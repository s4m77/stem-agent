"""Structural validation for generated workflow graphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from stemds.workflows.graph import GeneratedWorkflowSpec


ALLOWED_NODE_TYPES = {
    "schema_summary",
    "llm_plan",
    "llm_code",
    "python_execute",
    "llm_repair",
    "answer_normalize",
    "llm_answer_check",
    "stop",
}


@dataclass(slots=True)
class ValidationResult:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_generated_workflow(spec: GeneratedWorkflowSpec) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not spec.workflow_id.strip():
        errors.append("workflow_id is required")
    if spec.limits.max_llm_calls > 5:
        errors.append("max_llm_calls must be <= 5")
    if spec.limits.max_repairs > 2:
        errors.append("max_repairs must be <= 2")
    if spec.limits.max_llm_calls < 1:
        errors.append("max_llm_calls must be >= 1")
    if spec.limits.max_repairs < 0:
        errors.append("max_repairs must be >= 0")
    if spec.limits.timeout_sec <= 0:
        errors.append("timeout_sec must be positive")

    node_ids = [node.id for node in spec.nodes]
    if not node_ids:
        errors.append("workflow must contain at least one node")
    if len(node_ids) != len(set(node_ids)):
        errors.append("node ids must be unique")
    if any(not node_id.strip() for node_id in node_ids):
        errors.append("node ids must be non-empty")

    node_by_id = {node.id: node for node in spec.nodes}
    node_types = [node.type for node in spec.nodes]
    for node in spec.nodes:
        if node.type not in ALLOWED_NODE_TYPES:
            errors.append(f"unsupported node type for {node.id}: {node.type}")
    if "llm_code" not in node_types:
        errors.append("workflow must contain at least one llm_code node")
    if "python_execute" not in node_types:
        errors.append("workflow must contain at least one python_execute node")
    if "stop" not in node_types:
        errors.append("workflow must contain a stop node")
    if "llm_repair" in node_types and spec.limits.max_repairs == 0:
        warnings.append("llm_repair node present but max_repairs is 0")
    if any(node_type in {"answer_normalize", "llm_answer_check"} for node_type in node_types):
        warnings.append("answer_normalize and llm_answer_check are no-op/TODO nodes in v0")

    adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for source, target in spec.edges:
        if source not in node_by_id:
            errors.append(f"edge references unknown source node: {source}")
            continue
        if target not in node_by_id:
            errors.append(f"edge references unknown target node: {target}")
            continue
        adjacency[source].append(target)
        indegree[target] += 1

    if not errors:
        stop_nodes = {node.id for node in spec.nodes if node.type == "stop"}
        start_nodes = [node_id for node_id, degree in indegree.items() if degree == 0] or node_ids
        if not any(_can_reach_stop(start, stop_nodes, adjacency) for start in start_nodes):
            errors.append("workflow must have a path to a stop node")

        for cycle in _find_cycles(adjacency):
            if not _is_allowed_repair_cycle(cycle, node_by_id, spec.limits.max_repairs):
                cycle_text = " -> ".join(cycle)
                errors.append(f"unsupported cycle: {cycle_text}")

    return ValidationResult(valid=not errors, errors=errors, warnings=warnings)


def _can_reach_stop(start: str, stop_nodes: set[str], adjacency: dict[str, list[str]]) -> bool:
    visited: set[str] = set()
    stack = [start]
    while stack:
        node_id = stack.pop()
        if node_id in stop_nodes:
            return True
        if node_id in visited:
            continue
        visited.add(node_id)
        stack.extend(adjacency.get(node_id, []))
    return False


def _find_cycles(adjacency: dict[str, list[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    seen_cycles: set[tuple[str, ...]] = set()
    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def dfs(node_id: str) -> None:
        visited.add(node_id)
        active.append(node_id)
        active_set.add(node_id)
        for next_id in adjacency.get(node_id, []):
            if next_id not in visited:
                dfs(next_id)
            elif next_id in active_set:
                index = active.index(next_id)
                cycle = active[index:] + [next_id]
                key = _cycle_key(cycle[:-1])
                if key not in seen_cycles:
                    seen_cycles.add(key)
                    cycles.append(cycle)
        active.pop()
        active_set.remove(node_id)

    for node_id in adjacency:
        if node_id not in visited:
            dfs(node_id)
    return cycles


def _cycle_key(nodes: list[str]) -> tuple[str, ...]:
    if not nodes:
        return tuple()
    rotations = [tuple(nodes[index:] + nodes[:index]) for index in range(len(nodes))]
    return min(rotations)


def _is_allowed_repair_cycle(cycle: list[str], node_by_id: dict, max_repairs: int) -> bool:
    if max_repairs <= 0:
        return False
    cycle_nodes = cycle[:-1]
    if len(cycle_nodes) != 2:
        return False
    cycle_types = {node_by_id[node_id].type for node_id in cycle_nodes}
    return cycle_types == {"python_execute", "llm_repair"}
