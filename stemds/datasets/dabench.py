"""Adapter for the public InfiAgent-DABench/DAEval validation data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stemds.datasets.base import JSONLWriterMixin
from stemds.tasks import DataAnalysisTask


@dataclass(slots=True)
class DABenchDiscovery:
    questions_path: Path
    labels_path: Path
    csv_root: Path
    csv_count: int


class DABenchAdapter(JSONLWriterMixin):
    name = "dabench"

    def __init__(
        self,
        root_dir: Path,
        metadata_path: Path | None = None,
        labels_path: Path | None = None,
        csv_root: Path | None = None,
    ) -> None:
        self.root_dir = root_dir
        self._metadata_path = metadata_path
        self._labels_path = labels_path
        self._csv_root = csv_root
        self.discovery = self.discover()

    def discover(self) -> DABenchDiscovery:
        questions_path = self._metadata_path or self._find_metadata_file("questions")
        labels_path = self._labels_path or self._find_metadata_file("labels")
        csv_root = self._csv_root or self._find_csv_root()
        return DABenchDiscovery(
            questions_path=questions_path,
            labels_path=labels_path,
            csv_root=csv_root,
            csv_count=len(list(csv_root.glob("*.csv"))),
        )

    def load_tasks(self) -> list[DataAnalysisTask]:
        questions = _read_jsonl(self.discovery.questions_path)
        labels = _labels_by_id(_read_jsonl(self.discovery.labels_path))
        tasks: list[DataAnalysisTask] = []
        for question in questions:
            original_id = question["id"]
            label = labels.get(str(original_id))
            if label is None:
                raise ValueError(f"Missing DABench label for question id {original_id}")
            tasks.append(self._convert_item(question, label))
        return tasks

    def _convert_item(self, question: dict[str, Any], label: dict[str, Any]) -> DataAnalysisTask:
        file_name = str(question["file_name"])
        csv_path = self.discovery.csv_root / file_name
        if not csv_path.exists():
            raise ValueError(f"Missing DABench CSV for question id {question['id']}: {csv_path}")

        common_answers = label.get("common_answers", [])
        answer, answer_type, tolerance = _convert_common_answers(common_answers)
        concepts = [str(concept) for concept in question.get("concepts", [])]
        tags = [_normalize_tag(concept) for concept in concepts]
        level = question.get("level")
        if level:
            tags.append(f"level_{_normalize_tag(str(level))}")

        return DataAnalysisTask(
            task_id=f"dabench_{question['id']}",
            dataset_path=_display_path(csv_path),
            question=_compose_question(question),
            answer=answer,
            answer_type=answer_type,
            tolerance=tolerance,
            tags=tags,
            notes="Converted from InfiAgent-DABench/DAEval public validation data.",
            metadata={
                "adapter": self.name,
                "original_id": question["id"],
                "source_file": _display_path(self.discovery.questions_path),
                "label_file": _display_path(self.discovery.labels_path),
                "csv_root": _display_path(self.discovery.csv_root),
                "file_name": file_name,
                "concepts": concepts,
                "constraints": question.get("constraints"),
                "output_format": question.get("format"),
                "level": level,
                "common_answers": common_answers,
                "raw_answer_type": "single" if len(common_answers) == 1 else "multi",
            },
        )

    def _find_metadata_file(self, kind: str) -> Path:
        candidates: list[Path] = []
        for data_dir in _candidate_data_dirs(self.root_dir):
            candidates.extend(sorted(data_dir.glob(f"*{kind}*.jsonl")))
        if not candidates:
            raise ValueError(f"Could not find DABench {kind} JSONL under {self.root_dir}")
        return candidates[0]

    def _find_csv_root(self) -> Path:
        candidates: list[Path] = []
        for data_dir in _candidate_data_dirs(self.root_dir):
            if list(data_dir.glob("*.csv")):
                candidates.append(data_dir)
            candidates.extend(path for path in data_dir.iterdir() if path.is_dir() and list(path.glob("*.csv")))
        if not candidates:
            raise ValueError(f"Could not find DABench CSV directory under {self.root_dir}")
        return sorted(candidates, key=lambda path: ("tables" not in path.name.lower(), -len(list(path.glob("*.csv"))), str(path)))[0]


def _candidate_data_dirs(root_dir: Path) -> list[Path]:
    candidates = [
        root_dir,
        root_dir / "data",
        root_dir / "examples" / "DA-Agent" / "data",
    ]
    return [path for path in candidates if path.exists() and path.is_dir()]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _labels_by_id(labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(label["id"]): label for label in labels}


def _compose_question(question: dict[str, Any]) -> str:
    parts = [str(question["question"]).strip()]
    constraints = question.get("constraints")
    if constraints:
        parts.append(f"Constraints:\n{str(constraints).strip()}")
    output_format = question.get("format")
    if output_format:
        parts.append(f"Required output format:\n{str(output_format).strip()}")
    return "\n\n".join(parts)


def _convert_common_answers(common_answers: list[Any]) -> tuple[str | float | int | bool, str, float | None]:
    if len(common_answers) == 1:
        raw_value = str(common_answers[0][1])
        inferred_value = _infer_scalar_answer(raw_value)
        if isinstance(inferred_value, bool):
            return inferred_value, "boolean", None
        if isinstance(inferred_value, int | float) and not isinstance(inferred_value, bool):
            return inferred_value, "number", 1e-6
        return str(inferred_value), "string", None

    canonical = "\n".join(f"@{name}[{value}]" for name, value in common_answers)
    return canonical, "string", None


def _infer_scalar_answer(value: str) -> str | float | int | bool:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    numeric_text = normalized.replace(",", "")
    if re.fullmatch(r"[-+]?\d+", numeric_text):
        return int(numeric_text)
    if re.fullmatch(r"[-+]?(\d+(\.\d*)?|\.\d+)(e[-+]?\d+)?", numeric_text, flags=re.IGNORECASE):
        return float(numeric_text)
    return normalized


def _normalize_tag(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())
