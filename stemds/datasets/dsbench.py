"""Exploratory DSBench inspection and adapter skeleton.

DSBench is intentionally treated as external raw benchmark data. This module
does not make DSBench required for tests or for the DABench headline result.
It provides enough structure to inspect a local checkout and to convert only
simple, unambiguous tabular analysis tasks when such metadata is present.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stemds.datasets.base import JSONLWriterMixin
from stemds.ml.tasks import MLEngineeringTask
from stemds.tasks import DataAnalysisTask

SUPPORTED_SIMPLE_TABLE_SUFFIXES = {".csv"}
TABULAR_SUFFIXES = {".csv", ".tsv", ".xlsx", ".xls", ".xlsm", ".xlsb", ".parquet", ".json", ".jsonl"}
NON_TABULAR_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".ipynb", ".py", ".pdf", ".txt", ".md"}


@dataclass(slots=True)
class DSBenchInspection:
    root: str
    exists: bool
    directory_structure: list[str] = field(default_factory=list)
    metadata_files: list[str] = field(default_factory=list)
    task_files: list[str] = field(default_factory=list)
    data_files: list[str] = field(default_factory=list)
    zip_files: list[str] = field(default_factory=list)
    zip_member_count: int = 0
    zip_sample_members: list[str] = field(default_factory=list)
    detected_modalities: list[str] = field(default_factory=list)
    data_analysis_records: int = 0
    data_modeling_records: int = 0
    simple_subset_records: int = 0
    analysis_record_example: dict[str, Any] | None = None
    modeling_record_example: dict[str, Any] | None = None
    simple_record_example: dict[str, Any] | None = None
    example_data_file: str | None = None
    data_analysis_available: bool = False
    data_modeling_available: bool = False
    conversion_supported: bool = False
    recommended_strategy: str = ""
    blockers: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            "# DSBench Inspection",
            "",
            "This report is exploratory. DABench remains the validated StemDS headline benchmark.",
            "",
            "## Summary",
            f"- Root: `{self.root}`",
            f"- Exists: `{self.exists}`",
            f"- Data-analysis metadata records: `{self.data_analysis_records}`",
            f"- Data-modeling metadata records: `{self.data_modeling_records}`",
            f"- Simple convertible records: `{self.simple_subset_records}`",
            f"- Conversion supported now: `{self.conversion_supported}`",
            "",
            "## Relevant Directories And Files",
            *_bullet_paths("Directory structure", self.directory_structure),
            *_bullet_paths("Metadata files", self.metadata_files),
            *_bullet_paths("Task files", self.task_files),
            *_bullet_paths("Data files", self.data_files),
            *_bullet_paths("Zip files", self.zip_files),
            *_bullet_paths("Zip sample members", self.zip_sample_members[:20]),
            "",
            "## Detected Modalities",
            *([f"- `{modality}`" for modality in self.detected_modalities] or ["- None detected."]),
            "",
            "## Example Records",
            "### Data Analysis",
            _json_block(self.analysis_record_example),
            "",
            "### Data Modeling",
            _json_block(self.modeling_record_example),
            "",
            "### Simple Convertible",
            _json_block(self.simple_record_example),
            "",
            "## Example Data File",
            f"- `{self.example_data_file}`" if self.example_data_file else "- None detected.",
            "",
            "## Recommended Conversion Strategy",
            self.recommended_strategy or "No safe conversion strategy detected yet.",
            "",
            "## Blockers And Unknowns",
            *([f"- {item}" for item in self.blockers] or ["- No blockers recorded."]),
            "",
            "## Unknowns",
            *([f"- {item}" for item in self.unknowns] or ["- No unknowns recorded."]),
            "",
        ]
        return "\n".join(lines)


class DSBenchAdapter(JSONLWriterMixin):
    """Skeleton adapter for DSBench.

    Conceptual modes:
    - data_analysis: convert tasks with a table, question, and scalar/string answer
      into `DataAnalysisTask`.
    - data_modeling: convert tasks with a dataset, target, and metric into
      `MLEngineeringTask` when that mapping is explicit.

    The public DSBench layout inspected so far does not expose a simple universal
    task schema. This adapter therefore converts only records with explicit
    simple tabular fields and otherwise reports unsupported layouts.
    """

    name = "dsbench"

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.inspection = inspect_dsbench(root_dir)

    def inspect(self) -> DSBenchInspection:
        return self.inspection

    def load_tasks(self) -> list[DataAnalysisTask]:
        tasks, _skipped = self.convert_simple_tabular_subset()
        return tasks

    def convert_simple_tabular_subset(self, limit: int | None = None) -> tuple[list[DataAnalysisTask], list[str]]:
        metadata_path = _find_simple_metadata(self.root_dir)
        if metadata_path is None:
            return [], [
                "No simple tabular metadata file found. Expected simple_tasks.jsonl or tasks.jsonl "
                "with task_id, dataset_path, question, and answer fields."
            ]

        tasks: list[DataAnalysisTask] = []
        skipped: list[str] = []
        for index, record in enumerate(_read_jsonl(metadata_path), start=1):
            if limit is not None and len(tasks) >= limit:
                break
            try:
                task = self._convert_simple_record(record, metadata_path=metadata_path)
            except ValueError as exc:
                skipped.append(f"{metadata_path}:{index}: {exc}")
                continue
            tasks.append(task)
        return tasks, skipped

    def convert_data_analysis_tasks(self, limit: int | None = None) -> tuple[list[DataAnalysisTask], list[str]]:
        """Convert unambiguous table/question/answer DSBench records.

        Currently delegates to the simple metadata path. The native DSBench
        analysis metadata points to question files and mixed Excel/image/table
        assets inside archives, so converting it safely requires a dedicated
        extractor and answer-contract mapping.
        """

        return self.convert_simple_tabular_subset(limit=limit)

    def convert_data_modeling_tasks(self, limit: int | None = None) -> tuple[list[MLEngineeringTask], list[str]]:
        """Placeholder for future DSBench data-modeling conversion."""

        return [], [
            "DSBench data-modeling metadata does not expose local dataset paths, target columns, "
            "or metrics in a form that can be mapped to MLEngineeringTask without additional rules."
        ]

    def _convert_simple_record(self, record: dict[str, Any], metadata_path: Path) -> DataAnalysisTask:
        missing = [field_name for field_name in ["task_id", "dataset_path", "question", "answer"] if field_name not in record]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")

        dataset_path = _resolve_dataset_path(self.root_dir, metadata_path, str(record["dataset_path"]))
        if not dataset_path.exists():
            raise ValueError(f"dataset_path does not exist: {dataset_path}")
        if dataset_path.suffix.lower() not in SUPPORTED_SIMPLE_TABLE_SUFFIXES:
            raise ValueError(f"unsupported simple table suffix: {dataset_path.suffix}")

        answer = record["answer"]
        answer_type = str(record.get("answer_type") or _infer_answer_type(answer))
        tolerance = record.get("tolerance")
        if tolerance is None and answer_type == "number":
            tolerance = 1e-6

        return DataAnalysisTask(
            task_id=f"dsbench_{record['task_id']}",
            dataset_path=_display_path(dataset_path),
            question=str(record["question"]),
            answer=answer,
            answer_type=answer_type,
            tolerance=tolerance,
            tags=[str(tag) for tag in record.get("tags", ["dsbench", "simple_tabular"])],
            notes="Converted from an explicit simple DSBench-compatible tabular metadata record.",
            metadata={
                "adapter": self.name,
                "source_file": _display_path(metadata_path),
                "original_id": str(record["task_id"]),
                "raw_record": record,
            },
        )


def inspect_dsbench(root: Path) -> DSBenchInspection:
    root = Path(root)
    if not root.exists():
        return DSBenchInspection(
            root=str(root),
            exists=False,
            recommended_strategy="Clone DSBench into external/DSBench before inspection.",
            blockers=[f"Root does not exist: {root}"],
            unknowns=["Benchmark structure unavailable."],
        )

    files = [path for path in sorted(root.rglob("*")) if path.is_file() and "__pycache__" not in path.parts]
    relative_files = [_display_relative(path, root) for path in files]
    metadata_files = _detect_metadata_files(files, root)
    task_files = [path for path in relative_files if _looks_like_task_file(path)]
    zip_paths = [path for path in files if path.suffix.lower() == ".zip"]
    zip_files = [_display_relative(path, root) for path in zip_paths]

    data_files = [
        path
        for path in relative_files
        if Path(path).suffix.lower() in TABULAR_SUFFIXES | {".jpg", ".jpeg", ".png", ".gif", ".ipynb"}
        and path not in metadata_files
    ][:80]

    analysis_records, analysis_example = _read_optional_jsonl(root / "data_analysis" / "data.json")
    modeling_records, modeling_example = _read_optional_jsonl(root / "data_modeling" / "data.json")
    simple_path = _find_simple_metadata(root)
    simple_records, simple_example = _read_optional_jsonl(simple_path) if simple_path else (0, None)

    zip_member_count, zip_sample_members, zip_suffix_counts = _inspect_zips(zip_paths)
    suffix_counts = Counter(Path(path).suffix.lower() for path in relative_files if Path(path).suffix)
    suffix_counts.update(zip_suffix_counts)
    detected_modalities = _modalities_from_suffixes(suffix_counts)

    example_data_file = _first_data_file(relative_files, zip_sample_members)
    conversion_supported = simple_records > 0
    blockers = _build_blockers(
        analysis_records=analysis_records,
        modeling_records=modeling_records,
        simple_records=simple_records,
        detected_modalities=detected_modalities,
    )
    unknowns = _build_unknowns(analysis_records=analysis_records, modeling_records=modeling_records)

    recommended_strategy = _recommended_strategy(conversion_supported, detected_modalities)

    return DSBenchInspection(
        root=str(root),
        exists=True,
        directory_structure=_top_level_structure(root),
        metadata_files=metadata_files,
        task_files=task_files[:80],
        data_files=data_files,
        zip_files=zip_files,
        zip_member_count=zip_member_count,
        zip_sample_members=zip_sample_members,
        detected_modalities=detected_modalities,
        data_analysis_records=analysis_records,
        data_modeling_records=modeling_records,
        simple_subset_records=simple_records,
        analysis_record_example=analysis_example,
        modeling_record_example=modeling_example,
        simple_record_example=simple_example,
        example_data_file=example_data_file,
        data_analysis_available=analysis_records > 0,
        data_modeling_available=modeling_records > 0,
        conversion_supported=conversion_supported,
        recommended_strategy=recommended_strategy,
        blockers=blockers,
        unknowns=unknowns,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: expected object")
            records.append(payload)
    return records


def _read_optional_jsonl(path: Path | None) -> tuple[int, dict[str, Any] | None]:
    if path is None or not path.exists():
        return 0, None
    records = _read_jsonl(path)
    return len(records), records[0] if records else None


def _find_simple_metadata(root: Path) -> Path | None:
    for candidate in [
        root / "simple_tasks.jsonl",
        root / "tasks.jsonl",
        root / "data_analysis" / "simple_tasks.jsonl",
    ]:
        if candidate.exists():
            return candidate
    return None


def _resolve_dataset_path(root: Path, metadata_path: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    for base in [metadata_path.parent, root]:
        resolved = base / candidate
        if resolved.exists():
            return resolved
    return metadata_path.parent / candidate


def _detect_metadata_files(files: list[Path], root: Path) -> list[str]:
    detected: list[str] = []
    for path in files:
        lower_name = path.name.lower()
        if path.suffix.lower() in {".json", ".jsonl"} and (
            lower_name in {"data.json", "tasks.jsonl", "simple_tasks.jsonl"}
            or "metadata" in lower_name
            or "question" in lower_name
        ):
            detected.append(_display_relative(path, root))
    return detected[:80]


def _looks_like_task_file(path: str) -> bool:
    name = Path(path).name.lower()
    return bool(re.search(r"question|task|instruction|prompt", name))


def _inspect_zips(zip_paths: list[Path]) -> tuple[int, list[str], Counter[str]]:
    total_members = 0
    sample_members: list[str] = []
    suffix_counts: Counter[str] = Counter()
    for path in zip_paths:
        try:
            with zipfile.ZipFile(path) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/") and not name.startswith("__MACOSX")]
        except zipfile.BadZipFile:
            continue
        total_members += len(names)
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix:
                suffix_counts[suffix] += 1
            if len(sample_members) < 80:
                sample_members.append(f"{_display_path(path)}::{name}")
    return total_members, sample_members, suffix_counts


def _modalities_from_suffixes(suffix_counts: Counter[str]) -> list[str]:
    modalities: set[str] = set()
    if any(suffix in suffix_counts for suffix in {".csv", ".tsv", ".parquet", ".json", ".jsonl"}):
        modalities.add("tabular_text")
    if any(suffix in suffix_counts for suffix in {".xlsx", ".xls", ".xlsm", ".xlsb"}):
        modalities.add("excel")
    if any(suffix in suffix_counts for suffix in {".jpg", ".jpeg", ".png", ".gif"}):
        modalities.add("image")
    if ".ipynb" in suffix_counts:
        modalities.add("notebook")
    if ".py" in suffix_counts:
        modalities.add("code")
    if ".txt" in suffix_counts:
        modalities.add("text_questions")
    return sorted(modalities)


def _first_data_file(relative_files: list[str], zip_sample_members: list[str]) -> str | None:
    for path in [*zip_sample_members, *relative_files]:
        inner_path = path.split("::")[-1]
        name = Path(inner_path).name.lower()
        suffix = Path(inner_path).suffix.lower()
        if name == "data.json":
            continue
        if suffix in TABULAR_SUFFIXES | {".jpg", ".png", ".ipynb"}:
            return path
    return None


def _top_level_structure(root: Path) -> list[str]:
    paths: list[str] = []
    for path in sorted(root.iterdir()):
        if path.name.startswith(".git"):
            continue
        marker = "/" if path.is_dir() else ""
        paths.append(f"{path.name}{marker}")
    return paths


def _build_blockers(
    analysis_records: int,
    modeling_records: int,
    simple_records: int,
    detected_modalities: list[str],
) -> list[str]:
    blockers: list[str] = []
    if analysis_records and simple_records == 0:
        blockers.append(
            "Native data-analysis metadata lists question ids and answers, but question text and data files "
            "live separately in archives with mixed file types."
        )
    if "excel" in detected_modalities or "image" in detected_modalities:
        blockers.append("Detected Excel/image assets; current DataAnalysisTask supports CSV-style single-table tasks only.")
    if modeling_records:
        blockers.append(
            "Data-modeling metadata lists competitions, but local train/test files, target columns, and metrics "
            "are not explicit in the top-level metadata."
        )
    if simple_records == 0:
        blockers.append("No explicit simple CSV/question/answer subset metadata was found.")
    return blockers


def _build_unknowns(analysis_records: int, modeling_records: int) -> list[str]:
    unknowns: list[str] = []
    if analysis_records:
        unknowns.append("How to map ModelOff-style multiple-choice/numeric answers to StemDS scalar answer contracts.")
        unknowns.append("Whether all required analysis assets can be read without Excel/image-specific tooling.")
    if modeling_records:
        unknowns.append("Which DSBench modeling metric and target field should map to each MLEngineeringTask.")
    return unknowns


def _recommended_strategy(conversion_supported: bool, detected_modalities: list[str]) -> str:
    if conversion_supported:
        return (
            "A simple tabular subset metadata file was detected. Convert those records first, then expand the "
            "adapter only after defining explicit rules for Excel, image, notebook, and modeling tasks."
        )
    if {"excel", "image"} & set(detected_modalities):
        return (
            "Treat DSBench as a future multi-modal/multi-file benchmark extension. First add a dedicated extractor "
            "for archived question files and Excel workbooks, then define a conservative subset that can be scored "
            "with StemDS metrics."
        )
    return (
        "No clean subset was detected. Add explicit metadata with dataset_path, question, answer, and answer_type "
        "before converting to StemDS JSONL."
    )


def _infer_answer_type(answer: Any) -> str:
    if isinstance(answer, bool):
        return "boolean"
    if isinstance(answer, int | float) and not isinstance(answer, bool):
        return "number"
    return "string"


def _json_block(value: Any) -> str:
    if value is None:
        return "```json\nnull\n```"
    return "```json\n" + json.dumps(value, indent=2, sort_keys=True) + "\n```"


def _bullet_paths(title: str, values: list[str]) -> list[str]:
    lines = [f"### {title}"]
    lines.extend([f"- `{value}`" for value in values] or ["- None detected."])
    return lines


def _display_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return _display_path(path)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path.resolve())
