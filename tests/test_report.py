from __future__ import annotations

import json

from stemds.reporting.report import extract_metrics, render_experiment_report, write_experiment_report


def test_extract_metrics_handles_common_shapes() -> None:
    assert extract_metrics({"metrics": {"answer_accuracy": 0.1}}) == {"answer_accuracy": 0.1}
    assert extract_metrics({"aggregate_metrics": {"composite_score": 0.2}}) == {"composite_score": 0.2}
    assert extract_metrics({"answer_accuracy": 0.3, "other": 1}) == {"answer_accuracy": 0.3}


def test_report_generator_handles_required_minimal_fake_artifacts(tmp_path) -> None:
    generic = tmp_path / "generic.json"
    workflow_search = tmp_path / "workflow_search.json"
    workflow_test = tmp_path / "workflow_test.json"
    output = tmp_path / "report.md"
    generic.write_text(json.dumps({"metrics": {"answer_accuracy": 0.3, "composite_score": 0.2, "total_tasks": 2}}))
    workflow_search.write_text(
        json.dumps(
            {
                "frozen_workflow": {"workflow_id": "code_then_repair"},
                "results": [
                    {
                        "workflow_id": "direct_code",
                        "metrics": {
                            "answer_accuracy": 0.3,
                            "composite_score": 0.2,
                            "execution_success_rate": 0.9,
                            "invalid_code_rate": 0.1,
                        },
                    },
                    {
                        "workflow_id": "code_then_repair",
                        "accepted": True,
                        "metrics": {
                            "answer_accuracy": 0.5,
                            "composite_score": 0.45,
                            "execution_success_rate": 1.0,
                            "invalid_code_rate": 0.0,
                        },
                    },
                ],
            }
        )
    )
    workflow_test.write_text(
        json.dumps(
            {
                "workflow": {"workflow_id": "code_then_repair"},
                "metrics": {
                    "answer_accuracy": 0.5,
                    "composite_score": 0.45,
                    "execution_success_rate": 1.0,
                    "invalid_code_rate": 0.0,
                    "total_tasks": 2,
                },
            }
        )
    )

    warnings = write_experiment_report(
        generic_path=generic,
        workflow_search_path=workflow_search,
        workflow_test_path=workflow_test,
        output_path=output,
    )

    markdown = output.read_text()
    assert warnings == []
    assert "Generic baseline" in markdown
    assert "Workflow search" in markdown
    assert "Frozen workflow" in markdown
    assert "code_then_repair" in markdown
    assert "Limitations" in markdown


def test_report_generator_handles_missing_optional_artifacts(tmp_path) -> None:
    generic = tmp_path / "generic.json"
    workflow_search = tmp_path / "workflow_search.json"
    workflow_test = tmp_path / "workflow_test.json"
    generic.write_text(json.dumps({"metrics": {"answer_accuracy": 0.3, "total_tasks": 1}}))
    workflow_search.write_text(json.dumps({"frozen_workflow": {"workflow_id": "direct_code"}, "results": []}))
    workflow_test.write_text(json.dumps({"metrics": {"answer_accuracy": 0.3, "total_tasks": 1}}))

    markdown, warnings = render_experiment_report(
        generic_path=generic,
        workflow_search_path=workflow_search,
        workflow_test_path=workflow_test,
        seed_skills_path=tmp_path / "missing_seed.json",
    )

    assert warnings
    assert "Not provided." in markdown
