"""Markdown report helpers."""

from __future__ import annotations

from stemds.metrics import aggregate_metrics, TaskEvalResult


def render_metrics_summary(results: list[TaskEvalResult]) -> str:
    lines = ["# StemDS Run Summary", ""]
    for key, value in aggregate_metrics(results).items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"

