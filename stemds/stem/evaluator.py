"""Stem evaluator placeholder."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class StemEvaluationSummary:
    baseline_score: float
    developed_score: float


class StemEvaluator:
    def evaluate(self) -> StemEvaluationSummary:
        raise NotImplementedError("Stem evaluation is not implemented in the initial skeleton.")

