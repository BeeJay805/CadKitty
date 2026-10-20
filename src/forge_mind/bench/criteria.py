"""Pass criteria — what makes a trial (and a benchmark) pass.

Loaded from a benchmark's pass_criteria.yaml. A trial passes when validation
passes, all required checks are present-and-passing, and it stays within the
time/cost/iteration budget. A benchmark passes when at least
``min_passing_trials`` of its trials pass.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from forge_mind.bench.types import TrialOutput


class PassCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trials: int = Field(default=10, ge=1)
    min_passing_trials: int = Field(default=9, ge=0)
    max_iterations_per_trial: int = Field(default=3, ge=1)
    max_wall_seconds: float = Field(default=60.0, gt=0)
    max_cost_usd_per_trial: float = Field(default=0.30, ge=0)
    required_checks: list[str] = Field(default_factory=list)
    reference_dsl: str | None = Field(
        default=None,
        description="Repo-relative path to the known-good DSL, used by --use-handwritten.",
    )


def load_criteria(path: Path) -> PassCriteria:
    data = yaml.safe_load(path.read_text()) or {}
    return PassCriteria.model_validate(data)


def evaluate_trial(
    output: TrialOutput,
    criteria: PassCriteria,
) -> tuple[bool, dict[str, bool]]:
    """Evaluate a trial output against criteria. Returns (passed, per-criterion)."""
    report = output.report

    no_error = output.error is None
    validation_pass = report is not None and report.passed

    if report is not None:
        passing_ids = {c.id for c in report.checks if c.status.value == "pass"}
        required_present = all(rc in passing_ids for rc in criteria.required_checks)
    else:
        required_present = not criteria.required_checks

    within_time = output.total_duration_ms <= criteria.max_wall_seconds * 1000.0
    within_cost = output.total_cost_usd <= criteria.max_cost_usd_per_trial
    within_iters = output.iterations <= criteria.max_iterations_per_trial

    criteria_results = {
        "no_error": no_error,
        "validation_pass": validation_pass,
        "required_checks_pass": required_present,
        "within_time": within_time,
        "within_cost": within_cost,
        "within_iterations": within_iters,
    }
    return all(criteria_results.values()), criteria_results


__all__ = ["PassCriteria", "evaluate_trial", "load_criteria"]
