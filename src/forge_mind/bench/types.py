"""Harness data types.

Two layers:
- TrialOutput: raw result a Runner produces (report + timings + artifacts).
- TrialResult: the harness's evaluation of a TrialOutput against pass criteria
  (serializable summary for the report / history).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.validator.report import ValidationReport


@dataclass
class StageTiming:
    """Per-stage telemetry. tokens/cost are 0 for non-LLM stages."""

    stage: str
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


@dataclass
class TrialOutput:
    """What a Runner returns for one trial (raw, pre-evaluation)."""

    report: ValidationReport | None
    stage_timings: list[StageTiming] = field(default_factory=list)
    artifacts: dict[str, Path] = field(default_factory=dict)
    iterations: int = 1
    first_failing_stage: str | None = None
    error: str | None = None

    @property
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.stage_timings)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.stage_timings)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.stage_timings)


@dataclass
class TrialResult:
    """Harness evaluation of one trial. JSON-serializable summary."""

    trial_index: int
    passed: bool
    overall_status: str
    first_failing_stage: str | None
    failed_check_ids: list[str]
    criteria_results: dict[str, bool]
    tier_pass_counts: dict[str, str]  # tier -> "passed/total"
    iterations: int
    total_duration_ms: float
    total_tokens: int
    total_cost_usd: float
    artifacts: dict[str, str]
    error: str | None = None
    # Shape-similarity vs the benchmark's reference mesh (ADR-0019 follow-on), REPORTED not
    # gated: chamfer_distance_mm / volumetric_iou / f1_at_threshold (+ params). None when the
    # benchmark has no reference or the comparison could not be computed.
    shape_similarity: dict[str, float] | None = None


@dataclass
class BenchmarkSummary:
    """Aggregate over all trials of one benchmark run."""

    bench_id: str
    runner: str
    run_id: str
    n_trials: int
    n_passed: int
    pass_rate: float
    min_passing_trials: int
    benchmark_passed: bool
    mean_duration_ms: float
    mean_tokens: float
    mean_cost_usd: float
    trials: list[TrialResult]


__all__ = [
    "BenchmarkSummary",
    "StageTiming",
    "TrialOutput",
    "TrialResult",
]
