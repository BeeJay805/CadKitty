"""Per-trial failure records and the aggregated failure dataset."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge_mind.analysis.taxonomy import Expectations, FailureLabel, classify, primary_failed_check
from forge_mind.pipeline import PipelineResult
from forge_mind.validator.report import ValidationReport


@dataclass
class TrialRecord:
    """Everything we record about one end-to-end trial."""

    trial_index: int
    passed: bool
    first_failing_stage: str | None
    primary_label: str
    primary_check_id: str | None
    validator_message: str | None
    op_id: str | None
    failed_check_ids: list[str]
    cost_usd: float
    duration_ms: float
    total_tokens: int
    artifacts: dict[str, str]
    error: str | None = None
    # Which Blueprint->program path produced this trial. Defaults to "unknown" so a
    # legacy record that predates this field is not falsely asserted as a real path;
    # every record created by the current harness sets it explicitly from the result.
    gen_path: str = "unknown"


@dataclass
class FailureDataset:
    """Aggregate over all trials of one analysis run."""

    bench_id: str
    run_id: str
    n_trials: int
    n_passed: int
    pass_rate: float
    mean_cost_usd: float
    mean_duration_ms: float
    label_counts: dict[str, int]
    stage_counts: dict[str, int]
    gen_path_counts: dict[str, int]  # which generation path produced each trial
    top_check_ids: list[tuple[str, int]]  # [(check_id, count), ...]
    representative_artifacts: dict[str, str]  # label -> trial dir of a representative failure
    records: list[TrialRecord] = field(default_factory=list)


def _relevant_op_id(report: ValidationReport | None, primary_check_id: str | None) -> str | None:
    """Best-effort op_id for the failure: the primary check's op_ref, else the
    first suggested fix that names an op.
    """
    if report is None:
        return None
    if primary_check_id is not None:
        for c in report.checks:
            if c.id == primary_check_id and c.op_ref:
                return c.op_ref
    for fix in report.suggested_fixes:
        if fix.op_id:
            return fix.op_id
    return None


def analyze_result(
    trial_index: int,
    result: PipelineResult,
    trial_dir: Path,
    expectations: Expectations,
    duration_ms: float,
    artifacts: dict[str, str],
) -> TrialRecord:
    """Build a TrialRecord from a finished pipeline run."""
    label = classify(result, expectations)

    primary_check = primary_failed_check(result.report) if result.report is not None else None
    primary_check_id = primary_check.id if primary_check is not None else None
    validator_message = primary_check.message if primary_check is not None else None
    failed_check_ids = result.report.failed_check_ids() if result.report is not None else []
    op_id = _relevant_op_id(result.report, primary_check_id)

    return TrialRecord(
        trial_index=trial_index,
        passed=result.passed,
        first_failing_stage=result.failed_stage,
        primary_label=label.value,
        primary_check_id=primary_check_id,
        validator_message=validator_message,
        op_id=op_id,
        failed_check_ids=failed_check_ids,
        cost_usd=result.total_cost_usd,
        duration_ms=duration_ms,
        total_tokens=result.total_tokens,
        artifacts=artifacts,
        error=result.error,
        gen_path=result.gen_path,
    )


def aggregate(bench_id: str, run_id: str, records: list[TrialRecord]) -> FailureDataset:
    """Roll up trial records into a FailureDataset."""
    n = len(records)
    n_passed = sum(1 for r in records if r.passed)

    label_counts = Counter(r.primary_label for r in records)
    stage_counts = Counter(r.first_failing_stage or "none" for r in records if not r.passed)
    gen_path_counts = Counter(r.gen_path for r in records)
    check_counts: Counter[str] = Counter()
    for r in records:
        check_counts.update(r.failed_check_ids)

    # One representative failed-trial artifact dir per label (first occurrence).
    representative: dict[str, str] = {}
    for r in records:
        if r.passed:
            continue
        representative.setdefault(r.primary_label, r.artifacts.get("dir", ""))

    return FailureDataset(
        bench_id=bench_id,
        run_id=run_id,
        n_trials=n,
        n_passed=n_passed,
        pass_rate=(n_passed / n if n else 0.0),
        mean_cost_usd=(sum(r.cost_usd for r in records) / n if n else 0.0),
        mean_duration_ms=(sum(r.duration_ms for r in records) / n if n else 0.0),
        label_counts=dict(label_counts.most_common()),
        stage_counts=dict(stage_counts.most_common()),
        gen_path_counts=dict(gen_path_counts.most_common()),
        top_check_ids=[(cid, cnt) for cid, cnt in check_counts.most_common(10)],
        representative_artifacts=representative,
        records=records,
    )


def dataset_to_dict(dataset: FailureDataset) -> dict[str, object]:
    return asdict(dataset)


# Labels we never expect from the auto-classifier; surfaced so a human can add
# them when reviewing artifacts.
MANUAL_LABELS = [
    FailureLabel.VALIDATOR_FALSE_POSITIVE.value,
    FailureLabel.VALIDATOR_FALSE_NEGATIVE.value,
    FailureLabel.VALID_BUT_MECHANICALLY_UGLY.value,
]


__all__ = [
    "MANUAL_LABELS",
    "FailureDataset",
    "TrialRecord",
    "aggregate",
    "analyze_result",
    "dataset_to_dict",
]
