"""Read-only research assistant: benchmark/result data -> next engineering task.

Reads existing benchmark/analysis result files (the ``*-report.json`` snapshots
under ``benchmarks/analysis/`` and the auto-runner outputs under
``benchmarks/results/_*``), normalises across the different report shapes
(``FailureDataset``, ``BestOfKDataset``, revision report), aggregates pass rate
/ failure mix / cost / latency / regression-vs-baseline, then applies a small
rule engine to recommend the next scoped engineering task.

Strictly read-only: it never mutates source, calls an LLM, or runs trials. It
writes one markdown file (``next_action.md``) that summarises the diagnosis and
an implementation prompt a human can hand to a coding agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Normalised report view (one shape across FailureDataset / BestOfK / revision)
# ---------------------------------------------------------------------------


class ReportKind(StrEnum):
    FAILURE = "failure"  # analysis/records.py FailureDataset
    BESTOFK = "bestofk"  # analysis/bestofk.py BestOfKDataset
    REVISION = "revision"  # analysis/revision.py revision report
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResultReport:
    """A normalised view over the three on-disk report shapes."""

    path: Path
    bench_id: str
    run_id: str
    kind: ReportKind
    n_trials: int
    n_passed: int
    pass_rate: float
    mean_cost_usd: float
    mean_duration_ms: float
    # Failures by primary label (most-common-first). For best-of-k this is the
    # CANDIDATE label distribution, not the winner's -- noted in the diagnosis.
    label_counts: dict[str, int]
    # Top failed check ids on the "winner" side (best-of-k winner or
    # post-revision); for plain analysis runs this is the only side.
    top_check_ids: list[tuple[str, int]]
    # Stage attribution. Empty for best-of-k/revision reports.
    stage_counts: dict[str, int]
    # Validator messages + errors mined from the matching JSONL (best-effort).
    # Used by component-registry / pattern heuristics.
    validator_messages: list[str] = field(default_factory=list)
    trial_errors: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


# Auto-generated outputs (``write_dataset``) use ``report.json``; committed
# snapshots use ``<prefix>-report.json``. Both endings are matched.
_REPORT_GLOBS = ("*-report.json", "report.json")


def find_reports(directory: Path) -> list[Path]:
    """All report JSON files directly under ``directory``, oldest mtime first."""
    if not directory.is_dir():
        return []
    seen: set[Path] = set()
    for pattern in _REPORT_GLOBS:
        for p in directory.glob(pattern):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen, key=lambda p: p.stat().st_mtime)


def _match_jsonl(report_path: Path) -> Path | None:
    """Find the per-trial JSONL sibling for a report.json file, if present."""
    parent = report_path.parent
    name = report_path.name
    # `<prefix>-report.json` -> `<prefix>-records.jsonl` or `<prefix>-trials.jsonl`.
    if name.endswith("-report.json"):
        stem = name[: -len("-report.json")]
        for suffix in ("-records.jsonl", "-trials.jsonl"):
            candidate = parent / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate
    # `report.json` -> `records.jsonl` or `trials.jsonl`.
    if name == "report.json":
        for suffix in ("records.jsonl", "trials.jsonl"):
            candidate = parent / suffix
            if candidate.is_file():
                return candidate
    return None


def _detect_kind(raw: dict[str, Any]) -> ReportKind:
    if "revised_pass_rate" in raw or "top_failed_after_revision" in raw:
        return ReportKind.REVISION
    if "best_of_k_pass_rate" in raw or "candidate_label_counts" in raw:
        return ReportKind.BESTOFK
    if "label_counts" in raw and "pass_rate" in raw:
        return ReportKind.FAILURE
    return ReportKind.UNKNOWN


def _as_check_pairs(raw_list: list[Any] | None) -> list[tuple[str, int]]:
    """Coerce ``[[id, n], ...]`` (json) to ``[(id, n), ...]``."""
    out: list[tuple[str, int]] = []
    if not raw_list:
        return out
    for item in raw_list:
        if isinstance(item, list | tuple) and len(item) == 2:
            cid, count = item[0], item[1]
            if isinstance(cid, str) and isinstance(count, int):
                out.append((cid, count))
    return out


def _load_jsonl(path: Path) -> tuple[list[str], list[str]]:
    """Mine ``validator_message`` and ``error`` fields from a records/trials JSONL.

    Best-effort: returns (validator_messages, errors). Both lists may be empty
    when the file's shape doesn't carry those fields (best-of-k trials.jsonl
    has no validator_message, only check_ids).
    """
    messages: list[str] = []
    errors: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            msg = obj.get("validator_message")
            if isinstance(msg, str) and msg:
                messages.append(msg)
            err = obj.get("error")
            if isinstance(err, str) and err:
                errors.append(err)
    except OSError:
        return [], []
    return messages, errors


def load_report(path: Path) -> ResultReport:
    """Read one report.json (any shape) into a normalised ResultReport."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a JSON object")

    kind = _detect_kind(raw)
    bench_id = str(raw.get("bench_id", path.parent.name))
    run_id = str(raw.get("run_id", path.stem))
    n_trials = int(raw.get("n_trials", 0))

    # Pass rate -- prefer best-of-k winner rate if present (it's the path that
    # actually ships); fall back to the failure-dataset pass_rate; revision
    # reports get their revised rate.
    pass_rate = 0.0
    if "revised_pass_rate" in raw:
        pass_rate = float(raw["revised_pass_rate"])
    elif "best_of_k_pass_rate" in raw:
        pass_rate = float(raw["best_of_k_pass_rate"])
    elif "pass_rate" in raw:
        pass_rate = float(raw["pass_rate"])
    n_passed = round(pass_rate * n_trials)
    if "n_passed" in raw and kind == ReportKind.FAILURE:
        n_passed = int(raw["n_passed"])

    mean_cost = float(raw.get("mean_cost_usd", 0.0))
    mean_dur = float(raw.get("mean_duration_ms", 0.0))

    # Label counts: failure -> label_counts; best-of-k/revision ->
    # candidate_label_counts (per-candidate, not per-trial).
    label_src = raw.get("label_counts") or raw.get("candidate_label_counts") or {}
    label_counts: dict[str, int] = {
        str(k): int(v) for k, v in label_src.items() if isinstance(v, int | float)
    }

    # Top check ids. Failure datasets use top_check_ids; best-of-k uses
    # top_failed_after (the winner column -- the surface the model actually
    # ships); revision uses top_failed_after_revision.
    top_src: list[Any] | None = (
        raw.get("top_failed_after_revision")
        or raw.get("top_failed_after")
        or raw.get("top_check_ids")
    )
    top_check_ids = _as_check_pairs(top_src)

    stage_src = raw.get("stage_counts") or {}
    stage_counts: dict[str, int] = {
        str(k): int(v) for k, v in stage_src.items() if isinstance(v, int | float)
    }

    jsonl_path = _match_jsonl(path)
    messages, errors = _load_jsonl(jsonl_path) if jsonl_path else ([], [])

    return ResultReport(
        path=path,
        bench_id=bench_id,
        run_id=run_id,
        kind=kind,
        n_trials=n_trials,
        n_passed=n_passed,
        pass_rate=pass_rate,
        mean_cost_usd=mean_cost,
        mean_duration_ms=mean_dur,
        label_counts=label_counts,
        top_check_ids=top_check_ids,
        stage_counts=stage_counts,
        validator_messages=messages,
        trial_errors=errors,
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Diagnosis: dominant failure mode + (optional) regression vs baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Diagnosis:
    """A summary of the failure mix; the rule engine consumes this."""

    dominant_label: str | None  # most-common non-"passed" label, or None
    dominant_label_share: float  # fraction of n_trials
    dominant_check: tuple[str, int] | None  # top failed check id, or None
    dominant_check_share: float  # fraction of n_trials
    component_signal: bool  # validator/error text mentions component/pattern
    component_evidence: list[str]  # short snippets backing component_signal
    regression: bool
    improvement: bool
    delta_summary: dict[str, float]  # baseline -> current deltas (current - baseline)
    baseline_path: Path | None


_DOMINANCE_THRESHOLD = 0.30  # >=30% of trials -> "dominant"
_REGRESSION_PASS_RATE_PTS = 0.05  # >5 percentage points drop
_REGRESSION_COST_FRACTION = 0.50  # >50% mean-cost jump

# Lowercase substrings that suggest "component or board-specific" trouble.
# Matched against validator messages + trial errors (best-effort).
_COMPONENT_KEYWORDS = (
    "unknown component",
    "no such component",
    "component not found",
    "not in registry",
    "unsupported component",
    "missing component",
    "board not recognised",
    "board not recognized",
    "unsupported board",
    "unknown board",
    "unknown pattern",
    "unsupported pattern",
)


def _dominant_label(label_counts: dict[str, int], n_trials: int) -> tuple[str | None, float]:
    """The most-common non-``passed`` label, with its share of n_trials."""
    if n_trials <= 0:
        return None, 0.0
    # Iterate in input (most-common-first) order; ignore "passed".
    for label, count in label_counts.items():
        if label == "passed":
            continue
        return label, count / n_trials
    return None, 0.0


def _dominant_check(
    top_check_ids: list[tuple[str, int]], n_trials: int
) -> tuple[tuple[str, int] | None, float]:
    if n_trials <= 0 or not top_check_ids:
        return None, 0.0
    cid, count = top_check_ids[0]
    return (cid, count), count / n_trials


def _component_signal(report: ResultReport) -> tuple[bool, list[str]]:
    evidence: list[str] = []
    for text in (*report.validator_messages, *report.trial_errors):
        low = text.lower()
        if any(kw in low for kw in _COMPONENT_KEYWORDS):
            evidence.append(text.strip())
    # Dedupe while preserving order; cap to keep next_action.md readable.
    seen: set[str] = set()
    unique: list[str] = []
    for s in evidence:
        if s in seen:
            continue
        seen.add(s)
        unique.append(s)
        if len(unique) >= 5:
            break
    return bool(unique), unique


def diagnose(report: ResultReport, baseline: ResultReport | None = None) -> Diagnosis:
    """Build a Diagnosis from a (current, optional baseline) pair."""
    dom_label, dom_label_share = _dominant_label(report.label_counts, report.n_trials)
    dom_check, dom_check_share = _dominant_check(report.top_check_ids, report.n_trials)
    has_component_signal, evidence = _component_signal(report)

    regression = False
    improvement = False
    delta: dict[str, float] = {}
    baseline_path: Path | None = None
    if baseline is not None and baseline.n_trials > 0:
        baseline_path = baseline.path
        pass_delta = report.pass_rate - baseline.pass_rate
        cost_delta = report.mean_cost_usd - baseline.mean_cost_usd
        dur_delta = report.mean_duration_ms - baseline.mean_duration_ms
        delta = {
            "pass_rate": pass_delta,
            "mean_cost_usd": cost_delta,
            "mean_duration_ms": dur_delta,
        }
        if pass_delta <= -_REGRESSION_PASS_RATE_PTS:
            regression = True
        if (
            baseline.mean_cost_usd > 0
            and cost_delta / baseline.mean_cost_usd >= _REGRESSION_COST_FRACTION
        ):
            regression = True
        if pass_delta >= _REGRESSION_PASS_RATE_PTS:
            improvement = True

    return Diagnosis(
        dominant_label=dom_label,
        dominant_label_share=dom_label_share,
        dominant_check=dom_check,
        dominant_check_share=dom_check_share,
        component_signal=has_component_signal,
        component_evidence=evidence,
        regression=regression,
        improvement=improvement,
        delta_summary=delta,
        baseline_path=baseline_path,
    )


# ---------------------------------------------------------------------------
# Rule engine: Diagnosis -> Recommendation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recommendation:
    area: str  # short tag: layout-helper, interface, component-registry, ...
    title: str
    summary: str  # 1-2 sentence rationale
    implementation_prompt: str
    constraints: list[str]
    stop_conditions: list[str]


_BASE_CONSTRAINTS = (
    "Do NOT change validator thresholds (min_wall, overhang, envelope).",
    "Do NOT add new CAD primitives, components, or patterns unless the task explicitly says to.",
    "Keep ruff + mypy --strict + non-LLM pytest green before committing.",
    "Touch only files inside the area named above; do not opportunistically refactor adjacent code.",
)


def _measurement_recommendation(report: ResultReport, diag: Diagnosis) -> Recommendation:
    label_summary = (
        ", ".join(f"{k}={v}" for k, v in list(report.label_counts.items())[:4]) or "(none)"
    )
    return Recommendation(
        area="measurement",
        title=f"Gather more measurement on {report.bench_id} before coding",
        summary=(
            "No single failure mode dominates "
            f"(top label share {diag.dominant_label_share * 100:.0f}%, "
            f"top check share {diag.dominant_check_share * 100:.0f}%). "
            "Coding into noise will burn budget. Re-measure with a larger N or a "
            "different benchmark variant before committing engineering effort."
        ),
        implementation_prompt=(
            f"Re-run the existing benchmark `{report.bench_id}` with a larger trial "
            "count (n>=30 if affordable) using the existing CLI commands "
            "(`forge-mind analyze`, `forge-mind candidates`, `forge-mind abtest`). "
            "Do NOT change prompts, the DSL, the interpreter, validator thresholds, "
            "or any code. The goal is a tighter measurement, not a fix. "
            f"Current label mix: {label_summary}. Commit the new report under "
            "`benchmarks/analysis/<bench>/` and re-run `forge-mind analyze-next` "
            "afterwards."
        ),
        constraints=list(_BASE_CONSTRAINTS),
        stop_conditions=[
            "A new committed report.json under benchmarks/analysis/<bench>/.",
            "Dominant-label share or dominant-check share crosses 30% (then re-run analyze-next).",
        ],
    )


def _layout_helper_recommendation(report: ResultReport, defect: str) -> Recommendation:
    if defect == "wall_too_thin":
        title = "Add a layout helper that guarantees min-wall lands around holes/edges"
        details = (
            "The dominant failure is thin walls (`features.min_wall_thickness`). "
            "Past evidence (FINDINGS.md) shows these are hole-to-edge webs the LLM "
            "lays out wrong, not a single box's thickness. Fix the LayoutSpec "
            "interpreter (or add a new helper used by interpreters) that places "
            "interface holes a guaranteed >= min_wall from every edge, mirroring "
            "the ADR-0005 land rule for standoffs."
        )
        area = "layout-helper"
    elif defect == "overhang_too_steep":
        title = "Add a printability/layout rule that prevents overhang failures"
        details = (
            "Overhang (`printability.overhang_max_angle`) is the dominant failure. "
            "Add an interpreter rule (or shared helper) that picks a print "
            "orientation / fillet / chamfer that keeps overhangs <= 45 deg by "
            "construction, instead of relying on the LLM to choose right."
        )
        area = "layout-helper"
    else:  # envelope_failed
        title = "Tighten the envelope/feasibility rule at the blueprint stage"
        details = (
            "Bounding-box overflow (`mesh.bbox_within_max`) is dominant. Past "
            "evidence: envelope is a blueprint-stage lever, not a DSL one, because "
            "every candidate of a blueprint shares the same envelope choice. Add "
            "a feasibility check that flags envelopes with insufficient margin "
            "before the LayoutSpec is generated."
        )
        area = "feasibility-rule"

    return Recommendation(
        area=area,
        title=title,
        summary=details,
        implementation_prompt=(
            f"You are working on Forge-Mind, bench `{report.bench_id}`, run `{report.run_id}`.\n\n"
            f"Problem: {details}\n\n"
            "Scope:\n"
            "1. Add the helper/rule inside the existing LayoutSpec interpreter package "
            "(`src/forge_mind/layout/`) or as a shared fact under `src/forge_mind/layout/` "
            "/`interfaces/`. Choose the one that already owns this concern.\n"
            "2. Add a unit test that reproduces the failure on a synthetic spec and shows "
            "the helper makes the failure structurally impossible.\n"
            "3. Do NOT touch prompts, validator thresholds, or non-layout code.\n"
            "4. Re-run only the non-LLM pytest suite; LLM benchmarks remain a human "
            "decision.\n"
        ),
        constraints=[
            *_BASE_CONSTRAINTS,
            "Stay inside src/forge_mind/layout/ (and tests/unit/layout/).",
        ],
        stop_conditions=[
            "Synthetic-spec unit test reproduces the defect before the change and passes after.",
            "Full non-LLM pytest suite stays green; ruff + mypy --strict still pass.",
            "No new pattern, component, or validator threshold is added.",
        ],
    )


def _interface_recommendation(report: ResultReport, diag: Diagnosis) -> Recommendation:
    cid = diag.dominant_check[0] if diag.dominant_check else "interfaces.*"
    return Recommendation(
        area="interface",
        title=f"Strengthen the interface place/check for `{cid}`",
        summary=(
            f"Dominant failure is `{cid}` (interface family). The interpreter is "
            "placing geometry that the interface check rejects. Tighten the shared "
            "interface fact (ADR-0006: place() + check() pair) so the placement and "
            "the check measure the same thing."
        ),
        implementation_prompt=(
            f"You are working on Forge-Mind, bench `{report.bench_id}`, run `{report.run_id}`.\n\n"
            f"Failing check: `{cid}`.\n\n"
            "Scope:\n"
            "1. Locate the shared interface fact under `src/forge_mind/interfaces/` (or the "
            "interpreter helper under `src/forge_mind/layout/` that calls it) that owns this "
            "interface's place() and check() pair.\n"
            "2. Audit the two functions for a mismatch (coordinate frame, tolerance, "
            "rounding, anchor) and fix the smaller side. Reuse existing facts; do NOT add a "
            "new component.\n"
            "3. Add a unit test that builds a minimal PartProgram exercising the interface "
            "and asserts the check now passes.\n"
            "4. Do NOT change the validator's tolerance values or add a new interface type.\n"
        ),
        constraints=[
            *_BASE_CONSTRAINTS,
            "Stay inside src/forge_mind/interfaces/ and src/forge_mind/layout/.",
        ],
        stop_conditions=[
            "A unit test fails before the fix and passes after.",
            "Full non-LLM pytest suite stays green; ruff + mypy --strict still pass.",
            "No new interface type, component, or validator threshold is added.",
        ],
    )


def _component_registry_recommendation(report: ResultReport, diag: Diagnosis) -> Recommendation:
    sample = diag.component_evidence[0] if diag.component_evidence else "missing-component evidence"
    return Recommendation(
        area="component-registry",
        title="Add or expose the missing board/component in the component registry",
        summary=(
            "Trial messages indicate the interpreter is being asked for a board/component "
            "the registry does not expose. Example: "
            f"`{sample[:160]}`. Add the missing component fact (ADR-0007) so the "
            "interpreter can look it up by name -- do not change the interpreter."
        ),
        implementation_prompt=(
            f"You are working on Forge-Mind, bench `{report.bench_id}`, run `{report.run_id}`.\n\n"
            "Symptom: trial messages mention an unknown/missing component or board. "
            f"Representative evidence: `{sample}`.\n\n"
            "Scope:\n"
            "1. Identify the missing component (board, motor, standoff family, etc.) from "
            "the evidence, then add it to the component registry under "
            "`src/forge_mind/layout/` (per ADR-0007 Stage D), reusing existing component "
            "shapes if at all possible.\n"
            "2. Wire it into the existing interpreter only via the registry lookup it "
            "already does; do NOT add a new interpreter or pattern.\n"
            "3. Add a unit test that constructs a LayoutSpec referencing the new component "
            "and asserts the interpreter resolves it.\n"
            "4. Do NOT add a new validator check or change thresholds.\n"
        ),
        constraints=[
            *_BASE_CONSTRAINTS,
            "Stay inside the component-registry module and its tests.",
        ],
        stop_conditions=[
            "Registry lookup unit test passes (was failing or absent before).",
            "Full non-LLM pytest suite stays green; ruff + mypy --strict still pass.",
            "No new pattern or validator threshold is added.",
        ],
    )


def _new_pattern_recommendation(report: ResultReport, diag: Diagnosis) -> Recommendation:
    label = diag.dominant_label or "layoutspec_invalid"
    return Recommendation(
        area="pattern-interpreter",
        title="Plan a new LayoutSpec pattern interpreter for the unsupported shape family",
        summary=(
            f"The dominant failure label is `{label}` -- the LLM is asking for a "
            "shape family the current patterns cannot express. Coding requires a "
            "design pass first: do NOT just bolt on a pattern. Author an ADR that "
            "scopes the new pattern, then build it behind the existing "
            "discriminated-union LayoutSpec + LayoutInterpreter contract."
        ),
        implementation_prompt=(
            f"You are working on Forge-Mind, bench `{report.bench_id}`, run `{report.run_id}`.\n\n"
            f"Symptom: the dominant failure label is `{label}` and the existing patterns "
            "(l_bracket_motor_mount, pi4_case, ...) cannot represent what the model is "
            "trying to produce.\n\n"
            "Scope (planning first, code second):\n"
            "1. Write an ADR under `docs/decision_log/` that scopes the new pattern: which "
            "part family it covers, what its spec fields are, what feasibility rules the "
            "interpreter enforces, and which interfaces it composes from the registry.\n"
            "2. Wait for ADR sign-off (this command does not approve it).\n"
            "3. Only after sign-off: implement the new pattern + a hand-authored fixture "
            "that the existing validator passes; add a smoke test.\n"
            "4. Do NOT widen the LLM prompt to cover the new pattern in the same change.\n"
        ),
        constraints=[
            *_BASE_CONSTRAINTS,
            "Land the ADR in its own commit before any interpreter code.",
        ],
        stop_conditions=[
            "ADR file lands and is approved by the human reviewer.",
            "Hand-authored fixture for the new pattern passes the existing validator.",
            "No prompt / validator-threshold change ships in the same commit.",
        ],
    )


def _high_pass_recommendation(report: ResultReport) -> Recommendation:
    return Recommendation(
        area="benchmark-coverage",
        title="Add a new benchmark; this one is saturated",
        summary=(
            f"Pass rate is {report.pass_rate * 100:.0f}% with no dominant failure. "
            "Further engineering on this bench is unlikely to move the number. "
            "Add a new benchmark (different part family) to test whether the "
            "architecture generalises beyond what is already measured."
        ),
        implementation_prompt=(
            f"You are working on Forge-Mind. `{report.bench_id}` is saturated "
            f"({report.pass_rate * 100:.0f}% pass on the latest run). Scope a NEW "
            "benchmark following the same layout as the existing `benchmarks/bench-00X-*/` "
            "folders (prompt.txt, ground_truth.json, pass_criteria.yaml, expected interfaces). "
            "Do NOT add new interpreters or prompts yet -- the benchmark is the spec, the "
            "next analyze-next run will say what code to write.\n"
        ),
        constraints=[
            *_BASE_CONSTRAINTS,
            "Benchmark assets only -- no source-code change in this commit.",
        ],
        stop_conditions=[
            "A new benchmark folder lands with prompt/ground-truth/criteria + a handwritten reference.",
            "`forge-mind bench <new-bench-id> --use-handwritten` passes on the reference.",
        ],
    )


_LABEL_RULES: dict[str, str] = {
    "wall_too_thin": "wall_too_thin",
    "overhang_too_steep": "overhang_too_steep",
    "envelope_failed": "envelope_failed",
}


def recommend(report: ResultReport, diag: Diagnosis) -> Recommendation:
    """Apply the rule engine: Diagnosis -> next-action Recommendation."""
    # 0. Empty data -> ask for more.
    if report.n_trials == 0:
        return _measurement_recommendation(report, diag)

    # 1. Saturated -> add coverage rather than tune.
    if report.pass_rate >= 0.95 and not diag.regression:
        return _high_pass_recommendation(report)

    # 2. Component-registry signal trumps generic label rules (it's evidence-
    #    backed by validator/error text and points to a specific fix surface).
    if diag.component_signal:
        return _component_registry_recommendation(report, diag)

    # 3. Layoutspec-stage failures -> new pattern (LLM produced something the
    #    union/interpreter does not understand or cannot satisfy).
    if diag.dominant_label in {"layoutspec_invalid", "layout_infeasible"} and (
        diag.dominant_label_share >= _DOMINANCE_THRESHOLD
    ):
        return _new_pattern_recommendation(report, diag)

    # 4. Label-driven layout helpers for the three measured-dominant defects.
    if diag.dominant_label in _LABEL_RULES and diag.dominant_label_share >= _DOMINANCE_THRESHOLD:
        return _layout_helper_recommendation(report, _LABEL_RULES[diag.dominant_label])

    # 5. Interface failure dominant.
    if diag.dominant_label == "interface_failed" and (
        diag.dominant_label_share >= _DOMINANCE_THRESHOLD
    ):
        return _interface_recommendation(report, diag)

    # 6. Check-id is dominant even when the label rule did not fire (e.g.
    #    best-of-k reports do not produce label_counts mapped to our buckets).
    if diag.dominant_check is not None and diag.dominant_check_share >= _DOMINANCE_THRESHOLD:
        cid = diag.dominant_check[0]
        if cid.startswith("interfaces."):
            return _interface_recommendation(report, diag)
        if cid == "features.min_wall_thickness":
            return _layout_helper_recommendation(report, "wall_too_thin")
        if cid == "printability.overhang_max_angle":
            return _layout_helper_recommendation(report, "overhang_too_steep")
        if cid == "mesh.bbox_within_max":
            return _layout_helper_recommendation(report, "envelope_failed")

    # 7. Nothing dominant -> measure more.
    return _measurement_recommendation(report, diag)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _format_evidence_rows(report: ResultReport, diag: Diagnosis) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = [
        ("benchmark", report.bench_id),
        ("run id", report.run_id),
        ("report kind", report.kind.value),
        ("n trials", str(report.n_trials)),
        ("pass rate", f"{report.pass_rate * 100:.0f}% ({report.n_passed}/{report.n_trials})"),
        ("mean cost", f"${report.mean_cost_usd:.4f}/trial"),
        ("mean latency", f"{report.mean_duration_ms / 1000:.1f}s/trial"),
    ]
    if diag.dominant_label is not None:
        rows.append(
            (
                "dominant label",
                f"{diag.dominant_label} ({diag.dominant_label_share * 100:.0f}% of trials)",
            )
        )
    if diag.dominant_check is not None:
        cid, count = diag.dominant_check
        rows.append(("dominant check", f"{cid} ({count}/{report.n_trials} trials)"))
    if diag.component_signal:
        rows.append(("component signal", f"{len(diag.component_evidence)} matching messages"))
    return rows


def render_next_action(report: ResultReport, diag: Diagnosis, rec: Recommendation) -> str:
    lines: list[str] = []
    lines.append(f"# Next action — {report.bench_id} ({report.run_id})")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append(rec.summary)
    lines.append("")

    lines.append("## Evidence")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in _format_evidence_rows(report, diag):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    if report.label_counts:
        lines.append("### Failure labels")
        lines.append("")
        lines.append("| label | count |")
        lines.append("|---|---|")
        for label, count in report.label_counts.items():
            lines.append(f"| {label} | {count} |")
        lines.append("")

    if report.top_check_ids:
        lines.append("### Top failed check_ids")
        lines.append("")
        lines.append("| check_id | count |")
        lines.append("|---|---|")
        for cid, count in report.top_check_ids:
            lines.append(f"| {cid} | {count} |")
        lines.append("")

    if report.stage_counts:
        lines.append("### Failures by first-failing stage")
        lines.append("")
        lines.append("| stage | count |")
        lines.append("|---|---|")
        for stage, count in report.stage_counts.items():
            lines.append(f"| {stage} | {count} |")
        lines.append("")

    if diag.baseline_path is not None:
        lines.append(f"## Comparison vs baseline `{diag.baseline_path.name}`")
        lines.append("")
        verdict = "regression" if diag.regression else "improvement" if diag.improvement else "flat"
        lines.append(f"Verdict: **{verdict}**.")
        lines.append("")
        lines.append("| metric | baseline | current | delta |")
        lines.append("|---|---|---|---|")
        # Re-load minimal baseline numbers (already in diag.delta_summary).
        b_pass = report.pass_rate - diag.delta_summary.get("pass_rate", 0.0)
        b_cost = report.mean_cost_usd - diag.delta_summary.get("mean_cost_usd", 0.0)
        b_dur = report.mean_duration_ms - diag.delta_summary.get("mean_duration_ms", 0.0)
        lines.append(
            f"| pass rate | {b_pass * 100:.0f}% | {report.pass_rate * 100:.0f}% | "
            f"{diag.delta_summary.get('pass_rate', 0.0) * 100:+.0f} pts |"
        )
        lines.append(
            f"| mean cost | ${b_cost:.4f} | ${report.mean_cost_usd:.4f} | "
            f"${diag.delta_summary.get('mean_cost_usd', 0.0):+.4f} |"
        )
        lines.append(
            f"| mean latency | {b_dur / 1000:.1f}s | {report.mean_duration_ms / 1000:.1f}s | "
            f"{diag.delta_summary.get('mean_duration_ms', 0.0) / 1000:+.1f}s |"
        )
        lines.append("")

    if diag.component_evidence:
        lines.append("### Component / pattern evidence")
        lines.append("")
        for snippet in diag.component_evidence:
            lines.append(f"- `{snippet}`")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"- **Area:** {rec.area}")
    lines.append(f"- **Task:** {rec.title}")
    lines.append("")

    lines.append("## Implementation prompt")
    lines.append("")
    lines.append("```")
    lines.append(rec.implementation_prompt.rstrip())
    lines.append("```")
    lines.append("")

    lines.append("## Constraints")
    lines.append("")
    for c in rec.constraints:
        lines.append(f"- {c}")
    lines.append("")

    lines.append("## Stop conditions")
    lines.append("")
    for s in rec.stop_conditions:
        lines.append(f"- {s}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level entry point used by the CLI
# ---------------------------------------------------------------------------


def analyze_next(directory: Path, out_path: Path | None = None) -> Path:
    """Read the newest report under ``directory``, diagnose, and write
    ``next_action.md`` (next to the report unless ``out_path`` is given).

    Returns the path written. Raises ``FileNotFoundError`` if no report
    exists under ``directory``.
    """
    reports = find_reports(directory)
    if not reports:
        raise FileNotFoundError(f"No report.json found in {directory}")

    current = load_report(reports[-1])
    baseline = load_report(reports[-2]) if len(reports) >= 2 else None
    diag = diagnose(current, baseline)
    rec = recommend(current, diag)

    target = out_path if out_path is not None else (directory / "next_action.md")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_next_action(current, diag, rec) + "\n", encoding="utf-8")
    return target


__all__ = [
    "Diagnosis",
    "Recommendation",
    "ReportKind",
    "ResultReport",
    "analyze_next",
    "diagnose",
    "find_reports",
    "load_report",
    "recommend",
    "render_next_action",
]
