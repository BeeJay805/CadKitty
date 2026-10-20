"""Run N end-to-end trials and produce a failure dataset.

Measurement only: runs the existing pipeline unchanged, saves every artifact, and
labels each trial. No prompt/DSL/revision changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from forge_mind.analysis.records import FailureDataset, TrialRecord, aggregate, analyze_result
from forge_mind.analysis.report import write_dataset
from forge_mind.analysis.taxonomy import Expectations, FailureLabel
from forge_mind.pipeline import run_pipeline


def load_expectations(bench_dir: Path) -> Expectations:
    """Read the ground-truth requirements used to root-cause failures."""
    gt = json.loads((bench_dir / "ground_truth.json").read_text())
    required = {ri["type"] for ri in gt.get("required_interfaces", [])}

    envelope = (60.0, 60.0, 40.0)
    min_wall = 4.0
    for dim in gt.get("dimensions", []):
        if dim["feature"] == "bounding_box" and isinstance(dim.get("value_mm"), list):
            v = dim["value_mm"]
            envelope = (float(v[0]), float(v[1]), float(v[2]))
        elif dim["feature"] == "wall_thickness":
            min_wall = float(dim["value_mm"])

    return Expectations(
        required_interface_types=required,
        envelope_cap_mm=envelope,
        min_wall_mm=min_wall,
    )


def _save_artifacts(result, trial_dir: Path, duration_ms: float) -> dict[str, str]:  # type: ignore[no-untyped-def]
    """Persist every intermediate so failures are inspectable. Returns paths."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {"dir": str(trial_dir)}

    if result.intent is not None:
        p = trial_dir / "intent.json"
        p.write_text(result.intent.model_dump_json(indent=2), encoding="utf-8")
        artifacts["intent"] = str(p)
    if result.blueprint is not None:
        p = trial_dir / "blueprint.json"
        p.write_text(result.blueprint.model_dump_json(indent=2), encoding="utf-8")
        artifacts["blueprint"] = str(p)
    if result.program is not None:
        p = trial_dir / "part.partdsl.json"
        p.write_text(result.program.model_dump_json(indent=2), encoding="utf-8")
        artifacts["partdsl"] = str(p)
    if result.model is not None:
        artifacts["stl"] = str(result.model.stl_path)
    if result.report is not None:
        p = trial_dir / "report.json"
        p.write_text(result.report.model_dump_json(indent=2), encoding="utf-8")
        artifacts["report"] = str(p)

    telemetry = {
        "total_duration_ms": duration_ms,
        "total_cost_usd": result.total_cost_usd,
        "total_tokens": result.total_tokens,
        "failed_stage": result.failed_stage,
        "usages": [asdict(u) for u in result.usages],
    }
    tp = trial_dir / "telemetry.json"
    tp.write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    artifacts["telemetry"] = str(tp)

    if result.model is not None:
        try:
            from forge_mind.viz.render import render_part_to_png

            png = render_part_to_png(result.model.stl_path, trial_dir / "part.png")
            artifacts["render"] = str(png)
        except Exception:
            pass

    return artifacts


def run_failure_analysis(
    bench_dir: Path,
    n_trials: int,
    output_root: Path,
    run_id: str,
    *,
    progress: bool = True,
    use_layoutspec: bool = False,
) -> FailureDataset:
    """Run ``n_trials`` end-to-end trials and write a failure dataset.

    ``use_layoutspec`` routes the Blueprint->program step through the LayoutSpec
    path (matching ``forge-mind run``); default off keeps measuring direct-DSL.
    """
    prompt = (bench_dir / "prompt.txt").read_text(encoding="utf-8").strip()
    expectations = load_expectations(bench_dir)

    run_dir = output_root / bench_dir.name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    live_jsonl = run_dir / "records.live.jsonl"

    records: list[TrialRecord] = []
    for i in range(n_trials):
        trial_dir = run_dir / f"trial-{i}"
        t0 = time.perf_counter()
        try:
            # Pass the flag only when set, so the default path's call site stays
            # 2-arg identical to before (the direct-DSL measurement and its test
            # double contract are unchanged).
            result = (
                run_pipeline(prompt, trial_dir, use_layoutspec=True)
                if use_layoutspec
                else run_pipeline(prompt, trial_dir)
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            artifacts = _save_artifacts(result, trial_dir, duration_ms)
            record = analyze_result(i, result, trial_dir, expectations, duration_ms, artifacts)
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            record = TrialRecord(
                trial_index=i,
                passed=False,
                first_failing_stage="harness",
                primary_label=FailureLabel.UNKNOWN.value,
                primary_check_id=None,
                validator_message=None,
                op_id=None,
                failed_check_ids=[],
                cost_usd=0.0,
                duration_ms=duration_ms,
                total_tokens=0,
                artifacts={"dir": str(trial_dir)},
                error=f"{type(exc).__name__}: {exc}",
            )
        records.append(record)

        # Append live so a crash mid-run keeps partial data.
        with live_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record)) + "\n")
        if progress:
            verdict = "PASS" if record.passed else f"FAIL/{record.primary_label}"
            print(f"  trial {i}: {verdict}  ${record.cost_usd:.4f}  {duration_ms / 1000:.1f}s")

    dataset = aggregate(bench_dir.name, run_id, records)
    write_dataset(dataset, run_dir)
    return dataset


__all__ = ["load_expectations", "run_failure_analysis"]
