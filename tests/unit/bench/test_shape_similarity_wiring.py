"""Wiring of shape-similarity metrics into the benchmark harness (reported, not gated)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.bench.report import render_markdown
from forge_mind.bench.runner import _compile_reference, _trial_shape_similarity
from forge_mind.bench.types import BenchmarkSummary, TrialResult
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import Metadata, PartProgram

# These compile reference solids + sample meshes for similarity (~2s each, above the 1s `slow`
# threshold). Marked slow to keep the inner loop fast; still in the full `-m "not llm"` gate.
pytestmark = pytest.mark.slow


def _box_stl(name: str, w: float, d: float, h: float, out: Path) -> Path:
    program = PartProgram(
        metadata=Metadata(name=name, units="mm"),
        operations=[Box(op_id="b", width=w, depth=d, height=h)],
    )
    return compile_program(program, out).stl_path


def test_shape_similarity_identical_mesh(tmp_path: Path) -> None:
    stl = _box_stl("box", 10.0, 20.0, 4.0, tmp_path / "a")
    ss = _trial_shape_similarity(stl, stl)
    assert ss is not None
    assert ss["chamfer_distance_mm"] < 0.2  # same surface -> ~0
    assert ss["volumetric_iou"] > 0.95  # same occupancy -> ~1
    assert ss["f1_at_threshold"] > 0.95


def test_shape_similarity_differs_for_different_shapes(tmp_path: Path) -> None:
    gen = _box_stl("gen", 10.0, 20.0, 4.0, tmp_path / "gen")
    ref = _box_stl("ref", 20.0, 20.0, 8.0, tmp_path / "ref")  # double the size in two dims
    ss = _trial_shape_similarity(gen, ref)
    assert ss is not None
    assert ss["chamfer_distance_mm"] > 0.5  # genuinely different shape
    assert ss["volumetric_iou"] < 0.9


def test_shape_similarity_none_without_reference(tmp_path: Path) -> None:
    stl = _box_stl("box", 10.0, 10.0, 4.0, tmp_path / "a")
    assert _trial_shape_similarity(stl, None) is None
    assert _trial_shape_similarity(None, stl) is None


def test_compile_reference(tmp_path: Path) -> None:
    ref_dsl = tmp_path / "ref.partdsl.json"
    program = PartProgram(
        metadata=Metadata(name="ref", units="mm"),
        operations=[Box(op_id="b", width=8.0, depth=8.0, height=4.0)],
    )
    ref_dsl.write_text(program.model_dump_json())
    ref_stl = _compile_reference(ref_dsl, tmp_path / "out")
    assert ref_stl is not None and ref_stl.exists()
    # No reference -> None (the metric is conditional on a reference existing).
    assert _compile_reference(None, tmp_path / "none") is None


def test_markdown_renders_shape_section_only_when_present() -> None:
    def _summary(shape: dict[str, float] | None) -> BenchmarkSummary:
        trial = TrialResult(
            trial_index=0,
            passed=True,
            overall_status="pass",
            first_failing_stage=None,
            failed_check_ids=[],
            criteria_results={"validation_pass": True},
            tier_pass_counts={"mesh": "6/6"},
            iterations=1,
            total_duration_ms=1.0,
            total_tokens=0,
            total_cost_usd=0.0,
            artifacts={},
            shape_similarity=shape,
        )
        return BenchmarkSummary(
            bench_id="b",
            runner="handwritten",
            run_id="r",
            n_trials=1,
            n_passed=1,
            pass_rate=1.0,
            min_passing_trials=1,
            benchmark_passed=True,
            mean_duration_ms=1.0,
            mean_tokens=0.0,
            mean_cost_usd=0.0,
            trials=[trial],
        )

    with_md = render_markdown(
        _summary({"chamfer_distance_mm": 0.0, "volumetric_iou": 1.0, "f1_at_threshold": 1.0})
    )
    assert "Shape similarity vs reference" in with_md
    # Absent when no trial carries scores (existing benches without a reference are unchanged).
    assert "Shape similarity vs reference" not in render_markdown(_summary(None))


def test_report_json_includes_shape_similarity_field(tmp_path: Path) -> None:
    # asdict serialization carries the new field (null when absent) without other changes.
    from forge_mind.bench.report import write_reports

    trial = TrialResult(
        trial_index=0,
        passed=True,
        overall_status="pass",
        first_failing_stage=None,
        failed_check_ids=[],
        criteria_results={},
        tier_pass_counts={},
        iterations=1,
        total_duration_ms=1.0,
        total_tokens=0,
        total_cost_usd=0.0,
        artifacts={},
        shape_similarity={
            "chamfer_distance_mm": 0.1,
            "volumetric_iou": 0.9,
            "f1_at_threshold": 0.95,
        },
    )
    summary = BenchmarkSummary(
        bench_id="b",
        runner="r",
        run_id="r",
        n_trials=1,
        n_passed=1,
        pass_rate=1.0,
        min_passing_trials=1,
        benchmark_passed=True,
        mean_duration_ms=1.0,
        mean_tokens=0.0,
        mean_cost_usd=0.0,
        trials=[trial],
    )
    _md, json_path = write_reports(summary, tmp_path)
    data = json.loads(json_path.read_text())
    assert data["trials"][0]["shape_similarity"]["volumetric_iou"] == 0.9
