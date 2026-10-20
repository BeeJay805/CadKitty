"""bench-003 deterministic foundation: the hand-authored NEMA17 -> 2020
adapter-plate reference compiles, passes every required validator check
(both interfaces reused from bench-001, no new checker code), and the bench
harness drives it to 10/10 trials. No LLM, no interpreter, no new pattern.
"""

from __future__ import annotations

import json
from pathlib import Path

from examples.handwritten_nema17_2020_adapter import build_adapter_plate_reference

from forge_mind.bench.criteria import load_criteria
from forge_mind.bench.runner import run_benchmark
from forge_mind.bench.runners import HandwrittenRunner
from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate

REPO = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO / "benchmarks" / "bench-003-nema17-2020-adapter-plate-v1"

REQUIRED_ADAPTER_CHECKS = [
    "mesh.watertight",
    "mesh.single_body",
    "mesh.bbox_within_max",
    "features.min_wall_thickness",
    "printability.overhang_max_angle",
    "interfaces.motor_face.bolt_holes_present",
    "interfaces.motor_face.bolt_pattern_31mm_square",
    "interfaces.motor_face.shaft_bore_present",
    "interfaces.extrusion_face.two_m5_holes",
    "interfaces.extrusion_face.m5_spacing",
]


def _validate(program: PartProgram, work: Path) -> ValidationReport:
    model = compile_program(program, work)
    return validate(model, program)


# ---------------------------------------------------------------------------
# Reference + per-check assertions
# ---------------------------------------------------------------------------


def test_adapter_reference_passes_all_required_checks(tmp_path: Path) -> None:
    report = _validate(build_adapter_plate_reference(), tmp_path)
    assert report.passed, f"unexpected failures: {report.failed_check_ids()}"
    for cid in REQUIRED_ADAPTER_CHECKS:
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_nema17_interface_checks_pass(tmp_path: Path) -> None:
    report = _validate(build_adapter_plate_reference(), tmp_path)
    for cid in (
        "interfaces.motor_face.bolt_holes_present",
        "interfaces.motor_face.bolt_pattern_31mm_square",
        "interfaces.motor_face.shaft_bore_present",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_extrusion_interface_checks_pass(tmp_path: Path) -> None:
    report = _validate(build_adapter_plate_reference(), tmp_path)
    for cid in (
        "interfaces.extrusion_face.two_m5_holes",
        "interfaces.extrusion_face.m5_spacing",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


def test_bbox_within_envelope(tmp_path: Path) -> None:
    model = compile_program(build_adapter_plate_reference(), tmp_path)
    sx, sy, sz = model.bounding_box.size.as_tuple()
    # 65 x 45 x 4 inside the 70 x 50 x 8 envelope (ADR-0008 design target).
    assert sx <= 70.0 and sy <= 50.0 and sz <= 8.0
    bbox_check = validate(model, build_adapter_plate_reference()).check("mesh.bbox_within_max")
    assert bbox_check is not None and bbox_check.status.value == "pass"


# ---------------------------------------------------------------------------
# Broken-variant checks: the existing bench-001 checkers actually catch the
# defect on a flat adapter plate (no new validator code is needed for this).
# ---------------------------------------------------------------------------


def test_wrong_m5_spacing_fails_extrusion_check(tmp_path: Path) -> None:
    # A GROSSLY-wrong spacing (30 mm on a 20 mm slot) is so far off that the misplaced holes
    # leave the localized slot region (ADR-0017), so the defect is caught EARLIER -- by the
    # count check -- and the part is still rejected. (Pre-localization this tripped m5_spacing;
    # localization re-routes a gross misplacement to two_m5_holes: the holes are nowhere near
    # the declared 20 mm slot. The good reference's checks stay byte-identical.)
    bad = build_adapter_plate_reference(tnut_spacing=30.0)
    report = _validate(bad, tmp_path)
    assert not report.passed
    count = report.check("interfaces.extrusion_face.two_m5_holes")
    assert count is not None and count.status.value == "fail"


def test_subtly_wrong_m5_spacing_still_fails_spacing_check(tmp_path: Path) -> None:
    # HONESTY CHECK (ADR-0017): a SUBTLY-wrong spacing (24 mm on a 20 mm slot) keeps the holes
    # INSIDE the localized region (24/2 = 12 <= region 12.5) but outside the 2 mm slot
    # tolerance -- so localization did NOT kill the spacing check: two_m5_holes still finds the
    # pair and m5_spacing catches the in-region misplacement.
    bad = build_adapter_plate_reference(tnut_spacing=24.0)
    report = _validate(bad, tmp_path)
    count = report.check("interfaces.extrusion_face.two_m5_holes")
    spacing = report.check("interfaces.extrusion_face.m5_spacing")
    assert count is not None and count.status.value == "pass", "holes should still be in-region"
    assert spacing is not None and spacing.status.value == "fail", "spacing check must still fire"


# ---------------------------------------------------------------------------
# Bench harness end-to-end: 10/10 trials via the handwritten runner.
# ---------------------------------------------------------------------------


def _run_harness(tmp_path: Path, n_trials: int = 10):  # type: ignore[no-untyped-def]
    criteria = load_criteria(BENCH_DIR / "pass_criteria.yaml")
    assert criteria.reference_dsl is not None
    runner = HandwrittenRunner(REPO / criteria.reference_dsl)
    return (
        run_benchmark(
            bench_dir=BENCH_DIR,
            runner=runner,
            criteria=criteria,
            run_id="test",
            output_root=tmp_path,
            n_trials=n_trials,
        ),
        criteria,
    )


def test_bench003_handwritten_passes_all_trials(tmp_path: Path) -> None:
    summary, _ = _run_harness(tmp_path, n_trials=10)
    assert summary.n_trials == 10
    assert summary.n_passed == 10, (
        f"expected 10/10, got {summary.n_passed}; "
        f"first failure: {next((t for t in summary.trials if not t.passed), None)}"
    )
    assert summary.benchmark_passed
    assert summary.pass_rate == 1.0


def test_every_required_check_is_satisfied(tmp_path: Path) -> None:
    summary, criteria = _run_harness(tmp_path, n_trials=1)
    trial = summary.trials[0]
    for required in criteria.required_checks:
        assert required not in trial.failed_check_ids


def test_history_appended(tmp_path: Path) -> None:
    _run_harness(tmp_path, n_trials=2)
    history = tmp_path / "bench-003-nema17-2020-adapter-plate-v1" / "history.jsonl"
    assert history.exists()
    rows = history.read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["n_trials"] == 2
