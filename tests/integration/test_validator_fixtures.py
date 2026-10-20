"""Phase 2 EXIT GATE.

The validator must:
1. Pass the good fixture cleanly (overall PASS, no failing checks).
2. Fail each broken fixture overall.
3. Fail it for the CORRECT, specific check id.
4. Introduce NO collateral failures in unrelated tiers (the defect is isolated,
   so only its tier should break).

Each broken fixture is the good fixture with exactly one defect (see
scripts/gen_broken_fixtures.py), so a collateral failure would mean a tier is
coupled to something it shouldn't be.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.report import Tier
from forge_mind.validator.run import validate

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "handwritten"


def _validate(name: str, out_dir: Path):  # type: ignore[no-untyped-def]
    program = PartProgram.model_validate(
        json.loads((FIXTURES / f"{name}.partdsl.json").read_text())
    )
    model = compile_program(program, out_dir)
    return validate(model, program)


def test_good_fixture_passes(tmp_path: Path) -> None:
    report = _validate("nema17_mount_good", tmp_path)
    assert report.passed, f"good fixture should pass; failed: {report.failed_check_ids()}"
    assert report.failed_check_ids() == []
    # Every tier present and green.
    for tier in (Tier.MESH, Tier.FEATURES, Tier.PRINTABILITY, Tier.INTERFACES):
        assert tier in report.tiers
        assert report.tiers[tier].n_failed == 0


# (fixture name, the check id that MUST fail, the tier that owns it)
BROKEN_CASES = [
    ("nema17_mount_broken_walls", "features.min_wall_thickness", Tier.FEATURES),
    (
        "nema17_mount_broken_holes",
        "interfaces.motor_face.bolt_pattern_31mm_square",
        Tier.INTERFACES,
    ),
    ("nema17_mount_broken_overhang", "printability.overhang_max_angle", Tier.PRINTABILITY),
]


@pytest.mark.parametrize(("name", "expected_failed_id", "broken_tier"), BROKEN_CASES)
def test_broken_fixture_fails_for_correct_reason(
    name: str, expected_failed_id: str, broken_tier: Tier, tmp_path: Path
) -> None:
    report = _validate(name, tmp_path)

    # Overall fails.
    assert not report.passed, f"{name} should fail overall"

    # The specific expected check failed.
    failed = report.failed_check_ids()
    assert expected_failed_id in failed, (
        f"{name}: expected {expected_failed_id!r} to fail; failed = {failed}"
    )

    # No collateral: every failing check belongs to the broken tier.
    for check in report.checks:
        if check.status.value == "fail":
            assert check.tier == broken_tier, (
                f"{name}: unexpected failure in tier {check.tier} "
                f"(check {check.id}); only {broken_tier} should fail"
            )


def test_broken_fixtures_emit_fix_hints(tmp_path: Path) -> None:
    """A failing validation should hand the Reviser something to act on."""
    for name in ("nema17_mount_broken_walls", "nema17_mount_broken_holes"):
        report = _validate(name, tmp_path)
        assert report.suggested_fixes, f"{name} produced no fix hints"
