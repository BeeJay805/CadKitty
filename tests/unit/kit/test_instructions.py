"""Make-it instructions + the combined Kit: deterministic, advisory, never a gate."""

from __future__ import annotations

from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.kit import build_kit
from forge_mind.kit.bom import build_bom
from forge_mind.kit.instructions import build_instructions
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature
from forge_mind.validator.report import (
    Check,
    CheckStatus,
    Tier,
    ValidationReport,
    build_report,
)

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _program(
    features: list[HardwareFeature],
    *,
    min_wall: float | None = 4.0,
    orientation: Vec3 = _UP_Z,
) -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="kit-test-plate", print_orientation=orientation),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=features,
        constraints=Constraints(min_wall_thickness=min_wall),
    )


def _hw(category: str, key: str, *, fit: str | None = None) -> HardwareFeature:
    return HardwareFeature(category=category, key=key, nominal_mm=0.0, fit=fit, where="t")


def _report(overhang_status: CheckStatus) -> ValidationReport:
    return build_report(
        [
            Check(
                id="printability.overhang_max_angle",
                tier=Tier.PRINTABILITY,
                status=overhang_status,
                message="overhang",
                actual="30 deg",
            )
        ],
        [],
    )


def test_instructions_are_always_advisory() -> None:
    program = _program([])
    instr = build_instructions(program, build_bom(program))
    assert instr.advisory is True
    assert "never a validation gate" in instr.caveat
    assert instr.print_orientation == Vec3(x=0.0, y=0.0, z=1.0)


def test_material_is_passed_through() -> None:
    program = _program([])
    instr = build_instructions(program, build_bom(program), material="PETG")
    assert instr.material == "PETG"
    assert "PETG" in instr.steps[0]


def test_min_wall_drives_perimeter_guidance() -> None:
    program = _program([], min_wall=4.0)
    instr = build_instructions(program, build_bom(program))
    assert instr.recommended_min_wall_mm == 4.0
    assert "4 mm" in instr.wall_note  # ceil(4.0/0.4) = 10 perimeters
    assert "10 perimeters" in instr.wall_note


def test_no_min_wall_has_a_graceful_note() -> None:
    program = _program([], min_wall=None)
    instr = build_instructions(program, build_bom(program))
    assert instr.recommended_min_wall_mm is None
    assert "No explicit minimum wall" in instr.wall_note


def test_insert_note_present_only_with_inserts() -> None:
    with_inserts = _program([_hw("heat_set_insert", "M3")])
    instr = build_instructions(with_inserts, build_bom(with_inserts))
    assert instr.insert_install_note is not None
    assert "PROVISIONAL" in instr.insert_install_note
    assert any("insert" in s for s in instr.steps)

    without = _program([_hw("clearance_hole", "M3", fit="normal")])
    instr2 = build_instructions(without, build_bom(without))
    assert instr2.insert_install_note is None


def test_steps_follow_assembly_order_for_each_hardware_kind() -> None:
    program = _program(
        [
            _hw("heat_set_insert", "M3"),
            _hw("bearing_seat", "608", fit="press"),
            _hw("magnet_pocket", "6x3"),
            _hw("captive_nut_pocket", "M5"),
            _hw("clearance_hole", "M3", fit="normal"),
            _hw("dowel_hole", "4", fit="slip"),
        ]
    )
    steps = build_instructions(program, build_bom(program)).steps
    joined = " | ".join(steps)
    # print first, then inserts, bearings, dowels, magnets, nuts, fasteners.
    assert steps[0].startswith("Print the part")
    assert joined.index("Heat-set") < joined.index("Press")
    assert joined.index("Press") < joined.index("magnet")
    assert joined.index("hex nut") < joined.index("screw")


def test_orientation_note_uses_passing_report() -> None:
    program = _program([])
    instr = build_instructions(program, build_bom(program), report=_report(CheckStatus.PASS))
    assert "support-free" in instr.orientation_note
    assert "+Z up" in instr.orientation_note


def test_orientation_note_warns_on_failing_overhang() -> None:
    program = _program([])
    instr = build_instructions(program, build_bom(program), report=_report(CheckStatus.FAIL))
    assert "flagged" in instr.orientation_note
    assert "support" in instr.orientation_note


def test_non_axis_orientation_renders_as_vector() -> None:
    program = _program([], orientation=Vec3(x=0.0, y=1.0, z=1.0))
    instr = build_instructions(program, build_bom(program))
    assert "(0, 1, 1)" in instr.orientation_note


def test_build_kit_bundles_bom_and_instructions() -> None:
    program = _program(
        [_hw("bearing_seat", "608", fit="press"), _hw("clearance_hole", "M3", fit="normal")]
    )
    kit = build_kit(program, material="ABS", report=_report(CheckStatus.PASS))
    assert kit.part_name == "kit-test-plate"
    assert kit.bom.total_items == 2
    assert kit.instructions.material == "ABS"
    assert "support-free" in kit.instructions.orientation_note
    # the whole kit round-trips as JSON for the UI.
    assert "608 deep-groove ball bearing" in kit.model_dump_json()
