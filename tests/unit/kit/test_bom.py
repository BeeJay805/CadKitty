"""Bill of Materials: every line pulled from the registry, aggregated, nothing invented."""

from __future__ import annotations

from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.pi4 import M25_CLEARANCE_RADIUS_MM
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.kit.bom import build_bom
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature

_UP = Vec3(x=0.0, y=0.0, z=1.0)


def _program(
    features: list[HardwareFeature], interfaces: list[InterfaceDeclaration] | None = None
) -> PartProgram:
    """A minimal valid program (one slab) carrying the given hardware features + face interfaces."""
    return PartProgram(
        metadata=Metadata(name="kit-test-plate"),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=features,
        interfaces=interfaces or [],
    )


def _hw(category: str, key: str, *, fit: str | None = None) -> HardwareFeature:
    # nominal_mm is irrelevant to the BOM (the BOM cites the registry, not the built dim).
    return HardwareFeature(category=category, key=key, nominal_mm=0.0, fit=fit, where="t")


def _face(interface_type: InterfaceType, **params: float) -> InterfaceDeclaration:
    return InterfaceDeclaration(
        name=interface_type.value,
        interface_type=interface_type,
        location=_UP,
        normal=_UP,
        params=params,
    )


def test_no_hardware_is_an_empty_bom() -> None:
    bom = build_bom(_program([]))
    assert bom.lines == []
    assert bom.unmapped == []
    assert bom.total_items == 0
    assert bom.summary() == "no registry-backed hardware"


def test_aggregates_repeated_features_into_one_line() -> None:
    bom = build_bom(_program([_hw("clearance_hole", "M3", fit="normal")] * 4))
    assert len(bom.lines) == 1
    line = bom.lines[0]
    assert line.component == "M3 socket-head cap screw"
    assert line.quantity == 4
    assert line.kind == "fastener"
    assert bom.total_items == 4
    assert "ISO 4762" in line.source  # cited straight from the fastener registry


def test_clearance_and_counterbore_same_key_merge_to_one_screw_line() -> None:
    # The screw you buy is the same M3 SHCS whether the hole is plain or counterbored.
    bom = build_bom(
        _program(
            [
                _hw("clearance_hole", "M3", fit="normal"),
                _hw("clearance_hole", "M3", fit="normal"),
                _hw("counterbore", "M3"),
            ]
        )
    )
    screw_lines = [line for line in bom.lines if line.component == "M3 socket-head cap screw"]
    assert len(screw_lines) == 1
    assert screw_lines[0].quantity == 3


def test_distinct_components_each_get_a_line_and_correct_label() -> None:
    bom = build_bom(
        _program(
            [
                _hw("bearing_seat", "608", fit="press"),
                _hw("clearance_hole", "M3", fit="normal"),
                _hw("clearance_hole", "M3", fit="normal"),
                _hw("heat_set_insert", "M3"),
                _hw("heat_set_insert", "M3"),
                _hw("captive_nut_pocket", "M5"),
                _hw("dowel_hole", "4", fit="slip"),
                _hw("magnet_pocket", "6x3"),
            ]
        )
    )
    by_component = {line.component: line for line in bom.lines}
    assert by_component["608 deep-groove ball bearing"].quantity == 1
    assert by_component["M3 socket-head cap screw"].quantity == 2
    assert by_component["M3 heat-set threaded insert"].quantity == 2
    assert by_component["M5 hex nut"].quantity == 1
    assert by_component["Ø4 mm dowel pin"].quantity == 1
    assert by_component["6x3 disc magnet"].quantity == 1


def test_provisional_components_are_surfaced_not_hidden() -> None:
    # Heat-set inserts + magnets are PROVISIONAL (manufacturer / no single standard) in the registry.
    bom = build_bom(_program([_hw("heat_set_insert", "M3"), _hw("magnet_pocket", "6x3")]))
    assert bom.has_provisional
    for line in bom.lines:
        assert line.provisional
        assert "[PROVISIONAL]" in line.render()
    # ISO-backed fasteners/bearings are NOT provisional.
    bolt = build_bom(_program([_hw("clearance_hole", "M3", fit="normal")])).lines[0]
    assert not bolt.provisional
    assert "[PROVISIONAL]" not in bolt.render()


def test_unknown_key_is_flagged_not_invented() -> None:
    # M99 is not a real fastener size: it must be FLAGGED, never turned into a line.
    bom = build_bom(_program([_hw("clearance_hole", "M99", fit="normal")]))
    assert bom.lines == []
    assert len(bom.unmapped) == 1
    assert "M99" in bom.unmapped[0]


def test_unmapped_category_is_flagged() -> None:
    bom = build_bom(_program([_hw("mystery_feature", "X")]))
    assert bom.lines == []
    assert any("mystery_feature" in u for u in bom.unmapped)


def test_lines_are_ordered_by_kind_then_component() -> None:
    # Out-of-order input -> deterministic fastener, insert, nut, bearing, dowel, magnet order.
    bom = build_bom(
        _program(
            [
                _hw("magnet_pocket", "6x3"),
                _hw("bearing_seat", "608", fit="press"),
                _hw("clearance_hole", "M3", fit="normal"),
            ]
        )
    )
    kinds = [line.kind for line in bom.lines]
    assert kinds == ["fastener", "bearing", "magnet"]


def test_summary_renders_all_lines() -> None:
    bom = build_bom(
        _program([_hw("clearance_hole", "M3", fit="normal")] * 4 + [_hw("bearing_seat", "608")])
    )
    summary = bom.summary()
    assert "4x M3 socket-head cap screw" in summary
    assert "1x 608 deep-groove ball bearing" in summary


# --- Fix 1: face-component BOM (motor / fan / extrusion / board mount) -----------------------


def test_legacy_motor_face_bills_four_m3_screws() -> None:
    bom = build_bom(_program([], [_face(InterfaceType.NEMA17_FACE)]))  # legacy: no clearance param
    line = bom.lines[0]
    assert line.component == "M3 socket-head cap screw"
    assert line.quantity == 4
    assert "ISO 4762" in line.source
    assert bom.unmapped == []


def test_nema23_motor_face_bills_m5_screws() -> None:
    # A registry NEMA 23 carries its M5 corner-bolt clearance radius in the declaration params.
    bom = build_bom(_program([], [_face(InterfaceType.NEMA17_FACE, bolt_clearance_radius=2.6)]))
    assert bom.lines[0].component == "M5 socket-head cap screw"
    assert bom.lines[0].quantity == 4


def test_fan_face_bills_four_m3_screws() -> None:
    bom = build_bom(_program([], [_face(InterfaceType.FAN_FACE, bolt_square=32.0)]))
    assert bom.lines[0].component == "M3 socket-head cap screw"
    assert bom.lines[0].quantity == 4


def test_legacy_extrusion_bills_two_m5_tnuts_and_two_bolts() -> None:
    bom = build_bom(_program([], [_face(InterfaceType.EXTRUSION_2020_SLOT, spacing=20.0)]))
    by_component = {line.component: line for line in bom.lines}
    assert by_component["M5 T-slot nut"].quantity == 2
    assert by_component["M5 T-slot nut"].kind == "tnut"
    assert "T-slot" in by_component["M5 T-slot nut"].source  # cited from the extrusion registry
    assert by_component["M5 socket-head cap screw"].quantity == 2


def test_extrusion_3030_bills_m6_tnuts_and_bolts() -> None:
    bom = build_bom(
        _program(
            [],
            [
                _face(
                    InterfaceType.EXTRUSION_2020_SLOT,
                    spacing=30.0,
                    bolt_clearance_radius=M6_CLEARANCE_RADIUS_MM,
                )
            ],
        )
    )
    by_component = {line.component: line for line in bom.lines}
    assert by_component["M6 T-slot nut"].quantity == 2
    assert by_component["M6 socket-head cap screw"].quantity == 2


def test_board_mount_bills_four_board_screws() -> None:
    bom = build_bom(
        _program(
            [],
            [
                _face(
                    InterfaceType.PI4_MOUNT,
                    hole_dx=58.0,
                    hole_dy=49.0,
                    screw_radius=M25_CLEARANCE_RADIUS_MM,
                )
            ],
        )
    )
    assert bom.lines[0].component == "M2.5 socket-head cap screw"
    assert bom.lines[0].quantity == 4


def test_port_cutout_is_not_a_buyable() -> None:
    bom = build_bom(_program([], [_face(InterfaceType.PI4_PORTS)]))
    assert bom.lines == []
    assert bom.unmapped == []  # a cutout, not an unmapped component


def test_face_screws_merge_with_hardware_screws_of_the_same_thread() -> None:
    # A motor face (4x M3) + two M3 clearance holes -> one "6x M3 socket-head cap screw" line.
    bom = build_bom(
        _program(
            [_hw("clearance_hole", "M3", fit="normal"), _hw("clearance_hole", "M3", fit="normal")],
            [_face(InterfaceType.NEMA17_FACE)],
        )
    )
    screws = [line for line in bom.lines if line.component == "M3 socket-head cap screw"]
    assert len(screws) == 1
    assert screws[0].quantity == 6


def test_unknown_face_thread_is_flagged_not_invented() -> None:
    # A motor face with a clearance radius matching no known thread -> flagged, no line.
    bom = build_bom(_program([], [_face(InterfaceType.NEMA17_FACE, bolt_clearance_radius=9.9)]))
    assert bom.lines == []
    assert len(bom.unmapped) == 1
    assert "not a known thread" in bom.unmapped[0]


# --- Fix 2: screw-length suggestion (deterministic where known, honest where not) -------------


def _program_mw(features: list[HardwareFeature], min_wall: float | None = 4.0) -> PartProgram:
    from forge_mind.dsl.program import Constraints

    return PartProgram(
        metadata=Metadata(name="kit-test-plate"),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=features,
        constraints=Constraints(min_wall_thickness=min_wall),
    )


def test_heat_set_insert_suggests_mating_screw_length() -> None:
    # M3 insert engagement = 4 mm; 4 mm wall -> an M3x8 screw (the canonical example).
    line = build_bom(_program_mw([_hw("heat_set_insert", "M3")])).lines[0]
    assert line.length_mm == 8.0
    assert "mating M3 screw" in line.length_note
    assert "suggestion" in line.length_note
    assert "~ 8 mm" in line.render()


def test_captive_nut_suggests_mating_bolt_length() -> None:
    # M5 nut thickness = 4.7 mm; 4 mm floor -> ceil(8.7) = 9 mm bolt.
    line = build_bom(_program_mw([_hw("captive_nut_pocket", "M5")])).lines[0]
    assert line.length_mm == 9.0
    assert "mating M5 bolt" in line.length_note


def test_plain_screw_length_is_honestly_ambiguous() -> None:
    # A clearance hole into an unknown mating part: no number, say so (no-false-PASS).
    line = build_bom(_program_mw([_hw("clearance_hole", "M3", fit="normal")])).lines[0]
    assert line.length_mm is None
    assert line.length_note == "length depends on what you're mounting to"


def test_face_mounting_screws_length_is_ambiguous() -> None:
    line = build_bom(_program([], [_face(InterfaceType.NEMA17_FACE)])).lines[0]
    assert line.length_mm is None
    assert line.length_note == "length depends on what you're mounting to"


def test_non_fastener_lines_have_no_length() -> None:
    bom = build_bom(_program_mw([_hw("bearing_seat", "608"), _hw("magnet_pocket", "6x3")]))
    for line in bom.lines:
        assert line.length_mm is None
        assert line.length_note is None


def test_missing_min_wall_gives_a_note_without_a_number() -> None:
    line = build_bom(_program_mw([_hw("heat_set_insert", "M3")], min_wall=None)).lines[0]
    assert line.length_mm is None
    assert "engagement" in line.length_note  # still honest guidance, just no concrete length
