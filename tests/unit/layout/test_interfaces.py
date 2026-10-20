"""Shared place()/check() interface facts (ADR-0006 Stage 3).

Proves each unified interface has a single source of truth: the placement helper
and the validator checker(s) read the SAME nominal facts, the helper emits the
expected geometry, and that geometry passes the real checker. Covers NEMA 17
(31 mm bolt square + shaft bore) and the M5 T-nut pair (2020-extrusion face).
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.interfaces import m5 as m5_facts
from forge_mind.dsl.interfaces import nema17 as facts
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.layout.interfaces import TNutPair, place_m5_pair, place_nema17_face
from forge_mind.types import Vec3
from forge_mind.validator.interfaces import extrusion_2020_slot as extrusion_checker
from forge_mind.validator.interfaces import m5_tnut as m5_checker
from forge_mind.validator.interfaces import nema17_face as checker
from forge_mind.validator.run import validate


def _cylinders(center: Vec3, thickness: float) -> list[Cylinder]:
    ops = place_nema17_face(center, thickness)
    out: list[Cylinder] = []
    for op in ops:
        assert isinstance(op, Cylinder)  # the helper only emits cylinders
        out.append(op)
    return out


# --- placement geometry ------------------------------------------------------


def test_place_emits_four_bolts_on_31mm_square_plus_shaft() -> None:
    cyls = _cylinders(Vec3(x=0.0, y=0.0, z=2.0), 4.0)
    bolts = [c for c in cyls if c.op_id.startswith("motor_bolt")]
    shaft = [c for c in cyls if c.op_id == "shaft_clearance"]
    assert len(bolts) == 4
    assert len(shaft) == 1
    # All 4 bolts sit at +/-15.5 in both in-plane axes (the 31 mm square).
    for b in bolts:
        assert abs(abs(b.position.x) - 15.5) < 1e-9
        assert abs(abs(b.position.y) - 15.5) < 1e-9
        assert b.radius == facts.M3_CLEARANCE_RADIUS_MM
        assert b.mode is OpMode.SUBTRACT
    assert shaft[0].position.as_tuple() == (0.0, 0.0, 2.0)
    assert shaft[0].radius == facts.SHAFT_BORE_RADIUS_MM
    assert shaft[0].mode is OpMode.SUBTRACT


def test_place_translates_with_center_and_keeps_through_thickness() -> None:
    cyls = _cylinders(Vec3(x=10.0, y=5.0, z=1.5), 3.0)
    by_id = {c.op_id: c for c in cyls}
    assert by_id["motor_bolt_ne"].position.as_tuple() == (10.0 + 15.5, 5.0 + 15.5, 1.5)
    assert by_id["motor_bolt_sw"].position.as_tuple() == (10.0 - 15.5, 5.0 - 15.5, 1.5)
    assert all(c.height == 3.0 for c in cyls)


def test_op_ids_match_bench001_naming() -> None:
    ids = {c.op_id for c in _cylinders(Vec3(x=0.0, y=0.0, z=2.0), 4.0)}
    assert ids == {
        "motor_bolt_nw",
        "motor_bolt_ne",
        "motor_bolt_sw",
        "motor_bolt_se",
        "shaft_clearance",
    }


# --- single source of truth --------------------------------------------------


def test_checker_and_placement_share_the_same_facts() -> None:
    # The checker's nominal constants ARE the shared facts (same objects), and the
    # placement helper builds from the same module -- they cannot drift.
    assert checker.BOLT_PITCH is facts.BOLT_PITCH_MM
    assert checker.M3_CLEARANCE_RADIUS is facts.M3_CLEARANCE_RADIUS_MM
    assert checker.SHAFT_BORE_RADIUS is facts.SHAFT_BORE_RADIUS_MM
    # And the placed bolt square is exactly BOLT_PITCH wide (placement -> same source).
    bolts = [c for c in _cylinders(Vec3(), 4.0) if c.op_id.startswith("motor_bolt")]
    assert (
        max(b.position.x for b in bolts) - min(b.position.x for b in bolts) == facts.BOLT_PITCH_MM
    )


# --- place() output passes check() -------------------------------------------


def test_placed_face_passes_the_real_checker(tmp_path: Path) -> None:
    # A plain plate with the placed NEMA 17 face must pass all three nema17 checks
    # through the real validator -- the round-trip proof of the unification.
    w = 4.0
    base = Box(
        op_id="base",
        mode=OpMode.ADD,
        position=Vec3(x=0.0, y=0.0, z=w / 2.0),
        width=60.0,
        depth=60.0,
        height=w,
    )
    program = PartProgram(
        metadata=Metadata(name="nema17_face_only", units="mm"),
        operations=[base, *place_nema17_face(Vec3(x=0.0, y=0.0, z=w / 2.0), w)],
        interfaces=[
            InterfaceDeclaration(
                name="motor_face",
                interface_type=InterfaceType.NEMA17_FACE,
                location=Vec3(x=0.0, y=0.0, z=w / 2.0),
                normal=Vec3(x=0.0, y=0.0, z=1.0),
            )
        ],
        constraints=Constraints(
            min_wall_thickness=w, max_overhang_deg=45.0, max_bbox=Vec3(x=60.0, y=60.0, z=10.0)
        ),
    )
    report = validate(compile_program(program, tmp_path), program)
    for cid in (
        "interfaces.motor_face.bolt_holes_present",
        "interfaces.motor_face.bolt_pattern_31mm_square",
        "interfaces.motor_face.shaft_bore_present",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid


# --- M5 T-nut pair: placement geometry ---------------------------------------


def _m5_cylinders(center: Vec3, spacing: float, thickness: float) -> list[Cylinder]:
    out: list[Cylinder] = []
    for op in place_m5_pair(center, spacing, thickness):
        assert isinstance(op, Cylinder)  # the helper only emits cylinders
        out.append(op)
    return out


def test_place_m5_pair_positions_radius_and_axis() -> None:
    # Two holes offset +/- spacing/2 along X at the shared y/z, drilled +Y.
    cyls = _m5_cylinders(Vec3(x=0.0, y=18.0, z=20.0), 20.0, 4.0)
    by_id = {c.op_id: c for c in cyls}
    assert set(by_id) == {"ext_bolt_left", "ext_bolt_right"}
    assert by_id["ext_bolt_left"].position.as_tuple() == (-10.0, 18.0, 20.0)
    assert by_id["ext_bolt_right"].position.as_tuple() == (10.0, 18.0, 20.0)
    for c in cyls:
        assert c.radius == m5_facts.M5_CLEARANCE_RADIUS_MM
        assert c.height == 4.0
        assert c.mode is OpMode.SUBTRACT
        assert (c.rotation.x, c.rotation.y, c.rotation.z) == (90.0, 0.0, 0.0)


def test_place_m5_pair_is_left_then_right() -> None:
    # Operation order is part of the contract (the compiler consumes it in order).
    ids = [c.op_id for c in _m5_cylinders(Vec3(), 20.0, 4.0)]
    assert ids == ["ext_bolt_left", "ext_bolt_right"]


def test_place_m5_pair_translates_with_center() -> None:
    by_id = {c.op_id: c for c in _m5_cylinders(Vec3(x=5.0, y=-2.0, z=7.0), 30.0, 3.0)}
    assert by_id["ext_bolt_left"].position.as_tuple() == (5.0 - 15.0, -2.0, 7.0)
    assert by_id["ext_bolt_right"].position.as_tuple() == (5.0 + 15.0, -2.0, 7.0)


# --- M5 single source of truth -----------------------------------------------


def test_m5_checkers_and_placement_share_the_same_fact() -> None:
    # Both the m5_tnut checker and the extrusion_2020_slot checker resolve their
    # M5 clearance radius to the one shared fact object; placement builds from it.
    assert m5_checker.M5_CLEARANCE_RADIUS is m5_facts.M5_CLEARANCE_RADIUS_MM
    assert extrusion_checker.M5_CLEARANCE_RADIUS is m5_facts.M5_CLEARANCE_RADIUS_MM
    placed = _m5_cylinders(Vec3(), 20.0, 4.0)
    assert all(c.radius == m5_facts.M5_CLEARANCE_RADIUS_MM for c in placed)


# --- M5 axis generalisation (ADR-0008): the bench-003 flat orientation ------


def test_place_m5_pair_along_y_drill_z_axis_and_offsets() -> None:
    # ALONG_Y_DRILL_Z: holes spaced along Y, drilled +Z (no rotation).
    ops = place_m5_pair(
        Vec3(x=-22.0, y=0.0, z=2.0),
        20.0,
        4.0,
        orientation=TNutPair.ALONG_Y_DRILL_Z,
        left_op_id="ext_bolt_back",
        right_op_id="ext_bolt_front",
    )
    by_id = {c.op_id: c for c in ops if isinstance(c, Cylinder)}
    assert set(by_id) == {"ext_bolt_back", "ext_bolt_front"}
    assert by_id["ext_bolt_back"].position.as_tuple() == (-22.0, -10.0, 2.0)
    assert by_id["ext_bolt_front"].position.as_tuple() == (-22.0, 10.0, 2.0)
    for c in by_id.values():
        assert c.radius == m5_facts.M5_CLEARANCE_RADIUS_MM
        assert c.height == 4.0
        assert c.mode is OpMode.SUBTRACT
        # No rotation: the default Z-axis cylinder is what we want.
        assert (c.rotation.x, c.rotation.y, c.rotation.z) == (0.0, 0.0, 0.0)


def test_place_m5_pair_default_orientation_unchanged() -> None:
    # The default keeps bench-001's exact behaviour (no breaking change).
    default = place_m5_pair(Vec3(), 20.0, 4.0)
    explicit = place_m5_pair(Vec3(), 20.0, 4.0, orientation=TNutPair.ALONG_X_DRILL_Y)
    by_id_default = {c.op_id: c for c in default if isinstance(c, Cylinder)}
    by_id_explicit = {c.op_id: c for c in explicit if isinstance(c, Cylinder)}
    assert by_id_default.keys() == by_id_explicit.keys()
    for op_id in by_id_default:
        d, e = by_id_default[op_id], by_id_explicit[op_id]
        assert d.position.as_tuple() == e.position.as_tuple()
        assert (d.rotation.x, d.rotation.y, d.rotation.z) == (
            e.rotation.x,
            e.rotation.y,
            e.rotation.z,
        )


def test_along_y_drill_z_placed_pair_passes_extrusion_checker(tmp_path: Path) -> None:
    # Round-trip: a flat plate with the new orientation's pair passes the
    # extrusion_2020_slot checker (no new validator code needed).
    w = 4.0
    base = Box(
        op_id="plate",
        mode=OpMode.ADD,
        position=Vec3(x=0.0, y=0.0, z=w / 2.0),
        width=60.0,
        depth=45.0,
        height=w,
    )
    pair = place_m5_pair(
        Vec3(x=0.0, y=0.0, z=w / 2.0),
        20.0,
        w,
        orientation=TNutPair.ALONG_Y_DRILL_Z,
        left_op_id="ext_bolt_back",
        right_op_id="ext_bolt_front",
    )
    program = PartProgram(
        metadata=Metadata(name="m5_pair_flat_only", units="mm"),
        operations=[base, *pair],
        interfaces=[
            InterfaceDeclaration(
                name="extrusion_face",
                interface_type=InterfaceType.EXTRUSION_2020_SLOT,
                location=Vec3(x=0.0, y=0.0, z=0.0),
                normal=Vec3(x=0.0, y=0.0, z=-1.0),
            )
        ],
        constraints=Constraints(
            min_wall_thickness=w, max_overhang_deg=45.0, max_bbox=Vec3(x=70.0, y=50.0, z=10.0)
        ),
    )
    report = validate(compile_program(program, tmp_path), program)
    for cid in (
        "interfaces.extrusion_face.two_m5_holes",
        "interfaces.extrusion_face.m5_spacing",
    ):
        check = report.check(cid)
        assert check is not None and check.status.value == "pass", cid
