"""Coverage expansion (row 61 follow-ups): pin-transparent loop detection, the four-bar
driver, unowned-cycle claim withholding, and the sequenced-DOF retry.

The scissor hid a textbook linkage behind 'the pin is a part'; the ladder's lock blocked
the fold BY DESIGN. These tests pin each new coverage seam on both its accept and refuse
sides -- no phantom verdicts in either direction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody
from forge_mind.types import Vec3
from forge_mind.validator.motion_sweep import (
    ChainSweeper,
    collapse_pin_edges,
    four_bar_systems,
    undriven_loop_joint_ids,
)


def _rev(a: str, b: str, rng: tuple[float, float] = (0, 360)) -> Joint:
    return Joint(type=JointType.REVOLUTE, a=a, b=b, range_deg=rng)


def _coax(a: str, b: str) -> Joint:
    return Joint(type=JointType.COAXIAL_INSERT, a=a, b=b)


# ---------- collapse_pin_edges ----------------------------------------------------------


def test_collapse_folds_a_link_pin_link_pivot() -> None:
    j1, j2 = _rev("linkA.bore", "pin.axis"), _coax("pin.axis", "linkB.bore")
    collapsed, hosts, origin = collapse_pin_edges([j1, j2])
    assert len(collapsed) == 1
    synth = collapsed[0]
    assert synth.type is JointType.REVOLUTE
    assert {synth.a_part(), synth.b_part()} == {"linkA", "linkB"}
    assert hosts == {"pin": "linkA"}
    assert origin[id(synth)] == (j1, j2)


def test_collapse_leaves_shared_and_slotted_pins_alone() -> None:
    shared = [  # a radial's shared crankpin: 3 partners -> untouched
        _rev("crank.pin", "rodA.big"),
        _rev("crank.pin", "rodB.big"),
        _rev("crank.pin", "rodC.big"),
    ]
    assert collapse_pin_edges(shared)[0] == shared
    slotted = [  # a pin that also slides is NOT a pure pivot
        _rev("linkA.bore", "pin.axis"),
        Joint(type=JointType.PRISMATIC, a="pin.axis", b="linkB.slot", range_mm=(0, 10)),
    ]
    assert collapse_pin_edges(slotted)[0] == slotted


# ---------- four-bar detection -----------------------------------------------------------


def _four_bar_joints(swap: bool = False) -> list[Joint]:
    def j(a: str, b: str) -> Joint:
        return _rev(b, a) if swap else _rev(a, b)

    return [
        j("frame.crank_pivot", "crank.journal"),
        j("crank.crankpin", "coupler.end_a"),
        j("coupler.end_b", "rocker.top"),
        j("rocker.bottom", "frame.rocker_pivot"),
        # a pure four-bar is topologically symmetric; the frame is disambiguated by its
        # higher full-graph degree, exactly like a real base (feet, mounts, more loops)
        Joint(type=JointType.FIXED, a="frame.foot", b="foot_pad.top"),
    ]


def test_four_bar_detection_is_side_order_agnostic() -> None:
    for swap in (False, True):
        systems = four_bar_systems(_four_bar_joints(swap))
        assert len(systems) == 1, f"swap={swap}"
        _j0, frame, crank, loops = systems[0]
        assert frame == "frame" and crank == "crank"
        assert loops[0].coupler == "coupler" and loops[0].rocker == "rocker"


def test_pin_jointed_four_bar_detected_after_collapse() -> None:
    """The scissor lesson end-to-end at the topology level: every pivot through its own
    pin part -> raw graph matches nothing; the collapsed graph is a textbook four-bar."""
    raw = [
        _rev("frame.p0", "pin0.axis"),
        _coax("pin0.axis", "crank.journal"),
        _rev("crank.pin_seat", "pin1.axis"),
        _coax("pin1.axis", "coupler.end_a"),
        _rev("coupler.end_b", "pin2.axis"),
        _coax("pin2.axis", "rocker.top"),
        _rev("rocker.bottom", "pin3.axis"),
        _coax("pin3.axis", "frame.p3"),
    ]
    assert four_bar_systems(raw) == []
    collapsed, hosts, _origin = collapse_pin_edges(raw)
    systems = four_bar_systems(collapsed)
    assert len(systems) == 1
    assert set(hosts) == {"pin0", "pin1", "pin2", "pin3"}


# ---------- unowned-cycle withholding ---------------------------------------------------


def test_triangle_loop_is_withheld_and_open_chain_is_not() -> None:
    tri = [_rev("a.p", "b.p"), _rev("b.q", "c.p"), _rev("c.q", "a.q")]
    withheld = undriven_loop_joint_ids(tri, owned_ids=set())
    assert withheld == {id(j) for j in tri}
    chain = [_rev("a.p", "b.p"), _rev("b.q", "c.p")]
    assert undriven_loop_joint_ids(chain, owned_ids=set()) == set()


def test_owned_loops_are_not_withheld() -> None:
    tri = [_rev("a.p", "b.p"), _rev("b.q", "c.p"), _rev("c.q", "a.q")]
    owned = {id(tri[0]), id(tri[1]), id(tri[2])}
    assert undriven_loop_joint_ids(tri, owned_ids=owned) == set()


def test_fixed_edge_closing_a_loop_still_counts() -> None:
    """A cycle closed through a FIXED weld is still rigid closure: its two moving
    joints cannot be tree-swept honestly."""
    loop = [
        _rev("base.p", "armA.p"),
        _rev("armA.q", "armB.p"),
        Joint(type=JointType.FIXED, a="armB.q", b="base.q"),
    ]
    withheld = undriven_loop_joint_ids(loop, owned_ids=set())
    assert withheld == {id(loop[0]), id(loop[1])}


# ---------- four-bar driver on real geometry --------------------------------------------
# Planar fixture, crank axis = X. Crank pivot at origin; rocker frame pivot at (y=40,z=0).
# crank r=15, coupler 50, rocker 35, ground 40 -> Grashof (15+50 < 35+40): full rotation.

_PIN_R = 2.5
_BORE_R = 2.9  # 0.4 mm running fit


def _cyl_x(radius: float, x0: float, x1: float, y: float, z: float) -> trimesh.Trimesh:
    c = trimesh.creation.cylinder(radius=radius, height=x1 - x0, sections=48)
    c.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, y, z])
    return c


def _ring_x(x0: float, x1: float, y: float, z: float, outer: float = 5.0) -> trimesh.Trimesh:
    ring = trimesh.boolean.difference(
        [_cyl_x(outer, x0, x1, y, z), _cyl_x(_BORE_R, x0 - 1, x1 + 1, y, z)]
    )
    assert isinstance(ring, trimesh.Trimesh) and ring.is_watertight
    return ring


def _bar(p: tuple[float, float], q: tuple[float, float], x0: float, x1: float) -> trimesh.Trimesh:
    """A box link from 2D point p to q (in the YZ plane), spanning [x0,x1] in X."""
    py, pz = p
    qy, qz = q
    length = math.hypot(qy - py, qz - pz)
    b = trimesh.creation.box(extents=[x1 - x0, 4.0, max(length - 12.0, 4.0)])
    ang = math.atan2(qy - py, qz - pz)
    b.apply_transform(trimesh.transformations.rotation_matrix(-ang, [1, 0, 0]))
    b.apply_translation([(x0 + x1) / 2, (py + qy) / 2, (pz + qz) / 2])
    return b


# home geometry: crankpin at (0,15); rocker end = circle-circle intersection
_PIN_HOME = (0.0, 15.0)
_R0 = (40.0, 0.0)
_D12 = math.hypot(_R0[0] - _PIN_HOME[0], _R0[1] - _PIN_HOME[1])
_A = (50.0**2 - 35.0**2 + _D12**2) / (2 * _D12)
_H = math.sqrt(50.0**2 - _A**2)
_E1 = ((_R0[0] - _PIN_HOME[0]) / _D12, (_R0[1] - _PIN_HOME[1]) / _D12)
# e2 = cross(x_axis, e1) in the YZ plane -> (-e1_z, e1_y) in (y,z) coords
_RE = (
    _PIN_HOME[0] + _E1[0] * _A - _E1[1] * _H,
    _PIN_HOME[1] + _E1[1] * _A + _E1[0] * _H,
)


def _port(name: str, y: float, z: float, x: float = 0.0) -> Port:
    return Port(
        name=name, origin=Vec3(x=x, y=y, z=z), axis=Vec3(x=1, y=0, z=0), normal=Vec3(x=0, y=0, z=1)
    )


def _union(parts: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    """Rigid-body union for driver fixtures. Shells may be DISJOINT on purpose: a rigid
    transform moves the whole mesh, and the sweeper only samples/contains -- physical
    connectedness is irrelevant to the pose math being validated here."""
    out = trimesh.boolean.union(parts)
    assert isinstance(out, trimesh.Trimesh) and out.is_watertight
    return out


def _fb_model(obstacle: trimesh.Trimesh | None = None) -> AssemblyModel:
    frame_shells = [
        _cyl_x(_PIN_R, 8.0, 18.0, 0.0, 0.0),  # crank pivot pin
        _cyl_x(_PIN_R, 28.0, 38.0, *_R0),  # rocker pivot pin
    ]
    plate = trimesh.creation.box(extents=[40.0, 160.0, 5.0])
    plate.apply_translation([20.0, 20.0, -70.0])
    frame_shells.append(plate)
    if obstacle is not None:
        frame_shells.append(obstacle)
    frame = _union(frame_shells)
    crank = _union(
        [
            _ring_x(10.0, 16.0, 0.0, 0.0),
            _bar((0.0, 0.0), _PIN_HOME, 10.0, 16.0),
            _cyl_x(_PIN_R, 18.0, 28.0, *_PIN_HOME),
        ]
    )
    coupler = _union(
        [
            _ring_x(20.0, 26.0, *_PIN_HOME),
            _bar(_PIN_HOME, _RE, 20.0, 26.0),
            _cyl_x(_PIN_R, 28.0, 38.0, *_RE),
        ]
    )
    rocker = _union(
        [
            _ring_x(30.0, 36.0, *_RE),
            _bar(_RE, _R0, 30.0, 36.0),
            _ring_x(30.0, 36.0, *_R0),
        ]
    )

    def body(pid: str, mesh: trimesh.Trimesh, ports: list[Port]) -> PlacedBody:
        return PlacedBody(
            id=pid,
            program=None,  # type: ignore[arg-type]
            cad=None,  # type: ignore[arg-type]
            mesh=mesh,
            transform=np.eye(4),
            ports=ports,
        )

    return AssemblyModel(
        bodies=[
            body("frame", frame, [_port("crank_pivot", 0, 0), _port("rocker_pivot", *_R0)]),
            body("crank", crank, [_port("journal", 0, 0), _port("crankpin", *_PIN_HOME)]),
            body("coupler", coupler, [_port("end_a", *_PIN_HOME), _port("end_b", *_RE)]),
            body("rocker", rocker, [_port("top", *_RE), _port("bottom", *_R0)]),
        ],
        joints=_four_bar_joints(),
    )


@pytest.mark.slow
def test_four_bar_clean_full_rotation_passes() -> None:
    model = _fb_model()
    sweeper = ChainSweeper(model)
    assert sweeper.all_sampled_ok()
    (j0, _frame, crank_id, loops) = four_bar_systems(model.joints)[0]
    ok, detail = sweeper.sweep_four_bar(j0, crank_id, loops, 0.0, 360.0)
    assert ok is True, detail
    assert "four-bar coupled" in detail or "loop-closure" in detail


@pytest.mark.slow
def test_four_bar_obstruction_fails_with_depth() -> None:
    """A post inside the crank's swept disk (R<=17 about the pivot), clear of the home
    pose (arm at home occupies y ~ 0 +-2; ring reaches R5): the rotating arm must strike
    it mid-cycle and the detail must carry pair + angle + measured depth."""
    post = trimesh.creation.box(extents=[4.0, 6.0, 6.0])
    post.apply_translation([13.0, 10.0, -5.0])  # x in the arm's 10..16 lane
    model = _fb_model(obstacle=post)
    sweeper = ChainSweeper(model)
    (j0, _frame, crank_id, loops) = four_bar_systems(model.joints)[0]
    ok, detail = sweeper.sweep_four_bar(j0, crank_id, loops, 0.0, 360.0)
    assert ok is False
    assert "hits frame at" in detail and "depth" in detail


@pytest.mark.slow
def test_four_bar_unclosable_range_earns_no_claim() -> None:
    """Shrink the coupler's declared geometry by moving the rocker pivot far away: the
    circles cannot intersect through the range -> (None, reason), never a verdict."""
    model = _fb_model()
    # rebuild joints against a bogus far-away rocker pivot port on the frame
    far = _port("rocker_pivot", 200.0, 0.0)
    frame = model.bodies[0]
    frame.ports[1] = far
    sweeper = ChainSweeper(model)
    (j0, _frame, crank_id, loops) = four_bar_systems(model.joints)[0]
    ok, detail = sweeper.sweep_four_bar(j0, crank_id, loops, 0.0, 360.0)
    assert ok is None
    assert "cannot close" in detail or "not drivable" in detail


# ---------- sequenced-DOF retry (the ladder's lock, end to end) --------------------------


def _seq_model() -> AssemblyModel:
    """A door that swings 0..120 deg about the origin and a latch pivoted at (y=0,z=45)
    whose home pose hangs INTO the door's swept annulus (blocking ~4..14 deg); at its 90
    deg extreme the latch swings fully outside the annulus (r > 40). The door alone must
    FAIL; the sequenced retry (latch posed at its extreme) must turn it into a PASS that
    names the required order."""
    base = _union(
        [
            _cyl_x(_PIN_R, 8.0, 18.0, 0.0, 0.0),  # door hinge pin
            _cyl_x(_PIN_R, 8.0, 18.0, 0.0, 45.0),  # latch pivot pin
            _plate_at((13.0, 0.0, -60.0)),
        ]
    )
    door_panel = trimesh.creation.box(extents=[6.0, 4.0, 30.0])
    door_panel.apply_translation([13.0, 0.0, 22.0])
    door = _union([_ring_x(10.0, 16.0, 0.0, 0.0), door_panel])
    latch = _union([_ring_x(10.0, 16.0, 0.0, 45.0), _bar((0.0, 45.0), (5.0, 20.0), 10.0, 16.0)])

    def body(pid: str, mesh: trimesh.Trimesh, ports: list[Port]) -> PlacedBody:
        return PlacedBody(
            id=pid,
            program=None,  # type: ignore[arg-type]
            cad=None,  # type: ignore[arg-type]
            mesh=mesh,
            transform=np.eye(4),
            ports=ports,
        )

    return AssemblyModel(
        bodies=[
            body("base", base, [_port("door_pivot", 0, 0), _port("latch_pivot", 0, 45)]),
            body("door", door, [_port("hinge", 0, 0)]),
            body("latch", latch, [_port("hinge", 0, 45)]),
        ],
        joints=[
            _rev("base.door_pivot", "door.hinge", rng=(0, 120)),
            _rev("base.latch_pivot", "latch.hinge", rng=(0, 90)),
        ],
    )


def _plate_at(center: tuple[float, float, float]) -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=[40.0, 160.0, 5.0])
    plate.apply_translation(list(center))
    return plate


@pytest.mark.slow
def test_sequenced_dof_retry_names_the_required_order() -> None:
    from forge_mind.validator.assembly_checks import validate_assembly

    model = _seq_model()
    report = validate_assembly(model)
    door_check = report.check("assembly.sweep.base__door")
    assert door_check is not None
    assert door_check.status.value == "pass", door_check.actual
    assert "SEQUENCED" in (door_check.actual or "")
    assert "latch" in (door_check.actual or "")


@pytest.mark.slow
def test_unremovable_blocker_stays_an_honest_fail() -> None:
    """A blocking bar that belongs to NO other moving joint (welded to the base) must
    keep the plain FAIL -- the sequenced retry only fires when a sequence exists."""
    model = _seq_model()
    # weld the latch: replace its revolute with FIXED, so no candidate K covers the blocker
    model.joints[1] = Joint(type=JointType.FIXED, a="base.latch_pivot", b="latch.hinge")
    from forge_mind.validator.assembly_checks import validate_assembly

    report = validate_assembly(model)
    door_check = report.check("assembly.sweep.base__door")
    assert door_check is not None
    assert door_check.status.value == "fail"
    assert "SEQUENCED" not in (door_check.actual or "")
