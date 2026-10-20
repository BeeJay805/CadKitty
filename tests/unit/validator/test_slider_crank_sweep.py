"""Slider-crank coupled driver (the i4-finish audit's soundness fix).

The tree sweep is structurally unsound on a closed crank-rod-piston loop: it rigidly
spins the rods with the crank, full-circles a rod about its pin, and never moves a piston
-- phantom FAILs and unearned PASSes both. These tests prove the loop-closure driver on a
real synthetic engine cylinder built from booleans (0.5 mm design margins, so a pose-math
error > ~0.65 mm reads as a collision):

  * detection finds the loop regardless of joint side order;
  * a clean cylinder sweeps 360 deg CLEAR -- including past an obstacle that the OLD
    full-circle rod sweep would have phantom-hit (the live run's failure class);
  * a real obstruction in the piston's travel FAILS with pair + angle + depth named;
  * a loop that does not close (wrist off the bore axis) earns NO claim (None, reason);
  * the full-scan diagnostics name the worst pair with measured depth.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody
from forge_mind.types import Vec3
from forge_mind.validator.motion_sweep import (
    ChainSweeper,
    slider_crank_systems,
)

# ---- geometry (mm): crank axis = X through the origin, bore axis = Z at x=23 ----------
THROW = 12.0
ROD_LEN = 42.0
WRIST_Z = THROW + ROD_LEN  # 54: wrist home = TDC


def _cyl_x(radius: float, x0: float, x1: float, y: float = 0.0, z: float = 0.0) -> trimesh.Trimesh:
    """A cylinder along X spanning [x0, x1] at (y, z)."""
    c = trimesh.creation.cylinder(radius=radius, height=x1 - x0, sections=64)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    c.apply_translation([(x0 + x1) / 2, y, z])
    return c


def _ring_x(
    r_bore: float, r_out: float, x0: float, x1: float, y: float, z: float
) -> trimesh.Trimesh:
    ring = trimesh.boolean.difference(
        [_cyl_x(r_out, x0, x1, y, z), _cyl_x(r_bore, x0 - 1, x1 + 1, y, z)]
    )
    assert isinstance(ring, trimesh.Trimesh) and ring.is_watertight
    return ring


def _crank() -> trimesh.Trimesh:
    """Journal + crankpin + two cheeks BESIDE the rod's x-band (a real crank carries the
    rod between its cheeks; a web through the rod plane collides at BDC -- the driver
    itself caught that when this fixture first ran)."""
    # journal SEGMENTS between the throws (a continuous journal through the rod's x-band
    # is crossed by the rod shank at BDC -- the second fixture bug this driver caught):
    journal_l = _cyl_x(5.0, -15.0, 18.4)  # main journal, left of cheek 1
    journal_r = _cyl_x(5.0, 27.6, 35.0)  # stub, right of cheek 2
    pin = _cyl_x(4.0, 18.0, 28.0, y=0.0, z=THROW)  # crankpin at z=+12 (home)
    cheeks = []
    for x0, x1 in ((18.0, 19.8), (26.2, 28.0)):  # rod ring spans x 20..26: 0.2 mm air
        cheek = trimesh.creation.box(extents=[x1 - x0, 12.0, THROW])
        cheek.apply_translation([(x0 + x1) / 2, 0.0, THROW / 2])
        cheeks.append(cheek)
    out = trimesh.boolean.union([journal_l, journal_r, pin, *cheeks])
    assert isinstance(out, trimesh.Trimesh) and out.is_watertight
    return out


def _rod() -> trimesh.Trimesh:
    big = _ring_x(4.4, 6.5, 20.0, 26.0, 0.0, THROW)  # around the crankpin, 0.4 mm fit
    small = _ring_x(3.4, 6.0, 20.0, 26.0, 0.0, WRIST_Z)  # around the piston pin
    shank = trimesh.creation.box(extents=[6.0, 6.0, WRIST_Z - THROW - 8.0])
    shank.apply_translation([23.0, 0.0, (THROW + WRIST_Z) / 2])
    out = trimesh.boolean.union([big, shank, small])
    assert isinstance(out, trimesh.Trimesh) and out.is_watertight
    return out


def _piston() -> trimesh.Trimesh:
    skirt = trimesh.creation.cylinder(radius=14.0, height=30.0, sections=64)
    skirt.apply_translation([23.0, 0.0, WRIST_Z + 9.0])  # z 48..78
    slot = trimesh.creation.box(extents=[8.0, 14.0, 18.0])
    slot.apply_translation([23.0, 0.0, WRIST_Z - 1.0])  # slot z 44..62 for the rod end
    body = trimesh.boolean.difference([skirt, slot])
    pin = _cyl_x(3.0, 17.0, 29.0, y=0.0, z=WRIST_Z)  # wrist pin bridges the slot
    out = trimesh.boolean.union([body, pin])
    assert isinstance(out, trimesh.Trimesh) and out.is_watertight
    return out


def _frame(extra: trimesh.Trimesh | None = None) -> trimesh.Trimesh:
    plate = trimesh.creation.box(extents=[80.0, 40.0, 6.0])
    plate.apply_translation([10.0, 0.0, -30.0])  # well below the crank orbit
    if extra is None:
        return plate
    out = trimesh.boolean.union([plate, extra])
    assert isinstance(out, trimesh.Trimesh)
    return out


def _port(name: str, origin: tuple[float, float, float], axis: tuple[float, float, float]) -> Port:
    ax = np.asarray(axis, dtype=float)
    n = (
        np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(ax, [0, 0, 1]))) < 0.9
        else np.array([0.0, 1.0, 0.0])
    )
    return Port(
        name=name,
        origin=Vec3(x=origin[0], y=origin[1], z=origin[2]),
        axis=Vec3(x=axis[0], y=axis[1], z=axis[2]),
        normal=Vec3(x=n[0], y=n[1], z=n[2]),
    )


def _body(pid: str, mesh: trimesh.Trimesh, ports: list[Port]) -> PlacedBody:
    return PlacedBody(
        id=pid,
        program=None,  # type: ignore[arg-type]
        cad=None,  # type: ignore[arg-type]
        mesh=mesh,
        transform=np.eye(4),
        ports=ports,
    )


def _joints(swap_sides: bool = False) -> list[Joint]:
    def j(t: JointType, a: str, b: str, **kw: object) -> Joint:
        if swap_sides:
            a, b = b, a
        return Joint(type=t, a=a, b=b, **kw)  # type: ignore[arg-type]

    return [
        j(JointType.REVOLUTE, "frame.main_bearing", "crank.main_journal", range_deg=(0, 360)),
        j(JointType.REVOLUTE, "crank.crankpin", "rod.big_end", range_deg=(0, 360)),
        j(JointType.REVOLUTE, "piston.wrist", "rod.small_end", range_deg=(-20, 20)),
        j(JointType.PRISMATIC, "frame.cyl_slide", "piston.slide", range_mm=(0, 24)),
    ]


def _model(
    frame_extra: trimesh.Trimesh | None = None, wrist_off_axis_mm: float = 0.0
) -> AssemblyModel:
    frame_ports = [
        _port("main_bearing", (0, 0, 0), (1, 0, 0)),
        _port("cyl_slide", (23, 0, 30), (0, 0, 1)),
    ]
    crank_ports = [
        _port("main_journal", (0, 0, 0), (1, 0, 0)),
        _port("crankpin", (23, 0, THROW), (1, 0, 0)),
    ]
    rod_ports = [
        _port("big_end", (23, 0, THROW), (1, 0, 0)),
        _port("small_end", (23, wrist_off_axis_mm, WRIST_Z), (1, 0, 0)),
    ]
    piston_ports = [
        _port("wrist", (23, wrist_off_axis_mm, WRIST_Z), (1, 0, 0)),
        _port("slide", (23, 0, 60), (0, 0, 1)),
    ]
    return AssemblyModel(
        bodies=[
            _body("frame", _frame(frame_extra), frame_ports),
            _body("crank", _crank(), crank_ports),
            _body("rod", _rod(), rod_ports),
            _body("piston", _piston(), piston_ports),
        ],
        joints=_joints(),
    )


def test_detection_finds_the_loop_regardless_of_side_order() -> None:
    for swap in (False, True):
        systems = slider_crank_systems(_joints(swap_sides=swap))
        assert len(systems) == 1, f"swap={swap}"
        _j0, frame, crank, loops = systems[0]
        assert frame == "frame" and crank == "crank"
        assert len(loops) == 1
        assert loops[0].rod == "rod" and loops[0].piston == "piston"


def test_no_loop_in_a_plain_hinge() -> None:
    hinge = [Joint(type=JointType.REVOLUTE, a="base.hub", b="arm.bore", range_deg=(0, 90))]
    assert slider_crank_systems(hinge) == []


@pytest.fixture(scope="module")
def clean_sweep() -> tuple[bool | None, str]:
    """One shared clean-fixture sweep (module-scoped: the booleans + 19 poses are the
    expensive part; every assertion about the clean run reads this single result)."""
    model = _model()
    sweeper = ChainSweeper(model)
    assert sweeper.all_sampled_ok()
    (j0, _frame, crank_id, loops) = slider_crank_systems(model.joints)[0]
    return sweeper.sweep_slider_crank(j0, crank_id, loops, 0.0, 360.0)


def test_clean_cylinder_sweeps_360_clear(clean_sweep: tuple[bool | None, str]) -> None:
    ok, detail = clean_sweep
    assert ok is True, detail
    assert "loop-closure driven" in detail


def test_no_phantom_from_full_circle_rod_poses() -> None:
    """The live run's failure class: an obstacle placed where a FULL-CIRCLE rod rotation
    about its pin would sweep (R~30 from the pin, y=30) but the REAL +-16.6 deg rod swing
    never reaches (rod material stays inside |y| <= ~19). The old tree sweep phantom-hits
    it; the loop-closure driver must sweep CLEAR."""
    obstacle = trimesh.creation.box(extents=[6.0, 6.0, 6.0])
    obstacle.apply_translation([23.0, 30.0, THROW])
    model = _model(frame_extra=obstacle)
    sweeper = ChainSweeper(model)
    (j0, _frame, crank_id, loops) = slider_crank_systems(model.joints)[0]
    ok, detail = sweeper.sweep_slider_crank(j0, crank_id, loops, 0.0, 360.0)
    assert ok is True, f"phantom collision: {detail}"


def test_real_obstruction_fails_with_pair_angle_and_depth() -> None:
    """A bar inside the piston skirt's swept annulus at z 38..42, offset to y=11 so only
    the PISTON reaches it (the rod's widest excursion at that height is |y| < 8; a first
    draft at y=0 sat in the rod's own path and the sweep dutifully reported BOTH pairs).
    Clear at home (piston z 48..78), struck mid-stroke. The detail must name the pair,
    the angle, and a measured depth."""
    bar = trimesh.creation.box(extents=[16.0, 6.0, 4.0])
    bar.apply_translation([23.0, 11.0, 40.0])
    model = _model(frame_extra=bar)
    sweeper = ChainSweeper(model)
    (j0, _frame, crank_id, loops) = slider_crank_systems(model.joints)[0]
    ok, detail = sweeper.sweep_slider_crank(j0, crank_id, loops, 0.0, 360.0)
    assert ok is False
    assert "piston hits frame at" in detail
    assert "depth" in detail and "mm" in detail


def test_unclosable_loop_earns_no_claim() -> None:
    """Wrist 5 mm off its bore axis: the loop cannot close as declared. The driver must
    return None (claim NOT earned -- WARN upstream), never a pass/fail verdict."""
    model = _model(wrist_off_axis_mm=5.0)
    sweeper = ChainSweeper(model)
    (j0, _frame, crank_id, loops) = slider_crank_systems(model.joints)[0]
    ok, detail = sweeper.sweep_slider_crank(j0, crank_id, loops, 0.0, 360.0)
    assert ok is None
    assert "off" in detail and "bore axis" in detail
