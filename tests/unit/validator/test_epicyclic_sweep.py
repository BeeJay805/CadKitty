"""The epicyclic coupled driver (north-star planetary): gear meshes riding a MOVING
carrier were the v1 'coupled pair within a moving chain' withhold -- now driven
closed-form from the Willis equation. Two-sided: a correct train EARNS its sweep; a
mis-sized member shows real depth; a mis-specified train stays honestly not-drivable.
Gears are modeled as pitch cylinders with backlash -- the sweep certifies SOLIDS along
the trajectory (tooth meshing belongs to validate_gear_mesh_kinematic)."""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import trimesh

from forge_mind.agentic.assembly_adapter import AuthoredPart, build_assembly_model
from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.types import Vec3
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.motion_sweep import (
    ChainSweeper,
    epicyclic_ratios,
    epicyclic_systems,
)

_STATIONS = [
    (0.0, 30.0),
    (30.0 * math.cos(math.radians(-30)), -15.0),
    (-30.0 * math.cos(math.radians(-30)), -15.0),
]


def _port(name: str, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Port:
    return Port(
        name=name,
        origin=Vec3(x=x, y=y, z=z),
        axis=Vec3(x=0.0, y=0.0, z=1.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )


def _cyl(tmp: Path, name: str, r: float, h: float, x: float = 0.0, y: float = 0.0, z: float = 0.0):
    stl = tmp / f"{name}.stl"
    m = trimesh.creation.cylinder(radius=r, height=h)
    m.apply_translation((x, y, z))
    m.export(str(stl))
    return stl


def _train(tmp: Path, planet_r: float = 17.8) -> object:
    """Frame with an integral ring (annulus), sun, three planets, carrier disc below."""
    parts: list[AuthoredPart] = []
    ring_stl = tmp / "frame.stl"
    trimesh.creation.annulus(r_min=49.0, r_max=56.0, height=12.0).export(str(ring_stl))
    parts.append(
        AuthoredPart(
            id="frame",
            stl_path=ring_stl,
            step_path=None,
            ports=[_port("sun_axis"), _port("car_axis"), _port("ring_axis")],
        )
    )
    parts.append(
        AuthoredPart(
            id="sun", stl_path=_cyl(tmp, "sun", 11.8, 10.0), step_path=None, ports=[_port("axis")]
        )
    )
    carrier_ports = [_port("axis", z=-10.0)]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        carrier_ports.append(_port(f"pin{i}", x=sx, y=sy))
    parts.append(
        AuthoredPart(
            id="carrier",
            stl_path=_cyl(tmp, "carrier", 36.0, 4.0, z=-10.0),
            step_path=None,
            ports=carrier_ports,
        )
    )
    joints: list[Joint] = [
        Joint(type=JointType.REVOLUTE, a="frame.sun_axis", b="sun.axis", range_deg=(0, 360)),
        Joint(type=JointType.REVOLUTE, a="frame.car_axis", b="carrier.axis", range_deg=(0, 360)),
    ]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        r = planet_r if i == 1 else 17.8
        parts.append(
            AuthoredPart(
                id=f"planet{i}",
                stl_path=_cyl(tmp, f"planet{i}", r, 10.0, x=sx, y=sy),
                step_path=None,
                ports=[_port("axis", x=sx, y=sy)],
            )
        )
        joints.append(
            Joint(
                type=JointType.REVOLUTE,
                a=f"carrier.pin{i}",
                b=f"planet{i}.axis",
                range_deg=(0, 360),
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="sun.axis",
                b=f"planet{i}.axis",
                module=2.0,
                teeth_a=12,
                teeth_b=18,
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="frame.ring_axis",
                b=f"planet{i}.axis",
                module=2.0,
                teeth_a=48,
                teeth_b=18,
                internal=True,
            )
        )
    return build_assembly_model(parts, joints)


def test_ratios_are_the_willis_exact_values() -> None:
    k_c, k_p = epicyclic_ratios(12, 18, 48)
    assert abs(k_c - float(Fraction(1, 5))) < 1e-12  # the 5:1 carrier reduction
    assert abs(k_p - float(Fraction(-8, 15))) < 1e-12  # consistent via sun AND ring routes


def test_detection_finds_the_train_and_its_members(tmp_path: Path) -> None:
    model = _train(tmp_path)
    systems = epicyclic_systems(model.joints)
    assert len(systems) == 1
    _sun_joint, frame_id, sun_id, sets = systems[0]
    assert (frame_id, sun_id) == ("frame", "sun")
    st = sets[0]
    assert (st.z_sun, st.z_planet, st.z_ring) == (12, 18, 48)
    assert len(st.planet_joints) == 3 and len(st.ring_meshes) == 3
    assert all(j.b_part().startswith("planet") for j in st.planet_joints)


def test_correct_train_earns_a_full_carrier_revolution(tmp_path: Path) -> None:
    model = _train(tmp_path)
    sweeper = ChainSweeper(model)
    sun_joint, _frame, sun_id, sets = epicyclic_systems(model.joints)[0]
    ok, detail = sweeper.sweep_epicyclic(sun_joint, sun_id, sets, 0.0, 360.0)
    assert ok is True, detail
    assert "epicyclic" in detail and "0.200x" in detail
    # the claim is the CARRIER span (the adversarial review's false-PASS class: the
    # sun's 360 deg is only k_c x of the orbit -- the sweep must drive the full turn)
    assert "carrier 0..360 deg" in detail and "sun driven 0..1800 deg" in detail


def test_frame_feature_beyond_the_sun_range_arc_is_caught(tmp_path: Path) -> None:
    """The adversarial review's demonstrated false PASS: with thin planets and a frame
    boss at an orbital angle the old sun-range sweep never reached (carrier only turned
    k_c*360 = 72 deg), the train certified 'clear' yet jams on the next crank turn. The
    full-carrier-revolution sweep must FAIL it."""
    parts: list[AuthoredPart] = []
    frame_stl = tmp_path / "frame.stl"
    annulus = trimesh.creation.annulus(r_min=45.0, r_max=52.0, height=12.0)
    boss = trimesh.creation.box(extents=(4.0, 4.0, 8.0))
    boss.apply_translation(
        (30.0 * math.cos(math.radians(190)), 30.0 * math.sin(math.radians(190)), 0.0)
    )
    trimesh.util.concatenate([annulus, boss]).export(str(frame_stl))
    parts.append(
        AuthoredPart(
            id="frame",
            stl_path=frame_stl,
            step_path=None,
            ports=[_port("sun_axis"), _port("car_axis"), _port("ring_axis")],
        )
    )
    parts.append(
        AuthoredPart(
            id="sun",
            stl_path=_cyl(tmp_path, "sun", 5.0, 10.0),
            step_path=None,
            ports=[_port("axis")],
        )
    )
    carrier_ports = [_port("axis", z=-10.0)]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        carrier_ports.append(_port(f"pin{i}", x=sx, y=sy))
    parts.append(
        AuthoredPart(
            id="carrier",
            stl_path=_cyl(tmp_path, "carrier", 36.0, 4.0, z=-10.0),
            step_path=None,
            ports=carrier_ports,
        )
    )
    joints: list[Joint] = [
        Joint(type=JointType.REVOLUTE, a="frame.sun_axis", b="sun.axis", range_deg=(0, 360)),
        Joint(type=JointType.REVOLUTE, a="frame.car_axis", b="carrier.axis", range_deg=(0, 360)),
    ]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        parts.append(
            AuthoredPart(
                id=f"planet{i}",
                stl_path=_cyl(tmp_path, f"planet{i}", 5.0, 10.0, x=sx, y=sy),
                step_path=None,
                ports=[_port("axis", x=sx, y=sy)],
            )
        )
        joints.append(
            Joint(
                type=JointType.REVOLUTE,
                a=f"carrier.pin{i}",
                b=f"planet{i}.axis",
                range_deg=(0, 360),
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="sun.axis",
                b=f"planet{i}.axis",
                module=2.0,
                teeth_a=12,
                teeth_b=18,
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="frame.ring_axis",
                b=f"planet{i}.axis",
                module=2.0,
                teeth_a=48,
                teeth_b=18,
                internal=True,
            )
        )
    model = build_assembly_model(parts, joints)
    sweeper = ChainSweeper(model)
    sun_joint, _frame, sun_id, sets = epicyclic_systems(model.joints)[0]
    assert sweeper.static_overlaps() == []  # assembles cleanly at home
    ok, detail = sweeper.sweep_epicyclic(sun_joint, sun_id, sets, 0.0, 360.0)
    assert ok is False and "planet" in detail  # struck within the full carrier turn


def test_oversized_planet_fails_with_measured_depth(tmp_path: Path) -> None:
    model = _train(tmp_path, planet_r=21.0)  # eats the 0.4mm sun gap: interpenetrates
    sweeper = ChainSweeper(model)
    sun_joint, _frame, sun_id, sets = epicyclic_systems(model.joints)[0]
    ok, detail = sweeper.sweep_epicyclic(sun_joint, sun_id, sets, 0.0, 360.0)
    assert ok is False
    assert "epicyclic" in detail


def test_misspecified_tooth_counts_stay_withheld(tmp_path: Path) -> None:
    model = _train(tmp_path)
    sweeper = ChainSweeper(model)
    sun_joint, _frame, sun_id, sets = epicyclic_systems(model.joints)[0]
    bad = [
        sets[0].__class__(
            carrier_joint=sets[0].carrier_joint,
            planet_joints=sets[0].planet_joints,
            owned_joints=sets[0].owned_joints,
            emit_joints=sets[0].emit_joints,
            sun_meshes=sets[0].sun_meshes,
            ring_meshes=sets[0].ring_meshes,
            z_sun=12,
            z_planet=18,
            z_ring=50,  # ring != sun + 2*planet: the train cannot close
        )
    ]
    ok, detail = sweeper.sweep_epicyclic(sun_joint, sun_id, bad, 0.0, 360.0)
    assert ok is None and "tooth counts do not close" in detail


def test_validate_assembly_drives_the_train_end_to_end(tmp_path: Path) -> None:
    """The wiring proof: the ownership block detects the train, the driver runs, and
    range_of_motion is an EARNED PASS -- not the old 'within a moving chain' withhold."""
    report = validate_assembly(_train(tmp_path))
    rom = report.check("assembly.range_of_motion")
    assert rom is not None and rom.status.value == "pass", rom and rom.message
    sweep_ids = [c.id for c in report.checks if c.id.startswith("assembly.sweep.")]
    assert any("carrier" in cid for cid in sweep_ids)
    assert sum("planet" in cid for cid in sweep_ids) >= 3


def test_carrier_journaled_on_the_sun_with_pin_parts_is_driven(tmp_path: Path) -> None:
    """Iteration 6's legitimate topology, previously MISSED by detection: the carrier
    rides the SUN's front journal (no frame<->carrier revolute exists) and the planets
    spin on separate press-fit PIN parts (the pin-is-a-part rule). Detection must accept
    the sun as the carrier's host + either endpoint orientation, the pin collapse must
    feed it, and validate_assembly must EARN the pass end-to-end."""
    parts: list[AuthoredPart] = []
    ring_stl = tmp_path / "housing.stl"
    trimesh.creation.annulus(r_min=49.0, r_max=56.0, height=12.0).export(str(ring_stl))
    parts.append(
        AuthoredPart(
            id="housing",
            stl_path=ring_stl,
            step_path=None,
            ports=[_port("sun_bore"), _port("ring")],
        )
    )
    parts.append(
        AuthoredPart(
            id="sun",
            stl_path=_cyl(tmp_path, "sun", 11.8, 10.0),
            step_path=None,
            ports=[_port("rear_journal"), _port("front_journal", z=6.0)],
        )
    )
    carrier_ports = [_port("hub_bore", z=-10.0)]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        carrier_ports.append(_port(f"pin_hole_{i}", x=sx, y=sy, z=-8.0))
    parts.append(
        AuthoredPart(
            id="carrier",
            stl_path=_cyl(tmp_path, "carrier", 36.0, 4.0, z=-10.0),
            step_path=None,
            ports=carrier_ports,
        )
    )
    joints: list[Joint] = [
        # NOTE the orientations: housing hosts the sun (a=housing), the SUN hosts the
        # carrier (a=sun) -- no housing<->carrier revolute anywhere.
        Joint(
            type=JointType.REVOLUTE, a="housing.sun_bore", b="sun.rear_journal", range_deg=(0, 360)
        ),
        Joint(
            type=JointType.REVOLUTE, a="sun.front_journal", b="carrier.hub_bore", range_deg=(0, 360)
        ),
    ]
    for i, (sx, sy) in enumerate(_STATIONS, start=1):
        parts.append(
            AuthoredPart(
                id=f"planet_{i}",
                stl_path=_cyl(tmp_path, f"planet_{i}", 17.8, 10.0, x=sx, y=sy),
                step_path=None,
                ports=[_port("gear", x=sx, y=sy), _port("bore", x=sx, y=sy)],
            )
        )
        parts.append(
            AuthoredPart(
                id=f"pin_{i}",
                # sits in the carrier-to-planet axial gap: joined by declaration
                stl_path=_cyl(tmp_path, f"pin_{i}", 1.5, 2.8, x=sx, y=sy, z=-6.5),
                step_path=None,
                ports=[_port("mount", x=sx, y=sy, z=-8.0), _port("journal", x=sx, y=sy)],
            )
        )
        joints.append(
            Joint(type=JointType.COAXIAL_INSERT, a=f"carrier.pin_hole_{i}", b=f"pin_{i}.mount")
        )
        joints.append(
            Joint(
                type=JointType.REVOLUTE,
                a=f"pin_{i}.journal",
                b=f"planet_{i}.bore",
                range_deg=(0, 360),
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="sun.gear" if False else f"planet_{i}.gear",
                b="sun.gear",
                module=2.0,
                teeth_a=18,
                teeth_b=12,
            )
        )
        joints.append(
            Joint(
                type=JointType.GEAR_MESH,
                a="housing.ring",
                b=f"planet_{i}.gear",
                module=2.0,
                teeth_a=48,
                teeth_b=18,
                internal=True,
            )
        )
    # the sun needs a 'gear' port for the flipped external-mesh endpoints above
    parts[1] = AuthoredPart(
        id="sun",
        stl_path=parts[1].stl_path,
        step_path=None,
        ports=[*parts[1].ports, _port("gear")],
    )
    report = validate_assembly(build_assembly_model(parts, joints))
    rom = report.check("assembly.range_of_motion")
    assert rom is not None and rom.status.value == "pass", rom and rom.message
    assert any(
        "epicyclic" in (c.actual or "") for c in report.checks if c.id.startswith("assembly.sweep.")
    )
