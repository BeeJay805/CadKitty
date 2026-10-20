"""Hardware kinematic attachment: PROOF that a hardware envelope fixed-joined to a
moving link is carried by the sweep (ADR-0124 wave 1; the L82-filed '2+-servo blocker').

The sweep's mover set comes from the undirected joint-graph cut over ``_parts`` -- and
``_parts`` includes hardware meshes, while joints legally reference hardware ports. So a
servo fixed to a moving link should ALREADY ride it. These tests pin that contract with
an obstacle placed where ONLY the riding hardware reaches (beyond the link's own arc):

  * with the fixed joint, the sweep FAILS naming the hardware in the colliding pair;
  * without the joint edge (hardware static at home), the same sweep is CLEAR --
    proving the carry comes from the joint graph, not coincidence.
"""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody, PlacedHardware
from forge_mind.types import Vec3
from forge_mind.validator.motion_sweep import ChainSweeper


def _v(x: float, y: float, z: float) -> Vec3:
    return Vec3(x=x, y=y, z=z)


def _body(pid: str, mesh: trimesh.Trimesh, ports: list[Port]) -> PlacedBody:
    return PlacedBody(
        id=pid,
        program=None,  # type: ignore[arg-type]
        cad=None,  # type: ignore[arg-type]
        mesh=mesh,
        transform=np.eye(4),
        ports=ports,
    )


def _box(
    extents: tuple[float, float, float], center: tuple[float, float, float]
) -> trimesh.Trimesh:
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(list(center))
    return b


def _model(with_attachment: bool) -> tuple[AssemblyModel, Joint]:
    # base: a pedestal below the pivot (static side of the revolute)
    base = _body(
        "base",
        _box((20, 20, 6), (0, 0, -6)),
        [Port(name="pivot", origin=_v(0, 0, 0), axis=_v(0, 0, 1), normal=_v(1, 0, 0))],
    )
    # link: a slim bar from the pivot out to x=40 (never reaches past y<=40 when rotated)
    link = _body(
        "link",
        _box((40, 8, 4), (20, 0, 2)),
        [
            Port(name="hub", origin=_v(0, 0, 0), axis=_v(0, 0, 1), normal=_v(1, 0, 0)),
            Port(name="tip", origin=_v(40, 0, 2), axis=_v(0, 0, 1), normal=_v(1, 0, 0)),
        ],
    )
    # obstacle: at y=+44 AND lifted to z 6..16 -- outside the link's arc in BOTH radius
    # and height; only the riding hardware (z 6..16, radii 35..45) can reach it
    obstacle = _body(
        "obstacle",
        _box((8, 8, 10), (0, 44, 11)),
        [Port(name="anchor", origin=_v(0, 44, 0), axis=_v(0, 0, 1), normal=_v(1, 0, 0))],
    )
    # the riding hardware: a 10 mm cube envelope centered on the link tip (x=40)
    hw_mesh = _box((10, 10, 10), (0, 0, 0))
    hw_tf = np.eye(4)
    hw_tf[0, 3] = 40.0
    hw_tf[2, 3] = 11.0
    servo = PlacedHardware(
        id="ride_servo",
        transform=hw_tf,
        ports=[Port(name="mount", origin=_v(40, 0, 11), axis=_v(0, 0, 1), normal=_v(1, 0, 0))],
        mesh=hw_mesh,
        kind="mg996r_servo",
    )
    pivot = Joint(type=JointType.REVOLUTE, a="base.pivot", b="link.hub")
    joints = [pivot]
    if with_attachment:
        joints.append(Joint(type=JointType.FIXED, a="link.tip", b="ride_servo.mount"))
    model = AssemblyModel(bodies=[base, link, obstacle], joints=joints, hardware=[servo])
    return model, pivot


def test_attached_hardware_rides_and_collides() -> None:
    model, pivot = _model(with_attachment=True)
    ok, detail = ChainSweeper(model).sweep(pivot, 0.0, 90.0)
    assert ok is False
    assert "ride_servo" in detail and "obstacle" in detail


def test_without_the_joint_edge_the_sweep_is_clear() -> None:
    model, pivot = _model(with_attachment=False)
    ok, detail = ChainSweeper(model).sweep(pivot, 0.0, 90.0)
    assert ok is True, detail
