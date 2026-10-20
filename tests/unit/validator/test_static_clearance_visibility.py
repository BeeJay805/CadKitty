"""A check that VANISHES is indistinguishable from a check that passed.

Found 2026-07-27 (research integration): `assembly.static_clearance` rides the same
ChainSweeper as the motion sweep, so any condition that refuses the sweeper (a hardware node
without an envelope mesh -- the legacy ADR-0012 shape -- or an un-sampleable solid) dropped the
clearance row out of the report entirely. The motion claim itself was always honest about it
(`assembly.range_of_motion` WARNs and names the cause), but a consumer looking for clearance
found nothing at all. It now WARNs with the reason.

Refusing to CLAIM clearance in that state stays correct and is pinned here too: an un-meshed
envelope is a body we cannot measure against, so "clear" would over-claim exactly what we
cannot see (no-false-PASS)."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.assembly.joints import Joint, JointType, Port
from forge_mind.compiler.compile_assembly import AssemblyModel, PlacedBody, PlacedHardware
from forge_mind.types import Vec3
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import CheckStatus


def _port(name: str, x: float, y: float, z: float) -> Port:
    return Port(
        name=name,
        origin=Vec3(x=x, y=y, z=z),
        axis=Vec3(x=0.0, y=0.0, z=1.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
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


def _model(*, servo_mesh: trimesh.Trimesh | None) -> AssemblyModel:
    """A frame + an arm driven by one servo. `servo_mesh=None` is the legacy frames-only
    hardware shape that used to make the clearance row disappear."""
    frame = trimesh.creation.box(extents=(40.0, 40.0, 10.0))
    arm = trimesh.creation.box(extents=(30.0, 8.0, 8.0))
    arm.apply_translation((0.0, 0.0, 30.0))
    return AssemblyModel(
        bodies=[
            _body("frame", frame, [_port("servo_seat", 0.0, 0.0, 5.0)]),
            _body("arm", arm, [_port("horn_seat", 0.0, 0.0, 30.0)]),
        ],
        joints=[
            Joint(
                type=JointType.FIXED,
                a="frame.servo_seat",
                b="servo_1.mount",
            ),
            Joint(
                type=JointType.REVOLUTE,
                a="servo_1.horn",
                b="arm.horn_seat",
                range_deg=(-45.0, 45.0),
            ),
        ],
        hardware=[
            PlacedHardware(
                id="servo_1",
                transform=np.eye(4),
                ports=[_port("mount", 0.0, 0.0, 5.0), _port("horn", 0.0, 0.0, 30.0)],
                mesh=servo_mesh,
                kind="mg996r_servo",
            )
        ],
    )


def _check(report: object, check_id: str) -> object | None:
    return next((c for c in report.checks if c.id == check_id), None)  # type: ignore[attr-defined]


def test_meshless_hardware_still_reports_a_clearance_row() -> None:
    """The regression: with a frames-only servo the row used to be ABSENT."""
    report = validate_assembly(_model(servo_mesh=None))
    row = _check(report, "assembly.static_clearance")
    assert row is not None, "assembly.static_clearance must never vanish from the report"
    assert row.status is CheckStatus.WARN  # type: ignore[attr-defined]
    assert "NOT measured" in row.message  # type: ignore[attr-defined]
    assert "servo_1" in row.message  # names the cause  # type: ignore[attr-defined]


def test_meshless_hardware_never_claims_clearance() -> None:
    """No-false-PASS: an unmeasurable envelope must not yield a PASS on either row."""
    report = validate_assembly(_model(servo_mesh=None))
    for cid in ("assembly.static_clearance", "assembly.range_of_motion"):
        row = _check(report, cid)
        assert row is not None and row.status is not CheckStatus.PASS  # type: ignore[attr-defined]


def test_meshed_hardware_measures_clearance_normally() -> None:
    """The lever changes nothing when the envelope exists: the row is measured, not warned."""
    servo = trimesh.creation.box(extents=(20.0, 20.0, 20.0))
    servo.apply_translation((0.0, 0.0, 17.0))
    report = validate_assembly(_model(servo_mesh=servo))
    row = _check(report, "assembly.static_clearance")
    assert row is not None
    assert row.status in (CheckStatus.PASS, CheckStatus.FAIL)  # type: ignore[attr-defined]
    assert "NOT measured" not in row.message  # type: ignore[attr-defined]
