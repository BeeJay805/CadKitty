"""Cited hardware mount patterns the assembly gate verifies.

"Every mount matches its cited part's real pattern": when a printed part bolts to a
hardware node (a servo's tab flange, a motor's face, a horn), the printed part must
carry the REGISTRY's hole pattern for that hardware -- measured on the compiled B-rep,
never assumed from the program. The pattern numbers are single-sourced from the same
modules the generators read (``assembly.servo``, ``dsl.interfaces.nema17``), so the
geometry builder and the gate can never disagree about what the cited part requires.

A pattern is keyed by (hardware kind, port name): the gate checks it only when a
printed part is actually joined at that port -- hardware with nothing bolted to a port
claims nothing there.
"""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.assembly.servo import (
    MG996R_HORN_PLANE_Z_MM,
    MG996R_HORN_SCREW_BC_D_MM,
    MG996R_HORN_SCREW_D_MM,
    MG996R_HUB_CENTER_BORE_D_MM,
    MG996R_MOUNT_HOLE_DIAMETER_MM,
    MG996R_SOURCE,
    mg996r_mount_hole_points_local,
)
from forge_mind.dsl.interfaces.nema17 import BOLT_PITCH_MM as _NEMA17_BOLT_PITCH_MM

Vec3T = tuple[float, float, float]


@dataclass(frozen=True)
class MountPattern:
    """One cited hole pattern on one hardware port. ``holes`` are (point in the
    HARDWARE's local frame, acceptable hole diameter range mm); ``axis_local`` is the
    common drilling axis in the hardware frame."""

    port: str
    axis_local: Vec3T
    holes: list[tuple[Vec3T, tuple[float, float]]]
    label: str
    source: str


def _mg996r_patterns() -> list[MountPattern]:
    tab_d = MG996R_MOUNT_HOLE_DIAMETER_MM
    tab_holes: list[tuple[Vec3T, tuple[float, float]]] = [
        (p, tab_d) for p in mg996r_mount_hole_points_local()
    ]
    horn_holes: list[tuple[Vec3T, tuple[float, float]]] = [
        (
            (0.0, 0.0, MG996R_HORN_PLANE_Z_MM),
            (MG996R_HUB_CENTER_BORE_D_MM - 0.3, MG996R_HUB_CENTER_BORE_D_MM + 0.3),
        )
    ]
    r = MG996R_HORN_SCREW_BC_D_MM / 2.0
    for dx, dy in ((r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)):
        horn_holes.append(
            (
                (dx, dy, MG996R_HORN_PLANE_Z_MM),
                (MG996R_HORN_SCREW_D_MM - 0.3, MG996R_HORN_SCREW_D_MM + 0.4),
            )
        )
    return [
        MountPattern(
            port="mount",
            axis_local=(0.0, 0.0, 1.0),
            holes=tab_holes,
            label="MG996R tab screws (4x on 49.5 x 10.0 mm)",
            source=MG996R_SOURCE,
        ),
        MountPattern(
            port="horn",
            axis_local=(0.0, 0.0, 1.0),
            holes=horn_holes,
            label="MG996R round-horn interface (center bore + 4 horn screws)",
            source=MG996R_SOURCE,
        ),
    ]


def _nema17_patterns() -> list[MountPattern]:
    half = _NEMA17_BOLT_PITCH_MM / 2.0
    holes: list[tuple[Vec3T, tuple[float, float]]] = [
        ((sx * half, sy * half, 0.0), (2.9, 3.8)) for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ]
    return [
        MountPattern(
            port="face",
            axis_local=(0.0, 0.0, 1.0),
            holes=holes,
            label="NEMA 17 face screws (4x M3 on 31.0 mm square)",
            source="NEMA ICS 16 via dsl/interfaces/nema17",
        )
    ]


_PATTERNS: dict[str, list[MountPattern]] = {
    "mg996r_servo": _mg996r_patterns(),
    "nema17_motor": _nema17_patterns(),
}


def patterns_for(kind: str | None) -> list[MountPattern]:
    """The cited patterns for a hardware kind ([] when none are registered -- an
    unknown kind claims nothing and the gate stays silent rather than guessing)."""
    if kind is None:
        return []
    return _PATTERNS.get(kind, [])


__all__ = ["MountPattern", "patterns_for"]
