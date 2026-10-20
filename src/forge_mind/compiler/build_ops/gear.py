"""GearSolid op -> build123d Part: the catalog's involute generator as a compiler op.

The tooth geometry is OWNED by ``agentic.gears.spur_gear`` (the same generator the
132-gear catalog and the kinematic mesh gate build with), so a printed gear and the
gear the mesh check certifies can never disagree. This builder only positions the
disc like every other op (rotation then translation).
"""

from __future__ import annotations

from typing import Any

from build123d import Pos, Rot

from forge_mind.agentic.gears import spur_gear
from forge_mind.dsl.primitives import GearSolid


def build_gear_solid(op: GearSolid) -> Any:
    """The gear disc with z in [0, thickness], axis +Z at the local origin (this op's
    documented contract), then posed by the op's rotation + position. ``spur_gear``
    itself is Z-CENTERED (the author-facing frame); the lift below restores [0, t]
    so every DSL program keeps its placement math."""
    body = Pos(0.0, 0.0, op.thickness / 2.0) * spur_gear(
        op.module,
        op.teeth,
        op.thickness,
        pressure_angle=op.pressure_angle,
        bore=op.bore_d,
        clock_deg=op.clock_deg,
    )
    if any((op.rotation.x, op.rotation.y, op.rotation.z)):
        body = Rot(X=op.rotation.x, Y=op.rotation.y, Z=op.rotation.z) * body
    if any((op.position.x, op.position.y, op.position.z)):
        body = Pos(op.position.x, op.position.y, op.position.z) * body
    return body


__all__ = ["build_gear_solid"]
