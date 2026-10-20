"""Cylinder primitive -> build123d Part."""

from __future__ import annotations

from typing import Any

from build123d import Cylinder as _B123dCylinder
from build123d import Pos, Rot

from forge_mind.dsl.primitives import Cylinder


def build_cylinder(op: Cylinder) -> Any:
    """Return a build123d ``Part`` for the given Cylinder op, in world coordinates."""
    body = _B123dCylinder(radius=op.radius, height=op.height)
    if any((op.rotation.x, op.rotation.y, op.rotation.z)):
        body = Rot(X=op.rotation.x, Y=op.rotation.y, Z=op.rotation.z) * body
    if any((op.position.x, op.position.y, op.position.z)):
        body = Pos(op.position.x, op.position.y, op.position.z) * body
    return body
