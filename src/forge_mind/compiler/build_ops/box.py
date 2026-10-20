"""Box primitive -> build123d Part."""

from __future__ import annotations

from typing import Any

from build123d import Box as _B123dBox
from build123d import Pos, Rot

from forge_mind.dsl.primitives import Box


def build_box(op: Box) -> Any:
    """Return a build123d ``Part`` for the given Box op, in world coordinates.

    Construction order: build at origin, rotate by ``op.rotation``, translate to
    ``op.position``. Matches the order documented on ``Operation.position``.
    """
    body = _B123dBox(op.width, op.depth, op.height)
    if any((op.rotation.x, op.rotation.y, op.rotation.z)):
        body = Rot(X=op.rotation.x, Y=op.rotation.y, Z=op.rotation.z) * body
    if any((op.position.x, op.position.y, op.position.z)):
        body = Pos(op.position.x, op.position.y, op.position.z) * body
    return body
