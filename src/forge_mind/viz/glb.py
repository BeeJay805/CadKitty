"""glTF orientation. A kernel mesh is Z-up (the CAD / build123d convention), but glTF viewers assume
Y-up, so a part exported as-is renders lying on its back. Rotate the mesh -90 deg about X (Z-up ->
Y-up) IMMEDIATELY before a GLB export.

The STL and STEP stay Z-up (slicers and CAD expect Z-up); ONLY the viewer GLB is converted, so every
generated part matches the hand-made Y-up fixture and the frontend needs no change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# -90 deg about X maps (x, y, z) -> (x, z, -y). Exact integer entries -- no trig rounding.
_ZUP_TO_YUP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def to_gltf_yup(mesh: Any) -> Any:
    """Rotate a Z-up kernel mesh IN PLACE to glTF's Y-up convention and return it. Call right before a
    GLB export; never on the STL/STEP (those stay Z-up). Pass a ``.copy()`` if the same mesh is also
    being written as an STL."""
    mesh.apply_transform(_ZUP_TO_YUP)
    return mesh


__all__ = ["to_gltf_yup"]
