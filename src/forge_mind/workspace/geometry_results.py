"""Normalize CAD Boolean results without discarding disconnected witnesses."""

from __future__ import annotations

import math
from typing import Any


def boolean_volume(shape: Any) -> float:
    """Sum a shape or ShapeList; invalid kernel results must never imply clearance."""
    if shape is None:
        return 0.0
    if isinstance(shape, (list, tuple)):
        volume = math.fsum(boolean_volume(item) for item in shape)
    else:
        volume = float(shape.volume)
    if not math.isfinite(volume) or volume < 0:
        raise ValueError("Invalid CAD Boolean volume")
    return volume


def boolean_overlap_components(shape: Any) -> list[dict[str, Any]]:
    """Return every positive solid component in the current CAD-world frame.

    None and nonvolumetric results have no positive components. Invalid positive
    results, unsupported decomposition, or nonfinite readback must raise.
    """
    if shape is None:
        return []
    if isinstance(shape, (list, tuple)):
        return [component for item in shape for component in boolean_overlap_components(item)]
    total = boolean_volume(shape)
    if total == 0:
        return []
    if not shape.is_valid:
        raise ValueError("Invalid positive CAD Boolean shape")
    solids = list(shape.solids())
    components: list[dict[str, Any]] = []
    for solid in solids:
        volume = boolean_volume(solid)
        if volume == 0:
            continue
        if not solid.is_valid:
            raise ValueError("Invalid positive CAD overlap component")
        box = solid.bounding_box()
        lower, upper = [float(v) for v in box.min], [float(v) for v in box.max]
        if (
            len(lower) != 3
            or len(upper) != 3
            or not all(math.isfinite(v) for v in [*lower, *upper])
            or any(a > b for a, b in zip(lower, upper, strict=True))
        ):
            raise ValueError("Invalid CAD overlap component bounds")
        components.append(
            {
                "volume_mm3": volume,
                "bbox_min_mm": lower,
                "bbox_max_mm": upper,
                "frame": "cad_z_up_mm",
            }
        )
    measured = math.fsum(component["volume_mm3"] for component in components)
    if not components or not math.isclose(total, measured, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError("CAD overlap component volumes do not match Boolean volume")
    return components
