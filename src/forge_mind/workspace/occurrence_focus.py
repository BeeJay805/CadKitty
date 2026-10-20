"""Pure selection and camera contracts for saved occurrence focus images."""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FOCUS_SCOPE = (
    "Selected isolated occurrence only; other assembly bodies are omitted. "
    "The marker shows projected saved entity bounds, not an exact entity outline. "
    "All selected-body triangles remain opaque and depth-tested, with viewport clipping only. "
    "Saved display tessellation is used; no exact face highlight, visibility, "
    "measurement or physical-fit claim."
)
_REGION_MARKER = "projected saved entity bounds; not exact entity outline"
_VIEW_ANGLES = {
    "isometric": (26.0, -58.0),
    "front": (2.0, -90.0),
    "right": (2.0, 0.0),
    "rear": (2.0, 90.0),
    "top": (89.0, -90.0),
    "bottom": (-89.0, 90.0),
}
_FRAME_KEYS = frozenset(
    {
        "request",
        "body_index",
        "body_signature",
        "entity_id",
        "entity_bounds",
        "body_bounds",
        "center_mm",
        "radius_mm",
        "scope",
    }
)


class OccurrenceFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["face", "edge"]
    signature: Annotated[
        str, Field(strict=True, min_length=64, max_length=64, pattern=r"^[a-f0-9]{64}$")
    ]
    view: Literal["isometric", "front", "right", "rear", "top", "bottom"] = "isometric"
    padding: Annotated[float, Field(strict=True, allow_inf_nan=False, ge=1.1, le=8.0)] = 2.0


def _digest(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Focus requires a valid saved signature")
    return value


def _number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("Focus requires finite numeric coordinates")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Focus requires finite numeric coordinates") from exc
    if not math.isfinite(number):
        raise ValueError("Focus requires finite numeric coordinates")
    return number


def _vector(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("Focus requires three coordinates")
    return [_number(item) for item in value]


def _bounds(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ValueError("Focus requires exact min/max bounds")
    lower, upper = _vector(value["min"]), _vector(value["max"])
    if any(a > b for a, b in zip(lower, upper, strict=True)):
        raise ValueError("Focus bounds must be ordered")
    return {"min": lower, "max": upper}


def _row_bounds(row: dict[str, Any]) -> dict[str, list[float]]:
    return _bounds({"min": row.get("min"), "max": row.get("max")})


def _center(bounds: dict[str, list[float]]) -> list[float]:
    return [_number(a / 2 + b / 2) for a, b in zip(bounds["min"], bounds["max"], strict=True)]


def _diagonal(bounds: dict[str, list[float]]) -> float:
    diagonal = _number(math.dist(bounds["min"], bounds["max"]))
    if diagonal <= 0:
        raise ValueError("Focus requires a positive bounds diagonal")
    return diagonal


def _identity(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("Focus requires a valid saved entity ID")
    return value


def validate_focus_frame(value: Any) -> dict[str, Any]:
    """Normalize JSON vectors and reject changed or inconsistent derived fields."""
    if not isinstance(value, dict) or set(value) != _FRAME_KEYS:
        raise ValueError("Focus frame has missing or unexpected fields")
    request = value["request"]
    if not isinstance(request, dict) or set(request) != {"kind", "signature", "view", "padding"}:
        raise ValueError("Focus frame requires a complete request")
    focus = OccurrenceFocus.model_validate(request)
    _digest(focus.signature)
    index = value["body_index"]
    if type(index) is not int or index < 0:
        raise ValueError("Focus requires a nonnegative saved body index")
    body_signature = _digest(value["body_signature"])
    entity_id = _identity(value["entity_id"])
    entity = _bounds(value["entity_bounds"])
    body = _bounds(value["body_bounds"])
    diagonal = _diagonal(entity)
    tolerance = max(_diagonal(body) * 1e-6, 1e-6)
    if any(a < b - tolerance for a, b in zip(entity["min"], body["min"], strict=True)) or any(
        a > b + tolerance for a, b in zip(entity["max"], body["max"], strict=True)
    ):
        raise ValueError("Focus entity bounds lie outside the selected body")
    center = _center(entity)
    radius = _number(max(diagonal / 2 * focus.padding, 1e-6))
    if _vector(value["center_mm"]) != center or _number(value["radius_mm"]) != radius:
        raise ValueError("Focus frame center or radius does not match saved bounds")
    if value["scope"] != FOCUS_SCOPE:
        raise ValueError("Focus frame has a different display scope")
    return {
        "request": focus.model_dump(mode="json"),
        "body_index": index,
        "body_signature": body_signature,
        "entity_id": entity_id,
        "entity_bounds": entity,
        "body_bounds": body,
        "center_mm": center,
        "radius_mm": radius,
        "scope": FOCUS_SCOPE,
    }


def resolve_focus(
    topology: dict[str, Any], body_index: int, focus: OccurrenceFocus
) -> dict[str, Any]:
    """Resolve inventory identity; mesh picking scope does not determine fact coverage."""
    if type(body_index) is not int or body_index < 0 or not isinstance(focus, OccurrenceFocus):
        raise ValueError("Focus requires a typed selector and nonnegative body index")
    if not isinstance(topology, dict) or topology.get("frame") != "cad_z_up_mm":
        raise ValueError("Focus requires saved CAD Z-up topology")
    rows = topology.get("entities")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Focus requires a valid saved entity inventory")
    # Equality can equate bool and int. Check the actual saved type before accepting identity.
    bodies = [
        row for row in rows if row.get("kind") == "body" and row.get("body_index") == body_index
    ]
    if len(bodies) != 1 or type(bodies[0].get("body_index")) is not int:
        raise ValueError("Focus body is missing or ambiguous")
    body = bodies[0]
    signature = _digest(body.get("signature"))
    if body.get("body_signature") != signature:
        raise ValueError("Focus body signatures disagree")
    body_id = _identity(body.get("id"))
    if sum(row.get("id") == body_id for row in rows) != 1:
        raise ValueError("Focus body ID is ambiguous")
    selected = [
        row
        for row in rows
        if row.get("kind") == focus.kind
        and row.get("body_index") == body_index
        and row.get("signature") == focus.signature
    ]
    if len(selected) != 1 or type(selected[0].get("body_index")) is not int:
        raise ValueError("Focus entity is missing or ambiguous in the selected body")
    entity = selected[0]
    if entity.get("body_signature") != signature:
        raise ValueError("Focus entity belongs to another saved body signature")
    identity = _identity(entity.get("id"))
    if sum(row.get("id") == identity for row in rows) != 1:
        raise ValueError("Focus entity ID is ambiguous")
    bounds = _row_bounds(entity)
    frame = {
        "request": focus.model_dump(mode="json"),
        "body_index": body_index,
        "body_signature": signature,
        "entity_id": identity,
        "entity_bounds": bounds,
        "body_bounds": _row_bounds(body),
        "center_mm": _center(bounds),
        "radius_mm": _number(max(_diagonal(bounds) / 2 * focus.padding, 1e-6)),
        "scope": FOCUS_SCOPE,
    }
    return validate_focus_frame(frame)


def focus_view_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    """Return camera basis columns: screen right, screen up, toward the viewer."""
    frame = validate_focus_frame(frame)
    view = frame["request"]["view"]
    elevation, azimuth = (math.radians(angle) for angle in _VIEW_ANGLES[view])
    se, ce, sa, ca = math.sin(elevation), math.cos(elevation), math.sin(azimuth), math.cos(azimuth)
    return {
        "view": view,
        "basis": [[-sa, -se * ca, ce * ca], [ca, -se * sa, ce * sa], [0.0, ce, se]],
        "context_center_mm": _center(frame["body_bounds"]),
        "context_radius_mm": _number(max(_diagonal(frame["body_bounds"]) / 2 * 1.1, 1e-6)),
        "detail_center_mm": frame["center_mm"],
        "detail_radius_mm": frame["radius_mm"],
        "all_selected_body_triangles_depth_tested": True,
        "region_marker": _REGION_MARKER,
    }
