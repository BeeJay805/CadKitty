"""Bounded pure views of saved individual cylindrical surfaces and port comparisons.

The caller verifies ownership, revision and artifact provenance before using this view.
No geometry is parsed, no surfaces are aggregated, and no acceptance verdict changes.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

MAX_PAGE_COMPARISONS = 2000
MAX_COUNT_COMPARISONS = 100_000
SURFACE_SCOPE = (
    "Individual exported cylindrical faces, not merged hole features. Surface IDs are "
    "occurrence/revision-local extraction ordinals, not stable topology IDs. Axial extents "
    "are face-vertex projections onto axis_direction relative to the world origin, not "
    "distances from axis_point_mm. They do not prove rectangular coverage, full circumference, "
    "throughness, continuity, bore depth, or usable fastening."
)


def _number(value: Any, name: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three finite numbers")
    return (_number(value[0], name), _number(value[1], name), _number(value[2], name))


def _axis(value: Any, name: str) -> tuple[float, float, float]:
    axis = _vector(value, name)
    if abs(math.hypot(*axis) - 1) > 1e-6:
        raise ValueError(f"{name} must be a unit axis")
    return axis


def _cross_length(a: tuple[float, float, float], c: tuple[float, float, float]) -> float:
    return math.hypot(
        a[1] * c[2] - a[2] * c[1], a[2] * c[0] - a[0] * c[2], a[0] * c[1] - a[1] * c[0]
    )


def _surface(row: Any) -> dict[str, Any]:
    if not isinstance(row, dict) or not isinstance(row.get("surface_id"), str):
        raise ValueError("Malformed saved cylindrical surface")
    radius = _number(row.get("radius_mm"), "radius_mm")
    diameter = _number(row.get("diameter_mm"), "diameter_mm")
    if radius <= 0 or diameter != 2 * radius:
        raise ValueError("Saved cylindrical radius and diameter disagree")
    _vector(row.get("axis_point_mm"), "axis_point_mm")
    _axis(row.get("axis_direction"), "axis_direction")
    if row.get("concave") is not None and type(row["concave"]) is not bool:
        raise ValueError("Saved concavity must be boolean or null")
    if "concave" not in row or "axial_extent_mm" not in row:
        raise ValueError("Saved surface must retain unknown concavity and extent explicitly")
    extent = row["axial_extent_mm"]
    if extent is not None:
        if not isinstance(extent, (list, tuple)) or len(extent) != 2:
            raise ValueError("Saved axial extent must contain two finite numbers or null")
        if _number(extent[0], "axial_extent_mm") > _number(extent[1], "axial_extent_mm"):
            raise ValueError("Saved axial extent is reversed")
    return row


def _comparison(port: dict[str, Any], surface: dict[str, Any]) -> dict[str, Any]:
    axis = _axis(port.get("axis"), "port axis")
    origin = _vector(port.get("origin"), "port origin")
    diameter = _number(port.get("diameter_mm"), "port diameter_mm")
    ca = _axis(surface["axis_direction"], "axis_direction")
    cp = _vector(surface["axis_point_mm"], "axis_point_mm")
    delta = tuple(a - c for a, c in zip(origin, cp, strict=True))
    parallel_error = _cross_length(axis, ca)
    radial_error = _cross_length((delta[0], delta[1], delta[2]), ca)
    diameter_error = abs(2 * surface["radius_mm"] - diameter)
    projection = sum(a * c for a, c in zip(origin, ca, strict=True))
    extent = surface["axial_extent_mm"]
    outside = max(extent[0] - projection, projection - extent[1], 0) if extent is not None else None
    matches = {
        "material_side": surface["concave"] == (port["kind"] == "bore"),
        "diameter": diameter_error < 1e-5,
        "axis_parallel": parallel_error < 1e-6,
        "axis_radial": radial_error < 1e-5,
        "axial_origin": extent is not None and extent[0] - 1e-5 <= projection <= extent[1] + 1e-5,
    }
    numbers = (parallel_error, radial_error, diameter_error, projection)
    if not all(math.isfinite(value) for value in numbers) or (
        outside is not None and not math.isfinite(outside)
    ):
        raise ValueError("Port comparison exceeds finite numeric range")
    return {
        "surface_id": surface["surface_id"],
        "predicate_matches": matches,
        "all_existing_predicates_match": all(matches.values()),
        "material_side_known": surface["concave"] is not None,
        "expected_concave": port["kind"] == "bore",
        "diameter_error_mm": diameter_error,
        "radius_delta_mm": surface["radius_mm"] - diameter / 2,
        "axis_cross_norm": parallel_error,
        "axis_radial_error_mm": radial_error,
        "axial_origin_projection_mm": projection,
        "axial_extent_known": extent is not None,
        "axial_outside_distance_mm": outside,
    }


def occurrence_facts(
    measurement: dict[str, Any],
    occurrence_path: str,
    *,
    surface_offset: int = 0,
    surface_limit: int = 20,
) -> dict[str, Any]:
    """Project saved same-revision assembly facts; never replace existing port status.

    Port declarations are complete. Surface comparisons are explicitly bounded separately;
    global matching counts are unavailable if their independent computation budget is exceeded.
    """
    if not isinstance(occurrence_path, str) or not occurrence_path or len(occurrence_path) > 2048:
        raise ValueError("occurrence_path must be a nonempty bounded string")
    if type(surface_offset) is not int or surface_offset < 0:
        raise ValueError("surface_offset must be a nonnegative integer")
    if type(surface_limit) is not int or not 1 <= surface_limit <= 100:
        raise ValueError("surface_limit must be an integer from 1 to 100")
    if not isinstance(measurement, dict) or not isinstance(measurement.get("facts"), dict):
        raise ValueError("Measurement must contain saved facts")
    assembly = measurement["facts"].get("assembly")
    result: dict[str, Any] = {
        "projection": "occurrence_facts",
        "occurrence_path": occurrence_path,
        "frame": "cad_z_up_mm_world",
        "surface_scope": SURFACE_SCOPE,
        "comparison_scope": "Existing port predicates only. Multiple matching faces are not multiple physical holes. Stored port statuses remain authoritative; no closest-candidate pass is inferred.",
        "predicate_tolerances": {
            "diameter_strict_mm": 1e-5,
            "axis_cross_norm_strict": 1e-6,
            "axis_radial_strict_mm": 1e-5,
            "axial_inclusive_mm": 1e-5,
        },
        "offset": surface_offset,
        "limit": surface_limit,
    }
    if assembly is None:
        return {
            **result,
            "availability": "unavailable",
            "reason": "No saved assembly facts",
            "cylindrical_surfaces": [],
            "ports": [],
            "total": None,
            "next_offset": None,
        }
    if (
        not isinstance(assembly, dict)
        or not isinstance(assembly.get("occurrences"), list)
        or not isinstance(assembly.get("ports"), dict)
    ):
        raise ValueError("Malformed saved assembly facts")
    rows = [
        row
        for row in assembly["occurrences"]
        if isinstance(row, dict) and row.get("path") == occurrence_path
    ]
    if len(rows) != 1:
        raise ValueError("Occurrence path must identify exactly one saved occurrence")
    occurrence = rows[0]
    for key in ("definition", "name"):
        if not isinstance(occurrence.get(key), str) or not occurrence[key]:
            raise ValueError(f"Saved occurrence {key} is missing")
    if type(occurrence.get("body_index")) is not int or occurrence["body_index"] < 0:
        raise ValueError("Saved occurrence body_index must be nonnegative")
    if _number(occurrence.get("volume_mm3"), "volume_mm3") < 0:
        raise ValueError("Saved occurrence volume is negative")
    _vector(occurrence.get("center_mm"), "center_mm")
    lo, hi = (_vector(occurrence.get(key), key) for key in ("bbox_min_mm", "bbox_max_mm"))
    if any(a > c for a, c in zip(lo, hi, strict=True)):
        raise ValueError("Saved occurrence bounds are reversed")
    prefix = occurrence_path + "."
    ports: list[dict[str, Any]] = []
    for key, port in assembly["ports"].items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if not isinstance(port, dict) or port.get("kind") not in {"bore", "shaft"}:
            raise ValueError("Malformed saved port declaration")
        if port.get("status") not in {"pass", "fail", "unverified"} or not isinstance(
            port.get("method"), str
        ):
            raise ValueError("Malformed saved port result")
        _axis(port.get("axis"), "port axis")
        _vector(port.get("origin"), "port origin")
        if _number(port.get("diameter_mm"), "port diameter_mm") <= 0:
            raise ValueError("Port diameter must be positive")
        ports.append(
            {
                "port_id": key,
                "declaration": deepcopy(port),
                "surface_comparisons": [],
                "matching_surface_count": None,
                "matching_surface_count_scope": "unavailable: saved surfaces missing",
            }
        )
    metadata = {
        key: deepcopy(occurrence[key])
        for key in (
            "path",
            "definition",
            "name",
            "body_index",
            "volume_mm3",
            "center_mm",
            "bbox_min_mm",
            "bbox_max_mm",
        )
        if key in occurrence
    }
    if "cylindrical_surfaces" not in occurrence:
        return {
            **result,
            "availability": "unavailable",
            "reason": "Legacy occurrence has no saved cylindrical surface facts; no native reconstruction performed",
            "occurrence": metadata,
            "cylindrical_surfaces": [],
            "ports": ports,
            "total": None,
            "next_offset": None,
        }
    raw = occurrence["cylindrical_surfaces"]
    if not isinstance(raw, list):
        raise ValueError("Saved cylindrical surfaces must be a list")
    surfaces = [_surface(row) for row in raw]
    if len({row["surface_id"] for row in surfaces}) != len(surfaces):
        raise ValueError("Saved cylindrical surface IDs are ambiguous")
    page = surfaces[surface_offset : surface_offset + surface_limit]
    comparison_limit = MAX_PAGE_COMPARISONS // len(ports) if ports else len(page)
    compared = page[:comparison_limit]
    count_available = len(ports) * len(surfaces) <= MAX_COUNT_COMPARISONS
    for port in ports:
        declaration = port["declaration"]
        port["surface_comparisons"] = [_comparison(declaration, row) for row in compared]
        port["comparisons_complete_for_page"] = len(compared) == len(page)
        count = (
            sum(_comparison(declaration, row)["all_existing_predicates_match"] for row in surfaces)
            if count_available
            else None
        )
        port["matching_surface_count"] = count
        port["matching_surface_count_scope"] = (
            "all saved individual surfaces"
            if count_available
            else "unavailable: comparison budget exceeded"
        )
        port["multiple_matching_surfaces"] = count > 1 if count is not None else None
    return {
        **result,
        "availability": "available",
        "occurrence": metadata,
        "cylindrical_surfaces": deepcopy(page),
        "ports": ports,
        "total": len(surfaces),
        "returned_count": len(page),
        "next_offset": surface_offset + len(page)
        if surface_offset + len(page) < len(surfaces)
        else None,
        "omitted_surface_count": len(surfaces) - len(page),
        "port_count": len(ports),
        "all_declared_ports_included": True,
        "comparison_omitted_count": len(ports) * (len(page) - len(compared)),
        "comparison_budget": MAX_PAGE_COMPARISONS,
        "matching_count_budget": MAX_COUNT_COMPARISONS,
    }
