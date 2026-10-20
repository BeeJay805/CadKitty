"""Saved-STEP planar wire observations, with continuous coverage error bounds.

Native imports occur only in compare_saved_profile. This module never decides
acceptance. The service authenticates the immutable image observation separately.
"""

from __future__ import annotations

import hashlib
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

from forge_mind.workspace.photo_profile import _calibration, _project, profile_digest
from forge_mind.workspace.photo_profile_models import (
    MAX_CONTOUR_POINTS,
    OBSERVER_ERROR_MM,
    SCOPE,
    PhotoProfileRequest,
    PhotoProfileSelector,
)

MAX_SAMPLES = 50000
MAX_DISTANCE_WORK = 100_000_000
NATIVE_NUMERICAL_ERROR_MM = 1e-7


def _observed_loops(observation: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    raw = dict(observation)
    identity = raw.pop("observation_sha256", None)
    if identity != profile_digest(raw) or raw.get("method") != "calibrated_photo_profile_v1":
        raise ValueError("Profile observation digest or method changed")
    request = PhotoProfileRequest.model_validate(raw["request"])
    if (
        raw.get("request_sha256") != profile_digest(request.model_dump(mode="json"))
        or raw.get("vision_image_sha256") != request.vision_image_sha256
    ):
        raise ValueError("Profile observation request/image binding changed")
    width, height = raw["image_size_px"]
    calibration = _calibration(request, width, height)
    if calibration != raw.get("calibration"):
        raise ValueError("Profile calibration differs from deterministic replay")
    selected = [c for c in raw["components"] if c["id"] == raw["selected_component"]]
    if len(selected) != 1 or selected[0]["touches_region_boundary"]:
        raise ValueError("Profile needs one complete selected component")
    loops: list[dict[str, Any]] = selected[0]["loops"]
    if (
        not loops
        or loops[0]["role"] != "outer"
        or any(loop["role"] != "inner" for loop in loops[1:])
    ):
        raise ValueError("Profile loop inventory is incomplete")
    count = 0
    for loop in loops:
        pixel = np.asarray(loop["points_px"], dtype=float)
        metric = np.asarray(loop["points_mm"], dtype=float)
        if (
            pixel.ndim != 2
            or pixel.shape[1] != 2
            or len(pixel) < 4
            or metric.shape != pixel.shape
            or not np.isfinite(metric).all()
            or not np.array_equal(pixel[0], pixel[-1])
            or not np.array_equal(metric[0], metric[-1])
        ):
            raise ValueError("Profile needs finite complete closed metric loops")
        count += len(pixel)
        if count > MAX_CONTOUR_POINTS or not np.allclose(
            _project(pixel, calibration["pixel_to_plane_mm"]), metric, rtol=0, atol=1e-10
        ):
            raise ValueError("Profile metric points differ from the frozen pixel projection")
    return loops


def _polyline_curves(points: Any) -> list[dict[str, Any]]:
    import numpy as np

    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or len(points) < 4
        or not np.isfinite(points).all()
        or not np.array_equal(points[0], points[-1])
    ):
        raise ValueError("Expected a finite closed profile polyline")
    return [
        {"kind": "line", "a": a.tolist(), "b": b.tolist()}
        for a, b in pairwise(points)
        if np.linalg.norm(b - a) > 1e-12
    ]


def _samples(curves: list[dict[str, Any]]) -> tuple[Any, float]:
    import numpy as np

    points = []
    length = 0.0
    total = 0
    for curve in curves:
        if curve["kind"] == "line":
            a, b = np.array(curve["a"]), np.array(curve["b"])
            arc_length = float(np.linalg.norm(b - a))
        elif curve["kind"] == "circle":
            arc_length = 2 * math.pi * curve["radius_mm"]
        else:
            raise ValueError("Unsupported analytic profile curve")
        if not math.isfinite(arc_length) or arc_length <= 0:
            raise ValueError("Degenerate analytic profile curve")
        intervals = max(1, math.ceil(arc_length / (2 * OBSERVER_ERROR_MM)))
        total += intervals + 1
        if total > MAX_SAMPLES:
            raise ValueError("Continuous profile comparison exceeds its sample budget")
        t = np.linspace(0, 1, intervals + 1)
        if curve["kind"] == "line":
            points.append(a + t[:, None] * (b - a))
        else:
            angle = t * 2 * math.pi
            points.append(
                np.asarray(curve["center_mm"])
                + curve["radius_mm"] * np.column_stack((np.cos(angle), np.sin(angle)))
            )
        length += arc_length
    if not points:
        raise ValueError("Profile boundary is empty")
    return np.concatenate(points), length


def _distances(points: Any, curves: list[dict[str, Any]]) -> Any:
    import numpy as np

    if len(points) * len(curves) > MAX_DISTANCE_WORK:
        raise ValueError("Continuous profile comparison exceeds its distance-work budget")
    line_a = np.array([c["a"] for c in curves if c["kind"] == "line"], dtype=float).reshape(-1, 2)
    line_b = np.array([c["b"] for c in curves if c["kind"] == "line"], dtype=float).reshape(-1, 2)
    circles = [c for c in curves if c["kind"] == "circle"]
    result = []
    for offset in range(0, len(points), 64):
        batch = points[offset : offset + 64]
        best = np.full(len(batch), np.inf)
        if len(line_a):
            delta = line_b - line_a
            lengths = np.sum(delta * delta, axis=1)
            if np.any(lengths <= 0):
                raise ValueError("Degenerate target segment")
            relative = batch[:, None, :] - line_a[None, :, :]
            t = np.clip(np.sum(relative * delta[None, :, :], axis=2) / lengths, 0, 1)
            best = np.linalg.norm(relative - t[:, :, None] * delta[None, :, :], axis=2).min(axis=1)
        for circle in circles:
            best = np.minimum(
                best,
                np.abs(np.linalg.norm(batch - circle["center_mm"], axis=1) - circle["radius_mm"]),
            )
        if not np.isfinite(best).all():
            raise ValueError("Profile target boundary is empty or nonfinite")
        result.append(best)
    return np.concatenate(result)


def compare_profile_curves(
    observed: list[dict[str, Any]], saved: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bound symmetric boundary distance over complete line/full-circle curves.

    Distance to a closed target set is 1-Lipschitz. Every source arc point lies
    within half an interval's arc length of an evaluated endpoint. Consequently
    max sampled distance <= max continuous distance <= sampled max + error.
    This is a coverage bound, not a finite-sample PASS or an image uncertainty.
    """
    import numpy as np

    source_curves = [curve for loop in observed for curve in _polyline_curves(loop["points_mm"])]
    saved_curves = [curve for loop in saved for curve in loop["curves"]]
    directions = []
    for name, source, target in (
        ("image_to_step", source_curves, saved_curves),
        ("step_to_image", saved_curves, source_curves),
    ):
        samples, length = _samples(source)
        distances = _distances(samples, target)
        maximum = float(distances.max())
        worst = int(np.argmax(distances))
        directions.append(
            {
                "direction": name,
                "lower_mm": max(0.0, maximum - NATIVE_NUMERICAL_ERROR_MM),
                "upper_mm": maximum + OBSERVER_ERROR_MM + NATIVE_NUMERICAL_ERROR_MM,
                "sample_count": len(samples),
                "complete_boundary_length_mm": length,
                "worst_point_mm": samples[worst].tolist(),
            }
        )
    return {
        "directed_residuals": directions,
        "symmetric_residual_interval_mm": [
            max(d["lower_mm"] for d in directions),
            max(d["upper_mm"] for d in directions),
        ],
        "continuous_coverage_error_mm": OBSERVER_ERROR_MM,
        "numerical_error_mm": NATIVE_NUMERICAL_ERROR_MM,
        "coverage_method": "All analytic line/full-circle arcs; maximum arc interval <= 2*coverage error; distance-to-set is 1-Lipschitz",
        "observed_inner_loop_count": len(observed) - 1,
        "saved_inner_loop_count": len(saved) - 1,
        "inner_loop_count_difference": len(saved) - len(observed),
    }


def _saved_face(
    shape: Any, selector: PhotoProfileSelector
) -> tuple[Any, list[dict[str, Any]], int]:
    import build123d as b
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    if not shape.is_valid:
        raise ValueError("Saved STEP shape is invalid")
    solids = list(shape.solids())
    if selector.body_index >= len(solids):
        raise ValueError("Selected body is unavailable")
    body = solids[selector.body_index]
    faces = []
    for face in body.faces():
        if face.geom_type != b.GeomType.PLANE:
            continue
        normal = tuple(face.normal_at())
        if abs(normal[0]) <= 1e-9 and abs(normal[1]) <= 1e-9 and normal[2] > 1 - 1e-9:
            bounds = face.bounding_box()
            if (
                abs(bounds.min.Z - selector.z_mm) <= NATIVE_NUMERICAL_ERROR_MM
                and abs(bounds.max.Z - selector.z_mm) <= NATIVE_NUMERICAL_ERROR_MM
            ):
                faces.append(face)
    if len(faces) != 1:
        raise ValueError("Selected +Z planar face is missing or ambiguous")
    face = faces[0]
    wires = [face.outer_wire(), *face.inner_wires()]
    saved = []
    for index, wire in enumerate(wires):
        if not wire.is_closed:
            raise ValueError("Saved planar face has an open boundary wire")
        curves: list[dict[str, Any]] = []
        for edge in wire.edges():
            if edge.geom_type == b.GeomType.LINE:
                a, z = tuple(edge.position_at(0)), tuple(edge.position_at(1))
                if (
                    abs(a[2] - selector.z_mm) > NATIVE_NUMERICAL_ERROR_MM
                    or abs(z[2] - selector.z_mm) > NATIVE_NUMERICAL_ERROR_MM
                ):
                    raise ValueError("Saved boundary leaves the selected plane")
                curves.append({"kind": "line", "a": list(a[:2]), "b": list(z[:2])})
            elif edge.geom_type == b.GeomType.CIRCLE:
                circle = BRepAdaptor_Curve(edge.wrapped).Circle()
                radius = float(circle.Radius())
                axis = circle.Axis().Direction()
                if (
                    abs(abs(axis.Z()) - 1) > 1e-9
                    or abs(float(edge.length) - 2 * math.pi * radius) > NATIVE_NUMERICAL_ERROR_MM
                ):
                    raise ValueError("Only complete coplanar circles are supported")
                center = circle.Location()
                curves.append(
                    {"kind": "circle", "center_mm": [center.X(), center.Y()], "radius_mm": radius}
                )
            else:
                raise ValueError("Unsupported saved boundary curve; complete coverage unavailable")
        if not curves:
            raise ValueError("Saved face has an empty wire")
        saved.append({"role": "outer" if index == 0 else "inner", "curves": curves})
    return body, saved, len(solids)


def compare_saved_profile(
    revision_directory: Path,
    observation: dict[str, Any],
    selector: PhotoProfileSelector | dict[str, Any],
) -> dict[str, Any]:
    """Read one exact saved STEP; never alter geometry or refit its camera."""
    selector = PhotoProfileSelector.model_validate(selector)
    from forge_mind.workspace.engine_identity import engine_fingerprint

    observed = _observed_loops(observation)
    step = revision_directory / selector.step_artifact
    from forge_mind.workspace.processing import regular_file

    if (
        not regular_file(step, 64 * 1024 * 1024)
        or step.resolve().parent != revision_directory.resolve()
    ):
        raise ValueError("Saved profile STEP must be an owned bounded regular file")
    raw = step.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    import build123d as b

    body, saved, body_count = _saved_face(b.import_step(step), selector)
    comparison = compare_profile_curves(observed, saved)
    if hashlib.sha256(step.read_bytes()).hexdigest() != digest:
        raise ValueError("Saved profile STEP changed during observation")
    bounds = body.bounding_box()
    # These projected polylines are display evidence only. The residual bounds
    # above cover the complete analytic curves, independent of this overlay.
    projected_loops = []
    for loop in saved:
        points, _ = _samples(loop["curves"])
        projected_loops.append(
            {
                "role": loop["role"],
                "points_px": _project(
                    points, observation["calibration"]["plane_mm_to_pixel"]
                ).tolist(),
                "connected_polyline": False,
            }
        )
    guidance = [
        {
            "direction": item["direction"],
            "location_plane_mm": item["worst_point_mm"],
            "residual_interval_mm": [item["lower_mm"], item["upper_mm"]],
            "action": "Inspect the saved profile near this location against the fixed image observation; keep calibration unchanged.",
        }
        for item in comparison["directed_residuals"]
    ]
    result = {
        "schema_version": 1,
        "method": "saved_step_photo_profile_v1",
        "engine_fingerprint": engine_fingerprint(),
        "status": "measured",
        "scope": SCOPE
        + " Only the explicitly selected visible +Z planar face is compared; hidden depth and through-cut truth are not verified.",
        "observation_sha256": observation["observation_sha256"],
        "vision_image_sha256": observation["vision_image_sha256"],
        "step_sha256": digest,
        "selector": selector.model_dump(mode="json"),
        "body_count": body_count,
        "bounds_mm": {"min": list(bounds.min), "max": list(bounds.max)},
        "saved_loops": saved,
        "overlay_geometry": {
            "vision_image_sha256": observation["vision_image_sha256"],
            "saved_boundary_points": projected_loops,
            "scope": "Projected boundary points for a reference-image overlay; unordered edges must not be joined across gaps.",
        },
        "repair_guidance": guidance,
        "coverage": {
            "all_selected_face_wires": True,
            "all_wire_edges_supported": True,
            "camera_refitted": False,
        },
        **comparison,
    }
    result["comparison_sha256"] = profile_digest(result)
    return result
