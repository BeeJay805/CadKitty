"""Deterministic, source-hashed planar measurements and editable proposals.

Pixels and fitted geometry remain derived observations. The host authenticates
source ownership and persists this output; its digest is not an authority token.
All image/scientific imports are lazy. No native CAD is imported here.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from itertools import pairwise
from typing import Any

from forge_mind.workspace.photo_profile_models import (
    CALIBRATION_RESIDUAL_PX,
    MAX_COMPONENTS,
    MAX_CONTOUR_POINTS,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_PROPOSAL_POINTS,
    SCOPE,
    PhotoProfileRequest,
)


def profile_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _project(points: Any, matrix: Any) -> Any:
    import numpy as np

    points = np.asarray(points, dtype=float)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-10):
        raise ValueError("Profile projection reaches an undefined plane horizon")
    result = homogeneous[:, :2] / homogeneous[:, 2, None]
    if not np.isfinite(result).all() or np.abs(result).max() > 1e6:
        raise ValueError("Profile projection exceeds finite metric bounds")
    return result


def _normalization(points: Any) -> tuple[Any, Any]:
    import numpy as np

    center = points.mean(axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    spread = float(distances.mean())
    if spread < 1e-8:
        raise ValueError("Calibration anchors have no usable spread")
    scale = math.sqrt(2) / spread
    transform = np.array(
        [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]]
    )
    return _project(points, transform), transform


def _calibration(request: PhotoProfileRequest, width: int, height: int) -> dict[str, Any]:
    import numpy as np

    if not request.assumptions.planar:
        raise ValueError("Metric profile requires an explicitly declared common plane")
    if request.assumptions.distortion != "negligible_declared":
        raise ValueError("Unknown lens distortion leaves metric reconstruction advisory")
    if len(request.calibration_anchors) < 4:
        raise ValueError("Metric profile needs four calibration anchors")
    all_points = [*request.calibration_anchors, *request.check_points]
    if any(not (0 <= p.pixel[0] < width and 0 <= p.pixel[1] < height) for p in all_points):
        raise ValueError("Calibration pixel is outside the retained image")
    pixel = np.array([p.pixel for p in request.calibration_anchors])
    metric = np.array([p.plane_mm for p in request.calibration_anchors])
    source, ts = _normalization(pixel)
    target, tt = _normalization(metric)
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, target, strict=True):
        rows.extend(
            ([-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v])
        )
    _, singular, vectors = np.linalg.svd(np.asarray(rows), full_matrices=True)
    if singular[7] < 1e-8 or singular[0] / singular[7] > 1e8:
        raise ValueError("Calibration anchors are degenerate or ill-conditioned")
    matrix = np.linalg.inv(tt) @ vectors[-1].reshape(3, 3) @ ts
    if abs(matrix[2, 2]) < 1e-12:
        raise ValueError("Calibration plane normalization is unstable")
    matrix /= matrix[2, 2]
    inverse = np.linalg.inv(matrix)
    corners = (
        np.array([[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]])
        @ matrix.T
    )
    if np.min(corners[:, 2]) <= 1e-8:
        raise ValueError("Calibration horizon crosses the retained image")
    checks = []
    for role, points in (("anchor", request.calibration_anchors), ("check", request.check_points)):
        if not points:
            continue
        predicted = _project([p.plane_mm for p in points], inverse)
        residuals = np.linalg.norm(predicted - np.array([p.pixel for p in points]), axis=1)
        if float(residuals.max()) > CALIBRATION_RESIDUAL_PX:
            raise ValueError(
                "Calibration or independent check residual exceeds the fixed pixel guard"
            )
        checks.append({"role": role, "residuals_px": residuals.tolist()})
    return {
        "pixel_to_plane_mm": matrix.tolist(),
        "plane_mm_to_pixel": inverse.tolist(),
        "checks": checks,
        "independent_check_status": "observed" if request.check_points else "not_run",
        "residual_guard_px": CALIBRATION_RESIDUAL_PX,
        "scope": "Projective plane under declared planarity and negligible-distortion assumptions",
    }


def _area(points: Any) -> float:
    import numpy as np

    return float(np.sum(points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]) / 2)


def _circle_traversal_bound(
    points: Any, center: Any, radius: float, residual: float
) -> float | None:
    """Bound an ordered simple contour, not a sorted set of radial samples.

    One signed turn supplies angular coverage. Each actual chord stays in the
    radial band given by its endpoint residual and its own angular span. Local
    backtracking is allowed for raster stairs; zero/multiple turns are not.
    """
    import numpy as np

    if not np.array_equal(points[0], points[-1]):
        return None
    local = points - center
    vectors = local[:-1]
    following = np.roll(vectors, -1, axis=0)
    if np.any(np.linalg.norm(vectors, axis=1) <= 1e-12):
        return None
    angles = np.arctan2(
        vectors[:, 0] * following[:, 1] - vectors[:, 1] * following[:, 0],
        np.sum(vectors * following, axis=1),
    )
    turn = math.fsum(float(angle) for angle in angles)
    signed_area = _area(local)
    if not math.isclose(abs(turn), 2 * math.pi, rel_tol=0, abs_tol=1e-7):
        return None
    if turn * signed_area <= 0:
        return None
    gap = float(np.abs(angles).max())
    if gap >= math.pi / 4:
        return None
    bound = residual + radius * (1 - math.cos(gap / 2))
    if not math.isfinite(bound) or bound >= radius:
        return None
    # This is the area band from the same radial/chord error, not an authored
    # area allowance. Translation above avoids subtracting large world origins.
    arithmetic = 64 * float(np.finfo(float).eps) * len(points) * max(1.0, radius * radius)
    if not (
        math.pi * (radius - bound) ** 2 - arithmetic
        <= abs(signed_area)
        <= math.pi * (radius + bound) ** 2 + arithmetic
    ):
        return None
    return bound


def _fit_loop(points: Any, request: PhotoProfileRequest) -> dict[str, Any]:
    import numpy as np
    from skimage.measure import approximate_polygon

    if request.allow_circle_fits and len(points) >= 16:
        unique = points[:-1]
        center0 = unique.mean(axis=0)
        local = unique - center0
        a = np.column_stack((2 * local, np.ones(len(local))))
        solution, _, rank, _ = np.linalg.lstsq(a, np.sum(local * local, axis=1), rcond=None)
        if rank == 3:
            center = solution[:2] + center0
            radius_squared = float(solution[2] + np.dot(solution[:2], solution[:2]))
            if radius_squared > 0:
                radius = math.sqrt(radius_squared)
                residual = float(np.abs(np.linalg.norm(unique - center, axis=1) - radius).max())
                fit_bound = _circle_traversal_bound(points, center, radius, residual)
                if fit_bound is not None and fit_bound <= request.fit_tolerance_mm:
                    return {
                        "kind": "circle",
                        "center_mm": center.tolist(),
                        "radius_mm": radius,
                        "max_fit_residual_mm": residual,
                        "contour_chord_guard_mm": fit_bound,
                    }
    simplified = approximate_polygon(points, tolerance=request.fit_tolerance_mm)
    if (
        len(simplified) < 4
        or len(simplified) > MAX_PROPOSAL_POINTS
        or not np.array_equal(simplified[0], simplified[-1])
    ):
        raise ValueError("Profile polygon is degenerate or exceeds the editable point bound")
    return {
        "kind": "polygon",
        "points_mm": simplified.tolist(),
        "fit_tolerance_mm": request.fit_tolerance_mm,
    }


def _pixel_features(points: Any, loop_id: str) -> dict[str, Any]:
    """Detect stable polygon vertices; fitted line intersections remain hypotheses."""
    import numpy as np
    from skimage.measure import approximate_polygon

    span = np.ptp(points, axis=0)
    tolerance = max(1.0, float(min(span)) * 0.005)
    polygon = approximate_polygon(points, tolerance=tolerance)
    if len(polygon) < 4 or len(polygon) > 65:
        return {"vertices": [], "scope": "No bounded dominant polygon was detected"}
    vertices = polygon[:-1]
    # Retain the measured contour and its ordering. Refine a quadrilateral only
    # from the intervening complete contour edges, never from a bounding box.
    if len(vertices) == 4:
        indices = [int(np.argmin(np.linalg.norm(points - p, axis=1))) for p in vertices]
        lines: list[tuple[Any, float]] = []
        for index, following in zip(indices, indices[1:] + indices[:1], strict=True):
            edge = (
                points[index : following + 1]
                if following > index
                else np.vstack((points[index:-1], points[: following + 1]))
            )
            if len(edge) < 8:
                lines = []
                break
            edge = edge[2:-2]
            center = edge.mean(axis=0)
            _, _, vectors = np.linalg.svd(edge - center, full_matrices=False)
            normal = vectors[-1]
            lines.append((normal, float(normal @ center)))
        if len(lines) == 4:
            refined: list[Any] = []
            for i in range(4):
                matrix = np.vstack((lines[i - 1][0], lines[i][0]))
                if abs(float(np.linalg.det(matrix))) < 1e-4:
                    refined = []
                    break
                point = np.linalg.solve(matrix, [lines[i - 1][1], lines[i][1]])
                if float(np.linalg.norm(point - vertices[i])) > 2 * tolerance:
                    refined = []
                    break
                refined.append(point)
            if refined:
                vertices = np.asarray(refined)
    return {
        "vertices": [
            {"id": f"{loop_id}_vertex_{i}", "pixel": p.tolist()} for i, p in enumerate(vertices)
        ],
        "simplification_tolerance_px": tolerance,
        "scope": "Derived dominant contour vertices/line intersections; not owner measurements or independent checks",
    }


def _rectangle_fit(points: Any, size: tuple[float, float], matrix: Any) -> dict[str, Any]:
    """Require complete two-way contour support before constraining a rectangle."""
    import numpy as np

    from forge_mind.workspace.photo_profile_native import compare_profile_curves

    width, height = size
    rectangle = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [0.0, 0.0]]
    # Pixel uncertainty is a prospective software bound, not a physical claim.
    metric = np.asarray(points)
    inverse = np.linalg.inv(np.asarray(matrix))
    pixels = _project(metric[:: max(1, len(metric) // 256)], inverse)
    delta = _project(np.add(pixels, [1, 0]), matrix) - _project(pixels, matrix)
    delta_y = _project(np.add(pixels, [0, 1]), matrix) - _project(pixels, matrix)
    pixel_mm = max(
        float(np.linalg.norm(delta, axis=1).max()), float(np.linalg.norm(delta_y, axis=1).max())
    )
    budget = 3 * pixel_mm + 0.02
    if budget > 0.5 or budget > min(size) / 50:
        raise ValueError("Rectangle image uncertainty exceeds the fixed reconstruction bound")
    local = metric - [width / 2, height / 2]
    a, b = local[:-1], local[1:]
    turns = np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))
    turn = math.fsum(float(value) for value in turns)
    area = _area(metric)
    # A C/U-shaped thin band can lie close to every rectangle edge while
    # leaving its center empty. Boundary distance alone cannot fill its region.
    if (
        not math.isclose(abs(turn), 2 * math.pi, rel_tol=0, abs_tol=1e-7)
        or area * turn <= 0
        or not (width - 2 * budget) * (height - 2 * budget)
        <= abs(area)
        <= (width + 2 * budget) * (height + 2 * budget)
    ):
        raise ValueError("Observed contour does not preserve the rectangle's material region")
    raw = {"role": "outer", "points_mm": metric.tolist()}
    exact = {
        "role": "outer",
        "curves": [{"kind": "line", "a": a, "b": b} for a, b in pairwise(rectangle)],
    }
    comparison = compare_profile_curves([raw], [exact])
    if comparison["symmetric_residual_interval_mm"][1] > budget:
        raise ValueError(
            "Complete observed outer contour is inconsistent with the source rectangle"
        )
    return {
        "kind": "polygon",
        "points_mm": rectangle,
        "constraint": "source_rectangle_hypothesis",
        "support_budget_mm": budget,
        "support_residual_interval_mm": comparison["symmetric_residual_interval_mm"],
    }


def _proposal(loops: list[dict[str, Any]], thickness: float) -> dict[str, Any]:
    from forge_mind.workspace.document import DesignDocument

    features: list[dict[str, Any]] = []
    output = ""
    for index, loop in enumerate(loops):
        fit = loop["fit"]
        prefix = f"loop_{index}"
        if fit["kind"] == "circle":
            x, y = fit["center_mm"]
            points = [{"id": "center", "x": x, "y": y}]
            entities = [
                {"id": "circle", "kind": "circle", "points": ["center"], "radius": fit["radius_mm"]}
            ]
        else:
            vertices = fit["points_mm"][:-1]
            # Each separate sketch/extrude uses +XY construction orientation.
            # Reverse only the copied proposal, never the observed/fitted loops.
            signed_area = sum(
                a[0] * b[1] - b[0] * a[1]
                for a, b in zip(vertices, vertices[1:] + vertices[:1], strict=True)
            )
            if signed_area < 0:
                vertices = list(reversed(vertices))
            points = [{"id": f"p{i}", "x": x, "y": y} for i, (x, y) in enumerate(vertices)]
            entities = [
                {"id": f"e{i}", "kind": "line", "points": [f"p{i}", f"p{(i + 1) % len(vertices)}"]}
                for i in range(len(vertices))
            ]
        features.extend(
            [
                {
                    "id": prefix + "_sketch",
                    "name": "Derived visible " + loop["role"],
                    "kind": "sketch",
                    "sketch": {"points": points, "entities": entities},
                },
                {
                    "id": prefix + "_solid",
                    "name": "Explicit depth",
                    "kind": "extrude",
                    "inputs": [prefix + "_sketch"],
                    "bindings": {"amount": "thickness_mm"},
                },
            ]
        )
        if index == 0:
            output = prefix + "_solid"
        else:
            cut = prefix + "_cut"
            features.append(
                {
                    "id": cut,
                    "name": "Explicit through cut",
                    "kind": "cut",
                    "inputs": [output, prefix + "_solid"],
                }
            )
            output = cut
    return DesignDocument.model_validate(
        {
            "name": "Derived photo profile proposal",
            "variables": {"thickness_mm": thickness},
            "features": features,
            "outputs": [output],
        }
    ).model_dump(mode="json")


def analyze_profile(
    image_png: bytes, request: PhotoProfileRequest | dict[str, Any]
) -> dict[str, Any]:
    """Measure all visible threshold components; select explicitly before proposing CAD."""
    import numpy as np
    from PIL import Image
    from skimage.measure import approximate_polygon, find_contours, label, regionprops

    from forge_mind.workspace.engine_identity import engine_fingerprint

    request = PhotoProfileRequest.model_validate(request)
    if (
        not 0 < len(image_png) <= MAX_IMAGE_BYTES
        or hashlib.sha256(image_png).hexdigest() != request.vision_image_sha256
    ):
        raise ValueError("Profile image does not match its retained normalized hash or byte bound")
    with Image.open(io.BytesIO(image_png)) as image:
        if (
            image.format != "PNG"
            or image.width * image.height > MAX_IMAGE_PIXELS
            or max(image.size) > 2048
            or getattr(image, "n_frames", 1) != 1
        ):
            raise ValueError("Profile requires a bounded single normalized PNG image")
        width, height = image.size
        if image.getexif().get(274, 1) != 1:
            raise ValueError("Profile PNG must use normalized pixel orientation")
        gray = np.array(image.convert("L"))
    x0, y0, x1, y1 = request.region or (0, 0, width, height)
    if x1 > width or y1 > height:
        raise ValueError("Profile region lies outside the retained image")
    region = gray[y0:y1, x0:x1]
    mask = (
        region <= request.threshold if request.foreground == "dark" else region >= request.threshold
    )
    labels, count = label(mask, connectivity=1, return_num=True)
    if count > MAX_COMPONENTS:
        raise ValueError("Visible component inventory exceeds its explicit bound")
    selected: int | None = 1 if count == 1 else None
    if request.seed_pixel is not None:
        sx, sy = request.seed_pixel
        if not (x0 <= sx < x1 and y0 <= sy < y1):
            raise ValueError("Component seed lies outside the requested region")
        selected = int(labels[int(sy) - y0, int(sx) - x0]) or None
        if selected is None:
            raise ValueError("Component seed does not identify foreground material")
    warnings = [
        "Derived threshold contours are observations, not confirmed source dimensions or physical fit."
    ]
    calibration = None
    try:
        calibration = _calibration(request, width, height)
    except ValueError as exc:
        warnings.append(str(exc))
    if calibration and not request.check_points:
        warnings.append(
            "No independent calibration check points were supplied; proposed coordinates remain unverified estimates."
        )
    components = []
    total_points = 0
    for properties in regionprops(labels):
        component_id = int(properties.label)
        row0, column0, row1, column1 = properties.bbox
        component = properties.image
        clipped = row0 == 0 or column0 == 0 or row1 == region.shape[0] or column1 == region.shape[1]
        contours = find_contours(np.pad(component, 1), 0.5, fully_connected="low")
        pixels = [
            np.column_stack((c[:, 1] - 1 + x0 + column0, c[:, 0] - 1 + y0 + row0)) for c in contours
        ]
        pixels.sort(key=lambda p: abs(_area(p)), reverse=True)
        loops = []
        for i, points in enumerate(pixels):
            total_points += len(points)
            if total_points > MAX_CONTOUR_POINTS:
                raise ValueError("Complete contour inventory exceeds its explicit point bound")
            if not np.array_equal(points[0], points[-1]):
                raise ValueError("Visible contour is not closed")
            loop: dict[str, Any] = {
                "id": f"component_{component_id}_loop_{i}",
                "role": "outer" if i == 0 else "inner",
                "points_px": points.tolist(),
                "closed": True,
            }
            display_points = approximate_polygon(points, tolerance=0.5)
            if len(display_points) < 4:
                display_points = points
            loop["pixel_outline"] = {
                "points_px": display_points.tolist(),
                "display_tolerance_px": 0.5,
                "scope": "Simplified display outline only; all raw points retained for measurement/comparison",
            }
            loop["detected_features"] = _pixel_features(points, loop["id"])
            if calibration:
                metric = _project(points, calibration["pixel_to_plane_mm"])
                loop.update(points_mm=metric.tolist(), signed_area_mm2=_area(metric))
                try:
                    loop["fit"] = (
                        _rectangle_fit(
                            metric, request.outer_rectangle_mm, calibration["pixel_to_plane_mm"]
                        )
                        if i == 0 and request.outer_rectangle_mm is not None
                        else _fit_loop(metric, request)
                    )
                except ValueError as exc:
                    loop["fit"] = None
                    warnings.append(str(exc))
            loops.append(loop)
        components.append(
            {
                "id": component_id,
                "pixel_count": int(properties.area),
                "seed_pixel": [
                    int(properties.coords[0, 1]) + x0,
                    int(properties.coords[0, 0]) + y0,
                ],
                "bounds_px": [column0 + x0, row0 + y0, column1 - 1 + x0, row1 - 1 + y0],
                "touches_region_boundary": clipped,
                "inner_loop_count": max(0, len(loops) - 1),
                "loops": loops,
            }
        )
        if calibration and loops:
            outer = np.asarray(loops[0]["points_mm"])
            low, high = outer.min(axis=0), outer.max(axis=0)
            components[-1]["bounds_mm"] = {"min": low.tolist(), "max": high.tolist()}
            components[-1]["dimensions_mm"] = (high - low).tolist()
            components[-1]["visible_area_mm2"] = abs(loops[0]["signed_area_mm2"]) - sum(
                abs(loop["signed_area_mm2"]) for loop in loops[1:]
            )
    chosen = next((c for c in components if c["id"] == selected), None)
    proposal = None
    if (
        chosen
        and calibration
        and not chosen["touches_region_boundary"]
        and all(loop.get("fit") for loop in chosen["loops"])
    ):
        points_needed = sum(len(loop["fit"].get("points_mm", [])) or 1 for loop in chosen["loops"])
        if points_needed > MAX_PROPOSAL_POINTS:
            warnings.append("Complete editable proposal exceeds its aggregate point bound.")
        elif (
            request.thickness_mm is not None
            and request.through_cuts is not None
            and (not chosen["inner_loop_count"] or request.through_cuts)
        ):
            proposal = _proposal(chosen["loops"], request.thickness_mm)
        else:
            warnings.append(
                "Editable solid requires explicit thickness and through-cut semantics; hidden depth remains unknown."
            )
    if selected is None:
        warnings.append(
            "Select one complete foreground component explicitly before proposing a solid."
        )
    if chosen and chosen["touches_region_boundary"]:
        warnings.append(
            "Selected component touches the image/region boundary; complete outline is unavailable."
        )
    result = {
        "schema_version": 1,
        "method": "calibrated_photo_profile_v1",
        "engine_fingerprint": engine_fingerprint(),
        "scope": SCOPE,
        "status": "measured" if calibration and request.check_points else "advisory",
        "vision_image_sha256": request.vision_image_sha256,
        "request": request.model_dump(mode="json"),
        "request_sha256": profile_digest(request.model_dump(mode="json")),
        "image_size_px": [width, height],
        "region_px": [x0, y0, x1, y1],
        "calibration": calibration,
        "components": components,
        "complete_visible_component_count": count,
        "selected_component": selected,
        "proposal_document": proposal,
        "warnings": warnings,
    }
    result["observation_sha256"] = profile_digest(result)
    return result
