"""Pure source and output contract for optional appearance-only product photos."""

from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any

FRAME = "cad_z_up_mm_world"
SCOPE = (
    "Appearance-only product image of the complete saved STL; the uniform preset is not "
    "a production material declaration or engineering, manufacturing, or physical approval."
)
PRESETS = ("metal", "plastic")
VERSION = 1
OPPOSED_VIEW = "opposed_upper"
MAX_IMAGE_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
REQUEST_KEYS = {"project_id", "revision_id", "mesh_sha256", "preset", "frame", "version"}
CAPTURED_REQUEST_KEYS = REQUEST_KEYS | {
    "package_id",
    "package_sha256",
    "part_id",
    "file_id",
    "quantity",
}
CAPTURED_SCOPE = (
    "Appearance-only product image of one captured made definition STL in its local mm frame; "
    "the uniform preset is not a production material declaration or engineering, "
    "manufacturing, or physical approval."
)
ASSEMBLY_REQUEST_KEYS = {
    "version",
    "project_id",
    "revision_id",
    "package_id",
    "package_sha256",
    "preset",
    "frame",
    "source_kind",
    "source",
    "selection",
}
ASSEMBLY_SOURCE_KEYS = {
    "saved_document_sha256",
    "captured_document_sha256",
    "assembly_step_sha256",
    "assembly_facts_sha256",
    "topology_sha256",
}
ASSEMBLY_SCOPE = (
    "Appearance-only image of all used made occurrences in their saved world poses, from saved "
    "selection/display tessellation. Bought, reference and unspecified geometry is omitted. "
    "The uniform preset is not a production material or engineering, manufacturing or physical approval."
)
SOURCE_TESSELLATION_KEYS = {
    "purpose",
    "relative_linear_deflection",
    "angular_deflection_radians",
    "selected_body_count",
    "discarded_zero_area_triangles",
    "vertex_precision",
}
DISPLAY_VERTEX_PRECISION = (
    "Saved JSON vertices converted float64 to float32 to float64; no placement transform"
)


class PhotoUnavailableError(RuntimeError):
    """An explicitly optional graphics failure, never source or access failure."""

    def __init__(self, code: str, reason: str) -> None:
        if code not in {
            "renderer_unavailable",
            "execution_unqualified",
            "render_failed",
            "source_unavailable",
            "render_limit",
        }:
            raise ValueError("Invalid optional product photo failure code")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Invalid optional product photo failure reason")
        self.code = code
        self.reason = reason[:500]
        super().__init__(self.reason)


def _checksum(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _finite_json(value: Any, depth: int = 0) -> None:
    if depth > 64:
        raise ValueError("Product photo JSON nesting exceeds its bound")
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _finite_json(item, depth + 1)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _finite_json(item, depth + 1)
        return
    raise ValueError("Product photo requires finite JSON values and string object keys")


def _bounded_json(value: Any, maximum: int) -> None:
    _finite_json(value)
    try:
        content = json.dumps(value, allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise ValueError("Product photo JSON cannot be serialized") from exc
    if len(content) > maximum:
        raise ValueError("Product photo JSON exceeds its byte limit")


def _number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _matrix(value: Any, rows: int, columns: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == rows
        and all(
            isinstance(row, list) and len(row) == columns and all(_number(item) for item in row)
            for row in value
        )
    )


def validated_request(value: dict[str, Any]) -> dict[str, Any]:
    """Validate exact request fields without changing supplied values."""
    if not isinstance(value, dict):
        raise ValueError("Unexpected or missing product photo request fields")
    keys = set(value)
    base_keys = keys - {"view"}
    if base_keys not in (
        REQUEST_KEYS,
        CAPTURED_REQUEST_KEYS,
        ASSEMBLY_REQUEST_KEYS,
    ):
        raise ValueError("Unexpected or missing product photo request fields")
    if "view" in keys:
        if (
            type(value["version"]) is not int
            or value["version"] != 2
            or not isinstance(value["view"], str)
            or value["view"] != OPPOSED_VIEW
        ):
            raise ValueError("Unsupported opposed product photo request")
    elif type(value["version"]) is not int or value["version"] != VERSION:
        raise ValueError("Unsupported product photo contract version")
    for key in ("project_id", "revision_id"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{32}", value[key]):
            raise ValueError("Invalid product photo project or revision identifier")
    assembly = base_keys == ASSEMBLY_REQUEST_KEYS
    if not assembly and not _checksum(value["mesh_sha256"]):
        raise ValueError("Invalid product photo mesh checksum")
    captured = base_keys == CAPTURED_REQUEST_KEYS
    if (
        not isinstance(value["preset"], str)
        or value["preset"] not in PRESETS
        or (
            value["frame"] not in ("definition_local", "document_local")
            if captured
            else value["frame"] != FRAME
        )
    ):
        raise ValueError("Product photo requires a supported uniform preset and saved CAD frame")
    if captured or assembly:
        if not isinstance(value["package_id"], str) or not re.fullmatch(
            r"[a-f0-9]{32}", value["package_id"]
        ):
            raise ValueError("Invalid product photo package identifier")
        if not _checksum(value["package_sha256"]):
            raise ValueError("Invalid product photo package checksum")
    if captured:
        part_id = value["part_id"]
        if not isinstance(part_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", part_id):
            raise ValueError("Invalid product photo made definition identifier")
        if (
            value["file_id"] != f"part-{part_id}-stl"
            or type(value["quantity"]) is not int
            or value["quantity"] <= 0
        ):
            raise ValueError("Invalid product photo captured file or quantity")
    if assembly:
        if value["source_kind"] != "saved_topology_made_assembly":
            raise ValueError("Invalid product photo assembly source kind")
        source, selection = value["source"], value["selection"]
        if (
            not isinstance(source, dict)
            or set(source) != ASSEMBLY_SOURCE_KEYS
            or not all(_checksum(item) for item in source.values())
        ):
            raise ValueError("Invalid product photo assembly source bindings")
        if (
            not isinstance(selection, dict)
            or set(selection) != {"selected", "omitted", "total_bodies"}
            or type(selection["total_bodies"]) is not int
            or not 0 < selection["total_bodies"] <= 4096
        ):
            raise ValueError("Invalid product photo assembly selection")
        seen_paths: set[str] = set()
        seen_indices: set[int] = set()
        for key in ("selected", "omitted"):
            rows = selection[key]
            if not isinstance(rows, list) or (key == "selected" and not rows):
                raise ValueError("Invalid product photo occurrence inventory")
            paths: list[str] = []
            for row in rows:
                if not isinstance(row, dict) or set(row) != {
                    "occurrence_path",
                    "part_id",
                    "role",
                    "body_index",
                }:
                    raise ValueError("Invalid product photo occurrence identity")
                path, part, role, index = (
                    row["occurrence_path"],
                    row["part_id"],
                    row["role"],
                    row["body_index"],
                )
                if (
                    not isinstance(path, str)
                    or not re.fullmatch(
                        r"[A-Za-z][A-Za-z0-9_]{0,63}(?:/[A-Za-z][A-Za-z0-9_]{0,63}){0,63}", path
                    )
                    or not isinstance(part, str)
                    or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", part)
                    or (
                        role != "make"
                        if key == "selected"
                        else role not in ("buy", "reference", "unspecified")
                    )
                    or type(index) is not int
                    or not 0 <= index < selection["total_bodies"]
                    or path in seen_paths
                    or index in seen_indices
                ):
                    raise ValueError("Contradictory product photo occurrence mapping")
                seen_paths.add(path)
                seen_indices.add(index)
                paths.append(path)
            if paths != sorted(paths):
                raise ValueError("Product photo occurrence rows must be sorted")
        if seen_indices != set(range(selection["total_bodies"])):
            raise ValueError("Product photo selection does not cover every saved body")
    return dict(value)


def opposed_request(primary: dict[str, Any]) -> dict[str, Any]:
    """Copy a validated legacy primary request into the single opposed-view variant."""
    request = validated_request(primary)
    if request["version"] != VERSION or "view" in request:
        raise ValueError("Opposed product photo requires a legacy primary request")
    request = deepcopy(request)
    request.update(version=2, view=OPPOSED_VIEW)
    return validated_request(request)


def legacy_request(request: dict[str, Any]) -> dict[str, Any]:
    """Return an independent validated v1 copy for source-family dispatch/comparison."""
    result = deepcopy(validated_request(request))
    result.pop("view", None)
    result["version"] = VERSION
    return validated_request(result)


def _opposed_camera(camera: dict[str, Any], bounds: list[list[float]]) -> None:
    """Validate actual opposed camera basis/center, not just its request label."""
    position, target, up = camera["position"]
    center = [bounds[0][axis] / 2 + bounds[1][axis] / 2 for axis in range(3)]
    if not all(_number(item) for item in center) or any(
        not math.isclose(
            target[axis], center[axis], rel_tol=0, abs_tol=max(1e-8, 4 * math.ulp(center[axis]))
        )
        for axis in range(3)
    ):
        raise ValueError("Opposed product photo camera must target its geometry bounds center")
    direction = [position[axis] - target[axis] for axis in range(3)]
    length = math.hypot(*direction)
    if not all(_number(item) for item in direction) or not math.isfinite(length) or length <= 0:
        raise ValueError("Opposed product photo requires a finite nonzero camera direction")
    outward_length = math.hypot(-1.25, 1.65, 1.6)
    outward = [item / outward_length for item in (-1.25, 1.65, 1.6)]
    horizontal = math.hypot(outward[0], outward[1])
    expected_up = [
        -outward[0] * outward[2] / horizontal,
        -outward[1] * outward[2] / horizontal,
        horizontal,
    ]
    if any(abs(direction[axis] / length - outward[axis]) > 1e-8 for axis in range(3)) or any(
        abs(up[axis] - expected_up[axis]) > 1e-8 for axis in range(3)
    ):
        raise ValueError("Opposed product photo camera has the wrong actual direction or up basis")
    if (
        camera["projection"] != "perspective"
        or abs(camera["view_angle_degrees"] - 35.0) > 1e-10
        or camera["distance_mm"] <= 0
        or not math.isclose(camera["distance_mm"], length, rel_tol=1e-8, abs_tol=1e-8)
        or not 0 <= camera["maximum_absolute_ndc"] <= 1 / 1.20 + 1e-12
    ):
        raise ValueError(
            "Opposed product photo requires its fitted positive 35-degree perspective camera"
        )


def validate_receipt(receipt: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Validate bound image metadata and bounded finite render facts, never CAD acceptance."""
    request = validated_request(request)
    keys = set(request)
    base_keys = set(legacy_request(request))
    if not isinstance(receipt, dict) or set(receipt) != keys | {
        "scope",
        "image",
        "rendering",
    }:
        raise ValueError("Unexpected or missing product photo receipt fields")
    echoed = validated_request({key: receipt[key] for key in keys})
    scope = (
        ASSEMBLY_SCOPE
        if base_keys == ASSEMBLY_REQUEST_KEYS
        else CAPTURED_SCOPE
        if base_keys == CAPTURED_REQUEST_KEYS
        else SCOPE
    )
    if echoed != request or receipt["scope"] != scope:
        raise ValueError("Product photo output does not match its source request and scope")
    image = receipt["image"]
    if (
        not isinstance(image, dict)
        or set(image) != {"file", "sha256", "width", "height"}
        or image["file"] != "photo.png"
        or not _checksum(image["sha256"])
        or type(image["width"]) is not int
        or type(image["height"]) is not int
        or image["width"] != 1024
        or image["height"] != 1024
    ):
        raise ValueError("Product photo requires a bound 1024 by 1024 PNG")
    rendering = receipt["rendering"]
    if not isinstance(rendering, dict):
        raise ValueError("Product photo rendering facts must be an object")
    if base_keys == ASSEMBLY_REQUEST_KEYS:
        tessellation = rendering.get("source_tessellation")
        if (
            not isinstance(tessellation, dict)
            or set(tessellation) != SOURCE_TESSELLATION_KEYS
            or not isinstance(tessellation["purpose"], str)
            or not 0 < len(tessellation["purpose"]) <= 512
            or any(
                not _number(tessellation[key]) or tessellation[key] <= 0
                for key in ("relative_linear_deflection", "angular_deflection_radians")
            )
            or type(tessellation["selected_body_count"]) is not int
            or tessellation["selected_body_count"] != len(request["selection"]["selected"])
            or type(tessellation["discarded_zero_area_triangles"]) is not int
            or tessellation["discarded_zero_area_triangles"] < 0
            or tessellation["vertex_precision"] != DISPLAY_VERTEX_PRECISION
        ):
            raise ValueError("Product photo requires bound display tessellation facts")
    versions = rendering.get("versions")
    if (
        not isinstance(versions, dict)
        or not versions
        or any(not isinstance(value, str) or not value.strip() for value in versions.values())
    ):
        raise ValueError("Product photo requires renderer version facts")
    camera = rendering.get("camera")
    if (
        not isinstance(camera, dict)
        or not _matrix(camera.get("position"), 3, 3)
        or any(
            not _number(camera.get(key))
            for key in ("view_angle_degrees", "distance_mm", "maximum_absolute_ndc")
        )
        or not isinstance(camera.get("projection"), str)
        or not camera["projection"].strip()
    ):
        raise ValueError("Product photo requires finite camera and projection facts")
    lights = rendering.get("lights")
    if (
        not isinstance(lights, list)
        or not lights
        or any(not isinstance(light, dict) or not light for light in lights)
    ):
        raise ValueError("Product photo requires lighting facts")
    for key in ("texture", "ssao"):
        if not isinstance(rendering.get(key), dict) or not rendering[key]:
            raise ValueError("Product photo requires texture and ambient-occlusion facts")
    if (
        not isinstance(rendering.get("shadow_method"), str)
        or not rendering["shadow_method"].strip()
    ):
        raise ValueError("Product photo requires shadow method facts")
    geometry = rendering.get("geometry")
    if (
        not isinstance(geometry, dict)
        or type(geometry.get("triangles")) is not int
        or not 1 <= geometry["triangles"] <= 500_000
        or not _checksum(geometry.get("triangle_sha256"))
        or geometry.get("triangle_positions_unchanged") is not True
    ):
        raise ValueError("Product photo requires unchanged bounded triangle geometry")
    bounds = geometry.get("bounds_mm")
    if (
        not isinstance(bounds, list)
        or not _matrix(bounds, 2, 3)
        or any(bounds[0][axis] > bounds[1][axis] for axis in range(3))
    ):
        raise ValueError("Product photo requires finite ordered geometry bounds")
    if "view" in request:
        _opposed_camera(camera, bounds)
    _bounded_json(receipt, MAX_RECEIPT_BYTES)
    return dict(receipt)
