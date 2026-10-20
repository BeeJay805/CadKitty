"""Appearance-only PBR rendering of hash-bound saved STL in an isolated worker.

Scene parameters come from visually reviewed render_probe_v4.py,
SHA256 3186bb062ceafe5bd5f448eb5747ee31e61041737bc763cbe44917c1f7371049.
This optional renderer never rebuilds CAD or substitutes an inspection diagram.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import stat
import struct
import sys
from pathlib import Path
from typing import Any

from forge_mind.workspace.product_photo_contract import (
    ASSEMBLY_REQUEST_KEYS,
    ASSEMBLY_SCOPE,
    CAPTURED_REQUEST_KEYS,
    CAPTURED_SCOPE,
    DISPLAY_VERTEX_PRECISION,
    MAX_IMAGE_BYTES,
    MAX_RECEIPT_BYTES,
    OPPOSED_VIEW,
    SCOPE,
    PhotoUnavailableError,
    legacy_request,
    validate_receipt,
    validated_request,
)

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_TRIANGLES = 500_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def read_bytes(path: Path, maximum: int) -> bytes:
    """Refuse linked/oversized staged inputs and bound growth during the read."""
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not 0 < info.st_size <= maximum
    ):
        raise ValueError("Product-photo file is missing, linked or oversized")
    with path.open("rb") as source:
        data = source.read(maximum + 1)
    if not 0 < len(data) <= maximum:
        raise ValueError("Product-photo file exceeded its byte budget during read")
    return data


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate product-photo JSON key")
        result[key] = value
    return result


def _nonfinite(value: str) -> Any:
    raise ValueError("Nonfinite product-photo JSON number: " + value)


def validated_triangles(value: Any) -> Any:
    """Validate triangle coordinates without repairing or moving the surface."""
    import numpy as np

    triangles = np.asarray(value, dtype="<f8").copy()
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected STL triangle coordinates")
    if not 0 < len(triangles) <= MAX_TRIANGLES or not np.isfinite(triangles).all():
        raise ValueError("Invalid or excessive product-photo triangle input")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        area = np.linalg.norm(cross, axis=1)
    if not np.isfinite(cross).all() or not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError("Degenerate or unrepresentable product-photo triangle")
    return triangles


def fitted_camera(value: Any, *, view: str | None = None) -> dict[str, Any]:
    """Fit every actual point at its perspective depth, with a 20% margin."""
    if view is not None and view != OPPOSED_VIEW:
        raise ValueError("Unsupported product-photo camera view")
    import numpy as np

    points = np.asarray(value, dtype="<f8")
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Camera requires finite three-dimensional points")
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) / 2
    extent = float(np.linalg.norm(high - low))
    if not np.isfinite(center).all() or not math.isfinite(extent) or extent <= 0:
        raise ValueError("Camera requires a finite positive extent")
    outward = np.array([1.25, -1.65, 1.6] if view is None else [-1.25, 1.65, 1.6])
    forward = -outward / np.linalg.norm(outward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - center
    projected = np.column_stack((relative @ right, relative @ up))
    view_angle = 35.0
    half_angle_tan = np.tan(np.deg2rad(view_angle / 2))
    along = relative @ (-forward)
    distance = float((np.abs(projected).max(axis=1) * 1.20 / half_angle_tan + along).max())
    depth = distance - along
    if not math.isfinite(distance) or distance <= 0 or np.any(depth <= 0):
        raise ValueError("Invalid fitted perspective camera depth")
    ndc = projected / (depth[:, None] * half_angle_tan)
    maximum_ndc = float(np.abs(ndc).max())
    if not math.isfinite(maximum_ndc) or maximum_ndc > 1 / 1.20 + 1e-12:
        raise ValueError("Perspective camera violates requested margin")
    camera = [(center - forward * distance).tolist(), center.tolist(), up.tolist()]
    if not np.isfinite(np.asarray(camera)).all():
        raise ValueError("Camera contains a nonfinite derived coordinate")
    return {
        "position": camera,
        "view_angle_degrees": view_angle,
        "distance_mm": distance,
        "projection": "perspective",
        "fit": "All source vertex depths/right/up coordinates; 20% extent margin",
        "projected_bounds_mm": [projected.min(axis=0).tolist(), projected.max(axis=0).tolist()],
        "maximum_absolute_ndc": maximum_ndc,
    }


def require_graphics_context(window: Any) -> str:
    """Refuse missing OpenGL support; return only three compact context facts."""
    try:
        if window.SupportsOpenGL() != 1:
            message = window.GetOpenGLSupportMessage()
            raise PhotoUnavailableError(
                "renderer_unavailable", "Product-photo rendering unavailable: " + str(message)[:400]
            )
        context = "\n".join(window.ReportCapabilities().splitlines()[:3])[:1024]
        if not context.strip():
            raise PhotoUnavailableError(
                "renderer_unavailable",
                "Product-photo rendering unavailable: graphics context is unknown",
            )
        return context
    except (AttributeError, OSError) as exc:
        raise PhotoUnavailableError(
            "renderer_unavailable",
            "Product-photo rendering unavailable: graphics context could not be checked",
        ) from exc


def _render(
    input_bytes: bytes, preset: str, *, view: str | None = None
) -> tuple[bytes, dict[str, Any]]:
    if view is not None and view != OPPOSED_VIEW:
        raise ValueError("Unsupported product-photo camera view")
    try:
        import trimesh
    except (ImportError, OSError) as exc:
        raise PhotoUnavailableError(
            "renderer_unavailable", "Optional triangle loader could not load"
        ) from exc
    source = trimesh.load(io.BytesIO(input_bytes), file_type="stl", force="mesh", process=False)
    if not isinstance(source, trimesh.Trimesh):
        raise ValueError("Product photo requires a triangle mesh")
    if view is None:
        return _render_triangles(source.triangles, preset)
    return _render_triangles(source.triangles, preset, view=view)


def _render_triangles(
    value: Any, preset: str, *, view: str | None = None
) -> tuple[bytes, dict[str, Any]]:
    if view is not None and view != OPPOSED_VIEW:
        raise ValueError("Unsupported product-photo camera view")
    # These optional imports occur only inside the isolated rendering worker.
    try:
        import numpy as np
        import pyvista as pv
        from PIL import Image
        from vtkmodules.vtkFiltersCore import vtkPolyDataTangents
    except (ImportError, OSError) as exc:
        raise PhotoUnavailableError(
            "renderer_unavailable",
            "Product-photo rendering unavailable: optional PyVista/VTK renderer could not load",
        ) from exc
    receipt: dict[str, Any] = {}

    receipt["versions"] = {
        key: importlib.metadata.version(key)
        for key in ("pyvista", "vtk", "numpy", "trimesh", "pillow")
    }
    triangles = validated_triangles(value)
    camera = (
        fitted_camera(triangles.reshape(-1, 3))
        if view is None
        else fitted_camera(triangles.reshape(-1, 3), view=view)
    )
    triangle_sha = hashlib.sha256(triangles.tobytes()).hexdigest()
    # Merge only EXACT coincident render vertices, then split shading normals at creases.
    points, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = np.column_stack((np.full(len(triangles), 3), inverse.reshape(-1, 3)))
    normal_mesh = pv.PolyData(points, faces).compute_normals(
        split_vertices=True,
        feature_angle=30,
        consistent_normals=False,
        auto_orient_normals=False,
        flip_normals=False,
        non_manifold_traversal=False,
    )
    indices = normal_mesh.faces.reshape(-1, 4)[:, 1:]
    if not np.array_equal(normal_mesh.points[indices], triangles):
        raise ValueError("Normal processing changed triangle coordinates or order")
    # Duplicate render corners for UV seams, without changing any triangle position.
    surface = pv.PolyData(
        triangles.reshape(-1, 3),
        np.column_stack((np.full(len(triangles), 3), np.arange(len(triangles) * 3).reshape(-1, 3))),
    )
    surface.point_data.active_normals = normal_mesh.point_data["Normals"][indices].reshape(-1, 3)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    if np.any(np.linalg.norm(cross, axis=1) == 0):
        raise ValueError("Degenerate input triangle")
    axes = np.argmax(np.abs(cross), axis=1)
    uv = np.empty((len(triangles), 3, 2))
    for axis in range(3):
        mask = axes == axis
        uv[mask] = triangles[mask][:, :, [i for i in range(3) if i != axis]] / 12.0
    surface.active_texture_coordinates = uv.reshape(-1, 2)
    tangents = vtkPolyDataTangents()
    tangents.SetInputData(surface)
    tangents.Update()
    surface = pv.wrap(tangents.GetOutput())
    if surface.GetPointData().GetTangents() is None:
        raise ValueError("Normal-map tangents unavailable")
    current = np.asarray(surface.points[surface.faces.reshape(-1, 4)[:, 1:]], dtype="<f8")
    if not np.array_equal(current, triangles):
        raise ValueError("Tangent processing changed triangles")

    rng = np.random.default_rng(8712)
    grain = rng.normal(size=(512, 512))
    for _ in range(6):
        grain = (
            4 * grain
            + np.roll(grain, 1, 0)
            + np.roll(grain, -1, 0)
            + np.roll(grain, 1, 1)
            + np.roll(grain, -1, 1)
        ) / 8
    grain = np.clip(grain / (3 * grain.std()), -1, 1)
    base = np.array([0.82, 0.84, 0.87] if preset == "metal" else [0.055, 0.24, 0.31])
    color = np.clip(base[None, None, :] * (1 + 0.035 * grain[:, :, None]), 0, 1)
    roughness = np.clip((0.30 if preset == "metal" else 0.62) + 0.065 * grain, 0, 1)
    orm = np.stack(
        (np.ones_like(grain), roughness, np.full_like(grain, 1 if preset == "metal" else 0)),
        axis=-1,
    )
    dx = np.roll(grain, -1, 1) - np.roll(grain, 1, 1)
    dy = np.roll(grain, -1, 0) - np.roll(grain, 1, 0)
    normal = np.stack((-0.23 * dx, -0.23 * dy, np.ones_like(grain)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    maps = {"color": color, "orm": orm, "normal": normal * 0.5 + 0.5}
    textures = {}
    # PyVista's native texture factory and decorated plotter methods lack complete stubs.
    texture_factory: Any = pv.Texture
    for name, values in maps.items():
        array = np.rint(values * 255).astype(np.uint8)
        texture = texture_factory(array)
        texture.interpolate = True
        texture.mipmap = True
        texture.repeat = True
        texture.SetUseSRGBColorSpace(name == "color")
        textures[name] = texture

    # Local equirectangular studio panels for broad PBR reflections, not downloaded HDR.
    v, u = np.mgrid[0:512, 0:1024] / np.array([512, 1024])[:, None, None]
    studio = np.full_like(u, 0.32)
    studio += 0.55 * np.clip((v - 0.5) * 4, 0, 1)
    for cx, cy, wx, wy in ((0.21, 0.23, 0.13, 0.15), (0.68, 0.28, 0.11, 0.18)):
        du = ((u - cx + 0.5) % 1) - 0.5
        studio += 0.68 * np.exp(-((du / wx) ** 4 + ((v - cy) / wy) ** 4))
    # Dark studio gaps provide contrast in reflected metal, not painted object marks.
    strip = ((u - 0.40 + 0.5) % 1) - 0.5
    studio -= 0.20 * np.exp(-((strip / 0.10) ** 4 + ((v - 0.30) / 0.20) ** 4))
    env_array = np.repeat(
        np.rint(np.clip(studio, 0, 1) * 255).astype(np.uint8)[:, :, None], 3, axis=2
    )
    environment = texture_factory(env_array)
    environment.interpolate = True
    environment.mipmap = True

    low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
    center = (low + high) / 2
    scale = float(np.linalg.norm(high - low))
    if not np.isfinite(center).all() or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Nonfinite or zero product-photo extent")
    plotter: Any = None
    try:
        plotter = pv.Plotter(off_screen=True, window_size=[1024, 1024], lighting="none")
        require_graphics_context(plotter.render_window)
        plotter.set_background("white")
        plotter.set_environment_texture(environment, is_srgb=True, show_background=False)
        plotter.renderer.SetEnvironmentUp(0, 0, 1)
        plotter.renderer.SetEnvironmentRight(1, 0, 0)
        actor = plotter.add_mesh(
            surface, color="white", pbr=True, metallic=1.0, roughness=1.0, smooth_shading=False
        )
        prop = actor.GetProperty()
        prop.SetBaseColorTexture(textures["color"])
        prop.SetORMTexture(textures["orm"])
        prop.SetNormalTexture(textures["normal"])
        prop.SetNormalScale(1.0)
        floor = pv.Plane(
            center=(float(center[0]), float(center[1]), float(low[2])),
            direction=(0, 0, 1),
            i_size=scale * 5,
            j_size=scale * 5,
        )
        plotter.add_mesh(floor, color="white", ambient=0.97, diffuse=0.15, specular=0.0)
        lights = []
        # Nine directional samples approximate a broad light's cast-shadow penumbra.
        for x in (-0.22, 0.0, 0.22):
            for y in (-0.22, 0.0, 0.22):
                position = center + scale * np.array([-1.2 + x, 1.5 + y, 2.3])
                values = {
                    "position": position.tolist(),
                    "focal_point": center.tolist(),
                    "intensity": 0.05,
                    "positional": False,
                }
                plotter.add_light(pv.Light(**values))
                lights.append(values)
        fill = {
            "position": (center + scale * np.array([1.6, -1.0, 1.5])).tolist(),
            "focal_point": center.tolist(),
            "intensity": 0.12,
            "positional": False,
        }
        plotter.add_light(pv.Light(**fill))
        lights.append(fill)
        plotter.enable_shadows()
        plotter.enable_ssao(radius=scale * 0.045, bias=scale * 0.003, kernel_size=128, blur=True)
        plotter.enable_anti_aliasing("ssaa")
        plotter.camera_position = camera["position"]
        plotter.camera.parallel_projection = False
        plotter.camera.view_angle = camera["view_angle_degrees"]
        plotter.reset_camera_clipping_range()
        receipt.update(
            camera=camera,
            lights=lights,
            texture={
                "tile_mm": 12,
                "seed": 8712,
                "map_pixels": 512,
                "normal_gradient_scale": 0.23,
                "roughness_variation": 0.065,
                "scope": "Color/roughness/shading normal only; no displacement",
            },
            ssao={
                "radius_mm": scale * 0.045,
                "bias_mm": scale * 0.003,
                "kernel_size": 128,
                "blur": True,
            },
            shadow_method="Native shadow maps with nine directional samples; restrained blurred SSAO; visual softness unqualified",
            geometry={
                "triangles": len(triangles),
                "bounds_mm": [low.tolist(), high.tolist()],
                "triangle_sha256": triangle_sha,
                "triangle_positions_unchanged": True,
                "display_normal_crease_degrees": 30,
            },
        )
        image = plotter.screenshot(return_img=True)
        if image is None or image.shape[:2] != (1024, 1024):
            raise ValueError("Unexpected output dimensions")
        receipt["graphics_context"] = require_graphics_context(plotter.render_window)
        receipt["image_dimensions"] = [1024, 1024]
    except (PhotoUnavailableError, MemoryError, TimeoutError):
        raise
    except (RuntimeError, OSError) as exc:
        raise PhotoUnavailableError(
            "render_failed", "Product-photo render failed: " + str(exc)[:400]
        ) from exc
    finally:
        if plotter is not None:
            plotter.close()
    if not np.array_equal(
        np.asarray(surface.points[surface.faces.reshape(-1, 4)[:, 1:]], dtype="<f8"), triangles
    ):
        raise ValueError("Render surface changed during rendering")
    encoded = io.BytesIO()
    Image.fromarray(image).save(encoded, format="PNG")
    return encoded.getvalue(), receipt


def render_product_photo(request_path: Path) -> dict[str, Any]:
    """Render an immutable staged mesh and publish only the image and receipt."""
    raw_request = read_bytes(request_path, MAX_RECEIPT_BYTES)
    value = json.loads(raw_request, object_pairs_hook=_object, parse_constant=_nonfinite)
    if not isinstance(value, dict):
        raise ValueError("Expected a product-photo request object")
    request = validated_request(value)
    base_request = legacy_request(request)
    view = request.get("view")
    directory = request_path.parent
    assembly = set(base_request) == ASSEMBLY_REQUEST_KEYS
    mesh_path = directory / ("topology.json" if assembly else "part.stl")
    source_hash = request["source"]["topology_sha256"] if assembly else request["mesh_sha256"]
    image_path = directory / "photo.png"
    receipt_path = directory / "receipt.json"
    if any(p.exists() or p.is_symlink() for p in (image_path, receipt_path)):
        raise ValueError("Product-photo output already exists")
    mesh_bytes = read_bytes(mesh_path, MAX_INPUT_BYTES)
    if hashlib.sha256(mesh_bytes).hexdigest() != source_hash:
        raise ValueError("Staged product-photo mesh does not match its source hash")
    if not assembly and len(mesh_bytes) >= 84:
        count = struct.unpack_from("<I", mesh_bytes, 80)[0]
        if len(mesh_bytes) == 84 + 50 * count and not 0 < count <= MAX_TRIANGLES:
            raise ValueError("Invalid or excessive binary STL triangle count")
    if assembly:
        from forge_mind.workspace.occurrence_render import (
            TopologyMeshLimitError,
            selected_topology_meshes,
        )

        topology = json.loads(mesh_bytes, object_pairs_hook=_object, parse_constant=_nonfinite)
        if not isinstance(topology, dict):
            raise ValueError("Expected saved topology object")
        counts = topology.get("topology_counts")
        if (
            not isinstance(counts, dict)
            or type(counts.get("bodies")) is not int
            or counts["bodies"] != request["selection"]["total_bodies"]
        ):
            raise ValueError("Saved topology body count differs from bound selection")
        tessellation = topology.get("pick_tessellation")
        if not isinstance(tessellation, dict):
            raise ValueError("Missing saved display tessellation metadata")
        for key in ("relative_linear_deflection", "angular_deflection_radians"):
            value = tessellation.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid saved display tessellation deflection")
        purpose = tessellation.get("purpose")
        if not isinstance(purpose, str) or not 0 < len(purpose) <= 512:
            raise ValueError("Invalid saved display tessellation purpose")
        try:
            mesh, count, discarded = selected_topology_meshes(
                topology, [row["body_index"] for row in request["selection"]["selected"]]
            )
        except TopologyMeshLimitError as exc:
            raise PhotoUnavailableError(
                "render_limit", "Saved display tessellation exceeds product-photo limits"
            ) from exc
        if count != request["selection"]["total_bodies"]:
            raise ValueError("Saved topology body count differs from bound selection")
        image, rendering = (
            _render_triangles(mesh.triangles, request["preset"])
            if view is None
            else _render_triangles(mesh.triangles, request["preset"], view=view)
        )
        rendering["source_tessellation"] = {
            "purpose": purpose,
            "relative_linear_deflection": tessellation["relative_linear_deflection"],
            "angular_deflection_radians": tessellation["angular_deflection_radians"],
            "selected_body_count": len(request["selection"]["selected"]),
            "discarded_zero_area_triangles": discarded,
            "vertex_precision": DISPLAY_VERTEX_PRECISION,
        }
    else:
        image, rendering = (
            _render(mesh_bytes, request["preset"])
            if view is None
            else _render(mesh_bytes, request["preset"], view=view)
        )
    if not 0 < len(image) <= MAX_IMAGE_BYTES or not image.startswith(PNG_SIGNATURE):
        raise ValueError("Product-photo PNG is missing, invalid or oversized")
    if read_bytes(request_path, MAX_RECEIPT_BYTES) != raw_request:
        raise ValueError("Product-photo request changed during rendering")
    if hashlib.sha256(read_bytes(mesh_path, MAX_INPUT_BYTES)).hexdigest() != source_hash:
        raise ValueError("Product-photo source changed during rendering")
    receipt = validate_receipt(
        {
            **request,
            "scope": ASSEMBLY_SCOPE
            if assembly
            else CAPTURED_SCOPE
            if set(base_request) == CAPTURED_REQUEST_KEYS
            else SCOPE,
            "image": {
                "file": "photo.png",
                "sha256": hashlib.sha256(image).hexdigest(),
                "width": 1024,
                "height": 1024,
            },
            "rendering": rendering,
        },
        request,
    )
    data = json.dumps(receipt, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(data) > MAX_RECEIPT_BYTES:
        raise ValueError("Product-photo receipt exceeds its byte budget")
    with image_path.open("xb") as output:
        output.write(image)
    with receipt_path.open("xb") as output:
        output.write(data)
    return receipt


if __name__ == "__main__":
    path = Path(sys.argv[1])
    try:
        render_product_photo(path)
    except PhotoUnavailableError as exc:
        # Typed optional failures cross the process boundary as bound data, not log text.
        request = validated_request(
            json.loads(
                read_bytes(path, MAX_RECEIPT_BYTES),
                object_pairs_hook=_object,
                parse_constant=_nonfinite,
            )
        )
        data = {"request": request, "code": exc.code, "reason": exc.reason}
        with (path.parent / "unavailable.json").open("xb") as output:
            output.write(json.dumps(data, allow_nan=False).encode("utf-8"))
