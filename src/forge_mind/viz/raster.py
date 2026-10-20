"""Opaque orthographic triangle rasterization using the existing NumPy runtime."""

from __future__ import annotations

import math

import numpy as np

WORKSPACE_PREVIEW_PIXELS = 512
MAX_RASTER_PIXELS = 2048
_TRIANGLE_BATCH = 4096


def rasterize_orthographic(
    triangles: np.ndarray, colors: np.ndarray, radius: float, size: int = WORKSPACE_PREVIEW_PIXELS
) -> np.ndarray:
    """Return RGB pixels, with positive Z toward the viewer and row zero at bottom.

    Every triangle is evaluated. No mesh sampling or winding culling occurs.
    Zero projected area covers no pixels. Depth ties retain the first face.
    Depth/RGB buffers depend on panel resolution. Projected geometry and colors
    use linear face-count storage. Temporary barycentric arrays cover only one
    clipped triangle bounding box at a time.
    The resolution bound is a raster memory limit, not a geometry size limit.
    """
    triangles = np.asarray(triangles, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not len(triangles):
        raise ValueError("Orthographic raster requires nonempty (faces, 3, 3) triangles")
    if colors.shape != (len(triangles), 3):
        raise ValueError("Orthographic raster requires one RGB color per triangle")
    if not np.isfinite(triangles).all() or not np.isfinite(colors).all():
        raise ValueError("Orthographic raster requires finite vertices and colors")
    if (colors < 0).any() or (colors > 1).any():
        raise ValueError("Orthographic raster RGB colors must be between zero and one")
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Orthographic raster requires a finite positive frame radius")
    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_RASTER_PIXELS:
        raise ValueError(f"Orthographic raster resolution must be 1 to {MAX_RASTER_PIXELS} pixels")

    screen = (triangles[:, :, :2] / radius + 1) * (size / 2)
    if not np.isfinite(screen).all():
        raise ValueError("Orthographic raster projection must remain finite")
    rgb = np.full((size, size, 3), 255, dtype=np.uint8)
    depth = np.full((size, size), -np.inf)
    palette = np.rint(colors * 255).astype(np.uint8)
    for start in range(0, len(screen), _TRIANGLE_BATCH):
        batch = screen[start : start + _TRIANGLE_BATCH]
        a, b, c = batch[:, 0], batch[:, 1], batch[:, 2]
        areas = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (
            a[:, 1] - c[:, 1]
        )
        if not np.isfinite(areas).all():
            raise ValueError("Orthographic raster projected area must remain finite")
        lower = np.clip(np.ceil(batch.min(axis=1) - 0.5), 0, size).astype(int)
        upper = np.clip(np.floor(batch.max(axis=1) - 0.5), -1, size - 1).astype(int)
        # These projected faces have no pixel-center bounding region. This is
        # exact screen coverage exclusion, not mesh decimation or a face limit.
        candidates = np.flatnonzero((areas != 0) & (upper >= lower).all(axis=1))
        for local in candidates:
            i = start + local
            a, b, c = screen[i]
            x0, y0 = lower[local]
            x1, y1 = upper[local] + 1
            xx = np.arange(x0, x1)[None, :] + 0.5
            yy = np.arange(y0, y1)[:, None] + 0.5
            u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / areas[local]
            v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / areas[local]
            w = 1 - u - v
            inside = (u >= -1e-10) & (v >= -1e-10) & (w >= -1e-10)
            z = u * triangles[i, 0, 2] + v * triangles[i, 1, 2] + w * triangles[i, 2, 2]
            patch = depth[y0:y1, x0:x1]
            visible = inside & (z > patch)
            patch[visible] = z[visible]
            rgb[y0:y1, x0:x1][visible] = palette[i]
    return rgb
