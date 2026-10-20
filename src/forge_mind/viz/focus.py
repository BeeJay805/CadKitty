"""Opaque saved-body context/detail display; markers are bounds, not face outlines."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Rectangle

from forge_mind.viz.raster import rasterize_orthographic
from forge_mind.viz.render import _RENDER_LOCK, _shaded_facecolors
from forge_mind.workspace.occurrence_focus import focus_view_metadata, validate_focus_frame

FOCUS_PANEL_PIXELS = 768
MAX_FOCUS_IMAGE_BYTES = 1024 * 1024


def _projected_bounds(bounds: dict[str, Any], center: np.ndarray, basis: np.ndarray) -> np.ndarray:
    corners = np.asarray(list(product(*zip(bounds["min"], bounds["max"], strict=True))))
    projected = (corners - center) @ basis
    return np.stack((projected[:, :2].min(axis=0), projected[:, :2].max(axis=0)))


def render_occurrence_focus(mesh: Any, png_path: Path, frame: dict[str, Any]) -> dict[str, Any]:
    """Render every selected-body triangle twice, with XY viewport clipping only."""
    frame = validate_focus_frame(frame)
    metadata = focus_view_metadata(frame)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Occurrence focus requires a nonempty full body mesh")
    triangles = np.asarray(mesh.triangles, dtype=float)
    if not np.isfinite(triangles).all():
        raise ValueError("Occurrence focus requires finite triangle positions")
    body = frame["body_bounds"]
    lower, upper = np.asarray(body["min"]), np.asarray(body["max"])
    tolerance = max(float(np.linalg.norm(upper - lower)) * 1e-6, 1e-6)
    if (mesh.bounds[0] < lower - tolerance).any() or (mesh.bounds[1] > upper + tolerance).any():
        raise ValueError("Focus mesh lies outside its saved body frame")
    if png_path.exists() or png_path.is_symlink():
        raise ValueError("Occurrence focus output already exists")
    basis = np.asarray(metadata["basis"], dtype=float)
    colors = _shaded_facecolors(mesh.face_normals)
    centers = [np.asarray(metadata[key]) for key in ("context_center_mm", "detail_center_mm")]
    radii = [metadata[key] for key in ("context_radius_mm", "detail_radius_mm")]
    with _RENDER_LOCK:
        fig, axes = plt.subplots(1, 2, figsize=(14, 8.4))
        try:
            for ax, center, radius, title in zip(
                axes,
                centers,
                radii,
                ("Complete selected body", "Local detail · viewport crop"),
                strict=True,
            ):
                # Do not select triangles by the region box or discard hidden geometry.
                pixels = rasterize_orthographic(
                    (triangles - center) @ basis, colors, radius, size=FOCUS_PANEL_PIXELS
                )
                ax.imshow(
                    pixels,
                    origin="lower",
                    extent=(-radius, radius, -radius, radius),
                    interpolation="nearest",
                )
                marked = _projected_bounds(frame["entity_bounds"], center, basis)
                ax.add_patch(
                    Rectangle(
                        marked[0],
                        *(marked[1] - marked[0]),
                        fill=False,
                        edgecolor="#B04A00",
                        linewidth=1.6,
                        linestyle="--",
                    )
                )
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
                ax.set_aspect("equal")
                ax.set_axis_off()
                ax.set_title(title, fontsize=14)
            offset = (centers[1] - centers[0]) @ basis
            window_radius = radii[1]
            axes[0].add_patch(
                Rectangle(
                    offset[:2] - window_radius,
                    2 * window_radius,
                    2 * window_radius,
                    fill=False,
                    edgecolor="#6B49A5",
                    linewidth=1.5,
                )
            )
            fig.suptitle(
                f"Occurrence focus · {metadata['view']} · same camera in both panels", fontsize=15
            )
            fig.text(
                0.5,
                0.12,
                "Orange dashes: projected entity bounds, not an exact outline. Purple: detail window.",
                ha="center",
                fontsize=12,
                parse_math=False,
            )
            fig.text(
                0.5,
                0.075,
                "Isolated occurrence; other bodies omitted. Hidden surfaces stay hidden.",
                ha="center",
                fontsize=12,
                parse_math=False,
            )
            fig.text(
                0.5,
                0.035,
                "Saved display mesh · world CAD Z-up mm · not a measurement or physical-fit check",
                ha="center",
                fontsize=12,
                parse_math=False,
            )
            fig.subplots_adjust(left=0.025, right=0.975, top=0.9, bottom=0.19, wspace=0.07)
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png_path, dpi=120, facecolor="white")
        finally:
            plt.close(fig)
    if not 0 < png_path.stat().st_size <= MAX_FOCUS_IMAGE_BYTES:
        raise ValueError("Occurrence focus PNG exceeded its output bound")
    return metadata
