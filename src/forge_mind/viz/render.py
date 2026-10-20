"""Render an STL to a shaded multi-view PNG, headless.

Six panels — isometric, front, right, rear, top (hole layout), bottom — exactly the views the
critic/judge prompts describe. Faces are shaded by a simple Lambert term so depth reads clearly
without a GPU. Captions (the exact prompt, per the standing convention) render in a dedicated
band BELOW the panels that the figure grows to fit — overlap is geometrically impossible.
"""

from __future__ import annotations

import threading
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display required

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from forge_mind.viz.raster import rasterize_orthographic

# (label, elevation, azimuth) for each panel — SIX views, exactly what the critic/judge prompts
# describe ("isometric, front, right, rear, top, bottom"). The golden-run autopsy (TEST_LOG row
# 37 follow-up) caught the old 3-view render contradicting both prompts: the judge reviewed
# "all six panels" of a three-panel image and hallucinated the rest.
_VIEWS = [
    ("isometric", 26, -58),
    ("front", 2, -90),
    ("right", 2, 0),
    ("rear", 2, 90),
    ("top (hole layout)", 89, -90),
    ("bottom", -89, 90),
]

# Caption layout: ~chars per wrapped line at fontsize 10 on a 15-inch figure, and the vertical
# inches one wrapped line consumes. Conservative on purpose — the band must always be tall
# enough, because matplotlib will happily overdraw axes with fig.text (the row-37 occlusion:
# a 20-line problem-mode prompt flooded the hole-layout panel and blinded the judge).
_CAPTION_CHARS_PER_LINE = 105
_CAPTION_LINE_IN = 0.19
_BASE_COLOR = np.array([0.33, 0.56, 0.86])  # steel blue
_LIGHT = np.array([0.35, -0.5, 0.78])
_AMBIENT = 0.25  # floor so back faces aren't pure black


def _shaded_facecolors(
    face_normals: np.ndarray, base_colors: np.ndarray | None = None
) -> np.ndarray:
    """TWO-SIDED Lambert (|n.l|) + a fill light. Boolean results and STL exports carry
    patches of flipped triangle winding; a one-sided dot rendered those faces BLACK (the
    founder's 'buggy game prop' scatter). Lighting by the face PLANE instead of its
    winding is exact for opaque display shading, and the fill light lifts the faces that
    are edge-on to the key light."""
    key = _LIGHT / np.linalg.norm(_LIGHT)
    fill = np.array([-0.5, 0.6, 0.35])
    fill = fill / np.linalg.norm(fill)
    i_key = np.abs(face_normals @ key)
    i_fill = np.abs(face_normals @ fill)
    intensity = _AMBIENT + (1.0 - _AMBIENT) * np.clip(0.75 * i_key + 0.35 * i_fill, 0.0, 1.0)
    palette = _BASE_COLOR[None, :] if base_colors is None else np.asarray(base_colors, dtype=float)
    if palette.shape not in ((1, 3), (len(face_normals), 3)) or not np.isfinite(palette).all():
        raise ValueError("Preview requires finite RGB colors for every triangle")
    if (palette < 0).any() or (palette > 1).any():
        raise ValueError("Preview RGB colors must be between zero and one")
    colors = palette * intensity[:, None]
    clipped: np.ndarray = np.clip(colors, 0.0, 1.0)
    return clipped


def _wrap_caption(caption: str) -> str:
    """Wrap the caption OURSELVES (deterministic line count) so the figure can reserve exactly
    enough vertical room — matplotlib's wrap=True gives no line count and overdraws axes."""
    lines: list[str] = []
    for raw in caption.splitlines() or [""]:
        wrapped = textwrap.wrap(raw, width=_CAPTION_CHARS_PER_LINE) or [""]
        lines.extend(wrapped)
    return "\n".join(lines)


# pyplot state is process-global and NOT thread-safe; parallel part authoring
# (FORGE_PARALLEL_AUTHORS) renders from worker threads. One render at a time --
# a single-threaded caller pays one uncontended acquire.
_RENDER_LOCK = threading.Lock()


def render_part_to_png(
    stl_path: Path, png_path: Path, title: str | None = None, caption: str | None = None
) -> Path:
    with _RENDER_LOCK:
        return _render_part_to_png_unlocked(stl_path, png_path, title=title, caption=caption)


def _render_part_to_png_unlocked(
    stl_path: Path, png_path: Path, title: str | None = None, caption: str | None = None
) -> Path:
    """Render ``stl_path`` to ``png_path`` (SIX shaded views). Returns png_path.

    ``caption`` (e.g. the exact prompt/recipe that produced the part) is printed BELOW the views
    in a band the figure GROWS to fit — the caption can never overlap the geometry panels, no
    matter how long the prompt is (the row-37 occlusion fix). None -> no caption.

    Instrument rule: renders fed to MODEL eyes (critic / acceptance judge) must pass
    ``caption=None`` — the model judges geometry, and the brief already reaches it as text;
    prompt-as-pixels only occludes. The captioned render is the HUMAN artifact.
    """
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"could not load a single mesh from {stl_path}")

    triangles = mesh.triangles  # (n, 3, 3)
    facecolors = _shaded_facecolors(mesh.face_normals)

    extents = mesh.extents  # (3,)
    center = mesh.bounds.mean(axis=0)
    radius = float(np.max(extents)) / 2.0 * 1.1

    panels_h = 10.0  # two rows of three views
    wrapped = _wrap_caption(caption) if caption else None
    caption_h = (wrapped.count("\n") + 1) * _CAPTION_LINE_IN + 0.35 if wrapped is not None else 0.0
    fig_h = panels_h + caption_h
    fig = plt.figure(figsize=(15, fig_h))
    band = caption_h / fig_h  # bottom fraction reserved for the caption; panels live above it

    for i, (label, elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        coll = Poly3DCollection(triangles, facecolors=facecolors, linewidths=0.0)
        coll.set_edgecolor("none")
        ax.add_collection3d(coll)

        # Equal aspect: a cube around the part center.
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=11)

    dims = f"{extents[0]:.0f} x {extents[1]:.0f} x {extents[2]:.0f} mm"
    vol = f"{mesh.volume:.0f} mm^3"
    suptitle = title or stl_path.stem
    fig.suptitle(f"{suptitle}    [{dims}, {vol}]", fontsize=13)
    if wrapped is not None:
        # Panels are laid out strictly ABOVE the caption band; the pre-wrapped text is drawn
        # top-down INSIDE the band, so overlap is geometrically impossible.
        fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=band + 0.02)
        fig.text(0.5, band - 0.01, wrapped, ha="center", va="top", fontsize=10)
    else:
        fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), dpi=120, facecolor="white")
    plt.close(fig)
    return png_path


def _orthographic_basis(elevation: float, azimuth: float) -> np.ndarray:
    """Right/up/toward-viewer axes, matching the existing CAD view directions."""
    el, az = np.radians([elevation, azimuth])
    return np.column_stack(
        (
            [-np.sin(az), np.cos(az), 0],
            [-np.sin(el) * np.cos(az), -np.sin(el) * np.sin(az), np.cos(el)],
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)],
        )
    )


def render_workspace_preview(
    stl_path: Path,
    png_path: Path,
    *,
    frame_bounds: np.ndarray | None = None,
    inspection_label: str | None = None,
    frame_scope: str = "complete-view frame",
    display_mesh: trimesh.Trimesh | None = None,
    face_colors: np.ndarray | None = None,
    legend: tuple[tuple[str, str], ...] = (),
    appearance_note: str | None = None,
) -> Path:
    """Six opaque orthographic views with per-pixel nearest-surface depth.

    Every preview triangle contributes to the depth test. The raster remains a
    display instrument, not geometric acceptance. Other renderers retain their
    existing perspective and caption behavior.
    """
    with _RENDER_LOCK:
        mesh = trimesh.load(str(stl_path), force="mesh") if display_mesh is None else display_mesh
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError("Workspace preview requires a nonempty triangle mesh")
        bounds = mesh.bounds if frame_bounds is None else np.asarray(frame_bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or (bounds[1] < bounds[0]).any():
            raise ValueError("Workspace preview requires finite ordered frame bounds")
        tolerance = max(float(np.linalg.norm(bounds[1] - bounds[0])) * 1e-6, 1e-6)
        if (mesh.bounds[0] < bounds[0] - tolerance).any() or (
            mesh.bounds[1] > bounds[1] + tolerance
        ).any():
            raise ValueError("Inspection mesh lies outside its complete-view frame")
        center = bounds.mean(axis=0)
        triangles = mesh.triangles - center
        colors = _shaded_facecolors(mesh.face_normals, face_colors)
        # One rotation-safe scale for all six panels, including diagonal views.
        radius = max(float(np.linalg.norm(bounds[1] - bounds[0])) / 2 * 1.1, 1e-6)
        if len(legend) > 8 or any(
            len(label.splitlines()) > 2 or any(len(line) > 32 for line in label.splitlines())
            for _, label in legend
        ):
            raise ValueError("Preview legend exceeds its readable bound")
        if appearance_note is not None and len(appearance_note) > 240:
            raise ValueError("Preview note exceeds its readable bound")
        annotated = bool(legend or appearance_note)
        # Two 12-point lines per entry, four rows at most, plus a separate note.
        # Grow the figure instead of reducing the text or covering geometry.
        legend_band = 0.8 + 0.55 * ((len(legend) + 1) // 2) if annotated else 0.0
        figure_height = 10 + legend_band
        fig, axes = plt.subplots(2, 3, figsize=(15, figure_height))
        try:
            for ax, (label, elevation, azimuth) in zip(axes.flat, _VIEWS, strict=True):
                projected = triangles @ _orthographic_basis(elevation, azimuth)
                ax.imshow(
                    rasterize_orthographic(projected, colors, radius),
                    origin="lower",
                    extent=(-radius, radius, -radius, radius),
                    interpolation="nearest",
                )
                ax.set_xlim(-radius, radius)
                ax.set_ylim(-radius, radius)
                ax.set_aspect("equal")
                ax.set_axis_off()
                ax.set_title(label, fontsize=11)
            dims = " x ".join(f"{value:.0f}" for value in bounds[1] - bounds[0])
            heading = inspection_label or "CAD revision · orthographic preview"
            dimension_scope = f"{frame_scope}: " if inspection_label else ""
            fig.suptitle(f"{heading}    [{dimension_scope}{dims} mm]", fontsize=13)
            if legend:
                key = fig.legend(
                    handles=[Patch(facecolor=color, label=label) for color, label in legend],
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.7 / figure_height),
                    ncol=2,
                    fontsize=12,
                    frameon=False,
                )
                for text in key.get_texts():
                    text.set_parse_math(False)
            if appearance_note:
                fig.text(
                    0.5,
                    0.12 / figure_height,
                    textwrap.fill(appearance_note, width=90),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    parse_math=False,
                )
            fig.subplots_adjust(
                left=0.02,
                right=0.98,
                top=0.94,
                bottom=(legend_band + 0.2) / figure_height if annotated else 0.02,
            )
            png_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(png_path), dpi=120, facecolor="white")
        finally:
            plt.close(fig)
    return png_path


__all__ = ["render_part_to_png", "render_workspace_preview"]
