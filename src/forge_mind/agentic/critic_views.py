"""Richer critic input: a six-panel sheet (4 shaded views + 2 SECTION cuts) so the
vision critic can see internal features and per-axis scale (lever d, ADR-0045).

The single 3-view render hides exactly what long-tail parts get wrong: internal
bores/pockets (invisible from outside), wrong per-axis proportions (no scale
reference), and back-side features. The section cuts slice the mesh at its bbox
center on two planes and draw the cut outlines -- a bore, cavity, or wall reads
directly. Each panel is labeled with its measured extents in mm.

OPT-IN via ``FORGE_CRITIC_VIEWS=1``: only the CRITIC's image changes; the iteration
log render and the acceptance judge's input stay byte-identical (flag off = nothing
changes anywhere). Failure here can never break the loop -- the caller falls back to
the standard render.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import trimesh

from forge_mind.viz.render import _shaded_facecolors


def critic_views_enabled() -> bool:
    return os.environ.get("FORGE_CRITIC_VIEWS") == "1"


_VIEWS = (  # label, elev, azim -- the four shaded orientations
    ("isometric", 30, -60),
    ("front", 0, -90),
    ("top", 90, -90),
    ("side", 0, 0),
)
_SECTIONS = (  # label, plane normal (the cut at bbox center)
    ("section X (YZ cut)", (1.0, 0.0, 0.0)),
    ("section Z (XY cut)", (0.0, 0.0, 1.0)),
)


def render_critic_sheet(stl_path: Path, out_png: Path, title: str) -> Path:
    """Write the six-panel critic sheet for ``stl_path`` and return ``out_png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"could not load a single mesh from {stl_path}")
    extents = mesh.extents
    center = mesh.bounds.mean(axis=0)
    radius = float(np.max(extents)) / 2.0 * 1.15
    facecolors = _shaded_facecolors(mesh.face_normals)

    fig = plt.figure(figsize=(16.5, 9.0))
    for i, (label, elev, azim) in enumerate(_VIEWS):
        ax = fig.add_subplot(2, 3, i + 1 if i < 2 else i + 2, projection="3d")
        coll = Poly3DCollection(mesh.triangles, facecolors=facecolors, linewidths=0.0)
        coll.set_edgecolor("none")
        ax.add_collection3d(coll)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(
            f"{label}  [{extents[0]:.0f} x {extents[1]:.0f} x {extents[2]:.0f} mm]", fontsize=10
        )

    # Section cuts: slice at the bbox center; draw every cut segment in the plane's
    # own 2D so internal voids (bores, pockets, walls) read as closed outlines.
    for s_ix, (label, normal) in enumerate(_SECTIONS):
        ax2 = fig.add_subplot(2, 3, 3 if s_ix == 0 else 6)
        n = np.asarray(normal)
        seg = trimesh.intersections.mesh_plane(  # type: ignore[no-untyped-call]
            mesh, plane_normal=n, plane_origin=center
        )
        if len(seg):
            # Project the 3D segments onto the cut plane's 2D basis.
            u = np.array([0.0, 1.0, 0.0]) if normal[0] else np.array([1.0, 0.0, 0.0])
            v = np.cross(n, u)
            u = np.cross(v, n)
            pts = seg.reshape(-1, 3) - center
            uv = np.column_stack([pts @ u, pts @ v]).reshape(-1, 2, 2)
            for a, b in uv:
                ax2.plot([a[0], b[0]], [a[1], b[1]], color="#1f3b57", linewidth=1.0)
            ax2.set_aspect("equal")
        else:
            ax2.text(0.5, 0.5, "no material on this cut", ha="center", va="center")
        ax2.set_title(f"{label} at center -- internal features visible", fontsize=10)
        ax2.grid(True, linewidth=0.3, alpha=0.5)
        ax2.tick_params(labelsize=7)
        ax2.set_xlabel("mm", fontsize=7)

    fig.suptitle(
        f"{title}    overall {extents[0]:.0f} x {extents[1]:.0f} x {extents[2]:.0f} mm "
        f"(grid + axis ticks are millimeters)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=110, facecolor="white")
    plt.close(fig)
    return out_png


__all__ = ["critic_views_enabled", "render_critic_sheet"]
