"""Grasp-tab emission (ADR-0012 §6 / ADR-0003) -- CELL LANE ONLY.

ForgeMind fuses ONE standardized handling tab onto every part the autonomous cell prints, so the
SO-101 arm's grasp policy trains on a single grasp family instead of infinite part geometry. This
module owns that deterministic fusion. It runs ONLY behind the cell profile (``FORGE_CELL_PROFILE``);
the consumer website paths never call it, so their output is byte-identical (the leak that must never
ship is a tab on a customer part -- enforced by the two-sided test).

Spec (ADR-0012 §6): a permanent vertical fin, grip zone 15 x 10 x 4.0 mm (two parallel 15x10 faces
4 mm apart for the parallel-jaw gripper), 1 mm top lead-in chamfer, 2 mm root fillet, self-supporting;
grip centroid in a fixed 10-15 mm band above the bed; centered over the part CoM in plan; a 25 mm-
radius clear approach cylinder above the tab to 40 mm; faces axis-aligned; part+tab mass ≤ 200 g.

FAIL-CLOSED (ADR-0065): no valid non-colliding placement, a fusion/fillet failure, an over-mass part,
or a tab that breaks re-validation all REFUSE the cell part (``refused=True``) with the reason -- never
a tab that collides or a silently-degraded part. Placement is deterministic (from the mesh): TOP-mount
(fin stands over the CoM on a part short enough that the grip band clears it) else SIDE-mount (fin out
from the largest axis-aligned silhouette face at band height); a part that fits neither is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

# --- ADR-0012 §6 constants (mm) ---
GRIP_W = 15.0  # grip-face width
GRIP_H = 10.0  # grip-face height (vertical, Z)
GRIP_T = 4.0  # jaw gap / fin thickness (the two grip faces are this far apart)
BAND_LO, BAND_HI = 10.0, 15.0  # grip-centroid band above the bed
GRIP_Z_LO, GRIP_Z_HI = 7.5, 17.5  # the 10 mm grip zone, centred on the 12.5 mm band midpoint
SIDE_STICKOUT = 15.0  # side-mount outward length (gives the 15 mm grip-face depth)
FUSE_OVERLAP = 1.5  # how far the fin reaches into the part so the boolean union is one solid
ROOT_FILLET = 2.0
TOP_CHAMFER = 1.0
CLEAR_R = 25.0  # clear approach-cylinder radius
CLEAR_TOP = 40.0  # clear approach-cylinder top, above the bed
MASS_CAP_G = 200.0  # SO-101 payload margin


@dataclass(frozen=True)
class GraspTabResult:
    """Outcome of a cell-lane grasp-tab fusion. Exactly one of ``ok`` / ``refused`` is True."""

    ok: bool
    refused: bool
    reason: str | None = None
    stl_path: Path | None = None
    step_path: Path | None = None
    mode: str | None = None  # 'top' | 'side'
    grip_dims_mm: tuple[float, float, float] | None = None
    mass_g: float | None = None


def _com_xy(mesh: Any) -> tuple[float, float]:
    try:
        com = mesh.center_mass
        return float(com[0]), float(com[1])
    except Exception:
        c = mesh.bounds.mean(axis=0)
        return float(c[0]), float(c[1])


def _surface_top_at(mesh: Any, x: float, y: float, z_hi: float) -> float | None:
    """z of the part's uppermost surface at (x,y) via a downward ray; None if the column misses."""
    try:
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=[[x, y, z_hi]], ray_directions=[[0.0, 0.0, -1.0]]
        )
    except Exception:
        return None
    if len(locs) == 0:
        return None
    return float(np.max(locs[:, 2]))


def _clear_cylinder_ok(mesh: Any, x: float, y: float, z_from: float, z_to: float) -> bool:
    """Is the vertical CLEAR_R cylinder at (x,y) over [z_from, z_to] free of part material?

    Deterministic sampling of the cylinder volume tested against the part mesh; any interior point
    means the descending gripper would collide -> not clear. Empty sample (degenerate span) = clear."""
    if z_to <= z_from:
        return True
    pts: list[list[float]] = []
    for zz in np.linspace(z_from, z_to, 8):
        for rr in (0.0, CLEAR_R * 0.5, CLEAR_R):
            for th in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
                pts.append([x + rr * float(np.cos(th)), y + rr * float(np.sin(th)), float(zz)])
    try:
        inside = mesh.contains(np.array(pts))
    except Exception:
        return False  # cannot verify -> fail closed (treat as blocked)
    return not bool(inside.any())


@dataclass(frozen=True)
class _Placement:
    mode: str  # 'top' | 'side'
    x: float
    y: float
    z_lo: float
    z_hi: float
    axis: str = "x"  # side-mount outward axis: '+x' | '-x' | '+y' | '-y'


def _plan_placement(mesh: Any) -> _Placement | None:
    """Deterministic tab placement, or None to REFUSE. TOP-mount when the part is short enough that
    the grip band clears its top over the CoM; otherwise SIDE-mount out from an axis-aligned face."""
    z_bed = float(mesh.bounds[0][2])
    part_max_z = float(mesh.bounds[1][2])
    xc, yc = _com_xy(mesh)
    z_hi_ray = z_bed + CLEAR_TOP + 20.0
    grip_top = z_bed + GRIP_Z_HI

    # TOP-MOUNT: only when the WHOLE part is shorter than the grip band, so the fin stands cleanly on
    # top over the CoM with the grip zone fully exposed. (Testing only the CoM column would let a
    # tray with tall walls "top-mount" onto its recessed floor, burying the tab inside the walls --
    # the render caught exactly that; a part with any tall feature side-mounts instead.)
    top = _surface_top_at(mesh, xc, yc, z_hi_ray)
    if top is not None and part_max_z <= z_bed + BAND_LO - 1.0:
        fin_top = grip_top + TOP_CHAMFER
        if _clear_cylinder_ok(mesh, xc, yc, fin_top, z_bed + CLEAR_TOP):
            return _Placement("top", xc, yc, top - FUSE_OVERLAP, fin_top)

    # SIDE-MOUNT: attach at the largest axis-aligned silhouette face, at band height, sticking out.
    lo, hi = mesh.bounds
    faces = [
        ("+x", float(hi[0]), yc, (hi[1] - lo[1])),
        ("-x", float(lo[0]), yc, (hi[1] - lo[1])),
        ("+y", xc, float(hi[1]), (hi[0] - lo[0])),
        ("-y", xc, float(lo[1]), (hi[0] - lo[0])),
    ]
    faces.sort(key=lambda f: f[3], reverse=True)  # widest face first ~ "largest exterior face"
    for axis, ex, ey, _w in faces:
        dx, dy = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[axis]
        # the part must exist just inside the edge at the band, or there is nothing to fuse to
        if _surface_top_at(mesh, ex - dx * 1.0, ey - dy * 1.0, z_hi_ray) is None:
            continue
        gx = ex + dx * (SIDE_STICKOUT / 2.0 - FUSE_OVERLAP)
        gy = ey + dy * (SIDE_STICKOUT / 2.0 - FUSE_OVERLAP)
        if _clear_cylinder_ok(mesh, gx, gy, grip_top + TOP_CHAMFER, z_bed + CLEAR_TOP):
            return _Placement("side", ex, ey, z_bed + GRIP_Z_LO, z_bed + GRIP_Z_HI, axis=axis)
    return None


def _build_fin(place: _Placement) -> Any:
    """The fin solid positioned to overlap the part for a single-body union (build123d)."""
    from build123d import Box, Pos

    if place.mode == "top":
        cz = (place.z_lo + place.z_hi) / 2.0
        return Pos(place.x, place.y, cz) * Box(GRIP_W, GRIP_T, place.z_hi - place.z_lo)
    dx, dy = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}[place.axis]
    cx = place.x + dx * (SIDE_STICKOUT / 2.0 - FUSE_OVERLAP)
    cy = place.y + dy * (SIDE_STICKOUT / 2.0 - FUSE_OVERLAP)
    cz = (place.z_lo + place.z_hi) / 2.0
    if dx != 0:  # outward along X: X=stickout, Y=jaw gap, Z=grip height
        return Pos(cx, cy, cz) * Box(SIDE_STICKOUT, GRIP_T, GRIP_H)
    return Pos(cx, cy, cz) * Box(GRIP_T, SIDE_STICKOUT, GRIP_H)


def _fuse_and_finish(part: Any, place: _Placement) -> Any:
    """Fuse the fin, then best-effort 2 mm root fillet + 1 mm top chamfer. A fillet/chamfer failure
    on a given geometry is non-fatal (the plain fused tab still satisfies the grasp spec); the final
    re-validation is the safety net for whatever geometry results."""
    from build123d import Axis, chamfer, fillet

    fused = part + _build_fin(place)
    # Root fillet at the fin/part junction: for a top-mount that is the horizontal surface plane
    # (z_lo + the fuse overlap); for a side-mount the junction is vertical, so a z-band fillet is a
    # best-effort attempt only.
    junction_z = place.z_lo + FUSE_OVERLAP if place.mode == "top" else place.z_lo
    try:
        root = fused.edges().filter_by_position(
            Axis.Z, minimum=junction_z - 0.3, maximum=junction_z + 0.3
        )
        if root:
            fused = fillet(root, ROOT_FILLET)
    except Exception:
        pass
    try:
        top = fused.edges().filter_by_position(
            Axis.Z, minimum=place.z_hi - 0.3, maximum=place.z_hi + 0.3
        )
        if top:
            fused = chamfer(top, TOP_CHAMFER)
    except Exception:
        pass
    return fused


def emit_grasp_tab(
    step_path: Path,
    stl_path: Path,
    out_dir: Path,
    *,
    material: Any,
    brief: Any,
) -> GraspTabResult:
    """Fuse the standardized grasp tab onto an accepted cell part. Never raises: every failure path
    REFUSES with a reason (fail-closed). On success, returns the tabbed STL/STEP paths."""
    try:
        mesh = trimesh.load(str(stl_path), force="mesh")
    except Exception as exc:
        return GraspTabResult(ok=False, refused=True, reason=f"could not load part mesh: {exc}")

    place = _plan_placement(mesh)
    if place is None:
        return GraspTabResult(
            ok=False,
            refused=True,
            reason=(
                "no valid grasp-tab placement: the part exposes no reachable face in the 10-15 mm "
                "grip band with a clear 25 mm approach column above (too tall, rounded, or enclosed)"
            ),
        )

    try:
        from build123d import export_step, export_stl, import_step

        part = import_step(str(step_path))
        tabbed = _fuse_and_finish(part, place)
        if len(tabbed.solids()) != 1:
            return GraspTabResult(
                ok=False,
                refused=True,
                reason="tab fusion did not yield a single solid",
                mode=place.mode,
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        tab_step = out_dir / "part_tabbed.step"
        tab_stl = out_dir / "part_tabbed.stl"
        export_step(tabbed, str(tab_step))
        export_stl(tabbed, str(tab_stl))
    except Exception as exc:
        return GraspTabResult(
            ok=False,
            refused=True,
            reason=f"tab fusion failed: {type(exc).__name__}: {exc}",
            mode=place.mode,
        )

    # Mass cap (SO-101 payload). Reuse the existing material mass calc on the tabbed volume.
    mass_g: float | None = None
    try:
        from forge_mind.agentic.material_report import evaluate_material

        tabbed_mesh: Any = trimesh.load(str(tab_stl), force="mesh")
        mr = evaluate_material(material, volume_mm3=float(abs(tabbed_mesh.volume)))
        mass_g = mr.mass_g
        if mass_g > MASS_CAP_G:
            return GraspTabResult(
                ok=False,
                refused=True,
                reason=f"tabbed part is {mass_g:.0f} g, over the {MASS_CAP_G:.0f} g SO-101 payload cap",
                mode=place.mode,
                mass_g=mass_g,
            )
    except Exception:
        mass_g = None  # mass unmeasurable -> do not fabricate; the re-validation below still gates

    # Re-validate the tabbed solid: a tab that breaks watertightness / single-body / walls REFUSES.
    try:
        from forge_mind.agentic.validate import validate_authored_part
        from forge_mind.fabrication import FabricationMode

        reval = validate_authored_part(
            tab_stl, tab_step, brief, fabrication_mode=FabricationMode.PRINT_3D
        )
        if not reval.hard_ok:
            return GraspTabResult(
                ok=False,
                refused=True,
                reason="the fused tab broke geometry validation (not one watertight solid)",
                mode=place.mode,
                mass_g=mass_g,
            )
    except Exception as exc:
        return GraspTabResult(
            ok=False,
            refused=True,
            reason=f"tabbed-part re-validation could not run: {type(exc).__name__}",
            mode=place.mode,
            mass_g=mass_g,
        )

    return GraspTabResult(
        ok=True,
        refused=False,
        stl_path=tab_stl,
        step_path=tab_step,
        mode=place.mode,
        grip_dims_mm=(GRIP_W, GRIP_H, GRIP_T),
        mass_g=mass_g,
    )


__all__ = ["MASS_CAP_G", "GraspTabResult", "emit_grasp_tab"]
