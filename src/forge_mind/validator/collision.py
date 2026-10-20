"""Two-mesh interference & clearance -- "do these parts collide, and by how much?".

Decision-grade geometry for a pair of meshes (e.g. two uploaded STLs): the penetration is the
EXACT overlap VOLUME from a boolean intersection (manifold3d, already a dependency), and the
clearance is the minimum surface distance from ``trimesh`` proximity. This complements the
internal assembly gate (``validator/assembly_checks.py``, which samples interior points to a
yes/no verdict for a generated ``AssemblyModel``): here we expose the check for two arbitrary
meshes and report the volume, not just a boolean.

Honest by construction: it REFUSES (``method="refused"`` with the reason) rather than guessing
when a mesh is not a closed solid (boolean volume is then undefined) or the backend fails -- it
never reports a fake clearance/penetration. Frame assumption (stated in the result): both meshes
are in the SAME coordinate frame; pass ``b_offset_mm`` to translate B if the two STLs were each
exported about their own origin.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import trimesh
from pydantic import BaseModel, computed_field

__all__ = ["InterferenceReport", "interference_report", "min_surface_clearance_mm"]

# Overlap below this is boolean numerical noise / surface contact, not a real collision.
_EPS_VOLUME_MM3 = 1.0e-3


class InterferenceReport(BaseModel):
    """Whether two meshes collide, by how much (volume), or how close they are (clearance)."""

    collide: bool
    penetration_volume_mm3: float | None  # exact overlap volume when both meshes are solids
    min_clearance_mm: float | None  # nearest-surface gap, reported when they do NOT collide
    method: Literal["boolean+proximity", "proximity-only", "refused"]
    summary: str
    caveats: list[str]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def assessed(self) -> bool:
        """True only when the verdict is trustworthy: volume-backed (the boolean ran) or the bounding
        boxes are provably disjoint. ``refused`` never is; a ``proximity-only`` verdict is trustworthy
        only when the bboxes were disjoint (recorded as ``penetration_volume_mm3 == 0.0``) -- with
        OVERLAPPING bboxes the volume is None (unverified), so ``collide=False`` there means "could
        not rule out a collision", NOT "safe". Consumers must gate on this, not ``collide`` alone
        (audit M1, ADR-0065)."""
        if self.method == "refused":
            return False
        if self.method == "proximity-only":
            return self.penetration_volume_mm3 == 0.0
        return True


def _bboxes_overlap(a: trimesh.Trimesh, b: trimesh.Trimesh) -> bool:
    amin, amax = a.bounds
    bmin, bmax = b.bounds
    return bool(np.all(amax >= bmin) and np.all(bmax >= amin))


def min_surface_clearance_mm(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float | None:
    """Minimum surface-to-surface distance (mm) between two placed meshes, both directions.

    Same proximity technique as ``assembly_checks._min_surface_gap``; ``None`` if it cannot be
    measured (so a caller skips rather than guesses). Note: unsigned -- if the meshes overlap this
    still returns a small number, so use it for the NON-colliding case (gate collision on volume).
    """
    try:
        qa: Any = trimesh.proximity.ProximityQuery(a)  # type: ignore[no-untyped-call]
        qb: Any = trimesh.proximity.ProximityQuery(b)  # type: ignore[no-untyped-call]
        d_ab = np.abs(qa.signed_distance(b.vertices))
        d_ba = np.abs(qb.signed_distance(a.vertices))
    except Exception:  # pragma: no cover - proximity backend issue
        return None
    if d_ab.size == 0 or d_ba.size == 0:
        return None
    return float(min(d_ab.min(), d_ba.min()))


def _intersection_volume_mm3(a: trimesh.Trimesh, b: trimesh.Trimesh) -> float | None:
    """Exact overlap volume (mm^3) via a boolean intersection. ``None`` if it can't be trusted
    (a mesh isn't a closed solid, or the boolean backend fails) -- never a fabricated number."""
    if not (a.is_volume and b.is_volume):
        return None
    try:
        inter = trimesh.boolean.intersection([a, b])
    except Exception:  # pragma: no cover - boolean backend issue
        return None
    if inter is None or inter.is_empty or len(inter.faces) == 0:
        return 0.0
    vol = float(abs(inter.volume))
    return vol


def interference_report(
    a: trimesh.Trimesh,
    b: trimesh.Trimesh,
    *,
    b_offset_mm: tuple[float, float, float] | None = None,
) -> InterferenceReport:
    """Assess interference/clearance between meshes ``a`` and ``b`` (in a shared frame).

    Reports the exact penetration VOLUME when both meshes are closed solids; otherwise falls back
    to a proximity-only verdict and says so. Refuses (does not guess) when neither can be measured.
    """
    caveats = ["Both meshes are assumed to be in the same coordinate frame (as assembled)."]
    if b_offset_mm is not None:
        b = b.copy()
        b.apply_translation(np.asarray(b_offset_mm, dtype=float))
        caveats.append(f"Mesh B translated by {b_offset_mm} mm before the check.")

    if not _bboxes_overlap(a, b):
        gap = min_surface_clearance_mm(a, b)
        return InterferenceReport(
            collide=False,
            penetration_volume_mm3=0.0,
            min_clearance_mm=gap,
            method="proximity-only",
            summary=(
                f"No collision (bounding boxes are disjoint); nearest surfaces ~{gap:.2f} mm apart."
                if gap is not None
                else "No collision (bounding boxes are disjoint)."
            ),
            caveats=caveats,
        )

    vol = _intersection_volume_mm3(a, b)
    if vol is not None:
        if vol > _EPS_VOLUME_MM3:
            return InterferenceReport(
                collide=True,
                penetration_volume_mm3=round(vol, 3),
                min_clearance_mm=None,
                method="boolean+proximity",
                summary=(
                    f"COLLISION: the parts overlap by {vol:.2f} mm^3 -- they cannot be assembled "
                    "as positioned. Move/resize a mating feature to clear the interference."
                ),
                caveats=caveats,
            )
        gap = min_surface_clearance_mm(a, b)
        return InterferenceReport(
            collide=False,
            penetration_volume_mm3=0.0,
            min_clearance_mm=gap,
            method="boolean+proximity",
            summary=(
                f"No collision; nearest surfaces ~{gap:.2f} mm apart (a tight clearance)."
                if gap is not None
                else "No collision (no overlap volume)."
            ),
            caveats=caveats,
        )

    # Boolean undefined (a mesh isn't a closed solid). Fall back to proximity; be explicit.
    gap = min_surface_clearance_mm(a, b)
    if gap is None:
        return InterferenceReport(
            collide=False,
            penetration_volume_mm3=None,
            min_clearance_mm=None,
            method="refused",
            summary="Cannot assess interference: a mesh is not a closed solid and proximity failed.",
            caveats=[*caveats, "Repair the mesh to a watertight solid for a volume-based verdict."],
        )
    caveats.append(
        "A mesh is not a closed solid, so penetration VOLUME is unavailable; this is a "
        "surface-distance estimate, not a watertight-boolean result."
    )
    return InterferenceReport(
        collide=False,
        penetration_volume_mm3=None,
        min_clearance_mm=gap,
        method="proximity-only",
        summary=f"Bounding boxes overlap; nearest surfaces ~{gap:.2f} mm apart (volume unverified).",
        caveats=caveats,
    )
