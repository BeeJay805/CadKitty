"""Geometry-level symmetry verification (Sol consultation 2026-07-24, P0.2 / task 3,
decision-independent): does a declared left/right pair actually come out GEOMETRICALLY
congruent under reflection, measured on the built B-rep/mesh -- not merely name-paired?

The H3 humanoid shipped shin_foot_L 58 mm wide and shin_foot_R 32 mm wide: a mirrored
pair that was not a mirror. The plan-level ``_plan_mirror_issues`` check catches a MISSING
sibling (roster asymmetry) but is blind to two present-but-unequal parts. This closes that
gap on the actual geometry.

Standalone by design: this returns its own fail-closed verdict and is NOT wired into the
live acceptance/disposition path. Moving the ship verdict onto geometry is the intent-gate
reclassification (Sol task 1), which is held pending the stack maturing. This module is the
kind of deterministic, model-free geometry validator that reclassification will later
consume.

Discipline (repo cad-validator-designer skill): every check returns its measured value(s)
+ the limit + evidence, and returns ``not_run`` -- never PASS -- on any missing/unmeasurable
input. It never guesses.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

Status = Literal["pass", "fail", "not_run"]

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class GeomVerdict:
    """One geometry check's result, in the check-spec shape: id, category, status, the
    measured quantities, the limits applied, and a human message. ``not_run`` is a real
    outcome (fail-closed) -- it is never a silent PASS."""

    check_id: str
    category: str
    status: Status
    message: str
    measured: dict[str, float] = field(default_factory=dict)
    limit: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "pass"


def _is_usable_mesh(mesh: Any) -> bool:
    """A mesh we can measure: a Trimesh with faces. Anything else -> not_run upstream."""
    try:
        return (
            mesh is not None
            and hasattr(mesh, "faces")
            and hasattr(mesh, "vertices")
            and len(mesh.faces) > 0
            and len(mesh.vertices) > 0
        )
    except Exception:
        return False


def _prep_centered(mesh: Any) -> Any:
    """A copy with consistent outward winding, centered at its centroid, so congruence is
    tested SHAPE-only (independent of where the part sits in the assembly). Uses the
    volumetric centroid when the mesh is a valid solid, else the area/vertex centroid."""
    m = mesh.copy()
    with contextlib.suppress(Exception):
        m.fix_normals()
    try:
        center = m.center_mass if m.is_watertight and m.volume > 0 else m.centroid
    except Exception:
        center = m.centroid
    m.apply_translation(-np.asarray(center, dtype=float))
    return m


def _reflect(mesh: Any, axis_i: int) -> Any:
    """Reflect a (centered) mesh across the axis plane through the origin, then restore
    outward winding (reflection inverts orientation). Stays centered at the origin."""
    m = mesh.copy()
    t = np.eye(4)
    t[axis_i, axis_i] = -1.0
    m.apply_transform(t)
    with contextlib.suppress(Exception):
        m.fix_normals()
    return m


def _probe_points(mesh: Any, n_samples: int, seed: int) -> np.ndarray:
    """Probe points for one surface: every REFERENCED vertex plus a SEEDED random sample.

    Referenced vertices only: a mesh built with ``process=False`` can carry orphan vertices no
    face uses, which are not part of the surface being compared -- probing them would fail two
    geometrically identical surfaces on a point that isn't on either one.

    Vertices are included because a localized defect (a bump, a missing chamfer, a shifted
    boss) is bounded by vertices, and random face sampling can miss it by luck. The random
    component is seeded so the same input always yields the same verdict: unseeded sampling
    made identical inputs range 0.299-0.479 mm across six runs (measured), which makes the
    evidence unreproducible and unusable for a frozen acceptance manifest."""
    faces = np.asarray(mesh.faces)
    referenced = np.unique(faces.reshape(-1))
    verts: np.ndarray = np.asarray(mesh.vertices, dtype=float)[referenced]
    if n_samples == 0:
        return verts
    import trimesh

    sampled, _ = trimesh.sample.sample_surface(mesh, n_samples, seed=seed)
    stacked: np.ndarray = np.vstack([verts, np.asarray(sampled, dtype=float)])
    return stacked


def _two_sided_max_surface_dev(
    a: Any, b: Any, n_samples: int, seed: int = 0
) -> tuple[float, float]:
    """Two-sided Hausdorff-ish deviation between two surfaces: probe each surface, measure the
    nearest distance to the other, take the max (and mean) over both directions. Both surfaces
    are probed so a feature present on one but not the other is caught from whichever side
    carries it.

    NOTE (Sol's open item): this is a SAMPLED estimate with vertex coverage, not a certified
    bound -- it can still under-report a defect on a large smooth face between vertices. A
    certified two-sided bound is queued; until then a PASS here is strong evidence, not proof."""
    import trimesh

    pa = _probe_points(a, n_samples, seed)
    pb = _probe_points(b, n_samples, seed)
    _, da, _ = trimesh.proximity.closest_point(b, pa)  # type: ignore[no-untyped-call]
    _, db, _ = trimesh.proximity.closest_point(a, pb)  # type: ignore[no-untyped-call]
    max_dev = float(max(float(np.max(da)), float(np.max(db))))
    mean_dev = float((float(np.mean(da)) + float(np.mean(db))) / 2.0)
    return max_dev, mean_dev


def mirror_congruence(
    mesh_left: Any,
    mesh_right: Any,
    *,
    mirror_axis: str = "y",
    tol_mm: float = 0.5,
    vol_tol_frac: float = 0.02,
    n_samples: int = 800,
) -> GeomVerdict:
    """Are ``mesh_left`` and ``mesh_right`` geometrically congruent under reflection across
    ``mirror_axis``? Measures, on the centered meshes, the max sampled surface deviation
    between the reflected left and the right, plus the bbox-extent delta and the volume
    delta. PASS only when ALL are within tolerance.

    ``mirror_axis`` is a parameter (default 'y') -- this check does NOT commit to any one
    global frame convention (that freeze is held pending ratification); the caller states the
    plane. ``tol_mm`` defaults to a print-tolerance-scale 0.5 mm; ``vol_tol_frac`` bounds the
    relative volume mismatch.

    Fail-closed: returns ``not_run`` (never PASS) when either mesh is missing/empty or the
    surface sampling cannot run.
    """
    axis = mirror_axis.lower()
    check_id = f"symmetry.mirror_congruence.{axis}"
    cat = "assembly"
    limit = {"tol_mm": tol_mm, "vol_tol_frac": vol_tol_frac}
    if axis not in _AXIS_INDEX:
        return GeomVerdict(
            check_id, cat, "not_run", f"unknown mirror_axis {mirror_axis!r} (use x|y|z)", {}, limit
        )
    if not isinstance(n_samples, int) or n_samples < 0:
        # A negative count is a malformed measurement request, not a request for vertex-only
        # mode (n_samples=0 is that, explicitly). Fail closed rather than silently degrade.
        return GeomVerdict(
            check_id, cat, "not_run", f"invalid n_samples {n_samples!r} (must be >= 0)", {}, limit
        )
    for name, mesh in (("left", mesh_left), ("right", mesh_right)):
        if not _is_usable_mesh(mesh):
            return GeomVerdict(check_id, cat, "not_run", f"{name} mesh missing or empty", {}, limit)
    try:
        left_c = _prep_centered(mesh_left)
        right_c = _prep_centered(mesh_right)
        left_reflected = _reflect(left_c, _AXIS_INDEX[axis])
        vol_l = abs(float(left_reflected.volume))
        vol_r = abs(float(right_c.volume))
        vol_delta = abs(vol_l - vol_r) / max(vol_r, 1e-9)
        ext_delta = float(np.max(np.abs(left_reflected.extents - right_c.extents)))
        max_dev, mean_dev = _two_sided_max_surface_dev(left_reflected, right_c, n_samples)
    except Exception as exc:  # any measurement failure is not_run, never a PASS
        return GeomVerdict(
            check_id, cat, "not_run", f"measurement failed: {type(exc).__name__}: {exc}", {}, limit
        )
    measured = {
        "max_surface_dev_mm": max_dev,
        "mean_surface_dev_mm": mean_dev,
        "bbox_extent_delta_mm": ext_delta,
        "volume_delta_frac": vol_delta,
    }
    congruent = max_dev <= tol_mm and ext_delta <= tol_mm and vol_delta <= vol_tol_frac
    if congruent:
        msg = (
            f"L/R congruent under {axis}-reflection: max surface dev {max_dev:.3f} mm "
            f"(<= {tol_mm}), extent delta {ext_delta:.3f} mm, volume delta {vol_delta:.3%}"
        )
        return GeomVerdict(check_id, cat, "pass", msg, measured, limit)
    msg = (
        f"L/R NOT congruent under {axis}-reflection: max surface dev {max_dev:.3f} mm "
        f"(limit {tol_mm}), extent delta {ext_delta:.3f} mm, volume delta {vol_delta:.3%} "
        f"(limit {vol_tol_frac:.1%}) -- a mirrored pair that is not a mirror (H3 shin class)"
    )
    return GeomVerdict(check_id, cat, "fail", msg, measured, limit)


__all__ = ["GeomVerdict", "mirror_congruence"]
