"""Defect confirmers (ADR-0083): independent geometric proof that an injected defect
actually materialized in the built mutant.

A mutant trial is VALID only when its confirmer returns confirmed=True; everything else
is discarded (invalid trial), so operator sloppiness costs yield, never label truth.

Independence rule: the class whose validator check is under audit gets an instrument on a
DIFFERENT code path. Through-ness (ADR-0082 ``stack_pierces``) is confirmed by trimesh
containment sampling along the bore axis on the tessellated STL -- not by the validator's
B-rep probe. Hole counts/positions reuse the B-rep cylinder extraction that the frozen
bench already trusts as its measurement layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.audit.mutations import MutantCandidate

_BLIND_SAMPLES = 41
_DRIFT_CONFIRM_MM = 3.0
_ENVELOPE_GROWTH = 1.25
_BBOX_MOVE_REL = 0.06
_VOLUME_MOVE_REL = 0.03


@dataclass(frozen=True)
class Confirmation:
    confirmed: bool
    evidence: str


def _mesh(path: Path) -> Any | None:
    try:
        m: Any = trimesh.load(str(path), force="mesh")
        return m if bool(m.is_watertight) else None
    except Exception:
        return None


def _concave_stacks(step_path: Path) -> list[Any]:
    """Concave (hole-wall) coaxial cylinder stacks from a STEP -- the same B-rep
    measurement layer the frozen bench uses. Empty on any failure (measured nothing)."""
    try:
        from build123d import import_step

        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

        part = import_step(str(step_path))
        return [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]
    except Exception:
        return []


def _axis_line_points(mesh: Any, stack: Any, pad: float = 1.0) -> Any:
    """Sample points along the stack's axis LINE across the PART's projected extent.

    Through-ness is a whole-part property: the floor of a shortened bore lies outside
    the bore-wall stack's own axial extent, so the probe must span the mesh, not the
    stack. The mesh's bbox corners projected onto the axis bound the segment."""
    a = np.asarray(stack.axis_point, dtype=float)
    d = np.asarray(stack.axis_dir, dtype=float)
    d = d / np.linalg.norm(d)
    lo3, hi3 = np.asarray(mesh.bounds, dtype=float)
    corners = np.array(
        [[x, y, z] for x in (lo3[0], hi3[0]) for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])]
    )
    ts_all = corners @ d
    lo, hi = float(np.min(ts_all)), float(np.max(ts_all))
    if hi - lo <= 2 * pad:
        ts = np.array([(lo + hi) / 2.0])
    else:
        ts = np.linspace(lo + pad, hi - pad, _BLIND_SAMPLES)
    t0 = float(np.dot(a, d))
    return a[None, :] + d[None, :] * (ts - t0)[:, None]


def _line_is_open(mesh: Any, stack: Any) -> bool:
    """True when no point along the bore's axis line sits inside solid material."""
    try:
        pts = _axis_line_points(mesh, stack)
        return not bool(np.any(mesh.contains(pts)))
    except Exception:
        return False


def _perp_line_distance(s1: Any, s2: Any) -> float:
    """Distance between two (near-)parallel stack axis LINES, perpendicular to the axis."""
    a1 = np.asarray(s1.axis_point, dtype=float)
    a2 = np.asarray(s2.axis_point, dtype=float)
    d = np.asarray(s1.axis_dir, dtype=float)
    d = d / np.linalg.norm(d)
    v = a2 - a1
    return float(np.linalg.norm(v - np.dot(v, d) * d))


def _bbox_extents(mesh: Any) -> Any:
    b = np.asarray(mesh.bounds, dtype=float)
    return b[1] - b[0]


# ------------------------------------------------------------- per-class confirmers ---


def _confirm_blind(base_step: Path, base_mesh: Any, mut_mesh: Any) -> Confirmation:
    through = [s for s in _concave_stacks(base_step) if _line_is_open(base_mesh, s)]
    if not through:
        return Confirmation(False, "baseline has no measurable through bore")
    for s in through:
        if not _line_is_open(mut_mesh, s):
            return Confirmation(
                True,
                f"bore d={2 * float(s.radius):.2f} axis line now blocked by material "
                "(was open end-to-end in the baseline)",
            )
    return Confirmation(False, "every baseline through bore is still open in the mutant")


def _diam_counts(stacks: list[Any], tol: float = 0.3) -> list[tuple[float, int]]:
    out: list[tuple[float, int]] = []
    for s in stacks:
        d = 2.0 * float(s.radius)
        for i, (d0, n) in enumerate(out):
            if abs(d - d0) <= tol:
                out[i] = (d0, n + 1)
                break
        else:
            out.append((d, 1))
    return out


def _confirm_dropped(
    base_step: Path, mut_step: Path, base_mesh: Any, mut_mesh: Any
) -> Confirmation:
    base = _diam_counts(_concave_stacks(base_step))
    mut = _diam_counts(_concave_stacks(mut_step))
    for d0, n0 in base:
        n1 = next((n for d1, n in mut if abs(d1 - d0) <= 0.3), 0)
        if n1 < n0:
            return Confirmation(True, f"holes d={d0:.2f}: {n0} -> {n1}")
    dv = abs(float(mut_mesh.volume) - float(base_mesh.volume)) / max(float(base_mesh.volume), 1e-9)
    if dv >= _VOLUME_MOVE_REL:
        return Confirmation(True, f"non-hole feature removed: volume moved {dv:.1%}")
    return Confirmation(False, "no hole-count drop and volume unchanged")


def _confirm_off_footprint(base_step: Path, mut_step: Path) -> Confirmation:
    base = _concave_stacks(base_step)
    mut = _concave_stacks(mut_step)
    if not base or len(base) != len(mut):
        return Confirmation(False, "stack count changed or unmeasurable -- not a pure drift")
    worst = 0.0
    for m in mut:
        same_d = [b for b in base if abs(2 * float(b.radius) - 2 * float(m.radius)) <= 0.3]
        if not same_d:
            return Confirmation(False, "a mutant bore has no diameter match in the baseline")
        worst = max(worst, min(_perp_line_distance(b, m) for b in same_d))
    if worst >= _DRIFT_CONFIRM_MM:
        return Confirmation(True, f"bore pattern drifted {worst:.1f} mm (size/count intact)")
    return Confirmation(False, f"max bore drift {worst:.1f} mm < {_DRIFT_CONFIRM_MM} mm")


def _confirm_envelope(base_mesh: Any, mut_mesh: Any) -> Confirmation:
    b, m = _bbox_extents(base_mesh), _bbox_extents(mut_mesh)
    ratios = m / np.maximum(b, 1e-9)
    worst = float(np.max(ratios))
    if worst >= _ENVELOPE_GROWTH:
        axis = "XYZ"[int(np.argmax(ratios))]
        return Confirmation(True, f"{axis} extent grew x{worst:.2f} vs baseline")
    return Confirmation(False, f"max axis growth x{worst:.2f} < x{_ENVELOPE_GROWTH}")


def _confirm_stated_dim(
    cand: MutantCandidate, base_step: Path, mut_step: Path, base_mesh: Any, mut_mesh: Any
) -> Confirmation:
    old = float(cand.meta.get("old", 0.0))
    new = float(cand.meta.get("new", 0.0))
    if old <= 0:
        return Confirmation(False, "no old value in candidate meta")
    base = _diam_counts(_concave_stacks(base_step))
    mut = _diam_counts(_concave_stacks(mut_step))
    # strong: the stated bore itself moved (param may be a radius -> k in {1, 2})
    for k in (1.0, 2.0):
        d_old, d_new = k * old, k * new
        had = any(abs(d - d_old) <= 0.25 for d, _ in base)
        lost = not any(abs(d - d_old) <= 0.25 for d, _ in mut)
        gained = any(abs(d - d_new) <= 0.25 for d, _ in mut)
        if had and lost and gained:
            return Confirmation(True, f"stated bore d={d_old:g} now measures d={d_new:g}")
    # medium: a stated envelope dimension moved by ~the injected factor
    b, m = _bbox_extents(base_mesh), _bbox_extents(mut_mesh)
    rel = np.abs(m - b) / np.maximum(b, 1e-9)
    if float(np.max(rel)) >= _BBOX_MOVE_REL:
        axis = "XYZ"[int(np.argmax(rel))]
        return Confirmation(
            True, f"{axis} extent moved {float(np.max(rel)):.1%} off the stated dim"
        )
    return Confirmation(False, "neither a stated bore nor an envelope extent measurably moved")


def _thickness_p5(mesh: Any) -> float | None:
    """5th-percentile max-sphere thickness over surface samples. The raw MIN is an edge
    artifact magnet (a sharp corner reads ~0 on any part); the low percentile tracks the
    part's genuinely thin regions instead."""
    try:
        pts, _ = mesh.sample(400, return_index=True)
        thickness_fn: Any = trimesh.proximity.thickness
        th = thickness_fn(mesh, pts)
        finite = th[np.isfinite(th)]
        if finite.size < 50:
            return None
        return float(np.percentile(finite, 5))
    except Exception:
        return None


def _confirm_subfloor(base_mesh: Any, mut_mesh: Any, floor_mm: float) -> Confirmation:
    """Differential: the mutant's thin-region thickness must sit under the floor AND have
    DROPPED materially vs the baseline -- shared edge artifacts cancel, so a spiky good
    part cannot self-confirm."""
    p5_base = _thickness_p5(base_mesh)
    p5_mut = _thickness_p5(mut_mesh)
    if p5_base is None or p5_mut is None:
        return Confirmation(False, "thickness unmeasurable on baseline or mutant")
    if p5_mut <= 0.8 * floor_mm and p5_mut <= 0.7 * p5_base:
        return Confirmation(
            True,
            f"thin-region thickness {p5_base:.2f} -> {p5_mut:.2f} mm (floor {floor_mm} mm)",
        )
    return Confirmation(
        False,
        f"thin-region thickness {p5_mut:.2f} mm (baseline {p5_base:.2f}) not a confirmed drop",
    )


def _confirm_undersized_bore(
    cand: MutantCandidate, base_step: Path, mutant_step: Path
) -> Confirmation:
    old = float(cand.meta.get("old", 0.0))
    new = float(cand.meta.get("new", 0.0))
    base_d = [2.0 * float(stack.radius) for stack in _concave_stacks(base_step)]
    mutant_d = [2.0 * float(stack.radius) for stack in _concave_stacks(mutant_step)]
    if any(abs(d - old) <= 0.2 for d in base_d) and any(abs(d - new) <= 0.2 for d in mutant_d):
        return Confirmation(True, f"clearance bore diameter {old:.2f} -> {new:.2f} mm")
    return Confirmation(False, "planned clearance bore did not measurably shrink")


def confirm(
    cand: MutantCandidate,
    *,
    baseline_stl: Path,
    baseline_step: Path,
    mutant_stl: Path,
    mutant_step: Path,
    min_wall_mm: float | None,
) -> Confirmation:
    """Dispatch to the class confirmer. A non-watertight mutant is never confirmed
    (the runner should have filtered it; belt and braces here)."""
    base_mesh = _mesh(baseline_stl)
    mut_mesh = _mesh(mutant_stl)
    if base_mesh is None or mut_mesh is None:
        return Confirmation(False, "baseline or mutant mesh unloadable / not watertight")
    if cand.defect_class == "blind_where_through":
        return _confirm_blind(baseline_step, base_mesh, mut_mesh)
    if cand.defect_class == "dropped_named_feature":
        return _confirm_dropped(baseline_step, mutant_step, base_mesh, mut_mesh)
    if cand.defect_class == "off_footprint_bore":
        return _confirm_off_footprint(baseline_step, mutant_step)
    if cand.defect_class == "envelope_violation":
        return _confirm_envelope(base_mesh, mut_mesh)
    if cand.defect_class == "off_brief_stated_dim":
        return _confirm_stated_dim(cand, baseline_step, mutant_step, base_mesh, mut_mesh)
    if cand.defect_class == "sub_floor_wall":
        return _confirm_subfloor(base_mesh, mut_mesh, float(min_wall_mm or 1.2))
    if cand.defect_class == "undersized_clearance_bore":
        return _confirm_undersized_bore(cand, baseline_step, mutant_step)
    return Confirmation(False, f"unknown defect class {cand.defect_class!r}")


__all__ = ["Confirmation", "confirm"]
