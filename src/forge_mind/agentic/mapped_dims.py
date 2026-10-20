"""Brief->feature POSITION mapping (ADR-0012 E1, the off_footprint_bore owning check).

The stated-dim owning check (validate._stated_extent_mismatch) owns overall EXTENTS; this owns
FEATURE POSITIONS -- where a stated hole pattern or bore sits. The reconcile that produced it
(2026-07-07) found the honest boundary and is worth stating plainly:

  * A position defect only EXISTS when the user stated the position. clip_tube's `plate_hole_pitch`,
    hook_j's `screw_spacing`, pulley's `aux_hole_pcd` drift MODEL-INVENTED holes the prompt never
    mentioned -- moving them makes a different valid part, not a defective one. So the check (and the
    audit that measures it) fire ONLY on positions traceable to a user number (the ADR-0080 anchor
    invariant, applied to positions).
  * A position defect is only CATCHABLE when the feature is a measurable BORE. Slot spacings, rib
    spacings, and blade pitch ANGLES (sd_card, din_rail, impeller) have no reliable B-rep
    measurement -- they stay fail-open with the reason named, never a guessed verdict.
  * Value-only anchoring is noisy (din_rail's slot_spacing 40 == the stated depth 40 by accident),
    so a number qualifies only when it appears in a POSITION phrase attached to a hole/bore feature.

Net: the check owns stated bore PATTERN pitch / PCD and stated bore HEIGHT-above-base. Its yield on
single-part corpora is small by nature; its value lands on interface parts (a bracket that must bolt
to a case at 76.2 mm), where a drifted pattern means the part does not fit.
"""

from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass
from typing import Any

from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.spatial import _DIA_TOL_MM

# Band shared with the stated-dim extent check: a position off by more than max(8%, 1.0mm) fails.
_POS_REL = 0.08
_POS_ABS_MM = 1.0
_AXIS_PARALLEL = 0.996  # ~5 deg: two bores count as one pattern pair when their axes align
# A position phrase feeds the bore check only from a hole-ish feature (name or a standard tag).
_HOLE_WORDS = ("hole", "bore", "bolt", "drill", "mount", "screw", "clearance", "pivot")

# Position phrases, each capturing the stated number. Matched against a hole/bore feature's dims
# text (+ the prompt), so a body/slot dimension never feeds the bore check.
_PITCH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:center[\s-]*to[\s-]*center|c[\s-]*to[\s-]*c|c-c|pitch|apart)"
    r"|(?:pitch|spacing)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mm",
    re.IGNORECASE,
)
_PCD_RE = re.compile(
    r"(?:pcd|bcd|bolt\s*circle)[^0-9]{0,10}(\d+(?:\.\d+)?)"
    r"|(\d+(?:\.\d+)?)\s*mm\s*(?:pcd|bcd|bolt\s*circle)",
    re.IGNORECASE,
)
_HEIGHT_RE = re.compile(
    r"cent(?:er|re)(?:ed|d)?\s*(?:at\s*)?(\d+(?:\.\d+)?)\s*mm\s*(?:above|over|from)\s*"
    r"(?:the\s*)?(?:base|bottom|floor)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PositionSpec:
    """One user-stated, bore-measurable position: its feature, kind, value, and (when the feature
    states one) the bore diameter that identifies which measured bores belong to it."""

    label: str
    kind: str  # "pitch" | "pcd" | "height"
    value_mm: float
    bore_d_mm: float | None


# The feature's own bore diameter, brief-standard phrasing "22 mm diameter" first, then Ø-prefixed.
_DIA_TXT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*mm\s*(?:dia(?:meter)?|bore)|[øØ⌀]\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)


def _is_hole_feature(name: str, standard: str | None) -> bool:
    low = name.lower()
    return standard is not None or any(w in low for w in _HOLE_WORDS)


def stated_bore_positions(
    brief: DesignBrief, prompt: str, stated: frozenset[float]
) -> list[PositionSpec]:
    """Parse user-stated bore positions from the brief's HOLE features (+ prompt context).

    A position qualifies only when its number is in ``stated`` (user-anchored) and it appears in a
    position phrase on a hole/bore feature -- so model-invented holes and coincidental value matches
    on non-bore features never produce a spec (see module docstring)."""
    specs: list[PositionSpec] = []

    def _anchored(v: float) -> bool:
        return any(abs(v - s) <= 0.05 for s in stated)

    for f in brief.features:
        if not _is_hole_feature(f.name, getattr(f, "standard", None)):
            continue
        dims = str(getattr(f, "approx_dims_mm", "") or "")
        dm = _DIA_TXT_RE.search(dims)
        bore_d = float(next(g for g in dm.groups() if g is not None)) if dm else None
        for rx, kind in ((_PITCH_RE, "pitch"), (_PCD_RE, "pcd"), (_HEIGHT_RE, "height")):
            m = rx.search(dims)
            if not m:
                continue
            val = float(next(g for g in m.groups() if g is not None))
            if _anchored(val):
                specs.append(PositionSpec(f.name, kind, val, bore_d))
    return specs


def _concave_stacks(part: Any) -> list[Any]:
    from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

    return [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]


def _bores_for(spec: PositionSpec, stacks: list[Any]) -> list[Any]:
    """The measured concave stacks that belong to this spec -- by diameter when the feature
    states one, else all bores (the pattern is whatever holes exist)."""
    if spec.bore_d_mm is None:
        return stacks
    matched = [s for s in stacks if abs(2.0 * s.radius - spec.bore_d_mm) <= _DIA_TOL_MM]
    return matched or stacks


def mapped_position_mismatch(
    part: Any,
    mesh: Any,
    brief: DesignBrief,
    prompt: str,
    stated: frozenset[float],
    positions: list[PositionSpec] | None = None,
) -> tuple[bool, str | None]:
    """Owning check for user-stated bore positions (ADR-0012 E1). Fail-open on every unmeasurable
    path (no B-rep, no matching bores, degenerate pattern) -- a position we cannot measure is never
    a FAIL, only the coarse checks bound it.

    ``positions`` (ADR-0088 slice 1): when the DimTable is enabled the validator passes the
    pre-parsed specs; None reparses here (the flag-off default, byte-identical)."""
    specs = positions if positions is not None else stated_bore_positions(brief, prompt, stated)
    if not specs or part is None:
        return False, None
    try:
        import numpy as np

        stacks = _concave_stacks(part)
    except Exception:
        return False, None
    if not stacks:
        return False, None

    offenses: list[str] = []
    for spec in specs:
        band = max(_POS_REL * spec.value_mm, _POS_ABS_MM)
        bores = _bores_for(spec, stacks)
        if spec.kind == "height":
            try:
                min_z = float(mesh.bounds[0][2])
            except Exception:
                continue
            heights = [float(s.axis_point[2]) - min_z for s in bores]
            if not heights:
                continue
            best = min(heights, key=lambda h: abs(h - spec.value_mm))
            if abs(best - spec.value_mm) > band:
                offenses.append(
                    f"'{spec.label}' sits {best:.1f} mm above the base vs the stated "
                    f"{spec.value_mm:g} mm (allowed {band:.1f} mm)"
                )
        elif spec.kind == "pitch":
            pitches = _pair_pitches(bores, np)
            if not pitches:
                continue
            best = min(pitches, key=lambda p: abs(p - spec.value_mm))
            if abs(best - spec.value_mm) > band:
                offenses.append(
                    f"'{spec.label}' pattern pitch nearest {best:.1f} mm vs the stated "
                    f"{spec.value_mm:g} mm (allowed {band:.1f} mm)"
                )
        elif spec.kind == "pcd":
            cands = _pcd_candidates(bores, np)
            if not cands:
                continue
            best = min(cands, key=lambda c: abs(c - spec.value_mm))
            off_by = abs(best - spec.value_mm)
            # Recognizability ceiling (L66): an offense means "the stated ring exists but is
            # MISPLACED". A nearest fit beyond 3x the band is not that ring at all -- it is a
            # different pattern or a detection gap, and absence is the presence/count checks'
            # jurisdiction, so this check stays silent (fail-open). Legacy env restores the
            # unceilinged pre-fix behavior.
            recognizable = off_by <= 3.0 * band or os.environ.get("FORGE_MAPPED_LEGACY") == "1"
            if band < off_by and recognizable:
                offenses.append(
                    f"'{spec.label}' bolt-circle diameter {best:.1f} mm vs the stated "
                    f"{spec.value_mm:g} mm (allowed {band:.1f} mm)"
                )
    if offenses:
        return True, (
            "MAPPED POSITION: a user-stated bore position is off -- "
            + "; ".join(offenses)
            + ". Place the pattern at the stated position; a stated location is the user's spec."
        )
    return False, None


def _pair_pitches(bores: list[Any], np: Any) -> list[float]:
    """Perpendicular center distances between parallel-axis bore pairs (mirrors env_checks
    ._hole_pitches: the mounting-pattern geometry)."""
    out: list[float] = []
    for i, a in enumerate(bores):
        for b in bores[i + 1 :]:
            da = np.asarray(a.axis_dir, dtype=float)
            db = np.asarray(b.axis_dir, dtype=float)
            if abs(float(np.dot(da, db))) < _AXIS_PARALLEL:
                continue
            rel = np.asarray(b.axis_point, dtype=float) - np.asarray(a.axis_point, dtype=float)
            perp = rel - float(np.dot(rel, da)) * da
            out.append(float(np.linalg.norm(perp)))
    return out


def _measured_pcd(bores: list[Any], np: Any) -> float | None:
    """The pitch-circle diameter of >=3 bore centers = 2 x mean radius from their centroid."""
    if len(bores) < 3:
        return None
    centers = np.asarray([b.axis_point for b in bores], dtype=float)
    centroid = centers.mean(axis=0)
    radii = np.linalg.norm(centers - centroid, axis=1)
    return float(2.0 * radii.mean())


def _pcd_candidates(bores: list[Any], np: Any) -> list[float]:
    """Candidate pitch-circle diameters, one per same-diameter ring with >=3 members.

    Row 50 measured the failure this prevents (hard_pipe_flange k2/k4): a central through-bore
    averaged into the bolt ring reads a true-70.0 pattern as ~61 -- a systematic false-FAIL on
    every bolt-circle-around-a-bore. Two containment rules: (1) bores cluster by DIAMETER first,
    so a 32 mm bore never joins a 5.5 mm ring (its singleton cluster simply yields no candidate);
    (2) within a cluster, on-axis outliers (radius < 0.35 x the cluster median -- a same-diameter
    center hole) are dropped and the ring re-centered before the mean. Nearest-to-stated selection
    happens at the caller, mirroring the pitch/height nearest-candidate semantics. Bores without
    a measurable radius share one cluster (fail-open: fakes and exotic stacks stay measurable)."""
    keys: list[float | None] = []
    clusters: list[list[Any]] = []
    for b in bores:
        r = getattr(b, "radius", None)
        d = float(2.0 * r) if r is not None else None
        for i, k in enumerate(keys):
            same = (k is None and d is None) or (
                k is not None and d is not None and abs(d - k) <= _DIA_TOL_MM
            )
            if same:
                clusters[i].append(b)
                break
        else:
            keys.append(d)
            clusters.append([b])
    out: list[float] = []
    for members in clusters:
        for ring in _coplanar_groups(members, np):
            if len(ring) < 3:
                continue
            centers = np.asarray([m.axis_point for m in ring], dtype=float)
            radii = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
            med = float(np.median(radii))
            if med > _POS_ABS_MM:
                keep = radii >= 0.35 * med
                if int(keep.sum()) >= 3:
                    centers = centers[keep]
                    radii = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
            out.append(float(2.0 * radii.mean()))
    return out


# The 2026-07-21 manifold false-FAIL (L66, two independent reproductions): diameter-only
# clustering pooled the head flange's 8 M6 holes with the outlet flange's 4 -- a two-plane
# pool whose ring fit read 68.0 vs the stated 48 both authors had CODED correctly. Same-size
# hole groups on different planes are different PATTERNS; split them before fitting.
_COPLANAR_GAP_MM = 10.0


def _coplanar_groups(members: list[Any], np: Any) -> list[list[Any]]:
    """Split one same-diameter cluster into coplanar rings: project centers onto the shared
    bore axis and gap-cluster the 1-D positions. FORGE_MAPPED_LEGACY=1 restores the pooled
    pre-fix behavior (the instant revert path -- no code change needed)."""
    if os.environ.get("FORGE_MAPPED_LEGACY") == "1" or len(members) < 2:
        return [members]
    axis = np.asarray(getattr(members[0], "axis_dir", (0.0, 0.0, 1.0)), dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return [members]
    axis = axis / norm
    pos = [float(np.dot(np.asarray(m.axis_point, dtype=float), axis)) for m in members]
    order = sorted(range(len(members)), key=lambda i: pos[i])
    groups: list[list[Any]] = [[members[order[0]]]]
    for prev, cur in itertools.pairwise(order):
        if pos[cur] - pos[prev] > _COPLANAR_GAP_MM:
            groups.append([])
        groups[-1].append(members[cur])
    return groups


__all__ = ["PositionSpec", "mapped_position_mismatch", "stated_bore_positions"]
