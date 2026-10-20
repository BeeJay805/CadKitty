"""Per-part FIDELITY tier: is each shipped part a genuine solid, and a realistic instance
of its stated role? (founder 2026-07-12, the Merlin valves). Function is the other tiers'
job (fit/motion/DFM); this tier catches the part that passes them all while being a lie:
a "ball valve" that is external massing with no flow passage, or a mesh that is not
actually a solid. Sibling of the intent-fidelity gate (validator/fidelity.py), which
compares a Blueprint against a PartProgram in the DSL lane -- this module measures MESHES.

Deterministic geometry only, measured on the real probe corpus (ADR-0091):

* ROBUST WATERTIGHTNESS -- the load-bearing check. A raw STL always reads
  ``is_watertight=False`` (the format duplicates vertices per triangle), so naive flagging
  false-fails every export: weld first, flag only what stays broken. Two genuine classes:
  OPEN BOUNDARY (edges bounding one face -- the volume is not enclosed) is the one
  unambiguous non-solid -> BLOCKER; NON-MANIFOLD FUSED edges (wrong euler, zero open
  boundary -- overlapping shells fused in place) measured on shipped-and-printable parts
  -> CAVEAT, report only.
* EXTERNAL-MASSING vs ROLE -- a part whose NAME says through-flow (valve/pump/nozzle/...)
  but whose topology has no through-passage (genus 0) is massing, not the thing: CAVEAT
  "modeled as external massing only". Keyed on the part name ONLY -- prompt text measured
  too noisy (incidental "mount"/"shaft" words negated the founder's own valve case) -- and
  negated by solid-subpart words (a "valve stem" is legitimately solid). Exact topology,
  never a face-count floor, so a simple bracket or pin can never false-FAIL.

The tier REPORTS by default and gates nothing except the open-boundary blocker, which the
caller folds into the clean-ship gate exactly like a DFM blocker (caps at caveats). An
unmeasurable part is reported as a caveat, never silently passed."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

# Name-keyed role expectations (word-boundary matched on the '_'/'-'-normalized name).
# THROUGH roles must have a through-passage (genus >= 1) to be more than massing; CAVITY
# roles get flagged only when ALSO near-convex (a solid brick "housing" -- a shell with an
# open cavity is strongly non-convex and passes). Conservative: unlisted roles are never
# checked; a listed role with any solid-subpart negation word is never checked.
_THROUGH_ROLES = ("valve", "pump", "manifold", "nozzle", "injector", "duct", "pipe", "tube")
_CAVITY_ROLES = ("housing", "enclosure", "gearbox")
_NEGATIONS = (
    "stem", "handle", "knob", "lever", "pin", "shaft", "bracket", "mount",
    "plate", "cap", "plug", "cover", "seat", "flange", "spacer", "standoff",
    "clamp", "clip", "gasket", "arm", "link", "rod", "axle", "washer",
)  # fmt: skip

# A solid "cavity-role" part is massing only when it is ALSO essentially its own convex
# hull (no meaningful concavity anywhere). Measured: real housings/shells sit <= 0.78.
_CONVEXITY_MASSING = 0.90
# A repaired open boundary whose total length is under this fraction of the bbox diagonal
# is a pinhole seam (slicer-repairable caveat), not a missing surface.
_SEAM_REL_BOUNDARY = 0.05
_EPS_MM = 1e-6


@dataclass
class FidelityFinding:
    part: str
    check: str  # solidity | external_massing
    severity: str  # blocker | caveat
    detail: str


@dataclass
class FidelityReport:
    tier: str  # "part-fidelity"
    verdict: str  # pass | caveats | fail | unchecked
    findings: list[FidelityFinding]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "verdict": self.verdict,
            "findings": [
                {"part": f.part, "check": f.check, "severity": f.severity, "detail": f.detail}
                for f in self.findings
            ],
        }


def _word_hit(text: str, words: tuple[str, ...]) -> str | None:
    for w in words:
        if re.search(rf"\b{w}s?\b", text):
            return w
    return None


def _welded(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A positional-weld copy with degenerate faces dropped -- undoes the raw-STL
    duplicate-vertex artifact without touching the caller's mesh."""
    w = mesh.copy()
    w.merge_vertices()
    w.update_faces(w.nondegenerate_faces())
    w.remove_unreferenced_vertices()
    return w


def _solid_for_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """The mesh to run exact topology on: itself if watertight, else its weld -- None if
    neither is a valid volume (solidity already reported the defect)."""
    if mesh.is_watertight:
        return mesh if mesh.is_volume else None
    w = _welded(mesh)
    return w if w.is_watertight and w.is_volume else None


def _open_boundary(mesh: trimesh.Trimesh) -> tuple[int, float]:
    """(count, total length mm) of real open-boundary edges (edges bounding exactly one
    face, zero-length degenerates excluded)."""
    edges = mesh.edges_sorted
    rows = trimesh.grouping.group_rows(edges, require_count=1)  # type: ignore[no-untyped-call]
    if len(rows) == 0:
        return 0, 0.0
    b = edges[rows]
    lengths = np.linalg.norm(mesh.vertices[b[:, 0]] - mesh.vertices[b[:, 1]], axis=1)
    real = lengths[lengths > _EPS_MM]
    return len(real), float(real.sum())


def _check_solidity(mesh: trimesh.Trimesh) -> tuple[str, str] | None:
    """None if a genuine solid (possibly after welding); else (severity, detail).
    BLOCKER only for an open boundary that survives welding and is not a cappable
    pinhole -- the one unambiguous not-a-solid. Everything else reports."""
    if mesh.is_watertight:
        if mesh.is_volume:
            return None
        return ("caveat", "watertight but not a valid volume (winding/orientation)")
    w = _welded(mesh)
    if w.is_watertight:
        return (
            None
            if w.is_volume
            else (
                "caveat",
                "watertight after welding but not a valid volume (winding/orientation)",
            )
        )
    n_open, boundary_mm = _open_boundary(w)
    if n_open > 0:
        diag = float(np.linalg.norm(w.bounding_box.extents))
        rel = boundary_mm / diag if diag > 0 else float("inf")
        filled = w.copy()
        try:
            filled.fill_holes()
            repaired = bool(filled.is_watertight)
        except Exception:
            repaired = False
        if repaired and rel <= _SEAM_REL_BOUNDARY:
            return (
                "caveat",
                f"pinhole open seam ({n_open} boundary edges, {boundary_mm:.1f} mm, "
                "trivially cappable -- slicer-repairable, verify the export)",
            )
        return (
            "blocker",
            f"not a solid: open boundary of {n_open} edges, {boundary_mm:.1f} mm "
            f"({rel:.0%} of the part diagonal) -- the volume is not enclosed",
        )
    return (
        "caveat",
        f"non-manifold fused geometry (euler {w.euler_number}, no open boundary) -- "
        "overlapping shells fused in place; usually slicer-repairable, verify the export",
    )


def _check_external_massing(mesh: trimesh.Trimesh, name: str) -> str | None:
    """Caveat detail if a role-named part is topologically bare massing, else None.
    Runs only on a single-body valid solid; exact genus, never sampled."""
    norm = re.sub(r"[_\-]+", " ", name.lower())
    if _word_hit(norm, _NEGATIONS) is not None:
        return None
    through = _word_hit(norm, _THROUGH_ROLES)
    cavity = _word_hit(norm, _CAVITY_ROLES)
    if through is None and cavity is None:
        return None
    solid = _solid_for_topology(mesh)
    if solid is None or solid.body_count != 1:
        return None
    genus = (2 - solid.euler_number) // 2
    if genus != 0:
        return None
    if through is not None:
        return (
            f"low-fidelity: '{through}' modeled as external massing only -- no "
            "through-passage (genus 0); a real one passes flow"
        )
    try:
        convexity = float(solid.volume) / float(solid.convex_hull.volume)
    except Exception:
        return None
    if convexity >= _CONVEXITY_MASSING:
        return (
            f"low-fidelity: '{cavity}' modeled as external massing only -- a solid "
            f"near-convex body (convexity {convexity:.2f}), no internal cavity"
        )
    return None


def fidelity_report(parts: list[tuple[str, trimesh.Trimesh]]) -> FidelityReport:
    """Run the fidelity tier over the assembly's (part name, mesh) pairs. Fail-open per
    part -- but an UNMEASURABLE part is reported as a caveat, never silently passed
    (a fidelity verdict that lies is worse than none)."""
    findings: list[FidelityFinding] = []
    measured = 0
    for name, mesh in parts:
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
            continue
        try:
            hit = _check_solidity(mesh)
            measured += 1
            if hit is not None:
                findings.append(FidelityFinding(name, "solidity", hit[0], hit[1]))
        except Exception as exc:
            findings.append(
                FidelityFinding(
                    name, "solidity", "caveat", f"fidelity unmeasurable ({type(exc).__name__})"
                )
            )
        try:
            massing = _check_external_massing(mesh, name)
            if massing is not None:
                findings.append(FidelityFinding(name, "external_massing", "caveat", massing))
        except Exception:
            pass  # advisory check: unmeasurable role richness stays silent
    if measured == 0 and not findings:
        return FidelityReport(tier="part-fidelity", verdict="unchecked", findings=[])
    if any(f.severity == "blocker" for f in findings):
        verdict = "fail"
    elif findings:
        verdict = "caveats"
    else:
        verdict = "pass"
    return FidelityReport(tier="part-fidelity", verdict=verdict, findings=findings)


__all__ = ["FidelityFinding", "FidelityReport", "fidelity_report"]
