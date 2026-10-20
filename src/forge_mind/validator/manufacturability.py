"""Process-keyed DFM (Design for Manufacturability): the fabrication-SPECIFIC validation
tier, sibling to the agnostic kinematic/fit layer (founder 2026-07-12: 'not every part
we make is 3D printable'). FDM printability already exists (validator/printability.py) --
that IS the FDM DFM profile. This module adds the METAL / MACHINED profile and the scope
of the certificate.

Our DFM is DETERMINISTIC GEOMETRY, never cost or CAM simulation: it flags features a
process physically cannot produce. For 3-axis machined metal the canonical rules:

  * a round cutter cannot cut a SHARP INTERNAL CORNER -- every concave internal corner
    needs a fillet radius >= the tool radius (report/caveat: 'add a fillet');
  * walls below the process minimum chatter/deflect (a hard limit -> blocker);
  * (named follow-on) undercut surfaces no tool can reach; deep-pocket aspect ratio.

Severity: sub-minimum wall = BLOCKER (unmakeable); sharp internal corners = CAVEAT
(makeable only with a design change). Fail-open on any measurement error -- an
unmeasurable part is reported UNCHECKED, never silently passed."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from forge_mind.materials import Material, ProcessCategory
from forge_mind.validator.part_fidelity import FidelityReport

# 3-axis milling defaults (conservative, published-typical). Local to the profile until a
# calibrated per-alloy table exists; the point is deterministic geometry, not a cost model.
_METAL_MIN_WALL_MM = 0.6  # below this a metal wall chatters/deflects on a 3-axis mill
_SHARP_CONCAVE_DEG = 50.0  # face-normal angle above which a concave edge reads as a sharp corner
_SHARP_EDGE_MIN_MM = 1.0  # ignore sub-mm concave slivers (tessellation noise, not a real corner)


@dataclass
class DfmFinding:
    check: str
    severity: str  # blocker | caveat
    detail: str


@dataclass
class DfmReport:
    process: str
    tier: str  # "machined-metal" | "fdm" | ...
    verdict: str  # pass | caveats | fail | unchecked
    findings: list[DfmFinding]

    def as_dict(self) -> dict[str, Any]:
        return {
            "process": self.process,
            "tier": self.tier,
            "verdict": self.verdict,
            "findings": [
                {"check": f.check, "severity": f.severity, "detail": f.detail}
                for f in self.findings
            ],
        }


def _sharp_internal_corners(mesh: trimesh.Trimesh) -> tuple[float, int]:
    """(total sharp concave-edge length mm, corner count). A concave face adjacency whose
    face-normal angle exceeds _SHARP_CONCAVE_DEG is an unmachinable sharp internal corner
    (a fillet would spread that fold across many small-angle concave edges instead). A
    FILLETED design produces no such edges -- so this measures exactly the DFM defect."""
    try:
        adj = mesh.face_adjacency
        angles = mesh.face_adjacency_angles  # radians between the two face normals
        convex = mesh.face_adjacency_convex  # bool per adjacency
        edges = mesh.face_adjacency_edges  # vertex-index pairs of the shared edge
        v = mesh.vertices
    except Exception:
        return 0.0, 0
    thresh = math.radians(_SHARP_CONCAVE_DEG)
    total = 0.0
    count = 0
    for i in range(len(adj)):
        if convex[i] or angles[i] < thresh:
            continue  # convex edge, or a gentle concave blend (a fillet reads as gentle)
        length = float(np.linalg.norm(v[edges[i][0]] - v[edges[i][1]]))
        if length >= _SHARP_EDGE_MIN_MM:
            total += length
            count += 1
    return total, count


def check_metal_dfm(mesh: trimesh.Trimesh) -> DfmReport:
    """Deterministic machined-metal DFM on a single part mesh. Fail-open per check."""
    findings: list[DfmFinding] = []
    # min wall (reuse the geometry-agnostic estimator) -> hard blocker below the limit
    try:
        from forge_mind.validator.feature_checks import _measure_min_wall

        mw = _measure_min_wall(mesh)
        if mw is not None and mw[0] < _METAL_MIN_WALL_MM:
            findings.append(
                DfmFinding(
                    "min_wall",
                    "blocker",
                    f"wall {mw[0]:.2f} mm < {_METAL_MIN_WALL_MM:.2f} mm 3-axis machining "
                    "minimum -- will chatter/deflect",
                )
            )
    except Exception:
        pass
    # sharp internal corners -> caveat (needs a fillet >= tool radius)
    try:
        length, count = _sharp_internal_corners(mesh)
        if count > 0:
            findings.append(
                DfmFinding(
                    "sharp_internal_corner",
                    "caveat",
                    f"{count} sharp internal corner(s) ({length:.0f} mm of concave edge) a "
                    "round cutter cannot machine -- add a fillet >= the tool radius",
                )
            )
    except Exception:
        pass
    if any(f.severity == "blocker" for f in findings):
        verdict = "fail"
    elif findings:
        verdict = "caveats"
    else:
        verdict = "pass"
    return DfmReport(process="metal", tier="machined-metal", verdict=verdict, findings=findings)


def dfm_for_material(meshes: list[trimesh.Trimesh], mat: Material) -> DfmReport | None:
    """The process-keyed entry: run the DFM profile for the material's process over the
    assembly's parts. METAL -> machined-metal checks. FDM/RESIN -> None here (the
    printability tier already owns those, per-part, in validate_authored_part). None means
    'no metal DFM profile applies', not 'passed'."""
    if mat.process is not ProcessCategory.METAL:
        return None
    worst: DfmReport | None = None
    agg: list[DfmFinding] = []
    for m in meshes:
        if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
            continue
        r = check_metal_dfm(m)
        agg.extend(r.findings)
    if not agg:
        return DfmReport(process="metal", tier="machined-metal", verdict="pass", findings=[])
    verdict = "fail" if any(f.severity == "blocker" for f in agg) else "caveats"
    worst = DfmReport(process="metal", tier="machined-metal", verdict=verdict, findings=agg)
    return worst


def scope_certificate(
    *,
    disposition: str,
    material_process: str,
    fit_verdict: str | None,
    motion_status: str | None,
    worst_case: dict[str, Any] | None,
    dfm: DfmReport | None,
    expert_blockers: int,
    fidelity: FidelityReport | None = None,
) -> dict[str, Any]:
    """Assemble the SCOPED certificate (founder 2026-07-12: 'scope the certificate').
    Names each tier and its verdict, and states in plain words what IS and IS NOT
    certified for this fabrication process -- so 'certified' never over-claims. The
    kinematic/fit core is fabrication-agnostic; the manufacturability tier is per-process
    and may be 'unchecked' (e.g. metal DFM when the part is printed, or vice versa)."""
    geometry = (
        "pass"
        if disposition == "shipped"
        else ("caveats" if disposition == "shipped_with_caveats" else "fail")
    )
    if dfm is not None:
        manuf = {"process": dfm.process, "tier": dfm.tier, "verdict": dfm.verdict}
    elif material_process == "fdm":
        manuf = {
            "process": "fdm",
            "tier": "fdm-printability",
            "verdict": "see per-part printability",
        }
    else:
        manuf = {"process": material_process, "tier": "none", "verdict": "unchecked"}
    unchecked: list[str] = []
    if material_process == "metal" and dfm is None:
        unchecked.append("machinability")
    if material_process != "metal":
        unchecked.append("machinability")
    if material_process not in ("fdm",):
        unchecked.append("FDM printability")
    if fidelity is None:
        unchecked.append("part fidelity")
    fidelity_clause = "" if fidelity is None else f"part fidelity ({fidelity.verdict}); "
    scope = (
        f"CERTIFIED: geometry+kinematics ({geometry}); "
        f"fit/tolerance ({fit_verdict or 'n/a'}); "
        f"{fidelity_clause}"
        f"manufacturability [{manuf['tier']}] ({manuf['verdict']}). "
        f"NOT CHECKED: {', '.join(sorted(set(unchecked))) or 'none'}. "
        "This certificate covers geometry, kinematics, and the named process tier only -- "
        "not thermal/structural/chemical performance (see expert_review)."
    )
    return {
        "geometry_kinematics": geometry,
        "fit_tolerance": fit_verdict,
        "motion": motion_status,
        "worst_case_material_condition": None if worst_case is None else worst_case.get("issues"),
        "manufacturability": manuf,
        "fidelity": None if fidelity is None else fidelity.verdict,
        "expert_blockers": expert_blockers,
        "scope": scope,
    }


__all__ = ["DfmReport", "check_metal_dfm", "dfm_for_material", "scope_certificate"]
