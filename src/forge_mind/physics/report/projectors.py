"""Projectors: turn an already-finished validator ``Check`` (and the reported stability) into contract
``Finding`` objects. PURE and READ-ONLY — they never load a mesh, run a check, or gate anything. The
measured/limit numbers are parsed from each check's OWN ``actual``/``expected`` strings (the check is the
source of truth), so a finding can never claim a value the validator did not compute. Anything that did
not run is ``not_run`` with a reason — never a neutral pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from forge_mind.types import Vec3

if TYPE_CHECKING:
    from forge_mind.physics.fea import LoadResponse

from forge_mind.physics.report.holes import (
    MIN_EDGE_DISTANCE_RATIO,
    MIN_PRINTABLE_HOLE_MM,
    DetectedHole,
    HoleClass,
    classify_hole,
)
from forge_mind.physics.report.mates import evaluate_mate
from forge_mind.physics.report.schema import (
    Comparator,
    Confidence,
    Finding,
    Fix,
    FixChange,
    Limit,
    MassG,
    Mate,
    Measured,
    PhysicsLocator,
    PhysicsSeverity,
    PhysicsStatus,
    PhysicsTier,
    Vitals,
)
from forge_mind.validator.intrinsic import StabilityView
from forge_mind.validator.report import Check, CheckStatus, Locator, PhysicsMetrics, StepProof

# Validator CheckStatus -> contract Finding status (typed so mypy keeps the Literal).
_STATUS_MAP: dict[CheckStatus, PhysicsStatus] = {
    CheckStatus.PASS: "pass",
    CheckStatus.FAIL: "fail",
    CheckStatus.WARN: "warn",
    CheckStatus.INFO: "info",
}

_ValueKind = Literal["float", "int", "str"]


@dataclass(frozen=True)
class _Spec:
    """Static, honest display metadata for a known intrinsic check id. ``severity``/``tier`` are FIXED
    properties of the check KIND (not re-derived from the verdict); the value/limit are still parsed from
    the live check, so the numbers always match what actually ran."""

    suite: str
    label: str
    severity: PhysicsSeverity
    tier: PhysicsTier
    method: str
    unit: str
    comparator: Comparator
    source: str
    value_kind: _ValueKind


# The checks the intrinsic (upload) path produces. Anything not here is projected generically (never
# dropped) — see ``_generic_spec``.
_SPECS: dict[str, _Spec] = {
    "mesh.watertight": _Spec(
        "printproof",
        "Watertight",
        "blocking",
        "verified",
        "mesh topology",
        "",
        "==",
        "manifold mesh (no boundary edges)",
        "str",
    ),
    "mesh.is_volume": _Spec(
        "printproof",
        "Solid volume",
        "blocking",
        "verified",
        "mesh topology",
        "",
        "==",
        "encloses a watertight solid",
        "str",
    ),
    "mesh.winding_consistent": _Spec(
        "printproof",
        "Consistent winding",
        "blocking",
        "verified",
        "mesh topology",
        "",
        "==",
        "consistent face winding",
        "str",
    ),
    "mesh.single_body": _Spec(
        "printproof",
        "Single body",
        "blocking",
        "verified",
        "connected-component count",
        "bodies",
        "==",
        "one connected shell",
        "int",
    ),
    "mesh.positive_volume": _Spec(
        "printproof",
        "Positive volume",
        "blocking",
        "verified",
        "mesh volume",
        "mm^3",
        ">=",
        "a positive solid volume",
        "float",
    ),
    "mesh.min_printable_thickness": _Spec(
        "printproof",
        "Minimum printable thickness",
        "blocking",
        "computed",
        "bounding-box smallest extent",
        "mm",
        ">=",
        "one FDM layer (0.1 mm floor)",
        "float",
    ),
    "features.min_wall_thickness": _Spec(
        "printproof",
        "Minimum wall",
        "warning",
        "computed",
        "ray-cast wall thickness (sampled)",
        "mm",
        ">=",
        "intrinsic printable floor (~2 perimeters @ 0.4 mm nozzle)",
        "float",
    ),
    "printability.overhang_max_angle": _Spec(
        "printproof",
        "Overhangs",
        "warning",
        "computed",
        "per-face normal vs build axis",
        "deg from vertical",
        "<=",
        "FDM unsupported overhang limit",
        "float",
    ),
    "mesh.loadable": _Spec(
        "printproof",
        "Readable mesh",
        "blocking",
        "verified",
        "mesh load",
        "",
        "==",
        "a single readable mesh",
        "str",
    ),
    "intrinsic.unreadable": _Spec(
        "printproof",
        "Readable model",
        "blocking",
        "verified",
        "mesh load",
        "",
        "==",
        "a readable model",
        "str",
    ),
}

# Validator Tier prefix -> suite, for any check not in _SPECS (defensive; keeps every check projected).
_TIER_SUITE = {"mesh": "printproof", "features": "printproof", "printability": "printproof"}


def _lead_number(s: str | None) -> float | None:
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", s)
    return float(m.group()) if m else None


def _coerce(num: float, kind: _ValueKind) -> int | float:
    return round(num) if kind == "int" else round(num, 3)


def _measured(check: Check, spec: _Spec) -> Measured | None:
    if spec.value_kind == "str":
        return Measured(value=check.actual or "", unit=spec.unit)
    num = _lead_number(check.actual)
    if num is None:  # e.g. "unmeasurable" — honest absence, not a fabricated value
        return None
    return Measured(value=_coerce(num, spec.value_kind), unit=spec.unit)


def _limit(check: Check, spec: _Spec) -> Limit | None:
    if spec.value_kind == "str":
        return Limit(
            value=check.expected or "",
            unit=spec.unit,
            comparator=spec.comparator,
            source=spec.source,
        )
    num = _lead_number(check.expected)
    if num is None:
        return None
    return Limit(
        value=_coerce(num, spec.value_kind),
        unit=spec.unit,
        comparator=spec.comparator,
        source=spec.source,
    )


def _locator(loc: Locator | None) -> PhysicsLocator | None:
    if loc is None:
        return None
    return PhysicsLocator(
        frame=loc.frame,
        point=loc.point,
        bbox_min=loc.bbox_min,
        bbox_max=loc.bbox_max,
        lens=loc.lens,
        part_id=None,
    )


def _fix(check: Check, spec: _Spec, measured: Measured | None, limit: Limit | None) -> Fix | None:
    """An honest, concrete fix for a FAILing known check. None for pass/info/warn-without-an-obvious-fix.
    Only synthesised for the cases where the fix is unambiguous from geometry."""
    if check.status != CheckStatus.FAIL:
        return None
    if check.id == "features.min_wall_thickness" and measured is not None and limit is not None:
        return Fix(
            text=f"Thicken the thinnest wall to at least {limit.value} {limit.unit}.",
            # `from` is a Python keyword; build via the alias to keep the JSON key correct.
            change=FixChange.model_validate(
                {"target": "min_wall_thickness", "from": measured.value, "to": limit.value}
            ),
        )
    if check.id == "printability.overhang_max_angle":
        return Fix(
            text="Reorient the part, add support, or chamfer this face so the overhang is within limit.",
        )
    if check.id == "mesh.watertight":
        return Fix(text="Repair the mesh into a closed solid (no boundary edges) before printing.")
    if check.id == "mesh.single_body" and measured is not None:
        return Fix(text=f"Merge or remove the {measured.value} disconnected bodies into one solid.")
    return None


def project_check(check: Check) -> Finding:
    """Project ONE validator Check to a contract Finding. Status carried through faithfully; severity/tier
    are the check kind's fixed honest classification (never a re-derivation of the verdict)."""
    spec = _SPECS.get(check.id)
    if spec is None:
        # Defensive generic projection — never silently drop a check we didn't enumerate.
        suite = _TIER_SUITE.get(check.tier.value, "printproof")
        spec = _Spec(
            suite=suite,
            label=check.id.rsplit(".", 1)[-1].replace("_", " ").title(),
            severity="blocking" if check.status == CheckStatus.FAIL else "advisory",
            tier="computed",
            method="validator check",
            unit="",
            comparator="==",
            source="validator",
            value_kind="str",
        )
    measured = _measured(check, spec)
    limit = _limit(check, spec)
    return Finding(
        id=f"{spec.suite}.{check.id}",
        suite=spec.suite,
        label=spec.label,
        status=_STATUS_MAP[check.status],
        severity=spec.severity,
        measured=measured,
        limit=limit,
        locator=_locator(check.locator),
        explanation=check.message,
        fix=_fix(check, spec, measured, limit),
        confidence=Confidence(tier=spec.tier, method=spec.method),
    )


def project_printproof(checks: list[Check]) -> list[Finding]:
    """Every intrinsic check routes to PrintProof in slice 1 (mesh/feature/printability tiers)."""
    return [project_check(c) for c in checks]


def _stability_finding(stability: StabilityView | None) -> Finding:
    if stability is None:
        return Finding(
            id="loadproof.stability",
            suite="loadproof",
            label="Stand-or-tip balance",
            status="not_run",
            severity="advisory",
            explanation="No clear flat base was found to rest the part on, so balance was not assessed.",
            confidence=Confidence(tier="advisory", method="not run: no resting base"),
            reason_not_run="No clear flat base to rest on.",
        )
    status: PhysicsStatus = "pass" if stability.stable else "warn"
    if stability.stable:
        explanation = (
            f"Resting on its base the part is stable: it would take a {stability.tip_angle_deg:.0f} "
            f"degree tilt to tip, with the center of mass {stability.margin_mm:.1f} mm inside the "
            "footprint edge."
        )
        fix = None
    else:
        explanation = (
            f"The part is top-heavy on its base: it tips at only {stability.tip_angle_deg:.0f} degrees "
            "and the center of mass is near (or past) the footprint edge."
        )
        fix = Fix(
            text="Widen the base or lower the center of mass so it stands without tipping.",
        )
    return Finding(
        id="loadproof.stability",
        suite="loadproof",
        label="Stand-or-tip balance",
        status=status,
        severity="advisory",  # geometry-only + orientation-dependent: REPORTED, never gates the verdict
        measured=Measured(value=round(stability.tip_angle_deg, 1), unit="deg to tip"),
        limit=None,
        locator=PhysicsLocator(point=stability.com_mm, lens="balance"),
        explanation=explanation,
        fix=fix,
        confidence=Confidence(tier="computed", method="center of mass vs base polygon"),
    )


def _fea_not_run_finding(load_requested: bool) -> Finding:
    # ADR-0058/0064: the illustrative FEA field never enters a `measured` slot, and a real-load solve runs
    # only when a load is stated AND the solver is validated. Either gap -> not_run (never a fake number).
    if load_requested:
        reason = (
            "A load was given but the validated mesh-FEA solve did not run (the FEA solver is not "
            "available/validated on this server, or the mesh could not be solved)."
        )
    else:
        reason = "No load specified — a mesh FEA solve needs a stated load."
    return Finding(
        id="loadproof.fea",
        suite="loadproof",
        label="Stress & deflection under load",
        status="not_run",
        severity="advisory",
        explanation="A mesh FEA solve estimates how much the part deflects and where it is most stressed "
        "under a stated load.",
        confidence=Confidence(tier="advisory", method="mesh FEA (not run)"),
        reason_not_run=reason,
    )


def _deflection_finding(lr: LoadResponse) -> Finding:
    return Finding(
        id="loadproof.deflection",
        suite="loadproof",
        label="Deflection under load",
        status="info",
        severity="advisory",  # a measured number for the user to judge; no universal serviceability limit
        measured=Measured(value=lr.max_deflection_mm, unit="mm"),
        limit=None,
        locator=PhysicsLocator(
            point=Vec3(
                x=lr.deflection_at_mm[0], y=lr.deflection_at_mm[1], z=lr.deflection_at_mm[2]
            ),
            lens="load",
        ),
        explanation=(
            f"Under the stated {lr.load_n:.0f} N the part deflects {lr.max_deflection_mm} mm at this "
            f"point ({lr.load_case})."
        ),
        fix=None,
        confidence=Confidence(tier="computed", method="P2 mesh FEA (validated vs closed-form)"),
    )


def _load_stress_finding(lr: LoadResponse) -> Finding:
    return Finding(
        id="loadproof.stress",
        suite="loadproof",
        label="Peak stress under load",
        status="info",
        severity="advisory",  # no certifiable FDM yield -> reported, never an auto 'safe/fail' verdict
        measured=Measured(value=lr.max_von_mises_mpa, unit="MPa"),
        limit=None,
        locator=PhysicsLocator(
            point=Vec3(x=lr.stress_at_mm[0], y=lr.stress_at_mm[1], z=lr.stress_at_mm[2]),
            lens="stress",
        ),
        explanation=(
            f"Peak stress {lr.max_von_mises_mpa} MPa at this point under {lr.load_n:.0f} N. {lr.note}"
        ),
        fix=None,
        confidence=Confidence(tier="computed", method="P2 mesh FEA (isotropic, validated)"),
    )


def project_loadproof(
    stability: StabilityView | None,
    load_response: LoadResponse | None = None,
    *,
    load_requested: bool = False,
) -> list[Finding]:
    """LoadProof: the geometry-only stand-or-tip finding (real, computed) + the load response. With a
    stated load AND a validated solve, emit absolute deflection + peak stress (located, P2-validated); else
    not_run (fail closed — never a fabricated load number)."""
    findings = [_stability_finding(stability)]
    if load_response is not None:
        findings.append(_deflection_finding(load_response))
        findings.append(_load_stress_finding(load_response))
    else:
        findings.append(_fea_not_run_finding(load_requested))
    return findings


def _fitproof_not_run(reason: str, mate_label: str = "Fit & assembly") -> Finding:
    return Finding(
        id="fitproof.intent",
        suite="fitproof",
        label=mate_label,
        status="not_run",
        severity="advisory",
        explanation=(
            "Fit checks (ISO 273 clearance, ISO 286 shaft/bearing fit) run once you say what this part "
            "mates with and we can measure the matching hole."
        ),
        confidence=Confidence(tier="advisory", method="ISO 273 / ISO 286 (not run)"),
        reason_not_run=reason,
    )


def project_fitproof(mates: list[Mate], holes: list[DetectedHole]) -> list[Finding]:
    """Decision-grade fit: each stated mate vs the nearest measured hole (ISO 273 bolts / ISO 286 shafts /
    bearing-seat). No mate -> not_run. A mate with no parseable spec or no matching hole -> a per-mate
    not_run (fail closed; never a fabricated pass)."""
    if not mates:
        return [_fitproof_not_run("No mating part or interface specified.")]
    findings: list[Finding] = []
    for i, mate in enumerate(mates):
        ev = evaluate_mate(mate.kind, mate.spec, holes)
        if ev is None:
            reason = (
                "Hole geometry needs a STEP upload (an STL has no feature data)."
                if not holes
                else f"No hole near the {mate.kind} '{mate.spec}', or the spec could not be parsed."
            )
            findings.append(_fitproof_not_run(reason, mate_label=f"{mate.kind} fit"))
            continue
        severity: PhysicsSeverity = "blocking" if ev.status == "fail" else "warning"
        findings.append(
            Finding(
                id=f"fitproof.mate_{i}_{ev.label.split()[0].lower().replace('.', '_')}",
                suite="fitproof",
                label=ev.label,
                status=ev.status,
                severity=severity if ev.status in {"fail", "warn"} else "advisory",
                measured=Measured(value=round(ev.measured_mm, 3), unit="mm dia"),
                limit=Limit(
                    value=round(ev.limit_mm, 3),
                    unit="mm dia",
                    comparator=ev.comparator,
                    source=ev.source,
                ),
                locator=PhysicsLocator(point=ev.hole.center_mm, lens="interference"),
                explanation=ev.explanation,
                fix=Fix(text=ev.fix_text) if ev.fix_text else None,
                confidence=Confidence(tier="verified", method=ev.method),
            )
        )
    return findings


def _boltbrain_not_run(has_brep: bool) -> Finding:
    """No holes to reason about. On a STEP we DID look (and found none); on an STL we can't look."""
    if has_brep:
        explanation = "No cylindrical fastener holes were detected in this STEP."
        reason = "No fastener holes detected in the B-rep."
    else:
        explanation = (
            "Fastener checks (clearance vs tapped, edge distance, thread engagement) need the part's "
            "hole geometry, which an STL upload does not carry."
        )
        reason = "Hole geometry needs a STEP upload (an STL has no feature data)."
    return Finding(
        id="boltbrain.holes",
        suite="boltbrain",
        label="Fastener intelligence",
        status="not_run",
        severity="advisory",
        explanation=explanation,
        confidence=Confidence(tier="advisory", method="hole detection (not run)"),
        reason_not_run=reason,
    )


def project_boltbrain(holes: list[DetectedHole], *, has_brep: bool) -> list[Finding]:
    """Identify each detected bore against the metric fastener registry (ISO 273 clearance / ISO 261 tap
    drill). Decision-grade IDENTIFICATION (a deterministic registry match), located on the hole. It is
    INFO, not pass/fail: without intent we report what the hardware IS, not whether it's the RIGHT size
    (that arrives with the mate spec). No holes -> not_run (we looked / can't look)."""
    if not holes:
        return [_boltbrain_not_run(has_brep)]

    # Group by classification so 4 identical M3 holes read as one finding, located on a representative.
    groups: dict[str, list[DetectedHole]] = {}
    classes: dict[str, HoleClass | None] = {}
    for h in holes:
        cls = classify_hole(h.diameter_mm)
        key = f"{cls.size} {cls.kind}" if cls is not None else "non-standard"
        groups.setdefault(key, []).append(h)
        classes[key] = cls

    findings: list[Finding] = []
    for key in sorted(groups):
        hs = groups[key]
        cls = classes[key]
        rep = hs[0]
        dia = round(sum(h.diameter_mm for h in hs) / len(hs), 2)
        n = len(hs)
        plural = "s" if n > 1 else ""
        loc = PhysicsLocator(point=rep.center_mm, lens="hole")
        if cls is not None:
            findings.append(
                Finding(
                    id=f"boltbrain.holes_{cls.size.lower().replace('.', '_')}_{cls.kind.split()[0]}",
                    suite="boltbrain",
                    label=f"{cls.size} {cls.kind}",
                    status="info",
                    severity="advisory",
                    measured=Measured(value=dia, unit="mm dia"),
                    limit=Limit(
                        value=cls.nominal_value_mm,
                        unit="mm dia",
                        comparator="==",
                        source=f"ISO 273/261 {cls.size} {cls.kind}",
                    ),
                    locator=loc,
                    explanation=(
                        f"{n} hole{plural} of {dia} mm diameter — matches {cls.size} {cls.kind}."
                    ),
                    fix=None,
                    confidence=Confidence(
                        tier="verified", method="bore diameter vs fastener registry"
                    ),
                )
            )
        else:
            findings.append(
                Finding(
                    id=f"boltbrain.holes_nonstandard_{round(dia * 10)}",
                    suite="boltbrain",
                    label="Non-standard hole",
                    status="info",
                    severity="advisory",
                    measured=Measured(value=dia, unit="mm dia"),
                    limit=None,
                    locator=loc,
                    explanation=(
                        f"{n} hole{plural} of {dia} mm diameter — matches no standard metric fastener "
                        "(M2-M8)."
                    ),
                    fix=Fix(
                        text="If this is a fastener hole, resize it to a standard clearance or tap-drill "
                        f"diameter near {dia} mm."
                    ),
                    confidence=Confidence(
                        tier="verified", method="bore diameter vs fastener registry"
                    ),
                )
            )

    findings.append(_edge_distance_finding(holes))
    smallest = min(holes, key=lambda h: h.diameter_mm)
    if smallest.diameter_mm < MIN_PRINTABLE_HOLE_MM:
        findings.append(_min_hole_finding(smallest))
    return findings


def _edge_distance_finding(holes: list[DetectedHole]) -> Finding:
    """Edge distance for the WORST (thinnest-walled) hole. center-to-edge / diameter must be >= 1.5.
    not_run if the surrounding wall could not be measured (fail closed — never a silent pass)."""
    walled = [h for h in holes if h.wall_mm is not None]
    if not walled:
        return Finding(
            id="boltbrain.edge_distance",
            suite="boltbrain",
            label="Edge distance",
            status="not_run",
            severity="advisory",
            explanation="The material around the holes could not be measured for an edge-distance check.",
            confidence=Confidence(
                tier="advisory", method="radial wall ray-cast (no clean measurement)"
            ),
            reason_not_run="Could not measure the wall around the detected holes.",
        )
    worst = min(walled, key=lambda h: (h.wall_mm or 0.0) / h.diameter_mm)
    wall = worst.wall_mm or 0.0
    ratio = round(0.5 + wall / worst.diameter_mm, 2)  # (radius + wall) / diameter
    ok = ratio >= MIN_EDGE_DISTANCE_RATIO
    return Finding(
        id="boltbrain.edge_distance",
        suite="boltbrain",
        label="Edge distance",
        status="pass" if ok else "warn",
        severity="advisory" if ok else "warning",
        measured=Measured(value=ratio, unit="x diameter"),
        limit=Limit(
            value=MIN_EDGE_DISTANCE_RATIO,
            unit="x diameter",
            comparator=">=",
            source="fastener edge-distance rule (center >= 1.5x diameter from the edge)",
        ),
        locator=PhysicsLocator(point=worst.center_mm, lens="edge"),
        explanation=(
            f"The closest hole sits {ratio}x its diameter from the part edge"
            + (
                ", clear of the edge."
                if ok
                else f" — under {MIN_EDGE_DISTANCE_RATIO}x, so it can blow out under load or torque."
            )
        ),
        fix=None
        if ok
        else Fix(
            text="Move this hole inboard or add a boss so its center is at least 1.5x its diameter from "
            "the edge."
        ),
        confidence=Confidence(tier="computed", method="radial wall ray-cast"),
    )


def _min_hole_finding(hole: DetectedHole) -> Finding:
    return Finding(
        id="boltbrain.min_hole",
        suite="boltbrain",
        label="Small hole printability",
        status="warn",
        severity="warning",
        measured=Measured(value=round(hole.diameter_mm, 2), unit="mm dia"),
        limit=Limit(
            value=MIN_PRINTABLE_HOLE_MM,
            unit="mm dia",
            comparator=">=",
            source="FDM small-hole limit (~0.4 mm nozzle)",
        ),
        locator=PhysicsLocator(point=hole.center_mm, lens="hole"),
        explanation=(
            f"A {round(hole.diameter_mm, 2)} mm hole prints unreliably on FDM — it can close up or need "
            "drilling to size."
        ),
        fix=Fix(
            text="Enlarge the hole to >= 2 mm, or plan to drill/ream it to size after printing."
        ),
        confidence=Confidence(tier="computed", method="bore diameter vs FDM nozzle floor"),
    )


def project_step_integrity(step_proof: StepProof) -> Finding:
    """Is the uploaded STEP genuine editable CAD (an analytic B-rep solid), or a tessellated lump? A
    deterministic census of the ISO 10303-21 file (reuses the StepProof evidence). Advisory: it does not
    block printing — it gates whether feature-level (fit/fastener) checks can ever run on this upload."""
    is_brep = step_proof.is_brep is True and step_proof.tessellated is not True
    schema = step_proof.schema_id or "STEP"
    faces = step_proof.face_count
    measured: Measured
    fix: Fix | None
    status: PhysicsStatus
    if is_brep:
        status = "pass"
        measured = (
            Measured(value=faces, unit="analytic faces")
            if faces is not None
            else Measured(value="B-rep solid", unit="")
        )
        explanation = (
            "The upload is genuine editable CAD: a "
            + schema
            + " analytic B-rep solid"
            + (f" with {faces} faces." if faces is not None else ".")
        )
        fix = None
    else:
        status = "warn"
        measured = Measured(value="tessellated", unit="")
        explanation = (
            "This STEP carries tessellated/triangulated geometry, not an analytic B-rep — it has no "
            "editable features or hole data for fit/fastener checks."
        )
        fix = Fix(
            text="Re-export from your CAD tool as an analytic B-rep STEP (AP214/AP242), not a mesh STEP."
        )
    return Finding(
        id="printproof.step_integrity",
        suite="printproof",
        label="Editable CAD (B-rep)",
        status=status,
        severity="advisory",
        measured=measured,
        limit=Limit(
            value="analytic B-rep", unit="", comparator="==", source="ISO 10303 (AP214/AP242)"
        ),
        locator=None,
        explanation=explanation,
        fix=fix,
        confidence=Confidence(tier="verified", method="STEP text census (ISO 10303-21)"),
    )


def project_vitals(
    metrics: PhysicsMetrics | None,
    surface_area_mm2: float | None,
) -> Vitals:
    """Vitals from the Z-up validator metrics (CoM is in the same stl_z_up frame as every locator) +
    the surface area (not carried on PhysicsMetrics). Slicer-only quantities stay None."""
    if metrics is None:
        return Vitals(surface_area_mm2=surface_area_mm2)
    return Vitals(
        volume_mm3=round(metrics.volume_mm3, 3),
        mass_g=MassG(material=metrics.material, grams=round(metrics.mass_g, 3)),
        surface_area_mm2=round(surface_area_mm2, 3) if surface_area_mm2 is not None else None,
        center_of_mass_mm=metrics.center_of_mass,
    )


__all__ = [
    "project_boltbrain",
    "project_check",
    "project_fitproof",
    "project_loadproof",
    "project_printproof",
    "project_step_integrity",
    "project_vitals",
]
