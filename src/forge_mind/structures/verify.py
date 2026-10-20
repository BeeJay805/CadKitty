"""C3: the uniform gate -- every author's artifact runs the SAME checks, per-check verdicts.

Authors propose; THIS disposes. Per-check verdicts (the ADR-0083 audit lesson: never one
``passed`` bool) over the deterministic core: watertight, positive volume/mass, the spec's OWN
load case vs its deflection bound (tri-state; a claim requires True), min-wall, and a real slice
(tri-state). A check whose prerequisite is absent degrades to an EXPLICIT "n/a", never silently.

``claimable`` is the strict conjunction: every deterministic check PASS and every tri-state
actually True (an unproven axis blocks the CLAIM but not the search -- the polarity split
ADR-0096 established). Full ``validate_authored_part`` (brief-semantic tiers) remains the agentic
path's gate; wiring these artifacts into it is recorded follow-up in ADR-0097.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import trimesh

from forge_mind.materials import resolve_material, strength_allowable
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.loadcase import describe_load_case, run_load_case
from forge_mind.structures.objectives import score_mesh
from forge_mind.structures.sections import print_gates
from forge_mind.structures.spec import RequirementSpec

# The static design factor for the strength screen (ADR-0107). A ductile-material static safety
# factor is 1.5-2.0 for well-known loads (Shigley, Mechanical Engineering Design); 2.0 is chosen
# for FDM material/process uncertainty. A conservative screen, not a certification.
STRESS_FOS_MIN = 2.0


def _stress_safety(lr: Any, material_id: str, measured: dict[str, Any]) -> str:
    """Yield-safety screen (ADR-0107): peak von Mises vs a CITED material allowable via a safety
    factor FoS = allowable / von_mises, pass iff FoS >= ``STRESS_FOS_MIN``. Closes the claim-level
    gap where "holds X kg" proved only stiffness (won't sag), never strength (won't yield) -- a
    stubby/notched part can pass deflection and still over-stress. Reuses the SAME solve as the
    deflection check (no added latency). Fail-closed tri-state: failed solve -> "unproven"; a
    material with no defensible allowable -> "n/a" + a VISIBLE basis note (strength simply not
    screened, never a guessed pass); a non-finite/non-positive stress -> "unproven"; else pass/fail."""
    if lr is None:
        return "unproven"
    allowable, basis = strength_allowable(resolve_material(material_id))
    if allowable is None:
        measured["stress_basis"] = (
            f"strength NOT screened: no defensible printed-strength basis for {material_id}"
        )
        return "n/a"
    vm = float(lr.max_von_mises_mpa)
    if not math.isfinite(vm) or vm <= 0.0:
        return "unproven"
    fos = allowable / vm
    measured["stress_allowable_mpa"] = allowable
    measured["stress_fos"] = round(fos, 2)
    measured["stress_basis"] = basis
    return "pass" if fos >= STRESS_FOS_MIN else "fail"


def verify_artifact(result: AuthorResult, spec: RequirementSpec) -> dict[str, Any]:
    """Run the uniform deterministic gate on one author's artifact against ITS OWN spec.

    Returns ``{"checks": {name: "pass"|"fail"|"unproven"|"n/a", ...}, "measured": {...},
    "claimable": bool}``. The load case and bound come from the SPEC (the single source for
    authoring and gating -- a candidate is never measured under a friendlier case than it was
    asked to meet)."""
    checks: dict[str, str] = {}
    measured: dict[str, Any] = {}

    if result.stl is None or not Path(result.stl).exists():
        return {"checks": {"artifact": "fail"}, "measured": {}, "claimable": False}
    checks["artifact"] = "pass"
    checks["step_present"] = (
        "pass" if (result.step is not None and Path(result.step).exists()) else "fail"
    )

    mesh: Any = trimesh.load(str(result.stl), force="mesh")
    part = score_mesh(mesh, material=spec.material_id, load_case=None, stl_label=str(result.stl))
    checks["watertight"] = "pass" if part.watertight else "fail"
    checks["positive_volume"] = "pass" if part.mass_g is not None else "fail"
    measured["mass_g"] = part.mass_g

    # Load-response checks share ONE FEA solve: STIFFNESS (won't sag past the bound) and STRENGTH
    # (won't yield -- ADR-0107). The case is echoed in HUMAN units on every report (Fable's
    # mitigation: a person can spot '50 kg' where '0.5 kg' was meant).
    if spec.load is None:
        checks["deflection_bound"] = "n/a"
        checks["stress_safety"] = "n/a"
    else:
        measured["load_case_echo"] = describe_load_case(spec.load)
        lr = run_load_case(mesh, spec.load)
        if lr is not None:
            measured["deflection_mm"] = lr.max_deflection_mm
            measured["von_mises_mpa"] = lr.max_von_mises_mpa
        # Stiffness vs the spec's OWN bound (tri-state; no bound claimed -> n/a).
        if spec.deflection_bound_mm is None:
            checks["deflection_bound"] = "n/a"
        elif lr is None:
            checks["deflection_bound"] = "unproven"
        else:
            checks["deflection_bound"] = (
                "pass" if lr.max_deflection_mm <= spec.deflection_bound_mm else "fail"
            )
        # Strength: peak stress vs a cited material allowable (reuses lr; fail-closed).
        checks["stress_safety"] = _stress_safety(lr, spec.material_id, measured)

    gates = print_gates(
        Path(result.stl), min_wall_required_mm=resolve_material(spec.material_id).min_wall_mm
    )
    measured["min_wall_mm"] = gates["min_wall_mm"]
    checks["min_wall"] = "pass" if gates["min_wall_pass"] else "fail"
    # tri-state: True/False from a real slice; None (no slicer installed) = unproven -- a
    # slicer-less box cannot CLAIM printability, but the search may continue.
    checks["slice"] = (
        "unproven" if gates["slice_ok"] is None else ("pass" if gates["slice_ok"] else "fail")
    )

    # Lighter-than-reference (only when the spec carries a reference).
    if spec.ref_mass_g is None:
        checks["lighter_than_reference"] = "n/a"
    elif part.mass_g is None:
        checks["lighter_than_reference"] = "unproven"
    else:
        checks["lighter_than_reference"] = "pass" if part.mass_g < spec.ref_mass_g else "fail"
        measured["mass_vs_reference_pct"] = round(100 * (part.mass_g / spec.ref_mass_g - 1), 1)

    claimable = all(v in ("pass", "n/a") for v in checks.values())
    return {"checks": checks, "measured": measured, "claimable": claimable}


def _minimal_brief(spec: RequirementSpec, *, part_name: str) -> Any:
    """Synthesize the minimal DesignBrief the full validator needs from a structures spec
    (Fable: deterministic authors synthesize the brief; brief-semantic checks that need more
    than this degrade to their own explicit N/A inside the validator)."""
    from forge_mind.agentic.brief import BriefFeature, DesignBrief

    length, depth, height = spec.envelope_mm
    mat = resolve_material(spec.material_id)
    return DesignBrief(
        object_name=part_name,
        summary=f"A physics-authored structural part ({part_name}).",
        visual_description=(
            "An engineered structural member: straight members/flanges with open cutouts, "
            "solid mounting regions at the interfaces."
        ),
        overall_envelope_mm=f"{length:.0f} x {depth:.0f} x {height:.0f} (spec envelope)",
        features=[
            BriefFeature(
                name="structure",
                purpose="carry the declared load between the interfaces",
                approx_dims_mm=f"span {length:.0f}, section up to {depth:.0f} x {height:.0f}",
                method="computed by the physics/topology author",
            )
        ],
        min_wall_mm=mat.min_wall_mm,
        material=mat.name,
        print_orientation_note="print on the largest flat face",
        target="cosmetic",  # one monolithic body (the structural authors emit single solids)
    )


def verify_full(
    result: AuthorResult, spec: RequirementSpec, *, part_name: str = "structural part"
) -> dict[str, Any]:
    """The uniform gate PLUS the engine's full authored-part validator (ADR-0098 follow-up wired).

    Runs ``verify_artifact`` first, then ``validate_authored_part`` behind a synthesized minimal
    brief and merges its verdicts as additional per-check entries (``full_*``). Fail-soft: if the
    full validator errors, the deterministic gate's verdicts stand and ``full_validator`` reads
    "unproven" -- the extra tier can add scrutiny, never silently remove it."""
    out = verify_artifact(result, spec)
    if result.stl is None or result.step is None:
        out["checks"]["full_validator"] = "n/a"
        return out
    try:
        from forge_mind.agentic.validate import validate_authored_part

        validation = validate_authored_part(
            Path(result.stl),
            Path(result.step),
            _minimal_brief(spec, part_name=part_name),
            material=resolve_material(spec.material_id),
        )
        out["checks"]["full_validator"] = "pass" if validation.passed else "fail"
        out["checks"]["full_hard_ok"] = "pass" if validation.hard_ok else "fail"
        out["measured"]["full_validator_issues"] = list(validation.issues)[:8]
    except Exception as exc:  # fail-soft: extra scrutiny only, never less
        out["checks"]["full_validator"] = "unproven"
        out["measured"]["full_validator_error"] = f"{type(exc).__name__}: {exc}"
    out["claimable"] = all(v in ("pass", "n/a") for v in out["checks"].values())
    return out


__all__ = ["verify_artifact", "verify_full"]
