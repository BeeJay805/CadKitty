"""Author self-check epilogue (ADR-0045 build agenda): the sandbox measures the authored
part against the brief's OWN targets immediately after the build -- bbox extents vs the
envelope, measured concave bores vs the planned hole diameters -- and writes the structured
deltas into ``result.json``. The loop hands those PRECISE numbers to the revision prompt as
the author's own measurements, so a miss is seen the moment it is built, before the
validator weighs in.

Honesty boundaries:
  - The self-check is FEEDBACK ONLY. It never gates ``passed``; the validator's measured
    gates (genus, size-aware bore audit, envelope conformance) remain the sole authority.
  - Measured in the SANDBOX CHILD on the live B-rep (the part object in hand), so the
    numbers describe exactly the artifact that was exported -- no re-load drift.
  - Best-effort by construction: an epilogue failure writes an ``error`` note and never
    fails the build.
  - OPT-IN via FORGE_AUTHOR_SELFCHECK (unset = deployed default; no spec rides job.json,
    the child runs no epilogue, prompts are byte-identical).
"""

from __future__ import annotations

import math
import os
from typing import Any

from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.spatial import _planned_hole_specs
from forge_mind.fabrication import FabricationMode
from forge_mind.materials import resolve_material
from forge_mind.validator.feature_checks import WALL_GATE_MIN_FRAC, wall_gate_triggered


def enabled() -> bool:
    """Is the self-check lever ON? (FORGE_AUTHOR_SELFCHECK=1; unset/0 = deployed default)."""
    return os.environ.get("FORGE_AUTHOR_SELFCHECK", "").strip().lower() in {"1", "true", "yes"}


def build_spec(
    brief: DesignBrief, fabrication_mode: FabricationMode | None = None
) -> dict[str, Any] | None:
    """The measurement targets the sandbox child checks the built part against, derived
    deterministically from the brief: the parsed envelope triple, the planned through-hole
    groups (count + target diameter where the brief states one), and the WALL FLOOR. None
    when the brief offers nothing measurable (no epilogue rides the job).

    The wall floor (2026-07-27, L102's largest kill class -- 5 of 8 H5B failures were
    sub-floor walls the author never measured until the validator failed it) is the
    EFFECTIVE floor the validator will apply: ``fabrication_mode.min_wall_override`` relaxes
    a declared wall to the structural-sanity floor on the GENERAL (machined, not-FDM) lane,
    so passing the declared value there would tell the author a valid part is about to fail
    (review-caught -- the refs bench runs GENERAL). Measured in the child with the GATE'S OWN
    function, against the GATE'S OWN floor: one number, one truth."""
    envelope = parse_envelope_mm(brief.overall_envelope_mm)
    holes = [
        {"label": s.label, "count": s.count, "d_mm": s.target_d_mm}
        for s in _planned_hole_specs(brief)
    ]
    if envelope is None and not holes:
        return None
    # The wall floor rides specs that ALREADY exist; it deliberately does NOT make a spec out
    # of nothing (review-caught): `min_wall_mm` is a required brief field, so keying spec
    # existence on it would make every brief truthy -- and `cadcoder_best_initial` treats a
    # truthy spec as "a perfect score of 4 is reachable" while `_score_draft` scores only
    # envelope + holes, so a wall-only spec would silently defeat its early exit and force
    # all N generator calls. Envelope/holes remain the existence test; wall is an extra field.
    material = resolve_material(brief.material)
    mode = fabrication_mode or FabricationMode.PRINT_3D
    floor = mode.effective_min_wall(brief.min_wall_mm, material.min_wall_mm)
    return {
        "envelope": list(envelope) if envelope is not None else None,
        "holes": holes,
        "wall_floor_mm": float(floor) if floor else None,
        "wall_min_area_mm2": material.min_feature_mm**2,
    }


def _pct_below_threshold(frac: float) -> str:
    """Render a SUB-threshold area fraction so the digits can never read as the threshold
    itself (review-caught: 2.9991% printed '3.0%' beside the words 'BELOW the 3% threshold').
    Floor-rounded, so the shown value is always <= the measured one, hence always strictly
    under the cutoff; a fraction that floors away entirely is reported as a bound."""
    shown = math.floor(frac * 10000) / 100.0
    return "<0.01%" if shown <= 0 else f"{shown:.2f}%"


def scoring_spec(spec: dict[str, Any] | None) -> dict[str, Any] | None:
    """The same spec with the WALL target dropped -- for the CAD-Coder best-of-N DRAFT scorer.

    Review-caught cost defect: `_score_draft` reads envelope + bores only, so carrying the
    wall floor into N draft builds would pay a 4096-ray STL measurement per draft that
    nothing reads (latency + timeout risk on complex meshes). The author-feedback path keeps
    the wall; scoring does not. Copies (never mutates the caller's spec)."""
    if spec is None:
        return None
    return {**spec, "wall_floor_mm": None}


def feedback_block(selfcheck: dict[str, Any] | None) -> str:
    """Render the child's measured deltas as the author-facing block ('' when there is
    nothing to say). The numbers are presented as the author's OWN measurements -- the
    tightest feedback loop there is."""
    if not selfcheck:
        return ""
    lines: list[str] = []
    env = selfcheck.get("envelope")
    if isinstance(env, dict):
        built = env.get("built_sorted") or []
        target = env.get("target_sorted") or []
        deltas = env.get("delta_mm") or []
        if built and target and deltas:
            built_s = " x ".join(f"{b:.0f}" for b in built)
            target_s = " x ".join(f"{t:.0f}" for t in target)
            delta_s = ", ".join(f"{d:+.1f}" for d in deltas)
            lines.append(
                f"envelope: built {built_s} mm vs brief {target_s} mm (deltas {delta_s} mm, "
                "sorted dims)"
            )
    for h in selfcheck.get("holes") or []:
        want = h.get("want")
        matched = h.get("matched")
        d = h.get("d_mm")
        if want is None or matched is None:
            continue
        size = f" Ø{d:.2f}mm" if isinstance(d, (int, float)) else ""
        verdict = "ALL PRESENT" if matched >= want else f"only {matched} of {want} present"
        lines.append(f"'{h.get('label', 'holes')}': {want} planned{size} bore(s) -- {verdict}")
    wall = selfcheck.get("wall")
    if isinstance(wall, dict) and wall.get("floor_mm") is not None:
        floor = float(wall["floor_mm"])
        frac = wall.get("thin_fraction")
        thinnest = wall.get("thinnest_mm")
        thin_area = wall.get("thin_area_mm2")
        thin_count = wall.get("thin_sample_count")
        min_area = wall.get("min_area_mm2")
        if isinstance(frac, (int, float)) and frac > 0 and isinstance(thinnest, (int, float)):
            # The wording is decided by the SHARED gate threshold, not by the rendered digits
            # (review-caught: 0.0286 printed as "3%" and claimed a failure the gate passes).
            # Sub-threshold walls are reported as measured-but-accepted, with enough precision
            # that the number cannot read as the threshold itself.
            gates = wall_gate_triggered(
                float(frac),
                float(thin_area) if isinstance(thin_area, (int, float)) else 0.0,
                int(thin_count) if isinstance(thin_count, int) else 0,
                float(min_area) if isinstance(min_area, (int, float)) else math.inf,
            )
            if gates:
                lines.append(
                    f"WALL: {frac * 100:.1f}% of your surface measures a WALL thinner than the "
                    f"{floor:.2f} mm floor (thinnest {thinnest:.2f} mm, "
                    f"{float(thin_area or 0):.2f} mm^2 thin area) -- the shared "
                    "fraction/area criterion says this FAILS the wall "
                    "gate. THICKEN that cluster; do not thin the rest of the part."
                )
            else:
                lines.append(
                    f"WALL: {_pct_below_threshold(frac)} of your surface is thinner than the "
                    f"{floor:.2f} mm floor (thinnest {thinnest:.2f} mm) -- BELOW the "
                    f"{WALL_GATE_MIN_FRAC * 100:.0f}% fraction threshold and the minimum "
                    "physical area, so the wall gate "
                    "ACCEPTS this; leave it alone unless another finding names it."
                )
        elif isinstance(frac, (int, float)):
            lines.append(f"WALL: no sub-floor wall measured (floor {floor:.2f} mm) -- clear.")
    bores = selfcheck.get("bores")
    if isinstance(bores, list) and bores:
        listed = "; ".join(
            f"Ø{b['d_mm']:.2f} at ({b['at'][0]:.0f}, {b['at'][1]:.0f}, {b['at'][2]:.0f})"
            for b in bores[:8]
            if isinstance(b, dict) and "d_mm" in b and "at" in b
        )
        if listed:
            lines.append(f"measured concave bores: {listed}")
    elif selfcheck.get("holes"):
        lines.append("measured concave bores: NONE (any cylinders present are solid pins/bosses)")
    if not lines:
        return ""
    body = "\n".join(f"  - {ln}" for ln in lines)
    return (
        "YOUR OWN MEASUREMENTS (sandbox self-check of the part you JUST built):\n"
        f"{body}\n"
        "Fix the shortfalls these numbers name FIRST; they are measured from your own "
        "geometry, not a reviewer's opinion."
    )


__all__ = ["build_spec", "enabled", "feedback_block", "scoring_spec"]
