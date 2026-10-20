"""Render SpecBrief spec_dims into the author prompt (the dead-carrier fix, L102).

The measured gap (2026-07-27 sprint, W2D): the SpecBrief pipeline PRODUCES per-feature typed
dims — H5B's upper_arm_r carried {"role": "thickness", "value_mm": 2.8, "of": "shell wall"}
on the exact lofted feature that shipped 0.36-0.64mm walls — but `_brief_to_user`
(author.py) interpolates only the prose `approx_dims_mm`, never `spec_dims`. Five of the
eight H5B failures were thin-wall kills on features whose typed floor existed and was never
shown to the author. Structure binds; prose does not (ADR-0129, 6-for-6 measured).

This module is the pure renderer; the two-line author.py wiring lands flag-gated
(FORGE_SPEC_DIM_PROMPT, default OFF pending its pre-registered A/B — W2D experiment 4).
Nothing imports this module until that wiring lands (introduced dark by construction).

KNOWN LIMITATION (stated, not patched): the agreement guard proves the VALUE traces to the
feature's prose, not that its ROLE tag is right — a researcher mis-tagging a 60mm length as
thickness=60 would promote as a hard wall instruction when 60 appears anywhere in the prose.
Role-correctness cannot be checked by prose proximity without rebuilding the drift-prone
prose-instrument class ADR-0130 retired; it is the emission contract's job, and the A/B that
arms this flag carries a guardrail metric for exactly this case (a HARD line contradicting
the realized feature class counts against the arm).
"""

from __future__ import annotations

import os
from typing import Any

from forge_mind.agentic.brief import spec_dim_agrees


def spec_dim_prompt_enabled() -> bool:
    return os.environ.get("FORGE_SPEC_DIM_PROMPT", "").strip().lower() in ("1", "true", "yes")


def material_prompt_enabled() -> bool:
    return os.environ.get("FORGE_MATERIAL_PROMPT", "").strip().lower() in ("1", "true", "yes")


def material_line(material: str | None) -> str | None:
    """The brief's MATERIAL and its density, for the author prompt; None when there is nothing
    honest to say (L103's dominant mass cause).

    Measured 2026-07-27: `brief.material` is read by the VALIDATOR only (validate.py) and has
    never reached the author, so a part authored against a stated target mass is sized at a
    density the author privately assumed -- five refs-bench parts missed reference mass by
    EXACTLY the steel/aluminium ratio (2.89x).

    The mass half now has a typed home too: `DesignBrief.target_mass_g` (added 2026-07-27) plus
    `target_volume_line` below. Before it, a stated mass survived research only if the researcher
    happened to keep it in prose -- the two archived refs-bench briefs show both outcomes:
    ppp0103 kept "mass reaches 96.13 g" AND a density and scored -0%; ppp0106 kept neither and
    scored +187%.

    KNOWN BOUNDARIES (review-caught, deliberately not papered over): the CAD-Coder lane
    (`FORGE_GENERATOR=cadcoder`, dark) authors from the raw prompt, not the brief, so it never
    sees these lines. And `produce_functional_part` hands `decompose_functional` the prompt +
    visual_description rather than the whole brief, so a MULTI-PART object's per-part cards do
    not inherit `target_mass_g` -- allocating a whole-object mass across parts is a real design
    question (which part owns how much?), not a plumbing gap, so the carrier is single-part-lane
    only until that is answered.

    The density is stated ONLY when the string
    resolves to a real registry entry: `resolve_material` falls back to PLA for anything
    unknown, and telling the author "PLA, 1.24 g/cm3" for a 6061 bracket would be a confident
    lie -- worse than the silence it replaces."""
    name = (material or "").strip()
    if not name:
        return None
    from forge_mind.materials import UnknownMaterialError, get_material

    try:
        mat = get_material(name)
    except UnknownMaterialError:
        return f"Material: {name} (density not in the registry -- do NOT assume one)."
    return (
        f"Material: {name} -- density {mat.density_g_cm3:g} g/cm^3. Size the part for THIS "
        "material: any stated target mass is mass at this density, so volume = mass / density."
    )


def target_volume_line(target_mass_g: float | None, material: str | None) -> str | None:
    """A stated target mass restated as a VOLUME target at the material's density.

    The other half of L103's mass leak (the first being that the author never saw the density
    at all). A volume target is what the author can actually build to and what the kernel
    measures directly, so an author that hits it hits the mass no matter what it privately
    assumes. None when there is nothing honest to say: no stated mass, no material, or a
    material whose density the registry does not know (a guessed density would produce a
    confidently wrong volume -- worse than no target at all)."""
    if not target_mass_g or target_mass_g <= 0:
        return None
    name = (material or "").strip()
    if not name:
        return None
    from forge_mind.materials import UnknownMaterialError, get_material

    try:
        mat = get_material(name)
    except UnknownMaterialError:
        return (
            f"Target mass: {target_mass_g:g} g -- but '{name}' has no registry density, so do "
            "NOT derive a volume from it; match the stated dimensions instead."
        )
    vol_cm3 = target_mass_g / mat.density_g_cm3
    # 3 significant figures, so the two units never disagree (review-caught: `.1f` printed
    # "0.1 cm^3" beside "127 mm^3", and "0.0 cm^3" beside "25 mm^3" -- two different targets).
    return (
        f"TARGET MASS {target_mass_g:g} g at {mat.density_g_cm3:g} g/cm^3 -> the finished solid "
        f"must measure about {vol_cm3:#.3g} cm^3 ({vol_cm3 * 1000.0:.0f} mm^3) of SOLID material. "
        "Build to that VOLUME: it is what the kernel measures. (Solid-equivalent basis -- a "
        "printed part at partial infill weighs less; machined/solid parts hit the mass exactly.)"
    )


def spec_dim_lines(feature: Any) -> list[str]:
    """One compact, demanding line per typed dim on a feature; [] when none exist.

    Renders every role the brief emits (thickness/diameter/length/width/height/...) —
    the thickness role is the measured killer, but a selective renderer would silently
    drop the next role the same way spec_dims itself was dropped."""
    dims = getattr(feature, "spec_dims", None) or []
    prose = str(getattr(feature, "approx_dims_mm", "") or "")
    lines: list[str] = []
    for d in dims:
        role = str(getattr(d, "role", "") or "").strip()
        of = str(getattr(d, "of", "") or "").strip()
        value = getattr(d, "value_mm", None)
        if not role or value is None:
            continue
        # The schema's own law: a spec_dim is a RESTATEMENT; one that does not trace to the
        # feature's prose is the 'second liar' and must never be promoted (review-caught --
        # rendering it as HARD would drive the author toward an invented number).
        if not spec_dim_agrees(d, prose):
            continue
        target = f" of the {of}" if of else ""
        unit = "deg" if role == "angle" else "mm"  # SpecDim.value_mm is degrees for angle
        lines.append(
            f"  - SPEC {role}{target}: {float(value):g} {unit} -- this is a HARD dimension; "
            "the geometry must measure it, not approximate it."
        )
    return lines
