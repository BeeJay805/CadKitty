"""Vision critic for the iterate loop: render the part and return a structured,
ACTIONABLE critique (per-feature verdict + prioritized fixes) the author revises from.

Distinct from ``judge.vlm_judge`` (advisory feature-presence, calibrated to assume-present
and never gate): the critic's job is to DRIVE improvement, so it names what to change. It
still does not gate the pipeline -- the deterministic validator does that; the critic only
decides "does this look right enough to stop iterating".
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.critic_views import critic_views_enabled, render_critic_sheet
from forge_mind.config import SONNET
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.viz.render import render_part_to_png

CRITIC_MAX_TOKENS = 4000  # dense multi-feature parts earn longer reviews (2026-07-10)


class FeatureVerdict(BaseModel):
    """Whether one brief feature reads correctly in the render."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(..., description="The brief feature being judged.")
    present: bool = Field(
        ..., description="True if plausibly present AND roughly the right shape in the render."
    )
    note: str = Field(default="", description="If absent/wrong: what is wrong. Empty if fine.")


class DesignCritique(BaseModel):
    """Structured critique of a render against the intended object + feature list."""

    model_config = ConfigDict(extra="forbid")

    looks_right: bool = Field(
        ...,
        description="True ONLY if the whole part clearly reads as the intended object with its key "
        "features present and decently proportioned.",
    )
    resemblance: str = Field(
        ...,
        description="Overall resemblance. Default scale: 'strong' | 'partial' | 'weak'. With the "
        "widened ordinal (FORGE_CRITIC_ORDINAL=1): 'none' | 'weak' | 'fair' | 'partial' | "
        "'good' | 'strong'.",
    )
    feature_verdicts: list[FeatureVerdict] = Field(default_factory=list)
    prioritized_fixes: list[str] = Field(
        default_factory=list,
        description="The most impactful GEOMETRIC changes first; each names the feature and a "
        "concrete change. Empty when looks_right is true.",
    )
    summary: str = Field(..., description="One-paragraph overall assessment.")


CRITIC_SYSTEM = """You are a meticulous CAD design reviewer. You are shown ONE image: a \
six-panel render (isometric, front, right, rear, top, bottom) of a single 3D part, plus the INTENDED real object \
and its feature list. Judge how well the modeled part matches the intended REAL object, and say \
what to CHANGE to make it better.

HOW TO LOOK
- Isometric for overall shape; front for side profile / proportions / taper; top for layout and \
holes. A feature evident in ONE view counts as present.
- IGNORE rendering artifacts: faceting on curved surfaces, shading, color, aliasing, jagged \
silhouettes. A faceted cylinder is still a cylinder. Never critique smoothness, print quality, \
or render aesthetics.

WHAT TO JUDGE
- For EACH listed feature: is it plausibly present AND roughly the right shape/proportion? If \
missing or clearly wrong, say what is wrong (e.g. "claw is a straight block; it should curve and \
split into two tines with a V notch").
- Overall: does the silhouette and proportion read as the intended object? A claw hammer has a \
compact heavy head and a long slender handle, not a dumbbell.

OUTPUT
- looks_right: true ONLY if the part clearly reads as the intended object with its key features \
present and decently proportioned. If a key feature is missing or badly shaped, set false.
- resemblance: 'strong' | 'partial' | 'weak'.
- feature_verdicts: present + a short note per feature.
- prioritized_fixes: the few most impactful GEOMETRIC changes, most important first, each naming \
the feature and a concrete change ("lengthen the handle to ~3x the head height", "make the head \
a round cylindrical poll", "curve the claw and cut a V notch"). If looks_right is false this MUST \
contain at least one concrete fix; if looks_right is true it MUST be empty. Do not nitpick \
millimeters or smoothness.
- summary: one short paragraph.

Judge SHAPE and FEATURE PRESENCE, not exact dimensions. Be specific and honest."""


# Step 1 of the reference-model spine (OPT-IN via FORGE_REFERENCE_MODEL=1): when a reference
# mesh of the real target part is supplied, the critic judges the BUILD against that REAL
# ground truth instead of only the model's own text idea of the object. Flag off (or no
# reference) = byte-identical to the text-only critique above.
_REFERENCE_SYSTEM = """You are a meticulous CAD design reviewer. The image has TWO stacked \
renders: the TOP ("BUILD") is the modeled part under review; the BOTTOM ("REFERENCE") is a render \
of the REAL target object. Judge how well the BUILD matches the REFERENCE's SHAPE, PROPORTIONS, and \
FEATURE LAYOUT, and say what to CHANGE so the build matches the reference better.

DIMENSIONS: the reference shows the correct FORM, not the size. NEVER tell the author to match the \
reference's absolute dimensions -- the user's stated sizes WIN. Judge shape, proportion, and which \
features go where.

HOW TO LOOK
- Compare the BUILD's silhouette and proportions against the REFERENCE's. IGNORE rendering \
artifacts: faceting on curved surfaces, shading, color, aliasing. A faceted cylinder is still a \
cylinder. Never critique smoothness or render aesthetics.

WHAT TO JUDGE
- For EACH listed feature: is it present in the BUILD and roughly the shape/placement the REFERENCE \
shows? If missing or clearly wrong, say what is wrong.
- Overall: does the BUILD's form read like the REFERENCE object?

OUTPUT
- looks_right: true ONLY if the BUILD clearly matches the REFERENCE's form with its key features \
present and decently proportioned.
- resemblance: 'strong' | 'partial' | 'weak' (of the BUILD vs the REFERENCE).
- feature_verdicts: present + a short note per feature.
- prioritized_fixes: the few most impactful GEOMETRIC changes to make the BUILD match the \
REFERENCE, most important first, each naming the feature and a concrete change. Non-empty iff \
looks_right is false.
- summary: one short paragraph.

Judge SHAPE and FEATURE LAYOUT against the reference, not exact dimensions. Be specific and honest."""


_REF_RENDER_MAX_FACES = 250_000  # a reference heavier than this is skipped (no decimation backend)


def reference_enabled() -> bool:
    """OPT-IN flag for reference-model critique (Step 1). Flag off = text-only, byte-identical."""
    return os.environ.get("FORGE_REFERENCE_MODEL") == "1"


def critic_ordinal_enabled() -> bool:
    """OPT-IN via FORGE_CRITIC_ORDINAL=1 (ADR-0117): widen resemblance from the 3-point scale
    to a 6-point ordinal. Measured motivation: qwen2.5vl sat at 'weak' for EVERY built
    iteration across 40+ samples -- a 3-point scale cannot register incremental progress, so
    the fidelity tie-break (which compares resemblance ordinals) never saw movement. Flag off
    = byte-identical prompts."""
    return os.environ.get("FORGE_CRITIC_ORDINAL") == "1"


_ORDINAL_ADDENDUM = """

RESEMBLANCE SCALE (use EXACTLY one of these six words, judged against what is VISIBLE):
- none: not the object at all (a plain block / unrelated shape)
- weak: barely suggestive -- one feature at most reads
- fair: SOME declared features read, the silhouette is still wrong
- partial: about half the features read and the proportions are roughly right
- good: most features read; small shape/proportion gaps remain
- strong: clearly the intended object
Move UP one step whenever a previously-missing feature becomes visible -- register progress."""


def critique_part(
    stl_path: Path,
    brief: DesignBrief,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    work_dir: Path | None = None,
    render_path: Path | None = None,
    reference_stl: Path | None = None,
    measured: str | None = None,
) -> tuple[DesignCritique, LLMUsage]:
    """Render ``stl_path`` and return a structured critique against ``brief``.

    If ``render_path`` is given it is reused (the loop already rendered the part);
    otherwise a render is made under ``work_dir`` (or the STL's directory).

    ``reference_stl`` (Step 1, OPT-IN via FORGE_REFERENCE_MODEL): a mesh of the REAL target
    part. When supplied and the flag is on, its render is stacked under the build render and
    the critic judges the build against that ground truth. Fail-closed: any reference-render
    failure falls back to the text-only critique (flag off = byte-identical).
    """
    # Lever d (ADR-0045, OPT-IN via FORGE_CRITIC_VIEWS=1): the critic gets a six-panel
    # sheet (4 shaded views + 2 section cuts, mm-labeled) so internal features and
    # per-axis scale are visible. ONLY the critic's image changes; the iteration log
    # render + the judge input are untouched. Best-effort: any failure falls back to
    # the standard render below (flag off = byte-identical behavior).
    png: Path | None = None
    if critic_views_enabled():
        try:
            out_dir = work_dir or stl_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            png = render_critic_sheet(
                stl_path, out_dir / f"{stl_path.stem}__critic_views.png", brief.object_name
            )
        except Exception:
            png = None
    if png is None and render_path is not None and render_path.exists():
        png = render_path
    if png is None:
        out_dir = work_dir or stl_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"{stl_path.stem}__critic.png"
        render_part_to_png(stl_path, png, title=brief.object_name, caption=brief.summary)

    system = CRITIC_SYSTEM
    user = _critic_user(brief)
    if reference_stl is not None and reference_enabled():
        out_dir = work_dir or stl_path.parent
        composite = composite_reference(png, reference_stl, out_dir, brief.object_name)
        if composite is not None:  # fail-closed: a render failure keeps the text-only critique
            png = composite
            system = _REFERENCE_SYSTEM
            user = _reference_user(brief)

    if critic_ordinal_enabled():
        # appended LAST so it rides both the text-only and the reference system prompts
        system = system + _ORDINAL_ADDENDUM

    # ADR-0119 (FORGE_CRITIC_MEASURED, caller-gated): the validator's measured context rides
    # the critique so fixes anchor on measured deficits instead of re-demanding features the
    # instruments already verified (the L66 manifold flange re-demand class). Appended to the
    # USER prompt so it rides both the text-only and reference paths; None = byte-identical.
    if measured:
        user = f"{user}\n\n{measured}"

    client = client or make_client()
    return client.generate_structured_with_image(
        system=system,
        user=user,
        image_png=png.read_bytes(),
        output_model=DesignCritique,
        model=model or SONNET,
        max_tokens=CRITIC_MAX_TOKENS,
    )


def _critic_user(brief: DesignBrief) -> str:
    lines = [
        "Here is the rendered six-view image of the modeled part.",
        "",
        f"INTENDED OBJECT: {brief.object_name}",
        f"It should look like: {brief.visual_description}",
        "",
        "FEATURES it should have:",
    ]
    lines.extend(f"  - {f.name}: {f.purpose} ({f.approx_dims_mm})" for f in brief.features)
    lines.append("")
    lines.append("Judge how well the render matches this, and list the most impactful fixes.")
    return "\n".join(lines)


def _reference_user(brief: DesignBrief) -> str:
    lines = [
        "The TOP render is the modeled BUILD under review; the BOTTOM render is the REFERENCE "
        "of the real target object.",
        "",
        f"INTENDED OBJECT: {brief.object_name}",
        f"User size constraints (these OVERRIDE the reference's size): {brief.overall_envelope_mm}",
        "",
        "FEATURES it should have:",
    ]
    lines.extend(f"  - {f.name}: {f.purpose} ({f.approx_dims_mm})" for f in brief.features)
    lines.append("")
    lines.append(
        "Judge how well the BUILD matches the REFERENCE's shape and feature layout (not its size), "
        "and list the most impactful fixes."
    )
    return "\n".join(lines)


def composite_reference(
    part_png: Path, reference_stl: Path, out_dir: Path, object_name: str
) -> Path | None:
    """Render the reference mesh and stack it under the build render into ONE labeled image
    (BUILD on top, REFERENCE below). Returns the composite path, or None on ANY failure
    (fail-closed: the caller falls back to the build-only, text-only critique).

    Shared by the vision critic (build-vs-reference critique) and the multimodal author (the
    build-vs-reference image the author revises against, ADR-0066)."""
    try:
        import trimesh
        from PIL import Image, ImageDraw

        out_dir.mkdir(parents=True, exist_ok=True)
        # Face guard (fail-closed): there is no mesh-decimation backend here, so a giant artist
        # mesh would hang the matplotlib render. Skip it -> the caller falls back to text-only.
        ref_mesh = trimesh.load(str(reference_stl), force="mesh")
        if not hasattr(ref_mesh, "faces") or len(ref_mesh.faces) > _REF_RENDER_MAX_FACES:
            return None
        # Render the reference ONCE, cached next to the (stable) reference file so the iteration
        # loop reuses it instead of re-rendering every pass.
        ref_png = reference_stl.parent / f"{reference_stl.stem}__ref_render.png"
        if not ref_png.exists():
            render_part_to_png(reference_stl, ref_png, title=f"REFERENCE: {object_name}")

        build = Image.open(part_png).convert("RGB")
        ref = Image.open(ref_png).convert("RGB")
        width = max(build.width, ref.width)

        def _fit(im: Any) -> Any:
            if im.width == width:
                return im
            return im.resize((width, round(im.height * width / im.width)))

        build, ref = _fit(build), _fit(ref)
        band = 30
        canvas = Image.new("RGB", (width, build.height + ref.height + 2 * band), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "BUILD (under review)", fill="black")
        canvas.paste(build, (0, band))
        draw.text((8, band + build.height + 8), "REFERENCE (real target shape)", fill="black")
        canvas.paste(ref, (0, 2 * band + build.height))
        composite = out_dir / f"{part_png.stem}__vs_reference.png"
        canvas.save(composite)
        return composite
    except Exception:
        return None


__all__ = [
    "CRITIC_SYSTEM",
    "DesignCritique",
    "FeatureVerdict",
    "composite_reference",
    "critique_part",
    "reference_enabled",
]
