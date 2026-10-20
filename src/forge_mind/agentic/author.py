"""Author stage: Opus writes build123d code from a DesignBrief.

First pass = a clean, VALID blocked-out massing (correct proportions + the primary
features in the right places, with simple-but-RIGHT methods: revolve a round head,
taper a handle -- not a stack of boxes). Detail emerges in the render-critique loop:
each later call gets the previous script + the critique/measurements and makes a
SURGICAL edit (treat the prior script as a frozen base; change only what is flagged;
keep what already works). The full script is the reproducible artifact; the loop
records the diff between versions as the visible "patch".
"""

from __future__ import annotations

import os
import re

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic import feature_plan
from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.cheatsheet import (
    EDGE_FINISH_ADDENDUM,
    author_skills_block,
    edge_finish_enabled,
    onecall_enabled,
)
from forge_mind.agentic.exemplars import exemplars_block
from forge_mind.agentic.params import param_opt_note_enabled
from forge_mind.config import FABLE, HAIKU, OPUS, OPUS_5, SONNET
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client


class AuthoredCode(BaseModel):
    """The author's output: a short construction plan + a complete build123d script."""

    model_config = ConfigDict(extra="forbid")

    approach: str = Field(
        ..., description="1-3 sentences: how you build it (the construction plan)."
    )
    code: str = Field(
        ...,
        description="Complete build123d script. Binds the finished solid to `result`. "
        "The kernel + Forge-Mind helpers are already imported; no imports needed.",
    )


class FeatureMapEntry(BaseModel):
    """One plan-to-code mapping row (lever f): a checklist feature -> the code that builds it."""

    model_config = ConfigDict(extra="forbid")

    feature: str = Field(..., description="The checklist feature name, copied VERBATIM.")
    where: str = Field(
        ...,
        description="The exact identifier or short code fragment in the script that builds "
        "this feature; it must appear in the script verbatim.",
    )


class AuthoredCodeWithPlan(AuthoredCode):
    """AuthoredCode + the mandatory plan-to-code mapping. Used ONLY when
    FORGE_FEATURE_PLAN=1 -- the deployed default keeps the plain AuthoredCode schema."""

    feature_map: list[FeatureMapEntry] = Field(
        ..., description="EXACTLY one entry per feature-checklist item."
    )


_SKILLS = author_skills_block(one_call=False)
_SKILLS_ONECALL = author_skills_block(one_call=True)

_SYSTEM_INITIAL = f"""You are an expert build123d CAD engineer. You write Python that models \
the requested part as accurately as the REAL object -- thin where it is thin, round where it \
is round, never a blocky approximation.

{_SKILLS}

THIS PASS: produce a clean, VALID blocked-out massing -- correct overall proportions and the \
PRIMARY features in the right places, using simple-but-RIGHT methods (revolve a round head, \
taper a handle; do not approximate a round thing with a box). Keep it ONE watertight solid. \
Build EVERY functional feature the brief lists NOW -- holes, bores, slots, pockets, and mounting \
features are part of the massing, NOT deferrable 'detail', and a repeated feature means the brief's \
full COUNT of it (four bolt holes = four holes). Only COSMETIC refinement (fillets, chamfers, \
surface polish) waits for a later pass. Bind the finished solid to `result`."""
# NOTE (2026-07-18, measured at n=8): a "VERIFIED API RULES" block appended here killed specific
# first-pass hallucinations (revolve(angle=) went 3/3 -> 0) but net-zeroed the SHIP rate (25% with
# and without, arms J vs M) while regressing the pulley class from validator-clean 3/3 to 0/8
# builds (it pushed the model from buildable extruded discs toward revolve half-profiles it
# cannot draw validly). Reverted on that evidence; the same corrections still reach the model
# through the error-triggered hints (kernel_hints.api_misuse_hint), which fire only when the
# mistake actually happened. Arms J-N: benchmarks/2026-07-18_ctx_untruncation_ab/.

_SYSTEM_REVISE = f"""You are an expert build123d CAD engineer improving an EXISTING part.

{_SKILLS}

You are given the current script, how the part currently looks and measures, and a prioritized \
critique. Make the SMALLEST set of edits that addresses the critique: treat the current script \
as a frozen base, change only what is flagged, and KEEP everything the critique called correct. \
Add the requested DETAIL (curves, tapers, claws, panels, fillets, chamfers) using the right \
methods. If a previous fillet/operation raised in the kernel, back off the radius or reorder -- \
do not delete a correct feature to dodge an error. If the validator reports a wall or edge \
thinner than the minimum, find the knife-thin feature (a claw tip, blade, chamfer, or a sliver \
where a loft/boolean nearly closes) and give it a small flat or radius so NO edge is thinner than \
the minimum wall -- real tools have blunt-ish edges, not zero-thickness knives.

DIAGNOSE the failure before you edit, then apply its fix:
- a MISSING feature (a hole/pocket that should be there is not): the cut was too shallow (extend it \
PAST both faces so it goes fully through), or a selector retargeted after a prior boolean (re-pick the \
face by axis/position, not a list index), or the mode was wrong (SUBTRACT vs ADD).
- DISCONNECTED bodies / not a sealed solid: a feature only touches the surface -- start it several mm \
INSIDE the body it joins so the union FUSES into one solid; close every sketch profile with make_face().
- a kernel / fillet RAISE: the radius exceeds the local thickness or hits a renumbered edge -- narrow \
the edge filter, apply the fillet LAST, and back off the radius; never delete the feature to dodge it.
- WRONG size: check radius-vs-diameter and the extrude amount, and match the brief's reference dims.
- an INVALID / self-intersecting B-rep: an over-large fillet or a degenerate boolean made a bad face -- \
rebuild that one feature with a smaller radius or a cleaner, deeper overlap.

Return the COMPLETE updated script binding `result`. Never regress a feature that was already right."""

# ADR-0133 fix 2 (FORGE_ONECALL_FEATURES): the one-call variants are the SAME prompts with
# only the skills block swapped -- constructed by replacement so there is no second template
# to drift. Both prebuilt at import; the flag is read at CALL time (monkeypatch-friendly).
_SYSTEM_INITIAL_ONECALL = _SYSTEM_INITIAL.replace(_SKILLS, _SKILLS_ONECALL)
_SYSTEM_REVISE_ONECALL = _SYSTEM_REVISE.replace(_SKILLS, _SKILLS_ONECALL)


def _edge_finish_block() -> str:
    """ADR-0133 fix 3: the fillet-first teaching, call-time gated (the _doctrine_block
    pattern -- flag off = empty string = byte-identical prompt)."""
    return f"\n\n{EDGE_FINISH_ADDENDUM}" if edge_finish_enabled() else ""


def _system_initial() -> str:
    base = _SYSTEM_INITIAL_ONECALL if onecall_enabled() else _SYSTEM_INITIAL
    return base + _edge_finish_block()


def _system_revise() -> str:
    base = _SYSTEM_REVISE_ONECALL if onecall_enabled() else _SYSTEM_REVISE
    return base + _edge_finish_block()


# Multimodal-author vision notes (OPT-IN via FORGE_AUTHOR_VISION, ADR-0066): appended to the
# revise system prompt ONLY when the loop hands author_revision a render of the current build, so
# the author revises WITH its own eyes instead of on lossy text alone. Two variants: build-only,
# and (when FORGE_REFERENCE_MODEL is also on) build-stacked-over-the-real-reference -- the same
# BUILD-vs-REFERENCE composite the vision critic already uses. Flag off = no image = these are
# never appended (byte-identical to the text-only revise prompt).
_REVISE_VISION_NOTE = """
YOU CAN SEE YOUR BUILD. The attached image is a six-panel render (isometric, front, right, rear, \
top, bottom) of the CURRENT script's output -- the exact part the critique below is about. Match \
each critique item to what you SEE, and also fix anything the render shows is wrong that the text \
missed (a wrong silhouette, a misplaced or missing feature, bad proportions). IGNORE rendering \
artifacts (faceting on curved surfaces, shading, aliasing): a faceted cylinder is still a cylinder \
-- never 'fix' smoothness."""

_REVISE_VISION_REFERENCE_NOTE = """
YOU CAN SEE YOUR BUILD AGAINST THE REAL TARGET. The attached image has TWO stacked six-panel \
renders: the TOP is your CURRENT build; the BOTTOM is a render of the REAL reference object. \
Compare them directly and change the code so the TOP's silhouette, proportions, and feature \
placement move toward the BOTTOM. The reference shows the correct FORM, NOT the size -- keep the \
user's stated dimensions; match shape, proportion, and WHICH feature goes WHERE. IGNORE rendering \
artifacts (faceting on curved surfaces, shading, aliasing)."""

# Appended to the author's user message ONLY in PRINT_3D mode (ADR-0031): steer the author to DESIGN
# for FDM printing. In GENERAL (the default) it is omitted -- the author models the real object
# faithfully and is not constrained by printer physics (a hammer's head overhangs by nature).
_PRINT_GUIDANCE = (
    "\n\nFABRICATION TARGET: FDM 3D PRINTING -- design this to print cleanly:\n"
    "- Choose a print orientation (which face sits on the bed) that puts a large FLAT face down and "
    "minimizes overhangs, and build the part in that orientation.\n"
    "- Avoid steep UNSUPPORTED overhangs (a face more than ~45 deg from vertical) and long unsupported "
    "bridges; prefer self-supporting chamfers over horizontal ledges, and add a fillet/gusset where a "
    "feature would otherwise overhang.\n"
    "- Keep every wall, rib, and web at or above the brief's minimum wall thickness so it prints solid."
)


def _print_guidance(fabrication_mode: FabricationMode) -> str:
    """The FDM design block, or '' when the part is not being 3D-printed (GENERAL)."""
    return _PRINT_GUIDANCE if fabrication_mode.enforces_printability else ""


# The parametric contract (ADR-0073, OPT-IN via FORGE_PARAM_OPT): the author exposes every key
# dimension as a named float in a PARAMS dict so the numerical optimizer can fit them against the
# reference form after authoring. Structure from the model, numbers from the optimizer. Flag off
# (the deployed default) = never appended = byte-identical prompts.
_PARAMS_CONTRACT_NOTE = """
PARAMETRIC CONTRACT: start the script with a literal dict of every key dimension, e.g.
PARAMS = {"shaft_d": 55.0, "web_thickness": 18.0, "throw_offset": 42.0, "n_webs": 5}
and use PARAMS["..."] everywhere instead of repeating numbers. FLOATS for dimensions in mm; INTS
for counts/teeth. After you write the script, a numerical optimizer will tune the FLOAT values
against the real object's measured form -- so put your effort into the STRUCTURE (which features
exist, how they attach, what repeats) and route every dimension through PARAMS so it is tunable."""


def _params_note(has_reference: bool = True) -> str:
    # has_reference gates the note in CONDITIONAL mode (ADR-0108); ignored in the default blanket
    # mode. A render passed to the author IS a reference-rich part -> the note's proven case.
    return _PARAMS_CONTRACT_NOTE if param_opt_note_enabled(has_reference) else ""


_FENCE = re.compile(r"^\s*```(?:python)?\s*\n(.*?)\n\s*```\s*$", re.DOTALL)

# The author's output-token budget. 8000 was sized for Opus (the 4000 cap truncated the
# fan-shroud script mid-JSON and crashed the parse -- same failure mode, one tier down).
# Candidate authors that write longer scripts (the fable-5 A/B truncated at 8000 on
# wall_hook/fan_shroud and author_failed instantly) need more headroom; billing is per
# ACTUAL token, so the only effect of headroom is that long scripts complete. Raised
# 8000 -> 16000 by founder direction 2026-07-10 ("big visions" headroom; supersedes the
# earlier A/B-gated default) -- FORGE_AUTHOR_MAX_TOKENS still overrides.
_DEFAULT_AUTHOR_MAX_TOKENS = 16000


def author_max_tokens() -> int:
    raw = os.environ.get("FORGE_AUTHOR_MAX_TOKENS")
    if raw:
        try:
            return max(int(raw), 1000)
        except ValueError:
            pass
    return _DEFAULT_AUTHOR_MAX_TOKENS


_MODEL_ALIASES = {
    "opus": OPUS,
    "opus5": OPUS_5,  # selectable A/B arm; ADR-0076's Fable-over-Opus verdict predates Opus 5
    "fable": FABLE,
    "sonnet": SONNET,
    "haiku": HAIKU,
}


def author_default_model() -> str:
    """The agentic author's model, overridable at RUNTIME via ``FORGE_AUTHOR_MODEL`` (an alias
    ``opus``/``fable``/``sonnet``/``haiku``, or a full model id); default ``OPUS``. Read at call time
    (like ``author_max_tokens``), so a bench / A/B can flip the author model per run with ZERO code
    change -- the same one-line ENV-flip pattern ``FORGE_BRAIN_MODEL`` uses for the invention brain.
    An explicit ``model=`` argument always wins over this default."""
    raw = os.environ.get("FORGE_AUTHOR_MODEL", "").strip()
    return _MODEL_ALIASES.get(raw.lower(), raw or FABLE)


def _doctrine_block() -> str:
    """The proactive construction doctrine (FORGE_DOCTRINE=1; ADR-0122 candidate). Read at
    call time like every author flag; off = the empty string = byte-identical prompt."""
    from forge_mind.agentic.cheatsheet import CONSTRUCTION_DOCTRINE, doctrine_enabled

    return f"\n\n{CONSTRUCTION_DOCTRINE}" if doctrine_enabled() else ""


def author_vision_enabled() -> bool:
    """OPT-IN flag for the multimodal author (ADR-0066): when on, the loop renders the build the
    author is revising and hands it to author_revision, so the author revises WITH vision. Flag
    off = no image is ever passed = byte-identical to the text-only revise path."""
    return os.environ.get("FORGE_AUTHOR_VISION") == "1"


def author_decompose_enabled() -> bool:
    """OPT-IN flag for the repeated-unit decomposition guidance (ADR-0067). Off = the guidance is
    never appended = byte-identical author prompt."""
    return os.environ.get("FORGE_DECOMPOSE") == "1"


# General repeated/phased-unit construction guidance (OPT-IN via FORGE_DECOMPOSE). The measured
# crankshaft failure: the author approximates a repeated-unit part (4 phased crank throws) as ONE
# axisymmetric revolve, which erases each unit's offset/phase -- the pins come out concentric, the
# counterweights a plain disc, and no amount of measured critique moves it because the whole
# construction is wrong. This teaches the GENERAL fix (build one unit, array it) for ANY repeated-unit
# part -- blades, teeth, throws, lobes, spokes, fins -- NOT a per-family crankshaft templater.
_DECOMPOSE_GUIDANCE = """

REPEATED-UNIT PARTS -- build ONE unit, then ARRAY it (never approximate as one axisymmetric solid).
If the brief describes N identical or phased units in a regular arrangement -- blades, gear/ratchet \
teeth, crank throws, cam lobes, spokes, fins, vanes, auger flights -- model ONE representative unit \
as a correct 3D solid, then REPLICATE it, deriving the count and spacing/phase from the brief:
- around an axis at one height: `with PolarLocations(radius, count): <sketch>` then extrude, or fuse \
`Rot(0, 0, i * 360 / N) * unit` copies.
- stepped ALONG an axis with a rotational phase (crank throws, cam stacks, staged rotors): build the \
unit centered on the axis, then fuse placed copies --
    result = Part()
    for z, phase in stations:            # stations (axial position, phase) from the brief's count
        result += Pos(0, 0, z) * Rot(0, 0, phase) * unit()
- NEVER revolve a part whose units carry an offset or phase (an offset crank pin, a swept blade, an \
asymmetric sawtooth tooth): a revolve is axisymmetric and ERASES that feature -- the pins end up \
concentric, the blades a plain disc. Build the true unit and array it.
Keep it ONE fused watertight solid: each placed unit must OVERLAP its neighbour or the core so the \
union seals into one body."""


def _decompose_guidance() -> str:
    """The repeated-unit guidance block, or '' when FORGE_DECOMPOSE is off (byte-identical default)."""
    return _DECOMPOSE_GUIDANCE if author_decompose_enabled() else ""


def clean_code(code: str) -> str:
    """Strip a stray markdown code fence if the model wrapped its script in one, then repair a
    stray-indent artifact if that is all that stops the script from compiling."""
    match = _FENCE.match(code.strip())
    cleaned = match.group(1) if match else code.strip()
    return _fix_stray_indent(cleaned)


def _fix_stray_indent(code: str) -> str:
    """Zero-LLM repair for a MEASURED structured-output artifact (2026-07-18): the local model
    occasionally emits ONE spurious leading space on a top-level line ("` groove_depth = 4`"),
    and exec dies with IndentationError while the code is otherwise sound. Self-gating (the
    deterministic-repair doctrine): only when compile() raises IndentationError, only the named
    line, only when stripping its leading whitespace makes the script compile -- any other
    outcome returns the original and the LLM repair path stands."""
    try:
        compile(code, "<authored_part>", "exec")
        return code
    except IndentationError as exc:
        lineno = exc.lineno or 0
        lines = code.splitlines()
        if not (1 <= lineno <= len(lines)) or not lines[lineno - 1][:1].isspace():
            return code
        candidate_lines = list(lines)
        candidate_lines[lineno - 1] = candidate_lines[lineno - 1].lstrip()
        candidate = "\n".join(candidate_lines)
        try:
            compile(candidate, "<authored_part>", "exec")
        except SyntaxError:
            return code
        return candidate
    except SyntaxError:
        return code  # not an indent artifact; the sandbox reports it and the LLM repairs


def _plan_user_suffix(brief: DesignBrief) -> str:
    """The feature-checklist prompt block (lever f), or '' when the lever is off --
    the deployed default prompt stays byte-identical with FORGE_FEATURE_PLAN unset."""
    if not feature_plan.enabled():
        return ""
    return feature_plan.checklist_block(feature_plan.feature_checklist(brief))


def _swept_idiom_suffix(brief: DesignBrief) -> str:
    """FORGE_SWEEP_IDIOM=1 (ADR-0133 fix 1, default OFF = byte-identical prompt): when
    the brief's TYPED fields name a swept form, the verified swept idiom rides the FIRST
    author prompt instead of arriving only after a burned iteration (sweep-callers accept
    at 27-30%, the corpus's worst class; 5/5 archived hook seeds burned their first draft
    on the same signature). Initial pass only -- the revise path already receives the
    idiom error-triggered."""
    if os.environ.get("FORGE_SWEEP_IDIOM") != "1":
        return ""
    from forge_mind.agentic.idioms import swept_form_idiom

    text = " ".join(
        [brief.object_name, brief.summary, *(f"{f.name} {f.method}" for f in brief.features)]
    )
    idiom = swept_form_idiom(text)
    return f"\n\n{idiom}" if idiom else ""


def _output_model(brief: DesignBrief) -> type[AuthoredCode]:
    """AuthoredCodeWithPlan when the feature-plan lever is ON and there is something to
    map; the plain AuthoredCode schema otherwise (deployed default, byte-identical)."""
    if feature_plan.enabled() and feature_plan.feature_checklist(brief):
        return AuthoredCodeWithPlan
    return AuthoredCode


def _recode(authored: AuthoredCode) -> AuthoredCode:
    """Re-wrap with the fence-stripped code, PRESERVING the plan mapping subclass."""
    if isinstance(authored, AuthoredCodeWithPlan):
        return AuthoredCodeWithPlan(
            approach=authored.approach,
            code=clean_code(authored.code),
            feature_map=authored.feature_map,
        )
    return AuthoredCode(approach=authored.approach, code=clean_code(authored.code))


# Multimodal author at the FIRST pass (ADR-0069): when the loop has a real reference and vision is
# on, show it to author_initial so the massing starts FROM the right form instead of a blind guess
# (the concavity gap is born at iteration 1). Appended to the initial system prompt; the call then
# goes through the image path. No image = byte-identical to the text-only first pass.
_INITIAL_VISION_NOTE = """
YOU CAN SEE THE REAL TARGET. The attached image is a render of the REAL object you are modeling. \
Build your first massing to MATCH its overall form, proportions, and where the features sit -- the \
silhouette, the concave features (pockets, crescents, hollows, the gaps between blades/webs/spokes), \
and which way it is elongated. The reference shows the correct FORM, NOT the size -- use the user's \
stated dimensions. IGNORE rendering artifacts (faceting on curved surfaces, shading, aliasing)."""


def author_initial(
    brief: DesignBrief,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    max_tokens: int | None = None,  # None -> author_max_tokens() (env-overridable budget)
    strategy: str = "",
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    image_png: bytes | None = None,
) -> tuple[AuthoredCode, LLMUsage]:
    """First pass: a valid blocked-out massing of the brief.

    ``strategy`` (optional) appends a distinct MASSING approach so best-of-N drafts
    diversify (different fusion / construction strategies, not just sampling noise).
    ``fabrication_mode`` PRINT_3D appends FDM design guidance; GENERAL omits it.

    ``image_png`` (ADR-0069): a render of the REAL reference object. When given, the massing is
    authored WITH vision (the system prompt gains a "you can see the real target" note and the call
    goes through the image path), so it starts from the right form. None = byte-identical to text.
    """
    client = client or make_client()
    user = _brief_to_user(brief)
    if strategy:
        user += f"\n\nMASSING STRATEGY for this draft: {strategy}"
    # Exemplar memory (lever e, ADR-0045): validated reference builds of similar
    # parts, OPT-IN via FORGE_AUTHOR_EXEMPLARS (a library dir; unset = '' = the
    # deployed default, byte-identical prompt). Best-effort by construction.
    user += exemplars_block(f"{brief.object_name} {brief.summary}")
    user += _print_guidance(fabrication_mode)
    user += _plan_user_suffix(brief)
    user += _swept_idiom_suffix(brief)
    system = (
        _system_initial()
        + _decompose_guidance()
        + _params_note(image_png is not None)
        + _doctrine_block()
    )
    if image_png is not None:
        code, usage = client.generate_structured_with_image(
            system=system + _INITIAL_VISION_NOTE,
            user=user,
            image_png=image_png,
            output_model=_output_model(brief),
            model=model or author_default_model(),
            max_tokens=max_tokens or author_max_tokens(),
        )
        return _recode(code), usage
    code, usage = client.generate_structured(
        system=system,
        user=user,
        output_model=_output_model(brief),
        model=model or author_default_model(),
        max_tokens=max_tokens or author_max_tokens(),
    )
    return _recode(code), usage


def author_revision(
    brief: DesignBrief,
    previous_code: str,
    feedback: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    max_tokens: int | None = None,  # same env-overridable budget as author_initial
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
    image_png: bytes | None = None,
    image_has_reference: bool = False,
) -> tuple[AuthoredCode, LLMUsage]:
    """Revision pass: surgically edit ``previous_code`` to address ``feedback``.

    ``fabrication_mode`` PRINT_3D appends FDM design guidance; GENERAL omits it.

    ``image_png`` (multimodal author, ADR-0066): a six-panel render of the build being revised.
    When given, the author revises WITH vision -- the revise prompt gains a "you can see your
    build" note and the call goes through the vision path. ``image_has_reference`` selects the
    note variant: True when the image is a BUILD-over-REFERENCE composite (FORGE_REFERENCE_MODEL
    also on), False for a build-only render. ``image_png=None`` (the deployed default) is
    byte-identical to the text-only revise call below."""
    client = client or make_client()
    user = (
        (
            f"{_brief_to_user(brief)}\n\n"
            f"CURRENT SCRIPT:\n```python\n{previous_code}\n```\n\n"
            f"WHAT TO FIX (prioritized):\n{feedback}\n\n"
            "Return the complete updated script that addresses this."
        )
        + _print_guidance(fabrication_mode)
        + _plan_user_suffix(brief)
    )
    if image_png is not None:
        note = _REVISE_VISION_REFERENCE_NOTE if image_has_reference else _REVISE_VISION_NOTE
        code, usage = client.generate_structured_with_image(
            system=f"{_system_revise()}\n{note}{_decompose_guidance()}",
            user=user,
            image_png=image_png,
            output_model=_output_model(brief),
            model=model or author_default_model(),
            max_tokens=max_tokens or author_max_tokens(),
        )
        return _recode(code), usage
    code, usage = client.generate_structured(
        system=_system_revise() + _decompose_guidance() + _params_note(),
        user=user,
        output_model=_output_model(brief),
        model=model or author_default_model(),
        max_tokens=max_tokens or author_max_tokens(),
    )
    return _recode(code), usage


def _brief_to_user(brief: DesignBrief) -> str:
    # FORGE_SPEC_DIM_PROMPT (dark; L102's thin-wall class): the brief's typed per-feature
    # spec_dims never reached the author -- five of H5B's eight failures died on walls whose
    # typed floor existed unseen. When armed, each feature's agreement-guarded dims render
    # as HARD lines directly under it.
    from forge_mind.agentic.spec_dim_prompt import (
        material_line,
        material_prompt_enabled,
        spec_dim_lines,
        spec_dim_prompt_enabled,
        target_volume_line,
    )

    spec_on = spec_dim_prompt_enabled()
    lines = [
        f"Build this part: {brief.object_name}",
        f"Summary: {brief.summary}",
        f"It should look like: {brief.visual_description}",
        f"Overall envelope: {brief.overall_envelope_mm}",
        f"Minimum wall / rib: {brief.min_wall_mm} mm",
    ]
    # FORGE_MATERIAL_PROMPT (dark; L103's dominant mass cause): brief.material has always been
    # validator-only, so the author sized target masses at a privately assumed density.
    if material_prompt_enabled():
        mat_line = material_line(brief.material)
        if mat_line:
            lines.append(mat_line)
        # ...and the stated target mass as a VOLUME target (the typed `target_mass_g` carrier;
        # the same flag gates both halves because neither is usable alone -- a density with no
        # mass and a mass with no density both leave the author guessing).
        vol_line = target_volume_line(getattr(brief, "target_mass_g", None), brief.material)
        if vol_line:
            lines.append(vol_line)
    lines.append("Features (name: purpose | dims | method):")
    for f in brief.features:
        lines.append(
            f"  - {f.name}{f'  x{f.count}' if f.count > 1 else ''}: {f.purpose} | "
            f"{f.approx_dims_mm} | method: {f.method}"
        )
        if spec_on:
            lines.extend(spec_dim_lines(f))
    return "\n".join(lines)


__all__ = [
    "AuthoredCode",
    "AuthoredCodeWithPlan",
    "FeatureMapEntry",
    "author_decompose_enabled",
    "author_default_model",
    "author_initial",
    "author_revision",
    "author_vision_enabled",
    "clean_code",
]
