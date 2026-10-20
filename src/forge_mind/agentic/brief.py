"""Research -> DesignBrief: a cheap-model stage that studies the real object and
emits the structured spec the author builds from.

Function-first (after ``assembly.plan.FunctionalPlan``): each feature names WHAT it
is, what it is FOR, its reference dimensions, and the build123d METHOD that realizes
it. The author consumes this; the vision critic and acceptance judge compare the
render against ``visual_description`` + the feature list.

Wave 1 research is the model's own world-knowledge expressed as structure -- there is
no web / image fetch yet. ``extra_context`` is the seam where retrieved reference text
(specs, dimensions) will be injected for novel or obscure parts in a later wave.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import BRIEF_MAX_TOKENS, SONNET
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

# SpecGraph slice 2 flag (ADR-0088). OFF (default) -> the researcher emits a plain DesignBrief,
# byte-identical to today. ON -> a SpecBrief with typed spec_dims + stable feature_id, whose
# EMISSION VALIDITY (structured fields agreeing with the model's own prose) is the pre-registered
# entry gate BEFORE any verdict consumes them (the skeptic's "second liar" guard).
_SPEC_BRIEF_FLAG = "FORGE_SPEC_BRIEF"


def spec_brief_enabled() -> bool:
    return bool(os.environ.get(_SPEC_BRIEF_FLAG))


# Form-quality wording at the BRIEF layer (FORGE_FIDELITY_BRIEF=1, default OFF; off =
# the research system byte-identical). The visual_description IS the vision critic's
# and acceptance judge's comparison target -- the L88 humanoid passed every gate while
# reading as a flat-plate prototype because nothing in the reference described the
# DESIGNED form. Generalized at founder direction (2026-07-23: every object class,
# not just robots). Independent flag from FORGE_FIDELITY_DOCTRINE (plan layer) so a
# stacked arm stays bisectable by env stamps.
def fidelity_brief_enabled() -> bool:
    return os.environ.get("FORGE_FIDELITY_BRIEF") == "1"


_FIDELITY_BRIEF_ADDENDUM = """

FORM QUALITY (crafted-product fidelity, every object): describe the object as the
DESIGNED, CRAFTED product it would ship as -- name its real form language: what is
tapered, crowned, domed, swept, shelled, and filleted, and where the surface details
live (recesses, panel lines, vents, grips, badges). Never describe it as a stack of
flat plates or bare primitive solids unless the real object truly is one. The
visual_description is the reviewer's comparison target: write it so a crude
primitives-only approximation FAILS the comparison and only the product-grade,
detailed form passes."""


class BriefFeature(BaseModel):
    """One functional feature of the part, named by function (not by primitive)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ..., description="Short feature name, e.g. 'claw', 'striking face', 'handle'."
    )
    purpose: str = Field(..., description="What this feature is FOR (its function).")
    approx_dims_mm: str = Field(
        ..., description="Reference dimensions in mm, free text, e.g. 'face 25x25, head 110 long'."
    )
    method: str = Field(
        ...,
        description="build123d approach that makes the RIGHT shape: revolve | loft | sweep | "
        "box+fillet | pocket | shell | bolt_holes | ...",
    )
    count: int = Field(
        default=1,
        ge=1,
        description="How many INSTANCES of this feature (4 bolt holes -> 4; one handle -> 1).",
    )
    standard: str | None = Field(
        default=None,
        description="If this feature mates a STANDARD part, its designation so the modeler uses the "
        "EXACT ISO/DIN dimension instead of guessing: a fastener 'M2'..'M8' (clearance hole / tap / "
        "counterbore / captive nut), a bearing '608'/'625'/'6800', a dowel '2'..'8', a magnet "
        "'6x3'. Omit for non-standard / freeform features.",
    )
    # Estimate self-report (ADR-0040): HOW the dims were reasoned when no standard applies.
    # Informational only — code derives the provenance TIER from anchors the model cannot
    # author (registry rows / the user's confirmed facts); this never raises a tier.
    estimate_basis: (
        Literal["anchor_scaled", "functional_requirement", "proportion_rule", "default_guess"]
        | None
    ) = Field(
        default=None,
        description="For NON-standard dims: how they were derived. anchor_scaled = scaled from a "
        "named real reference; functional_requirement = forced by a constraint the part must meet; "
        "proportion_rule = a stated convention; default_guess = it looked right. Omit for "
        "standard-backed features.",
    )
    estimate_anchor: str | None = Field(
        default=None,
        description="The named reference or constraint the estimate hangs on, e.g. 'typical 4-cyl "
        "bore ~80mm' or 'must clear an M3 cap head'. Omit when estimate_basis is omitted.",
    )


class SpecDim(BaseModel):
    """A typed restatement of ONE number the feature's prose already states (SpecGraph slice 2).

    The point is the TYPE: ``role`` says whether 3.4 is a diameter or a depth, so no downstream
    check re-guesses it from prose. This is a RESTATEMENT, not a new fact -- every value must
    also appear in the feature's ``approx_dims_mm``; the emission gate measures exactly that
    agreement, and a spec_dim that does not trace to the prose is the 'second liar' the gate
    rejects (never promoted to a verdict input)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal[
        "diameter", "length", "width", "height", "depth", "thickness", "pitch", "pcd", "angle"
    ] = Field(..., description="What KIND of dimension this is.")
    value_mm: float = Field(..., description="The value in mm (degrees for role=angle).")
    of: str = Field(
        default="",
        description="Short label of what it measures, e.g. 'bore', 'plate length'. Optional.",
    )


class SpecFeature(BriefFeature):
    """A BriefFeature with the slice-2 structured fields. IS-A BriefFeature: every consumer that
    reads a plain feature keeps working; only spec-aware code reads the new fields."""

    feature_id: str | None = Field(
        default=None,
        description="A STABLE, UNIQUE id for this feature, e.g. 'case_mount_holes' vs "
        "'side_mount_holes'. This is the identity two same-diameter hole groups need so one is "
        "never confused for the other.",
    )
    spec_dims: list[SpecDim] | None = Field(
        default=None,
        description="Each number in approx_dims_mm, restated with its TYPE. Restate ONLY numbers "
        "the prose already states -- never invent one.",
    )


class DesignBrief(BaseModel):
    """The spec the author builds from: a researched, function-first plan of a part."""

    model_config = ConfigDict(extra="forbid")

    object_name: str = Field(..., description="What the part IS, e.g. 'claw hammer'.")
    summary: str = Field(..., description="One-line description of the whole part.")
    visual_description: str = Field(
        ...,
        description="What it should LOOK like when rendered -- silhouette, proportions, what is "
        "round vs flat vs tapered, where it is thin. The vision critic's comparison target.",
    )
    overall_envelope_mm: str = Field(
        ..., description="Approx overall L x W x H in mm with a one-line rationale."
    )
    features: list[BriefFeature] = Field(..., min_length=1, max_length=12)
    min_wall_mm: float = Field(..., gt=0, description="Minimum wall/rib thickness target (mm).")
    material: str = Field(default="PLA", description="Suggested print material.")
    # 2026-07-27 (L103): a stated target mass used to reach the author only if the researcher
    # happened to keep it in PROSE. Measured on the refs bench: ppp0103's brief kept "mass
    # reaches 96.13 g" and scored -0% mass error; ppp0106's dropped it entirely and scored
    # +187%. Typed so it cannot be lost -- structure binds, prose does not (ADR-0129 law 2).
    target_mass_g: float | None = Field(
        default=None,
        gt=0,
        description="The part's stated target / verification MASS to HIT, ALWAYS IN GRAMS. Fill "
        "it only for an explicit equality target, and CONVERT to grams first: '2 kg' -> 2000, "
        "'1.5 lb' -> 680, 'want_mass=96.13' (already g) -> 96.13. Leave null for a BOUND ('under "
        "100 g', 'as light as possible' -- those are limits, not targets, and belong in the "
        "feature prose), and null when no mass is stated: NEVER invent or derive one.",
    )
    print_orientation_note: str = Field(
        ..., description="How to orient it for printing (which face goes down)."
    )
    target: Literal["cosmetic", "functional"] = Field(
        default="cosmetic",
        description="cosmetic = ONE monolithic body that LOOKS like the object (a vase, a bracket, "
        "a panel). functional = an assembly of SEPARATE parts joined by interfaces -- a "
        "tool/mechanism/multi-material object where the parts are genuinely distinct (a hammer is a "
        "head + a handle; a gripper is a base + fingers). A GRASPABLE part (a handle, pull, knob, "
        "lever, grip) mounted ON a base/plate/backplate is functional: the graspable part + the "
        "mounting part. Pick functional ONLY when the object is naturally several parts; otherwise "
        "cosmetic.",
    )


class SpecBrief(DesignBrief):
    """A DesignBrief whose features carry the slice-2 structured fields (SpecGraph, ADR-0088).

    Selected as the researcher's output model ONLY under FORGE_SPEC_BRIEF; a SpecBrief IS-A
    DesignBrief and its features ARE BriefFeatures, so the author / validator / every existing
    consumer treat it exactly as today -- the structured fields are additive and read only by
    spec-aware code once the emission gate has cleared."""

    # Narrowing the element type is intentional (SpecFeature IS-A BriefFeature) and pydantic
    # validates it at runtime; mypy sees only list invariance, so the override is annotated.
    features: list[SpecFeature] = Field(..., min_length=1, max_length=12)  # type: ignore[assignment]


# Appended to the researcher system prompt ONLY under the flag: ask for the structured fields, and
# bind them to the anti-tautology rule (restate the prose, never invent).
_SPEC_BRIEF_ADDENDUM = """

STRUCTURED FIELDS (in addition to everything above):
- feature_id: give EACH feature a short, stable, UNIQUE id (snake_case), e.g. 'case_mount_holes',
  'side_mount_holes'. Two features that differ only by where they sit MUST get different ids.
- spec_dims: for each feature, restate every NUMBER already in its approx_dims_mm as a typed dim
  {role, value_mm, of}. role is one of diameter/length/width/height/depth/thickness/pitch/pcd/angle.
  RESTATE ONLY numbers the prose already contains -- do not invent, round, or add a number that is
  not in approx_dims_mm. If a feature's prose has no numbers, leave its spec_dims empty."""


_SYSTEM = """You are a senior mechanical product designer. You RESEARCH a real-world object \
and turn it into a precise build plan a CAD engineer will model in build123d.

Given a part request, produce a DesignBrief that COPIES the real object's form:
- object_name: name the real thing precisely.
- summary: one line.
- visual_description: what the finished part LOOKS like from the outside -- silhouette, \
proportions, what is round vs flat vs tapered, where it is THIN. A reviewer compares a \
render against this, so be concrete and visual.
- overall_envelope_mm: realistic overall size L x W x H in mm using REAL reference \
dimensions (a claw hammer is ~330mm long; a phone stand ~100mm tall).
- features: the functional parts. For EACH give name, purpose (what it does), reference \
dimensions in mm, and the build123d METHOD that makes it the RIGHT shape -- revolve for \
round/forged, loft for organic transitions, sweep for curved members, pocket for \
recesses/panels, shell for hollow, box+fillet for blocks. Prefer methods that yield THIN, \
detailed geometry over solid blocks. For a REPEATED feature (bolt holes, panels, fins, slots) \
set its COUNT (e.g. four bolt holes -> count 4). When a feature mates a STANDARD part (a bolt, \
bearing, hex nut, dowel, magnet), set its `standard` to the designation (e.g. 'M3', '608') so the \
modeler uses the exact ISO/DIN dimension, never a guess. BUT when the prompt STATES a hole \
diameter as a number ('pivot bore 6 mm diameter'), that number is EXACT -- carry it in the dims \
and set NO fastener standard for it (tagging a stated 6 mm bore 'M6' silently redrills it to the \
6.6 mm clearance size; measured failure, bench row 17). A THREADED / tapped / set-screw hole \
('M8 threaded bore', 'M4 set-screw') keeps its M-designation and is modeled at the thread's \
NOMINAL major diameter (M8 -> 8 mm), NOT the tap drill or clearance -- keep the word \
'threaded'/'tapped'/'set-screw' in the feature name so the modeler picks nominal (bench row 18).
- min_wall_mm: the thinnest wall/rib that still prints (2-3mm for FDM PLA unless the object \
genuinely demands thinner).
- print_orientation_note: which face goes down to print with the fewest supports.
- target: 'cosmetic' for a single monolithic object (a vase, bracket, panel); 'functional' \
for a tool/mechanism that is genuinely SEVERAL separate parts (a hammer = head + handle; a gripper \
= base + fingers; pliers = two arms + a pivot). A GRASPABLE part -- a handle, pull, knob, lever, or \
grip -- mounted ON a base/plate/backplate is 'functional' (the graspable part + the mounting part), \
the SAME head+handle structure as the hammer. But a graspable object ALONE -- a knob, a handle, a \
pull that IS the whole request, with no second part it mounts on -- is ONE body: 'cosmetic'. Pick \
'functional' ONLY when the object is naturally multi-part -- when forcing it into one fused body \
would fight its real construction.

Research the REAL object: real proportions, real thinness, real features. Do NOT simplify \
it into a block -- name the curves, tapers, claws, flutes, panels, and ribs the real object \
has. If the object is novel (not a standard product), plan it from function: what must it \
do, and what shape achieves that.

NEVER decline a part for lack of a published standard or spec -- plan the best honest \
attempt. For every feature whose dims are NOT standard-backed, self-report HOW you derived \
them via estimate_basis + estimate_anchor: 'anchor_scaled' (scaled from a named real \
reference -- name it), 'functional_requirement' (forced by a constraint -- name it), \
'proportion_rule' (a stated convention), or 'default_guess' (it looked right). Be honest: \
the system labels the part's confidence from EVIDENCE; your self-report explains the \
reasoning but cannot upgrade it."""


# ---- SpecGraph slice 2 emission-validity scoring (the anti-tautology gate) ----
_PROSE_NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def spec_dim_agrees(dim: SpecDim, prose: str) -> bool:
    """True when the spec_dim's value appears in the feature's own prose (a faithful restatement,
    not an invented number). The whole gate rests on this: a structured field that does not trace
    to the prose is the 'second liar' and must never reach a verdict."""
    nums = [float(n) for n in _PROSE_NUM_RE.findall(prose or "")]
    return any(abs(dim.value_mm - n) <= 0.05 for n in nums)


def feature_spec_agreement(feature: BriefFeature) -> tuple[int, int]:
    """(agreed spec_dims, total spec_dims) for one feature; (0, 0) when it emitted none."""
    dims = getattr(feature, "spec_dims", None) or []
    prose = str(getattr(feature, "approx_dims_mm", "") or "")
    return sum(1 for d in dims if spec_dim_agrees(d, prose)), len(dims)


# Words that signal a genuine SECOND part the graspable piece mounts on. Deliberately TIGHTER than
# _MOUNT_WORDS below: 'body' is excluded because every single-piece object has a "body" feature --
# it is not evidence of a second part (the knob misroute: 'knob body' must not read as a mount).
_SECOND_PART_WORDS = ("base", "plate", "backplate", "mount", "stand", "bracket")


def _downgrade_lone_graspable(brief: DesignBrief) -> DesignBrief:
    """No-false-route guard: a graspable object that IS the whole request stays ONE body.

    The researcher's target rule ("a knob/handle/pull mounted ON a base is functional") measurably
    over-triggers on the WORD alone -- the bench's 25mm knob routed to the multi-part assembler in
    2 of 3 trials and died there (0/3). Deterministic downgrade: when the OBJECT NAME itself is the
    graspable thing and NO feature names a second part it mounts on, target becomes 'cosmetic'.
    A real multi-part object is untouched: the hammer's name ('claw hammer') is not graspable, and
    a 'door pull' WITH a backplate feature keeps its second part.
    """
    if brief.target != "functional":
        return brief
    name = brief.object_name.lower()
    if not any(g in name for g in _GRASPABLE_WORDS):
        return brief
    feature_names = [f.name.lower() for f in brief.features]
    if any(any(w in fn for w in _SECOND_PART_WORDS) for fn in feature_names):
        return brief
    return brief.model_copy(update={"target": "cosmetic"})


def research_brief(
    prompt: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    extra_context: str | None = None,
) -> tuple[DesignBrief, LLMUsage]:
    """Research the requested object and emit a :class:`DesignBrief` (one cheap call).

    ``extra_context`` (optional) is reference material -- retrieved specs / dimensions --
    injected verbatim for the model to copy from. Wave 1 leaves it None.
    """
    client = client or make_client()
    user = f"Part request:\n{prompt.strip()}\n"
    if extra_context:
        user += f"\nReference material you may use:\n{extra_context.strip()}\n"
    user += "\nProduce the DesignBrief that copies the real object."
    # SpecGraph slice 2 (flag-gated): emit a SpecBrief (typed spec_dims + feature_id) with the
    # addendum. OFF -> the exact call as today (plain DesignBrief), byte-identical.
    spec_on = spec_brief_enabled()
    brief, usage = client.generate_structured(
        system=_SYSTEM
        + (_SPEC_BRIEF_ADDENDUM if spec_on else "")
        + (_FIDELITY_BRIEF_ADDENDUM if fidelity_brief_enabled() else ""),
        user=user,
        output_model=SpecBrief if spec_on else DesignBrief,
        model=model or SONNET,
        max_tokens=BRIEF_MAX_TOKENS,  # generous by design -- see the config.py rationale
    )
    # Single chokepoint for BOTH routes (produce_part and the loop's own research): the
    # lone-graspable downgrade runs here so no caller can route a one-piece object multi-part.
    return _downgrade_lone_graspable(brief), usage


# ----------------------------------------------------------------------------
# Brief coherence: a CHEAP, no-LLM sanity pass over a DesignBrief. These are
# WARNINGS (never raised) for the author/orchestrator to read -- they catch a
# brief that contradicts itself BEFORE the kernel spends time building it. They
# FAIL-OPEN on anything unparseable so a noisy field never produces a false alarm.
# ----------------------------------------------------------------------------

# "60 x 60 x 40", "60x60x40mm", "Ø60 x 40", "120 by 30 by 15" -- three numbers
# separated by x / X / by / *, with optional units/noise around them.
_NUM = r"(\d+(?:\.\d+)?)"
_SEP = r"\s*(?:x|by|\*)\s*"
# An optional axis letter trailing a dimension: the brief routinely writes
# "120 L x 40 W x 85 H mm" (measured: ~half the arbitrary_v1 corpus). Without this the
# envelope gate parsed None and FAILED OPEN on those parts -- a real no-false-PASS hole
# the ADR-0083 audit found (envelope escapes 45%; the L/W/H briefs were the escapes).
# Per-number units ("110 mm L x 80 mm W x 17.5 mm H") are equally real -- row 50's
# golden_ssd_bracket k4 false-FAIL: that exact form failed to match, so the scanner slid
# past the primary statement and latched onto a compact "110x80x4" triple inside the
# trailing rationale, gating a correct ~17.5mm-tall part against Z~4mm. With per-number
# units allowed, leftmost-match-wins lands on the primary statement again.
_UNIT = r"(?:\s*mm\.?)?"
_AXIS = r"(?:\s*[LWHD])?"
_DIM = _NUM + _UNIT + _AXIS
_ENVELOPE_RE = re.compile(_DIM + _SEP + _DIM + _SEP + _DIM, re.IGNORECASE)
# The ROUND-part form the researcher actually writes: "76 mm dia x 14 mm tall",
# "Ø76 x 14 high". Measured failure (timing-gear run): this parsed as None, the
# envelope loss term vanished, and NOTHING anchored the part's absolute size.
_ENVELOPE_DIA_RE = re.compile(
    r"(?:"
    + _NUM
    + r"\s*(?:mm\s*)?dia(?:meter)?\.?|[øØ⌀]\s*"
    + _NUM
    + r")[^0-9]{0,24}?"
    + _NUM
    + r"\s*(?:mm\s*)?(?:tall|high|height|thick|long|deep)",
    re.IGNORECASE,
)


def parse_envelope_mm(text: str) -> tuple[float, float, float] | None:
    """Parse an ``L x W x H`` triple in mm out of free text, tolerant of trailing rationale.

    Accepts the forms the brief actually emits -- "60 x 60 x 40 mm", "60x60x40mm",
    "120 by 30 by 15 (handle + base)" -- and ignores anything wrapped around the numbers.
    Returns ``(l, w, h)`` as floats, or ``None`` when no clean three-number triple is found
    (the caller treats None as "could not check", never as a failure).
    """
    if not text:
        return None
    m = _ENVELOPE_RE.search(text)
    if m is not None:
        try:
            dims = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
        except (TypeError, ValueError):
            return None
        if any(d <= 0 for d in dims):
            return None
        return dims
    d = _ENVELOPE_DIA_RE.search(text)
    if d is None:
        return None
    try:
        dia = float(d.group(1) if d.group(1) is not None else d.group(2))
        height = float(d.group(3))
    except (TypeError, ValueError):
        return None
    if dia <= 0 or height <= 0:
        return None
    return (dia, dia, height)  # a round part: dia x dia x height


# Graspable parts and the mounting parts they sit on -- the head+handle structure
# the classifier calls 'functional'. A brief carrying BOTH while marked 'cosmetic'
# is the measured mislabel (a teardrop pull on a round base built as one body).
_GRASPABLE_WORDS = ("handle", "pull", "knob", "lever", "grip")
_MOUNT_WORDS = ("base", "plate", "backplate", "mount", "body", "stand", "bracket")


def _max_dim_mm(text: str) -> float | None:
    """The largest single number (mm) appearing in a free-text dimension string, or None."""
    nums = [float(n) for n in re.findall(_NUM, text or "")]
    nums = [n for n in nums if n > 0]
    return max(nums) if nums else None


def _standard_in_registry(designation: str) -> bool | None:
    """Is ``designation`` a key in ANY standards registry? None = cannot tell (fail-open).

    Recognizes the families the brief uses (a fastener 'M3', a bearing '608', a dowel '4', a
    magnet '6x3'). Returns True/False only when the FIRST token clearly looks like one of those
    families; returns None for free text we cannot confidently classify, so noisy designations
    never produce a false warning.
    """
    from forge_mind.dsl.standards import ALL_REGISTRIES

    token = designation.strip().split()[0] if designation.strip() else ""
    if not token:
        return None
    # Normalize the fastener prefix to the registry's capital-M key ('m3' -> 'M3').
    norm = token
    if re.fullmatch(r"[mM]\d+(?:\.\d+)?", token):
        norm = "M" + token[1:]
    # Only judge tokens that match a known family shape; otherwise fail-open.
    looks_known = bool(
        re.fullmatch(r"M\d+(?:\.\d+)?", norm)  # fastener / standoff / hex-nut / washer
        or re.fullmatch(r"\d+", norm)  # bearing / dowel / retaining ring
        or re.fullmatch(r"\d+x\d+", norm.lower())  # magnet
    )
    if not looks_known:
        return None
    key = norm.lower() if "x" in norm.lower() else norm
    return any(reg.has(key) for reg in ALL_REGISTRIES.values())


def validate_brief(brief: DesignBrief) -> list[str]:
    """Return coherence WARNINGS for a :class:`DesignBrief` (never raises, never fails-closed).

    Conservative -- it flags only CLEAR self-contradictions a senior reviewer would catch at a
    glance, and fails OPEN (emits nothing) on any field it cannot parse:

      1. ``min_wall_mm`` below the resolved material's manufacturable floor.
      2. ``overall_envelope_mm`` grossly smaller than the largest feature's ``approx_dims_mm``.
      3. a feature ``standard`` designation that is not in the standards registry.
      4. an obviously-multipart object (a graspable part ON a mount) marked ``target='cosmetic'``.
    """
    from forge_mind.materials import resolve_material

    warnings: list[str] = []

    # 1. min_wall below the material floor.
    try:
        mat = resolve_material(brief.material)
        if brief.min_wall_mm < mat.min_wall_mm:
            warnings.append(
                f"min_wall_mm {brief.min_wall_mm:.2f} is below {mat.name}'s "
                f"manufacturable floor {mat.min_wall_mm:.2f}mm -- thin walls/ribs will not "
                "reliably form; raise min_wall_mm or pick a finer-detail material."
            )
    except Exception:
        pass  # fail-open: never a false warning on an odd material string

    # 2. envelope grossly inconsistent with the largest feature.
    env = parse_envelope_mm(brief.overall_envelope_mm)
    if env is not None:
        env_max = max(env)
        feat_max = 0.0
        worst = None
        for f in brief.features:
            d = _max_dim_mm(f.approx_dims_mm)
            if d is not None and d > feat_max:
                feat_max, worst = d, f.name
        # only flag a GROSS overrun (feature > 1.5x the whole envelope) to stay conservative.
        if worst is not None and feat_max > env_max * 1.5:
            warnings.append(
                f"feature {worst!r} is {feat_max:.0f}mm but the overall envelope's largest "
                f"dimension is only {env_max:.0f}mm -- the envelope cannot contain the feature; "
                "reconcile overall_envelope_mm with the feature dimensions."
            )

    # 3. unknown 'standard' designation.
    for f in brief.features:
        if not f.standard:
            continue
        known = _standard_in_registry(f.standard)
        if known is False:
            warnings.append(
                f"feature {f.name!r} cites standard {f.standard!r}, which is not in the "
                "standards registry -- the modeler cannot fetch its exact ISO/DIN dimension; "
                "use a known designation or drop the standard tag."
            )

    # 4. obviously-multipart object marked cosmetic.
    if brief.target == "cosmetic":
        names = [f.name.lower() for f in brief.features]
        has_graspable = any(any(g in n for g in _GRASPABLE_WORDS) for n in names)
        has_mount = any(any(mw in n for mw in _MOUNT_WORDS) for n in names)
        if has_graspable and has_mount:
            warnings.append(
                "target is 'cosmetic' but the features describe a graspable part mounted on a "
                "base/mount -- that is a multi-part (functional) object; a single fused body will "
                "drop the graspable part. Consider target='functional'."
            )

    return warnings


__all__ = [
    "BriefFeature",
    "DesignBrief",
    "parse_envelope_mm",
    "research_brief",
    "validate_brief",
]
