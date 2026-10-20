"""The Planner Agent: understand the request, decide what is known vs. missing, ask only the
questions that matter, and (when enough is known) emit a draft ``DesignBrief``.

Two brains implement one ``PlannerBrain`` Protocol so the LLM is swappable and the whole flow is
testable offline:

  - ``HeuristicPlannerBrain`` -- deterministic, NO LLM. Classifies the part by keywords, cites
    standards from the catalog, and selects questions. This is the CI default (mockable: it needs no
    key) and a genuinely useful fallback when no key is configured.
  - ``LlmPlannerBrain`` -- optional. Uses the LLM ONLY to UNDERSTAND (classify the part, extract which
    catalog standards apply), then delegates question selection + brief assembly to the SAME
    deterministic core. The LLM never formats the question schema or invents standards/dimensions
    (the repo's "LLM picks categorical, code derives" rule).

Both brains funnel through ``_plan_from_state`` so the question/brief logic lives in exactly one place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import SONNET
from forge_mind.intake.brief import all_request_text, assemble_brief, parse_measurements
from forge_mind.intake.constraints import infer_constraints
from forge_mind.intake.questions import q_image_scale, questions_for_part_type
from forge_mind.intake.schema import (
    ConstraintKind,
    DesignBrief,
    Feature,
    InferredConstraint,
    KnownStandard,
    Option,
    PlannerState,
    PreviewBehavior,
    Question,
    QuestionType,
    ScalarValue,
    Viewpoint,
)
from forge_mind.intake.session import DesignSession
from forge_mind.intake.standards import detect_standards, known_standards
from forge_mind.llm.cost import LLMUsage

if TYPE_CHECKING:
    from forge_mind.llm.provider import LLMClientLike

# Ask a small set; never overwhelm. Pad with optional questions only when the required set is small.
_MAX_QUESTIONS = 4
_MIN_BEFORE_PAD = 3

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class PlannerOutcome:
    """The result of one planner pass: the updated state, the questions to ask (empty when ready or
    refused), whether the brief is ready, the (draft or final) brief, an optional refusal/note, and
    any LLM usage incurred."""

    state: PlannerState
    questions: list[Question] = field(default_factory=list)
    ready: bool = False
    brief: DesignBrief | None = None
    refusal: str | None = None
    note: str | None = None
    usages: list[LLMUsage] = field(default_factory=list)
    # Structure-first interpretation surfaced to the UI alongside the questions (ADR-0025 result
    # contract). Both mirror ``state`` so the API payload can render them as first-class panels.
    features: list[Feature] = field(default_factory=list)
    constraints: list[InferredConstraint] = field(default_factory=list)


class PlannerBrain(Protocol):
    """Anything that can turn a session into a PlannerOutcome. Mockable for tests."""

    def analyze(self, session: DesignSession) -> PlannerOutcome: ...


# --------------------------------------------------------------------------------------------------
# Shared deterministic core (used by every brain)
# --------------------------------------------------------------------------------------------------


def classify_part_type(text: str | None, standards: list[KnownStandard]) -> str:
    """Classify the part family from the request + the standards already detected. Standards win
    (a NEMA mention => motor_mount) over loose keywords."""
    names = {s.name for s in standards}
    if names & {"NEMA-17", "NEMA-23"}:
        return "motor_mount"
    if any(n.endswith("bearing") for n in names):
        return "bearing_block"
    if any(n.endswith("fan") for n in names):
        return "fan_mount"

    t = (text or "").lower()
    keyword_map: list[tuple[tuple[str, ...], str]] = [
        (("motor mount", "stepper", "motor bracket"), "motor_mount"),
        (("adapter plate", "adapter"), "adapter_plate"),
        (("bearing",), "bearing_block"),
        (("fan",), "fan_mount"),
        (("enclosure", "case", "housing", "box"), "enclosure"),
        (("standoff", "spacer"), "standoff"),
        (("bracket", "mount", "holder", "clamp"), "bracket"),
        # Shape primitives -- LOWEST priority (a functional keyword above wins, so "adapter plate"
        # stays adapter_plate). Lets a plain "a 1 m cube" / "an 80 mm plate" get a concrete type + the
        # minimal primitive question set instead of falling through to generic.
        (("cube",), "cube"),
        (("plate", "sheet"), "plate"),
        (("cylinder", "rod", "shaft"), "cylinder"),
        (("tube", "pipe"), "tube"),
        (("disc", "disk"), "disc"),
        (("ring", "washer", "annulus"), "ring"),
        (("sphere", "ball"), "sphere"),
        (("cone",), "cone"),
        (("block", "bar"), "block"),
    ]
    for keywords, part_type in keyword_map:
        if any(k in t for k in keywords):
            return part_type
    return "generic"


def _has_explicit_screw(text: str | None) -> bool:
    return bool(text and (re.search(r"\bm[2-8]\b", text.lower()) or "1/4" in text.lower()))


def _resolved_by_standard(question_id: str, std_names: set[str], word_count: int) -> bool:
    has_nema = bool(std_names & {"NEMA-17", "NEMA-23"})
    has_extrusion = any("extrusion" in n for n in std_names)
    has_bearing = any(n.endswith("bearing") for n in std_names)
    has_fan = any(n.endswith("fan") for n in std_names)
    non_surface_object = has_nema or has_bearing or has_fan or ("Raspberry Pi 4" in std_names)
    return {
        "q_motor_size": has_nema,
        "q_mounting_surface": has_extrusion,
        "q_fastener_type": has_extrusion,
        "q_bearing_size": has_bearing,
        "q_fan_size": has_fan,
        # The object is named when it's a known device, or the request is already descriptive.
        "q_object_being_mounted": non_surface_object or word_count >= 8,
    }.get(question_id, False)


# Families where naming two members at once is a genuine same-feature ambiguity (you mount ONE motor
# on a motor mount), so the planner REFUSES rather than guessing. ONLY mutually-exclusive single-object
# families belong here: extrusion profiles and fan sizes are deliberately absent because an adapter
# plate or a dual-fan shroud legitimately bridges two of them. Extend the table, not the call sites.
_EXCLUSIVE_STANDARD_GROUPS: list[tuple[str, frozenset[str]]] = [
    ("motor frame size", frozenset({"NEMA-17", "NEMA-23"})),
]

# Unambiguous words for a cut-out. "diameter"/"dia" are OMITTED on purpose: a disc/cylinder BODY has a
# diameter too, so keying on them would false-refuse "an 80 mm diameter disc with a 5 mm hole".
_HOLE_WORDS = ("bore", "hole")
_BODY_WORDS = ("plate", "wide", "width", "square", "block", "cube", "body", "stock", "base", "disc")


def _conflicting_standards(std_names: set[str]) -> str | None:
    """Refuse when two members of one mutually-exclusive standard family are both named."""
    for label, group in _EXCLUSIVE_STANDARD_GROUPS:
        named = sorted(group & std_names)
        if len(named) >= 2:
            return (
                f"The request names two different {label}s ({' and '.join(named)}). "
                "Pick one so the geometry is unambiguous."
            )
    return None


def _mm_near(text: str, words: tuple[str, ...]) -> float | None:
    """The explicit '<n> mm' value adjacent to any of ``words`` (number before OR after), else None.
    Conservative: requires a literal millimetre value next to the keyword -- it never guesses."""
    word_re = "|".join(words)
    patterns = (
        rf"(\d+(?:\.\d+)?)\s*mm\b[^.\d]{{0,16}}\b(?:{word_re})\b",
        rf"\b(?:{word_re})\b[^.\d]{{0,16}}(\d+(?:\.\d+)?)\s*mm\b",
    )
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return float(m.group(1))
    return None


def _impossible_dimension(text: str | None) -> str | None:
    """Refuse ONE clear geometric impossibility: a hole/bore at least as large as the body it is cut
    into, when BOTH are stated explicitly in mm. Narrow by design -- deeper feasibility is caught
    downstream by the conformance gate; this just stops the obvious silent-wrong before it generates."""
    if not text:
        return None
    t = text.lower()
    hole = _mm_near(t, _HOLE_WORDS)
    body = _mm_near(t, _BODY_WORDS)
    if hole is not None and body is not None and hole >= body:
        return (
            f"A {hole:g} mm hole cannot fit in a {body:g} mm body. "
            "Lower the hole size or raise the outer size so the part is buildable."
        )
    return None


def _contradiction(session: DesignSession, std_names: set[str]) -> str | None:
    """Hard contradictions the planner REFUSES rather than asks about: two mutually-exclusive standards
    of one single-object family, or an explicitly-impossible dimension. Small and honest by design --
    a same-feature-two-values detector, never a constraint solver."""
    return _conflicting_standards(std_names) or _impossible_dimension(session.original_text_prompt)


def _is_number_str(s: str) -> bool:
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def _attach_preview_behavior(questions: list[Question]) -> list[Question]:
    """Deterministically tag geometry-affecting questions with preview metadata so the frontend shows
    a live 3D preview as the user hovers options. Heuristic + presentation-only: it never changes WHICH
    questions are asked, only annotates the dimensional ones (a question already carrying a
    preview_behavior is left untouched)."""
    out: list[Question] = []
    for q in questions:
        numeric_options = bool(q.options) and all(_is_number_str(o.value) for o in q.options)
        is_dim = q.unit == "mm" or q.type == QuestionType.NUMBER or numeric_options
        if q.preview_behavior is not None or not is_dim:
            out.append(q)
            continue
        title = (q.title or "").lower()
        if any(k in title for k in ("diameter", "bore", "hole")):
            ptype = "hole_diameter"
        elif "radius" in title:
            ptype = "radius"
        elif any(k in title for k in ("length", "long", "tall", "deep", "depth")):
            ptype = "length"
        else:
            ptype = "extrusion_thickness"
        out.append(
            q.model_copy(
                update={
                    "dimension_id": q.dimension_id or q.id,
                    "preview_behavior": PreviewBehavior(type=ptype, ghost_preview=True),
                }
            )
        )
    return out


def _plan_from_state(
    session: DesignSession,
    state: PlannerState,
    llm_questions: list[Question] | None = None,
) -> PlannerOutcome:
    """Given a classified state, pick the questions to ask (or decide ready) -- deterministic.

    When ``llm_questions`` is provided (the LLM brain authored idea-specific questions for this exact
    request) those become the candidate set; otherwise we fall back to the part-type templates. Either
    way the gate is identical: ready once every required candidate is answered (plus the image-scale
    rule), so generation is never blocked on a question the user has no way to answer."""
    std_names = {s.name for s in state.known_standards}
    word_count = len((session.original_text_prompt or "").split())
    answered = {a.question_id for a in session.answers}
    # Suppression reads EVERY channel (typed + speech + notes + labels), so a value stated in any of
    # them counts as already-given -- we never re-ask for something the capture already supplied.
    request_text = all_request_text(session)

    # Fold deterministic sketch constraints into the state -- idempotent (merge + dedupe with any the
    # vision model already proposed). Pure + cheap; runs on every pass so both brains benefit.
    state.inferred_constraints = _merge_constraints(
        state.inferred_constraints, _sketch_constraints(session)
    )

    # llm_questions is None only on the heuristic (no-LLM) path -> use the part-type templates.
    # When the LLM brain ran, use ITS questions even if empty (empty = nothing to ask, ready to
    # generate) so we never fall back to generic template questions.
    candidates = (
        llm_questions
        if llm_questions is not None
        else questions_for_part_type(state.part_type or "generic")
    )

    def resolved(q: Question) -> bool:
        if q.id in answered:
            return True
        if q.id == "q_fastener_type" and _has_explicit_screw(request_text):
            return True
        return _resolved_by_standard(q.id, std_names, word_count)

    required_unresolved = sorted(
        (q for q in candidates if q.required and not resolved(q)), key=lambda q: q.priority
    )
    optional_unresolved = sorted(
        (q for q in candidates if not q.required and not resolved(q)), key=lambda q: q.priority
    )

    # A drawing/image is shape-only: ask for one real dimension before generating (unless the user
    # already gave a measurement, confirmed scale, or answered the scale question).
    scale_known = (
        "q_image_scale" in answered
        or bool(parse_measurements(request_text))
        or (session.capture is not None and session.capture.scale is not None)
        or any(a.scale_confirmed for a in session.uploaded_assets)
        or any(s.scale_confirmed for s in session.sketch_assets)
    )
    if session.has_visual_input() and not scale_known:
        required_unresolved.insert(0, q_image_scale())

    state.missing_critical = [q.id for q in required_unresolved]

    if not required_unresolved:
        brief = assemble_brief(session, state, ready=True)
        return PlannerOutcome(
            state=state,
            questions=[],
            ready=True,
            brief=brief,
            features=list(state.features),
            constraints=list(state.inferred_constraints),
        )

    asked = required_unresolved[:_MAX_QUESTIONS]
    if len(required_unresolved) < _MIN_BEFORE_PAD:
        asked = (asked + optional_unresolved)[:_MAX_QUESTIONS]

    # Annotate the dimensional questions so the frontend can render a live 3D preview (additive).
    asked = _attach_preview_behavior(asked)

    draft = assemble_brief(session, state, ready=False)
    note = None
    if not (session.original_text_prompt or "").strip() and not session.has_visual_input():
        note = "Tell me what you want to make, or upload a drawing -- a few quick questions first."
    return PlannerOutcome(
        state=state,
        questions=asked,
        ready=False,
        brief=draft,
        note=note,
        features=list(state.features),
        constraints=list(state.inferred_constraints),
    )


# --------------------------------------------------------------------------------------------------
# Brains
# --------------------------------------------------------------------------------------------------


class HeuristicPlannerBrain:
    """Deterministic, no-LLM planner. The CI default and the no-key fallback."""

    def analyze(self, session: DesignSession) -> PlannerOutcome:
        text = session.original_text_prompt
        standards = detect_standards(text)
        std_names = {s.name for s in standards}

        contradiction = _contradiction(session, std_names)
        prev_rounds = session.planner_state.rounds if session.planner_state else 0
        part_type = classify_part_type(text, standards)
        state = PlannerState(
            classified=True,
            part_type=part_type,
            user_goal=(text or None),
            known_standards=standards,
            detected_interfaces=sorted(std_names),
            detected_fasteners=[],
            rounds=prev_rounds + 1,
        )
        if contradiction is not None:
            return PlannerOutcome(state=state, questions=[], ready=False, refusal=contradiction)
        return _plan_from_state(session, state)


class _LlmChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(..., description="Human-facing label, e.g. '8 spokes'.")
    value: str = Field(..., description="Stable machine value, e.g. '8'.")


class _LlmQuestion(BaseModel):
    """A clarifying question the planner authored for THIS specific request. Categorical/typed only --
    the deterministic layer renders + gates it; the LLM never picks a final dimension, it only asks."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="snake_case id, e.g. 'q_spoke_count'.")
    type: str = Field(..., description="one of: multiple_choice, dropdown, number, text, boolean.")
    title: str = Field(..., description="The question shown to the user.")
    options: list[_LlmChoice] = Field(
        default_factory=list, description="Required for multiple_choice/dropdown; else empty."
    )
    unit: str | None = Field(default=None, description="For number questions, e.g. 'mm'.")
    required: bool = Field(
        default=True, description="True only if the part cannot be built correctly without it."
    )
    recommended: str | None = Field(
        default=None,
        description=(
            "ALWAYS set this: the single best default for a typical user. For multiple_choice/dropdown "
            "it MUST exactly equal one of the option values; for number, a typical numeric value as a "
            "string; for boolean, 'true' or 'false'. Lets the user accept smart defaults and build now."
        ),
    )
    reason: str | None = Field(default=None, description="Why this matters to the part.")


class _LlmFeature(BaseModel):
    """One topological feature the planner identified -- structure, not numbers. Dimensions stay
    SYMBOLIC (names) so the model never invents a value to fill a structural gap (ADR-0025)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Short id, e.g. 'f_base', 'f_bore'.")
    kind: str = Field(
        ...,
        description="base_extrude | hole | counterbore | fillet | chamfer | boss | slot | rib | "
        "pocket | shell | gear | revolve.",
    )
    human_label: str = Field(..., description="Plain-English name shown to the user.")
    relationships: list[str] = Field(
        default_factory=list,
        description="Ties to other features by id: 'on_face_of:f_base', 'concentric_with:f_boss', "
        "'through_all', 'mirror_of:f_left'.",
    )
    symbolic_dims: list[str] = Field(
        default_factory=list,
        description="NAMED (never numeric) dimensions, e.g. 'plate_thickness', 'bore_d'.",
    )


class _LlmConstraint(BaseModel):
    """An intentional geometric relationship among features the build must preserve."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(
        ...,
        description="symmetry | equal | concentric | perpendicular | parallel | coincident | "
        "horizontal | vertical | tangent.",
    )
    refs: list[str] = Field(
        default_factory=list, description="Feature ids the relationship ties together."
    )


class _Understanding(BaseModel):
    """The LLM's understanding of the request -- structure-first + categorical (topology and named
    relationships, never final dimensions: code/questions bind the numbers)."""

    model_config = ConfigDict(extra="forbid")

    part_type: str = Field(
        ...,
        description=(
            "The part family. A FUNCTIONAL type when it mounts/holds/adapts something: motor_mount / "
            "bracket / adapter_plate / bearing_block / fan_mount / enclosure / standoff. A SHAPE "
            "PRIMITIVE for a plain geometric solid: cube / plate / cylinder / tube / disc / ring / "
            "sphere / cone / block. Pick the most specific (a '1 m cube' is 'cube'); use 'generic' "
            "only as a last resort."
        ),
    )
    user_goal: str = Field(..., description="One-line restatement of what the user wants.")
    standard_names: list[str] = Field(
        default_factory=list,
        description="Names of catalog standards that apply (must match the catalog exactly; never invent).",
    )
    questions: list[_LlmQuestion] = Field(
        default_factory=list,
        description=(
            "The FEWEST clarifying questions that genuinely change the part (aim for 2, max 3), SPECIFIC "
            "to this request (for a wheel: spoke count, hub bore; for a gear: tooth count, module). "
            "PREFER multiple_choice with concrete options the user picks in one click; use number only "
            "when a measurement is unavoidable. EVERY question MUST set `recommended` (a real default). "
            "Mark required only when the part cannot be built correctly without it. Empty if the request "
            "already pins down everything."
        ),
    )
    features: list[_LlmFeature] = Field(
        default_factory=list,
        description=(
            "STRUCTURE FIRST: the ordered topological feature list (what features exist + how they "
            "connect) BEFORE any dimensions. Build order: base -> additive -> subtractive -> "
            "finishing. Keep every dimension SYMBOLIC (a name like 'bore_d', never a number). This is "
            "the spine of the part; fixing topology before numbers roughly halves downstream error."
        ),
    )
    constraints: list[_LlmConstraint] = Field(
        default_factory=list,
        description=(
            "Intentional geometric relationships among the features -- right angles, equal sizes, "
            "shared centers, symmetry -- the build MUST preserve so the part survives editing. "
            "Reference features by their ids."
        ),
    )
    refusal: str | None = Field(
        default=None,
        description="A clear blocker if the request is impossible/contradictory; else null.",
    )


_ALLOWED_LLM_TYPES = {"multiple_choice", "dropdown", "number", "text", "boolean"}


def _recommended_default(
    kind: str, recommended: str | None, opts: list[Option]
) -> ScalarValue | None:
    """Coerce the LLM's ``recommended`` string into a typed ``default_value`` for the question.

    Choice questions always get a default (the recommended option, or the first one as a fallback) so a
    one-click "use recommended" path always exists. Numbers/booleans parse leniently; anything
    unparseable yields None (the question simply renders without a default)."""
    rec = (recommended or "").strip()
    if kind in ("multiple_choice", "dropdown"):
        if rec and any(o.value == rec for o in opts):
            return rec
        return opts[0].value if opts else None
    if kind == "number":
        try:
            return float(rec)
        except (TypeError, ValueError):
            return None
    if kind == "boolean":
        low = rec.lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
        return None
    return rec or None  # text


def _convert_llm_questions(items: list[_LlmQuestion]) -> list[Question]:
    """Validate + convert LLM-authored questions into real typed ``Question`` objects. Defensive: any
    malformed item is skipped, ids are normalised + de-duped, choice types must carry options, and the
    set is capped at ``_MAX_QUESTIONS``. Returns [] when nothing usable -> the planner falls back to
    the deterministic templates, so a flaky LLM never breaks the flow."""
    out: list[Question] = []
    seen: set[str] = set()
    for i, lq in enumerate(items or []):
        if len(out) >= _MAX_QUESTIONS:
            break
        kind = (lq.type or "").strip().lower()
        if kind not in _ALLOWED_LLM_TYPES:
            continue
        qid = re.sub(r"[^a-z0-9_]", "_", (lq.id or "").strip().lower())[:64] or f"q_detail_{i}"
        if not qid.startswith("q_"):
            qid = f"q_{qid}"[:64]
        if qid in seen:
            continue
        title = (lq.title or "").strip()
        if not title:
            continue
        opts = [
            Option(label=o.label.strip(), value=o.value.strip())
            for o in lq.options
            if o.label.strip() and o.value.strip()
        ]
        if kind in ("multiple_choice", "dropdown") and not opts:
            continue
        try:
            q = Question(
                id=qid,
                type=QuestionType(kind),
                title=title,
                required=bool(lq.required),
                options=opts,
                unit=lq.unit if kind == "number" else None,
                default_value=_recommended_default(kind, lq.recommended, opts),
                reason=lq.reason,
                affects=["geometry"],
                priority=1 if lq.required else 3,
            )
        except Exception:
            continue
        out.append(q)
        seen.add(qid)
    return out


_ALLOWED_CONSTRAINT_KINDS = {k.value for k in ConstraintKind}


def _convert_llm_features(items: list[_LlmFeature]) -> list[Feature]:
    """Validate + convert LLM-proposed features into typed ``Feature`` objects. Defensive: malformed
    items are skipped, ids normalised + de-duped, dims/relationships capped, the set capped at 12."""
    out: list[Feature] = []
    seen: set[str] = set()
    for i, lf in enumerate(items or []):
        if len(out) >= 12:
            break
        fid = re.sub(r"[^a-z0-9_]", "_", (lf.id or "").strip().lower())[:64] or f"f_{i}"
        if not fid.startswith("f_"):
            fid = f"f_{fid}"[:64]
        if fid in seen:
            continue
        kind = (lf.kind or "").strip().lower()
        label = (lf.human_label or "").strip()
        if not kind or not label:
            continue
        try:
            out.append(
                Feature(
                    id=fid,
                    kind=kind,
                    human_label=label,
                    relationships=[r.strip() for r in lf.relationships if r.strip()][:6],
                    symbolic_dims=[d.strip() for d in lf.symbolic_dims if d.strip()][:6],
                )
            )
        except Exception:
            continue
        seen.add(fid)
    return out


def _convert_llm_constraints(items: list[_LlmConstraint]) -> list[InferredConstraint]:
    """Validate + convert LLM-proposed constraints (source='vision'). Unknown kinds skipped; deduped."""
    out: list[InferredConstraint] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for lc in items or []:
        if len(out) >= 24:
            break
        kind = (lc.kind or "").strip().lower()
        if kind not in _ALLOWED_CONSTRAINT_KINDS:
            continue
        refs = sorted(r.strip() for r in lc.refs if r.strip())[:4]
        key = (kind, tuple(refs))
        if key in seen:
            continue
        try:
            out.append(
                InferredConstraint(
                    kind=ConstraintKind(kind), refs=refs, source="vision", confidence=0.8
                )
            )
        except Exception:
            continue
        seen.add(key)
    return out


def _sketch_constraints(session: DesignSession) -> list[InferredConstraint]:
    """Deterministic constraints from the latest sketch's strokes (empty if none / on any failure)."""
    if not session.sketch_assets:
        return []
    try:
        return infer_constraints(session.sketch_assets[-1].sketch)
    except Exception:
        return []


def _merge_constraints(
    a: list[InferredConstraint], b: list[InferredConstraint]
) -> list[InferredConstraint]:
    """Union two constraint lists, deduped by (kind, sorted refs); order-stable (``a`` before ``b``)."""
    out: list[InferredConstraint] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for c in [*a, *b]:
        key = (c.kind.value, tuple(sorted(c.refs)))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


_VIEWPOINT_HINTS = {
    Viewpoint.FRONT: "front view -- it pins WIDTH (X) and HEIGHT (Z); DEPTH (Y) is not visible and "
    "must come from the narration or a label.",
    Viewpoint.TOP: "top view -- it pins WIDTH (X) and DEPTH (Y); HEIGHT (Z) must come from the "
    "narration or a label.",
    Viewpoint.RIGHT: "right view -- it pins DEPTH (Y) and HEIGHT (Z); WIDTH (X) must come from the "
    "narration or a label.",
    Viewpoint.ISO: "isometric view -- foreshortened; use it for TOPOLOGY only and take all dimensions "
    "from the narration/labels.",
    Viewpoint.FREE: "unspecified view -- take dimensions from the narration/labels, not pixel sizes.",
}


def _compose_planner_user(session: DesignSession) -> str:
    """Build the planner's user message from the whole capture bundle, role-separated: the typed
    request + SPEECH TRANSCRIPT carry intent/function/dimensions; the sketch image is FORM/TOPOLOGY
    only; scale applies to in-plane lengths; the viewpoint says which axis a single view can't pin."""
    cap = session.capture
    parts: list[str] = []

    text = (session.original_text_prompt or "").strip()
    if text:
        parts.append(f"TYPED REQUEST:\n{text}")
    if cap and cap.speech_transcript and cap.speech_transcript.strip():
        parts.append(
            "SPEECH TRANSCRIPT (PRIMARY INTENT -- narrated while drawing; carries function, rationale, "
            f"and the dimensions a sketch can't show):\n{cap.speech_transcript.strip()}"
        )
    if cap and cap.labels:
        labels = "; ".join(label.strip() for label in cap.labels if label.strip())
        if labels:
            parts.append(f"ON-SKETCH LABELS: {labels}")
    if cap and cap.notes and cap.notes.strip():
        parts.append(f"NOTE:\n{cap.notes.strip()}")
    if cap and cap.scale is not None:
        ref = f" ('{cap.scale.ref_label}')" if cap.scale.ref_label else ""
        parts.append(
            f"SCALE CALIBRATION: the reference{ref} = {cap.scale.mm:g} mm. Apply it to IN-PLANE "
            "lengths only; never read the out-of-plane axis off the drawing."
        )
    if cap and cap.viewpoint is not None:
        parts.append("VIEWPOINT: " + _VIEWPOINT_HINTS.get(cap.viewpoint, "unspecified view."))
    if session.has_visual_input():
        parts.append(
            "A sketch/drawing is attached as an image -- read it for FORM and TOPOLOGY (which features "
            "exist and how they connect), NOT for exact dimensions."
        )

    return "\n\n".join(parts) if parts else "(no text; see attached drawing)"


def _planner_image_bytes(session: DesignSession) -> bytes | None:
    """The image to show the vision model: an uploaded image if present, else the rendered sketch
    preview -- so a DRAWN sketch reaches the model too (previously only uploads did)."""
    img = _first_image_bytes(session)
    if img is not None:
        return img
    for sk in reversed(session.sketch_assets):
        if sk.preview_path:
            p = Path(sk.preview_path)
            if p.is_file():
                try:
                    return p.read_bytes()
                except OSError:
                    continue
    return None


def _read_prompt(name: str) -> str:
    path = _PROMPTS_DIR / name
    return path.read_text(encoding="utf-8") if path.is_file() else ""


class LlmPlannerBrain:
    """Optional LLM brain. Uses the model ONLY to understand the request (classify + pick which
    catalog standards apply); question selection + brief assembly stay deterministic. Requires a
    configured LLM provider; constructed lazily so importing this module never needs one."""

    def __init__(self, *, model: str = SONNET, client: LLMClientLike | None = None) -> None:
        self._model = model
        self._client = client

    def analyze(self, session: DesignSession) -> PlannerOutcome:
        # Follow-up rounds: the questions were already authored on the first pass. Re-running the model
        # now would re-propose them with fresh ids (the model has no memory of what it asked), which
        # defeats the id-based "answered" filter and loops the UI ("asked the same thing twice"). So
        # reuse the already-authored set and just gate on the collected answers -- no LLM call, which
        # also makes it faster + cheaper, and guarantees the question loop terminates.
        if session.answers and session.questions and session.planner_state is not None:
            state = session.planner_state.model_copy(
                update={"rounds": session.planner_state.rounds + 1}
            )
            return _plan_from_state(session, state, llm_questions=list(session.questions))

        from forge_mind.llm.provider import make_client

        client = self._client if self._client is not None else make_client()

        system = _read_prompt("planner_system.md") or "You classify CAD part requests."
        user = _compose_planner_user(session)

        image = _planner_image_bytes(session)
        if image is not None:
            understanding, usage = client.generate_structured_with_image(
                system=system,
                user=user,
                image_png=image,
                output_model=_Understanding,
                model=self._model,
                max_tokens=2048,
            )
        else:
            understanding, usage = client.generate_structured(
                system=system,
                user=user,
                output_model=_Understanding,
                model=self._model,
                max_tokens=2048,
            )

        catalog = known_standards()
        # Only standards that exist in the catalog -- the LLM cannot invent one.
        chosen = [catalog[n] for n in understanding.standard_names if n in catalog]
        # Also merge any the deterministic detector found in the text (belt and suspenders).
        for s in detect_standards(session.original_text_prompt):
            if s.name not in {c.name for c in chosen}:
                chosen.append(s)

        prev_rounds = session.planner_state.rounds if session.planner_state else 0
        state = PlannerState(
            classified=True,
            part_type=understanding.part_type
            or classify_part_type(session.original_text_prompt, chosen),
            user_goal=understanding.user_goal or session.original_text_prompt,
            known_standards=chosen,
            detected_interfaces=sorted(c.name for c in chosen),
            rounds=prev_rounds + 1,
            features=_convert_llm_features(understanding.features),
            inferred_constraints=_convert_llm_constraints(understanding.constraints),
        )
        if understanding.refusal:
            return PlannerOutcome(
                state=state,
                questions=[],
                ready=False,
                refusal=understanding.refusal,
                usages=[usage],
            )
        outcome = _plan_from_state(
            session, state, llm_questions=_convert_llm_questions(understanding.questions)
        )
        outcome.usages = [usage]
        return outcome


def _first_image_bytes(session: DesignSession) -> bytes | None:
    """Load the first uploaded image's bytes (for the multimodal planner). Best-effort: a missing or
    unreadable file just means a text-only planner call."""
    for asset in session.uploaded_assets:
        if asset.path:
            p = Path(asset.path)
            if p.is_file():
                try:
                    return p.read_bytes()
                except OSError:
                    continue
    return None


class PlannerAgent:
    """Thin wrapper holding a brain. Defaults to the deterministic heuristic brain (no LLM)."""

    def __init__(self, brain: PlannerBrain | None = None) -> None:
        self._brain: PlannerBrain = brain or HeuristicPlannerBrain()

    def run(self, session: DesignSession) -> PlannerOutcome:
        return self._brain.analyze(session)


__all__ = [
    "HeuristicPlannerBrain",
    "LlmPlannerBrain",
    "PlannerAgent",
    "PlannerBrain",
    "PlannerOutcome",
    "classify_part_type",
]
