"""Assemble the final ``DesignBrief`` from a session's state + answers, gate generation on its
readiness, and turn a ready brief into the precise natural-language prompt the EXISTING pipeline
consumes.

The brief is built deterministically from what is KNOWN (the planner's classified state, the cited
standards, the user's answers) plus SAFE, clearly-marked defaults for the optional choices. Critical
dimensions record their ``source`` so a default is never read as a user-confirmed value.
"""

from __future__ import annotations

import re

from forge_mind.intake.constraints import summarize_constraints
from forge_mind.intake.questions import ALL_BUILDERS
from forge_mind.intake.schema import (
    AnswerValue,
    CriticalDimension,
    DesignBrief,
    DimensionSource,
    PlannerState,
    RequiredInterface,
    ScalarValue,
)
from forge_mind.intake.session import DesignSession

_DEFAULT_THICKNESS_MM = 5.0
_EXPLICIT_SCREW = re.compile(r"\bm([2-8])\b", re.IGNORECASE)

# --- server-side measurement parsing (unit-aware) -------------------------------------------------
# Any number with an EXPLICIT unit, normalized to mm. Conservative: a bare integer is never a size.
_UNIT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(millimet(?:er|re)s?|centimet(?:er|re)s?|met(?:er|re)s?|inch(?:es)?|mm|cm|m)\b",
    re.IGNORECASE,
)
# A feature word ADJACENT to a measurement binds the value to that feature's named dimension, so a
# label like "30 mm center bore" sets the bore diameter instead of sitting as freeform text.
_FEATURE_WORD: dict[str, str] = {
    "bore": "bore_d",
    "hole": "hole_d",
    "diameter": "diameter",
    "dia": "diameter",
    "radius": "radius",
    "thickness": "plate_thickness",
    "thick": "plate_thickness",
    "width": "width",
    "wide": "width",
    "height": "height",
    "tall": "height",
    "length": "length",
    "long": "length",
    "depth": "depth",
    "deep": "depth",
}


def _unit_to_mm(value: float, unit: str) -> float | None:
    u = unit.strip().lower()
    if u == "in" or u.startswith("inch"):
        return value * 25.4
    if u.startswith("mm") or u.startswith("millim"):
        return value
    if u.startswith("cm") or u.startswith("centim"):
        return value * 10.0
    if u == "m" or u.startswith("met"):
        return value * 1000.0
    return None


def extract_dimensions(text: str | None) -> list[tuple[str | None, float]]:
    """Each explicit measurement in ``text`` (unit-aware: mm/cm/m/inch) paired with the feature it
    labels, if any. A value next to a feature word binds to it ('30 mm center bore' -> ('bore_d', 30));
    an unbound value ('1 meter', '40 mm') returns (None, value). Values are normalized to mm."""
    if not text:
        return []
    low = text.lower()
    out: list[tuple[str | None, float]] = []
    for m in _UNIT_RE.finditer(low):
        mm = _unit_to_mm(float(m.group(1)), m.group(2))
        if mm is None or mm <= 0:
            continue
        window = low[max(0, m.start() - 24) : min(len(low), m.end() + 24)]
        name = next((canon for word, canon in _FEATURE_WORD.items() if word in window), None)
        out.append((name, mm))
    return out


def parse_measurements(text: str | None) -> list[float]:
    """Every explicit measurement in ``text``, normalized to mm. ``[]`` when the text states none."""
    return [value for _, value in extract_dimensions(text)]


def _capture_channels(session: DesignSession) -> list[tuple[str, DimensionSource]]:
    """Every channel the user can state a fact in, with provenance: typed prompt + speech + notes are
    USER; on-sketch labels are IMAGE (they sit on the drawing). Order sets first-value-wins precedence."""
    chans: list[tuple[str, DimensionSource]] = []
    if session.original_text_prompt:
        chans.append((session.original_text_prompt, DimensionSource.USER))
    cap = session.capture
    if cap is not None:
        if cap.speech_transcript:
            chans.append((cap.speech_transcript, DimensionSource.USER))
        if cap.notes:
            chans.append((cap.notes, DimensionSource.USER))
        if cap.labels:
            chans.append((" ; ".join(cap.labels), DimensionSource.IMAGE))
    return chans


def all_request_text(session: DesignSession) -> str:
    """All stated text across channels (typed + speech + notes + labels). The deterministic
    suppression/parsing reads THIS, so a value given in ANY channel counts, not just the typed prompt."""
    return "  ".join(text for text, _ in _capture_channels(session))


def _captured_dimensions(session: DesignSession) -> dict[str, CriticalDimension]:
    """Measurements the user STATED in any capture channel, folded into named dimensions with honest
    provenance. A value by a feature word binds to that feature; the largest UNBOUND value becomes
    ``overall_size`` so a plain '1 meter cube' still scales. First value per name wins."""
    out: dict[str, CriticalDimension] = {}
    unbound: list[tuple[float, DimensionSource]] = []
    for text, src in _capture_channels(session):
        for name, value in extract_dimensions(text):
            if name is None:
                unbound.append((value, src))
            elif name not in out:
                out[name] = CriticalDimension(name=name, value=value, unit="mm", source=src)
    overall_blockers = {"width", "height", "length", "diameter", "overall_size"}
    if unbound and not (out.keys() & overall_blockers):
        value, src = max(unbound, key=lambda vs: vs[0])
        out["overall_size"] = CriticalDimension(
            name="overall_size", value=value, unit="mm", source=src
        )
    return out


def answer_map(session: DesignSession) -> dict[str, AnswerValue]:
    """The latest answer per question id."""
    out: dict[str, AnswerValue] = {}
    for a in session.answers:
        out[a.question_id] = a.value
    return out


def _used_default(session: DesignSession, question_id: str) -> bool:
    a = session.answer_for(question_id)
    return a is not None and a.used_default


def _standard_names(state: PlannerState) -> set[str]:
    return {s.name for s in state.known_standards}


def _recommended_fastener(state: PlannerState) -> str:
    """The fastener a present standard recommends (rail fastener wins -- it carries the load path)."""
    names = _standard_names(state)
    if any("extrusion" in n for n in names):
        return "M5"
    if "NEMA-23" in names:
        return "M5"
    if "NEMA-17" in names:
        return "M3"
    return "M3"


def resolved_fastener(session: DesignSession, state: PlannerState) -> tuple[str, DimensionSource]:
    """Resolve the fastener size and its provenance (a user pick, an explicit mention, or a standard)."""
    ans = answer_map(session).get("q_fastener_type")
    if isinstance(ans, str) and ans not in ("", "recommended"):
        value = "1/4-20" if ans == "quarter_20" else ans
        return value, DimensionSource.USER
    text = all_request_text(session)
    m = _EXPLICIT_SCREW.search(text)
    if m:
        return f"M{m.group(1)}", DimensionSource.USER
    return _recommended_fastener(state), DimensionSource.STANDARD


def resolved_thickness(session: DesignSession) -> tuple[float, DimensionSource]:
    ans = answer_map(session).get("q_plate_thickness")
    if isinstance(ans, int | float) and not _used_default(session, "q_plate_thickness"):
        return float(ans), DimensionSource.USER
    return _DEFAULT_THICKNESS_MM, DimensionSource.DEFAULT


def _bool_choice(session: DesignSession, question_id: str, default: bool) -> bool:
    ans = answer_map(session).get(question_id)
    return bool(ans) if isinstance(ans, bool) else default


def _mounting_surface(session: DesignSession, state: PlannerState) -> str | None:
    ans = answer_map(session).get("q_mounting_surface")
    if isinstance(ans, str) and ans:
        return ans
    names = _standard_names(state)
    if "2020 aluminum extrusion" in names:
        return "2020_extrusion"
    if "2040 aluminum extrusion" in names:
        return "2040_extrusion"
    return None


_LABEL_STOPWORDS = frozenset(
    {"a", "an", "the", "my", "for", "of", "to", "with", "that", "this", "i", "need", "want"}
)


def _goal_label(goal: str) -> str:
    """A label slug from the USER'S OWN words (first content words of the goal). Used when the
    classified part_type is an ungrounded guess, so 'a hammer' is named 'hammer', never 'block'."""
    words = [w for w in re.findall(r"[a-z0-9]+", goal.lower()) if w not in _LABEL_STOPWORDS]
    return "_".join(words[:3]) or "part"


def _type_grounded(state: PlannerState, goal: str) -> bool:
    """Is the classified part_type EVIDENCED -- by a detected standard (a NEMA mention really is a
    motor mount) or by the type's own words appearing in the user's text? An unevidenced type is a
    classifier GUESS (the closed vocabulary mapping 'a hammer' to 'block'): it may still steer
    question selection, but it must never rename the part or prefix the generation prompt
    (ADR-0036 -- the label-consistency fix)."""
    pt = state.part_type
    if not pt or pt == "generic":
        return False
    names = _standard_names(state)
    if pt == "motor_mount" and names & {"NEMA-17", "NEMA-23"}:
        return True
    if pt == "bearing_block" and any(n.endswith("bearing") for n in names):
        return True
    if pt == "fan_mount" and any(n.endswith("fan") for n in names):
        return True
    t = goal.lower()
    return any(token in t for token in pt.split("_"))


def _slug(state: PlannerState, goal: str) -> str:
    if not _type_grounded(state, goal):
        return _goal_label(goal)
    parts: list[str] = []
    names = _standard_names(state)
    if "NEMA-17" in names:
        parts.append("nema17")
    elif "NEMA-23" in names:
        parts.append("nema23")
    if any("2020" in n for n in names):
        parts.append("2020")
    elif any("2040" in n for n in names):
        parts.append("2040")
    parts.append(state.part_type or "part")
    return "_".join(parts)


def _interfaces(session: DesignSession, state: PlannerState) -> list[RequiredInterface]:
    """Derive the mating interfaces from the part type + cited standards + the mounting-surface answer."""
    out: list[RequiredInterface] = []
    names = _standard_names(state)
    if "NEMA-17" in names or "NEMA-23" in names:
        motor = "NEMA-17" if "NEMA-17" in names else "NEMA-23"
        out.append(
            RequiredInterface(
                type=f"{motor.replace('-', '')}_face".lower(),
                fasteners="M3" if motor == "NEMA-17" else "M5",
                clearance="standard",
            )
        )
    surface = _mounting_surface(session, state)
    if surface in ("2020_extrusion", "2040_extrusion"):
        adjustable = _bool_choice(session, "q_adjustable_slots", True)
        out.append(
            RequiredInterface(
                type=f"{surface.replace('_extrusion', '')}_extrusion_mount",
                fasteners="M5",
                adjustability="slotted" if adjustable else "fixed",
            )
        )
    elif surface in ("flat_plate", "wood_surface"):
        fastener, _ = resolved_fastener(session, state)
        out.append(RequiredInterface(type=f"{surface}_mount", fasteners=fastener))
    for s in state.known_standards:
        if s.name.endswith("bearing"):
            out.append(RequiredInterface(type="bearing_seat", notes=s.name))
        if s.name.endswith("fan"):
            out.append(
                RequiredInterface(type="fan_face", fasteners=str(s.values.get("screws", "M3")))
            )
    return out


def _critical_dimensions(session: DesignSession, state: PlannerState) -> list[CriticalDimension]:
    captured = _captured_dimensions(session)
    dims: list[CriticalDimension] = []

    # Thickness: a value stated in the capture wins; else the q_plate_thickness answer; else default.
    if "plate_thickness" in captured:
        dims.append(captured.pop("plate_thickness"))
    else:
        thickness, t_src = resolved_thickness(session)
        dims.append(
            CriticalDimension(name="plate_thickness", value=thickness, unit="mm", source=t_src)
        )

    for s in state.known_standards:
        clearance = s.values.get("shaft_clearance_mm")
        if isinstance(clearance, int | float):
            dims.append(
                CriticalDimension(
                    name="shaft_clearance",
                    value=float(clearance),
                    unit="mm",
                    source=DimensionSource.STANDARD,
                )
            )
            break

    # The image-scale answer pins part width -- unless the capture already bound a width/overall size.
    scale = answer_map(session).get("q_image_scale")
    if (
        isinstance(scale, int | float)
        and "width" not in captured
        and "overall_size" not in captured
    ):
        # Provenance must reflect whether the user PICKED the scale or accepted the default: an
        # accepted default is DEFAULT, not USER (mirror resolved_thickness; audit M7).
        source = (
            DimensionSource.DEFAULT
            if _used_default(session, "q_image_scale")
            else DimensionSource.USER
        )
        dims.append(
            CriticalDimension(name="part_width", value=float(scale), unit="mm", source=source)
        )

    # Fold the remaining measurements the user stated in the capture (bore, overall size, ...).
    dims.extend(captured.values())
    return dims


def _extra_answer_notes(session: DesignSession) -> list[str]:
    """Fold answers to the LLM's idea-specific questions (ids that are NOT in the deterministic
    template bank) into readable notes, so every smart-question answer actually reaches the generation
    prompt. Template-question answers are already mapped into the brief's structured fields above, so
    they are skipped here -- the deterministic path stays unchanged."""
    known = set(ALL_BUILDERS)
    notes: list[str] = []
    for a in session.answers:
        if a.question_id in known:
            continue
        value: object = a.value
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if isinstance(value, list):
            if not value:
                continue
            value = ", ".join(str(v) for v in value)
        label = a.question_id.removeprefix("q_").replace("_", " ").strip() or "detail"
        notes.append(f"{label}: {value}")
    return notes


def _fabrication(session: DesignSession) -> str:
    """The session's fabrication intent: "print_3d" | "general". None (the toggle was never sent
    -- a legacy client) keeps the historical FDM default, so nothing changes until the frontend
    sends the field (ADR-0031 / ADR-0036)."""
    return session.fabrication or "print_3d"


def _constraints(session: DesignSession, state: PlannerState) -> list[str]:
    # The printability constraints are asserted ONLY when the part is actually being 3D-printed.
    # They were hardcoded for every session (the board-reported bug: the toggle OFF still produced
    # a brief claiming "Must be 3D printable").
    out = (
        ["Must be 3D printable", "Avoid unsupported overhangs where practical"]
        if _fabrication(session) == "print_3d"
        else []
    )
    names = _standard_names(state)
    if "NEMA-17" in names or "NEMA-23" in names:
        out.append("Motor holes must align with the stepper face pattern")
    if any("extrusion" in n for n in names):
        out.append("Rail slots must fit M5 hardware")
    special = answer_map(session).get("q_special_constraints")
    if isinstance(special, str) and special.strip():
        out.append(special.strip())
    out.extend(_extra_answer_notes(session))
    return out


def _design_choices(session: DesignSession, state: PlannerState) -> dict[str, ScalarValue]:
    return {
        "reinforced": _bool_choice(session, "q_reinforced", True),
        "rounded_edges": True,
        "adjustable_slots": _bool_choice(session, "q_adjustable_slots", True),
    }


def assemble_brief(session: DesignSession, state: PlannerState, *, ready: bool) -> DesignBrief:
    """Build the full DesignBrief from the classified state + answers + safe defaults."""
    goal = (session.original_text_prompt or state.user_goal or "a custom 3D-printed part").strip()
    material = answer_map(session).get("q_material_process")
    return DesignBrief(
        session_id=session.id,
        part_name=_slug(state, goal),
        user_goal=goal,
        part_type=state.part_type or "generic",
        # Echo the user's fabrication intent (was hardcoded to FDM regardless of the toggle).
        manufacturing_process=(
            "fdm_3d_printing" if _fabrication(session) == "print_3d" else "general_fabrication"
        ),
        material_assumption=str(material) if isinstance(material, str) else "PLA_or_PETG",
        known_standards=list(state.known_standards),
        required_interfaces=_interfaces(session, state),
        critical_dimensions=_critical_dimensions(session, state),
        design_choices=_design_choices(session, state),
        constraints=_constraints(session, state),
        features=list(state.features),
        geometry_constraints=summarize_constraints(state.inferred_constraints),
        intent_narration=(session.capture.speech_transcript if session.capture else None),
        unknowns=[] if ready else list(state.missing_critical),
        ready_for_blueprint=ready,
    )


def readiness(session: DesignSession) -> tuple[bool, list[str]]:
    """Whether generation may start. Blocks unless the brief is ready, no required question is
    unanswered, and no critical dimension is of UNKNOWN source. Returns (ok, reasons-if-not)."""
    reasons: list[str] = []
    brief = session.final_design_brief
    if brief is None:
        reasons.append("no design brief has been finalized yet")
    elif not brief.ready_for_blueprint:
        reasons.append("the design brief is not marked ready_for_blueprint")
    if brief is not None and brief.unknowns:
        reasons.append(f"unresolved unknowns: {', '.join(brief.unknowns)}")
    unanswered = session.unanswered_required()
    if unanswered:
        reasons.append("unanswered required questions: " + ", ".join(q.id for q in unanswered))
    if brief is not None:
        unknown_dims = [
            d.name for d in brief.critical_dimensions if d.source == DimensionSource.UNKNOWN
        ]
        if unknown_dims:
            reasons.append("critical dimensions with no source: " + ", ".join(unknown_dims))
    return (not reasons), reasons


_STD_PHRASE = {
    "NEMA-17": "a NEMA-17 stepper motor (42.3 mm face, 31 mm hole pattern, M3 screws)",
    "NEMA-23": "a NEMA-23 stepper motor (56.4 mm face, 47.14 mm hole pattern, M5 screws)",
    "2020 aluminum extrusion": "a 2020 aluminum T-slot extrusion rail (M5 T-nuts)",
    "2040 aluminum extrusion": "a 2040 aluminum T-slot extrusion rail (M5 T-nuts)",
}


def brief_to_prompt(brief: DesignBrief) -> str:
    """Render a ready brief as a precise, standards-resolved natural-language prompt for the existing
    ``run_pipeline``. Deterministic (no LLM): the planner already resolved the ambiguity, so this
    just states the resolved part plainly in the prompt style the IntentParser expects."""
    # Open with "A <part type>: <goal>", but (1) don't double the prefix when the goal already
    # carries it (a re-rendered brief produced "A generic: A generic: ..."), (2) don't use the
    # bare "A generic:" phrasing when no part family was resolved, and (3) don't assert an
    # UNGROUNDED type at all -- the closed-vocabulary classifier mapping "a hammer" to "block"
    # must not contaminate generation with "A block: a hammer..." (ADR-0036). A type is asserted
    # only when its own words appear in the goal.
    goal = brief.user_goal.strip()
    prefix = f"A {brief.part_type.replace('_', ' ')}: "
    type_tokens = brief.part_type.split("_")
    type_in_goal = any(tok in goal.lower() for tok in type_tokens)
    if brief.part_type == "generic" or goal.lower().startswith(prefix.lower()) or not type_in_goal:
        opening = goal
    else:
        opening = f"{prefix}{goal}"
    bits: list[str] = [opening.rstrip(".") + "."]

    if brief.intent_narration and brief.intent_narration.strip():
        bits.append(f'The maker described it aloud: "{brief.intent_narration.strip()}".')

    std_phrases = [_STD_PHRASE.get(s.name) for s in brief.known_standards]
    named = [p for p in std_phrases if p]
    if named:
        bits.append("It must fit " + "; ".join(named) + ".")

    for dim in brief.critical_dimensions:
        marker = " (default)" if dim.source == DimensionSource.DEFAULT else ""
        bits.append(f"{dim.name.replace('_', ' ').capitalize()}: {dim.value:g} {dim.unit}{marker}.")

    if brief.required_interfaces:
        iface = "; ".join(
            f"{i.type}"
            + (f" with {i.fasteners}" if i.fasteners else "")
            + (f", {i.adjustability}" if i.adjustability else "")
            for i in brief.required_interfaces
        )
        bits.append(f"Interfaces: {iface}.")

    if brief.features:
        feats = "; ".join(
            f"{f.human_label} ({f.kind})"
            + (f" [{', '.join(f.relationships)}]" if f.relationships else "")
            for f in brief.features
        )
        bits.append(f"Build it from these features, in order: {feats}.")

    if brief.geometry_constraints:
        bits.append(
            "Preserve these geometric relationships: " + "; ".join(brief.geometry_constraints) + "."
        )

    if brief.constraints:
        bits.append("Constraints: " + "; ".join(brief.constraints) + ".")

    process = brief.manufacturing_process.replace("_", " ")
    material = brief.material_assumption.replace("_", " ")
    bits.append(f"Manufacture by {process} in {material}.")
    return " ".join(bits)


def brief_to_context(brief: DesignBrief) -> str:
    """Render the typed brief as a STATED-FACTS reference block for the agentic engine's brief
    researcher (the ``research_brief`` ``extra_context`` seam, ADR-0036).

    This is the typed handoff the prose round-trip was losing: ``brief_to_prompt`` flattens the
    planner's resolved understanding into a paragraph and the agentic engine RE-GUESSES its own
    brief from it. This block carries the typed fields verbatim -- every line a user-confirmed or
    standards-resolved fact -- so generation copies them instead of re-deriving them. Deterministic
    (no LLM)."""
    lines = [
        "STATED FACTS from the guided intake (typed, user-confirmed or standards-resolved).",
        "Copy these exactly; do not contradict or re-derive them:",
        f"- object: {brief.user_goal.strip()}",
        (
            f"- fabrication: {brief.manufacturing_process.replace('_', ' ')}; "
            f"material: {brief.material_assumption.replace('_', ' ')}"
        ),
    ]
    lines.extend(f"- standard hardware: {s.name}" for s in brief.known_standards)
    lines.extend(
        f"- interface: {i.type}"
        + (f", fasteners {i.fasteners}" if i.fasteners else "")
        + (f", {i.adjustability}" if i.adjustability else "")
        for i in brief.required_interfaces
    )
    lines.extend(
        f"- dimension: {d.name.replace('_', ' ')} = {d.value:g} {d.unit}"
        f" (source: {getattr(d.source, 'value', str(d.source))})"
        for d in brief.critical_dimensions
    )
    lines.extend(
        f"- feature: {f.human_label} ({f.kind})"
        + (f" [{', '.join(f.relationships)}]" if f.relationships else "")
        for f in brief.features
    )
    lines.extend(f"- geometric relationship: {c}" for c in brief.geometry_constraints)
    lines.extend(f"- constraint: {c}" for c in brief.constraints)
    return "\n".join(lines)


__all__ = [
    "all_request_text",
    "answer_map",
    "assemble_brief",
    "brief_to_context",
    "brief_to_prompt",
    "extract_dimensions",
    "parse_measurements",
    "readiness",
    "resolved_fastener",
    "resolved_thickness",
]
