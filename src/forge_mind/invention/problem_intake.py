"""The problem-intake surface (ADR-0087 S6b, founder addendum 2026-07-06) — one question pass,
a cached tree, a typed form, and refusal as a first-class outcome.

The founder's rulings, implemented here as code:

  * **Atomic build**: every fact is gathered before ``produce`` runs; generation is never
    interrupted to ask for data (the S6a confirm gate enforces the spend side; this module
    enforces the gather side).
  * **ONE consolidated question pass**: :func:`start_problem` runs decompose -> registry ->
    ALL research -> conflicts ONCE and derives the COMPLETE remaining question set as a single
    typed form (:class:`IntakeForm`) — number+unit fields, registry-backed dropdowns, variant/
    conflict choices, free text only for irreducible ``ask`` leaves.
  * **Decompose once, cache the tree**: the validated tree is stored on the session
    (:class:`ProblemState`, an opaque blob on the ADR-0036 ``DesignSession``); the answer round
    (:func:`apply_answers`) re-enters ``compile_from_tree`` with the cached tree, the new user
    facts, and the round-1 registry/researched facts (sources INTACT — no laundering). It makes
    ZERO model calls and ZERO research calls: :class:`_NoModelClient` raises if anything tries.
  * **Refusal, not a second round (L5)**: a load-bearing fact still un-gettable after the single
    pass REFUSES, naming the fact and how to supply it. The ONLY re-ask is a CONTRADICTION
    (the user's answer vs a sourced fact), shown as a conflict choice, at most once.
  * **Deterministic classification**: :func:`classify_intake` routes a plain SPEC prompt back to
    today's ``/generate`` path with the prompt BYTE-UNTOUCHED; ambiguity resolves toward SPEC
    (the proven path). No model call is spent on classification.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.invention.compile_requirements import (
    CompiledRequirements,
    OpenQuestion,
    compile_from_tree,
)
from forge_mind.invention.decompose import DecomposeResult, decompose_problem
from forge_mind.invention.env_checks import EnvCheckResult, run_env_checks
from forge_mind.invention.problem_mode import ProduceLike, _load_geometry
from forge_mind.invention.requirements import Fact
from forge_mind.invention.research import ResearchProviderLike
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

# ---------------------------------------------------------------------------------------------
# The seed + the deterministic classifier
# ---------------------------------------------------------------------------------------------


class ProblemIntake(BaseModel):
    """The minimal rich seed: a goal plus the context that shrinks the question set (a named
    product OR key dims — later, an image). NOT a big static form.

    N1 (founder 2026-07-07): ``materials`` = which filaments the user can actually print with
    (empty = all known); the material question offers ONLY that set, and when a thermal branch
    is active the system RECOMMENDS from it (advisory, cited, ceiling-labeled — L6).
    ``process`` = the fabrication base fact; v1 validates FDM printing only, so ``cnc``
    REFUSES honestly instead of pretending."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: str = Field(..., min_length=1)
    context: str = ""  # required on the PROBLEM path (the endpoint enforces, with the exact ask)
    materials: tuple[str, ...] = ()  # the user's availability set; () = all MATERIAL_CHOICES
    process: Literal["print", "cnc"] = "print"


IntakeRoute = Literal["problem", "spec"]

# First-person / environment phrasing marks a PROBLEM; a bare geometric description is a SPEC.
_PROBLEM_MARKERS = re.compile(
    r"\b(i need|i'?m building|i am building|help me|in my|for my|to my|my \w+|one more|"
    r"i want to (?:mount|fit|hold|attach|add|route))\b",
    re.IGNORECASE,
)


def classify_intake(goal: str, context: str) -> IntakeRoute:
    """Deterministic, no model call. Ambiguity resolves toward SPEC — misrouting a problem to the
    proven executor is today's behavior; misrouting a spec into problem-mode is churn (P17)."""
    if context.strip():
        return "problem"  # the user deliberately supplied problem context
    if _PROBLEM_MARKERS.search(goal):
        return "problem"
    return "spec"


# Product-model tokens ("NR200", "H510 Flow") for research entity/variant detection.
_ENTITY_RE = re.compile(r"\b([A-Za-z]{1,8}[- ]?\d{2,4}[A-Za-z0-9\-]*)\b")


def detect_entity(text: str) -> str:
    m = _ENTITY_RE.search(text)
    return m.group(1) if m else ""


# Git-reviewed alias table: seed text -> registry keys, on an UNAMBIGUOUS match only (the addendum:
# a rich seed closes facts before any question is asked). Keys MUST exist in their registry — a
# golden test pins that (the P9 discipline).
_REGISTRY_ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "drive_form": {
        "2.5in_drive": ("2.5in", "2.5 in", '2.5"', "2.5-inch", "2.5 inch"),
        "3.5in_drive": ("3.5in", "3.5 in", '3.5"', "3.5-inch", "3.5 inch"),
    },
    "case_fan": {
        "fan_80": ("80mm fan", "80 mm fan"),
        "fan_92": ("92mm fan", "92 mm fan"),
        "fan_120": ("120mm fan", "120 mm fan"),
        "fan_140": ("140mm fan", "140 mm fan"),
    },
}


def seed_registry_keys(text: str) -> dict[str, str]:
    """Registry keys deterministically seeded from the user's own words — set only when exactly
    ONE key of a registry matches (ambiguity stays a dropdown question, never a guess)."""
    low = text.lower()
    out: dict[str, str] = {}
    for reg_name, keymap in _REGISTRY_ALIASES.items():
        hits = [key for key, aliases in keymap.items() if any(a in low for a in aliases)]
        if len(hits) == 1:
            out[reg_name] = hits[0]
    return out


# What each registry is ABOUT, for dropdown relevance: a registry may replace a number question
# with its dropdown only when the problem text actually mentions its subject — a router bracket
# must never be asked "which drive form?" just because a playbook template carries the hint.
_REGISTRY_SUBJECTS: dict[str, tuple[str, ...]] = {
    "drive_form": ("ssd", "hdd", "hard drive", "drive"),
    "case_fan": ("fan",),
}

# BASE intake facts (founder, 2026-07-07: "there are some stuff it always needs, like it always
# needs a material") — gathered on EVERY problem intake regardless of the tree. v1: material, a
# closed reviewed FDM choice; the list grows by git review, never at runtime. The user's pick is
# user-stated; a spec cannot confirm without one (no silent default — the ADR-0036 lesson).
MATERIAL_CHOICES: tuple[str, ...] = ("PLA", "PETG", "ABS", "ASA", "TPU", "Nylon")

_MATERIAL_RE = re.compile(r"\b(pla|petg|abs|asa|tpu|nylon)\b", re.IGNORECASE)


def detect_material(text: str) -> str | None:
    """The material seeded from the user's own words (rich seed shrinks the form), or None."""
    m = _MATERIAL_RE.search(text)
    if m is None:
        return None
    token = m.group(1).lower()
    return next(c for c in MATERIAL_CHOICES if c.lower() == token)


def relevant_registries(text: str) -> frozenset[str]:
    """The registries the problem text plausibly talks about (alias hit OR subject word) — the
    deterministic gate on which dropdowns may appear."""
    low = text.lower()
    out: set[str] = set()
    for reg_name, keymap in _REGISTRY_ALIASES.items():
        if any(a in low for aliases in keymap.values() for a in aliases):
            out.add(reg_name)
    for reg_name, subjects in _REGISTRY_SUBJECTS.items():
        if any(s in low for s in subjects):
            out.add(reg_name)
    return frozenset(out)


# ---------------------------------------------------------------------------------------------
# The typed form (derived from the compiler's typed OpenQuestions)
# ---------------------------------------------------------------------------------------------


class FormField(BaseModel):
    """One field of the consolidated intake form. ``kind`` drives the UI control: ``number`` ->
    value+unit input, ``choice`` -> dropdown over ``choices`` (registry rows / variants / conflict
    values), ``text`` -> free text for an irreducible ask leaf."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: Literal["number", "choice", "text"]
    question: str
    why: str = ""
    unit: str = ""
    choices: tuple[str, ...] = ()
    fact_name: str = ""
    registry: str = ""
    conflict: bool = False
    # N1: an ADVISORY recommendation among ``choices`` + the cited, ceiling-labeled reason —
    # the user still picks (their pick is the user-stated fact; the recommendation never is).
    recommended: str = ""
    reason: str = ""


class IntakeForm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fields: tuple[FormField, ...] = ()


def _thermal_active(compiled: CompiledRequirements) -> bool:
    """True when the tree actively gathers/checks thermal_airflow (the material-relevant signal
    we can detect WITHOUT a thermal instrument — heat context, not heat analysis)."""
    if compiled.tree is None:
        return False
    return any(
        leaf.domain == "thermal_airflow"
        and leaf.closure is not None
        and leaf.closure.kind in ("fact", "check")
        for leaf in compiled.tree.leaves()
    )


def _material_field(available: tuple[str, ...], compiled: CompiledRequirements) -> FormField:
    """The material choice over the USER'S availability set. When a thermal branch is active,
    RECOMMEND the set's highest cited HDT — advisory, cited, and ceiling-labeled (L6: we cite a
    datasheet number; we never claim thermal sufficiency)."""
    from forge_mind.dsl.standards import FILAMENTS

    choices = available or MATERIAL_CHOICES
    recommended = ""
    reason = ""
    if _thermal_active(compiled):
        rated = [
            (m, FILAMENTS.get(m).hdt_045_c)
            for m in choices
            if FILAMENTS.has(m) and FILAMENTS.get(m).hdt_045_c is not None
        ]
        if rated:
            top = max(rated, key=lambda mv: mv[1] or 0.0)
            recommended = top[0]
            reason = (
                f"this build touches a heat/airflow region: {top[0]} has the highest cited "
                f"heat-deflection rating of your set ({top[1]:g} °C @0.45 MPa, datasheet value, "
                "±10 °C spread) — advisory only; temperatures are NOT simulated"
            )
    return FormField(
        id="material",
        kind="choice",
        question="Which print material should this part be made of?",
        why="every build needs a material (base intake fact)",
        choices=choices,
        recommended=recommended,
        reason=reason,
    )


def form_from_questions(questions: Sequence[OpenQuestion]) -> IntakeForm:
    """The 1:1 deterministic projection question -> field, deduped by id (first wins)."""
    fields: list[FormField] = []
    seen: set[str] = set()
    for q in questions:
        if q.kind == "registry":
            field = FormField(
                id=f"registry:{q.registry}",
                kind="choice",
                question=q.question,
                why=q.why,
                choices=q.choices,
                registry=q.registry,
            )
        elif q.kind == "variant":
            field = FormField(
                id="variant", kind="choice", question=q.question, why=q.why, choices=q.choices
            )
        elif q.kind == "conflict":
            field = FormField(
                id=f"conflict:{q.fact_name}",
                kind="choice",
                question=q.question,
                why=q.why,
                choices=q.choices,
                fact_name=q.fact_name,
                unit=q.unit,
                conflict=True,
            )
        elif q.kind == "ask":
            field = FormField(id=f"ask:{q.node_id}", kind="text", question=q.question, why=q.why)
        elif q.choices:  # a fact offered as a CHOICE over cited standard values (N1 identity-first)
            field = FormField(
                id=q.fact_name,
                kind="choice",
                question=q.question,
                why=q.why,
                unit=q.unit or "mm",
                fact_name=q.fact_name,
                choices=q.choices,
            )
        else:  # "fact"
            field = FormField(
                id=q.fact_name,
                kind="number",
                question=q.question,
                why=q.why,
                unit=q.unit or "mm",
                fact_name=q.fact_name,
            )
        if field.id not in seen:
            seen.add(field.id)
            fields.append(field)
    return IntakeForm(fields=tuple(fields))


class IntakeAnswerItem(BaseModel):
    """One answer to one form field, by field id. Exactly the member matching the field's kind
    must be set (``value`` / ``choice`` / ``text``) — validated fail-closed on apply.

    ``unknown=True`` is the explicit "I cannot supply this": it satisfies the form's completeness
    check while withholding the value, so the leaf stays open and the answer round lands on the
    designed L5 outcome — a REFUSAL naming the fact — instead of a dead-ended form (the addendum's
    "still un-gettable after the single pass", stated by the user themselves)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    value: float | None = None
    unit: str = ""
    choice: str | None = None
    text: str | None = None
    unknown: bool = False


class IntakeAnswerError(ValueError):
    """A malformed/incomplete answer set — the endpoint surfaces it as a 422, nothing mutates."""


# ---------------------------------------------------------------------------------------------
# The session state (an opaque blob on the ADR-0036 DesignSession; this module owns its schema)
# ---------------------------------------------------------------------------------------------

ProblemStage = Literal["awaiting_answers", "spec_ready", "built", "refused"]


class BuildRecord(BaseModel):
    """The confirm-gated build's outcome: what the engine shipped + the advisory env findings +
    the assumption ledger that rides the report (L3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    part_name: str
    disposition: str
    artifacts: dict[str, str] = Field(default_factory=dict)
    env_results: tuple[EnvCheckResult, ...] = ()
    assumption_ledger: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class ProblemState(BaseModel):
    """Everything one problem intake knows, serialized onto the guided-intake session. The TREE is
    cached here — the answer round never re-decomposes (the addendum's churn rule)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: ProblemStage
    goal: str
    context: str
    entity: str = ""
    # The base intake fact (founder 2026-07-07): chosen by the user (or seeded from their own
    # words); a spec cannot confirm while this is None — no silent material default, ever.
    material: str | None = None
    registry_keys: dict[str, str] = Field(default_factory=dict)
    tree: DecomposeResult
    user_facts: tuple[Fact, ...] = ()  # latest answer wins per name (self-corrections collapse)
    known_facts: tuple[Fact, ...] = ()  # round-1 registry/researched facts, sources INTACT
    answered_asks: dict[str, str] = Field(default_factory=dict)  # node_id -> the user's text
    overridden: tuple[str, ...] = ()  # names where a confirmed answer beats a sourced fact
    conflict_round_used: bool = False
    form: IntakeForm | None = None
    compiled: CompiledRequirements | None = None
    refusal: str | None = None
    build: BuildRecord | None = None
    notes: tuple[str, ...] = ()


_T = TypeVar("_T", bound=BaseModel)


class _NoModelClient:
    """The answer round's client: it must never be needed. Any call is a bug — the round is
    deterministic by design (cached tree, no research), so this fails loudly, not silently."""

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[_T],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[_T, LLMUsage]:
        raise AssertionError(
            "the answer round must not call the model (founder addendum 2026-07-06)"
        )

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[_T],
        model: str,
        max_tokens: int,
    ) -> tuple[_T, LLMUsage]:
        raise AssertionError(
            "the answer round must not call the model (founder addendum 2026-07-06)"
        )


def _problem_text(goal: str, context: str) -> str:
    return f"{goal.strip()}\n\nContext: {context.strip()}" if context.strip() else goal.strip()


def _hollow_refusal(compiled: CompiledRequirements) -> str | None:
    """A 'ready' spec with ZERO measurable anchors is HOLLOW — a degenerate tree (every branch
    unvalidated / engine's-choice) produced nothing a build could rest on. Confirming it would
    be the P6 silent-assumption failure; the honest outcome is a refusal (run-3 bench evidence:
    a must-refuse prompt sailed to spec_ready with an empty ledger and zero questions)."""
    if compiled.status == "ready" and not compiled.ledger.anchors():
        return (
            "REFUSED: nothing measurable could be gathered — every requirement branch closed as "
            "unvalidated or the engine's choice, so a build would rest on zero facts. Restate "
            "the problem with the object being built and its measurable context."
        )
    return None


def start_problem(
    intake: ProblemIntake,
    *,
    client: LLMClientLike,
    model: str,
    provider: ResearchProviderLike | None,
) -> tuple[ProblemState, list[LLMUsage]]:
    """The single gather pass: decompose ONCE -> registry -> ALL research -> conflicts -> the
    complete typed form (or straight to spec_ready). The tree + round-1 sourced facts are cached
    on the returned state so the answer round is deterministic."""
    bad = [m for m in intake.materials if m not in MATERIAL_CHOICES]
    if bad:
        raise IntakeAnswerError(f"unknown material(s) {bad}; known: {list(MATERIAL_CHOICES)}")
    problem = _problem_text(intake.goal, intake.context)
    seed_text = f"{intake.goal} {intake.context}"
    entity = detect_entity(intake.context or intake.goal)
    registry_keys = seed_registry_keys(seed_text)
    usages: list[LLMUsage] = []
    if intake.process == "cnc":
        # The fabrication base fact, honestly: only FDM printing is a VALIDATED lane today.
        # Pretending to design-for-CNC would be an unvalidated claim (L6) — refuse with the gap.
        return (
            ProblemState(
                stage="refused",
                goal=intake.goal,
                context=intake.context,
                tree=DecomposeResult(root=None, notes=["cnc lane refused before decompose"]),
                refusal=(
                    "REFUSED: CNC is not a validated fabrication lane yet — only FDM 3D "
                    "printing is. A CNC part would carry manufacturability claims nothing "
                    "here can check. Choose process='print', or wait for the CNC lane."
                ),
            ),
            usages,
        )
    decomposed, usage = decompose_problem(problem, client=client, model=model)
    usages.append(usage)
    compiled = compile_from_tree(
        problem,
        decomposed,
        client=client,
        model=model,
        entity=entity,
        registry_keys=registry_keys,
        provider=provider,
        usages=usages,
        park_registries=relevant_registries(seed_text),
    )
    known = tuple(f for f in compiled.ledger.facts if f.source in ("registry", "researched"))
    material = detect_material(seed_text)
    if material is not None and intake.materials and material not in intake.materials:
        material = None  # the user named a material they cannot print with -> ask from their set
    hollow = _hollow_refusal(compiled)
    fields = form_from_questions(compiled.open_questions).fields
    if material is None and hollow is None:
        # the base fact rides the SAME single form, offered from the USER'S availability set
        fields = (*fields, _material_field(intake.materials, compiled))
    stage: ProblemStage = (
        "refused" if hollow is not None else "awaiting_answers" if fields else "spec_ready"
    )
    return (
        ProblemState(
            stage=stage,
            goal=intake.goal,
            context=intake.context,
            entity=entity,
            material=material,
            registry_keys=registry_keys,
            tree=decomposed,
            known_facts=known,
            form=IntakeForm(fields=fields) if fields and hollow is None else None,
            compiled=compiled,
            refusal=hollow,
        ),
        usages,
    )


def _apply_one(
    field: FormField,
    answer: IntakeAnswerItem,
    *,
    user_map: dict[str, Fact],
    registry_keys: dict[str, str],
    answered_asks: dict[str, str],
    overridden: set[str],
    notes: list[str],
) -> str | None:
    """Apply one validated answer to the state pieces; returns the new entity for a variant pick
    (None otherwise). Raises IntakeAnswerError on a malformed answer (fail-closed, no mutation of
    anything the caller keeps on error — the caller re-raises before persisting)."""
    if answer.unknown:
        return None  # explicitly un-gettable: the leaf stays open -> the refusal policy names it
    if field.kind == "number":
        if answer.value is None:
            raise IntakeAnswerError(f"field {field.id!r} needs a numeric value")
        fact = Fact(
            name=field.fact_name or field.id,
            value=answer.value,
            unit=answer.unit or field.unit or "mm",
            source="user",
        )
        user_map[fact.name.strip().lower()] = fact
        return None
    if field.kind == "text":
        if not (answer.text or "").strip():
            raise IntakeAnswerError(f"field {field.id!r} needs a text answer")
        answered_asks[field.id.removeprefix("ask:")] = str(answer.text).strip()
        return None
    # choice fields: registry dropdown / variant / conflict
    choice = (answer.choice or "").strip()
    if choice not in field.choices:
        raise IntakeAnswerError(
            f"field {field.id!r}: choice {choice!r} is not one of {list(field.choices)}"
        )
    if field.registry:
        registry_keys[field.registry] = choice
        return None
    if field.conflict:
        fact = Fact(
            name=field.fact_name, value=float(choice), unit=field.unit or "mm", source="user"
        )
        user_map[fact.name.strip().lower()] = fact
        overridden.add(field.fact_name.strip().lower())
        notes.append(
            f"user confirmed {field.fact_name} = {choice} {field.unit or 'mm'} over the "
            "conflicting sourced value (override recorded)"
        )
        return None
    if field.fact_name:  # a fact-choice over cited standard values (N1): the pick is user-stated
        fact = Fact(
            name=field.fact_name, value=float(choice), unit=field.unit or "mm", source="user"
        )
        user_map[fact.name.strip().lower()] = fact
        return None
    return choice  # a variant pick: the caller updates the entity


def apply_answers(state: ProblemState, answers: Sequence[IntakeAnswerItem]) -> ProblemState:
    """The answer round: fully deterministic (cached tree, no research, a client that refuses to
    be called). Every form field must be answered; then either the spec is ready, a CONTRADICTION
    re-asks once as a conflict, or the intake REFUSES naming the un-gettable fact(s) — never a
    second question round (L5, the addendum)."""
    if state.stage != "awaiting_answers":
        raise IntakeAnswerError(f"session is {state.stage!r}, not awaiting answers")
    if state.form is None:  # structurally impossible when awaiting; fail closed anyway
        raise IntakeAnswerError("no form to answer")
    by_id = {f.id: f for f in state.form.fields}
    answered = {a.id for a in answers}
    missing = sorted(set(by_id) - answered)
    if missing:
        raise IntakeAnswerError(f"unanswered field(s): {', '.join(missing)}")
    unknown = sorted(answered - set(by_id))
    if unknown:
        raise IntakeAnswerError(f"unknown field id(s): {', '.join(unknown)}")

    user_map: dict[str, Fact] = {f.name.strip().lower(): f for f in state.user_facts}
    registry_keys = dict(state.registry_keys)
    answered_asks = dict(state.answered_asks)
    overridden = {n.strip().lower() for n in state.overridden}
    notes = list(state.notes)
    entity = state.entity
    material = state.material
    material_unknown = False
    for a in answers:
        field = by_id[a.id]
        if field.id == "material":  # the base intake fact rides its own state, not the ledger
            if a.unknown:
                material_unknown = True
            else:
                choice = (a.choice or "").strip()
                if choice not in field.choices:
                    raise IntakeAnswerError(
                        f"material {choice!r} is not one of {list(field.choices)}"
                    )
                material = choice
            continue
        picked = _apply_one(
            field,
            a,
            user_map=user_map,
            registry_keys=registry_keys,
            answered_asks=answered_asks,
            overridden=overridden,
            notes=notes,
        )
        if picked is not None:
            entity = picked

    known = tuple(f for f in state.known_facts if f.name.strip().lower() not in overridden)
    compiled = compile_from_tree(
        _problem_text(state.goal, state.context),
        state.tree,
        client=_NoModelClient(),
        model="deterministic-answer-round",
        user_facts=tuple(user_map.values()),
        known_facts=known,
        answered_asks=answered_asks,
        entity=entity,
        registry_keys=registry_keys,
        provider=None,  # research had its one shot in the single pass — never re-runs
        park_registries=relevant_registries(f"{state.goal} {state.context}"),
    )

    conflicts = [q for q in compiled.open_questions if q.kind == "conflict"]
    others = [q for q in compiled.open_questions if q.kind != "conflict"]
    if material_unknown:  # the base fact is un-gettable by the user's own word -> L5 names it
        others.append(
            OpenQuestion(
                question="Which print material should this part be made of?",
                why="every build needs a material (base intake fact)",
                kind="ask",
                fact_name="material",
            )
        )
    base = state.model_copy(
        update={
            "entity": entity,
            "material": material,
            "registry_keys": registry_keys,
            "user_facts": tuple(user_map.values()),
            "known_facts": known,
            "answered_asks": answered_asks,
            "overridden": tuple(sorted(overridden)),
            "compiled": compiled,
            "notes": tuple(notes),
        }
    )
    if others:
        return base.model_copy(
            update={"stage": "refused", "form": None, "refusal": _refusal_text(others)}
        )
    if conflicts:
        if state.conflict_round_used:
            return base.model_copy(
                update={
                    "stage": "refused",
                    "form": None,
                    "refusal": _refusal_text(conflicts, contradiction=True),
                }
            )
        return base.model_copy(
            update={
                "stage": "awaiting_answers",
                "form": form_from_questions(conflicts),
                "conflict_round_used": True,
            }
        )
    hollow = _hollow_refusal(compiled)
    if hollow is not None:
        return base.model_copy(update={"stage": "refused", "form": None, "refusal": hollow})
    return base.model_copy(update={"stage": "spec_ready", "form": None})


def _refusal_text(questions: Sequence[OpenQuestion], *, contradiction: bool = False) -> str:
    gaps = "; ".join(
        f"{q.fact_name or q.question} ({q.why})" if q.why else (q.fact_name or q.question)
        for q in questions
    )
    if contradiction:
        return (
            "REFUSED: the answers remain contradictory after one conflict round — "
            f"{gaps}. Verify the value on your physical hardware and start a new intake."
        )
    return (
        "REFUSED: the following load-bearing fact(s) could not be gathered from you, the "
        f"registries, or cited research: {gaps}. Measure the hardware (or restate the goal "
        "with these values) and start a new intake."
    )


def confirm_and_build(state: ProblemState, *, produce: ProduceLike) -> ProblemState:
    """The ONLY spend path (L3): a spec_ready state, explicitly confirmed, builds the EXACT
    compiled spec the user saw, then runs the advisory env checks and assembles the ledger."""
    if state.stage != "spec_ready" or state.compiled is None:
        raise IntakeAnswerError(f"session is {state.stage!r}; only spec_ready can be confirmed")
    if state.material is None:
        raise IntakeAnswerError(
            "no material chosen — every build needs one (base intake fact); never defaulted"
        )
    compiled = state.compiled
    part = produce(compiled.prompt, compiled.extra_context, material=state.material)
    mesh, brep, load_notes = _load_geometry(part)
    env_results = tuple(run_env_checks(compiled.checks, mesh, brep))
    ledger = (
        *compiled.assumptions,
        *(f"env check {r.check_id}: {r.explanation}" for r in env_results if r.status == "flag"),
    )
    return state.model_copy(
        update={
            "stage": "built",
            "build": BuildRecord(
                part_name=part.part_name,
                disposition=part.disposition,
                artifacts=dict(part.artifacts),
                env_results=env_results,
                assumption_ledger=ledger,
                notes=tuple(load_notes),
            ),
        }
    )


__all__ = [
    "MATERIAL_CHOICES",
    "BuildRecord",
    "FormField",
    "IntakeAnswerError",
    "IntakeAnswerItem",
    "IntakeForm",
    "IntakeRoute",
    "ProblemIntake",
    "ProblemStage",
    "ProblemState",
    "apply_answers",
    "classify_intake",
    "confirm_and_build",
    "detect_entity",
    "detect_material",
    "form_from_questions",
    "relevant_registries",
    "seed_registry_keys",
    "start_problem",
]
