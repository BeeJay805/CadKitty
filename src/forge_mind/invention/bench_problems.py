"""problems_v1 — the Problem Solver intake bench (ADR-0087 S7): loader, canned user, and the
DETERMINISTIC scorer.

The bench drives the REAL S6b surface (``start_problem`` -> one ``apply_answers`` round, plus at
most one conflict round) against the frozen prompt set with canned research corpora, then scores
every outcome with plain code — no model judges anything:

  * **invented-fact rate (the gate: must be 0).** Every ledger fact must TRACE: ``user`` -> a
    canned answer the bench actually gave; ``registry`` -> a committed cited registry row's value;
    ``researched`` -> quote verbatim in the prompt's canned corpus + value+unit in the quote +
    citation = that document's URL. Anything else is INVENTED — the P1 failure, measured.
  * **refusal correctness (the gate: must be 100%).** Must-refuse prompts refuse naming the
    expected gap; solvable prompts never refuse (a refusal caused by an over-asked question the
    canned user could not answer is a SPURIOUS refusal — deliberate, P5).
  * route accuracy (P17), missing-branch rate (P16), fact recall/precision, question discipline,
    and the P14 thermal-ceiling check are reported alongside.

The scorer never calls a model; ``run_prompt`` calls one only through the injected client (live
runs use the provider factory; unit tests inject canned brains)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.standards import ALL_REGISTRIES
from forge_mind.invention.compile_requirements import CompiledRequirements
from forge_mind.invention.playbooks import PLAYBOOKS, FactTemplate
from forge_mind.invention.problem_intake import (
    IntakeAnswerItem,
    IntakeForm,
    ProblemIntake,
    ProblemState,
    apply_answers,
    classify_intake,
    start_problem,
)
from forge_mind.invention.requirements import Fact, normalize_measure

# the quote-verification primitives ARE the instrument — share them, never fork them
from forge_mind.invention.research import (
    Document,
    FakeResearchProvider,
    _norm,
    _value_in_text,
)
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike

# ---------------------------------------------------------------------------------------------
# the frozen content (loader)
# ---------------------------------------------------------------------------------------------


class ExpectedFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    closure: Literal["registry", "researched", "asked"]
    value: float | None = None
    unit: str = "mm"


class CannedAnswers(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    facts: dict[str, dict[str, float | str]] = Field(default_factory=dict)
    unknown_facts: tuple[str, ...] = ()
    registry_choices: dict[str, str] = Field(default_factory=dict)
    variant_choice: str | None = None
    conflict_choices: dict[str, str] = Field(default_factory=dict)
    default_text: str | None = None
    # Solvable prompts only: a real owner can measure their own hardware, so any UNCOVERED number
    # question gets this measurement (over-asking stays visible via question_budget). Must-refuse
    # prompts carry None -> uncovered questions answer unknown -> the refusal logic is measured.
    fallback_number: float | None = None
    # The base intake fact: every problem form carries the material choice; the canned user picks
    # this (default PLA) — material is never a refusal lever in this bench.
    material_choice: str | None = None


class BenchPrompt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    class_: str = Field(..., alias="class")
    goal: str
    context: str = ""
    expected_route: Literal["spec", "problem"]
    expected_domains: tuple[str, ...] = ()
    question_budget: int = 0
    expect_variant_question: bool = False
    expect_registry_dropdown: str | None = None
    expected_facts: dict[str, ExpectedFact] = Field(default_factory=dict)
    answers: CannedAnswers = Field(default_factory=CannedAnswers)
    must_refuse: bool = False
    expected_refusal_gap: str = ""
    corpus: str | None = None
    golden_build: bool = False


class Bench(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    frozen: str
    authored_by: str
    answer_policy: str
    prompts: tuple[BenchPrompt, ...]


def bench_dir() -> Path:
    return Path(__file__).parents[3] / "benchmarks" / "problems_v1"


def load_bench(root: Path | None = None) -> Bench:
    root = root or bench_dir()
    return Bench.model_validate_json((root / "problems_v1.json").read_text(encoding="utf-8"))


def load_corpus(prompt: BenchPrompt, root: Path | None = None) -> list[Document]:
    """Every canned document of the prompt's corpus (for scoring); [] when it has none."""
    if prompt.corpus is None:
        return []
    root = root or bench_dir()
    raw = json.loads((root / prompt.corpus).read_text(encoding="utf-8"))
    return [Document.model_validate(d) for docs in raw["by_query"].values() for d in docs]


def provider_for(prompt: BenchPrompt, root: Path | None = None) -> FakeResearchProvider:
    if prompt.corpus is None:
        return FakeResearchProvider({})
    root = root or bench_dir()
    raw = json.loads((root / prompt.corpus).read_text(encoding="utf-8"))
    return FakeResearchProvider(
        {k: [Document.model_validate(d) for d in docs] for k, docs in raw["by_query"].items()}
    )


# ---------------------------------------------------------------------------------------------
# the canned user + the drive loop
# ---------------------------------------------------------------------------------------------


def answers_for_form(form: IntakeForm, canned: CannedAnswers) -> list[IntakeAnswerItem]:
    """The frozen answer policy: cover what the set covers; everything else is honestly UNKNOWN."""
    out: list[IntakeAnswerItem] = []
    for f in form.fields:
        if f.id == "material":
            out.append(IntakeAnswerItem(id=f.id, choice=canned.material_choice or "PLA"))
        elif f.kind == "number":
            name = f.fact_name or f.id
            spec = canned.facts.get(name)
            if name in canned.unknown_facts:
                out.append(IntakeAnswerItem(id=f.id, unknown=True))
            elif spec is not None:
                out.append(
                    IntakeAnswerItem(
                        id=f.id,
                        value=float(spec["value"]),
                        unit=str(spec.get("unit", f.unit or "mm")),
                    )
                )
            elif canned.fallback_number is not None:
                out.append(
                    IntakeAnswerItem(id=f.id, value=canned.fallback_number, unit=f.unit or "mm")
                )
            else:
                out.append(IntakeAnswerItem(id=f.id, unknown=True))
        elif f.kind == "text":
            if canned.default_text:
                out.append(IntakeAnswerItem(id=f.id, text=canned.default_text))
            else:
                out.append(IntakeAnswerItem(id=f.id, unknown=True))
        elif f.conflict:
            picked = canned.conflict_choices.get(f.fact_name) or (f.choices[0] if f.choices else "")
            out.append(IntakeAnswerItem(id=f.id, choice=picked))
        elif f.fact_name and f.choices and not f.registry:
            # a fact-choice over standard values (N1): pick the canned value when it is one of
            # the offered choices, else the first cited class (never a made-up number)
            spec = canned.facts.get(f.fact_name)
            want = f"{float(spec['value']):g}" if spec is not None else None
            out.append(
                IntakeAnswerItem(id=f.id, choice=want if want in f.choices else f.choices[0])
            )
        elif f.registry:
            reg_choice = canned.registry_choices.get(f.registry)
            if reg_choice is not None and reg_choice in f.choices:
                out.append(IntakeAnswerItem(id=f.id, choice=reg_choice))
            else:
                out.append(IntakeAnswerItem(id=f.id, unknown=True))
        else:  # variant
            if canned.variant_choice is not None and canned.variant_choice in f.choices:
                out.append(IntakeAnswerItem(id=f.id, choice=canned.variant_choice))
            else:
                out.append(IntakeAnswerItem(id=f.id, unknown=True))
    return out


class PromptOutcome(BaseModel):
    """Everything one prompt produced, as the scorer needs it (and results.json records it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str
    route: Literal["spec", "problem"]
    round1_form: IntakeForm | None = None
    state: ProblemState | None = None  # the FINAL state (problem prompts only)
    usages: tuple[LLMUsage, ...] = ()
    error: str = ""


def run_prompt(
    prompt: BenchPrompt,
    *,
    client: LLMClientLike,
    model: str,
    root: Path | None = None,
) -> PromptOutcome:
    """Drive the real S6b surface: classify; if a problem, ONE gather pass + ONE answer round
    (+ at most one conflict round, per the addendum). Never raises — an error is an outcome."""
    route = classify_intake(prompt.goal, prompt.context)
    if route == "spec" or prompt.expected_route == "spec":
        return PromptOutcome(prompt_id=prompt.id, route=route)
    try:
        state, usages = start_problem(
            ProblemIntake(goal=prompt.goal, context=prompt.context),
            client=client,
            model=model,
            provider=provider_for(prompt, root),
        )
        round1 = state.form
        rounds = 0
        while state.stage == "awaiting_answers" and state.form is not None and rounds < 2:
            state = apply_answers(state, answers_for_form(state.form, prompt.answers))
            rounds += 1
        return PromptOutcome(
            prompt_id=prompt.id,
            route=route,
            round1_form=round1,
            state=state,
            usages=tuple(usages),
        )
    except Exception as exc:  # a crashed prompt is a scored failure, never a lost run
        return PromptOutcome(prompt_id=prompt.id, route=route, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------------------------
# the deterministic scorer
# ---------------------------------------------------------------------------------------------


def _template_named(fact_name: str) -> FactTemplate | None:
    want = fact_name.strip().lower()
    for pb in PLAYBOOKS.values():
        for t in pb.facts:
            if t.name.strip().lower() == want:
                return t
    return None


def _user_fact_traces(fact: Fact, canned: CannedAnswers) -> bool:
    """A user-tier fact must be something the canned user actually said (a number answer or a
    conflict choice) — anything else is provenance laundering."""
    name = fact.name.strip().lower()
    spec = {k.strip().lower(): v for k, v in canned.facts.items()}.get(name)
    fv, fu = fact.as_mm_or_g
    if spec is not None:
        ev, eu = normalize_measure(float(spec["value"]), str(spec.get("unit", "mm")))
        if eu == fu and abs(ev - fv) <= 1e-6:
            return True
    if (
        canned.fallback_number is not None
        and name not in {k.strip().lower() for k in canned.unknown_facts}
        and fu == "mm"
        and abs(canned.fallback_number - fv) <= 1e-6
    ):
        return True  # the canned owner's fallback measurement for an uncovered question
    choice = {k.strip().lower(): v for k, v in canned.conflict_choices.items()}.get(name)
    return choice is not None and abs(float(choice) - fact.value) <= 1e-6


def _registry_fact_traces(fact: Fact) -> bool:
    """A registry-tier fact must equal a committed cited row's attribute value."""
    template = _template_named(fact.name)
    if template is None or template.registry is None or template.registry_attr is None:
        return False
    reg = ALL_REGISTRIES.get(template.registry)
    if reg is None:
        return False
    for row in reg.all():
        value = getattr(row, template.registry_attr, None)
        if value is not None and abs(float(value) - fact.value) <= 1e-6:
            return True
    return False


def _researched_fact_traces(fact: Fact, corpus: list[Document]) -> bool:
    """A researched fact must anchor to THIS bench's canned corpus: quote verbatim in the cited
    document, value + unit in the quote (the same discipline verify_proposed_fact enforces)."""
    if not fact.quote or not fact.citation:
        return False
    doc = next((d for d in corpus if d.url == fact.citation), None)
    if doc is None:
        return False
    nq = _norm(fact.quote)
    if not nq or nq not in _norm(doc.text):
        return False
    return _value_in_text(fact.value, fact.quote) and fact.unit.strip().lower() in nq


def _derived_fact_traces(fact: Fact) -> bool:
    """A derived fact must BE a reviewed playbook default: same template name, exactly the
    committed default value, and the default citation — anything else is an invented number
    wearing the assumption tier."""
    if fact.citation != "reviewed playbook default":
        return False
    template = _template_named(fact.name)
    return (
        template is not None
        and template.ask_policy == "default"
        and template.default_mm is not None
        and abs(template.default_mm - fact.value) <= 1e-6
    )


def _choice_fact_traces(fact: Fact, form: IntakeForm | None) -> bool:
    """A user fact picked from a fact-CHOICE field traces when its value is one of the form's
    cited standard choices (the canned user picked it off the offered list, N1)."""
    if form is None:
        return False
    want = fact.name.strip().lower()
    for f in form.fields:
        if (
            f.kind == "choice"
            and f.fact_name
            and not f.conflict
            and not f.registry
            and f.fact_name.strip().lower() == want
        ):
            return f"{fact.value:g}" in f.choices
    return False


def invented_facts(
    compiled: CompiledRequirements,
    canned: CannedAnswers,
    corpus: list[Document],
    form: IntakeForm | None = None,
) -> list[str]:
    out: list[str] = []
    for fact in compiled.ledger.facts:
        if fact.source == "user":
            ok = _user_fact_traces(fact, canned) or _choice_fact_traces(fact, form)
        elif fact.source == "registry":
            ok = _registry_fact_traces(fact)
        elif fact.source == "researched":
            ok = _researched_fact_traces(fact, corpus)
        else:  # "derived": only a committed reviewed default may wear this tier
            ok = _derived_fact_traces(fact)
        if not ok:
            out.append(f"{fact.name} ({fact.source}: untraceable -> INVENTED)")
    return out


class FactVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    expected: str
    got: str
    ok: bool


class PromptScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_id: str
    error: str = ""
    route_ok: bool
    missing_domains: tuple[str, ...] = ()
    facts: tuple[FactVerdict, ...] = ()
    recall: float = 1.0
    precision: float = 1.0
    invented: tuple[str, ...] = ()
    question_count: int = 0
    budget_ok: bool = True
    registry_ask_violations: tuple[str, ...] = ()
    variant_ok: bool = True
    dropdown_ok: bool = True
    thermal_ceiling_ok: bool = True
    refusal_ok: bool = True
    spurious_refusal: bool = False
    final_stage: str = ""


def _closure_got(
    name: str, state: ProblemState, round1: IntakeForm | None
) -> tuple[str, Fact | None]:
    want = name.strip().lower()
    compiled = state.compiled
    if compiled is not None:
        for fact in compiled.ledger.facts:
            if fact.name.strip().lower() == want and fact.source in ("registry", "researched"):
                return fact.source, fact
    if round1 is not None:
        for f in round1.fields:
            asked_kinds = f.kind == "number" or (
                f.kind == "choice" and f.fact_name and not f.conflict and not f.registry
            )
            if asked_kinds and (f.fact_name or f.id).strip().lower() == want:
                return "asked", None  # number ask OR an identity-first fact-choice (N1)
    return "absent", None


def score_prompt(
    prompt: BenchPrompt, outcome: PromptOutcome, *, root: Path | None = None
) -> PromptScore:
    route_ok = outcome.route == prompt.expected_route
    if prompt.expected_route == "spec" or outcome.state is None:
        return PromptScore(
            prompt_id=prompt.id,
            error=outcome.error,
            route_ok=route_ok and not outcome.error,
            refusal_ok=not prompt.must_refuse,
            final_stage="spec" if outcome.route == "spec" else "error",
        )
    state = outcome.state
    compiled = state.compiled
    assert compiled is not None  # start_problem always attaches a compile result
    corpus = load_corpus(prompt, root)

    touched = compiled.tree.touched_domains() if compiled.tree is not None else set()
    missing = tuple(sorted(set(prompt.expected_domains) - touched))

    verdicts: list[FactVerdict] = []
    for name, exp in prompt.expected_facts.items():
        got, fact = _closure_got(name, state, outcome.round1_form)
        ok = got == ("asked" if exp.closure == "asked" else exp.closure)
        if ok and exp.value is not None and fact is not None:
            ev, eu = normalize_measure(exp.value, exp.unit)
            fv, fu = fact.as_mm_or_g
            ok = eu == fu and abs(ev - fv) <= 1e-6
        verdicts.append(FactVerdict(name=name, expected=exp.closure, got=got, ok=ok))
    recall = sum(1 for v in verdicts if v.ok) / len(verdicts) if verdicts else 1.0
    expected_names = {n.strip().lower() for n in prompt.expected_facts}
    resolved = [f for f in compiled.ledger.facts if f.source in ("registry", "researched")]
    precision = (
        sum(1 for f in resolved if f.name.strip().lower() in expected_names) / len(resolved)
        if resolved
        else 1.0
    )

    form = outcome.round1_form
    fields = form.fields if form is not None else ()
    question_count = len(fields)
    budget_ok = question_count <= prompt.question_budget if prompt.question_budget else True
    violations: list[str] = []
    for f in fields:
        if f.kind != "number":
            continue
        template = _template_named(f.fact_name or f.id)
        if template is None or template.registry is None or template.registry_attr is None:
            continue
        key = state.registry_keys.get(template.registry)
        if key is None:
            continue
        reg = ALL_REGISTRIES.get(template.registry)
        if (
            reg is not None
            and reg.has(key)
            and getattr(reg.get(key), template.registry_attr, None) is not None
        ):
            violations.append(f.fact_name or f.id)  # asked despite a closing registry row

    variant_ok = (not prompt.expect_variant_question) or any(f.id == "variant" for f in fields)
    dropdown_ok = (prompt.expect_registry_dropdown is None) or any(
        f.id == f"registry:{prompt.expect_registry_dropdown}" for f in fields
    )
    thermal_ok = ("thermal_airflow" not in touched) or any(
        "NOT simulated" in a for a in compiled.assumptions
    )

    refused = state.stage == "refused"
    if prompt.must_refuse:
        refusal_ok = refused and (
            prompt.expected_refusal_gap.lower() in (state.refusal or "").lower()
        )
        spurious = False
    else:
        refusal_ok = not refused
        spurious = refused

    return PromptScore(
        prompt_id=prompt.id,
        error=outcome.error,
        route_ok=route_ok,
        missing_domains=missing,
        facts=tuple(verdicts),
        recall=round(recall, 4),
        precision=round(precision, 4),
        invented=tuple(invented_facts(compiled, prompt.answers, corpus, outcome.round1_form)),
        question_count=question_count,
        budget_ok=budget_ok,
        registry_ask_violations=tuple(violations),
        variant_ok=variant_ok,
        dropdown_ok=dropdown_ok,
        thermal_ceiling_ok=thermal_ok,
        refusal_ok=refusal_ok,
        spurious_refusal=spurious,
        final_stage=state.stage,
    )


class BenchReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    prompts: int
    invented_fact_total: int
    refusal_correctness: float
    route_accuracy: float
    missing_branch_rate: float
    mean_recall: float
    mean_precision: float
    question_discipline: float  # fraction of problem prompts within budget + no registry-ask
    thermal_ceiling_rate: float
    spurious_refusals: int
    errors: int
    gate_pass: bool  # invented == 0 AND refusal correctness == 100%
    scores: tuple[PromptScore, ...]


def aggregate(version: str, scores: list[PromptScore]) -> BenchReport:
    problem_scores = [s for s in scores if s.final_stage not in ("spec",)]
    invented_total = sum(len(s.invented) for s in scores)
    refusal_correct = sum(1 for s in scores if s.refusal_ok) / len(scores) if scores else 0.0
    route_acc = sum(1 for s in scores if s.route_ok) / len(scores) if scores else 0.0
    missing_rate = (
        sum(1 for s in problem_scores if s.missing_domains) / len(problem_scores)
        if problem_scores
        else 0.0
    )
    discipline = (
        sum(1 for s in problem_scores if s.budget_ok and not s.registry_ask_violations)
        / len(problem_scores)
        if problem_scores
        else 1.0
    )
    thermal_rate = (
        sum(1 for s in problem_scores if s.thermal_ceiling_ok) / len(problem_scores)
        if problem_scores
        else 1.0
    )
    recalls = [s.recall for s in problem_scores]
    precisions = [s.precision for s in problem_scores]
    return BenchReport(
        version=version,
        prompts=len(scores),
        invented_fact_total=invented_total,
        refusal_correctness=round(refusal_correct, 4),
        route_accuracy=round(route_acc, 4),
        missing_branch_rate=round(missing_rate, 4),
        mean_recall=round(sum(recalls) / len(recalls), 4) if recalls else 1.0,
        mean_precision=round(sum(precisions) / len(precisions), 4) if precisions else 1.0,
        question_discipline=round(discipline, 4),
        thermal_ceiling_rate=round(thermal_rate, 4),
        spurious_refusals=sum(1 for s in scores if s.spurious_refusal),
        errors=sum(1 for s in scores if s.error),
        gate_pass=invented_total == 0 and refusal_correct == 1.0,
        scores=tuple(scores),
    )


__all__ = [
    "Bench",
    "BenchPrompt",
    "BenchReport",
    "CannedAnswers",
    "ExpectedFact",
    "PromptOutcome",
    "PromptScore",
    "aggregate",
    "answers_for_form",
    "bench_dir",
    "invented_facts",
    "load_bench",
    "load_corpus",
    "provider_for",
    "run_prompt",
    "score_prompt",
]
