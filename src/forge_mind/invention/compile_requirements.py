"""The requirements compiler (ADR-0087 S4): tree + playbooks + facts -> a cited, confirmable spec.

``compile_requirements`` walks the validated requirement tree and closes every leaf through the
honest resolution order — a fact is USER-stated if the user said it, REGISTRY if a cited spec row
carries it, RESEARCHED if the fail-closed verifier admitted it, and otherwise it becomes a QUESTION
or a visible assumption. Nothing is invented at any branch:

  * a registry attribute that is honestly ``None`` (the 2.5" drive's unpublished hole coordinates)
    falls through to research/ask — the registry's own honesty propagates into a question;
  * a check leaf compiles into a :class:`CheckRequest` ONLY when every parameterizing fact resolved;
    otherwise it demotes to a visible assumption ("check not compiled: fact X unresolved");
  * every touched domain's honesty ceiling rides the assumptions VERBATIM (thermal: geometric only)
    whether or not its checks compiled — considered is never presented as validated (P14);
  * fact conflicts (two sources, different values) become questions, never averages (P10).

The result carries everything downstream stages need: the FactLedger (-> the ADR-0040 stated-facts
block for ``produce_part``), the compiled checks (S5 implements them), the open questions (the
clarify round), the assumption ledger, the engine prompt, and the human card (the tree, closures
visible). ``status="needs_input"`` blocks the build until questions are answered (L3/L5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.standards import ALL_REGISTRIES
from forge_mind.invention.decompose import (
    Closure,
    DecomposeResult,
    RequirementNode,
    decompose_problem,
)
from forge_mind.invention.playbooks import PLAYBOOKS, FactTemplate
from forge_mind.invention.requirements import Fact, FactLedger
from forge_mind.invention.research import ResearchProviderLike, research_facts
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike


class CheckRequest(BaseModel):
    """A validator check the spec compiled into (S5 implements it against the built geometry)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    check_id: str
    params: dict[str, float] = Field(default_factory=dict)  # fact name -> normalized mm/g value
    domain: str
    description: str


class OpenQuestion(BaseModel):
    """One remaining question, TYPED (S6b, founder addendum 2026-07-06) so the consolidated
    intake form derives from it deterministically: ``fact`` -> a number+unit field, ``registry``
    -> a dropdown over cited registry rows (one pick closes every fact that registry carries),
    ``variant``/``conflict`` -> a choice, ``ask`` -> free text keyed by the leaf's node id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    why: str
    node_id: str = ""
    kind: Literal["fact", "ask", "variant", "conflict", "registry"] = "fact"
    fact_name: str = ""  # fact/conflict: the ledger name the answer closes
    unit: str = ""  # fact/conflict: the expected unit of a numeric answer
    registry: str = ""  # registry: the ALL_REGISTRIES name the dropdown draws from
    choices: tuple[str, ...] = ()  # registry/variant/conflict: the closed answer set


class CompiledRequirements(BaseModel):
    """The compiler's output: the confirmable spec + everything the build needs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ready", "needs_input"]
    tree: RequirementNode | None
    ledger: FactLedger
    checks: tuple[CheckRequest, ...] = ()
    open_questions: tuple[OpenQuestion, ...] = ()
    assumptions: tuple[str, ...] = ()
    heuristics: tuple[str, ...] = ()
    prompt: str = ""
    extra_context: str = ""
    card: str = ""
    notes: tuple[str, ...] = ()


# N1 identity-first phrasing for the registry dropdowns — a normal person answers these without
# measuring or researching anything (git-reviewed data; falls back to the generic text).
_REGISTRY_IDENTITY_ASK: dict[str, str] = {
    "drive_form": "What size drive is it?",
    "case_fan": "What size is the fan?",
}


def _norm_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def _template_for(domain: str, fact_name: str) -> FactTemplate | None:
    pb = PLAYBOOKS.get(domain)
    if pb is None:
        return None
    want = _norm_name(fact_name)
    for t in pb.facts:
        if _norm_name(t.name) == want:
            return t
    return None


def _ask_text(template: FactTemplate | None, fact_name: str, entity: str) -> str:
    ent = entity.strip() or "the assembly"
    if template is None:
        return f"Provide a value for '{fact_name}' (with units)."
    return template.ask.replace("{entity}", ent)


def _resolve_registry(
    template: FactTemplate, registry_keys: Mapping[str, str], notes: list[str]
) -> Fact | None:
    """A cited registry row closes the fact — unless the attribute is honestly None (unsourced),
    which falls through with a note (the registry's own honesty propagating)."""
    if template.registry is None or template.registry_attr is None:
        return None
    key = registry_keys.get(template.registry)
    if key is None:
        return None
    reg = ALL_REGISTRIES.get(template.registry)
    if reg is None or not reg.has(key):
        return None
    spec = reg.get(key)
    value = getattr(spec, template.registry_attr, None)
    if value is None:
        notes.append(
            f"registry {template.registry}/{key} knows the part but "
            f"{template.registry_attr!r} is not sourced — falling through to research/ask"
        )
        return None
    return Fact(
        name=template.name,
        value=float(value),
        unit=template.unit,
        source="registry",
        citation=str(getattr(spec, "source", f"{template.registry}:{key}")),
    )


def compile_requirements(
    problem: str,
    *,
    client: LLMClientLike,
    model: str,
    user_facts: Sequence[Fact] = (),
    entity: str = "",
    registry_keys: Mapping[str, str] | None = None,
    provider: ResearchProviderLike | None = None,
) -> tuple[CompiledRequirements, list[LLMUsage]]:
    """Decompose -> close every leaf -> assemble the confirmable spec. Deterministic outside the
    two model passes (decompose; research proposal), and fail-closed at every closure."""
    usages: list[LLMUsage] = []
    user_ctx = "\n".join(f"- {f.name} = {f.value} {f.unit} (user)" for f in user_facts)
    decomposed, usage = decompose_problem(problem, client=client, model=model, context=user_ctx)
    usages.append(usage)
    return compile_from_tree(
        problem,
        decomposed,
        client=client,
        model=model,
        user_facts=user_facts,
        entity=entity,
        registry_keys=registry_keys,
        provider=provider,
        usages=usages,
    ), usages


def compile_from_tree(
    problem: str,
    decomposed: DecomposeResult,
    *,
    client: LLMClientLike,
    model: str,
    user_facts: Sequence[Fact] = (),
    known_facts: Sequence[Fact] = (),
    answered_asks: Mapping[str, str] | None = None,
    entity: str = "",
    registry_keys: Mapping[str, str] | None = None,
    provider: ResearchProviderLike | None = None,
    usages: list[LLMUsage] | None = None,
    park_registries: AbstractSet[str] | None = None,
) -> CompiledRequirements:
    """The tree-entry compiler — PUBLIC (S6b): the answer round re-enters HERE with the session's
    CACHED tree plus the new facts; it must never re-decompose (re-decomposition = a new tree = new
    leaves = question churn, the exact failure the 2026-07-06 addendum forbids).

    ``known_facts`` are round-1 registry/researched facts re-entering with their sources INTACT
    (no laundering; a user answer contradicting one becomes a visible CONFLICT question, never a
    silent overwrite). ``answered_asks`` (node_id -> the user's text) closes ``ask`` leaves and
    rides the compiled prompt as explicit user decisions. With ``provider=None`` and a cached tree
    this function performs ZERO model calls — the answer round is fully deterministic.

    ``park_registries`` limits which registries may replace a number question with a DROPDOWN:
    a template's registry hint is only right when the payload actually IS that kind of part
    (a router bracket must never be asked "which drive form?"). None = park any known registry
    (the pre-S7 behavior); the intake surface passes the deterministic text-relevance set."""
    registry_keys = dict(registry_keys or {})
    answered_asks = dict(answered_asks or {})
    usages = usages if usages is not None else []
    notes: list[str] = list(decomposed.notes)
    root = decomposed.root
    if root is None:
        return CompiledRequirements(
            status="needs_input",
            tree=None,
            ledger=FactLedger(),
            open_questions=(
                OpenQuestion(
                    question="The problem could not be decomposed — please restate it with the "
                    "goal and the object it must work with.",
                    why="no valid requirement tree",
                ),
            ),
            notes=tuple(notes),
        )

    leaves = root.leaves()
    facts: list[Fact] = []
    by_name: dict[str, Fact] = {}
    # user facts seed FIRST (a user answer wins closure); known facts (round-1 registry/researched,
    # sources intact) still ENTER the ledger, so a contradicting answer surfaces as a CONFLICT
    # question below — visible, never silently dropped or averaged (P10 + the 2026-07-06 addendum).
    for f in (*user_facts, *known_facts):
        facts.append(f)
        by_name.setdefault(_norm_name(f.name), f)

    questions: list[OpenQuestion] = []
    assumptions: list[str] = []
    # registry name -> fact names parked on ONE dropdown (a cited row closes them all at once)
    missing_registries: dict[str, list[str]] = {}

    def _fact_question(
        template: FactTemplate | None, fname: str, leaf: RequirementNode, why: str
    ) -> OpenQuestion:
        return OpenQuestion(
            question=_ask_text(template, fname, entity),
            why=why,
            node_id=leaf.node_id,
            kind="fact",
            fact_name=fname,
            unit=template.unit if template is not None else "mm",
        )

    def _choice_from_registry(template: FactTemplate, fname: str, leaf: RequirementNode) -> bool:
        """N1 identity-first: the registry KEY is known and the template names a tuple attr of
        standard values (drive height classes) -> the question becomes a CHOICE over those cited
        values, never a free mm ask. Key unknown -> False (the number ask stays the fallback)."""
        if template.registry is None or template.registry_choice_attr is None:
            return False
        key = registry_keys.get(template.registry)
        if key is None:
            return False
        reg = ALL_REGISTRIES.get(template.registry)
        if reg is None or not reg.has(key):
            return False
        values = getattr(reg.get(key), template.registry_choice_attr, None)
        if not values:
            return False
        questions.append(
            OpenQuestion(
                question=template.choice_ask or _ask_text(template, fname, entity),
                why=leaf.statement,
                node_id=leaf.node_id,
                kind="fact",
                fact_name=fname,
                unit=template.unit,
                choices=tuple(f"{float(v):g}" for v in values),
            )
        )
        return True

    def _close_with_default(template: FactTemplate, fname: str) -> bool:
        """ask_policy='default': close with the REVIEWED default as a visible assumption (P6's
        designed path) instead of a question — the run-4 anti-quiz fix. source='derived': never
        an anchor tier (it cannot launder into the stated-facts block), but checks can read it."""
        if template.ask_policy != "default" or template.default_mm is None:
            return False
        fact = Fact(
            name=template.name,
            value=template.default_mm,
            unit=template.unit,
            source="derived",
            citation="reviewed playbook default",
            note=template.default_note,
        )
        facts.append(fact)
        by_name[fname] = fact
        detail = f": {template.default_note}" if template.default_note else ""
        assumptions.append(
            f"{template.name} assumed {template.default_mm:g} {template.unit} "
            f"(reviewed default{detail}) — edit if wrong"
        )
        return True

    def _park_on_registry(template: FactTemplate, fname: str, leaf: RequirementNode) -> bool:
        """Park a fact on its registry DROPDOWN when no key was chosen: one pick closes every fact
        the registry carries. Single-pass completeness (the addendum's no-second-round rule): when
        ANY row honestly lacks this attribute, the number question is ALSO asked NOW — a pick
        landing on that row must not discover a new question after the one answer round."""
        if template.registry is None or template.registry_attr is None:
            return False
        if park_registries is not None and template.registry not in park_registries:
            return False  # the problem text never mentions this kind of part -> plain number ask
        if template.registry in registry_keys:
            return False  # a key was chosen; _resolve_registry already had its say
        reg = ALL_REGISTRIES.get(template.registry)
        if reg is None or not reg.keys():
            return False
        missing_registries.setdefault(template.registry, []).append(fname)
        if any(getattr(row, template.registry_attr, None) is None for row in reg.all()):
            questions.append(
                OpenQuestion(
                    question=_ask_text(template, fname, entity)
                    + " (the standard does not publish this for every variant - measure yours)",
                    why=leaf.statement,
                    node_id=leaf.node_id,
                    kind="fact",
                    fact_name=fname,
                    unit=template.unit,
                )
            )
        return True

    # ---- close the fact leaves: user -> registry -> research -> dropdown/ask -----------------
    research_wanted: dict[str, list[tuple[RequirementNode, FactTemplate]]] = {}
    for leaf in leaves:
        cl = leaf.closure
        if cl is None or cl.kind != "fact":
            continue
        fname = _norm_name(cl.detail or leaf.statement)
        if fname in by_name:
            continue  # the user already stated it (or round 1 already pinned it)
        template = _template_for(leaf.domain, fname)
        if template is not None:
            resolved = _resolve_registry(template, registry_keys, notes)
            if resolved is not None:
                facts.append(resolved)
                by_name[fname] = resolved
                continue
            if _choice_from_registry(template, fname, leaf):
                continue
            if template.research_query is not None and provider is not None:
                query = template.research_query.replace("{entity}", entity or "").strip()
                research_wanted.setdefault(query, []).append((leaf, template))
                continue
            if _close_with_default(template, fname):
                continue
            if _park_on_registry(template, fname, leaf):
                continue
        questions.append(_fact_question(template, fname, leaf, leaf.statement))

    # ---- base gather (founder 2026-07-07: "there are some stuff it always needs") -----------
    # An ACTIVE domain (>=1 fact/check leaf) always gathers its playbook core facts through the
    # SAME honest chain — a forgetful tree cannot silently drop payload_width (the run-2/run-3
    # bench evidence). not_applicable / design_freedom branches stay silent (the P5 guard).
    active_domains = {
        leaf.domain
        for leaf in leaves
        if leaf.closure is not None
        and leaf.closure.kind in ("fact", "check")
        and leaf.domain in PLAYBOOKS
    }
    queued = (
        set(by_name)
        | {_norm_name(q.fact_name) for q in questions if q.fact_name}
        | {_norm_name(t.name) for wanted in research_wanted.values() for _leaf, t in wanted}
        | {_norm_name(n) for names in missing_registries.values() for n in names}
    )
    text_low = problem.lower()
    for domain in sorted(active_domains):
        playbook = PLAYBOOKS[domain]
        domain_relevant = any(t in text_low for t in playbook.triggers)
        for template in playbook.facts:
            fname = _norm_name(template.name)
            if fname in queued:
                continue
            if template.ask_policy == "leaf_only":
                continue  # in the vocabulary for naming convergence; only a tree leaf asks it
            # Relevance gates (run-4 anti-quiz fixes): a registry-backed core fact needs its
            # SUBJECT in the problem text (no fan dimensions for a router bracket); a
            # registry-less core fact needs its DOMAIN's triggers in the text.
            if template.registry is not None:
                if park_registries is not None and template.registry not in park_registries:
                    continue
            elif not domain_relevant:
                continue
            queued.add(fname)
            core = RequirementNode(
                node_id="",
                statement=f"core {domain} fact — always gathered when the domain is active",
                domain=domain,
                closure=Closure(kind="fact", detail=template.name),
            )
            resolved = _resolve_registry(template, registry_keys, notes)
            if resolved is not None:
                facts.append(resolved)
                by_name[fname] = resolved
                continue
            if _choice_from_registry(template, fname, core):
                continue
            if template.research_query is not None and provider is not None:
                query = template.research_query.replace("{entity}", entity or "").strip()
                research_wanted.setdefault(query, []).append((core, template))
                continue
            if _close_with_default(template, fname):
                continue
            if _park_on_registry(template, fname, core):
                continue
            questions.append(_fact_question(template, fname, core, core.statement))

    if research_wanted and provider is not None:
        research_items = list(research_wanted.items())
    else:
        research_items = []
    for query, wanted in research_items:
        assert provider is not None  # research_wanted only fills when a provider exists
        result, usage = research_facts(
            query,
            provider=provider,
            client=client,
            model=model,
            entity=entity,
            # bind the brain's proposals to the template names the compiler matches on (naming is
            # coordination; admission stays with the deterministic quote verifier)
            wanted_names=sorted({t.name for _leaf, t in wanted}),
        )
        usages.append(usage)
        notes.extend(result.dropped)
        notes.extend(result.notes)
        returned = {_norm_name(f.name): f for f in result.facts}
        for leaf, template in wanted:
            fname = _norm_name(template.name)
            fact = returned.get(fname)
            if fact is not None and fname not in by_name:
                facts.append(fact)
                by_name[fname] = fact
            elif fname not in by_name:
                if _close_with_default(template, fname):
                    continue  # research failed; the reviewed default closes it as an assumption
                if _park_on_registry(template, fname, leaf):
                    continue  # research failed but a cited registry row can still close it
                questions.append(
                    _fact_question(
                        template,
                        fname,
                        leaf,
                        f"{leaf.statement} (research could not verify a value)",
                    )
                )
        if len(result.variant_ambiguity) > 1:
            questions.append(
                OpenQuestion(
                    question=(
                        f"Multiple product variants were found ({', '.join(result.variant_ambiguity)}) — "
                        "which exact one do you have?"
                    ),
                    why="researched facts must attach to the right variant",
                    kind="variant",
                    choices=tuple(result.variant_ambiguity),
                )
            )

    # ---- one dropdown per parked registry: a cited row closes every waiting fact at once -----
    for reg_name in sorted(missing_registries):
        waiting = sorted(set(missing_registries[reg_name]))
        reg = ALL_REGISTRIES[reg_name]
        label = reg_name.replace("_", " ")
        questions.append(
            OpenQuestion(
                # identity-first, consumer-grade (N1): one pick, no measuring, no research
                question=_REGISTRY_IDENTITY_ASK.get(reg_name, f"Which {label} is it?"),
                why=f"one pick supplies: {', '.join(waiting)} (cited standard values)",
                kind="registry",
                registry=reg_name,
                choices=tuple(reg.keys()),
            )
        )

    ledger = FactLedger(facts=tuple(facts))
    for conflict in ledger.conflicts():
        av, au = conflict.a.as_mm_or_g
        bv, _bu = conflict.b.as_mm_or_g
        questions.append(
            OpenQuestion(
                question=(
                    f"Sources disagree on '{conflict.name}': {av:g} {au} "
                    f"({conflict.a.citation or 'you'}) vs {bv:g} {au} "
                    f"({conflict.b.citation or 'you'}). Which is right for your hardware?"
                ),
                why="conflicting facts are never averaged",
                kind="conflict",
                fact_name=conflict.name,
                unit=au,
                choices=(f"{av:g}", f"{bv:g}"),
            )
        )

    # ---- close the remaining leaves ----------------------------------------------------------
    checks: list[CheckRequest] = []
    for leaf in leaves:
        cl = leaf.closure
        if cl is None:
            continue
        if cl.kind == "check":
            pb = PLAYBOOKS.get(leaf.domain)
            tmpl = next((c for c in (pb.checks if pb else ()) if c.check_id == cl.detail), None)
            if tmpl is None:  # _validate demotes these already; belt-and-suspenders
                assumptions.append(f"{leaf.statement}: proposed check unavailable — not validated")
                continue
            params: dict[str, float] = {}
            missing: list[str] = []
            for pname in tmpl.params_from:
                fact = by_name.get(_norm_name(pname))
                if fact is None:
                    missing.append(pname)
                else:
                    params[pname] = fact.as_mm_or_g[0]
            if missing:
                assumptions.append(
                    f"{leaf.statement}: check '{tmpl.check_id}' NOT compiled — "
                    f"unresolved fact(s): {', '.join(missing)}"
                )
            else:
                checks.append(
                    CheckRequest(
                        check_id=tmpl.check_id,
                        params=params,
                        domain=leaf.domain,
                        description=tmpl.description,
                    )
                )
        elif cl.kind == "ask":
            if leaf.node_id in answered_asks:
                continue  # closed by the user's one-pass answer; rides the prompt as a decision
            questions.append(
                OpenQuestion(
                    question=cl.detail or leaf.statement,
                    why=leaf.statement,
                    node_id=leaf.node_id,
                    kind="ask",
                )
            )
        elif cl.kind == "unvalidated":
            assumptions.append(f"{leaf.statement}: NOT validated ({cl.detail or 'no instrument'})")

    # ---- ceilings + heuristics for every touched domain (P14: touched != validated) ----------
    heuristics: list[str] = []
    for domain in sorted(root.touched_domains()):
        pb = PLAYBOOKS[domain]
        if pb.honesty_ceiling:
            assumptions.append(pb.honesty_ceiling)
        heuristics.extend(pb.heuristics)

    anchors = ledger.anchors()
    constraint_lines = [f"- {f.name}: {f.as_mm_or_g[0]:g} {f.as_mm_or_g[1]}" for f in anchors]
    # answered ask leaves ride the prompt as explicit user decisions (never as invented facts — L1)
    decision_lines = [
        f"- {leaf.closure.detail or leaf.statement} -> user answered: {answered_asks[leaf.node_id]}"
        for leaf in leaves
        if leaf.closure is not None and leaf.closure.kind == "ask" and leaf.node_id in answered_asks
    ]
    prompt = "\n".join(
        [
            problem.strip(),
            "",
            "Hard constraints (measured or stated facts — hold these exactly):",
            *(constraint_lines or ["- (none resolved yet)"]),
            *(["", "User decisions (stated):", *decision_lines] if decision_lines else []),
            "",
            "Design guidance (advisory):",
            *[f"- {h}" for h in heuristics],
        ]
    )

    status: Literal["ready", "needs_input"] = "needs_input" if questions else "ready"
    return CompiledRequirements(
        status=status,
        tree=root,
        ledger=ledger,
        checks=tuple(checks),
        open_questions=tuple(questions),
        assumptions=tuple(assumptions),
        heuristics=tuple(heuristics),
        prompt=prompt,
        extra_context=ledger.to_stated_facts_block(),
        card=render_card(root, by_name, {q.node_id for q in questions}),
        notes=tuple(notes),
    )


def render_card(
    root: RequirementNode, resolved: Mapping[str, Fact], question_nodes: set[str]
) -> str:
    """The tree as the user's confirm card (text v1; the UI renders richly in S8). Every leaf shows
    HOW it closed — provenance visible, per the card's whole point."""
    lines: list[str] = []

    def tag(node: RequirementNode) -> str:
        cl = node.closure
        if cl is None:
            return ""
        if cl.kind == "fact":
            fact = resolved.get(_norm_name(cl.detail or node.statement))
            if fact is not None:
                label = {
                    "user": "you said",
                    "registry": "standard",
                    "researched": "researched - verify",
                    "derived": "assumed - default",
                }
                v, u = fact.as_mm_or_g
                return f"[ok: {v:g} {u}, {label.get(fact.source, fact.source)}]"
            return "[needs answer]"
        return {
            "check": f"[check: {cl.detail}]",
            "ask": "[needs answer]",
            "design_freedom": "[engine's choice]",
            "not_applicable": "[n/a]",
            "unvalidated": "[NOT validated]",
        }[cl.kind]

    def walk(node: RequirementNode, depth: int) -> None:
        lines.append(f"{'  ' * depth}- {node.statement} {tag(node)}".rstrip())
        for c in node.children:
            walk(c, depth + 1)

    walk(root, 0)
    return "\n".join(lines)


__all__ = [
    "CheckRequest",
    "CompiledRequirements",
    "OpenQuestion",
    "compile_from_tree",
    "compile_requirements",
    "render_card",
]
