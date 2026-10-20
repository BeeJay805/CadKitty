"""problems_v1 bench (ADR-0087 S7), offline: the freeze guard, content meta-validation (the bench
never demands the impossible), the canned user, the deterministic scorer (including adversarial
invented-fact cases), and one full offline drive of a golden prompt + a must-refuse prompt."""

from __future__ import annotations

import hashlib
from typing import Any

from forge_mind.dsl.standards import ALL_REGISTRIES
from forge_mind.invention.bench_problems import (
    BenchPrompt,
    CannedAnswers,
    aggregate,
    answers_for_form,
    bench_dir,
    invented_facts,
    load_bench,
    load_corpus,
    run_prompt,
    score_prompt,
)
from forge_mind.invention.compile_requirements import CompiledRequirements
from forge_mind.invention.decompose import ProposedNode, ProposedTree
from forge_mind.invention.playbooks import PLAYBOOKS
from forge_mind.invention.problem_intake import FormField, IntakeForm
from forge_mind.invention.requirements import Fact, FactLedger
from forge_mind.invention.research import ProposedFact, ResearchProposal
from forge_mind.llm.cost import LLMUsage

# ---------------------------------------------------------------------------------------------
# the freeze guard + content meta-validation
# ---------------------------------------------------------------------------------------------


def _frozen_sha(path: Any) -> str:
    """The freeze hash is over LF-NORMALIZED bytes so the verdict is identical on every checkout
    flavor (Windows CRLF working copies vs Linux CI) — the content is frozen, not the line endings.
    (Found the hard way: the S7 PR went red on CI over exactly this.)"""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_manifest_pins_the_frozen_content() -> None:
    root = bench_dir()
    lines = [
        ln
        for ln in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    assert lines, "empty manifest"
    for ln in lines:
        digest, name = ln.split(maxsplit=1)
        path = root / name.lstrip("*")
        assert _frozen_sha(path) == digest, (
            f"{name}: frozen content changed — that is a NEW bench version"
        )


def test_expected_domains_and_registry_facts_are_satisfiable() -> None:
    bench = load_bench()
    assert len(bench.prompts) >= 12
    for p in bench.prompts:
        for d in p.expected_domains:
            assert d in PLAYBOOKS, f"{p.id}: unknown domain {d!r}"
        for name, exp in p.expected_facts.items():
            if exp.closure != "registry":
                continue
            template = next(
                (
                    t
                    for pb in PLAYBOOKS.values()
                    for t in pb.facts
                    if t.name == name and t.registry is not None
                ),
                None,
            )
            assert template is not None, f"{p.id}: {name} has no registry template"
            assert template.registry is not None and template.registry_attr is not None
            reg = ALL_REGISTRIES[template.registry]
            values = [getattr(row, template.registry_attr, None) for row in reg.all()]
            assert exp.value is not None and any(
                v is not None and abs(float(v) - exp.value) <= 1e-6 for v in values
            ), f"{p.id}: expected registry value {exp.value} for {name} matches no cited row"


def test_researched_expectations_are_anchorable_in_their_corpus() -> None:
    # the meta-rule: a researched expectation must be quotable from the canned corpus, or the
    # bench would demand an invented fact
    bench = load_bench()
    for p in bench.prompts:
        for name, exp in p.expected_facts.items():
            if exp.closure != "researched":
                continue
            docs = load_corpus(p)
            assert docs, f"{p.id}: researched expectation {name} but no corpus"
            token = f"{exp.value:g} {exp.unit}"
            assert any(token in d.text for d in docs), (
                f"{p.id}: '{token}' for {name} is not stated in the corpus"
            )


def test_class_coverage_spans_the_adr_mix() -> None:
    bench = load_bench()
    classes = {p.class_ for p in bench.prompts}
    assert {
        "route_spec",
        "registry_only",
        "needs_research",
        "needs_clarify",
        "ambiguous_variant",
        "registry_dropdown",
        "must_refuse",
    } <= classes
    assert sum(1 for p in bench.prompts if p.must_refuse) >= 2
    assert sum(1 for p in bench.prompts if p.golden_build) == 3


# ---------------------------------------------------------------------------------------------
# the canned user
# ---------------------------------------------------------------------------------------------


def test_answers_for_form_covers_each_kind() -> None:
    canned = CannedAnswers(
        facts={"payload_thickness": {"value": 9.5, "unit": "mm"}},
        unknown_facts=("available_height",),
        registry_choices={"drive_form": "2.5in_drive"},
        variant_choice="NR200",
        default_text="bottom rail",
    )
    form = IntakeForm(
        fields=(
            FormField(
                id="payload_thickness",
                kind="number",
                question="t?",
                fact_name="payload_thickness",
                unit="mm",
            ),
            FormField(
                id="available_height",
                kind="number",
                question="h?",
                fact_name="available_height",
                unit="mm",
            ),
            FormField(
                id="uncovered_thing",
                kind="number",
                question="u?",
                fact_name="uncovered_thing",
                unit="mm",
            ),
            FormField(id="ask:n1", kind="text", question="where?"),
            FormField(
                id="registry:drive_form",
                kind="choice",
                question="which?",
                choices=("2.5in_drive", "3.5in_drive"),
                registry="drive_form",
            ),
            FormField(
                id="variant", kind="choice", question="which variant?", choices=("NR200", "NR200P")
            ),
        )
    )
    by_id = {a.id: a for a in answers_for_form(form, canned)}
    assert by_id["payload_thickness"].value == 9.5
    assert by_id["available_height"].unknown is True  # explicitly unknown wins over nothing
    assert by_id["uncovered_thing"].unknown is True  # no fallback -> honestly unknown, never faked
    assert by_id["ask:n1"].text == "bottom rail"
    assert by_id["registry:drive_form"].choice == "2.5in_drive"
    assert by_id["variant"].choice == "NR200"
    # a solvable prompt's fallback answers the uncovered number; explicit unknowns still win
    with_fb = canned.model_copy(update={"fallback_number": 30.0})
    by_id = {a.id: a for a in answers_for_form(form, with_fb)}
    assert by_id["uncovered_thing"].value == 30.0
    assert by_id["available_height"].unknown is True
    # the base intake fact: the canned user always picks a material (default PLA)
    material_form = IntakeForm(
        fields=(
            FormField(id="material", kind="choice", question="material?", choices=("PLA", "PETG")),
        )
    )
    (mat,) = answers_for_form(material_form, canned)
    assert mat.choice == "PLA"


# ---------------------------------------------------------------------------------------------
# the invented-fact verifier (adversarial)
# ---------------------------------------------------------------------------------------------


def _compiled_with(facts: tuple[Fact, ...]) -> CompiledRequirements:
    return CompiledRequirements(status="ready", tree=None, ledger=FactLedger(facts=facts))


def test_invented_detection_catches_all_three_launder_paths() -> None:
    bench = load_bench()
    nr200 = next(p for p in bench.prompts if p.id == "nr200_ssd")
    corpus = load_corpus(nr200)
    fabricated = (
        # a "user" fact the canned user never stated
        Fact(name="payload_thickness", value=12.0, unit="mm", source="user"),
        # a "registry" fact matching no cited row value
        Fact(name="payload_width", value=68.0, unit="mm", source="registry", citation="x"),
        # a "researched" fact whose quote is not in the corpus
        Fact(
            name="available_height",
            value=92.0,
            unit="mm",
            source="researched",
            citation="bench://nr200/spec",
            quote="the clearance is definitely 92 mm, trust me",
        ),
    )
    bad = invented_facts(_compiled_with(fabricated), nr200.answers, corpus)
    assert len(bad) == 3


def test_traceable_facts_are_not_invented() -> None:
    bench = load_bench()
    nr200 = next(p for p in bench.prompts if p.id == "nr200_ssd")
    corpus = load_corpus(nr200)
    good = (
        Fact(name="payload_thickness", value=9.5, unit="mm", source="user"),
        Fact(
            name="payload_width",
            value=69.85,
            unit="mm",
            source="registry",
            citation="SFF-8201 ...",
        ),
        Fact(
            name="available_height",
            value=92.0,
            unit="mm",
            source="researched",
            citation="bench://nr200/spec",
            quote="offers a clearance height of 92 mm for drive mounting",
        ),
    )
    assert invented_facts(_compiled_with(good), nr200.answers, corpus) == []


# ---------------------------------------------------------------------------------------------
# full offline drives (canned brain): a golden prompt + a must-refuse prompt
# ---------------------------------------------------------------------------------------------


def _tree_for(prompt_id: str) -> ProposedTree:
    n = ProposedNode
    if prompt_id == "car_dashboard":
        return ProposedTree(
            nodes=[
                n(node_id="r", parent_id=None, statement="mount an SSD in the dash"),
                n(
                    node_id="w",
                    parent_id="r",
                    statement="span the SSD",
                    domain="mount_interface",
                    closure_kind="fact",
                    closure_detail="payload_width",
                ),
                n(
                    node_id="t",
                    parent_id="r",
                    statement="thickness",
                    domain="mount_interface",
                    closure_kind="fact",
                    closure_detail="payload_thickness",
                ),
                n(
                    node_id="h",
                    parent_id="r",
                    statement="fit the cavity",
                    domain="fit_envelope",
                    closure_kind="fact",
                    closure_detail="available_height",
                ),
            ]
        )
    return ProposedTree(  # nr200_hdd35
        nodes=[
            n(node_id="r", parent_id=None, statement="mount a 3.5in HDD in the NR200"),
            n(
                node_id="w",
                parent_id="r",
                statement="span the HDD",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_width",
            ),
            n(
                node_id="l",
                parent_id="r",
                statement="HDD length",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_length",
            ),
            n(
                node_id="hp",
                parent_id="r",
                statement="bolt to the HDD holes",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="mount_hole_pitch",
            ),
            n(
                node_id="t",
                parent_id="r",
                statement="HDD thickness",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_thickness",
            ),
            n(
                node_id="h",
                parent_id="r",
                statement="fit under the shroud",
                domain="fit_envelope",
                closure_kind="fact",
                closure_detail="available_height",
            ),
        ]
    )


class _Brain:
    """Canned decompose + research proposal quoting the REAL corpus doc."""

    def __init__(self, prompt_id: str) -> None:
        self.prompt_id = prompt_id

    def generate_structured(
        self, *, output_model: type, model: str, user: str = "", **_: Any
    ) -> tuple[Any, LLMUsage]:
        usage = LLMUsage(model=model, input_tokens=1, output_tokens=1)
        if output_model is ProposedTree:
            return _tree_for(self.prompt_id), usage
        if output_model is ResearchProposal:
            return (
                ResearchProposal(
                    facts=[
                        ProposedFact(
                            name="available_height",
                            value=92.0,
                            unit="mm",
                            quote="offers a clearance height of 92 mm for drive mounting",
                            url="bench://nr200/spec",
                        )
                    ]
                ),
                usage,
            )
        raise AssertionError(output_model)


def test_offline_drive_golden_hdd35_scores_clean() -> None:
    bench = load_bench()
    prompt = next(p for p in bench.prompts if p.id == "nr200_hdd35")
    outcome = run_prompt(prompt, client=_Brain(prompt.id), model="m")
    score = score_prompt(prompt, outcome)
    assert score.error == "" and score.final_stage == "spec_ready"
    assert score.invented == ()  # registry + researched + user all trace
    assert score.recall == 1.0 and score.route_ok
    assert score.missing_domains == ()
    assert score.budget_ok and score.registry_ask_violations == ()
    assert score.refusal_ok and not score.spurious_refusal


def test_offline_drive_car_dashboard_refuses_with_the_gap() -> None:
    bench = load_bench()
    prompt = next(p for p in bench.prompts if p.id == "car_dashboard")
    outcome = run_prompt(prompt, client=_Brain(prompt.id), model="m")
    score = score_prompt(prompt, outcome)
    assert score.final_stage == "refused"
    assert score.refusal_ok  # refused AND names the expected gap
    assert score.invented == ()


def test_spec_prompts_never_touch_the_brain() -> None:
    bench = load_bench()
    spec = next(p for p in bench.prompts if p.id == "spec_plate")

    class _Never:
        def generate_structured(self, **_: Any) -> tuple[Any, LLMUsage]:
            raise AssertionError("spec routing must not call the model")

    outcome = run_prompt(spec, client=_Never(), model="m")
    score = score_prompt(spec, outcome)
    assert outcome.route == "spec" and score.route_ok


def test_aggregate_gates_on_invented_and_refusal() -> None:
    bench = load_bench()
    prompt = next(p for p in bench.prompts if p.id == "nr200_hdd35")
    refuse = next(p for p in bench.prompts if p.id == "car_dashboard")
    ok = score_prompt(prompt, run_prompt(prompt, client=_Brain(prompt.id), model="m"))
    ref = score_prompt(refuse, run_prompt(refuse, client=_Brain(refuse.id), model="m"))
    report = aggregate("problems_v1", [ok, ref])
    assert report.gate_pass and report.invented_fact_total == 0
    # flip: a spurious refusal breaks the gate
    broken = ok.model_copy(update={"refusal_ok": False, "spurious_refusal": True})
    report2 = aggregate("problems_v1", [broken, ref])
    assert not report2.gate_pass


def test_prompt_alias_seeding_matches_bench_assumptions() -> None:
    # the frozen set assumes which prompts seed which registry keys — pin it against the code
    from forge_mind.invention.problem_intake import seed_registry_keys

    bench = load_bench()
    by_id: dict[str, BenchPrompt] = {p.id: p for p in bench.prompts}
    assert seed_registry_keys(by_id["nr200_ssd"].goal + " " + by_id["nr200_ssd"].context) == {
        "drive_form": "2.5in_drive"
    }
    assert seed_registry_keys(by_id["nr200_hdd35"].goal + " " + by_id["nr200_hdd35"].context) == {
        "drive_form": "3.5in_drive"
    }
    assert seed_registry_keys(by_id["meshify_duct"].goal + " " + by_id["meshify_duct"].context) == {
        "case_fan": "fan_120"
    }
    assert (
        seed_registry_keys(by_id["c4000d_dropdown"].goal + " " + by_id["c4000d_dropdown"].context)
        == {}
    )  # no size named -> the dropdown must ask
