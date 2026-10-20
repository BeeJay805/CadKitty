"""The S6b intake surface (founder addendum 2026-07-06): deterministic spec/problem classification,
the single gather pass -> ONE typed form, the cached tree (the answer round NEVER re-decomposes and
NEVER calls the model), refusal as a first-class outcome, and the conflict-only re-ask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

from forge_mind.dsl.standards import ALL_REGISTRIES
from forge_mind.invention.decompose import (
    Closure,
    DecomposeResult,
    ProposedNode,
    ProposedTree,
    RequirementNode,
)
from forge_mind.invention.lifecycle import GeneratedPart
from forge_mind.invention.problem_intake import (
    _REGISTRY_ALIASES,
    FormField,
    IntakeAnswerError,
    IntakeAnswerItem,
    IntakeForm,
    ProblemIntake,
    ProblemState,
    apply_answers,
    classify_intake,
    confirm_and_build,
    detect_entity,
    seed_registry_keys,
    start_problem,
)
from forge_mind.invention.requirements import Fact
from forge_mind.invention.research import (
    Document,
    FakeResearchProvider,
    ProposedFact,
    ResearchProposal,
)
from forge_mind.kit import Kit, build_kit
from forge_mind.llm.cost import LLMUsage

# ---------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------

_NR200_DOC = Document(
    url="https://example.com/nr200-spec",
    title="Cooler Master NR200 specifications",
    text="The NR200 mini-ITX case: internal clearance height is 4 inches at the bottom rail.",
    fetched_at="2026-07-06",
)


def _ssd_tree() -> ProposedTree:
    n = ProposedNode
    return ProposedTree(
        nodes=[
            n(node_id="r", parent_id=None, statement="mount one more SSD in the NR200"),
            n(
                node_id="w",
                parent_id="r",
                statement="span the SSD width",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="payload_width",
            ),
            n(
                node_id="hp",
                parent_id="r",
                statement="bolt to the SSD's holes",
                domain="mount_interface",
                closure_kind="fact",
                closure_detail="mount_hole_pitch",
            ),
            n(
                node_id="t",
                parent_id="r",
                statement="match the SSD thickness",
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
            n(
                node_id="loc",
                parent_id="r",
                statement="pick the mounting location",
                closure_kind="ask",
                closure_detail="Which location - the bottom rail or behind the tray?",
            ),
        ]
    )


class _Client:
    """Canned brain: counts decompose calls (the no-re-decompose proof) and serves the research
    proposal. Any unexpected structured call fails the test."""

    def __init__(self, tree: ProposedTree, proposal: ResearchProposal | None = None) -> None:
        self.tree = tree
        self.proposal = proposal or ResearchProposal()
        self.decompose_calls = 0

    def generate_structured(
        self, *, output_model: type, model: str, **_: Any
    ) -> tuple[Any, LLMUsage]:
        usage = LLMUsage(model=model, input_tokens=1, output_tokens=1)
        if output_model is ProposedTree:
            self.decompose_calls += 1
            return self.tree, usage
        if output_model is ResearchProposal:
            return self.proposal, usage
        raise AssertionError(f"unexpected output_model {output_model}")


def _proposal() -> ResearchProposal:
    return ResearchProposal(
        facts=[
            ProposedFact(
                name="available_height",
                value=4.0,
                unit="inches",
                quote="internal clearance height is 4 inches",
                url=_NR200_DOC.url,
            )
        ]
    )


def _start(goal: str = "I need to mount one more SSD", context: str = "Cooler Master NR200"):
    client = _Client(_ssd_tree(), _proposal())
    state, usages = start_problem(
        ProblemIntake(goal=goal, context=context),
        client=client,
        model="m",
        provider=FakeResearchProvider({"nr200": [_NR200_DOC]}),
    )
    return state, client, usages


class _Produce:
    """A spy engine adapter: records calls, writes a small box STL so env checks can run."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, prompt: str, extra_context: str, material: str = "PLA") -> GeneratedPart:
        self.calls.append((prompt, extra_context, material))
        stl = self.tmp / "part.stl"
        trimesh.creation.box(extents=(90.0, 60.0, 8.0)).export(str(stl))
        return GeneratedPart(
            part_name="ssd bracket",
            disposition="shipped",
            kit=_kit(),
            artifacts={"stl": str(stl)},
        )


def _kit() -> Kit:
    from forge_mind.dsl.primitives import Box, OpMode
    from forge_mind.dsl.program import Constraints, Metadata, PartProgram
    from forge_mind.types import Vec3

    program = PartProgram(
        metadata=Metadata(name="ssd-bracket", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


# ---------------------------------------------------------------------------------------------
# the deterministic classifier + seeding (P17)
# ---------------------------------------------------------------------------------------------


def test_spec_prompts_classify_as_spec() -> None:
    # plain geometric descriptions belong to /generate — no problem markers, no context
    assert classify_intake("a 40 x 40 x 4 mm plate with four M3 holes at the corners", "") == "spec"
    assert classify_intake("hex standoff, M3 thread, 20mm tall", "") == "spec"


def test_problem_prompts_classify_as_problem() -> None:
    assert classify_intake("I need to mount one more SSD", "") == "problem"
    assert classify_intake("mount an SSD in my NR200", "") == "problem"
    # a supplied context field is a deliberate problem intake regardless of goal phrasing
    assert classify_intake("SSD bracket", "Cooler Master NR200") == "problem"


def test_registry_alias_table_keys_exist() -> None:
    # the P9 discipline: an alias pointing at a nonexistent registry row would poison seeding
    for reg_name, keymap in _REGISTRY_ALIASES.items():
        reg = ALL_REGISTRIES[reg_name]
        for key in keymap:
            assert reg.has(key), f"{reg_name} alias key {key!r} is not a registry row"


def test_seed_registry_keys_unambiguous_only() -> None:
    assert seed_registry_keys("I need to mount a 2.5in SSD") == {"drive_form": "2.5in_drive"}
    # both sizes mentioned -> ambiguous -> no seed (the dropdown will ask)
    assert seed_registry_keys("either a 2.5in or 3.5in drive") == {}


def test_detect_entity_finds_the_product_token() -> None:
    assert detect_entity("Cooler Master NR200 case") == "NR200"
    assert detect_entity("a plain shelf") == ""


# ---------------------------------------------------------------------------------------------
# the single gather pass -> ONE typed form
# ---------------------------------------------------------------------------------------------


def test_single_pass_builds_the_complete_typed_form() -> None:
    state, client, _ = _start()
    assert client.decompose_calls == 1
    assert state.stage == "awaiting_answers"
    assert state.form is not None
    by_id = {f.id: f for f in state.form.fields}
    # ONE registry dropdown closes payload_width + mount_hole_pitch at once
    drop = by_id["registry:drive_form"]
    assert drop.kind == "choice" and set(drop.choices) == {"2.5in_drive", "3.5in_drive"}
    # single-pass completeness: a row of drive_form honestly lacks the pitch -> asked NOW too
    assert by_id["mount_hole_pitch"].kind == "number"
    assert by_id["payload_thickness"].kind == "number" and by_id["payload_thickness"].unit == "mm"
    assert by_id["ask:loc"].kind == "text"
    # research already pinned available_height -> NOT on the form; cached as a known fact
    assert "available_height" not in by_id
    assert [f.name for f in state.known_facts] == ["available_height"]
    assert state.known_facts[0].source == "researched"


def test_rich_seed_shrinks_the_form() -> None:
    # "2.5in SSD" in the goal seeds drive_form -> payload_width resolves; the dropdown vanishes
    state, _, _ = _start(goal="I need to mount one more 2.5in SSD")
    assert state.registry_keys == {"drive_form": "2.5in_drive"}
    assert state.form is not None
    ids = {f.id for f in state.form.fields}
    assert "registry:drive_form" not in ids
    assert {"mount_hole_pitch", "payload_thickness", "ask:loc"} <= ids


# ---------------------------------------------------------------------------------------------
# the answer round: deterministic, no re-decompose, no model, no research
# ---------------------------------------------------------------------------------------------


def _answers_for(form: IntakeForm) -> list[IntakeAnswerItem]:
    out: list[IntakeAnswerItem] = []
    for f in form.fields:
        if f.kind == "number":
            out.append(IntakeAnswerItem(id=f.id, value=9.5 if "thickness" in f.id else 76.2))
        elif f.kind == "choice":
            out.append(IntakeAnswerItem(id=f.id, choice=f.choices[0]))
        else:
            out.append(IntakeAnswerItem(id=f.id, text="bottom rail"))
    return out


def test_answers_close_without_redecompose_or_model() -> None:
    state, client, _ = _start()
    assert state.form is not None
    done = apply_answers(state, _answers_for(state.form))
    # apply_answers uses a client that RAISES on any call — reaching spec_ready proves zero
    # model calls; the canned client's decompose count proves the tree was never re-built.
    assert client.decompose_calls == 1
    assert done.stage == "spec_ready" and done.form is None and done.compiled is not None
    ctx = done.compiled.extra_context
    assert "(source: researched)" in ctx  # round-1 research persisted, tier INTACT (no laundering)
    assert "(source: user)" in ctx
    assert "(source: standard)" in ctx  # the dropdown pick resolved the registry facts
    assert "user answered: bottom rail" in done.compiled.prompt  # the ask decision rides the prompt


def test_unknown_answer_flows_into_the_l5_refusal() -> None:
    # S7: "I cannot supply this" satisfies completeness but withholds the value, so the round
    # lands on the designed refusal naming the fact — never a dead-ended form, never a guess
    state, _, _ = _start()
    assert state.form is not None
    answers = [
        a if a.id != "payload_thickness" else IntakeAnswerItem(id="payload_thickness", unknown=True)
        for a in _answers_for(state.form)
    ]
    done = apply_answers(state, answers)
    assert done.stage == "refused"
    assert done.refusal is not None and "payload_thickness" in done.refusal


def test_incomplete_or_unknown_answers_fail_closed() -> None:
    state, _, _ = _start()
    assert state.form is not None
    try:
        apply_answers(state, [IntakeAnswerItem(id="payload_thickness", value=9.5)])
        raise AssertionError("incomplete answers must not apply")
    except IntakeAnswerError as exc:
        assert "unanswered" in str(exc)
    bad = [*_answers_for(state.form), IntakeAnswerItem(id="nope", value=1.0)]
    try:
        apply_answers(state, bad)
        raise AssertionError("unknown field ids must not apply")
    except IntakeAnswerError as exc:
        assert "unknown field" in str(exc)


# ---------------------------------------------------------------------------------------------
# refusal (L5) + the conflict-only re-ask
# ---------------------------------------------------------------------------------------------


def _leaf(node_id: str, statement: str, kind: str, detail: str) -> RequirementNode:
    return RequirementNode(
        node_id=node_id,
        statement=statement,
        domain="other",
        closure=Closure(kind=kind, detail=detail),  # type: ignore[arg-type]
    )


def _manual_state(
    *,
    leaves: tuple[RequirementNode, ...],
    form: IntakeForm,
    known: tuple[Fact, ...] = (),
    conflict_round_used: bool = False,
) -> ProblemState:
    root = RequirementNode(node_id="r", statement="goal", domain="other", children=leaves)
    return ProblemState(
        stage="awaiting_answers",
        goal="goal",
        context="ctx",
        tree=DecomposeResult(root=root),
        known_facts=known,
        form=form,
        conflict_round_used=conflict_round_used,
    )


def test_ungettable_fact_after_the_single_pass_refuses() -> None:
    # the tree carries a fact the (simulated) round-1 form never covered — after the one answer
    # round it is STILL open, so the intake REFUSES naming it (never a second question round)
    state = _manual_state(
        leaves=(
            _leaf("a", "known thing", "fact", "known_thing"),
            _leaf("b", "the unforeseen gap", "fact", "mystery_gap"),
        ),
        # the round-1 form only asked for known_thing
        form=IntakeForm(
            fields=(
                FormField(
                    id="known_thing",
                    kind="number",
                    question="known?",
                    fact_name="known_thing",
                    unit="mm",
                ),
            )
        ),
    )
    done = apply_answers(state, [IntakeAnswerItem(id="known_thing", value=5.0)])
    assert done.stage == "refused"
    assert done.refusal is not None and "mystery_gap" in done.refusal
    assert done.form is None


def test_contradiction_reasks_once_then_resolves_with_a_visible_override() -> None:
    researched = Fact(
        name="avail_h",
        value=101.6,
        unit="mm",
        source="researched",
        citation="https://example.com/spec",
        quote="4 inches",
    )
    state = _manual_state(
        leaves=(_leaf("a", "height", "fact", "avail_h"),),
        form=IntakeForm(
            fields=(
                FormField(
                    id="avail_h", kind="number", question="h?", fact_name="avail_h", unit="mm"
                ),
            )
        ),
        known=(researched,),
    )
    # the user's answer contradicts the researched value -> ONE conflict re-ask, never an average
    round2 = apply_answers(state, [IntakeAnswerItem(id="avail_h", value=95.0)])
    assert round2.stage == "awaiting_answers" and round2.conflict_round_used is True
    assert round2.form is not None and len(round2.form.fields) == 1
    conflict = round2.form.fields[0]
    assert conflict.conflict is True and set(conflict.choices) == {"95", "101.6"}
    # the user confirms their measurement -> override recorded VISIBLY, sourced fact suppressed
    done = apply_answers(round2, [IntakeAnswerItem(id=conflict.id, choice="95")])
    assert done.stage == "spec_ready"
    assert any("override" in n for n in done.notes)
    assert all(f.name != "avail_h" for f in done.known_facts)


def test_second_contradiction_refuses() -> None:
    researched = Fact(
        name="avail_h",
        value=101.6,
        unit="mm",
        source="researched",
        citation="https://example.com/spec",
    )
    state = _manual_state(
        leaves=(_leaf("a", "height", "fact", "avail_h"),),
        form=IntakeForm(
            fields=(
                FormField(
                    id="avail_h", kind="number", question="h?", fact_name="avail_h", unit="mm"
                ),
            )
        ),
        known=(researched,),
        conflict_round_used=True,  # the one allowed conflict round is already spent
    )
    done = apply_answers(state, [IntakeAnswerItem(id="avail_h", value=95.0)])
    assert done.stage == "refused"
    assert done.refusal is not None and "contradictor" in done.refusal


# ---------------------------------------------------------------------------------------------
# the base intake fact (founder 2026-07-07): material always gathered, never defaulted
# ---------------------------------------------------------------------------------------------


def test_material_field_rides_every_unseeded_form() -> None:
    state, _, _ = _start()
    assert state.material is None
    assert state.form is not None
    material = next(f for f in state.form.fields if f.id == "material")
    assert material.kind == "choice" and "PLA" in material.choices


def test_material_seeds_from_the_users_own_words() -> None:
    state, _, _ = _start(goal="I need to mount one more SSD, printed in PETG")
    assert state.material == "PETG"
    assert state.form is not None
    assert all(f.id != "material" for f in state.form.fields)  # the rich seed shrank the form


def test_material_answer_lands_on_the_state() -> None:
    state, _, _ = _start()
    assert state.form is not None
    done = apply_answers(state, _answers_for(state.form))
    assert done.material == "PLA"  # _answers_for picks the first choice


def test_unknown_material_refuses_naming_it() -> None:
    state, _, _ = _start()
    assert state.form is not None
    answers = [
        a if a.id != "material" else IntakeAnswerItem(id="material", unknown=True)
        for a in _answers_for(state.form)
    ]
    done = apply_answers(state, answers)
    assert done.stage == "refused"
    assert done.refusal is not None and "material" in done.refusal


def test_confirm_without_material_refuses(tmp_path: Path) -> None:
    # a synthetic spec_ready state with no material must not spend (no silent default, ever)
    state, _, _ = _start()
    ready = state.model_copy(update={"stage": "spec_ready", "form": None, "material": None})
    produce = _Produce(tmp_path)
    try:
        confirm_and_build(ready, produce=produce)
        raise AssertionError("confirm without a material must refuse")
    except IntakeAnswerError as exc:
        assert "material" in str(exc)
    assert produce.calls == []


# ---------------------------------------------------------------------------------------------
# N1: material planning over the user's availability set + the fabrication base fact
# ---------------------------------------------------------------------------------------------


def _thermal_tree() -> ProposedTree:
    n = ProposedNode
    return ProposedTree(
        nodes=[
            n(node_id="r", parent_id=None, statement="duct air from the fan"),
            n(
                node_id="k",
                parent_id="r",
                statement="do not block the fan",
                domain="thermal_airflow",
                closure_kind="check",
                closure_detail="keepout_clear",
            ),
        ]
    )


def test_material_recommended_from_the_users_set_when_thermal_active() -> None:
    client = _Client(_thermal_tree())
    state, _ = start_problem(
        ProblemIntake(
            goal="I want a duct for the front fan of my case",
            context="front 120 mm fan",
            materials=("PLA", "PETG"),
        ),
        client=client,
        model="m",
        provider=FakeResearchProvider({}),
    )
    assert state.form is not None
    material = next(f for f in state.form.fields if f.id == "material")
    assert material.choices == ("PLA", "PETG")  # ONLY what the user can print with
    assert material.recommended == "PETG"  # the highest cited HDT of THEIR set
    assert "70" in material.reason and "NOT simulated" in material.reason  # cited + ceiling (L6)


def test_material_not_recommended_without_a_thermal_branch() -> None:
    state, _, _ = _start()  # the SSD tree has no active thermal branch
    assert state.form is not None
    material = next(f for f in state.form.fields if f.id == "material")
    assert material.recommended == "" and material.reason == ""


def test_seeded_material_outside_the_availability_set_is_asked_not_kept() -> None:
    state, _, _ = _start(goal="I need to mount one more SSD, printed in PETG")
    assert state.material == "PETG"  # unrestricted set -> the seed holds
    client = _Client(_ssd_tree(), _proposal())
    restricted, _ = start_problem(
        ProblemIntake(
            goal="I need to mount one more SSD, printed in PETG",
            context="Cooler Master NR200",
            materials=("PLA",),
        ),
        client=client,
        model="m",
        provider=FakeResearchProvider({"nr200": [_NR200_DOC]}),
    )
    assert restricted.material is None  # they cannot print PETG -> ask from their set
    assert restricted.form is not None
    material = next(f for f in restricted.form.fields if f.id == "material")
    assert material.choices == ("PLA",)


def test_unknown_material_name_in_the_set_fails_closed() -> None:
    try:
        start_problem(
            ProblemIntake(goal="a bracket", context="ctx", materials=("PLA", "UNOBTAINIUM")),
            client=_Client(_ssd_tree()),
            model="m",
            provider=FakeResearchProvider({}),
        )
        raise AssertionError("an unknown material must not pass")
    except IntakeAnswerError as exc:
        assert "UNOBTAINIUM" in str(exc)


def test_cnc_process_refuses_honestly_before_any_model_call() -> None:
    class _Never:
        def generate_structured(self, **_: Any) -> tuple[Any, LLMUsage]:
            raise AssertionError("cnc must refuse before decompose")

    state, usages = start_problem(
        ProblemIntake(goal="a machined bracket", context="my mill", process="cnc"),
        client=_Never(),
        model="m",
        provider=FakeResearchProvider({}),
    )
    assert state.stage == "refused"
    assert state.refusal is not None and "CNC" in state.refusal
    assert usages == []  # no spend on a lane we cannot validate


def test_thickness_becomes_an_identity_choice_when_the_drive_is_known() -> None:
    # N1 identity-first: "2.5in SSD" seeds the registry key -> thickness is a CHOICE over the
    # row's cited height classes, never a free mm ask
    state, _, _ = _start(goal="I need to mount one more 2.5in SSD")
    assert state.form is not None
    thickness = next(f for f in state.form.fields if f.fact_name == "payload_thickness")
    assert thickness.kind == "choice" and thickness.choices == ("7", "9.5", "15")
    done = apply_answers(state, _answers_for(state.form))
    assert done.stage == "spec_ready"
    fact = next(f for f in done.compiled.ledger.facts if f.name == "payload_thickness")  # type: ignore[union-attr]
    assert fact.source == "user" and fact.value == 7.0  # the first cited class was picked


# ---------------------------------------------------------------------------------------------
# the confirm gate (L3): spec_ready + explicit confirm is the ONLY spend path
# ---------------------------------------------------------------------------------------------


def test_confirm_builds_the_exact_spec_and_runs_env_checks(tmp_path: Path) -> None:
    state, _, _ = _start()
    assert state.form is not None
    ready = apply_answers(state, _answers_for(state.form))
    assert ready.stage == "spec_ready"
    produce = _Produce(tmp_path)
    built = confirm_and_build(ready, produce=produce)
    assert built.stage == "built" and built.build is not None
    assert len(produce.calls) == 1
    prompt, ctx, material = produce.calls[0]
    assert ready.compiled is not None
    assert (prompt, ctx) == (ready.compiled.prompt, ready.compiled.extra_context)  # the EXACT spec
    assert material == "PLA"  # the canned pick reached the engine adapter, never a silent default
    assert built.build.disposition == "shipped"


def test_confirm_refuses_any_stage_but_spec_ready(tmp_path: Path) -> None:
    state, _, _ = _start()
    produce = _Produce(tmp_path)
    try:
        confirm_and_build(state, produce=produce)
        raise AssertionError("an unanswered session must not build")
    except IntakeAnswerError:
        pass
    assert produce.calls == []  # NO SPEND
