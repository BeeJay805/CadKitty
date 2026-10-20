"""Problem-mode orchestration (ADR-0087 S6a): the confirm gate is CODE — an unconfirmed or
question-bearing solve structurally cannot spend; answers loop back as STATED facts; the built
path runs the advisory env checks and assembles the assumption ledger."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import trimesh

from forge_mind.dsl.standards import ALL_REGISTRIES  # noqa: F401  (registry import sanity)
from forge_mind.invention.decompose import ProposedNode, ProposedTree
from forge_mind.invention.lifecycle import GeneratedPart
from forge_mind.invention.problem_mode import (
    AnsweredFact,
    answers_to_facts,
    problem_mode_enabled,
    solve_problem,
)
from forge_mind.invention.research import ResearchProposal
from forge_mind.kit import Kit, build_kit
from forge_mind.llm.cost import LLMUsage


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


def _tree() -> ProposedTree:
    n = ProposedNode
    return ProposedTree(
        nodes=[
            n(node_id="r", parent_id=None, statement="mount a 2.5in SSD"),
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
        ]
    )


class _Client:
    def __init__(self, tree: ProposedTree) -> None:
        self.tree = tree

    def generate_structured(
        self, *, output_model: type, model: str, **_: Any
    ) -> tuple[Any, LLMUsage]:
        usage = LLMUsage(model=model, input_tokens=1, output_tokens=1)
        if output_model is ProposedTree:
            return self.tree, usage
        if output_model is ResearchProposal:
            return ResearchProposal(), usage
        raise AssertionError(output_model)


class _Produce:
    """A spy engine adapter: records calls, writes a small box STL so env checks can run."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.calls: list[tuple[str, str]] = []

    def __call__(self, prompt: str, extra_context: str) -> GeneratedPart:
        self.calls.append((prompt, extra_context))
        stl = self.tmp / "part.stl"
        trimesh.creation.box(extents=(90.0, 60.0, 8.0)).export(str(stl))
        return GeneratedPart(
            part_name="ssd bracket",
            disposition="shipped",
            kit=_kit(),
            artifacts={"stl": str(stl)},
        )


def test_flag_default_off(monkeypatch: Any) -> None:
    monkeypatch.delenv("FORGE_PROBLEM_MODE", raising=False)
    assert problem_mode_enabled() is False
    monkeypatch.setenv("FORGE_PROBLEM_MODE", "1")
    assert problem_mode_enabled() is True


def test_questions_block_spend(tmp_path: Path) -> None:
    produce = _Produce(tmp_path)
    outcome, _ = solve_problem(
        "mount an SSD",
        client=_Client(_tree()),
        model="m",
        produce=produce,
        confirmed=True,  # even confirmed: open questions must still block (L5 before L3)
        registry_keys={"drive_form": "2.5in_drive"},
    )
    # payload_thickness has no registry/research route -> an open question remains
    assert outcome.kind == "needs_input"
    assert produce.calls == []  # NO SPEND


def test_unconfirmed_ready_spec_cannot_spend(tmp_path: Path) -> None:
    produce = _Produce(tmp_path)
    outcome, _ = solve_problem(
        "mount an SSD",
        client=_Client(_tree()),
        model="m",
        produce=produce,
        confirmed=False,
        # the base gather (S7) also asks the un-sourced core pitch — the canned user states both
        user_facts=answers_to_facts(
            [
                AnsweredFact(name="payload_thickness", value=9.5),
                AnsweredFact(name="mount_hole_pitch", value=61.7),
            ]
        ),
        registry_keys={"drive_form": "2.5in_drive"},
    )
    assert outcome.kind == "awaiting_confirmation"
    assert produce.calls == []  # L3: the card must be confirmed before any build
    assert outcome.compiled.card  # the card exists to confirm


def test_confirmed_ready_spec_builds_and_runs_env_checks(tmp_path: Path) -> None:
    produce = _Produce(tmp_path)
    outcome, usages = solve_problem(
        "mount an SSD",
        client=_Client(_tree()),
        model="m",
        produce=produce,
        confirmed=True,
        user_facts=answers_to_facts(
            [
                AnsweredFact(name="payload_thickness", value=9.5),
                AnsweredFact(name="mount_hole_pitch", value=61.7),
            ]
        ),
        registry_keys={"drive_form": "2.5in_drive"},
    )
    assert outcome.kind == "built"
    assert len(produce.calls) == 1
    prompt, ctx = produce.calls[0]
    assert "payload_width: 69.85 mm" in prompt  # the compiled constraints reached the engine
    assert "(source: standard)" in ctx  # the ADR-0040 block rode extra_context
    assert outcome.part is not None and outcome.part.disposition == "shipped"
    assert usages  # the decompose pass was accounted


def test_answers_become_stated_facts() -> None:
    (fact,) = answers_to_facts([AnsweredFact(name="payload_thickness", value=9.5)])
    assert fact.source == "user" and fact.as_mm_or_g == (9.5, "mm")
