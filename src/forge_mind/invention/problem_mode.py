"""Problem-mode orchestration (ADR-0087 S6a) — compile, confirm, build, verify; never spend early.

The pure spine of the Problem Solver, wired from the shipped slices:

    problem -> compile_requirements (S4: tree + playbooks + S1 ledger + S2 registry + S3 research)
            -> questions?            ->  kind="needs_input"   (NO build call — L5)
            -> ready, not confirmed  ->  kind="awaiting_confirmation" (NO build call — L3)
            -> ready + confirmed     ->  produce (the injected engine adapter)
            -> run_env_checks (S5, advisory) -> kind="built" with the assumption ledger

The CONFIRM GATE is enforced HERE, in code, not in UI politeness: an unconfirmed spec structurally
cannot spend, because ``produce`` is simply never invoked on that path. Answers to open questions
return as ``user_facts`` (STATED tier) and the caller re-solves — the loop the founder described,
closed. ``FORGE_PROBLEM_MODE`` (default OFF) is the dark-launch flag; server wiring (S6b) checks it
and the S7 bench decides whether it ever turns on.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

import trimesh
from pydantic import BaseModel, ConfigDict

from forge_mind.invention.compile_requirements import (
    CompiledRequirements,
    compile_requirements,
)
from forge_mind.invention.env_checks import EnvCheckResult, run_env_checks
from forge_mind.invention.lifecycle import GeneratedPart
from forge_mind.invention.requirements import Fact
from forge_mind.invention.research import ResearchProviderLike
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike


class ProduceLike(Protocol):
    """produce(prompt, extra_context, material=...) -> GeneratedPart. The REAL adapter wraps
    produce_part (api/core's GeneratedPart composition); tests inject a stub. artifacts["stl"] /
    ["step"] feed the env checks. ``material`` is the S7 base intake fact — the intake surface
    always passes the user's pick; this spine's own two-arg calls keep the engine default."""

    def __call__(self, prompt: str, extra_context: str, material: str = "PLA") -> GeneratedPart: ...


def problem_mode_enabled() -> bool:
    """The ADR-0087 dark-launch flag. OFF by default; S7's bench gates it ON."""
    return os.environ.get("FORGE_PROBLEM_MODE", "").strip() == "1"


class ProblemOutcome(BaseModel):
    """One solve attempt. ``kind`` says exactly how far it went and why it stopped."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    kind: Literal["needs_input", "awaiting_confirmation", "built"]
    compiled: CompiledRequirements
    part: GeneratedPart | None = None
    env_results: tuple[EnvCheckResult, ...] = ()
    # the full honesty surface for the report: compiler assumptions + advisory env flags
    assumption_ledger: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _load_geometry(part: GeneratedPart) -> tuple[trimesh.Trimesh | None, object | None, list[str]]:
    """Best-effort artifact loading for the env checks; failures become notes, never crashes
    (the checks themselves flag fail-closed on missing geometry)."""
    notes: list[str] = []
    mesh: trimesh.Trimesh | None = None
    brep: object | None = None
    stl = part.artifacts.get("stl")
    if stl:
        try:
            loaded = trimesh.load(stl, force="mesh")
            if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces) > 0:
                mesh = loaded
            else:
                notes.append("built STL unreadable as a mesh — env checks will flag")
        except Exception as exc:
            notes.append(f"built STL failed to load ({type(exc).__name__}) — env checks will flag")
    step = part.artifacts.get("step")
    if step:
        try:
            from build123d import import_step

            brep = import_step(step)
        except Exception as exc:
            notes.append(f"built STEP failed to load ({type(exc).__name__}) — pattern check flags")
    return mesh, brep, notes


def solve_problem(
    problem: str,
    *,
    client: LLMClientLike,
    model: str,
    produce: ProduceLike,
    confirmed: bool = False,
    user_facts: Sequence[Fact] = (),
    entity: str = "",
    registry_keys: Mapping[str, str] | None = None,
    provider: ResearchProviderLike | None = None,
) -> tuple[ProblemOutcome, list[LLMUsage]]:
    """One pass of the problem loop. Deterministic control flow; the confirm gate is code."""
    compiled, usages = compile_requirements(
        problem,
        client=client,
        model=model,
        user_facts=user_facts,
        entity=entity,
        registry_keys=registry_keys,
        provider=provider,
    )
    if compiled.status == "needs_input":
        return (
            ProblemOutcome(
                kind="needs_input",
                compiled=compiled,
                assumption_ledger=compiled.assumptions,
                notes=compiled.notes,
            ),
            usages,
        )
    if not confirmed:
        # L3: the user sees the card (compiled.card) and confirms BEFORE any build spend.
        return (
            ProblemOutcome(
                kind="awaiting_confirmation",
                compiled=compiled,
                assumption_ledger=compiled.assumptions,
                notes=compiled.notes,
            ),
            usages,
        )
    part = produce(compiled.prompt, compiled.extra_context)
    mesh, brep, load_notes = _load_geometry(part)
    env_results = tuple(run_env_checks(compiled.checks, mesh, brep))
    ledger = [
        *compiled.assumptions,
        *(f"env check {r.check_id}: {r.explanation}" for r in env_results if r.status == "flag"),
    ]
    return (
        ProblemOutcome(
            kind="built",
            compiled=compiled,
            part=part,
            env_results=env_results,
            assumption_ledger=tuple(ledger),
            notes=(*compiled.notes, *load_notes),
        ),
        usages,
    )


class AnsweredFact(BaseModel):
    """A user's answer to an open question, as the API collects it: name + value + unit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    unit: str = "mm"


def answers_to_facts(answers: Sequence[AnsweredFact]) -> tuple[Fact, ...]:
    """Answers become STATED facts — the loop's return edge (the user said it, tier earned)."""
    return tuple(Fact(name=a.name, value=a.value, unit=a.unit, source="user") for a in answers)


__all__ = [
    "AnsweredFact",
    "ProblemOutcome",
    "ProduceLike",
    "answers_to_facts",
    "problem_mode_enabled",
    "solve_problem",
]
