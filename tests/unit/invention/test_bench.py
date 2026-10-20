"""Invention benchmark runner (ADR-0048, P7): the offline harness runs deterministically with mocked
outputs (no spend), failure is attributed to the first failing stage, baselines are preserved, runs
resume, and the LIVE path fails CLEARLY when no model is available. Plus the CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.cli.main import app
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import (
    ConceptBrief,
    ConceptPlan,
    FeasibilityVerdict,
    GeneratedPart,
    Mechanism,
    build_invention_package,
)
from forge_mind.invention.bench import (
    BenchRun,
    LiveModelUnavailableError,
    _failure_stage,
    assert_live_model_available,
    default_problems,
    freeze_baseline,
    load_run,
    offline_bench,
    run_bench,
    save_run,
    score_bench,
)
from forge_mind.invention.package import InventionPackage
from forge_mind.kit import build_kit
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _plan(*, distinct: bool = True, n: int = 2) -> ConceptPlan:
    concepts = [
        ConceptBrief(
            concept_id="c1",
            name="clip",
            primary_mechanism=Mechanism.SNAP_FIT,
            approach="flex over edge",
            obvious_baseline="bracket",
            difference_from_obvious="no screws",
        )
    ]
    if n >= 2:
        concepts.append(
            ConceptBrief(
                concept_id="c2",
                name="saddle",
                primary_mechanism=Mechanism.FRICTION_FIT if distinct else Mechanism.SNAP_FIT,
                approach="friction",
                obvious_baseline="adhesive",
                difference_from_obvious="removable",
            )
        )
    return ConceptPlan(
        problem_restated="route conductors",
        feasibility_verdict=FeasibilityVerdict.LIKELY_FEASIBLE,
        feasibility_note="routine",
        concepts=concepts,
        recommended_concept_id="c1",
        recommendation_reason="simplest",
    )


def _kit() -> Any:
    program = PartProgram(
        metadata=Metadata(name="p", units="mm", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


def _package(plan: ConceptPlan, *, problem: str = "route two cables") -> InventionPackage:
    return build_invention_package(
        part_name="p",
        disposition="shipped",
        kit=_kit(),
        confidence=standard_lane_label(1),
        problem_statement=problem,
        concept=plan.recommended(),
    )


def _clear_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FORGE_FORBID_API_SPEND", raising=False)


def test_score_bench_full_marks_on_a_clean_lifecycle() -> None:
    plan = _plan(distinct=True)
    scores = score_bench(_package(plan), plan, problem="route two cables")
    assert scores.rubric_passed is True
    assert scores.concept_diversity == 1.0
    assert scores.package_completeness == 1.0
    assert scores.feasibility_honesty == 1.0


def test_no_divergence_lowers_diversity_and_fails_the_rubric() -> None:
    plan = _plan(distinct=False)  # two concepts, SAME mechanism
    scores = score_bench(_package(plan), plan, problem="route two cables")
    assert scores.concept_diversity == 0.5
    assert scores.rubric_passed is False


def test_score_bench_fails_when_problem_not_carried() -> None:
    # Regression (audit M9): the rubric's problem_carried check compares the package's statement to
    # the BENCHMARK problem. Passing the package's OWN statement as the reference made it always
    # equal (vacuous); with the real reference, a lifecycle that dropped the problem fails the rubric.
    plan = _plan(distinct=True)
    pkg = _package(plan, problem="route two cables")
    assert score_bench(pkg, plan, problem="route two cables").rubric_passed is True
    assert score_bench(pkg, plan, problem="an unrelated brief").rubric_passed is False


def test_failure_is_attributed_to_the_first_failing_stage() -> None:
    one = _plan(n=1)  # a single concept -> no divergence -> the concept stage fails
    assert _failure_stage(_package(one), one, problem="route two cables") == "concept"
    assert (
        _failure_stage(
            _package(_plan(distinct=True)), _plan(distinct=True), problem="route two cables"
        )
        is None
    )


def test_offline_bench_runs_the_fixture_deterministically() -> None:
    run = offline_bench()
    assert run.mode == "offline"
    assert run.entries  # one entry per frozen problem
    assert run.pass_rate == 1.0  # the deterministic stubs build a complete, diverse package
    assert run.total_cost_usd == 0.0
    # replay: a second offline run is byte-identical (mocked outputs, no model)
    assert offline_bench().model_dump() == run.model_dump()


def test_baseline_is_preserved(tmp_path: Path) -> None:
    run = offline_bench()
    baseline = tmp_path / "baseline.json"
    freeze_baseline(run, baseline)
    assert baseline.exists()
    with pytest.raises(FileExistsError):
        freeze_baseline(run, baseline)  # never clobber the reference numbers
    assert load_run(baseline).model_dump() == run.model_dump()


def test_run_resumes_and_skips_done_problems() -> None:
    full = offline_bench()
    # resuming from the full run produces no new entries (every problem already done)
    resumed = run_bench(
        default_problems(),
        client=_StubFrontDoor(),
        produce=_stub_produce,
        mode="offline",
        resume_from=full,
    )
    assert len(resumed.entries) == len(full.entries)


def test_live_bench_fails_clearly_without_a_model(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)
    with pytest.raises(LiveModelUnavailableError):
        assert_live_model_available()


# --- inline stubs for the resume test ---------------------------------------------------------


class _StubFrontDoor:
    def generate_structured(self, **_: Any) -> tuple[Any, LLMUsage]:
        return _plan(), LLMUsage(model="stub", input_tokens=0, output_tokens=0)

    def generate_structured_with_image(self, **_: Any) -> tuple[Any, LLMUsage]:
        raise AssertionError("no image")


def _stub_produce(prompt: str) -> GeneratedPart:
    return GeneratedPart(
        part_name="p", disposition="shipped", kit=_kit(), confidence=standard_lane_label(1)
    )


# --- the CLI ----------------------------------------------------------------------------------


def test_cli_offline_bench_runs() -> None:
    result = CliRunner().invoke(app, ["invention-bench"])
    assert result.exit_code == 0
    assert "pass_rate" in result.stdout


def test_cli_offline_bench_writes_results(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    result = CliRunner().invoke(app, ["invention-bench", "--out", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert load_run(out).mode == "offline"


def test_cli_live_bench_blocked_exits_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_model_env(monkeypatch)
    result = CliRunner().invoke(app, ["invention-bench", "--live"])
    assert result.exit_code == 1
    assert "blocked" in result.stdout.lower()


def test_bench_run_round_trips(tmp_path: Path) -> None:
    run = offline_bench()
    path = tmp_path / "r.json"
    save_run(run, path)
    assert isinstance(load_run(path), BenchRun)
    assert load_run(path).pass_rate == run.pass_rate
