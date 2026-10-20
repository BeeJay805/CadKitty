"""The standard-parts lane (ADR-0034): canonical standard parts route to the deterministic
LayoutSpec pipeline; everything else -- and every lane failure -- falls through to the agentic
engine untouched (strictly-no-worse semantics). All money-free (run_pipeline is stubbed)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from forge_mind.api import core
from forge_mind.layout.standard_match import standard_match

# ---------------------------------------------------------------- matcher ---


@pytest.mark.parametrize(
    "prompt",
    [
        "a NEMA-17 motor mount plate 5 mm thick with the 31 mm bolt circle and a bore",
        "an adapter plate from a NEMA-17 face to four M5 holes on a 50 mm square",
        "a 60 mm fan shroud that bolts to a 2020 aluminum extrusion with M5 hardware",
        "a raspberry pi 4 case with vent slots",
    ],
)
def test_standard_prompts_match(prompt: str) -> None:
    assert standard_match(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "a right-angle bracket 40 mm x 40 mm with two M4 clearance holes per leg",  # no ecosystem
        "a 25 mm knob with a 6 mm D-shaft bore and a knurled rim",
        "a NEMA-17 stepper motor",  # the MOTOR itself, not a part for it
        "a flower vase with a wavy rim",
        "",
    ],
)
def test_non_standard_prompts_do_not_match(prompt: str) -> None:
    assert standard_match(prompt) is False


# ------------------------------------------------------- adapter fall-through ---


@dataclass
class _FakeUsage:
    cost_usd: float = 0.05
    total_tokens: int = 100


@dataclass
class _FakeModel:
    stl_path: Path
    step_path: Path | None


@dataclass
class _FakePipelineResult:
    prompt: str = "x"
    refusal: str | None = None
    failed_stage: str | None = None
    error: str | None = None
    model: Any = None
    report: Any = None
    program: Any = None
    kit: Any = None
    spec: Any = None  # the parametric model (ADR-0037); part of the PipelineResult contract
    gen_path: str = "layoutspec"
    fallback_reason: str | None = None
    usages: list[Any] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.report is not None and getattr(self.report, "passed", False)

    @property
    def total_cost_usd(self) -> float:
        return sum(u.cost_usd for u in self.usages)


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, result: _FakePipelineResult) -> None:
    import forge_mind.pipeline as pipeline_mod

    monkeypatch.setattr(
        pipeline_mod, "run_pipeline", lambda prompt, out, **kw: result, raising=True
    )


def test_refusal_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, _FakePipelineResult(refusal="out of domain"))
    res, lane = core._generate_standard("p", tmp_path, lambda k, s: None)
    assert res is None
    assert lane.attempted is True
    assert lane.outcome == "refused"
    assert lane.refusal == "out of domain"


def test_stage_failure_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_pipeline(monkeypatch, _FakePipelineResult(failed_stage="compile"))
    res, lane = core._generate_standard("p", tmp_path, lambda k, s: None)
    assert res is None
    assert lane.outcome == "failed_stage"
    assert lane.error is not None and "compile" in lane.error


def test_failed_validation_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Report:
        passed = False

    _patch_pipeline(monkeypatch, _FakePipelineResult(report=_Report(), failed_stage="validate"))
    res, lane = core._generate_standard("p", tmp_path, lambda k, s: None)
    assert res is None
    assert lane.outcome == "validation_failed"


def test_direct_dsl_fallback_is_blocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR-0035 decision (a): a PASSING part from the direct-DSL fallback must NOT ship under the
    'standard:' label -- the lane returns None (agentic fall-through) and classifies the miss."""

    class _Report:
        passed = True

    _patch_pipeline(
        monkeypatch,
        _FakePipelineResult(
            report=_Report(),
            gen_path="direct-dsl",
            fallback_reason="LayoutError: infeasible spec",
        ),
    )
    res, lane = core._generate_standard("p", tmp_path, lambda k, s: None)
    assert res is None
    assert lane.outcome == "fallback_blocked"
    assert lane.gen_path == "direct-dsl"
    assert lane.fallback_reason == "LayoutError: infeasible spec"


def test_exception_is_classified_not_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bare 'except Exception: return None' swallow is gone: a surprise is captured,
    classified, and STILL falls through (strictly-no-worse)."""
    import forge_mind.pipeline as pipeline_mod

    def _boom(prompt: str, out: Path, **kw: Any) -> Any:
        raise RuntimeError("surprise")

    monkeypatch.setattr(pipeline_mod, "run_pipeline", _boom, raising=True)
    res, lane = core._generate_standard("p", tmp_path, lambda k, s: None)
    assert res is None
    assert lane.outcome == "exception"
    assert lane.error is not None and "RuntimeError" in lane.error


def test_outcome_to_record_drops_unset_fields() -> None:
    lane = core.StandardLaneOutcome(attempted=True, outcome="refused", refusal="why")
    row = lane.to_record()
    assert row == {"attempted": True, "outcome": "refused", "refusal": "why"}


# ------------------------------------------------------------ success path ---


def _make_real_report_and_program(tmp_path: Path) -> tuple[Any, Any, Path, Path]:
    """Compile a REAL tiny part (no LLM) so the projection runs against production shapes."""
    from forge_mind.compiler.compile import compile_program
    from forge_mind.dsl.primitives import Box
    from forge_mind.dsl.program import Metadata, PartProgram
    from forge_mind.types import Vec3 as FVec3
    from forge_mind.validator.run import validate

    program = PartProgram(
        metadata=Metadata(name="std_plate"),
        operations=[Box(op_id="p", width=40.0, depth=20.0, height=4.0, position=FVec3(z=2.0))],
    )
    model = compile_program(program, tmp_path / "pipe")
    report = validate(model, program)
    return report, program, Path(model.stl_path), Path(model.step_path)


def test_success_ships_files_validation_and_kit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_mind.kit import build_kit

    report, program, stl, step = _make_real_report_and_program(tmp_path)
    kit = build_kit(program, material="PLA")  # REAL kit -> proves the schema bridge at test time
    result = _FakePipelineResult(
        model=_FakeModel(stl_path=stl, step_path=step),
        report=report,
        program=program,
        kit=kit,
        usages=[_FakeUsage(cost_usd=0.11)],
    )
    _patch_pipeline(monkeypatch, result)
    phases: list[tuple[str, str]] = []

    out, lane = core._generate_standard(
        "a nema plate", tmp_path, lambda k, s: phases.append((k, s))
    )

    assert out is not None and out.status == "ok" and out.ok is not None
    assert (tmp_path / "part.stl").exists()
    assert (tmp_path / "part.glb").exists()
    assert (tmp_path / "part.step").exists()
    assert out.ok.gen_path == "standard:layoutspec"
    assert out.ok.cost_usd == 0.11
    assert out.ok.validation.passed is True
    assert out.ok.kit.part_name == "std_plate"
    assert out.ok.fabrication_mode == "print_3d"
    assert ("export", "active") in phases
    assert lane.outcome == "shipped"
    assert lane.gen_path == "standard:layoutspec"
    assert out.lane is lane


# ---------------------------------------------------------- generate_part wiring ---


def test_generate_part_short_circuits_on_standard_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = core.StandardLaneOutcome(attempted=True, outcome="shipped")
    sentinel = core.GenerateResult(
        status="error", error=core.GenerateError(stage="x", message="s"), lane=lane
    )
    monkeypatch.setattr(core, "_generate_standard", lambda p, d, ph: (sentinel, lane))

    def _no_agentic(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("agentic engine must not run when the standard lane ships")

    monkeypatch.setattr(core, "produce_part", _no_agentic)
    out = core.generate_part("a NEMA-17 motor mount plate", tmp_path)
    assert out is sentinel


def test_generate_part_falls_through_when_lane_declines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lane = core.StandardLaneOutcome(attempted=True, outcome="exception", error="boom")
    monkeypatch.setattr(core, "_generate_standard", lambda p, d, ph: (None, lane))
    called: dict[str, bool] = {"agentic": False}

    def _agentic_stub(*_a: Any, **_k: Any) -> Any:
        called["agentic"] = True
        raise RuntimeError("stop here -- agentic reached")

    monkeypatch.setattr(core, "produce_part", _agentic_stub)
    out = core.generate_part("a NEMA-17 motor mount plate", tmp_path)
    assert called["agentic"] is True  # the lane declined; the agentic engine got its shot
    assert out.status == "error"  # produce_part's raise is captured, never leaked
    assert out.lane is lane  # the lane attempt rides the agentic result (ADR-0035)


def test_generate_part_no_lane_for_non_standard_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _agentic_stub(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("agentic reached")

    monkeypatch.setattr(core, "produce_part", _agentic_stub)
    out = core.generate_part("a flower vase with a wavy rim", tmp_path)
    assert out.lane is None  # never matched -> no lane attempt recorded
