"""Runners — map a prompt to a TrialOutput.

A Runner is one strategy for producing a part. Phase 3 ships HandwrittenRunner
(the LLM bypass). The harness core never knows which runner it's driving.
"""

from __future__ import annotations

import abc
import json
import shutil
import time
from pathlib import Path

from forge_mind.bench.types import StageTiming, TrialOutput
from forge_mind.compiler.compile import CompileError, compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.run import validate


class Runner(abc.ABC):
    """Strategy for producing + validating a part for one trial."""

    #: short label used in reports and result dirs.
    name: str

    @abc.abstractmethod
    def run_trial(self, prompt: str, trial_dir: Path, trial_index: int) -> TrialOutput:
        raise NotImplementedError


class HandwrittenRunner(Runner):
    """LLM bypass: ignores the prompt, compiles + validates a fixed reference DSL.

    Establishes the measurement floor — the harness, timing, artifacts, and
    reporting all exercised end-to-end before any model is involved.
    """

    name = "handwritten"

    def __init__(self, reference_dsl: Path) -> None:
        self._reference_dsl = reference_dsl

    def run_trial(self, prompt: str, trial_dir: Path, trial_index: int) -> TrialOutput:
        trial_dir.mkdir(parents=True, exist_ok=True)
        timings: list[StageTiming] = []
        artifacts: dict[str, Path] = {}

        # Persist the DSL used (the reference), so every trial is self-describing.
        program_path = trial_dir / "program.partdsl.json"
        try:
            shutil.copyfile(self._reference_dsl, program_path)
            program = PartProgram.model_validate(json.loads(program_path.read_text()))
        except (OSError, ValueError) as exc:
            return TrialOutput(
                report=None,
                first_failing_stage="load_dsl",
                error=f"failed to load reference DSL: {exc}",
            )
        artifacts["program"] = program_path

        # Stage: compile.
        t0 = time.perf_counter()
        try:
            model = compile_program(program, trial_dir)
        except CompileError as exc:
            timings.append(StageTiming("compile", (time.perf_counter() - t0) * 1000))
            return TrialOutput(
                report=None,
                stage_timings=timings,
                artifacts=artifacts,
                first_failing_stage="compile",
                error=f"compile failed: {exc}",
            )
        timings.append(StageTiming("compile", (time.perf_counter() - t0) * 1000))
        artifacts["stl"] = model.stl_path
        artifacts["step"] = model.step_path

        # Stage: validate.
        t1 = time.perf_counter()
        report = validate(model, program)
        timings.append(StageTiming("validate", (time.perf_counter() - t1) * 1000))

        report_path = trial_dir / "report.json"
        report_path.write_text(report.model_dump_json(indent=2))
        artifacts["report"] = report_path

        # A handwritten trial has no LLM stages, so "first failing stage" is
        # simply validate if the geometry failed checks, else None.
        first_failing = None if report.passed else "validate"

        return TrialOutput(
            report=report,
            stage_timings=timings,
            artifacts=artifacts,
            iterations=1,
            first_failing_stage=first_failing,
        )


__all__ = ["HandwrittenRunner", "Runner"]
