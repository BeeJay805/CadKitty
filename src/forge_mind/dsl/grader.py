"""Grade a generated PartProgram: does it compile and pass the validator?

This is the real test of the DSL stage. Unlike the blueprint grader (which checks
invariants on a spec), here we run the actual deterministic core: compile to
geometry, validate against the part's declared constraints + interfaces, and
require the benchmark's required checks to pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.compiler.compile import CompileError, compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


@dataclass
class DSLGrade:
    passed: bool
    compiled: bool
    failures: list[str] = field(default_factory=list)
    report: ValidationReport | None = None
    stl_path: Path | None = None


def grade_program(
    program: PartProgram,
    output_dir: Path,
    required_checks: list[str],
) -> DSLGrade:
    """Compile + validate ``program``; pass iff it compiles and every required
    check passes.
    """
    try:
        model = compile_program(program, output_dir)
    except CompileError as exc:
        return DSLGrade(passed=False, compiled=False, failures=[f"compile failed: {exc}"])

    report = validate(model, program)
    passing = {c.id for c in report.checks if c.status.value == "pass"}
    missing = [rc for rc in required_checks if rc not in passing]

    failures: list[str] = []
    if missing:
        failures.append(f"required checks not passing: {missing}")
    if not report.passed:
        failures.extend(f"failed: {fid}" for fid in report.failed_check_ids())

    return DSLGrade(
        passed=not failures,
        compiled=True,
        failures=failures,
        report=report,
        stl_path=model.stl_path,
    )


__all__ = ["DSLGrade", "grade_program"]
