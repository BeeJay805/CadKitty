"""The LayoutSpec generation path, end to end (ADR-0004).

    LayoutSpec -> LayoutInterpreter -> PartProgram -> Compiler -> Validator

This is the NEW path in the A/B experiment. The OLD path stays exactly as is:

    Blueprint -> DSLGenerator -> PartProgram -> Compiler -> Validator

Both terminate in the same compiler + validator, so a future A/B harness compares
two ValidationReports. For this spike the LayoutSpec is hand-authored; the LLM
``LayoutSpecGenerator`` (Blueprint -> LayoutSpec) drops into the front of this
path later without changing anything downstream.

Failures are captured, not raised, so a caller always gets an inspectable result
(mirrors ``forge_mind.pipeline.run_pipeline``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from forge_mind.compiler.compile import CADModel, compile_program
from forge_mind.dsl.program import PartProgram
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import LayoutSpec, interpret_layout
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


@dataclass
class LayoutResult:
    spec: LayoutSpec
    program: PartProgram | None = None
    model: CADModel | None = None
    report: ValidationReport | None = None
    failed_stage: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.report is not None and self.report.passed


def run_layout_pipeline(spec: LayoutSpec, output_dir: Path) -> LayoutResult:
    """Interpret -> compile -> validate. Never raises on stage failure; records it."""
    result = LayoutResult(spec=spec)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        program = interpret_layout(spec)
        result.program = program
        (output_dir / "part.partdsl.json").write_text(
            program.model_dump_json(indent=2), encoding="utf-8"
        )
    except LayoutError as exc:
        result.failed_stage = "interpret"
        result.error = str(exc)
        return result

    try:
        model = compile_program(program, output_dir)
        result.model = model
    except Exception as exc:
        result.failed_stage = "compile"
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    try:
        result.report = validate(model, program, material=getattr(spec, "material", "PLA"))
    except Exception as exc:
        result.failed_stage = "validate"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    if not result.report.passed:
        result.failed_stage = "validate"
    return result


__all__ = ["LayoutResult", "run_layout_pipeline"]
