"""Fabrication mode gates the printability tier (ADR-0031): GENERAL (the not-3D-printed default)
skips the FDM overhang check entirely; PRINT_3D runs it. Also locks the FabricationMode logic that
was previously dead code (`enforces_printability` was defined but never called)."""

from __future__ import annotations

from typing import Any

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.fabrication import GENERAL_MIN_WALL_MM, FabricationMode
from forge_mind.types import Vec3
from forge_mind.validator.report import ValidationReport
from forge_mind.validator.run import validate


def _program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="plate"),
        operations=[Box(op_id="plate", width=40.0, depth=20.0, height=4.0, position=Vec3(z=2.0))],
    )


@pytest.fixture(scope="module")
def model(tmp_path_factory: Any) -> Any:
    return compile_program(_program(), tmp_path_factory.mktemp("fab"))


def _printability_ids(report: ValidationReport) -> list[str]:
    return [c.id for c in report.checks if c.id.startswith("printability.")]


def test_print_3d_runs_the_printability_tier(model: Any) -> None:
    report = validate(model, _program(), fabrication_mode=FabricationMode.PRINT_3D)
    assert _printability_ids(report)  # the overhang check is present


def test_general_skips_the_printability_tier(model: Any) -> None:
    report = validate(model, _program(), fabrication_mode=FabricationMode.GENERAL)
    assert _printability_ids(report) == []  # FDM overhang is not checked at all


def test_default_mode_keeps_printability_on(model: Any) -> None:
    # The shared validator default stays PRINT_3D so the DSL/template path is byte-identical.
    report = validate(model, _program())
    assert _printability_ids(report)


# --- the FabricationMode logic this ADR finally wires in --------------------


def test_enforces_printability() -> None:
    assert FabricationMode.PRINT_3D.enforces_printability is True
    assert FabricationMode.GENERAL.enforces_printability is False


def test_min_wall_override() -> None:
    assert FabricationMode.PRINT_3D.min_wall_override(3.0) is None  # PRINT_3D: unchanged
    assert FabricationMode.GENERAL.min_wall_override(3.0) == GENERAL_MIN_WALL_MM  # relaxed floor
    assert FabricationMode.GENERAL.min_wall_override(0.5) == 0.5  # never STRICTER than declared
    assert (
        FabricationMode.GENERAL.min_wall_override(None) is None
    )  # adds no check where none exists
