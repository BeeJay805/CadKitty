"""LayoutSpec schema validation (pure, no compile, no LLM)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.layout.spec import LBracketMotorMountSpec, Pattern

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench001_l_bracket.layoutspec.json"


def _spec_dict() -> dict:
    return json.loads(FIXTURE.read_text())


def test_bench001_fixture_parses() -> None:
    spec = LBracketMotorMountSpec.model_validate(_spec_dict())
    assert spec.pattern is Pattern.L_BRACKET_MOTOR_MOUNT
    assert spec.base is not None and spec.base.width_mm == 43.0
    assert spec.upright is not None and spec.upright.height_mm == 30.0


def test_defaults_applied() -> None:
    # gusset removed in ADR-0009 Task 2; the interpreter owns the decision.
    spec = LBracketMotorMountSpec.model_validate(
        {
            "name": "minimal",
            "envelope_mm": {"x": 60, "y": 60, "z": 40},
            "material": "PLA",
            "min_wall_mm": 4.0,
            "max_overhang_deg": 45.0,
            "base": {"width_mm": 43, "depth_mm": 50},
            "upright": {"width_mm": 43, "height_mm": 30},
        }
    )
    assert spec.nema17.name == "motor_face"
    assert spec.extrusion_2020.m5_spacing_mm == 20.0


def test_extra_fields_forbidden() -> None:
    d = _spec_dict()
    d["unexpected"] = 1
    with pytest.raises(ValidationError):
        LBracketMotorMountSpec.model_validate(d)


def test_negative_dimension_rejected() -> None:
    d = _spec_dict()
    d["base"]["width_mm"] = -1.0
    with pytest.raises(ValidationError):
        LBracketMotorMountSpec.model_validate(d)


def test_base_and_upright_now_optional() -> None:
    # ADR-0008-shaped fix (351a50b -> next commit): the LLM-facing fields
    # base + upright are optional. The interpreter derives them when omitted.
    # Existing fixtures with explicit values still validate (see
    # test_bench001_fixture_parses), but the minimal spec omits them.
    spec = LBracketMotorMountSpec.model_validate(
        {
            "name": "minimal-derived",
            "envelope_mm": {"x": 60, "y": 60, "z": 40},
            "material": "PLA",
            "min_wall_mm": 4.0,
            "max_overhang_deg": 45.0,
        }
    )
    assert spec.base is None
    assert spec.upright is None
    # Other defaults still apply.
    assert spec.nema17.name == "motor_face"
    assert spec.extrusion_2020.m5_spacing_mm == 20.0


def test_gusset_field_no_longer_accepted() -> None:
    # ADR-0009 Task 2: gusset was removed from the LLM-facing schema. An old
    # fixture that still tries to set it must be rejected by extra="forbid".
    with pytest.raises(ValidationError):
        LBracketMotorMountSpec.model_validate(
            {
                "name": "legacy-gusset-required",
                "envelope_mm": {"x": 60, "y": 60, "z": 40},
                "material": "PLA",
                "min_wall_mm": 4.0,
                "max_overhang_deg": 45.0,
                "gusset": "required",
            }
        )
