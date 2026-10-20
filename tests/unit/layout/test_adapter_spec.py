"""AdapterPlateSpec schema (ADR-0008) — pydantic validation, defaults, JSON
round-trip via the discriminated LayoutSpec union.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from forge_mind.layout.adapter_spec import AdapterPlateSpec
from forge_mind.layout.layoutspec import LayoutSpec
from forge_mind.layout.spec import Pattern

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench003_adapter_plate.layoutspec.json"

_ADAPTER: TypeAdapter[LayoutSpec] = TypeAdapter(LayoutSpec)


def _fixture_dict() -> dict[str, object]:
    return dict(json.loads(FIXTURE.read_text()))


def test_pattern_pins_to_adapter_plate() -> None:
    spec = AdapterPlateSpec.model_validate(_fixture_dict())
    assert spec.pattern is Pattern.ADAPTER_PLATE
    assert spec.pattern.value == "adapter_plate"


def test_union_discriminates_to_adapter_plate() -> None:
    parsed = _ADAPTER.validate_python(_fixture_dict())
    assert isinstance(parsed, AdapterPlateSpec)


def test_minimal_fields_with_defaults() -> None:
    # plate_width_mm and plate_depth_mm are now optional -- the interpreter
    # derives them from the offsets + min_wall_mm when omitted (ADR-0008
    # perturbation fix). Minimum LLM-facing fields are envelope + material +
    # min_wall + the two interface offsets.
    spec = AdapterPlateSpec.model_validate(
        {
            "pattern": "adapter_plate",
            "name": "minimal",
            "envelope_mm": {"x": 70.0, "y": 50.0, "z": 8.0},
            "material": "PLA",
            "min_wall_mm": 4.0,
            "motor_center_x_mm": 11.0,
            "extrusion_center_x_mm": -22.0,
        }
    )
    # Defaults: max_overhang_deg = 45, plate dims left null (interpreter
    # derives). print_orientation, nema17, extrusion_2020 are no longer spec
    # fields -- the interpreter constants them. ADR-0009 Task 1.
    assert spec.max_overhang_deg == 45.0
    assert spec.plate_width_mm is None
    assert spec.plate_depth_mm is None


def test_explicit_plate_dims_still_accepted() -> None:
    # Backward compat: existing fixtures that set explicit dims must still
    # validate as-is, and the interpreter must respect them.
    payload = _fixture_dict()
    spec = AdapterPlateSpec.model_validate(payload)
    assert spec.plate_width_mm == 65.0
    assert spec.plate_depth_mm == 45.0


def test_unknown_fields_rejected() -> None:
    payload = _fixture_dict()
    payload["nope"] = 1
    with pytest.raises(ValidationError):
        AdapterPlateSpec.model_validate(payload)


def test_non_positive_dims_rejected() -> None:
    payload = _fixture_dict()
    payload["plate_width_mm"] = 0.0
    with pytest.raises(ValidationError):
        AdapterPlateSpec.model_validate(payload)


def test_json_round_trip() -> None:
    parsed = _ADAPTER.validate_python(_fixture_dict())
    payload = json.loads(parsed.model_dump_json())
    reparsed = _ADAPTER.validate_python(payload)
    assert isinstance(reparsed, AdapterPlateSpec)
    assert reparsed == parsed
