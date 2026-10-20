"""Pi4CaseSpec schema validation (DRAFT model; no interpreter exists yet).

These lock the bench-002 spec shape so the (future) interpreter and the LLM
generator have a stable contract to target. Geometry is NOT produced here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.layout.pi4_spec import LidPolicy, Pi4CaseSpec, PortPolicy, VentilationPolicy
from forge_mind.layout.spec import Pattern

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "layout" / "bench002_pi4_case.layoutspec.json"


def _spec_dict() -> dict:
    return json.loads(FIXTURE.read_text())


def test_bench002_fixture_parses() -> None:
    spec = Pi4CaseSpec.model_validate(_spec_dict())
    assert spec.pattern is Pattern.ELECTRONICS_TRAY
    assert spec.board.length_mm == 85.0
    assert spec.standoffs.hole_dx_mm == 58.0
    assert spec.ports is PortPolicy.STANDARD
    assert spec.lid is LidPolicy.NONE


def test_defaults_are_pi4_mechanical_facts() -> None:
    spec = Pi4CaseSpec.model_validate(
        {
            "name": "minimal",
            "envelope_mm": {"x": 95, "y": 70, "z": 30},
            "material": "PLA",
            "min_wall_mm": 2.0,
            "clearance_mm": 0.5,
        }
    )
    assert spec.board.width_mm == 56.0
    assert spec.standoffs.hole_dy_mm == 49.0
    assert spec.standoffs.screw_clearance_mm == 2.7
    assert spec.ventilation is VentilationPolicy.SLOTS


def test_lid_policy_accepts_all_four() -> None:
    for value in ("none", "snap", "screw", "friction"):
        d = _spec_dict()
        d["lid"] = value
        assert Pi4CaseSpec.model_validate(d).lid is LidPolicy(value)


def test_extra_fields_forbidden() -> None:
    d = _spec_dict()
    d["unexpected"] = 1
    with pytest.raises(ValidationError):
        Pi4CaseSpec.model_validate(d)


def test_negative_clearance_rejected() -> None:
    d = _spec_dict()
    d["clearance_mm"] = -1.0
    with pytest.raises(ValidationError):
        Pi4CaseSpec.model_validate(d)
