"""Pydantic validation tests for DSL primitives.

We test that:
- Valid ops parse cleanly with defaults applied.
- Invalid ops (zero/negative dimensions, bad op_id, duplicate op_id, bad
  discriminator, unknown fields) are rejected with clear errors.
- Discriminated-union round-trip via PartProgram preserves concrete subtype.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram

# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


def test_box_minimal_valid() -> None:
    b = Box(op_id="motor_flange", width=42.0, depth=50.0, height=4.0)
    assert b.op == "box"
    assert b.mode == OpMode.ADD
    assert b.position.as_tuple() == (0.0, 0.0, 0.0)
    assert b.rotation.x == 0.0


def test_box_rejects_zero_dimension() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Box(op_id="x", width=0.0, depth=10.0, height=10.0)


def test_box_rejects_negative_dimension() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Box(op_id="x", width=10.0, depth=-1.0, height=10.0)


def test_box_rejects_bad_op_id() -> None:
    # Pattern requires identifier-shaped string.
    with pytest.raises(ValidationError):
        Box(op_id="123abc", width=10.0, depth=10.0, height=10.0)
    with pytest.raises(ValidationError):
        Box(op_id="has space", width=10.0, depth=10.0, height=10.0)
    with pytest.raises(ValidationError):
        Box(op_id="", width=10.0, depth=10.0, height=10.0)


def test_box_rejects_extra_field() -> None:
    with pytest.raises(ValidationError, match=r"extra_forbidden|Extra inputs are not permitted"):
        Box(op_id="x", width=10.0, depth=10.0, height=10.0, color="red")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Cylinder
# ---------------------------------------------------------------------------


def test_cylinder_minimal_valid() -> None:
    c = Cylinder(op_id="shaft", radius=11.0, height=4.0)
    assert c.op == "cylinder"
    assert c.radius == 11.0


def test_cylinder_rejects_zero_radius() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Cylinder(op_id="x", radius=0.0, height=4.0)


def test_cylinder_unit_error_guard() -> None:
    # 1500mm radius is almost certainly a units mistake.
    with pytest.raises(ValidationError, match="unit error"):
        Cylinder(op_id="x", radius=1500.0, height=4.0)


# ---------------------------------------------------------------------------
# PartProgram
# ---------------------------------------------------------------------------


def _meta() -> Metadata:
    return Metadata(name="test_part")


def test_program_rejects_empty_operations() -> None:
    with pytest.raises(ValidationError, match=r"at least 1 item|too_short"):
        PartProgram(metadata=_meta(), operations=[])


def test_program_rejects_duplicate_op_ids() -> None:
    box = Box(op_id="dup", width=10.0, depth=10.0, height=10.0)
    other = Cylinder(op_id="dup", radius=2.0, height=1.0)
    with pytest.raises(ValidationError, match="duplicate op_id"):
        PartProgram(metadata=_meta(), operations=[box, other])


def test_program_rejects_subtract_first() -> None:
    sub_only = Cylinder(op_id="x", mode=OpMode.SUBTRACT, radius=1.0, height=1.0)
    with pytest.raises(ValidationError, match="first operation must have mode='add'"):
        PartProgram(metadata=_meta(), operations=[sub_only])


def test_program_round_trip_via_json() -> None:
    program = PartProgram(
        metadata=_meta(),
        operations=[
            Box(op_id="base", width=10.0, depth=10.0, height=2.0),
            Cylinder(op_id="hole", mode=OpMode.SUBTRACT, radius=1.0, height=2.0),
        ],
    )
    dumped = program.model_dump_json()
    reloaded = PartProgram.model_validate_json(dumped)

    # Discriminated union preserves concrete subtype after round-trip.
    assert isinstance(reloaded.operations[0], Box)
    assert isinstance(reloaded.operations[1], Cylinder)
    assert reloaded.operations[1].mode == OpMode.SUBTRACT


def test_program_loads_nema17_fixture() -> None:
    """The hand-authored fixture must always parse — it's the Phase 1 anchor."""
    from pathlib import Path

    fixture = (
        Path(__file__).parents[2] / "fixtures" / "handwritten" / "nema17_mount_good.partdsl.json"
    )
    data = json.loads(fixture.read_text())
    program = PartProgram.model_validate(data)
    assert program.metadata.name == "nema17_mount_good"
    assert len(program.operations) == 9
