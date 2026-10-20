"""Unit tests for the DSLDraft -> PartProgram conversion (no LLM).

The whole point of DSLDraft is that holes are subtracted by construction, so the
conversion must guarantee it.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.dsl.draft import DraftHole, DraftInterface, DraftSolid, DSLDraft, draft_to_program
from forge_mind.dsl.interfaces.declaration import InterfaceType
from forge_mind.dsl.primitives import OpMode
from forge_mind.types import Rotation, Vec3

REPO = Path(__file__).resolve().parents[3]
BLUEPRINT = REPO / "tests" / "fixtures" / "blueprints" / "nema17_mount.blueprint.json"


def _blueprint() -> Blueprint:
    return Blueprint.model_validate(json.loads(BLUEPRINT.read_text()))


def test_solids_added_holes_subtracted() -> None:
    draft = DSLDraft(
        solids=[
            DraftSolid(
                op_id="plate",
                position=Vec3(z=2.0),
                rotation=Rotation(),
                width=43.0,
                depth=50.0,
                height=4.0,
            )
        ],
        holes=[
            DraftHole(
                op_id="bore",
                position=Vec3(z=2.0),
                rotation=Rotation(),
                radius=11.0,
                height=4.0,
            )
        ],
        interfaces=[
            DraftInterface(
                name="motor_face",
                interface_type=InterfaceType.NEMA17_FACE,
                location=Vec3(z=2.0),
                normal=Vec3(z=1.0),
            )
        ],
    )
    program = draft_to_program(draft, _blueprint())

    # Solids first (so the first op is add), then holes.
    assert program.operations[0].op == "box"
    assert program.operations[0].mode == OpMode.ADD
    assert program.operations[1].op == "cylinder"
    assert program.operations[1].mode == OpMode.SUBTRACT


def test_metadata_and_constraints_come_from_blueprint() -> None:
    bp = _blueprint()
    draft = DSLDraft(
        solids=[
            DraftSolid(
                op_id="plate",
                position=Vec3(z=2.0),
                rotation=Rotation(),
                width=43.0,
                depth=50.0,
                height=4.0,
            )
        ],
    )
    program = draft_to_program(draft, bp)
    assert program.metadata.name == bp.name
    assert program.constraints.min_wall_thickness == bp.min_wall_mm
    assert program.constraints.max_bbox is not None
    assert program.constraints.max_bbox.x == bp.envelope_mm.x


def test_interfaces_converted_with_empty_params() -> None:
    draft = DSLDraft(
        solids=[
            DraftSolid(
                op_id="p",
                position=Vec3(),
                rotation=Rotation(),
                width=10.0,
                depth=10.0,
                height=4.0,
            )
        ],
        interfaces=[
            DraftInterface(
                name="iface",
                interface_type=InterfaceType.M5_TNUT,
                location=Vec3(),
                normal=Vec3(y=1.0),
            )
        ],
    )
    program = draft_to_program(draft, _blueprint())
    assert len(program.interfaces) == 1
    assert program.interfaces[0].params == {}
