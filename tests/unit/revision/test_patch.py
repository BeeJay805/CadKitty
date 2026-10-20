"""Patch validation + application (pure, no compile, no LLM)."""

from __future__ import annotations

import pytest

from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.revision.patch import (
    Patch,
    PatchError,
    apply_patch,
    apply_patches,
    validate_patch,
)
from forge_mind.types import Vec3


def _program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="t", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[
            Box(op_id="slab", mode=OpMode.ADD, position=Vec3(z=1.0), width=40, depth=40, height=2),
        ],
        constraints=Constraints(min_wall_thickness=4.0, max_bbox=Vec3(x=60, y=60, z=40)),
    )


def _patch(**kw: object) -> Patch:
    base: dict[str, object] = {
        "op_id": "slab",
        "field_path": "height",
        "old_value": 2.0,
        "new_value": 6.0,
        "reason_check_id": "features.min_wall_thickness",
        "confidence": 0.9,
        "explanation": "thicken the slab",
    }
    base.update(kw)
    return Patch(**base)  # type: ignore[arg-type]


# --- validate_patch -------------------------------------------------------


def test_valid_op_scalar_patch_ok() -> None:
    validate_patch(_program(), _patch())  # no raise


def test_unknown_op_id_rejected() -> None:
    with pytest.raises(PatchError, match="no operation"):
        validate_patch(_program(), _patch(op_id="nope"))


def test_unknown_field_path_rejected() -> None:
    with pytest.raises(PatchError, match="does not exist"):
        validate_patch(_program(), _patch(field_path="radius", old_value=2.0))


def test_op_identity_fields_not_patchable() -> None:
    with pytest.raises(PatchError, match="identity"):
        validate_patch(_program(), _patch(field_path="op_id", old_value="slab", new_value="x"))


def test_program_level_field_must_be_allowlisted() -> None:
    # Changing a validator threshold (a constraint) is exactly what must be blocked.
    with pytest.raises(PatchError, match="not patchable"):
        validate_patch(
            _program(),
            _patch(
                op_id=None,
                field_path="constraints.min_wall_thickness",
                old_value=4.0,
                new_value=1.0,
            ),
        )


def test_operations_list_not_patchable() -> None:
    # No whole-program rewrite: 'operations' is not reachable as a program field.
    with pytest.raises(PatchError, match="not patchable"):
        validate_patch(
            _program(),
            _patch(op_id=None, field_path="operations", old_value="x", new_value="y"),
        )


def test_print_orientation_is_patchable() -> None:
    validate_patch(
        _program(),
        _patch(
            op_id=None,
            field_path="metadata.print_orientation",
            old_value={"x": 0.0, "y": 0.0, "z": 1.0},
            new_value={"x": 0.0, "y": 0.0, "z": -1.0},
        ),
    )


def test_stale_old_value_rejected() -> None:
    with pytest.raises(PatchError, match="stale patch"):
        validate_patch(_program(), _patch(old_value=99.0))


def test_scalar_vector_mismatch_rejected() -> None:
    with pytest.raises(PatchError, match="scalar"):
        validate_patch(_program(), _patch(new_value={"x": 1.0, "y": 2.0, "z": 3.0}))


# --- apply_patch ----------------------------------------------------------


def test_apply_scalar_patch_returns_new_program() -> None:
    prog = _program()
    out = apply_patch(prog, _patch())
    assert out.operations[0].height == 6.0
    assert prog.operations[0].height == 2.0  # original untouched (frozen)


def test_apply_type_check_rejects_constraint_violation() -> None:
    # width must be > 0; a patch that violates the field constraint must fail.
    with pytest.raises(PatchError, match="invalid"):
        apply_patch(_program(), _patch(field_path="width", old_value=40.0, new_value=-5.0))


def test_apply_print_orientation_vector_patch() -> None:
    out = apply_patch(
        _program(),
        _patch(
            op_id=None,
            field_path="metadata.print_orientation",
            old_value={"x": 0.0, "y": 0.0, "z": 1.0},
            new_value={"x": 0.0, "y": 0.0, "z": -1.0},
        ),
    )
    assert out.metadata.print_orientation == Vec3(x=0.0, y=0.0, z=-1.0)


def test_apply_patches_sequential() -> None:
    prog = _program()
    patches = [
        _patch(field_path="height", old_value=2.0, new_value=6.0),
        _patch(field_path="width", old_value=40.0, new_value=55.0),
    ]
    out = apply_patches(prog, patches)
    assert out.operations[0].height == 6.0
    assert out.operations[0].width == 55.0
