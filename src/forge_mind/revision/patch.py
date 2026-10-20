"""Patch — a single field-level edit to a PartProgram.

Phase 7 revision is *patch-only*: the reviser may not regenerate the program, it
may only propose small, targeted edits to existing operations (or the part's
print orientation). That constraint is enforced structurally here:

- A ``Patch`` carries a single ``field_path`` and a single ``new_value`` that can
  only be a scalar or a flat vector (``dict[str, float]``). It cannot carry an
  operation, a list of operations, or any nested structure -- so a patch can
  never replace the program wholesale.
- ``op_id=None`` (a program-level patch) is restricted to an allow-list that
  contains ONLY ``metadata.print_orientation``. Constraints (the validator's
  thresholds) and the operations list are deliberately not patchable, so a patch
  can never weaken a check to "pass" or rewrite the body.

Application is deterministic: dump the program to a JSON dict, set the one leaf,
and re-validate through Pydantic. The re-validation is the type-check -- a patch
that would make a field the wrong type (or violate a field constraint such as
``gt=0``) raises ``PatchError`` instead of producing a broken program.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from forge_mind.dsl.program import PartProgram

# A patch value is a scalar or a flat vector. Notably NOT a list or nested model:
# this is what makes "no whole-program rewrite" structural rather than a check.
# bool precedes float so a JSON ``true`` is not coerced to ``1.0``.
PatchValue = bool | float | str | dict[str, float]

# The only program-level (op_id=None) field a patch may touch. Reorientation is a
# legitimate design choice the printability tier itself suggests. Everything else
# at the program level (constraints, operations, version, metadata.name) is off
# limits: constraints would change validator thresholds, operations would be a
# rewrite.
PROGRAM_LEVEL_PATCHABLE = frozenset({"metadata.print_orientation"})

# Op fields a patch must never touch: identity and the type discriminator.
_OP_FIELDS_FORBIDDEN = frozenset({"op_id", "op"})


class PatchError(ValueError):
    """Raised when a patch is invalid or cannot be applied."""


class Patch(BaseModel):
    """A single field-level edit, addressed by op_id + dotted field_path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op_id: str | None = Field(
        default=None,
        description="Operation to edit. None = a program-level field (allow-listed).",
    )
    field_path: str = Field(
        ...,
        min_length=1,
        description="Dotted path to the field, e.g. 'height', 'position.z', "
        "'metadata.print_orientation'.",
    )
    old_value: PatchValue = Field(
        ...,
        description="Current value at the path; the patch is rejected if it no longer matches.",
    )
    new_value: PatchValue = Field(..., description="Replacement value (scalar or flat vector).")
    reason_check_id: str = Field(
        ...,
        description="The failed check id this patch is meant to repair, e.g. "
        "'features.min_wall_thickness'.",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Reviser confidence in [0, 1].")
    explanation: str = Field(..., min_length=1, max_length=500)


def _find_op(program: PartProgram, op_id: str) -> Any | None:
    for op in program.operations:
        if op.op_id == op_id:
            return op
    return None


def _resolve(node: Any, segments: list[str]) -> Any:
    """Walk ``segments`` into a dumped dict, raising KeyError if any is missing."""
    cur = node
    for seg in segments:
        if not isinstance(cur, dict) or seg not in cur:
            raise KeyError(seg)
        cur = cur[seg]
    return cur


def _values_match(current: Any, expected: PatchValue) -> bool:
    if isinstance(current, int | float) and isinstance(expected, int | float):
        return math.isclose(float(current), float(expected), rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(current, dict) and isinstance(expected, dict):
        if set(current) != set(expected):
            return False
        return all(_values_match(current[k], expected[k]) for k in current)
    return bool(current == expected)


def validate_patch(program: PartProgram, patch: Patch) -> None:
    """Raise PatchError unless the patch targets an existing, patchable field
    whose current value still matches ``patch.old_value``.

    Type-checking of ``new_value`` happens at application time (a re-validation of
    the whole program); here we enforce target existence, the patch-only scope,
    and the optimistic-concurrency guard.
    """
    segments = patch.field_path.split(".")
    if any(not s for s in segments):
        raise PatchError(f"malformed field_path: {patch.field_path!r}")

    if patch.op_id is None:
        # Program-level patch: only the explicit allow-list, nothing else.
        if patch.field_path not in PROGRAM_LEVEL_PATCHABLE:
            raise PatchError(
                f"program-level field {patch.field_path!r} is not patchable "
                f"(allowed: {sorted(PROGRAM_LEVEL_PATCHABLE)}). Constraints and the "
                "operations list are intentionally off limits."
            )
        root: Any = program.model_dump(mode="json")
    else:
        op = _find_op(program, patch.op_id)
        if op is None:
            raise PatchError(f"no operation with op_id={patch.op_id!r}")
        if segments[0] in _OP_FIELDS_FORBIDDEN:
            raise PatchError(
                f"field {segments[0]!r} is not patchable (op identity / type discriminator)"
            )
        root = op.model_dump(mode="json")

    try:
        current = _resolve(root, segments)
    except KeyError as exc:
        raise PatchError(f"field_path {patch.field_path!r} does not exist on the target") from exc

    # Refuse to replace a whole submodel-shaped field with a scalar or vice versa;
    # a leaf must be a scalar, a vector field must receive a flat vector dict.
    if isinstance(current, dict) and not isinstance(patch.new_value, dict):
        raise PatchError(
            f"field_path {patch.field_path!r} is a vector; new_value must be a flat "
            "{x,y,z}-style object, not a scalar"
        )
    if not isinstance(current, dict) and isinstance(patch.new_value, dict):
        raise PatchError(
            f"field_path {patch.field_path!r} is a scalar; new_value must be a scalar, not an object"
        )

    if not _values_match(current, patch.old_value):
        raise PatchError(
            f"stale patch: old_value {patch.old_value!r} does not match current "
            f"value {current!r} at {patch.field_path!r}"
        )


def read_value(program: PartProgram, op_id: str | None, field_path: str) -> Any:
    """Current JSON-dumped value at (op_id, field_path). Raises PatchError if the
    op or field does not exist. Used by the LLM reviser to fill ``old_value``
    itself rather than trusting the model to echo it.
    """
    segments = field_path.split(".")
    if op_id is None:
        root: Any = program.model_dump(mode="json")
    else:
        op = _find_op(program, op_id)
        if op is None:
            raise PatchError(f"no operation with op_id={op_id!r}")
        root = op.model_dump(mode="json")
    try:
        return _resolve(root, segments)
    except KeyError as exc:
        raise PatchError(f"field_path {field_path!r} does not exist on the target") from exc


def apply_patch(program: PartProgram, patch: Patch) -> PartProgram:
    """Apply one validated patch, returning a new PartProgram.

    Re-validates the whole program through Pydantic, which is the new_value
    type-check: a wrong type or a constraint violation (e.g. width <= 0) raises
    PatchError rather than yielding an invalid program.
    """
    validate_patch(program, patch)

    data: dict[str, Any] = program.model_dump(mode="json")
    segments = patch.field_path.split(".")

    if patch.op_id is None:
        parent: Any = data
        target_segments = segments
    else:
        ops = data["operations"]
        idx = next(i for i, o in enumerate(ops) if o["op_id"] == patch.op_id)
        parent = ops[idx]
        target_segments = segments

    cursor = parent
    for seg in target_segments[:-1]:
        cursor = cursor[seg]
    cursor[target_segments[-1]] = patch.new_value

    try:
        return PartProgram.model_validate(data)
    except ValidationError as exc:
        raise PatchError(
            f"patch makes the program invalid ({patch.field_path}={patch.new_value!r}): "
            f"{_first_error(exc)}"
        ) from exc


def apply_patches(program: PartProgram, patches: list[Patch]) -> PartProgram:
    """Apply patches in order. Each is validated against the running (already
    patched) program, so a later patch sees the effects of earlier ones.
    """
    current = program
    for patch in patches:
        current = apply_patch(current, patch)
    return current


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    e = errors[0]
    loc = ".".join(str(p) for p in e.get("loc", ()))
    return f"{loc}: {e.get('msg', '')}".strip(": ")


__all__ = [
    "PROGRAM_LEVEL_PATCHABLE",
    "Patch",
    "PatchError",
    "PatchValue",
    "apply_patch",
    "apply_patches",
    "read_value",
    "validate_patch",
]
