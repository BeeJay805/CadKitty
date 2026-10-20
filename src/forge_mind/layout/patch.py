"""Spec-level parametric patch — the /patch edit-survival core (ADR-0037).

A stored LayoutSpec IS the parametric model of a standard-lane part. An edit
("move the holes 5mm further apart") changes ONE spec field; everything
downstream re-derives through the SAME deterministic interpreter + strict
validator, and every other spec field stays byte-stable — an edit survives the
rest of the design instead of re-cutting it wholesale.

The moat holds: the LLM only picks WHICH catalog field and TRANSCRIBES the
user's stated value/amount; deterministic code does the arithmetic, the
re-validation, and the geometry. An edit that cannot be mapped, or that the
interpreter/validator judges infeasible, REFUSES with the reason — it never
silently re-cuts and never ships a wrong number.

Pieces (mirrors ``revision/patch.py``, which does this at the PartProgram
level for the Phase-7 reviser):

- ``editable_fields``: walk a spec instance into a closed, deterministic
  CATALOG of leaf fields (dotted path, kind, current value, choices).
- ``SpecEdit``: the mapper's structured output — a single (field, op, value).
- ``map_edit``: the one small LLM call + the deterministic gate that refuses
  any emission outside the catalog/kind rules.
- ``apply_spec_edit``: dump -> set ONE leaf -> re-validate the whole spec
  through the LayoutSpec union (Pydantic constraints are the type-check).

Nothing here ships a part: the caller re-runs interpret -> compile ->
validate and only a strict PASS becomes a new version (``api/patch.py``).
"""

from __future__ import annotations

import types
import typing
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from forge_mind.config import BUDGETS, HAIKU
from forge_mind.layout.layoutspec import LayoutSpec
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_SPEC_ADAPTER: TypeAdapter[Any] = TypeAdapter(LayoutSpec)

_PROMPTS = Path(__file__).parent / "prompts"
_MAPPER_PROMPT = (_PROMPTS / "spec_patch.md").read_text(encoding="utf-8")

# Never editable: the union discriminator is the part's identity, not a parameter.
_EXCLUDED_PATHS = frozenset({"pattern"})


class SpecPatchError(ValueError):
    """An edit that cannot be applied. The message is the user-facing refusal reason."""


@dataclass(frozen=True)
class EditableField:
    """One leaf of the editable-field catalog (deterministically derived)."""

    path: str  # dotted; list items by index ("interfaces.0.spacing_mm")
    kind: str  # "number" | "boolean" | "enum" | "string"
    current: Any  # JSON value currently at path (None for unset optionals)
    description: str
    choices: tuple[str, ...] | None = None  # enum members for kind="enum"


def _kind_from_annotation(ann: Any) -> tuple[str, tuple[str, ...] | None] | None:
    """Classify a field ANNOTATION into a catalog kind (used for None-valued leaves).

    Returns None when the annotation is not a supported editable kind — the field is
    simply left out of the catalog (conservative: only well-understood leaves are editable).
    """
    origin = typing.get_origin(ann)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(args) != 1:
            return None
        ann = args[0]
        origin = typing.get_origin(ann)
    if origin is Literal:
        return None  # literals pin identity (e.g. `pattern`), never editable
    if not isinstance(ann, type):
        return None
    if issubclass(ann, Enum):
        return "enum", tuple(str(m.value) for m in ann)
    if issubclass(ann, bool):
        return "boolean", None
    if issubclass(ann, int | float):
        return "number", None
    if issubclass(ann, str):
        return "string", None
    return None


def _kind_of_value(value: Any) -> tuple[str, tuple[str, ...] | None] | None:
    """Classify a present VALUE into a catalog kind. Enum precedes str (StrEnum is a str)."""
    if isinstance(value, Enum):
        return "enum", tuple(str(m.value) for m in type(value))
    if isinstance(value, bool):
        return "boolean", None
    if isinstance(value, int | float):
        return "number", None
    if isinstance(value, str):
        return "string", None
    return None


def _walk(model: BaseModel, prefix: str, out: list[EditableField]) -> None:
    for name, info in type(model).model_fields.items():
        path = f"{prefix}{name}"
        if path in _EXCLUDED_PATHS:
            continue
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            _walk(value, f"{path}.", out)
            continue
        if isinstance(value, list):
            # Parameter edits on existing items only. Adding/removing features is a
            # structural change — out of scope for the parametric patch (refused upstream).
            for i, item in enumerate(value):
                if isinstance(item, BaseModel):
                    _walk(item, f"{path}.{i}.", out)
            continue
        kind = _kind_from_annotation(info.annotation) if value is None else _kind_of_value(value)
        if kind is None:
            continue
        current = value.value if isinstance(value, Enum) else value
        out.append(
            EditableField(
                path=path,
                kind=kind[0],
                current=current,
                description=info.description or "",
                choices=kind[1],
            )
        )


def editable_fields(spec: Any) -> list[EditableField]:
    """The closed catalog of editable leaves for ``spec``, in declaration order."""
    out: list[EditableField] = []
    _walk(spec, "", out)
    return out


class SpecEdit(BaseModel):
    """The mapper's structured output: ONE categorical field pick + the user's value.

    Small by design (the ADR-0019/0020 grammar lesson). ``field_path`` is validated
    against the catalog AFTER the call — the deterministic gate, not the grammar,
    enforces the closed vocabulary.
    """

    model_config = ConfigDict(extra="forbid")

    mappable: bool = Field(
        ..., description="False when the edit does not map to exactly one catalog field."
    )
    reason: str | None = Field(
        default=None,
        description="When unmappable: one human sentence saying why and what IS editable.",
    )
    field_path: str | None = Field(
        default=None, description="The catalog path to edit (copied verbatim from the catalog)."
    )
    op: Literal["set", "increase", "decrease"] | None = Field(
        default=None,
        description="set = replace with the user's value; increase/decrease = "
        "current +/- the user's stated amount (numbers only).",
    )
    value: float | bool | str | None = Field(
        default=None,
        description="The user's stated value or amount, transcribed — never invented.",
    )


def render_catalog(spec: Any) -> str:
    """The catalog as prompt lines. Deterministic; the only spec view the mapper sees."""
    lines = []
    for f in editable_fields(spec):
        choices = f" choices={list(f.choices)}" if f.choices else ""
        desc = f" -- {f.description}" if f.description else ""
        lines.append(f"- {f.path} ({f.kind}, current={f.current!r}{choices}){desc}")
    return "\n".join(lines)


def _resolve(data: dict[str, Any], segments: list[str]) -> Any:
    cur: Any = data
    for seg in segments:
        if isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError) as exc:
                raise SpecPatchError(f"field path segment {seg!r} not found") from exc
        elif isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        else:
            raise SpecPatchError(f"field path segment {seg!r} not found")
    return cur


def _check_value_for(field: EditableField, op: str, value: Any) -> None:
    """The deterministic kind gate. Raises SpecPatchError with a user-facing reason."""
    if op in ("increase", "decrease"):
        if field.kind != "number":
            raise SpecPatchError(
                f"'{field.path}' is a {field.kind} field; increase/decrease applies to numbers only."
            )
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise SpecPatchError(
                f"increase/decrease needs the amount as a number (got {value!r}). "
                "State the amount, e.g. '5mm'."
            )
        if field.current is None:
            raise SpecPatchError(
                f"'{field.path}' has no current value to adjust; state the target value instead."
            )
        return
    # op == "set"
    if field.kind == "number" and (not isinstance(value, int | float) or isinstance(value, bool)):
        raise SpecPatchError(f"'{field.path}' needs a numeric value (got {value!r}).")
    if field.kind == "boolean" and not isinstance(value, bool):
        raise SpecPatchError(f"'{field.path}' needs true/false (got {value!r}).")
    if field.kind == "enum" and (
        not isinstance(value, str) or (field.choices and value not in field.choices)
    ):
        raise SpecPatchError(
            f"'{field.path}' must be one of {list(field.choices or ())} (got {value!r})."
        )
    if field.kind == "string" and not isinstance(value, str):
        raise SpecPatchError(f"'{field.path}' needs a text value (got {value!r}).")


def apply_spec_edit(spec: Any, edit: SpecEdit) -> tuple[Any, Any, Any]:
    """Apply one validated edit; returns ``(new_spec, old_value, new_value)``.

    Deterministic: dump -> set the ONE leaf -> re-validate through the LayoutSpec
    union. A wrong type or a constraint violation (e.g. ``gt=0``) raises
    :class:`SpecPatchError` instead of producing a broken spec. Every other field
    is byte-stable by construction (only one leaf of the dump is touched).
    """
    if not edit.mappable or edit.field_path is None or edit.op is None:
        raise SpecPatchError(edit.reason or "That edit does not map to a single spec parameter.")
    catalog = {f.path: f for f in editable_fields(spec)}
    field = catalog.get(edit.field_path)
    if field is None:
        raise SpecPatchError(
            f"'{edit.field_path}' is not an editable parameter of this part. "
            f"Editable: {', '.join(sorted(catalog))}."
        )
    _check_value_for(field, edit.op, edit.value)

    if edit.op == "increase":
        assert isinstance(edit.value, int | float)
        new_value: Any = float(field.current) + float(edit.value)
    elif edit.op == "decrease":
        assert isinstance(edit.value, int | float)
        new_value = float(field.current) - float(edit.value)
    else:
        new_value = edit.value

    data = spec.model_dump(mode="json")
    segments = edit.field_path.split(".")
    parent = _resolve(data, segments[:-1]) if len(segments) > 1 else data
    last = segments[-1]
    if isinstance(parent, list):
        parent[int(last)] = new_value
    else:
        parent[last] = new_value
    try:
        new_spec = _SPEC_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise SpecPatchError(
            f"that change is out of range ({edit.field_path}={new_value!r}): {_first_error(exc)}"
        ) from exc
    return new_spec, field.current, new_value


def _interface_summary(placement: dict[str, Any]) -> str:
    """A short ``ref@face`` label for the edit view (the full interface rides the returned spec)."""
    return f"{placement.get('ref', 'interface')}@{placement.get('face', '?')}"


def apply_spec_structural_edit(
    spec: Any,
    op: str,
    *,
    index: int | None = None,
    placement: dict[str, Any] | None = None,
) -> tuple[Any, str, str | None, str | None]:
    """Add or remove ONE item on the composable plate's ``interfaces`` list, deterministically
    (ADR-0060 slice C — the structural edit ADR-0037 deferred).

    Returns ``(new_spec, field_path, old_summary, new_summary)``; the rebuild driver feeds these into
    the SAME interpret -> compile -> validate -> ship spine a parameter edit uses. The whole spec is
    re-validated through the LayoutSpec union, so an unknown ref/face, a min/max-length violation, or
    a bad index becomes a :class:`SpecPatchError` refusal, never a broken spec. Nothing here builds a
    part; only a strict PASS downstream becomes a new version.

    Structural edits apply ONLY to the ``interfaces`` list (the composable plate's feature/interface
    menu); a spec without it (a tray, an arm, the legacy plates) refuses with a clear reason.
    """
    interfaces = getattr(spec, "interfaces", None)
    if not isinstance(interfaces, list):
        raise SpecPatchError(
            "this part has no add/remove-able features; structural edits apply to the composable "
            "plate's interfaces only."
        )
    data = spec.model_dump(mode="json")
    items: list[Any] = data["interfaces"]
    if op == "add":
        if not isinstance(placement, dict):
            raise SpecPatchError("an add needs the interface to place (a ref + face).")
        field_path = f"interfaces.{len(items)}"
        old_summary, new_summary = None, _interface_summary(placement)
        items.append(placement)
    elif op == "remove":
        if index is None:
            raise SpecPatchError("a remove needs the index of the interface to drop.")
        if not 0 <= index < len(items):
            raise SpecPatchError(
                f"interface index {index} is out of range (this part has {len(items)})."
            )
        field_path, old_summary, new_summary = (
            f"interfaces.{index}",
            _interface_summary(items[index]),
            None,
        )
        del items[index]
    else:
        raise SpecPatchError(f"unknown structural op {op!r} (expected 'add' or 'remove').")
    try:
        new_spec = _SPEC_ADAPTER.validate_python(data)
    except ValidationError as exc:
        raise SpecPatchError(f"that structural change is not valid: {_first_error(exc)}") from exc
    return new_spec, field_path, old_summary, new_summary


def map_edit(
    spec: Any,
    edit_text: str,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> tuple[SpecEdit, LLMUsage]:
    """Map ``edit_text`` to ONE catalog field via a single small structured call.

    The emission is NOT trusted: ``apply_spec_edit`` re-validates the path, kind, and
    value deterministically, so an out-of-vocabulary emission becomes a refusal,
    never a wrong part.
    """
    client = client or make_client()
    user = (
        f"Part: {getattr(spec, 'name', 'part')} (pattern: {spec.pattern.value})\n\n"
        f"EDITABLE FIELD CATALOG:\n{render_catalog(spec)}\n\n"
        f"Edit request: {edit_text}\n"
    )
    return client.generate_structured(
        system=_MAPPER_PROMPT,
        user=user,
        output_model=SpecEdit,
        model=model or HAIKU,
        max_tokens=BUDGETS["spec_patch"].max_tokens,
    )


def validate_spec_payload(spec_data: dict[str, Any]) -> Any:
    """Validate an inbound (client-supplied) spec dict through the LayoutSpec union.

    /patch is STATELESS — the spec rides the /generate response and comes back here.
    The union validation plus the downstream interpreter + strict gates mean an
    arbitrary payload can at worst build a valid part, never a wrong number.
    Raises :class:`SpecPatchError` with a user-facing message on an invalid payload.
    """
    try:
        return _SPEC_ADAPTER.validate_python(spec_data)
    except ValidationError as exc:
        raise SpecPatchError(
            f"this part does not carry a usable parametric model ({_first_error(exc)})"
        ) from exc


def _first_error(exc: ValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return str(exc)
    e = errors[0]
    loc = ".".join(str(p) for p in e.get("loc", ()))
    return f"{loc}: {e.get('msg', '')}".strip(": ")


__all__ = [
    "EditableField",
    "SpecEdit",
    "SpecPatchError",
    "apply_spec_edit",
    "apply_spec_structural_edit",
    "editable_fields",
    "map_edit",
    "render_catalog",
    "validate_spec_payload",
]
