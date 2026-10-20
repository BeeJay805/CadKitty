"""LLMReviser — proposes patches for a failing part using the LLM.

This is one implementation of the ``ProposeFn`` signature the revise() loop
takes; it is gated behind a flag at the call site, never invoked by the loop
itself. The model emits a generation-friendly ``PatchSet`` (a field, an op, and
either a number or a vector). We fill ``old_value`` from the LIVE program and
re-validate every proposed patch, dropping anything that does not target a real,
patchable field. So a hallucinated op_id or field simply gets discarded rather
than corrupting the program.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.config import BUDGETS, REVISER_MODEL
from forge_mind.dsl.program import PartProgram
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.revision.patch import Patch, PatchError, read_value, validate_patch
from forge_mind.revision.reviser import ProposeResult
from forge_mind.types import Vec3
from forge_mind.validator.report import ValidationReport

_PROMPT_PATH = Path(__file__).parent / "prompts" / "reviser_v1.md"


class ProposedPatch(BaseModel):
    """What the model emits per patch. Scalar fields use ``new_value_number``;
    ``metadata.print_orientation`` uses ``new_value_vector``.
    """

    model_config = ConfigDict(extra="forbid")

    op_id: str | None = Field(
        default=None, description="Operation to edit, or null for a program-level field."
    )
    field_path: str = Field(..., min_length=1)
    new_value_number: float | None = None
    new_value_vector: Vec3 | None = None
    reason_check_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., min_length=1, max_length=500)


class PatchSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patches: list[ProposedPatch] = Field(default_factory=list)


def _to_patch(program: PartProgram, prop: ProposedPatch) -> Patch | None:
    """Convert a proposed patch to a validated Patch, or None if it does not
    target a real, patchable field (we fill old_value from the live program).
    """
    try:
        old = read_value(program, prop.op_id, prop.field_path)
    except PatchError:
        return None

    if isinstance(old, dict):
        if prop.new_value_vector is None:
            return None
        new_value: float | dict[str, float] = {
            "x": prop.new_value_vector.x,
            "y": prop.new_value_vector.y,
            "z": prop.new_value_vector.z,
        }
    else:
        if prop.new_value_number is None:
            return None
        new_value = prop.new_value_number

    try:
        patch = Patch(
            op_id=prop.op_id,
            field_path=prop.field_path,
            old_value=old,
            new_value=new_value,
            reason_check_id=prop.reason_check_id,
            confidence=prop.confidence,
            explanation=prop.explanation,
        )
        validate_patch(program, patch)
    except (PatchError, ValueError):
        return None
    return patch


def _format_failures(report: ValidationReport) -> str:
    lines: list[str] = ["Failed checks:"]
    for c in report.checks:
        if c.status.value == "fail":
            detail = f" expected {c.expected}, actual {c.actual}" if c.expected else ""
            lines.append(f"- {c.id}: {c.message}{detail}")
    if report.suggested_fixes:
        lines.append("\nValidator fix hints:")
        for f in report.suggested_fixes:
            lines.append(f"- ({f.field}) {f.reason}")
    return "\n".join(lines)


class LLMReviser:
    """Proposes patches for a failing part. Matches the ProposeFn signature."""

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or REVISER_MODEL

    def __call__(self, program: PartProgram, report: ValidationReport) -> ProposeResult:
        user = (
            "Repair this part with minimal field patches.\n\n"
            f"{_format_failures(report)}\n\n"
            "PartProgram (JSON):\n"
            f"{program.model_dump_json(indent=2)}\n"
        )
        patchset, usage = self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=PatchSet,
            model=self._model,
            max_tokens=BUDGETS["reviser"].max_tokens,
        )
        patches = [p for p in (_to_patch(program, pp) for pp in patchset.patches) if p is not None]
        return ProposeResult(patches=patches, usage=usage)


__all__ = ["LLMReviser", "PatchSet", "ProposedPatch"]
