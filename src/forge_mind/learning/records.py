"""Typed mirror of the ``generations`` table (supabase/migrations/0001_generation_records.sql).

Keep this in sync with the migration by hand: if you add a column, add a field here (and vice
versa). ``to_row`` produces a Supabase insert payload; ``from_row`` parses a fetched row.

This is the durable record of ONE generation attempt -- success or failure. See DATA_MODEL.md
and ADR-0023. It is intentionally engine-agnostic (legacy LayoutSpec and agentic both fit).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TokenUsage(BaseModel):
    """LLM cost/token telemetry for one generation attempt."""

    model_config = ConfigDict(extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class GenerationRecord(BaseModel):
    """One row of the ``generations`` table.

    Required fields mirror the NOT NULL columns; everything else is optional so a record can
    be created at the start of a request (status ``ok``/``failed``) and updated later with the
    user's accept/reject decision and feedback.
    """

    # protected_namespaces=() so the ``model`` field (the model id) doesn't collide with
    # pydantic's reserved ``model_`` namespace.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # --- required (NOT NULL in the table) ---
    user_id: str
    prompt: str
    engine: str  # layoutspec | agentic
    status: str  # ok | failed | accepted | rejected

    # --- identity / linkage ---
    id: str | None = None
    project_id: str | None = None
    part_id: str | None = None
    created_at: str | None = None
    quality: str | None = None  # standard | studio

    # --- spec lineage ---
    intent: dict[str, Any] | None = None
    blueprint: dict[str, Any] | None = None
    spec: dict[str, Any] | None = None

    # --- result ---
    accepted: bool | None = None
    validation: dict[str, Any] | None = None
    files: dict[str, str] | None = None
    error_type: str | None = None
    error_message: str | None = None

    # --- telemetry ---
    model: str | None = None  # primary model id, e.g. claude-opus-4-8
    provider: str = "anthropic"
    prompt_version: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    duration_ms: int | None = None

    # --- learning ---
    feedback: str | None = None

    def to_row(self) -> dict[str, Any]:
        """Flatten to a ``generations`` insert payload, dropping unset (None) optionals.

        ``usage`` is flattened into the table's ``cost_usd``/``input_tokens``/``output_tokens``
        columns.
        """
        row: dict[str, Any] = {
            "user_id": self.user_id,
            "prompt": self.prompt,
            "engine": self.engine,
            "status": self.status,
            "provider": self.provider,
            "cost_usd": self.usage.cost_usd,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
        }
        optional: dict[str, Any] = {
            "id": self.id,
            "project_id": self.project_id,
            "part_id": self.part_id,
            "created_at": self.created_at,
            "quality": self.quality,
            "intent": self.intent,
            "blueprint": self.blueprint,
            "spec": self.spec,
            "accepted": self.accepted,
            "validation": self.validation,
            "files": self.files,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "duration_ms": self.duration_ms,
            "feedback": self.feedback,
        }
        row.update({k: v for k, v in optional.items() if v is not None})
        return row

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> GenerationRecord:
        """Parse a fetched ``generations`` row back into a record."""
        usage = TokenUsage(
            input_tokens=int(row.get("input_tokens") or 0),
            output_tokens=int(row.get("output_tokens") or 0),
            cost_usd=float(row.get("cost_usd") or 0.0),
        )
        return cls(
            user_id=row["user_id"],
            prompt=row["prompt"],
            engine=row["engine"],
            status=row["status"],
            usage=usage,
            id=row.get("id"),
            project_id=row.get("project_id"),
            part_id=row.get("part_id"),
            created_at=row.get("created_at"),
            quality=row.get("quality"),
            intent=row.get("intent"),
            blueprint=row.get("blueprint"),
            spec=row.get("spec"),
            accepted=row.get("accepted"),
            validation=row.get("validation"),
            files=row.get("files"),
            error_type=row.get("error_type"),
            error_message=row.get("error_message"),
            model=row.get("model"),
            provider=row.get("provider") or "anthropic",
            prompt_version=row.get("prompt_version"),
            duration_ms=row.get("duration_ms"),
            feedback=row.get("feedback"),
        )


__all__ = ["GenerationRecord", "TokenUsage"]
