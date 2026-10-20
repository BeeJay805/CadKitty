"""Shared immutable editing base; a captured plan is intent, never evidence."""

from __future__ import annotations

from pydantic import Field, model_validator

from forge_mind.workspace.models import Record


class EditBasis(Record):
    revision_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")
    document_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    plan_version: int = Field(ge=0, strict=True)
    plan_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    contract_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_version: int = Field(ge=0, strict=True)
    source_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def coherent_source(self) -> EditBasis:
        if (self.revision_id is None) != (self.document_hash is None):
            raise ValueError("An edit basis needs both revision_id and document_hash, or neither")
        if (self.plan_version == 0) != (self.plan_hash is None):
            raise ValueError("An edit basis needs a plan_hash for an initialized plan")
        return self
