"""Customer creation policy over the existing native author and durable workspace."""

from __future__ import annotations

import os
from typing import Any

from pydantic import Field, field_validator

from forge_mind.workspace.customer_references import (
    MAX_REFERENCE_BYTES,
    MAX_REFERENCE_PIXELS,
    REFERENCE_MEDIA_TYPES,
    CustomerReference,
)
from forge_mind.workspace.models import Contract, Record


class CustomerProjectRequest(Record):
    request_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    prompt: str = Field(min_length=1, max_length=100000)
    contract: Contract | None = None
    max_actions: int | None = Field(default=None, ge=1, strict=True)
    max_seconds: int | None = Field(default=None, ge=1, strict=True)
    approval_required: bool = Field(default=False, strict=True)
    printable: bool = Field(default=True, strict=True)
    reference: CustomerReference | None = None

    @field_validator("prompt")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be blank")
        return value


def customer_project(session: dict[str, Any]) -> bool:
    """Only server-created customer records inherit the complete-delivery policy."""
    return (
        session.get("project_kind") == "workspace"
        and session.get("creation_request_id") is not None
        and session.get("authoring_mode") == "native_operations"
    )


def hosted_customer_limits() -> dict[str, int] | None:
    """A server operator must set both caps; private-local jobs retain their policy."""
    result = {}
    for name, variable in (
        ("max_seconds", "FORGE_CUSTOMER_HOSTED_MAX_SECONDS"),
        ("max_actions", "FORGE_CUSTOMER_HOSTED_MAX_ACTIONS"),
    ):
        raw = os.environ.get(variable, "")
        if not raw.isascii() or not raw.isdigit() or int(raw) <= 0:
            return None
        result[name] = int(raw)
    return result


def customer_capabilities(executor: Any) -> dict[str, Any]:
    reference = {
        "max_reference_bytes": MAX_REFERENCE_BYTES,
        "max_reference_pixels": MAX_REFERENCE_PIXELS,
        "reference_media_types": REFERENCE_MEDIA_TYPES,
    }
    if executor.mode == "trusted_local":
        return {
            "enabled": True,
            "scope": "private_local",
            "reason": "Private loopback customer lifecycle; hosted and physical qualification remain separate.",
            **reference,
        }
    qualified = (
        executor.typed_container_verified is True
        and getattr(executor, "customer_container_verified", False) is True
        and hosted_customer_limits() is not None
    )
    return {
        "enabled": qualified,
        "scope": "qualified_isolated" if qualified else "unavailable",
        "reason": "Isolated customer geometry, proof and delivery workers qualified."
        if qualified
        else "Customer delivery requires isolated geometry, final proof and delivery qualification.",
        **reference,
    }
