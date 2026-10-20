"""Owned input lifecycle boundary; no target, CAD or approval is supplied here."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.models import Record


class AbandonInputRequest(Record):
    input_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    source_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    basis: EditBasis


def input_application_summary(document: dict[str, Any]) -> dict[str, Any]:
    """Expose stable pending identities without copying owner text or saved CAD."""
    from forge_mind.workspace.input_application import pending

    records: dict[str, dict[str, Any]] = {}
    for location, snapshot in (
        ("current", document),
        ("proposal", document.get("proposal") or {}),
        ("checkpoint", document.get("resume_candidate") or {}),
    ):
        item = pending(snapshot)
        if item is None:
            continue
        identity = item["input_id"]
        record = records.setdefault(
            identity,
            {
                key: item[key]
                for key in ("input_id", "source_count", "source_hash", "mode", "origin")
            },
        )
        record.setdefault("locations", []).append(location)
    return {
        "pending": list(records.values()),
        "scope": "Owner input application only; no geometry, acceptance or physical approval.",
    }
