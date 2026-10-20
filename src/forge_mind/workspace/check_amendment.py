"""Explicit criterion corrections preserve identity, history and frozen claims.

An authored reason records intent; it cannot prove that a predicate establishes
a requirement. Old saved plans load unchanged. New semantic request edits require
the actual predecessor hash at their shared admission boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def request_hash(request: Any) -> str:
    """Use the existing exact prepared-request hash, without normalization."""
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def same_criterion(left: Any, right: Any) -> bool:
    """Compare typed values, not serialization differences or numerical tolerances.

    Pydantic can emit an unvalidated numeric default as 0 and its round-trip as
    0.0. Defaults omitted by older callers also retain their schema meaning.
    Receipt identity and amendment predecessor hashes remain exact and separate.
    """
    from forge_mind.workspace.product_tools import _ADAPTER

    a = _ADAPTER.validate_python(left).model_dump(mode="json")
    b = _ADAPTER.validate_python(right).model_dump(mode="json")
    return bool(a == b)


class CriterionAmendment(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    previous_request_hash: str = Field(strict=True, pattern=r"^[a-f0-9]{64}$")
    reason: str = Field(strict=True, min_length=1, max_length=2000)
    basis: str = Field(
        strict=True,
        min_length=1,
        max_length=2000,
        description=(
            "Explicit source/requirement basis for correcting the predicate. Authored provenance "
            "only; never requirement acceptance, physical proof or a failure waiver."
        ),
    )

    @field_validator("reason", "basis")
    @classmethod
    def meaningful_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Criterion amendment reason and basis must contain non-whitespace text"
            )
        return value


def validate_check_amendments(plan: dict[str, Any], previous: dict[str, Any] | None) -> None:
    """Validate an admission against its predecessor, or an empty initial plan.

    Standalone saved-plan validation does not call this admission-only helper.
    """
    previous = previous or {}
    old = {
        check["id"]: (subsystem["id"], check)
        for subsystem in previous.get("subsystems", [])
        for check in subsystem.get("checks", [])
    }
    for subsystem in plan.get("subsystems", []):
        for check in subsystem.get("checks", []):
            predecessor = old.get(check["id"])
            amendment = check.get("criterion_amendment")
            if predecessor is None:
                if amendment is not None:
                    raise ValueError("A new check cannot amend an unknown same-ID predecessor")
                continue
            prior_subsystem, prior = predecessor
            if prior_subsystem != subsystem["id"]:
                raise ValueError("A criterion amendment cannot move a check to another subsystem")
            changed = not same_criterion(check["request"], prior["request"])
            if not changed:
                if amendment != prior.get("criterion_amendment"):
                    raise ValueError(
                        "Retain criterion amendment provenance when the request is unchanged"
                    )
                continue
            if amendment is None:
                raise ValueError(
                    "Changed prepared check requests require an explicit criterion_amendment"
                )
            parsed = CriterionAmendment.model_validate(amendment)
            if parsed.previous_request_hash != request_hash(prior["request"]):
                raise ValueError(
                    "Criterion amendment predecessor request hash changed; refresh the plan"
                )
            if amendment == prior.get("criterion_amendment"):
                raise ValueError("A new criterion change requires a new explicit amendment")
