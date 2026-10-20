"""Read-only handoff obligations from saved intent, never engineering approval."""

from __future__ import annotations

from typing import Any

from forge_mind.workspace.assembly import expand
from forge_mind.workspace.document import DesignDocument


def print_preparation(
    document: DesignDocument, guide: dict[str, Any], *, explicitly_requested: bool = False
) -> dict[str, Any]:
    """Require preparation for occurring made parts explicitly declared additive.

    A raw STL declaration alone does not request plate preparation. Production
    intent supplies this obligation; no prompt text or file name is interpreted.
    The trusted guidance reader verifies any current setup receipt.
    """
    if document.assembly is None:
        definitions = [("part", document.production)]
    else:
        occurring = {row["definition"] for row in expand(document)}
        definitions = [
            (part.id, part.production) for part in document.assembly.parts if part.id in occurring
        ]
    required = sorted(
        part_id
        for part_id, production in definitions
        if production is not None
        and production.role == "make"
        and (production.process == "additive" or explicitly_requested)
    )
    setup = (guide.get("facts") or {}).get("print_setup") or {}
    supported = document.assembly is not None
    return {
        "required": bool(required),
        "supported": supported,
        "part_ids": required,
        "status": (
            setup.get("status", "missing")
            if required and supported
            else "conversion_required"
            if required
            else "not_requested"
        ),
        "recovery": (
            "Use document with plan_patch_json to convert the current flat CAD to an explicit single-part assembly. Preserve its Features, variables, bindings and production intent, and supply explicit operation ownership. Then use print_setup query=prepare."
            if required and not supported
            else "Use print_setup query=prepare, then refresh instructions."
        ),
        "scope": "Explicit additive production requires current plate-aligned files; physical print success remains unverified.",
    }


def preflight_context(review: dict[str, Any]) -> dict[str, Any]:
    """Present existing finish findings without adding a second closure authority."""
    return {
        "revision_id": review.get("revision_id"),
        "plan_hash": review.get("plan_hash"),
        "work_reasons": list(review.get("work_reasons", [])),
        "print_preparation": review.get("print_preparation"),
        "scope": "Current saved finish obligations; scheduling guidance only, never approval.",
    }
