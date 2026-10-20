"""Pure literal author-history ownership checks; no readers or approval decisions."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from forge_mind.workspace.document import digest


def guidance_history_status(
    history: Any, revision_id: str, record: Any, *, present: bool = True
) -> dict[str, Any]:
    """Check a literal per-revision chain without trusting its reported status."""
    result: dict[str, Any] = {"status": "verified", "head_hash": None, "errors": []}
    if not present:
        result["status"] = "legacy_unverified" if record is not None else "unavailable"
        return result
    if not isinstance(history, list) or not history:
        return {
            **result,
            "status": "unverified",
            "errors": ["Guidance history is not a nonempty list"],
        }
    previous = None
    for version, entry in enumerate(history, 1):
        try:
            changed = (
                not isinstance(entry, dict)
                or set(entry)
                != {"version", "predecessor_hash", "revision_id", "record", "record_hash"}
                or type(entry.get("version")) is not int
                or entry["version"] != version
                or entry.get("revision_id") != revision_id
                or entry.get("predecessor_hash") != previous
                or entry.get("record_hash") != digest(entry.get("record"))
            )
            entry_hash = None if changed else digest(entry)
        except (TypeError, ValueError, OverflowError):
            changed, entry_hash = True, None
        if changed:
            return {
                **result,
                "status": "unverified",
                "errors": [f"Guidance history entry {version} identity or hash changed"],
            }
        previous = entry_hash
    try:
        head_changed = digest(history[-1]["record"]) != digest(record)
    except (TypeError, ValueError, OverflowError):
        return {
            **result,
            "status": "unverified",
            "errors": ["Guidance history head or current record is not valid JSON"],
        }
    if head_changed:
        return {
            **result,
            "status": "unverified",
            "errors": ["Guidance history head differs from current record"],
        }
    result["head_hash"] = previous
    return result


def guidance_record_revision(record: Any) -> str | None:
    """Return an explicit saved origin only when both identity fields exist."""
    basis = record.get("basis") if isinstance(record, dict) else None
    if not isinstance(basis, dict):
        return None
    if not all(
        isinstance(basis.get(key), str) and re.fullmatch(r"[a-f0-9]{32}", basis[key])
        for key in ("session_id", "revision_id")
    ):
        return None
    revision_id = basis["revision_id"]
    return revision_id if isinstance(revision_id, str) else None


def _present(history: Any, status: Any) -> bool:
    return not (
        history == []
        and isinstance(status, dict)
        and status.get("status") in {"legacy_unverified", "unavailable"}
    )


def _tracked(record: Any) -> bool:
    return isinstance(record, dict) and bool(
        {"instruction_plan", "instruction_plan_hash", "input_authority_hash"}.intersection(record)
    )


def guidance_record_provenance(
    context: dict[str, Any],
    *,
    source: Any,
    origin_history: Any = None,
    origin_history_present: bool = False,
) -> dict[str, Any] | None:
    """Recompute displayed-record origin and its literal chain independently of the target."""
    record = context.get("record")
    if record is None:
        return None
    basis = context["basis"]
    target = basis["revision_id"]
    origin = guidance_record_revision(record)
    same_origin = origin == target
    history = context["history"] if same_origin else origin_history
    present = (
        _present(history, context["history_status"]) if same_origin else origin_history_present
    )
    errors = []
    status: dict[str, Any]
    if origin is None:
        status = {
            "status": "unverified" if _tracked(record) or present else "legacy_unverified",
            "head_hash": None,
            "errors": ["Author-record origin is unavailable; no session or revision was inferred"],
        }
    else:
        status = guidance_history_status(history, origin, record, present=present)
        if record["basis"]["session_id"] != basis["session_id"]:
            errors.append("Author-record origin belongs to another session")
        try:
            if record.get("basis_hash") != digest(record["basis"]):
                errors.append("Author-record origin basis hash changed")
        except (TypeError, ValueError, OverflowError):
            errors.append("Author-record origin basis is invalid")
    if source == "target":
        if origin is not None and not same_origin:
            errors.append("Requested revision record differs from its map key")
    elif source == "selected_fallback":
        if (
            context.get("target_kind") != "selected_candidate"
            or context.get("selected_revision") != target
            or same_origin
        ):
            errors.append("Carried author record is not a selected-revision fallback")
    else:
        errors.append("Author-record selection source is invalid")
    try:
        record_hash = digest(record)
    except (TypeError, ValueError, OverflowError):
        record_hash = None
        errors.append("Author record cannot be hashed")
    if errors:
        status = {**status, "status": "unverified", "errors": [*status["errors"], *errors]}
    return {
        "source": source,
        "revision_id": origin,
        "record_hash": record_hash,
        "history": {"ref": "history"} if same_origin else deepcopy(history),
        "history_status": status,
    }


def verify_guidance_history_context(context: dict[str, Any]) -> dict[str, Any]:
    """Verify both ownership coordinates; return the displayed record's history status."""
    record: Any = context.get("record")
    history, reported = context["history"], context["history_status"]
    if not isinstance(reported, dict):
        raise ValueError("Customer guidance history status is invalid")
    provenance = context.get("record_provenance")
    has_provenance = "record_provenance" in context
    if not has_provenance and _tracked(record):
        origin = guidance_record_revision(record)
        record_basis = record.get("basis")
        try:
            same_target = (
                origin is not None
                and origin == context["basis"]["revision_id"]
                and isinstance(record_basis, dict)
                and record_basis.get("session_id") == context["basis"]["session_id"]
                and record.get("basis_hash") == digest(record_basis)
            )
        except (TypeError, ValueError, OverflowError):
            same_target = False
        if not same_target:
            raise ValueError("Customer guidance cross-origin authored record requires provenance")
    target_record = record
    if (
        has_provenance
        and isinstance(provenance, dict)
        and provenance.get("source") == "selected_fallback"
    ):
        target_record = None  # Never infer a missing target record from an orphan history head.
    verified = guidance_history_status(
        history,
        context["basis"]["revision_id"],
        target_record,
        present=_present(history, reported),
    )
    if verified != reported or verified["status"] == "unverified":
        raise ValueError("Customer guidance history integrity changed")
    record_status: dict[str, Any] = verified
    if has_provenance:
        if record is None:
            if provenance is not None:
                raise ValueError("Customer guidance record provenance has no author record")
        else:
            if not isinstance(provenance, dict) or set(provenance) != {
                "source",
                "revision_id",
                "record_hash",
                "history",
                "history_status",
            }:
                raise ValueError("Customer guidance record provenance is incomplete")
            origin_status = provenance["history_status"]
            expected = guidance_record_provenance(
                context,
                source=provenance["source"],
                origin_history=provenance["history"],
                origin_history_present=_present(provenance["history"], origin_status),
            )
            if (
                expected is None
                or expected != provenance
                or expected["history_status"]["status"] == "unverified"
            ):
                raise ValueError("Customer guidance record provenance integrity changed")
            record_status = expected["history_status"]
            if context.get("status") == "current" and (
                provenance["source"] != "target"
                or provenance["revision_id"] != context["basis"]["revision_id"]
            ):
                raise ValueError("Customer guidance current record differs from its target")
    if _tracked(record) and record_status["status"] != "verified":
        raise ValueError("Customer guidance authored history is missing")
    return record_status
