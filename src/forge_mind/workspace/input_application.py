"""Server-owned input consumption, separate from canonical Contract serialization."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from forge_mind.workspace.document import digest
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3

    from forge_mind.workspace.store import Store

FIELD = "contract_application"
CONTINUATION = "admitted_input_continuation"
SERVER_FIELDS = {
    "customer_construction_binding",
    FIELD,
    CONTINUATION,
    "selection_version",
    "input_application_id",
    "input_instructions",
    "input_origin",
    "intent_answer_binding",
    "baseline_contract_application",
    "resume_contract_application",
    "target_contract_application",
    "target_instructions",
}


def source(document: dict[str, Any], count: int | None = None) -> dict[str, Any]:
    instructions = document.get("instructions", [])
    count = len(instructions) if count is None else count
    if type(count) is not int or not 0 <= count <= len(instructions):
        raise WorkspaceError("Input application source prefix is invalid")
    text = str(document["prompt"]) + "".join(
        "\nUser revision instruction: " + instruction for instruction in instructions[:count]
    )
    return {"source_count": count, "source_hash": digest(text)}


def contract_hash(contract: dict[str, Any] | None) -> str | None:
    return Contract.model_validate(contract).fingerprint() if contract is not None else None


def baseline(document: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap a known steady snapshot; interrupted legacy state needs another guard."""
    existing = document.get(FIELD)
    if isinstance(existing, dict) and existing.get("pending") is None:
        previous = existing.get("applied")
        if (
            isinstance(previous, dict)
            and type(previous.get("source_count")) is int
            and source(document)
            == {key: previous.get(key) for key in ("source_count", "source_hash")}
            and previous.get("contract_hash") == contract_hash(document.get("contract"))
            and document.get("contract") is not None
        ):
            # Construction/plan publication may change geometry without changing
            # the consumed owner source or target. Keep its exact host receipt.
            return deepcopy(existing)
    applied = (
        {
            "input_id": None,
            **source(document),
            "contract_hash": contract_hash(document["contract"]),
        }
        if document.get("contract") is not None
        else None
    )
    return {"applied": applied, "pending": None}


def pending(document: dict[str, Any]) -> dict[str, Any] | None:
    application = document.get(FIELD) or {}
    return application.get("pending")


def authoring(payload: dict[str, Any]) -> bool:
    return not any(
        (
            payload.get("document") is not None,
            payload.get("engineering_operation"),
            payload.get("parameters"),
            payload.get("restore_only"),
        )
    )


def continuation(
    document: dict[str, Any], snapshot: dict[str, Any], location: str
) -> dict[str, Any]:
    """Transfer only exact aliases of the selected recovery target, never an ID alone."""

    def identity(value: dict[str, Any], *, current: bool = False) -> dict[str, Any]:
        instructions = value.get("instructions", [])
        if not isinstance(instructions, list) or any(
            not isinstance(item, str) for item in instructions
        ):
            raise WorkspaceError("Continuation owner history is invalid")
        application = value.get(FIELD)
        item = pending(value)
        basis = (item or {}).get("edit_basis") or {}
        plan = value.get("plan_snapshot") or value.get("operation_plan_snapshot")
        return {
            "revision_id": value.get("selected_revision") if current else value.get("revision_id"),
            "based_on": value.get("based_on"),
            "contract_hash": contract_hash(value.get("contract")),
            "instructions": instructions,
            "source": source({"prompt": document["prompt"], "instructions": instructions}),
            FIELD: application,
            "plan_snapshot": plan,
            "input_version": basis.get("input_version") if current else value.get("input_version"),
            "source_fingerprint": basis.get("source_fingerprint")
            if current
            else value.get("source_fingerprint"),
        }

    selected = identity(snapshot)
    current = identity(document, current=True)
    selected_application = selected[FIELD] or {}
    selected_input = (
        selected_application.get("pending") or selected_application.get("applied") or {}
    )
    current_applied = (document.get(FIELD) or {}).get("applied") or {}
    if (
        selected_input.get("input_id") is not None
        and current_applied.get("input_id") == selected_input["input_id"]
        and any(
            current[key] != selected[key]
            for key in ("contract_hash", "instructions", "source", FIELD)
        )
    ):
        raise WorkspaceError("Continuation owner input differs from its current applied tuple")
    predecessors = {}
    for name, value in (
        ("current", document),
        ("proposal", document.get("proposal")),
        ("resume_candidate", document.get("resume_candidate")),
    ):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise WorkspaceError("Continuation branch is invalid")
        actual = current if name == "current" else identity(value)
        expected = selected
        if name == "current":
            # The session has no candidate based_on field; its target and full
            # application/source/plan tuple still must match if it is pending.
            actual = {key: item for key, item in actual.items() if key != "based_on"}
            expected = {key: item for key, item in selected.items() if key != "based_on"}
        matches = actual == expected
        if pending(value) is not None and not matches:
            raise WorkspaceError("A competing pending owner input target must be resolved first")
        if name != "current" and matches:
            predecessors[name] = deepcopy(value)
    if location not in predecessors or predecessors[location] != snapshot:
        raise WorkspaceError("Continuation target changed before admission")
    return {
        "version": 1,
        "selected_revision": document.get("selected_revision"),
        "selection_version": document.get("selection_version", 0),
        "plan_version": document.get("plan_version", 0),
        "predecessors": predecessors,
        "target": deepcopy(snapshot),
    }


def stage(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    jid: str,
    document: dict[str, Any],
    payload: dict[str, Any],
    *,
    origin: str,
    answer_binding: dict[str, Any] | None,
    basis_document: dict[str, Any] | None = None,
) -> None:
    """Stage exactly once in enqueue's transaction, after existing admission checks."""
    application = deepcopy(document.get(FIELD) or baseline(document))
    previous = application.get("pending")
    instruction = payload.get("instruction")
    applied = application.get("applied")
    if previous is None and applied is not None:
        if (
            source(document, applied["source_count"])["source_hash"] != applied["source_hash"]
            or contract_hash(document.get("contract")) != applied["contract_hash"]
        ):
            raise WorkspaceError("Applied owner input snapshot changed")
        if (
            applied["source_count"] != len(document.get("instructions", []))
            and origin != "supplied"
        ):
            raise WorkspaceError("Select the saved candidate before extending its owner history")
    if previous is not None:
        if answer_binding is None and not payload.get("resume"):
            raise WorkspaceError("Resume the pending owner input before starting another edit")
        if source(document) != {key: previous[key] for key in ("source_count", "source_hash")}:
            raise WorkspaceError("Pending owner input source changed")
        if contract_hash(document.get("contract")) != previous["base_contract_hash"]:
            raise WorkspaceError("Pending owner input target changed")
        store._compare_edit_basis(con, sid, document, previous["edit_basis"], rid=None)
        if answer_binding is not None and previous["mode"] != "full":
            raise WorkspaceError("Resume the pending requirement edit before another owner answer")
    elif payload.get("resume") and instruction:
        raise WorkspaceError("Resume cannot include a competing owner instruction")

    if instruction:
        document["instructions"] = [*document.get("instructions", []), instruction]
    needs_application = previous is not None or bool(instruction) or origin == "supplied"
    needs_application = needs_application or document.get("contract") is None
    if needs_application:
        mode = (
            previous["mode"]
            if previous is not None
            else "supplied"
            if origin == "supplied"
            else "full"
            if document.get("contract") is None
            else "delta"
        )
        answers = list((previous or {}).get("owner_answers", []))
        if answer_binding is not None:
            answers.append(deepcopy(answer_binding))
        application["pending"] = {
            "input_id": previous["input_id"] if previous else uuid4().hex,
            **source(document),
            "base_contract_hash": contract_hash(document.get("contract")),
            "mode": mode,
            "origin": previous["origin"] if previous else origin,
            "instruction": (previous or {}).get("instruction", instruction or ""),
            "owner_answers": answers,
            "edit_basis": store._edit_basis(
                con, sid, document if basis_document is None else basis_document, rid=None
            ).model_dump(),
        }
        if mode == "supplied":
            original_job = (previous or {}).get("supplied_job_id", jid)
            row = con.execute(
                "SELECT payload FROM jobs WHERE id=? AND session=?", (original_job, sid)
            ).fetchone()
            if row is None:
                raise WorkspaceError("Supplied owner target is unavailable")
            provided = json.loads(row[0]).get("contract")
            provided_hash = contract_hash(provided)
            if provided is None or (
                previous is not None and previous.get("supplied_contract_hash") != provided_hash
            ):
                raise WorkspaceError("Supplied owner target changed")
            application["pending"].update(
                supplied_job_id=original_job, supplied_contract_hash=provided_hash
            )
            payload["contract"] = provided
        payload["input_application_id"] = application["pending"]["input_id"]
    elif payload.get("resume") and (application.get("applied") or {}).get("input_id"):
        payload["input_application_id"] = application["applied"]["input_id"]
    document[FIELD] = application
    payload["input_instructions"] = list(document.get("instructions", []))
    payload["input_origin"] = origin


def validate_pending(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    document: dict[str, Any],
    input_id: str,
) -> dict[str, Any]:
    item = pending(document)
    if item is None or item["input_id"] != input_id:
        raise WorkspaceError("Pending owner input identity changed")
    if source(document) != {key: item[key] for key in ("source_count", "source_hash")}:
        raise WorkspaceError("Pending owner input source changed")
    if contract_hash(document.get("contract")) != item["base_contract_hash"]:
        raise WorkspaceError("Pending owner input target changed")
    store._compare_edit_basis(con, sid, document, item["edit_basis"], rid=None)
    return item


def apply(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    document: dict[str, Any],
    input_id: str,
    contract: Contract,
) -> None:
    """Caller owns the transaction and current-worker check; publish both or neither."""
    item = validate_pending(store, con, sid, document, input_id)
    application = deepcopy(document[FIELD])
    application["applied"] = {
        "input_id": item["input_id"],
        "source_count": item["source_count"],
        "source_hash": item["source_hash"],
        "contract_hash": contract.fingerprint(),
        "owner_answers": item["owner_answers"],
    }
    from forge_mind.workspace.input_lineage import FIELD as LINEAGE
    from forge_mind.workspace.input_lineage import consumed

    if item["mode"] == "delta":
        job_id = document.get("job_id")
        row = con.execute(
            "SELECT payload FROM jobs WHERE id=? AND session=?", (job_id, sid)
        ).fetchone()
        payload = json.loads(row[0]) if row is not None else {}
        inherited = (payload.get(CONTINUATION) or {}).get("target") or {}
        origin_matches = payload.get("input_origin") == item["origin"] == "existing"
        if payload.get("input_origin") == "inherited":
            origin_matches = (
                inherited.get(FIELD) == document[FIELD]
                and inherited.get("instructions") == document.get("instructions", [])
                and contract_hash(inherited.get("contract")) == item["base_contract_hash"]
            )
        if (
            not isinstance(job_id, str)
            or payload.get("input_application_id") != input_id
            or not origin_matches
            or payload.get("input_instructions") != document.get("instructions", [])
        ):
            raise WorkspaceError("Consumed owner input admission changed")
        lineage = consumed(document, document[FIELD], item, application["applied"], job_id=job_id)
        if lineage is not None:
            application[LINEAGE] = lineage
        else:
            application.pop(LINEAGE, None)
    else:
        application.pop(LINEAGE, None)
    application["pending"] = None
    document.update(contract=contract.model_dump(), **{FIELD: application})


def copy_to_proposal(document: dict[str, Any], proposal: dict[str, Any]) -> None:
    """Attach only the application state for this exact candidate target/history."""
    if proposal.get("contract") != document.get("contract") or proposal.get(
        "instructions", []
    ) != document.get("instructions", []):
        return
    proposal[FIELD] = deepcopy(document.get(FIELD) or baseline(document))
