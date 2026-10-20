"""Source-bound interpretation records and atomic owner clarification admission.

These records describe an interpretation or a question. They grant no approval,
measurement, or permission to relax a frozen requirement.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from forge_mind.workspace.document import digest
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.intent_attribution import (
    IntentAttribution,
    IntentIssue,
    IntentResolution,
    validate_attribution,
)
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store, WorkspaceError

SCOPE = (
    "Source-bound interpretation and owner-supplied answers only; "
    "not owner approval, measured evidence, or verified subject attribution."
)


def request_text(session: dict[str, Any]) -> str:
    """Use the same immutable request history as extraction and authoring."""
    return str(session["prompt"]) + "".join(
        "\nUser revision instruction: " + instruction
        for instruction in session.get("instructions", [])
    )


def _record(document: dict[str, Any]) -> dict[str, Any] | None:
    record = document.get("intent_attribution")
    if record is None:
        return None
    if not isinstance(record, dict) or record.get("draft_hash") != digest(
        {key: value for key, value in record.items() if key != "draft_hash"}
    ):
        raise WorkspaceError("Intent attribution record identity is invalid")
    if record.get("source_hash") != digest(record.get("source_text")):
        raise WorkspaceError("Intent attribution source identity is invalid")
    target = Contract.model_validate(record["proposed_contract"])
    if record.get("contract_hash") != target.fingerprint():
        raise WorkspaceError("Intent attribution contract identity is invalid")
    return record


def _replace_record(document: dict[str, Any], values: dict[str, Any]) -> dict[str, Any]:
    previous = _record(document)

    def substantive(record: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in record.items() if key not in {"version", "draft_hash"}}

    if previous is not None and substantive(previous) == substantive(values):
        return previous
    record = {**values, "version": previous["version"] + 1 if previous else 1}
    record["draft_hash"] = digest(record)
    if previous is not None:
        document.setdefault("intent_attribution_history", []).append(previous)
    document["intent_attribution"] = record
    return record


def _owned_document(con: sqlite3.Connection, sid: str, owner: str) -> dict[str, Any]:
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    document: dict[str, Any] = json.loads(row[0])
    return document


def _write_proposal(
    store: Store,
    sid: str,
    owner: str,
    jid: str,
    attribution: IntentAttribution | list[IntentIssue],
    contract: Contract,
    source_text: str,
    expected_basis: EditBasis | dict[str, Any],
    *,
    conflict: bool,
) -> dict[str, Any]:
    contract = Contract.model_validate(contract.model_dump())
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        document = _owned_document(con, sid, owner)
        store._compare_edit_basis(con, sid, document, expected_basis)
        store._check_plan_writer(con, sid, document, jid, allow_intent_record=True)
        if request_text(document) != source_text:
            raise WorkspaceError("Intent source changed; read the current request history")
        previous = _record(document)
        if conflict:
            retain_previous = (
                previous is not None
                and previous["contract_hash"] == contract.fingerprint()
                and previous["source_hash"] == digest(source_text)
                and previous["status"] != "answered"
            )
            # Historical input facts remain intact. A changed request or contract
            # needs a new interpretation before its old map can be current again.
            retained = IntentAttribution.model_validate(
                previous["attribution"] if previous is not None and retain_previous else {}
            )
            issues = (
                {issue.id: issue for issue in retained.issues}
                if (retain_previous and previous and previous["status"] == "needs_input")
                else {}
            )
            assert isinstance(attribution, list)
            if not attribution:
                raise WorkspaceError("Intent conflict needs at least one issue")
            for raw_issue in attribution:
                issue = IntentIssue.model_validate(raw_issue)
                if not retain_previous and issue.fact_ids:
                    raise WorkspaceError(
                        "Input facts belong to an earlier interpretation; report this conflict "
                        "with current requirement IDs and no fact IDs"
                    )
                if issue.id in issues and issues[issue.id] != issue:
                    raise WorkspaceError("An open intent issue cannot be replaced")
                issues[issue.id] = issue
            resolved = retained.model_copy(update={"issues": list(issues.values())})
        else:
            if previous and previous["status"] == "needs_input":
                raise WorkspaceError(
                    "Owner clarification is required before another interpretation"
                )
            if (
                previous
                and previous["status"] == "answered"
                and previous["source_hash"] == digest(source_text)
            ):
                raise WorkspaceError("Apply the owner answer to the request before interpreting it")
            resolved = IntentAttribution.model_validate(attribution)
        from forge_mind.workspace.customer_interpretation import (
            validate_server_interpretation_proposal,
        )

        try:
            validate_server_interpretation_proposal(document, jid, source_text, contract, resolved)
            from forge_mind.workspace.photo_nominal import validate_photo_source_replay

            validate_photo_source_replay(
                document,
                jid,
                source_text,
                contract,
                resolved,
                require_photo=bool(document.get("customer_reference")) and previous is None,
            )
        except ValueError as exc:
            raise WorkspaceError(str(exc)) from exc
        validate_attribution(resolved, contract, source_text)
        record = _replace_record(
            document,
            {
                "source_hash": digest(source_text),
                "source_text": source_text,
                "proposed_contract": contract.model_dump(),
                "contract_hash": contract.fingerprint(),
                "attribution": resolved.model_dump(),
                "basis": store._edit_basis(con, sid, document).model_dump(),
                "status": "needs_input" if resolved.issues else "interpreted",
                "resolutions": [],
            },
        )
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(document), sid))
    return record


def record_proposal(
    store: Store,
    sid: str,
    owner: str,
    jid: str,
    attribution: IntentAttribution,
    contract: Contract,
    source_text: str,
    expected_basis: EditBasis | dict[str, Any],
) -> dict[str, Any]:
    return _write_proposal(
        store, sid, owner, jid, attribution, contract, source_text, expected_basis, conflict=False
    )


def record_conflict(
    store: Store,
    sid: str,
    owner: str,
    jid: str,
    issues: list[IntentIssue],
    contract: Contract,
    source_text: str,
    expected_basis: EditBasis | dict[str, Any],
) -> dict[str, Any]:
    return _write_proposal(
        store, sid, owner, jid, issues, contract, source_text, expected_basis, conflict=True
    )


def attribution_summary(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    attribution = record["attribution"]
    return {
        "version": record["version"],
        "draft_hash": record["draft_hash"],
        "status": record["status"],
        "questions": [
            {key: issue[key] for key in ("id", "question", "reason")}
            for issue in attribution["issues"]
        ],
        "subject_count": len(attribution["subjects"]),
        "fact_count": len(attribution["facts"]),
        "issue_count": len(attribution["issues"]),
        "scope": SCOPE,
    }


def read_attribution(store: Store, sid: str, owner: str) -> dict[str, Any]:
    return {"record": _record(store.session_record(sid, owner)), "scope": SCOPE}


def apply_resolution(
    document: dict[str, Any],
    payload: dict[str, Any],
    con: sqlite3.Connection,
    sid: str,
    owner: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consume owner text inside enqueue's transaction, before any admission write."""
    record = _record(document)
    if "intent_resolution" not in payload:
        if record is not None and record["status"] == "needs_input":
            raise WorkspaceError("Owner clarification is required before new work")
        return document, payload
    # This helper is not an independently callable owner-answer write surface.
    if not con.in_transaction:
        raise WorkspaceError("Intent resolution requires the job admission transaction")
    owned = con.execute("SELECT 1 FROM sessions WHERE id=? AND owner=?", (sid, owner)).fetchone()
    if owned is None:
        raise WorkspaceError("workspace not found")
    if record is None or record["status"] != "needs_input":
        raise WorkspaceError("No current intent question is awaiting an answer")
    allowed = {"intent_resolution", "max_seconds", "max_actions", "preview_only"}
    if set(payload) - allowed:
        raise WorkspaceError(
            "Intent resolution cannot include a competing instruction or operation"
        )
    resolution = IntentResolution.model_validate(payload["intent_resolution"])
    if (
        resolution.expected_version != record["version"]
        or resolution.draft_hash != record["draft_hash"]
    ):
        raise WorkspaceError("Intent questions changed; read the current version and hash")
    if request_text(document) != record["source_text"]:
        raise WorkspaceError("Intent source changed; the owner answer is stale")
    Store._compare_edit_basis(con, sid, document, record["basis"])
    expected = {issue["id"] for issue in record["attribution"]["issues"]}
    if {answer.issue_id for answer in resolution.answers} != expected:
        raise WorkspaceError("Supply exactly one answer for every current intent issue")
    # The question is model text. Only the owner-provided ID and raw answer enter
    # the new request, so a question cannot smuggle instructions into authoring.
    updated_payload = {key: value for key, value in payload.items() if key != "intent_resolution"}
    updated_payload["instruction"] = "\n".join(
        f"User clarification ({answer.issue_id}): {answer.answer}" for answer in resolution.answers
    )
    updated_payload["reconcile_intent"] = True
    updated_payload["intent_answer_binding"] = {
        "question_version": record["version"],
        "question_draft_hash": record["draft_hash"],
        "source_hash": record["source_hash"],
        "basis": record["basis"],
        "issue_ids": sorted(expected),
        "answers": [answer.model_dump() for answer in resolution.answers],
    }
    prior = con.execute(
        "SELECT payload FROM jobs WHERE id=? AND session=?", (document.get("job_id"), sid)
    ).fetchone()
    if prior is not None:
        prior_payload = json.loads(prior[0])
        for key in ("max_seconds", "max_actions"):
            if updated_payload.get(key) is None and prior_payload.get(key) is not None:
                updated_payload[key] = prior_payload[key]
        if "preview_only" not in updated_payload:
            updated_payload["preview_only"] = prior_payload.get("preview_only", False)
        if prior_payload.get("delivery_requested") is True:
            updated_payload["delivery_requested"] = True
    head = (document.get("proposal") or {}).get("revision_id") or document.get("selected_revision")
    if head is not None:
        updated_payload["revision_id"] = head
    updated_document = dict(document)
    updated_document["intent_attribution_history"] = list(
        document.get("intent_attribution_history", [])
    )
    _replace_record(
        updated_document,
        {
            **{key: value for key, value in record.items() if key not in {"version", "draft_hash"}},
            "status": "answered",
            "resolutions": [answer.model_dump() for answer in resolution.answers],
        },
    )
    return updated_document, updated_payload
