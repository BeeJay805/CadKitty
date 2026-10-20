"""Host-owned consumption history; historical interpretation grants no new proof."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from forge_mind.workspace.document import digest
from forge_mind.workspace.edit_contracts import EditBasis

FIELD = "intent_lineage"
KEYS = ("input_id", "source_count", "source_hash", "contract_hash")


def _tuple(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Consumed input tuple is invalid")
    result = {key: value[key] for key in KEYS}
    if type(result["source_count"]) is not int or result["source_count"] < 0:
        raise ValueError("Consumed source count is invalid")
    return result


def validated(
    document: dict[str, Any], application: dict[str, Any], intent: dict[str, Any]
) -> dict[str, Any] | None:
    """Check an existing receipt against exact saved sources, never infer a receipt."""
    from forge_mind.workspace.input_application import contract_hash, source

    receipt = application.get(FIELD)
    if receipt is None:
        return None
    if application.get("pending") is not None:
        raise ValueError("Input lineage has an unresolved input")
    if not isinstance(receipt, dict) or set(receipt) != {
        "version",
        "session_id",
        "intent_hash",
        "root",
        "steps",
    }:
        raise ValueError("Input lineage receipt is invalid")
    if (
        type(receipt["version"]) is not int
        or receipt["version"] != 1
        or receipt["session_id"] != document["id"]
        or receipt["intent_hash"] != digest(intent)
        or intent.get("status") not in {"interpreted", "answered"}
    ):
        raise ValueError("Input lineage interpretation changed")
    previous = _tuple(receipt["root"])
    if (
        receipt["root"] != previous
        or source(document, previous["source_count"])["source_hash"] != previous["source_hash"]
        or previous["source_hash"] != intent["source_hash"]
        or previous["contract_hash"] != intent["contract_hash"]
    ):
        raise ValueError("Input lineage original scope changed")
    steps = receipt["steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("Input lineage consumption is unavailable")
    predecessor = digest({key: value for key, value in receipt.items() if key != "steps"})
    identities: set[str] = set()
    for step in steps:
        if not isinstance(step, dict) or set(step) != {
            "before",
            "after",
            "job_id",
            "origin",
            "edit_basis",
            "predecessor_hash",
            "hash",
        }:
            raise ValueError("Input lineage transition is invalid")
        after = _tuple(step["after"])
        basis = EditBasis.model_validate(step["edit_basis"]).model_dump()
        if (
            step["before"] != previous
            or step["after"] != after
            or after["source_count"] != previous["source_count"] + 1
            or source(document, after["source_count"])["source_hash"] != after["source_hash"]
            or not isinstance(after["input_id"], str)
            or not after["input_id"]
            or after["input_id"] in identities
            or not isinstance(step["job_id"], str)
            or not step["job_id"]
            or step["origin"] not in {"existing", "inherited"}
            or basis != step["edit_basis"]
            or basis.get("contract_hash") != previous["contract_hash"]
            or step["predecessor_hash"] != predecessor
            or step["hash"] != digest({key: value for key, value in step.items() if key != "hash"})
        ):
            raise ValueError("Input lineage transition changed")
        identities.add(after["input_id"])
        previous, predecessor = after, step["hash"]
    if (
        previous != _tuple(application.get("applied"))
        or source(document) != {key: previous[key] for key in ("source_count", "source_hash")}
        or contract_hash(document.get("contract")) != previous["contract_hash"]
    ):
        raise ValueError("Input lineage current target changed")
    return receipt


def consumed(
    document: dict[str, Any],
    application: dict[str, Any],
    pending: dict[str, Any],
    applied: dict[str, Any],
    *,
    job_id: str,
) -> dict[str, Any] | None:
    """Mint only at the atomic host delta-consumption boundary after admission."""
    from forge_mind.workspace.input_application import source
    from forge_mind.workspace.intent_attribution_service import _record

    if pending["mode"] != "delta" or pending["origin"] not in {"existing", "inherited"}:
        return None
    intent = _record(document)
    if intent is None or intent.get("status") not in {"interpreted", "answered"}:
        return None
    if intent["source_hash"] == pending["source_hash"]:
        return None  # A new complete interpretation needs no historical receipt.
    before = _tuple(application.get("applied"))
    prior = {**document, "instructions": document.get("instructions", [])[: before["source_count"]]}
    # The pending tuple still names the prior Contract, so validation uses that
    # exact prior source and target rather than the newly admitted Contract.
    receipt = validated(prior, {**application, "pending": None}, intent)
    if receipt is None:
        if (
            before["source_hash"] != intent["source_hash"]
            or before["contract_hash"] != intent["contract_hash"]
        ):
            return None  # Unrecorded historical edits cannot acquire retroactive lineage.
        receipt = {
            "version": 1,
            "session_id": document["id"],
            "intent_hash": digest(intent),
            "root": before,
            "steps": [],
        }
    receipt = deepcopy(receipt)
    if (
        source(prior)["source_hash"] != before["source_hash"]
        or pending["source_count"] != before["source_count"] + 1
        or document.get("instructions", [])[-1:] != [pending["instruction"]]
    ):
        raise ValueError("Input lineage admitted source changed")
    step = {
        "before": before,
        "after": _tuple(applied),
        "job_id": job_id,
        "origin": pending["origin"],
        "edit_basis": deepcopy(pending["edit_basis"]),
        "predecessor_hash": receipt["steps"][-1]["hash"]
        if receipt["steps"]
        else digest({key: value for key, value in receipt.items() if key != "steps"}),
    }
    step["hash"] = digest(step)
    receipt["steps"].append(step)
    return receipt
