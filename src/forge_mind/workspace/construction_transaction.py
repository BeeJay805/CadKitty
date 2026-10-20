"""Explicit authored plan/CAD staging and one guarded publication, never acceptance."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.edit_service import check_author_facts
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.operation_plan import coverage, snapshot, validate_plan
from forge_mind.workspace.plan_history import request_hash
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import EditConflictError, WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


@dataclass(frozen=True)
class PreparedConstruction:
    basis: EditBasis
    document: DesignDocument
    patch: PlanPatch
    base_snapshot: dict[str, Any]
    successor_snapshot: dict[str, Any]
    author_intent: str
    authority: dict[str, Any]
    selection_version: int
    acceptance: dict[str, Any]
    request_hash: str


def construction_hash(document: DesignDocument, patch: PlanPatch, author_intent: str) -> str:
    return request_hash(
        {
            "document": document.model_dump(mode="json"),
            "patch": patch.model_dump(mode="json", exclude_unset=True),
            "author_intent": author_intent,
        }
    )


def validated_successor(prepared: PreparedConstruction) -> dict[str, Any]:
    """Recompute staged intent; mutable Python objects do not authorize themselves."""
    if (
        construction_hash(prepared.document, prepared.patch, prepared.author_intent)
        != prepared.request_hash
    ):
        raise WorkspaceError("Construction request changed after preparation")
    if prepared.patch.basis != prepared.basis:
        raise WorkspaceError("Construction patch basis differs from its original source")
    contract = Contract.model_validate(prepared.base_snapshot["contract"])
    if (
        prepared.base_snapshot["plan_hash"] != prepared.basis.plan_hash
        or contract.fingerprint() != prepared.basis.contract_hash
    ):
        raise WorkspaceError("Construction base snapshot differs from its edit basis")
    check_author_facts(prepared.patch, prepared.author_intent)
    original = DesignPlan.model_validate(prepared.base_snapshot["plan"])
    result = apply_plan_patch(
        original, prepared.patch, contract, previous_raw=prepared.base_snapshot["plan"]
    )
    validate_plan(
        result, contract, previous=prepared.base_snapshot["plan"], document=prepared.document
    )
    successor = snapshot(result, prepared.basis.plan_version + 1, contract)
    if successor != prepared.successor_snapshot:
        raise WorkspaceError("Construction successor snapshot changed after preparation")
    return successor


def publication_matches(proof: dict[str, Any], candidate: dict[str, Any], basis: EditBasis) -> bool:
    """Validate a durable successor separately from legacy same-plan publication."""
    try:
        patch = PlanPatch.model_validate(proof["patch"])
        document = DesignDocument.model_validate(candidate["document"])
        base = proof["base_snapshot"]
        target = proof["successor_snapshot"]
        if patch.basis != basis or base["plan_hash"] != basis.plan_hash:
            return False
        contract = Contract.model_validate(base["contract"])
        if contract.fingerprint() != basis.contract_hash:
            return False
        if base != {
            "plan": base["plan"],
            # A restored snapshot keeps its historical version. The EditBasis
            # separately binds the live write counter used by the successor.
            "plan_version": base["plan_version"],
            "plan_hash": request_hash(base["plan"]),
            "contract": contract.model_dump(mode="json"),
            "contract_hash": contract.fingerprint(),
        }:
            return False
        check_author_facts(patch, proof["author_intent"])
        revised = apply_plan_patch(
            DesignPlan.model_validate(base["plan"]),
            patch,
            contract,
            previous_raw=base["plan"],
        )
        validate_plan(revised, contract, previous=base["plan"], document=document)
        return bool(
            construction_hash(document, patch, proof["author_intent"]) == proof["request_hash"]
            and snapshot(revised, basis.plan_version + 1, contract) == target
            and (candidate.get("operation_record") or {}).get("plan_snapshot") == target
            and (candidate.get("operation_record") or {}).get("plan_hash") == target["plan_hash"]
        )
    except (ValueError, KeyError, TypeError):
        return False


def prepare_construction(
    service: WorkspaceService,
    sid: str,
    owner: str,
    jid: str,
    document: DesignDocument,
    patch: PlanPatch,
    *,
    author_intent: str,
) -> PreparedConstruction:
    patch = PlanPatch.model_validate(patch.model_dump(exclude_unset=True))
    document = DesignDocument.model_validate(document.model_dump())
    with service.store.connect() as con:
        con.execute("BEGIN")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            raise WorkspaceError("workspace not found")
        state = json.loads(row[0])
        if state.get("authoring_mode") != "native_operations":
            raise WorkspaceError("Authored construction requires native operations")
        service.store._check_plan_writer(con, sid, state, jid)
        service.store._compare_edit_basis(con, sid, state, patch.basis)
        if not state.get("design_plan"):
            raise WorkspaceError("Initialize a plan before authored construction")
        effective = (state.get("proposal") or {}).get("contract") or state.get("contract") or {}
        contract = Contract.model_validate(effective)
        base = service.store._edit_target_snapshot(state, contract.fingerprint())
        if base is None or base.get("plan_hash") != patch.basis.plan_hash:
            base = snapshot(state["design_plan"], patch.basis.plan_version, contract)
            # Preserve the actual stored predecessor, including accepted JSON
            # number spelling. Only the new successor uses typed serialization.
            base["plan"] = deepcopy(state["design_plan"])
            base["plan_hash"] = request_hash(base["plan"])
        check_author_facts(patch, author_intent)
        proposed = apply_plan_patch(
            DesignPlan.model_validate(base["plan"]),
            patch,
            contract,
            previous_raw=base["plan"],
        )
        validate_plan(proposed, contract, previous=base["plan"], document=document)
        if patch.basis.revision_id:
            from forge_mind.workspace.assembly import preserved_assembly

            original = con.execute(
                "SELECT document FROM revisions WHERE session=? AND id=?",
                (sid, patch.basis.revision_id),
            ).fetchone()
            source = json.loads(original[0]) if original else {}
            retained, reason = preserved_assembly(
                source.get("document") or {}, document.model_dump()
            )
            if not retained:
                raise WorkspaceError(reason)
        prepared = PreparedConstruction(
            basis=patch.basis,
            document=document,
            patch=patch,
            base_snapshot=deepcopy(base),
            successor_snapshot=snapshot(proposed, patch.basis.plan_version + 1, contract),
            author_intent=author_intent,
            authority=service.store._native_author_binding(state),
            selection_version=state.get("selection_version", 0),
            acceptance=service.store._preview_acceptance_identity(state),
            request_hash=construction_hash(document, patch, author_intent),
        )
    validated_successor(prepared)
    return prepared


def build_construction(
    service: WorkspaceService,
    sid: str,
    owner: str,
    jid: str,
    prepared: PreparedConstruction,
    *,
    explanation: str,
    timeout: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    from forge_mind.workspace.document_service import build_document, protection_results
    from forge_mind.workspace.engineering_models import EngineeringInputs

    successor = validated_successor(prepared)
    from forge_mind.workspace.customer_construction import stage_construction

    stage_construction(
        service,
        sid,
        owner,
        jid,
        prepared.document,
        Contract.model_validate(successor["contract"]),
        prepared.basis.revision_id,
        explanation,
        prepared=prepared,
    )
    if cancelled():
        raise WorkspaceError("Construction cancelled before building")
    if service.store.edit_basis(sid, owner) != prepared.basis:
        raise EditConflictError("Edit basis changed before construction")
    revision = build_document(
        service,
        sid,
        owner,
        prepared.document,
        Contract.model_validate(successor["contract"]),
        prepared.basis.revision_id,
        explanation,
        timeout,
        cancelled,
        plan_snapshot=successor,
    )
    if cancelled():
        raise WorkspaceError("Construction cancelled before publication")
    if not revision["measurement"]["geometry_valid"]:
        return {
            "action": "construction",
            "applied": False,
            "plan_applied": False,
            "cad_applied": False,
            "base_revision_id": prepared.basis.revision_id,
            "candidate_revision_id": revision["id"],
            "revision_id": prepared.basis.revision_id,
            "code": "invalid_geometry",
            "plan_hash": prepared.basis.plan_hash,
            "plan_version": prepared.basis.plan_version,
            "edit_basis": service.store.edit_basis(sid, owner).model_dump(),
            "measurement": revision["measurement"],
        }
    state = service.store.get(sid, owner)
    inputs = EngineeringInputs.model_validate(state["engineering"]["inputs"])
    if any(
        check["status"] != "pass" for check in protection_results(service, sid, revision, inputs)
    ):
        raise WorkspaceError("Construction changed protected geometry")
    if cancelled():
        raise WorkspaceError("Construction cancelled before publication")
    receipt = service.store.publish_construction(sid, owner, jid, prepared, revision["id"])
    receipt["edit_basis"] = service.store.edit_basis(sid, owner).model_dump()
    receipt["coverage"] = coverage(successor, prepared.document)
    receipt["measurement"] = revision["measurement"]
    return receipt
