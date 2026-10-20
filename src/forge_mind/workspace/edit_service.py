"""One guarded service path for focused author and connector edits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.feature_delta import FeatureDelta, apply_feature_delta
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import EditConflictError, WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def read_edit_basis(service: WorkspaceService, sid: str, owner: str) -> dict[str, Any]:
    basis: EditBasis = service.store.edit_basis(sid, owner)
    return basis.model_dump()


def check_edit_basis(service: WorkspaceService, sid: str, owner: str, basis: EditBasis) -> None:
    if service.store.edit_basis(sid, owner) != basis:
        raise EditConflictError("Edit basis changed; read the current edit basis before retrying")


def check_author_facts(patch: PlanPatch, author_intent: str) -> None:
    """Shared provenance boundary for authored plan patches and construction."""
    for fact in patch.dimensions:
        if fact.source == "measured_reference":
            raise WorkspaceError("A plan patch cannot assert a new measured reference")
        if fact.source == "customer" and fact.source_quote not in author_intent:
            raise WorkspaceError("Customer fact quote must occur in the current request history")


def patch_plan(
    service: WorkspaceService,
    sid: str,
    owner: str,
    patch: PlanPatch,
    jid: str | None = None,
    *,
    author_intent: str | None = None,
) -> dict[str, Any]:
    check_edit_basis(service, sid, owner, patch.basis)
    session = service.store.get(sid, owner)
    if jid is not None:
        check_author_facts(patch, author_intent or session["prompt"])
    if not session.get("design_plan"):
        raise WorkspaceError("Initialize a design plan before applying a plan patch")
    document = None
    if patch.basis.revision_id:
        revision = service._load_revision(sid, patch.basis.revision_id)
        document = DesignDocument.model_validate(revision["document"])
    effective = (session.get("proposal") or {}).get("contract") or session.get("contract") or {}
    contract = Contract.model_validate(effective)
    captured = service.store._edit_target_snapshot(session, contract.fingerprint())
    current_plan = (
        captured["plan"]
        if captured and captured.get("plan_hash") == patch.basis.plan_hash
        else session["design_plan"]
    )
    plan = apply_plan_patch(
        DesignPlan.model_validate(current_plan),
        patch,
        contract,
        document,
        previous_raw=current_plan,
    )
    args: dict[str, Any] = {
        "expected_plan_version": patch.basis.plan_version,
        "contract": contract,
        "expected_current_revision": patch.basis.revision_id,
        "expected_input_version": patch.basis.input_version,
        "expected_edit_basis": patch.basis,
    }
    binding = (
        service.store.set_design_plan_for_job(sid, owner, jid, plan, **args)
        if jid
        else service.store.set_design_plan(sid, owner, plan, **args)
    )
    return {
        "action": "plan_patch",
        "applied": True,
        "plan_applied": True,
        "cad_applied": False,
        "revision_id": patch.basis.revision_id,
        "plan_version": binding["plan_version"],
        "plan_hash": binding["plan_hash"],
        "edit_basis": read_edit_basis(service, sid, owner),
        "scope": "Intent changed; no geometry edit or prior rejected edit was applied",
    }


def enqueue_feature_edit(
    service: WorkspaceService,
    sid: str,
    owner: str,
    delta: FeatureDelta,
    max_seconds: int | None = None,
) -> dict[str, Any]:
    service.store.get(sid, owner)
    if service.executor.mode != "trusted_local" and not service.executor.typed_container_verified:
        raise WorkspaceError("Typed container is not qualified; no generation job was started")
    check_edit_basis(service, sid, owner, delta.basis)
    if delta.basis.revision_id is None:
        raise WorkspaceError("Build a typed document before a feature edit")
    revision = service._load_revision(sid, delta.basis.revision_id)
    document = apply_feature_delta(DesignDocument.model_validate(revision["document"]), delta)
    jid = service.store.enqueue(
        sid,
        owner,
        {
            "document": document.model_dump(),
            "revision_id": delta.basis.revision_id,
            "expected_current_revision": delta.basis.revision_id,
            "expected_input_version": delta.basis.input_version,
            "feature_edit_basis": delta.basis.model_dump(),
            "max_seconds": max_seconds,
            "preview_only": True,
            "explanation": "Apply an explicit revision-bound feature delta",
        },
    )
    service.start()
    return {
        "id": sid,
        "job_id": jid,
        "state": "queued",
        "applied": False,
        "base_revision_id": delta.basis.revision_id,
        "scope": "Admitted for construction; read the terminal edit result before claiming application",
    }
