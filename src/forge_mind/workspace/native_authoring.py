"""Compact native author transport; production models still validate every action."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer

from forge_mind.workspace.construction_reference import OPERATIONS
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.intent_capabilities import preparation_capability
from forge_mind.workspace.models import Action, Contract, DesignPlan
from forge_mind.workspace.plan_patch import PlanPatch

INSPECTION_QUERIES = (
    "",
    "evidence",
    "source",
    "diff",
    "topology",
    "requirements",
    "prepare_handoff",
)


class NativeAction(Action):
    """Keep every action available without embedding the full plan output schema."""

    plan: None = None
    plan_patch_json: str = Field(
        default="",
        description="Optional complete PlanPatch JSON for document, assembly_edit or feature_edit only. Its original EditBasis must match the CAD edit. Plan and CAD publish together or neither publishes.",
    )

    @model_serializer(mode="wrap")
    def omit_absent_patch(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        result: dict[str, Any] = handler(self)
        if not self.plan_patch_json:
            result.pop("plan_patch_json", None)
        return result


def _core_schema(model: type[BaseModel], fields: set[str]) -> dict[str, Any]:
    """Select existing fields and their referenced definitions, without new syntax."""
    full = model.model_json_schema()
    result = {k: deepcopy(v) for k, v in full.items() if k not in {"$defs", "properties"}}
    result["properties"] = {
        name: deepcopy(value) for name, value in full["properties"].items() if name in fields
    }
    result["required"] = [name for name in full.get("required", []) if name in fields]
    definitions = full.get("$defs", {})
    retained: dict[str, Any] = {}

    def retain(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("$ref", "")
            if reference.startswith("#/$defs/"):
                name = reference.removeprefix("#/$defs/")
                if name not in retained:
                    retained[name] = deepcopy(definitions[name])
                    retain(retained[name])
            for child in value.values():
                retain(child)
        elif isinstance(value, list):
            for child in value:
                retain(child)

    retain(result)
    if retained:
        result["$defs"] = retained
    return result


def native_bootstrap(contract: Contract | None = None) -> dict[str, Any]:
    """Supply usable initial syntax while richer optional plans remain discoverable."""
    from forge_mind.workspace.native_handoff_prepare import PrepareHandoffRequest

    result: dict[str, Any] = {
        "scope": "Core construction syntax only; planned effects are not measured evidence.",
        "plan_action": "kind=plan; document_json contains the DesignPlan JSON object",
        "plan_schema": _core_schema(
            DesignPlan,
            {
                "objective",
                "parts",
                "interfaces",
                "assembly_sequence",
                "verification",
                "unresolved",
                "operation_plan",
                "verification_bindings",
            },
        ),
        "document_schema": _core_schema(
            DesignDocument,
            {
                "version",
                "name",
                "units",
                "variables",
                "relations",
                "features",
                "outputs",
                "assembly",
                "production",
                "delivery_guidance",
            },
        ),
        "construction_patch_schema": _core_schema(
            PlanPatch, {"basis", "construction_assignments", "verification_bindings"}
        ),
        "prepare_handoff_schema": PrepareHandoffRequest.model_json_schema(),
        "feature_index": {
            kind: {
                "schema_query": f"feature:{kind}",
                **deepcopy(record),
            }
            for kind, record in OPERATIONS.items()
            if kind != "python"
        },
        "discovery": {
            "inspection_queries": list(INSPECTION_QUERIES),
            "inspect": "inspect accepts the listed inspection_queries and optional revision_id. Empty query or evidence reads saved measurements; source reads exact saved code and native document when present; diff reads the exact saved revision diff; topology reads the hash-checked saved topology index. Topology and source reads establish no new measurements, requirement pass or acceptance.",
            "prepared_print_capability": preparation_capability(),
            "requirements": "Only subject_ids in verification_subject_catalog are eligible for verification_bindings. Attribution subjects alone are not eligible. If the catalog is absent or empty, omit verification_bindings. For eligible typed subjects, save their explicit bindings before inspection. After saving CAD, inspect query=requirements runs the supported frozen geometric predicates on the current selected STEP within the existing author budget. Read actual failures and repair CAD before final instructions. It changes no targets, plan, selection, delivery or acceptance. Digital obligations remain unverified until actual final capture/readback. No automatic repeated heavy preflight runs.",
            "prepare_handoff": "After completing current plan/CAD work, inspect query=prepare_handoff with document_json matching prepare_handoff_schema combines current frozen inspection, explicitly requested all-made print preparation, and current guidance facts/schema in one action. Supply the exact current revision_id and explicit prepare_print boolean. Failed checks, changed inputs or other work blockers stop it. It never edits ownership, targets, CAD or instructions, and never captures final delivery. Review its current facts and submit guidance query=finish next; no separate guidance schema read is needed. Existing inspection and print tools remain available.",
            "plan_intent": "schema query=plan_intent; plan query=intent accepts a provisional part inventory only before any plan, draft or CAD. It lowers into canonical pending construction, with no evidence. Explicit plan patches extend this intent. The private staged controller can request this compact typed response directly.",
            "print_setup": "For additive made parts, start with an explicit assembly, even for one part. print_setup query=inspect with part_id measures saved made-definition planar poses. query=prepare creates all made plate-aligned STLs and an immutable setup receipt. Existing flat additive CAD must first convert through document plus an explicit plan_patch_json, preserving Features, variables and bindings with authored operation ownership. Use preparation before reviewing print instructions. Refresh guidance after preparation. No printer process or physical approval is inferred.",
            "full_plan": "schema query=plan includes prepared checks and requirement dispositions",
            "check_amendment": "To change a saved prepared-check request, supply criterion_amendment={previous_request_hash,reason,basis}. Copy previous_request_hash from that current check's completion.subsystems[].checks[].request_hash. The host computes it; do not invent or calculate it. The reason and basis are authored declarations. Old definitions and failed receipts remain historical. Changing a criterion does not change frozen requirements or prove a pass. schema query=plan_patch includes the typed fields.",
            "full_document": "schema query=document includes mechanisms and motion rigs",
            "construction": "schema query=construction or construction:<recipe_id>",
            "binding_transitions": "assembly_edit: promote_relations converts named derived values to inputs; derive_variables converts inputs to explicit relations. Query assembly_edit for syntax.",
            "focused_edits": "feature_edit document_json={basis:edit_basis,changes:[{part_id,append:[Feature],update:[Feature],output}]}. Append IDs must be new; update IDs must exist. Omitted fields and unrelated parts remain intact. Select an explicit new cut output. schema query=feature_delta supplies the complete contract. Read design pages for exact IDs when needed.",
            "plan_patch": "plan query=patch with document_json={basis:edit_basis,operation_steps:[OperationStep],...} upserts typed records by stable ID. schema query=plan_patch lists fields. It changes intent only; rejected CAD is never replayed. Refresh edit_basis after every save.",
            "construction_transaction": "For document, assembly_edit or feature_edit, optional plan_patch_json contains a complete PlanPatch with the current edit_basis. The explicit authored plan successor and CAD publish atomically. For feature_edit, both basis values must match. Use this to bind new Features or recapture current CAD with explicit ownership. No ownership, pass or acceptance is inferred. schema query=plan_patch supplies syntax. Other action kinds cannot carry this field.",
            "construction_assignments": "Use construction_patch_schema before the first CAD submission: plan_patch_json combines the current basis and explicit construction_assignments={step_id,feature_ids} entries. Include verification_bindings only for eligible subject_ids listed in verification_subject_catalog. If the catalog is absent or empty, omit verification_bindings. Assign only authored Features to an existing construction step. Assignments add ownership and retain every existing step field and requirement obligation. Do not resubmit a full operation_steps record merely to attach Features. Include any additional authored plan work in that same transaction using schema query=plan_patch; plan capture, checks and instructions still require current evidence.",
            "plan_draft": "plan query=draft_read|draft_patch|draft_commit uses schema query=plan_draft_read|plan_draft_patch|plan_draft_commit. draft_patch with current edit_basis, draft_version=0 and draft_hash=null begins an unpublished initial draft. Typed changes upsert stable records; later writes require the current draft version/hash. draft_read retrieves a collection or record_id without repeating the whole plan. Drafts retain rejected full plans beyond automatic context limits. Explicit draft_commit reruns complete validation and guarded atomic plan admission. Draft reads/patches apply no plan or CAD; failed commits retain the draft. Never infer acceptance or evidence from a draft.",
            "functional_interfaces": "Full plan schema includes functional_interfaces: requirement-linked participants, installed hardware, reachable output Features, operating states and duties. Completion derives missing construction and scoped proof from saved geometry. Nominal fit, animation and arithmetic cannot prove retention, locking or holding. Declare unsupported physical duties explicitly; never invent a pass.",
            "read_design": "Nested read_design options use schema query=read_design. Omit cursor for the first page; follow the returned opaque string verbatim. Invalid typed arguments return a recoverable rejection, never a replacement design.",
            "plan_reconciliation": "schema query=plan_reconciliation; plan query=reconcile appends explicit superseding facts with expected_plan_version and current revision_id. Do this before edited CAD. Changed plans invalidate instruction binding.",
            "guidance": "guidance query=read returns current measured facts and basis_hash. Use schema query=guidance, then guidance query=finish with document_json submits all four reviewed instructions and explicitly requests final validation/capture. Repair reasons return if work remains. No extra finish call required.",
            "all_actions": list(Action.model_json_schema()["properties"]["kind"]["enum"]),
        },
        "conventions": (
            "Feature inputs refer to earlier Features. Bindings name variables or relations, "
            "never expressions. Assembly documents use named parts and roots, with empty "
            "top-level features/outputs. Python Features and authored code remain forbidden. "
            "Use full schema queries for fields outside this core; no tool is hidden."
        ),
    }
    if contract is not None:
        from forge_mind.workspace.verification_admission import verification_binding_subject_ids

        result["discovery"]["verification_subject_catalog"] = {
            "contract_hash": contract.fingerprint(),
            "subject_ids": sorted(verification_binding_subject_ids(contract)),
            "scope": "Frozen binding identities only; not measured geometry, source confirmation or physical proof.",
        }
    return result
