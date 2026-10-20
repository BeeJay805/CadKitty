"""Finite software delivery-boundary observation, never arbitrary prose proof."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.nominal_inventory import captured_project

if TYPE_CHECKING:
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.models import Requirement

PROTOCOL = "nominal_delivery_scope_v1"
SCOPE = (
    "Observed structured-CAD project-export and optional unsliced print-package protocols only. "
    "This does not certify arbitrary authored prose, geometry, physical function, manufacture "
    "or activity outside this software delivery boundary."
)


def evaluate_delivery_scope(
    requirement: Requirement,
    *,
    document: DesignDocument,
    captured_delivery: dict[str, Any] | None = None,
    package_content: bytes = b"",
    print_context: dict[str, Any] | None = None,
    final: bool = False,
) -> dict[str, Any]:
    from forge_mind.workspace.verification_admission import criterion_hash

    criterion = requirement.verification
    if criterion is None or criterion.kind != "nominal_delivery_scope":
        raise ValueError("Scope proof requires its frozen software policy")
    result: dict[str, Any] = {
        "status": "unverified",
        "reason": "Software boundary observation awaits exact final captured outputs",
        "method": PROTOCOL,
        "criterion_hash": criterion_hash(criterion.model_dump(mode="json")),
        "facts": {"stage": "before_final_capture", "scope": SCOPE},
        "evidence_refs": [],
    }
    if not final:
        return result
    manifest, production, receipt = captured_project(
        document, captured_delivery or {}, package_content
    )
    # These are observations from actual read-back output protocols, not values
    # accepted merely because the requested policy says they should be present.
    checks = {
        "project_protocol": manifest.get("schema_version") == 1
        and manifest.get("bundle_kind") == "manufacturing_review",
        "captured_context": (manifest.get("context") or {}).get("status") == "captured",
        "project_receipt": receipt.get("status") == "draft_candidate",
        "project_not_manufacturing_ready": manifest.get("manufacturing_ready") is False
        and receipt.get("manufacturing_ready") is False
        and production.get("manufacturing_ready") is False,
        "project_physical_unverified": manifest.get("physical_readiness") == "unverified"
        and receipt.get("physical_readiness") == "unverified",
    }
    context = print_context or {}
    if context:
        record = context.get("record") or {}
        package = record.get("package") or {}
        setup = record.get("result") or {}
        checks["prepared_print_protocol"] = (
            context.get("status") == "current"
            and context.get("readback_verified") is True
            and package.get("schema_version") == 1
            and package.get("status") == "prepared"
            and package.get("manufacturing_ready") is False
            and package.get("physical_readiness") == "unverified"
            and package.get("revision_id") == receipt.get("revision_id")
            and (package.get("basis") or {}).get("algorithm") == "measured-planar-mesh-placement-v1"
            and setup.get("algorithm") == "measured-planar-mesh-placement-v1"
            and setup.get("operation") == "prepare"
            and setup.get("manufacturing_ready") is False
            and setup.get("physical_readiness") == "unverified"
        )
    result["facts"].update(stage="final_captured_software_boundary", observed_checks=checks)
    if not all(checks.values()):
        return {
            **result,
            "reason": "Captured output protocol exceeds or cannot establish nominal scope",
        }
    result.update(
        status="pass", reason="Captured software output protocols retain nominal-only limits"
    )
    result["evidence_refs"].append({"kind": "project_package", "receipt": receipt})
    if context:
        result["evidence_refs"].append(
            {"kind": "print_package", "receipt": context["record"]["package"]}
        )
    return result
