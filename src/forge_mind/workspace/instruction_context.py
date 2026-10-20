"""Pure instruction presentation; saved evidence and repair authority stay intact."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from forge_mind.workspace.customer_delivery import customer_delivery_view
from forge_mind.workspace.delivery_guidance import GuidanceUpdate
from forge_mind.workspace.instruction_plan import InstructionPlan
from forge_mind.workspace.native_review_context import compact_review_context


class ProtectedGuidanceUpdate(GuidanceUpdate):
    """Require supported references in the protected author instruction opportunity."""

    instruction_plan: InstructionPlan


def _without_feature_source(document: dict[str, Any]) -> dict[str, Any]:
    """Omit automatic construction source, retaining all non-source metadata."""
    result = deepcopy(document)
    if result.get("representation") == "summary_only_not_a_design_document":
        return result
    omitted = []
    if "features" in result:
        features = result.pop("features")
        result["feature_ids"] = [item["id"] for item in features]
        omitted.append("features")
    assembly = result.get("assembly")
    if isinstance(assembly, dict):
        for part in assembly.get("parts", []):
            if "features" in part:
                features = part.pop("features")
                part["feature_ids"] = [item["id"] for item in features]
                omitted.append("assembly.parts[].features")
    result["representation"] = "instruction_reference_only_not_a_design_document"
    result["omitted_source_fields"] = sorted(set(omitted))
    result["retrieve"] = (
        "Use read_design with the exact revision_id for complete saved pages; "
        "read_part retrieves a complete assembly definition. Never submit this presentation "
        "as replacement source."
    )
    return result


def customer_instruction_context(context: dict[str, Any]) -> dict[str, Any]:
    """Present current instruction facts without changing any verdict or saved input.

    The same projection supports early NativeAction review and the forced typed
    update. The caller selects that response contract. Explicit part, design and
    source reads remain complete; no complexity or instruction length cap applies.
    """
    result = compact_review_context(context)
    result["customer_delivery"] = customer_delivery_view(context.get("delivery_guidance"))
    for key in ("selected", "failed_attempt"):
        original = context.get(key)
        projected = result.get(key)
        if not isinstance(original, dict) or not isinstance(projected, dict):
            continue
        # Keep every current measurement and diagnostic, including nested values.
        if "measurement" in original:
            projected["measurement"] = deepcopy(original["measurement"])
        if isinstance(original.get("document"), dict):
            # Use original non-source metadata, not a bounded document preview:
            # production notes and physical limitations must survive intact.
            projected["document"] = _without_feature_source(original["document"])
            projected["document_is_summary"] = True
        if (
            "contract" in projected
            and isinstance(result.get("frozen_contract"), dict)
            and projected["contract"] == result["frozen_contract"]
        ):
            projected.pop("contract", None)
            projected["contract_in_context"] = "frozen_contract"
    guidance = result.get("delivery_guidance")
    original_guidance = context.get("delivery_guidance")
    # Retain the exact current basis, not a reconstructed or broadened binding.
    if (
        isinstance(guidance, dict)
        and isinstance(original_guidance, dict)
        and "basis" in original_guidance
    ):
        guidance["basis"] = deepcopy(original_guidance["basis"])
    attribution = result.get("request_attribution")
    if (
        isinstance(attribution, dict)
        and "proposed_contract" in attribution
        and isinstance(result.get("frozen_contract"), dict)
        and attribution["proposed_contract"] == result.get("frozen_contract")
    ):
        del attribution["proposed_contract"]
        attribution["proposed_contract_in_context"] = "frozen_contract"
    for name in ("history", "persistent_tool_observations"):
        for original, projected in zip(context.get(name, []), result.get(name, []), strict=True):
            # History is explicitly historical. Retain diagnostics even when they
            # are absent from current measurements or the compact repair focus.
            if isinstance(original.get("measurement"), dict):
                projected["measurement"] = deepcopy(original["measurement"])
    result["instruction_presentation"] = {
        "scope": (
            "Current customer_delivery and delivery_guidance facts describe separate evidence "
            "scopes. Historical tool observations and author declarations are not current "
            "workflow commands. No requirement satisfaction or physical approval is granted."
        ),
        "authority": (
            "Use the exact current revision_id and delivery_guidance.basis_hash. Keep frozen "
            "targets, failed checks, missing inputs and physical limits unchanged. Select exact "
            "IDs and compatible actions from delivery_guidance.instruction_catalog. The server "
            "resolves fact values and step text. Free prose remains unverified author notes."
        ),
        "retrieval": (
            "All NativeAction repair and retrieval tools remain available in native_authoring. "
            "Retrieve exact schemas and source when a repair requires them. Explicit reads "
            "are complete; automatic document presentations are not replacement CAD."
        ),
    }
    return result
