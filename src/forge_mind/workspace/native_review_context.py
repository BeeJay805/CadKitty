"""Pure review context reduction; full designs and tools remain available."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, get_args

from forge_mind.workspace.document import digest
from forge_mind.workspace.inspection import author_revision_context
from forge_mind.workspace.models import Action
from forge_mind.workspace.native_stages import _schema_record


def _inventory(value: Any, revision_id: Any) -> dict[str, Any]:
    """Summarize automatic inventories without interpreting their verdicts."""
    diagnostics: list[dict[str, Any]] = []

    def collect(item: Any) -> None:
        if isinstance(item, dict):
            if item.get("status") in ("fail", "failed", "invalid", "unverified") or any(
                item.get(key) for key in ("errors", "caveats", "unresolved")
            ):
                diagnostics.append(item)
            else:
                for child in item.values():
                    collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)

    collect(value)
    return {
        "is_summary": True,
        "item_count": len(value) if isinstance(value, (dict, list)) else None,
        "sha256": digest(value),
        "revision_id": revision_id,
        "retained_diagnostics": diagnostics,
        "retrieve": (
            "Use inspect with this revision_id for saved evidence; use read_design or "
            "read_part for full definitions. This summary is not a replacement document."
        ),
    }


def _measurement(value: dict[str, Any], revision_id: Any) -> dict[str, Any]:
    result = dict(value)
    facts = dict(value.get("facts") or {})
    # Retain dimensions, holes, assembly interfaces, mates and all top-level
    # findings/errors/caveats. Only automatic detail inventories are summarized.
    for key in ("feature_history", "bodies", "faces", "edges", "surfaces", "topology"):
        if key in facts and len(json.dumps(facts[key])) > 4000:
            facts[key] = _inventory(facts[key], revision_id)
    assembly = facts.get("assembly")
    if isinstance(assembly, dict):
        assembly = dict(assembly)
        occurrences = []
        for row in assembly.get("occurrences", []):
            occurrence = dict(row)
            surfaces = occurrence.get("cylindrical_surfaces")
            if surfaces is not None and len(json.dumps(surfaces)) > 4000:
                occurrence["cylindrical_surfaces"] = _inventory(surfaces, revision_id)
            occurrences.append(occurrence)
        if "occurrences" in assembly:
            assembly["occurrences"] = occurrences
        facts["assembly"] = assembly
    result["facts"] = facts
    return result


def _revision(value: dict[str, Any], plan: Any) -> dict[str, Any]:
    # Reuse the normal revision summary only for complete documents. Applying it
    # to a previous summary would hash the summary instead of the saved document.
    document = value.get("document") or {}
    result = (
        author_revision_context(value)
        if value.get("id")
        and document
        and not value.get("document_is_summary")
        and document.get("representation") != "summary_only_not_a_design_document"
        else dict(value)
    )
    for key in ("code", "diff"):
        result.pop(key, None)
    if "document_hash" in value:
        result["document_hash"] = value["document_hash"]
    if isinstance(value.get("measurement"), dict):
        result["measurement"] = _measurement(value["measurement"], value.get("id"))
    record = result.get("operation_record")
    if isinstance(record, dict):
        record = dict(record)
        snapshot = record.get("plan_snapshot")
        if isinstance(snapshot, dict) and snapshot.get("plan") == plan:
            record["plan_snapshot"] = {k: v for k, v in snapshot.items() if k != "plan"}
            record["operation_plan_in_context"] = "design_plan"
        result["operation_record"] = record
    return result


def _guidance_history(history: list[Any], revision_id: Any) -> dict[str, Any]:
    return {
        "is_summary": True,
        "entry_count": len(history),
        "sha256": digest(history),
        "revision_id": revision_id,
        "entries": [
            {
                key: row[key]
                for key in ("version", "predecessor_hash", "revision_id", "record_hash", "legacy")
                if key in row
            }
            for row in history
            if isinstance(row, dict)
        ],
        "scope": "Historical author declarations, never current instructions or proof.",
        "retrieve": "guidance query=read returns complete literal target and record-origin histories",
    }


def _guidance(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    basis = result.pop("basis", None)
    if isinstance(basis, dict) and "revision_id" in basis:
        result["revision_id"] = basis["revision_id"]
    record = result.get("record")
    if isinstance(record, dict):
        record = dict(record)
        record.pop("basis", None)
        result["record"] = record
    history = result.get("history")
    if isinstance(history, list):
        result["history"] = _guidance_history(history, result.get("revision_id"))
    provenance = result.get("record_provenance")
    if isinstance(provenance, dict):
        provenance = dict(provenance)
        origin_history = provenance.get("history")
        if isinstance(origin_history, list):
            provenance["history"] = _guidance_history(origin_history, provenance.get("revision_id"))
        result["record_provenance"] = provenance
    basis_hash, revision_id = result.get("basis_hash"), result.get("revision_id")
    result["review_basis_available"] = (
        isinstance(basis_hash, str)
        and len(basis_hash) == 64
        and all(c in "0123456789abcdef" for c in basis_hash)
        and isinstance(revision_id, str)
        and len(revision_id) == 32
        and all(c in "0123456789abcdef" for c in revision_id)
    )
    result["basis_inventory_omitted"] = isinstance(basis, dict)
    result["basis_scope"] = (
        "Use the exact basis_hash from guidance query=read. Status alone supplies no usable "
        "review basis. Omitted inventories remain host-verified; this summary grants no approval."
    )
    return result


def compact_review_context(context: dict[str, Any]) -> dict[str, Any]:
    """Reduce automatic context for an explicit review phase, without changing state.

    The caller chooses the phase. This function does not decide readiness, finish
    jobs, change requirements or limit design complexity. Explicit reads stay full.
    """
    result = deepcopy(context)
    bootstrap = result.get("native_authoring") or {}
    discovery = dict(bootstrap.get("discovery") or {})
    discovery.update(
        all_actions=list(get_args(Action.model_fields["kind"].annotation)),
        full_plan="schema query=plan; plans retain complete dimensions and obligations",
        full_document="schema query=document; read_design retrieves saved revision pages",
        guidance="schema query=guidance; guidance query=read refreshes the exact review basis",
        construction="schema query=construction or feature:<kind>; repairs remain available",
        binding_transitions="schema query=assembly_edit; read_design variables|relations",
        read_part="read_part with revision_id and part_id retrieves the complete part",
        read_design="read_design with revision_id and query retrieves complete requested pages",
        inspect="inspect with revision_id retrieves saved measured evidence",
    )
    result["native_authoring"] = {
        "discovery": discovery,
        "scope": "Review context only; every action remains available, including CAD repair.",
    }
    schemas = result.get("authoritative_schemas") or {}
    # The caller already bounds this cache. Review still permits CAD repair,
    # so explicitly fetched syntax must survive just as it does in construction.
    result["authoritative_schemas"] = schemas
    result["schema_catalog"] = [_schema_record(name, value) for name, value in schemas.items()]
    result["review_context"] = {
        "is_summary": True,
        "scope": (
            "Automatic construction context is reduced; requirements and physical limits remain open."
        ),
        "schema_retrieval": (
            "Exact cached syntax remains available; request schema for uncached syntax."
        ),
        "repair": (
            "If review finds a CAD defect, request the relevant schema or saved design and repair it."
        ),
        "size_scope": (
            "Context reduction is not a design complexity limit; explicit read responses stay complete."
        ),
    }
    for key in ("selected", "failed_attempt"):
        if isinstance(result.get(key), dict):
            result[key] = _revision(result[key], result.get("design_plan"))
    if isinstance(result.get("delivery_guidance"), dict):
        result["delivery_guidance"] = _guidance(result["delivery_guidance"])
    for key in ("history", "persistent_tool_observations"):
        for row in result.get(key, []):
            if not isinstance(row, dict):
                continue
            # Tool reads are explicit evidence requests. Never reduce their
            # returned definitions, source text or inspection observations here.
            if any(
                name in {"part", "read_part", "read_design"} or name.startswith("source")
                for name in row
            ):
                continue
            name = row.get("schema_read")
            if (
                isinstance(name, str)
                and name in schemas
                and "authoring_schema" in row
                and digest(row["authoring_schema"]) == digest(schemas[name])
            ):
                row.pop("authoring_schema")
            if (
                isinstance(row.get("guidance"), dict)
                and row.get("explicit_guidance_read") is not True
            ):
                row["guidance"] = _guidance(row["guidance"])
            if isinstance(row.get("measurement"), dict):
                row["measurement"] = _measurement(row["measurement"], row.get("revision_id"))
    return result
