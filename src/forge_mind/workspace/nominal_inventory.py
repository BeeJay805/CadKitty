"""Exact designed-definition inventory; no geometric or physical qualification."""

from __future__ import annotations

import hashlib
import io
import json
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

from forge_mind.workspace.document import DesignDocument, digest

if TYPE_CHECKING:
    from forge_mind.workspace.models import DesignPlan, Requirement

PROTOCOL = "nominal_subject_inventory_v1"


def captured_project(
    document: DesignDocument, delivery: dict[str, Any], content: bytes
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Check the existing reader's exact captured bytes, never a proposed receipt."""
    receipt = delivery.get("receipt") or {}
    if (
        delivery.get("status") != "delivered"
        or delivery.get("readback_verified") is not True
        or receipt.get("sha256") != hashlib.sha256(content).hexdigest()
        or receipt.get("bytes") != len(content)
    ):
        raise ValueError("Exact current captured project readback is required")
    with ZipFile(io.BytesIO(content)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Captured project contains duplicate members")
        manifest = json.loads(archive.read("manifest.json"))
        files = manifest["files"]
        if set(names) != set(files) | {"manifest.json"}:
            raise ValueError("Captured project members differ from its manifest")
        for name in ("document.json", "production.json"):
            data = archive.read(name)
            if files[name]["sha256"] != hashlib.sha256(data).hexdigest() or files[name][
                "size_bytes"
            ] != len(data):
                raise ValueError("Captured inventory inputs differ from manifest bytes")
        saved = DesignDocument.model_validate_json(archive.read("document.json"))
        if saved != document or manifest.get("document_hash") != digest(document.model_dump()):
            raise ValueError("Captured editable document differs from the selected document")
        production = json.loads(archive.read("production.json"))
    return manifest, production, receipt


def _quantities(rows: list[dict[str, Any]]) -> dict[str, int]:
    quantities: dict[str, int] = {}
    for row in rows:
        part_id, quantity = row["part_id"], row["quantity"]
        if part_id in quantities or type(quantity) is not int or quantity <= 0:
            raise ValueError("Made inventory needs unique positive integer occurrence quantities")
        quantities[part_id] = quantity
    return quantities


def evaluate_inventory(
    requirement: Requirement,
    *,
    plan: DesignPlan,
    document: DesignDocument,
    captured_delivery: dict[str, Any] | None = None,
    package_content: bytes = b"",
    final: bool = False,
) -> dict[str, Any]:
    from forge_mind.workspace.production import production_report
    from forge_mind.workspace.verification_admission import (
        criterion_hash,
        resolve_verification_inventory,
    )

    criterion = requirement.verification
    if criterion is None or criterion.kind != "nominal_subject_inventory":
        raise ValueError("Inventory proof requires its frozen inventory criterion")
    result: dict[str, Any] = {
        "status": "unverified",
        "reason": "Current design inventory still requires exact final capture/readback",
        "method": PROTOCOL,
        "criterion_hash": criterion_hash(criterion.model_dump(mode="json")),
        "facts": {"stage": "current_design_before_capture", "physical_readiness": "unverified"},
        "evidence_refs": [],
    }
    resolved = resolve_verification_inventory(requirement, plan, document)
    bound = {row["subject_id"]: row for row in resolved}
    if len(bound) != len(criterion.items):
        raise ValueError("Inventory requires every unique frozen subject binding")
    definitions = {part.id: part for part in document.assembly.parts} if document.assembly else {}
    expected: dict[str, int] = {}
    observations = []
    for item in criterion.items:
        binding = bound[item.subject_id]
        part_id = binding["part_id"]
        if part_id in expected:
            raise ValueError("Inventory subjects cannot alias one made definition")
        expected[part_id] = item.quantity
        name = definitions[part_id].name if document.assembly else document.name
        observations.append({**binding, "required_name": item.required_name, "actual_name": name})
        if name != item.required_name:
            return {**result, "status": "fail", "reason": "Exact requested definition name differs"}
    declared_made = (
        {part.id for part in document.assembly.parts if part.production.role == "make"}
        if document.assembly
        else {"part"}
    )
    actual = _quantities(production_report(document)["make"])
    result["facts"].update(
        bindings=observations, expected_made_quantities=expected, actual_made_quantities=actual
    )
    if actual != expected or declared_made != set(expected):
        return {
            **result,
            "status": "fail",
            "reason": "All made declarations and installed occurrence quantities must match exactly",
        }
    result["facts"]["current_design_status"] = "pass"
    if not final:
        return result
    manifest, production, receipt = captured_project(
        document, captured_delivery or {}, package_content
    )
    if (
        _quantities(production["make"]) != expected
        or _quantities(receipt["made_parts"]) != expected
        or {row["part_id"]: row["name"] for row in production["make"]}
        != {row["part_id"]: row["name"] for row in production_report(document)["make"]}
    ):
        return {**result, "status": "fail", "reason": "Captured made inventory differs from design"}
    result.update(
        status="pass",
        reason="Exact named designed definitions and all installed made quantities captured and read back",
    )
    result["facts"].update(
        stage="final_captured_inventory",
        captured_document_sha256=manifest["files"]["document.json"]["sha256"],
        captured_production_sha256=manifest["files"]["production.json"]["sha256"],
    )
    result["evidence_refs"].append({"kind": "project_package", "receipt": receipt})
    return result
