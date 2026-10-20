"""Pure preparation of a small deferred physical bench protocol."""

from __future__ import annotations

from typing import Any

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.production import production_report

SCOPE = (
    "Preparation and owner-reported observations only. No printer actuation, purchases, "
    "instrument attestation, load rating, accepted CAD change, or physical qualification."
)
CHECKS = (
    (
        "print_completion",
        "Print the required copies",
        "Record the actual printer, material, settings and completion or failure.",
    ),
    (
        "part_inspection",
        "Inspect and clean the parts",
        "Remove supports and loose strands. Record cracks, separated layers or blocked holes.",
    ),
    (
        "hardware_fit",
        "Check the actual hardware",
        "When hardware is needed, check passages, seating, engagement and access for its matching tools. Do not force a fit.",
    ),
    (
        "assembly_fit",
        "Assemble on the bench",
        "Follow the saved assembly steps with the actual object. Record loose fit, interference or visible damage.",
    ),
    (
        "removal",
        "Remove and reinstall the object",
        "Follow the saved removal steps while supporting the object. Record access, tools and any help needed.",
    ),
    (
        "wall_installation",
        "Review the installation surface",
        "If applicable, record the actual surface and chosen fasteners. If not performed, keep this check not_run.",
    ),
)
MISSING_INPUTS = (
    "Actual printer model, approved profile and readiness",
    "Actual material identity and available filament",
    "Printed specimen identity and completed print observations",
    "Actual hardware dimensions, matching tools and engagement",
    "Actual mating object dimensions and fit",
    "Installation surface, anchors and load suitability when applicable",
)


def build_physical_pilot(
    revision: dict[str, Any],
    guidance: dict[str, Any],
    print_setup: dict[str, Any] | None,
) -> dict[str, Any]:
    """Produce declarations and pending checks; never invent a physical result."""
    document = DesignDocument.model_validate(revision["document"])
    report = production_report(
        document, (revision.get("measurement", {}).get("facts") or {}).get("assembly")
    )
    if not report["make"]:
        raise ValueError("Physical preparation requires at least one declared made part")
    artifacts = revision["measurement"]["artifact_hashes"]
    made = []
    for row in report["make"]:
        stem = "definition_" + row["part_id"] if document.assembly else "part"
        name = stem + ".stl"
        if not artifacts.get(name):
            raise ValueError("Physical preparation requires saved STL for every made definition")
        made.append(
            {
                "part_id": row["part_id"],
                "name": row["name"],
                "quantity": row["quantity"],
                "source_file": "make/" + name,
                "source_sha256": artifacts[name],
                "material": row["production"].get("material"),
            }
        )
    hardware = [
        {
            "part_id": row["part_id"],
            "name": row["name"],
            "quantity": row["quantity"],
            "catalog_id": row.get("catalog_id"),
            "procurement": row["production"].get("procurement"),
        }
        for row in report["buy"]
    ]
    record = guidance["record"]
    protocol = {
        "schema_version": 1,
        "binding": {
            "revision_id": revision["id"],
            "document_hash": guidance["basis"]["document_hash"],
            "contract_hash": guidance["basis"]["contract_hash"],
            "guidance_basis_hash": guidance["basis_hash"],
            "guidance_record_hash": digest(record),
            "artifact_hashes": artifacts,
            "print_setup_hash": digest(print_setup) if print_setup is not None else None,
        },
        "made_parts": made,
        "hardware": hardware,
        "production_gaps": report["gaps"],
        "unresolved_production": [row["part_id"] for row in report["unresolved"]],
        "print_setup": print_setup,
        "instructions": record["guidance"],
        "checks": [
            {"check_id": identifier, "title": title, "instruction": instruction}
            for identifier, title, instruction in CHECKS
        ],
        "missing_inputs": list(MISSING_INPUTS),
        "physical_status": "deferred",
        "physical_qualification": "unverified",
        "scope": SCOPE,
    }
    protocol_hash = digest(protocol)
    lines = [
        "# Your print and bench-test packet",
        "",
        "Actual printing and physical testing are deferred. This preparation packet contains no specimen observations.",
        "Use the lab's approved printer setup. These files and instructions do not establish a safe working load.",
        "",
        "## Parts to print",
        "",
    ]
    for part in made:
        lines.append(
            f"- {part['name']}: {part['quantity']} copies; [{part['source_file']}]({part['source_file']})."
        )
    if print_setup is not None:
        lines += [
            "",
            "## Prepared print files",
            "",
            "These files are rigidly placed derivatives. Keep the original CAD and review the lab's slicer setup.",
        ]
        names = {part["part_id"]: part["name"] for part in made}
        for path, item in print_setup["files"].items():
            lines.append(
                f"- {names[item['part_id']]}: {item['quantity']} copies; [{path}]({path})."
            )
    lines += ["", "## Hardware and tools", ""]
    if hardware:
        lines.extend(f"- {item['name']}: {item['quantity']}." for item in hardware)
    else:
        lines.append(
            "No bought hardware occurs in the saved assembly. This does not mean hardware is unnecessary."
        )
    lines.append(
        "Use the tool type and size that match the actual purchased hardware. Confirm access before use."
    )
    for field, title in (
        ("print", "Printing"),
        ("installation", "Assembly and installation"),
        ("use_and_removal", "Use and removal"),
        ("unresolved", "Unknowns and checks"),
    ):
        lines += ["", "## " + title, "", record["guidance"][field]]
    lines += ["", "## Bench test and short results form", ""]
    for index, (_, title, instruction) in enumerate(CHECKS, 1):
        lines.append(f"{index}. {title}. {instruction}")
    lines += [
        "",
        "Record the specimen, printer/profile, material, actual hardware and tools.",
        "For each step, choose observed_pass, observed_fail, unknown or not_run and explain what happened.",
        "For a numeric reading, record the value, units and measurement method. Leave unmeasured values unknown.",
        "Do not record a physical pass from a CAD image, slicer result or an unprinted specimen.",
        "",
        "## Missing physical inputs",
        "",
    ]
    lines.extend("- " + item for item in MISSING_INPUTS)
    if print_setup is None:
        lines.append(
            "- No saved automatic print setup is bound to this packet. Source STL files remain available."
        )
    else:
        lines.append(
            "A saved print setup is bound to this packet. Its placement is not a printer or adhesion qualification."
        )
    lines += [
        "",
        "## Traceability",
        "",
        f"Revision: {revision['id']}",
        f"Protocol hash: {protocol_hash}",
        "",
        SCOPE,
    ]
    return {
        "protocol": protocol,
        "protocol_hash": protocol_hash,
        "packet_markdown": "\n".join(lines) + "\n",
        "physical_status": "deferred",
        "physical_qualification": "unverified",
        "pending_observations": [
            {
                "check_id": item[0],
                "status": "not_run",
                "reason": "Actual printing and physical testing are deferred",
            }
            for item in CHECKS
        ],
        "missing_inputs": list(MISSING_INPUTS),
        "scope": SCOPE,
    }
