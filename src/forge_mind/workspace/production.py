"""Occurrence-aware production declarations and make-only geometry packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.customer_delivery import customer_delivery_view
from forge_mind.workspace.document import DesignDocument, Feature
from forge_mind.workspace.instruction_plan import FAILED_CHECK_STATES
from forge_mind.workspace.product_models import PartProduction


def _legacy_catalog(features: list[Feature], outputs: list[str]) -> str | None:
    """Only an unchanged single catalog output establishes a purchased definition."""
    lookup = {feature.id: feature for feature in features}
    if len(outputs) != 1:
        return None
    feature = lookup.get(outputs[0])
    while feature and not feature.suppressed and feature.kind == "transform":
        feature = lookup.get(feature.inputs[0])
    return (
        feature.catalog_id
        if feature and not feature.suppressed and feature.kind == "hardware"
        else None
    )


def production_report(
    document: DesignDocument | None, assembly_facts: dict[str, Any] | None = None
) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    gaps: list[str] = []
    mass_gaps: list[str] = []
    mass = 0.0
    if document is None:
        gaps.append("No feature document declares production roles")
    else:
        from forge_mind.workspace.assembly import expand

        if document.assembly:
            occurrences = expand(document)
            definitions = [
                (
                    p.id,
                    p.name,
                    p.production,
                    p.features,
                    [p.output],
                    [r["path"] for r in occurrences if r["definition"] == p.id],
                )
                for p in document.assembly.parts
            ]
        else:
            definitions = [
                (
                    "part",
                    document.name,
                    document.production or PartProduction(),
                    document.features,
                    document.outputs or [f.id for f in document.features[-1:]],
                    ["part"],
                )
            ]
        measured = {r["path"]: r for r in (assembly_facts or {}).get("occurrences", [])}
        for part_id, name, production, features, outputs, paths in definitions:
            if not paths:
                continue
            catalog = _legacy_catalog(features, outputs)
            declared_catalog = production.procurement.catalog_id if production.procurement else None
            if declared_catalog:
                from forge_mind.agentic.catalog.registry import get_catalog_part

                try:
                    get_catalog_part(declared_catalog)
                except (KeyError, ValueError):
                    gaps.append(f"{part_id}: procurement catalog identity is unknown")
                if catalog and declared_catalog != catalog:
                    gaps.append(
                        f"{part_id}: procurement catalog identity conflicts with inserted geometry"
                    )
            role = production.role
            basis = "explicit declaration" if role != "unspecified" else "unresolved"
            if role == "unspecified" and catalog:
                role, basis = "buy", "unchanged legacy catalog geometry"
            line = {
                "part_id": part_id,
                "name": name,
                "quantity": len(paths),
                "role": role,
                "role_basis": basis,
                "production": production.model_dump(),
                "occurrence_paths": paths,
                "catalog_id": catalog,
            }
            lines.append(line)
            if role == "unspecified":
                gaps.append(f"{part_id}: production role unresolved")
            if role == "make":
                if production.material is None:
                    gaps.append(f"{part_id}: material unspecified")
                if production.process == "unspecified":
                    gaps.append(f"{part_id}: process unspecified")
            if (
                role == "buy"
                and not catalog
                and not (
                    production.procurement
                    and (
                        production.procurement.catalog_id
                        or (production.procurement.supplier and production.procurement.part_number)
                    )
                )
            ):
                gaps.append(f"{part_id}: procurement identity unspecified")
            if role == "reference":
                continue
            density = production.material.density_kg_m3 if production.material else None
            volumes = [measured.get(path, {}).get("volume_mm3") for path in paths]
            if (
                role == "unspecified"
                or density is None
                or any(
                    not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                    for v in volumes
                )
            ):
                mass_gaps.append(f"{part_id}: physical role, density or measured volume missing")
            else:
                mass += sum(volumes) * density / 1e9
    return {
        "lines": lines,
        **{
            role: [line for line in lines if line["role"] == role]
            for role in ("make", "buy", "reference")
        },
        "unresolved": [line for line in lines if line["role"] == "unspecified"],
        "setup_status": "incomplete" if gaps or not lines else "declared",
        "manufacturing_ready": False,
        "gaps": gaps,
        "mass_kg": mass if lines and not mass_gaps else None,
        "mass_scope": "Nominal CAD volume times declared per-part density; simplified purchased envelopes and omitted internals can differ from real mass",
        "mass_gaps": mass_gaps,
        "scope": "Declared production setup only. Process feasibility, tolerances, tooling, supplier suitability and physical acceptance remain unverified. Reference geometry excluded from physical mass.",
    }


def resolve_sources(report: dict[str, Any], sources: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the same source interpretation gaps in the inspector and exported handoff."""
    claims = {c["id"]: c for c in (sources or {}).get("claims", [])}
    bindings = []
    for line in report["lines"]:
        declared = line["production"]
        for section in (
            declared,
            declared.get("material") or {},
            declared.get("procurement") or {},
        ):
            for claim_id in section.get("source_claim_ids", []):
                claim = claims.get(claim_id)
                status = (
                    "missing"
                    if claim is None
                    else "confirmed_interpretation"
                    if claim.get("confirmation")
                    else "unconfirmed"
                )
                bindings.append(
                    {
                        "part_id": line["part_id"],
                        "claim_id": claim_id,
                        "status": status,
                        "scope": "Source interpretation only; geometry compliance requires a separate measured interface check",
                    }
                )
                if status != "confirmed_interpretation":
                    report["gaps"].append(f"{line['part_id']}: source claim {claim_id} is {status}")
    report["source_bindings"] = bindings
    report["source_fingerprint"] = (sources or {}).get("fingerprint")
    if report["gaps"]:
        report["setup_status"] = "incomplete"
    return report


def package_file_inventory(
    document: DesignDocument,
    artifact_hashes: dict[str, str],
    *,
    include_stl: bool = True,
    include_assembly_step: bool = True,
    include_import_assets: bool = False,
) -> dict[str, Any]:
    """Map saved geometry to project paths; saved provenance is not file delivery."""
    report = production_report(document)
    geometry: dict[str, Any] = {}
    for part in report["make"]:
        stem = "definition_" + part["part_id"] if document.assembly else "part"
        for extension in ("step", "stl") if include_stl else ("step",):
            source = f"{stem}.{extension}"
            geometry["make/" + source] = {
                "source_artifact": source,
                "source_present": source in artifact_hashes,
                "role": "make",
                "part_id": part["part_id"],
                "quantity": part["quantity"],
            }
    if include_assembly_step:
        geometry["assembly/part.step"] = {
            "source_artifact": "part.step",
            "source_present": "part.step" in artifact_hashes,
            "role": "overall_review",
            "part_id": None,
            "quantity": 1,
        }
    if include_import_assets:
        from forge_mind.workspace.handoff import referenced_import_asset_ids

        for asset_id in referenced_import_asset_ids(document.model_dump()):
            source = f"input_{asset_id}.step"
            geometry["inputs/" + source] = {
                "source_artifact": source,
                "source_present": source in artifact_hashes,
                "role": "import_source",
                "part_id": None,
                "asset_id": asset_id,
                "quantity": 1,
            }
    exported_sources = {entry["source_artifact"] for entry in geometry.values()}
    return {
        "status": "prospective_export_paths",
        "options": {
            "include_stl": include_stl,
            "include_assembly_step": include_assembly_step,
            "include_import_assets": include_import_assets,
        },
        "geometry": geometry,
        "metadata_files": [
            "document.json",
            "production.json",
            "purchasing.json",
            "sources.json",
            "engineering-evidence.json",
            "manifest.json",
            *(["README.md"] if include_assembly_step else []),
            *(["inputs/assets.json", "import-assets.json"] if include_import_assets else []),
        ],
        "excluded_saved_artifacts": sorted(
            name
            for name in artifact_hashes
            if name.endswith((".step", ".stl", ".stp", ".brep", ".glb"))
            and name not in exported_sources
        ),
        "print_preparation": "not established by this inventory",
        "scope": (
            "Prospective package-relative paths for the selected export options, not captured files. "
            "Source presence does not establish a valid export or captured delivery. "
            "Optional reports, imported inputs and separate prepared print files retain their own "
            "receipts. The captured manifest lists the exact final members. Saved artifact hashes "
            "remain provenance, including artifacts excluded from this package."
        ),
    }


def _package_readme(
    document: DesignDocument,
    report: dict[str, Any],
    members: set[str],
    guidance_context: dict[str, Any] | None = None,
) -> bytes:
    """Describe saved files and declarations; never infer manufacture or physical use."""
    lines = [
        "# Your draft CAD project",
        "",
        "Physical fit, printing and load capacity are not verified.",
        "",
    ]
    current_guidance = bool(guidance_context and guidance_context.get("status") == "current")
    file_inventory = package_file_inventory(
        document,
        (guidance_context or {}).get("basis", {}).get("artifact_hashes", {}),
        include_import_assets=True,
    )
    prospective = (guidance_context or {}).get("facts", {}).get("package_files") or {}
    geometry = file_inventory["geometry"]
    if "geometry" in prospective:
        # Retain the resolved prospective policy, plus any actual extra members.
        # A selected optional omission is not a missing required file.
        geometry = {
            path: entry
            for path, entry in geometry.items()
            if path in prospective["geometry"] or path in members
        }
    presentation_inventory = {
        **file_inventory,
        **prospective,
        "geometry": geometry,
    }
    delivery = customer_delivery_view(
        guidance_context, members=members, package_inventory=presentation_inventory
    )
    build_time_findings = (guidance_context or {}).get("facts", {}).get(
        "requirement_findings_stage"
    ) == "build_time_before_final_contract_verification"

    def literal(label: str, text: str) -> None:
        # Keep author text literal, including Markdown fences and HTML.
        fence = "`" * max(3, max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
        lines.extend(["", label, "", fence + "text", text, fence])

    def compiled_instructions() -> None:
        binding = delivery["instruction_binding"]
        lines.extend(
            [
                "",
                "## Captured file facts and build-time check findings"
                if build_time_findings
                else "## Current file and check facts",
                "",
                "Instruction binding status: " + binding["status"] + ".",
                "Bound references do not verify adjacent author notes or physical performance.",
            ]
        )
        for fact in binding["facts"]:
            lines.append("- " + fact["text"])
            lines.append("  Scope: " + fact["scope"])
        lines.extend(["", "## Print, install and use", ""])
        lines.append(
            "The current supported steps below cover file access and review. "
            "They do not establish a verified physical assembly or use procedure. "
            "Literal practical proposals remain visible in saved author history below."
        )
        if binding["steps"]:
            for index, step in enumerate(binding["steps"], 1):
                lines.append(f"{index}. {step['text']} ({step['actor']}; {step['state']})")
        else:
            lines.append(
                "No bound customer steps are supplied. Unverified notes are retained separately."
            )
        if binding["errors"]:
            literal("Instruction reference problems:", "\n".join(binding["errors"]))
        lines.append(binding["scope"])

    def procedure_proposals() -> None:
        proposal = delivery["procedure_proposal"]
        if proposal["status"] != "available_unverified":
            return
        lines.extend(["", "## Saved author proposal before final capture", "", proposal["scope"]])
        source = proposal["source"]
        lines.append(
            "Source revision: "
            + str(source["revision_id"])
            + "; guidance SHA-256: "
            + str(source["guidance_hash"])
            + "; reviewed basis: "
            + str(source["basis_hash"])
            + "."
        )
        omitted = delivery["project_files"]["omitted"]
        if omitted:
            literal("Files omitted from this download:", "\n".join(omitted))
            lines.append("Obtain required files before using the proposed directions below.")
        for field, label in (
            ("print", "Saved printing proposal (unverified):"),
            ("installation", "Saved installation proposal (unverified):"),
            ("use_and_removal", "Saved use and removal proposal (unverified):"),
            ("unresolved", "Saved limitations and proposed physical tests (unverified):"),
        ):
            literal(label, proposal["procedures"][field])

    if guidance_context is not None:
        lines += ["", "## File and instruction status", ""]
        lines.append("Project file status: " + delivery["project_files"]["status"] + ".")
        lines.append("Instruction reference state: " + delivery["guidance"]["status"] + ".")
        lines.append("Included project files retain their original CAD frames.")
        interface_rows = delivery["checks"]["interfaces"].get("rows", [])
        motion = delivery["checks"]["motion"]
        if (
            delivery["checks"]["failed"]
            or motion.get("status") in FAILED_CHECK_STATES
            or any(
                row.get("status") in FAILED_CHECK_STATES
                or row.get("failed_interface_ids")
                or any(duty.get("status") in FAILED_CHECK_STATES for duty in row.get("duties", []))
                for row in interface_rows
            )
        ):
            lines.append(
                "**Saved checks failed. Review the failures below before manufacture or use.**"
            )
        if build_time_findings:
            lines.extend(
                [
                    "",
                    "The saved build-time findings in this package do not report the later final contract proof.",
                    "Final proof runs after automatic package capture and exact readback. It is a separate result; no final pass or failure is claimed in this ZIP.",
                    "Read get_contract_verification for this project in the authenticated workspace. Match its project, revision and captured download SHA-256; a later export can have a different receipt.",
                ]
            )
            project_id = (guidance_context or {}).get("basis", {}).get("session_id")
            if isinstance(project_id, str) and re.fullmatch(r"[a-f0-9]{32}", project_id):
                lines.append(
                    f'Current proof reader: `get_contract_verification(project_id="{project_id}")` or authenticated `GET /workspace/sessions/{project_id}/contract-verification`.'
                )
        if motion:
            lines.extend(["", "### Current sampled motion evidence", ""])
            lines.append("Motion check status: " + str(motion.get("status", "unverified")) + ".")
            if motion.get("reason"):
                lines.append(str(motion["reason"]))
            if type(motion.get("collision_count")) is int:
                lines.append("Saved collision count: " + str(motion["collision_count"]) + ".")
            if motion.get("scope"):
                lines.append(str(motion["scope"]))
            for collision in motion.get("collisions", []):
                literal("Saved collision witness:", json.dumps(collision, ensure_ascii=False))
            for error in motion.get("errors", []):
                literal("Saved motion error:", str(error))
            lines.append(
                "Sampled motion does not establish continuous clearance or physical function."
            )
        facts = guidance_context.get("facts") or {}
        setup = delivery["print_files"]
        if setup["status"] != "unavailable":
            lines += ["", "### Separate prepared print files", ""]
            lines.append(
                "The plate-aligned STL files use a separate print package. "
                "The original CAD/STL files in this project keep their original frames."
            )
            lines.append("Print setup status: " + str(setup["status"]))
            for name, item in setup["files"].items():
                quantity = item["quantity"]
                noun = "copy" if quantity == 1 else "copies"
                entry = f"- {name}: {quantity} {noun}"
                if not current_guidance:
                    entry += f"; SHA-256 {item['sha256']}"
                lines.append(entry)
            lines.append(setup["scope"])
        else:
            lines.append("No current prepared print package is supplied.")
        historical_checks = facts.get("historical_check_amendments") or []
        if historical_checks:
            lines.extend(["", "### Historical check corrections", ""])
            lines.append(
                "These archived results describe earlier predicates. They do not change current check results or prove physical performance."
            )
            for check in historical_checks:
                lines.append(
                    "Historical check "
                    + str(check.get("check_id"))
                    + " ("
                    + str(check.get("tool"))
                    + "): "
                    + str(check.get("historical_status", "unverified"))
                    + "; "
                    + str(check.get("verification", "unverified_historical"))
                    + "."
                )
                if check.get("algorithm_scope"):
                    literal("Original measurement scope:", str(check["algorithm_scope"]))
                literal("Historical receipt limit:", str(check.get("reason", "Unverified history")))
                amendment = check.get("amendment") or {}
                literal(
                    "Authored correction reason (unverified):",
                    str(amendment.get("reason", "Not supplied")),
                )
                literal(
                    "Authored correction basis (unverified):",
                    str(amendment.get("basis", "Not supplied")),
                )
        compiled_instructions()
        procedure_proposals()
        if not current_guidance:
            lines.append(
                "These instructions have no current saved-facts review. Review before use."
            )
        if not current_guidance:
            lines += ["", "## Current saved CAD facts", ""]
        labels = {
            "extents_mm": "Overall X, Y, Z size (mm)",
            "bbox_min": "Overall minimum X, Y, Z coordinates (mm)",
            "bbox_max": "Overall maximum X, Y, Z coordinates (mm)",
            "volume_mm3": "Saved volume (mm3)",
            "solid_count": "Saved solid count",
            "step_valid": "Saved STEP validity",
            "hole_count": "Saved hole count within its measurement scope",
        }
        for key, value in (facts.get("measured") or {}).items():
            if current_guidance:
                continue
            lines.append(f"- {labels.get(key, key)}: {json.dumps(value, allow_nan=False)}")
        if not current_guidance:
            lines += ["", "### Declared dimensions", ""]
            lines.append(
                "Named input and derived values are declarations, not measured dimensions."
            )
            declaration_sets = [("Overall design", facts.get("declared_variables") or {})]
            declaration_sets.extend(
                ("Part " + str(part["part_id"]), part) for part in facts.get("parts", [])
            )
            for label, declarations in declaration_sets:
                for key, description in (("inputs", "Inputs"), ("evaluated", "Evaluated values")):
                    if declarations.get(key):
                        literal(
                            label + " / " + description + ":",
                            json.dumps(declarations[key], indent=2),
                        )
        if not current_guidance and facts.get("measured_scope"):
            lines.append(facts["measured_scope"])
        declarations = facts.get("declared_variables") or {}
        if not current_guidance and (
            declarations.get("inputs_truncated") or declarations.get("evaluated_truncated")
        ):
            lines.append(
                "This variable display is partial. The complete editable document is included."
            )
        if not current_guidance and facts.get("parts_truncated"):
            lines.append("This part display is partial. The complete parts list is included.")

    lines += ["", "## Files and quantities", ""]
    if current_guidance:
        lines.append("The files listed below are included in this package.")
    for part in report["make"]:
        files = [
            f"`{name}` ({label})"
            for ext, label in (("stl", "slicer mesh"), ("step", "CAD geometry"))
            for name, entry in file_inventory["geometry"].items()
            if entry["role"] == "make"
            and entry["part_id"] == part["part_id"]
            and name.endswith("." + ext)
            and name in members
        ]
        noun = "copy" if part["quantity"] == 1 else "copies"
        lines.append(f"- {part['part_id']}: make {part['quantity']} {noun}; " + ", ".join(files))
    if not report["make"]:
        lines.append("No parts to make are declared.")
    if not any(name.startswith("make/") and name.endswith(".stl") for name in members):
        lines.append("No STL print file is included.")
    lines += ["", "## Materials and hardware", ""]
    for part in report["make"]:
        material = part["production"].get("material")
        value = (
            "Not declared."
            if not material
            else material["name"]
            + ("\nGrade: " + material["grade"] if material.get("grade") else "")
        )
        literal(f"Declared material for {part['part_id']}:", value)
    if not report["buy"]:
        lines.append(
            "No hardware entries are saved in the parts list. "
            "Hardware requirements may appear in the saved instructions. "
            "This does not mean hardware is unnecessary."
        )
    for part in report["buy"]:
        details = [f"Declared name: {part['name']}", f"Quantity: {part['quantity']}"]
        procurement = dict(part["production"].get("procurement") or {})
        if part.get("catalog_id") and not procurement.get("catalog_id"):
            procurement["catalog_id"] = part["catalog_id"]
        for key, label in (
            ("supplier", "Supplier"),
            ("part_number", "Part number"),
            ("catalog_id", "Catalog ID"),
        ):
            if current_guidance and key == "catalog_id":
                continue
            if procurement.get(key):
                details.append(f"{label}: {procurement[key]}")
        literal(f"Declared bought part {part['part_id']}:", "\n".join(details))
    lines += [
        "",
        "## Reference-only parts",
        "",
        "Do not print these parts or the overall assembly.",
    ]
    for part in report["reference"]:
        literal(
            part["part_id"] + " (declared reference):",
            f"Declared name: {part['name']}\nQuantity: {part['quantity']}",
        )
    if not report["reference"]:
        lines.append("No reference-only parts are declared.")
    if not current_guidance and "customer-guidance.json" not in members:
        lines += [
            "",
            "## Unverified saved author notes",
            "",
            "These literal declarations can include old tasks or default wording.",
            "They are not current customer steps.",
            "File coordinates alone do not establish a print orientation.",
        ]
        sources = {p.id: p.source for p in document.assembly.parts} if document.assembly else {}
        for part in report["lines"]:
            notes = [sources[part["part_id"]]] if part["part_id"] in sources else []
            if part["production"].get("finish"):
                notes.append(part["production"]["finish"])
            if notes:
                literal(part["part_id"] + ":", "\n\n".join(notes))
    if guidance_context is not None:
        omitted = sorted(
            set(delivery["project_files"]["omitted"]) & file_inventory["geometry"].keys()
        )
        if omitted:
            literal("Project geometry omitted from this download:", "\n".join(omitted))
            lines.append(
                "The saved instructions below can require these omitted files. "
                "Obtain the required files before following the affected steps. "
                "Do not substitute the overall assembly for a made-part print file."
            )
        omitted_metadata = sorted(
            set(delivery["project_files"]["omitted"]) - file_inventory["geometry"].keys()
        )
        if omitted_metadata:
            literal("Project metadata omitted from this download:", "\n".join(omitted_metadata))
            lines.append("Use manifest.json for the exact included files.")
        excluded = [
            name
            for name in file_inventory["excluded_saved_artifacts"]
            if name.endswith((".step", ".stp", ".stl"))
        ]
        if excluded and (not current_guidance or "customer-guidance.json" not in members):
            literal("Saved geometry not packaged as separate files:", "\n".join(excluded))
    if guidance_context is not None:
        lines += [
            "",
            "## Build-time checks and unresolved physical claims"
            if build_time_findings
            else "## Checks that need review",
            "",
        ]
        lines.append("Quoted diagnostic text is not additional proof or a current instruction.")
        findings = (guidance_context.get("facts") or {}).get("requirement_findings") or []
        if not findings:
            lines.append(
                "No saved requirement results are supplied. "
                "Review the saved evidence before manufacture or use."
            )
        else:
            failed = sum(finding.get("status") == "fail" for finding in findings)
            unverified = sum(finding.get("status") == "unverified" for finding in findings)
            lines.append(
                ("Build-time record: " if build_time_findings else "")
                + f"{failed} failed requirement checks; {unverified} unverified requirement checks."
            )
        finding_groups: dict[str, list[dict[str, Any]]] = {}
        for finding in findings:
            if finding.get("status") not in {"fail", "unverified"}:
                continue
            key = json.dumps(
                [finding["status"], finding.get("reason"), finding.get("measured")],
                sort_keys=True,
                allow_nan=False,
            )
            finding_groups.setdefault(key, []).append(finding)
        for group in finding_groups.values():
            finding = group[0]
            status = "Failed" if finding["status"] == "fail" else "Unverified"
            identifiers = ", ".join(
                str(item.get("requirement_id", item.get("id", "Requirement"))) for item in group
            )
            if len(group) > 1 and status == "Unverified" and "customer-guidance.json" in members:
                identifiers = f"({len(group)})"
            details = [str(finding.get("reason") or "No reason supplied.")]
            if finding.get("measured") is not None:
                details.append(
                    "Saved measured value: " + json.dumps(finding["measured"], allow_nan=False)
                )
            noun = "check" if len(group) == 1 else "checks"
            literal(f"{status} {noun} {identifiers}:", "\n".join(details))
        interfaces = delivery["checks"]["interfaces"]
        unresolved_interfaces = [
            row for row in interfaces.get("rows", []) if row.get("status") != "nominal_checked"
        ]
        if unresolved_interfaces:
            lines.append(
                f"{len(unresolved_interfaces)} functional interfaces remain unresolved. "
                "The complete scoped check states remain in the technical records."
            )
        for interface in interfaces.get("rows", []):
            status = interface.get("status", "unverified")
            if status in {"missing_construction", "check_failed", "fail", "failed"}:
                literal(f"Interface {interface['id']}:", "Current construction status: " + status)
            for key, label in (
                ("missing_participant_paths", "Missing parts"),
                ("missing_installed_occurrence_paths", "Missing installed parts or hardware"),
                ("missing_features", "Missing required geometry"),
                ("missing_port_ids", "Missing connection surfaces"),
                ("missing_mate_ids", "Missing connections"),
                ("failed_interface_ids", "Failed interface checks"),
                ("pending_interface_ids", "Unverified interface checks"),
            ):
                if interface.get(key):
                    literal(label + ":", json.dumps(interface[key], ensure_ascii=False))
            for duty in interface.get("duties", []):
                label = f"Interface {interface['id']}, duty {duty['id']}"
                detail = duty.get("reason") or duty.get("status", "Unverified")
                if duty.get("status") in {"fail", "failed", "check_failed"}:
                    literal("Failed " + label + ":", str(detail))
                elif duty.get("status") == "missing_construction":
                    literal("Missing construction for " + label + ":", str(detail))
        functional_mapping = delivery["checks"]["coverage"]
        if functional_mapping.get("missing_requirement_ids"):
            missing = functional_mapping["missing_requirement_ids"]
            if "customer-guidance.json" in members:
                lines.append(
                    f"{len(missing)} requirements have no functional or external review mapping."
                )
            else:
                literal(
                    "Requirements with no functional or external review mapping:",
                    ", ".join(missing),
                )
        declared_notes: dict[str, dict[str, None]] = {
            "missing_inputs": {},
            "physical_validation": {},
            "unsupported_checks": {},
        }

        def declared_limits(value: Any) -> None:
            # Field roles retain source text; they cannot make prose a current task.
            if isinstance(value, dict):
                for key, children in value.items():
                    if key in declared_notes and isinstance(children, list):
                        for text in children:
                            if isinstance(text, str):
                                declared_notes[key].setdefault(text, None)
                    elif isinstance(children, (dict, list)):
                        declared_limits(children)
            elif isinstance(value, list):
                for child in value:
                    declared_limits(child)

        declared_limits(interfaces)
        declared_limits(functional_mapping)
        for detail in declared_notes["missing_inputs"]:
            literal("Missing information (unverified author declaration):", detail)
        lines.append("Physical fit, printing and load capacity are not verified.")

        lines += ["", "## Unverified author notes and history", ""]
        lines.append(
            "Original authored text can contain unsupported claims or old tasks. "
            "It is not part of the bound customer steps."
        )
        if "customer-guidance.json" in members:
            lines += [
                "Literal guidance, declared obligations and instruction history: "
                "[customer-guidance.json](customer-guidance.json).",
                "This file preserves declared unresolved work and earlier instructions as "
                "unverified author notes.",
            ]
        else:

            def authored_guidance(label: str, record: Any) -> None:
                if not isinstance(record, dict):
                    return
                guidance = record.get("guidance")
                if isinstance(guidance, dict):
                    for field, value in guidance.items():
                        if isinstance(value, str):
                            literal(label + " / " + field + " (unverified):", value)

            record = guidance_context.get("record")
            if record is not None:
                literal(
                    "Supplied author record (unverified history):",
                    json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False),
                )
            authored_guidance("Saved authored guidance", record)
            history = guidance_context.get("history") or []
            if history:
                literal(
                    "Supplied authored history records (unverified notes):",
                    json.dumps(history, indent=2, ensure_ascii=False, allow_nan=False),
                )
                for entry in history:
                    if isinstance(entry, dict):
                        authored_guidance("Historical authored guidance", entry.get("record"))
            for key in ("physical_validation", "unsupported_checks"):
                for detail in declared_notes[key]:
                    literal(
                        "Authored " + key + " (unverified history, not a current task):", detail
                    )
    if document.delivery_guidance is not None and "customer-guidance.json" not in members:
        lines += [
            "",
            "## Legacy customer guidance: unbound reference state",
            "",
            "These saved declarations have no current-facts acknowledgment. Review them before use.",
            "These unverified author notes are not current customer steps or physical proof.",
        ]
        for field, label in (
            ("print", "Printing:"),
            ("installation", "Installation:"),
            ("use_and_removal", "Use and removal:"),
            ("unresolved", "Unresolved choices and checks:"),
        ):
            value = getattr(document.delivery_guidance, field)
            literal(label, value if value is not None else "Not supplied.")
    lines += ["", "## Missing information", ""]
    if report["unresolved"]:
        lines.append(
            "Production role is not declared for: "
            + ", ".join(p["part_id"] for p in report["unresolved"])
        )
    if report["gaps"]:
        literal("Declared gaps:", "\n".join(report["gaps"]))
    else:
        lines.append("No production setup gaps are listed. Other fit or use questions can remain.")
    lines += ["", "## More detail", ""]
    for name, label in (
        ("document.json", "Editable source"),
        ("assembly/part.step", "Overall CAD"),
        ("production.json", "Part details"),
        ("purchasing.json", "Purchasing details"),
        ("engineering-evidence.json", "Saved review findings"),
    ):
        if name in members:
            lines.append(f"- {label}: [{name}]({name})")
    for name in (
        "engineering/report.json",
        "engineering/drawing.svg",
        "engineering/inspection.csv",
        "inspection-status.json",
        "import-assets.json",
    ):
        if name in members:
            lines.append(f"- [{name}]({name})")
    if "import-assets.json" not in members:
        lines.append("Imported source inputs, if required by document.json, are not included.")
    return ("\n".join(lines) + "\n").encode("utf-8")


def manufacture_bundle(
    document: DesignDocument,
    source: Path,
    target: Path,
    artifact_hashes: dict[str, str],
    assembly_facts: dict[str, Any] | None = None,
    sources: dict[str, Any] | None = None,
    engineering_evidence: dict[str, Any] | None = None,
    revision_context: dict[str, Any] | None = None,
    engineering_inputs: dict[str, Any] | None = None,
    inspection_bundle: dict[str, Any] | None = None,
    *,
    include_stl: bool = False,
    include_assembly_step: bool = False,
    include_import_assets: bool = False,
    guidance_context: dict[str, Any] | None = None,
) -> Path:
    """Write local make definitions and optional saved overall STEP; never run CAD."""
    from forge_mind.workspace.document import digest
    from forge_mind.workspace.handoff import (
        ASSEMBLY_FRAME,
        ASSEMBLY_SCOPE,
        IMPORT_ASSETS_SCOPE,
        SCOPE,
        import_asset_metadata,
        json_bytes,
        referenced_import_asset_ids,
        verify_manufacturing_handoff,
    )
    from forge_mind.workspace.processing import MAX_JOB_BYTES, regular_file

    report = resolve_sources(production_report(document, assembly_facts), sources)
    if type(include_stl) is not bool:
        raise ValueError("include_stl must be a boolean")
    if type(include_assembly_step) is not bool:
        raise ValueError("include_assembly_step must be a boolean")
    if type(include_import_assets) is not bool:
        raise ValueError("include_import_assets must be a boolean")
    inventory = package_file_inventory(
        document,
        artifact_hashes,
        include_stl=include_stl,
        include_assembly_step=include_assembly_step,
        include_import_assets=include_import_assets,
    )
    make_paths = {
        entry["source_artifact"]: name
        for name, entry in inventory["geometry"].items()
        if entry["role"] == "make"
    }
    import_paths = {
        entry["source_artifact"]: name
        for name, entry in inventory["geometry"].items()
        if entry["role"] == "import_source"
    }
    bounded = include_stl or include_assembly_step or include_import_assets
    if include_stl and not report["make"]:
        if document.assembly:
            raise ValueError("Named STL packaging requires assembly make definitions")
        raise ValueError("STL packaging requires explicitly declared make parts")
    flat_stl = include_stl and not document.assembly
    transferred = 0
    files: list[tuple[str, bytes]] = []
    for name, member in make_paths.items():
        if not name.endswith(".step"):
            continue
        path = source / name
        if bounded and regular_file(path) and path.stat().st_size > MAX_JOB_BYTES - transferred:
            raise ValueError("Manufacturing mesh package exceeds the worker artifact budget")
        if (
            not regular_file(path)
            or name not in artifact_hashes
            or (not bounded and file_sha256(path) != artifact_hashes[name])
        ):
            raise ValueError("Manufacturing definition missing or changed: " + name)
        if bounded:
            with path.open("rb") as stream:
                data = stream.read(MAX_JOB_BYTES - transferred + 1)
            transferred += len(data)
            if transferred > MAX_JOB_BYTES:
                raise ValueError("Manufacturing mesh artifact exceeded the transfer budget")
        else:
            data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != artifact_hashes[name]:
            raise ValueError("Manufacturing definition changed during packaging: " + name)
        files.append((member, data))
    members = dict(files)

    def read_bound(name: str) -> bytes:
        nonlocal transferred
        path = source / name
        if not regular_file(path) or name not in artifact_hashes:
            raise ValueError("Manufacturing mesh artifact missing: " + name)
        remaining = MAX_JOB_BYTES - transferred
        if remaining < 0 or path.stat().st_size > remaining:
            raise ValueError("Manufacturing mesh package exceeds the worker artifact budget")
        with path.open("rb") as stream:
            data = stream.read(remaining + 1)
        if len(data) > remaining:
            raise ValueError("Manufacturing mesh artifact exceeded the transfer budget")
        if hashlib.sha256(data).hexdigest() != artifact_hashes[name]:
            raise ValueError("Manufacturing mesh artifact changed: " + name)
        transferred += len(data)
        return data

    def encoded_json(value: Any) -> bytes:
        if not (include_assembly_step or include_import_assets or flat_stl):
            return json_bytes(value)
        remaining = MAX_JOB_BYTES - sum(len(data) for data in members.values())
        data = bytearray()
        for chunk in json.JSONEncoder(indent=2, sort_keys=True, allow_nan=False).iterencode(value):
            # ensure_ascii defaults true, so character length equals encoded byte length.
            if len(chunk) > remaining - len(data):
                raise ValueError("Project package metadata exceeds the worker artifact budget")
            data.extend(chunk.encode("utf-8"))
        return bytes(data)

    def add_member(name: str, data: bytes) -> None:
        if (include_assembly_step or include_import_assets or flat_stl) and len(
            data
        ) > MAX_JOB_BYTES - sum(len(v) for v in members.values()):
            raise ValueError("Project package exceeds the worker artifact budget")
        members[name] = data

    if include_assembly_step:
        for name, entry in inventory["geometry"].items():
            if entry["role"] == "overall_review":
                members[name] = read_bound(entry["source_artifact"])
    document_bytes = (
        b"" if include_stl or include_import_assets else encoded_json(document.model_dump())
    )
    if include_stl or include_import_assets:
        document_bytes = read_bound("document.json")
        if DesignDocument.model_validate_json(document_bytes) != document:
            raise ValueError("Manufacturing mesh source document differs from requested document")
    if flat_stl:
        members[make_paths["part.stl"]] = read_bound("part.stl")
    elif include_stl:
        from forge_mind.workspace.definition_meshes import (
            MANIFEST_NAME,
            definition_mesh_entry,
            validate_definition_mesh,
        )

        mesh_manifest = read_bound(MANIFEST_NAME)
        manifest_record = json.loads(mesh_manifest)
        if not isinstance(manifest_record, dict):
            raise ValueError("Manufacturing mesh manifest must be an object")
        mesh_revision = manifest_record.get("revision_id")
        if not isinstance(mesh_revision, str):
            raise ValueError("Manufacturing mesh revision identifier is missing")
        if revision_context and revision_context.get("revision_id") != mesh_revision:
            raise ValueError("Manufacturing mesh revision differs from captured project revision")
        failures = []
        for line in report["make"]:
            part_id = line["part_id"]
            stl_name = f"definition_{part_id}.stl"
            try:
                definition_mesh_entry(
                    document_bytes, mesh_manifest, mesh_revision, part_id, artifact_hashes
                )
                stl = read_bound(stl_name)
                validate_definition_mesh(
                    document_bytes,
                    mesh_manifest,
                    mesh_revision,
                    part_id,
                    artifact_hashes,
                    members[make_paths[f"definition_{part_id}.step"]],
                    stl,
                )
            except ValueError as exc:
                failures.append(f"{part_id}: {exc}")
                continue
            members[make_paths[stl_name]] = stl
        if failures:
            raise ValueError("Manufacturing STL package unavailable: " + "; ".join(failures))
        members[MANIFEST_NAME] = mesh_manifest
    if include_import_assets:
        asset_ids = referenced_import_asset_ids(json.loads(document_bytes))
        metadata_bytes = read_bound("assets.json")
        metadata = import_asset_metadata(metadata_bytes, asset_ids)
        add_member("inputs/assets.json", metadata_bytes)
        dependencies = []
        for asset_id in asset_ids:
            name = f"input_{asset_id}.step"
            data = read_bound(name)
            if hashlib.sha256(data).hexdigest() != metadata[asset_id]:
                raise ValueError(
                    "Imported dependency differs from saved asset metadata: " + asset_id
                )
            member = import_paths[name]
            add_member(member, data)
            dependencies.append(
                {
                    "asset_id": asset_id,
                    "file": member,
                    "sha256": metadata[asset_id],
                    "size_bytes": len(data),
                }
            )
        add_member(
            "import-assets.json",
            encoded_json(
                {
                    "schema_version": 1,
                    "document_sha256": hashlib.sha256(document_bytes).hexdigest(),
                    "assets": dependencies,
                    "scope": IMPORT_ASSETS_SCOPE,
                }
            ),
        )
    report["files"] = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in members.items()
        if name.startswith("make/")
    }
    add_member("document.json", document_bytes)
    for operation_member in ("operation-plan.json", "operation-record.json"):
        if operation_member in artifact_hashes:
            add_member(operation_member, read_bound(operation_member))
    add_member("production.json", encoded_json(report))
    add_member("purchasing.json", encoded_json({"lines": report["buy"]}))
    add_member("sources.json", encoded_json(sources or {"claims": [], "artifacts": []}))
    add_member(
        "engineering-evidence.json",
        encoded_json(
            engineering_evidence
            or {
                "physical_readiness": "unverified",
                "scope": "No engineering receipts supplied",
            }
        ),
    )
    if engineering_inputs is not None:
        add_member("engineering-inputs.json", encoded_json(engineering_inputs))
    if guidance_context is not None:
        add_member("customer-guidance.json", encoded_json(guidance_context))
    if (
        inspection_bundle is None
        and revision_context
        and "inspection_plan_hash" in revision_context
    ):
        if revision_context["inspection_plan_hash"] != digest(None):
            raise ValueError("Inspection intent is required for the supplied plan binding")
        inspection_bundle = {
            "plan": None,
            "artifacts": {},
            "status": {
                "status": "unavailable",
                "reason": "No inspection report supplied",
                "physical_readiness": "unverified",
            },
        }
    if inspection_bundle is not None:
        artifacts = inspection_bundle["artifacts"]
        if set(artifacts) not in (set(), {"report.json", "drawing.svg", "inspection.csv"}):
            raise ValueError("Inspection handoff requires exactly the report, drawing and CSV")
        add_member("inspection-intent.json", encoded_json(inspection_bundle["plan"]))
        add_member("inspection-status.json", encoded_json(inspection_bundle["status"]))
        for name, data in artifacts.items():
            add_member("engineering/" + name, data)
    if include_assembly_step:
        # These generated members are guaranteed in a successful final archive.
        # Rendering failure still aborts the capture before publication.
        add_member(
            "README.md",
            _package_readme(
                document, report, set(members) | {"README.md", "manifest.json"}, guidance_context
            ),
        )
    manifest = {
        "schema_version": 1,
        "bundle_kind": "manufacturing_review",
        "document_hash": digest(document.model_dump()),
        "context": revision_context
        or {"status": "unavailable", "scope": "No project snapshot supplied"},
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in sorted(members.items())
        },
        "manufacturing_ready": False,
        "physical_readiness": "unverified",
        "disposition": "Review handoff. Complete saved overall STEP is included separately; only make definitions appear in make/. Resolve declared gaps and review process, tolerances, inspection and purchasing before manufacture."
        if include_assembly_step
        else "Review handoff. Only make definitions are exported; buy and reference geometry are excluded. Resolve declared gaps and review process, tolerances, inspection and purchasing before manufacture.",
        "integrity_scope": SCOPE,
        **({"include_stl": True} if include_stl else {}),
        **({"include_import_assets": True} if include_import_assets else {}),
        **(
            {
                "include_assembly_step": True,
                "assembly": {
                    "file": "assembly/part.step",
                    "sha256": artifact_hashes["part.step"],
                    "frame": ASSEMBLY_FRAME,
                    "units": "mm",
                    "scope": ASSEMBLY_SCOPE,
                },
            }
            if include_assembly_step
            else {}
        ),
    }
    manifest_bytes = encoded_json(manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    # A failed build never replaces a previously usable package.
    import tempfile

    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".zip", delete=False) as temporary:
        pending = Path(temporary.name)
    try:
        with ZipFile(pending, "w", ZIP_DEFLATED) as archive:
            for name, data in sorted(members.items()):
                archive.writestr(name, data)
            archive.writestr("manifest.json", manifest_bytes)
        verify_manufacturing_handoff(pending)
        pending.replace(target)
    finally:
        pending.unlink(missing_ok=True)
    return target
