"""Offline manufacturing package integrity, without extraction or engineering approval."""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

from forge_mind.workspace.document import digest

SCOPE = (
    "Internal package integrity only; unsigned hashes do not authenticate origin or prove "
    "current server state, engineering correctness, manufacturing suitability or physical fitness."
)
ASSEMBLY_FRAME = "cad_z_up_mm_world"
ASSEMBLY_SCOPE = (
    "Complete saved overall geometry in world CAD Z-up millimeter coordinates; "
    "assembly geometry can include made, bought and reference parts. "
    "Manufactured definitions remain separate in local coordinates. No physical approval."
)
IMPORT_ASSETS_SCOPE = (
    "Exact referenced, untransformed STEP source bytes. Source coordinates and units are not "
    "measured by packaging; occurrence placement remains in document.json. Requires a compatible "
    "CAD runtime. Independent regeneration and physical fitness are not verified. Not a full backup."
)


def referenced_import_asset_ids(document: dict[str, Any]) -> list[str]:
    """Include every declared dependency, even suppressed or unused definitions."""
    features = list(document.get("features", []))
    for part in (document.get("assembly") or {}).get("parts", []):
        features.extend(part.get("features", []))
    return sorted({feature["asset_id"] for feature in features if feature.get("asset_id")})


def import_asset_metadata(data: bytes, asset_ids: list[str]) -> dict[str, str]:
    """Validate exact saved metadata identity/hash closure without loading CAD."""
    records = _json(data)
    if not isinstance(records, list) or len(records) > 100:
        raise ValueError("Imported asset metadata must be a bounded list")
    hashes: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Malformed imported asset metadata")
        asset_id, checksum = record.get("id"), record.get("sha256")
        if (
            not isinstance(asset_id, str)
            or re.fullmatch(r"[a-f0-9]{32}", asset_id) is None
            or asset_id in hashes
            or not isinstance(checksum, str)
            or re.fullmatch(r"[a-f0-9]{64}", checksum) is None
        ):
            raise ValueError("Duplicate or malformed imported asset identity/hash")
        hashes[asset_id] = checksum
    if set(hashes) != set(asset_ids):
        raise ValueError("Imported asset metadata differs from declared dependencies")
    return hashes


def _verify_import_assets(
    archive: ZipFile, files: dict[str, Any], document: dict[str, Any], document_bytes: bytes
) -> None:
    asset_ids = referenced_import_asset_ids(document)
    if "import-assets.json" not in files or "inputs/assets.json" not in files:
        raise ValueError("Imported source dependency map or metadata is missing")
    metadata = import_asset_metadata(archive.read("inputs/assets.json"), asset_ids)
    mapping = _json(archive.read("import-assets.json"))
    if (
        not isinstance(mapping, dict)
        or set(mapping) != {"schema_version", "document_sha256", "assets", "scope"}
        or type(mapping.get("schema_version")) is not int
        or mapping["schema_version"] != 1
        or mapping.get("document_sha256") != hashlib.sha256(document_bytes).hexdigest()
        or mapping.get("scope") != IMPORT_ASSETS_SCOPE
        or not isinstance(mapping.get("assets"), list)
        or len(mapping["assets"]) != len(asset_ids)
    ):
        raise ValueError("Imported dependency map binding or scope changed")
    seen: set[str] = set()
    expected_inputs = {"inputs/assets.json"}
    for row in mapping["assets"]:
        if not isinstance(row, dict) or set(row) != {"asset_id", "file", "sha256", "size_bytes"}:
            raise ValueError("Malformed imported dependency entry")
        asset_id = row["asset_id"]
        if not isinstance(asset_id, str) or asset_id not in metadata or asset_id in seen:
            raise ValueError("Duplicate or undeclared imported dependency")
        seen.add(asset_id)
        name = f"inputs/input_{asset_id}.step"
        if (
            row["file"] != name
            or name not in files
            or row["sha256"] != metadata[asset_id]
            or row["sha256"] != files[name]["sha256"]
            or type(row["size_bytes"]) is not int
            or row["size_bytes"] != files[name]["size_bytes"]
        ):
            raise ValueError("Imported dependency bytes/path/size binding changed")
        expected_inputs.add(name)
    if (
        seen != set(asset_ids)
        or {name for name in files if name.startswith("inputs/")} != expected_inputs
    ):
        raise ValueError("Package does not contain exactly its referenced imported inputs")


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("Duplicate JSON key: " + name)
        result[name] = value
    return result


def _json(data: bytes) -> Any:
    def invalid_constant(value: str) -> Any:
        raise ValueError("Invalid JSON constant: " + value)

    return json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid_constant)


def _member_name(name: str) -> bool:
    return bool(
        name
        and "\\" not in name
        and ":" not in name
        and not any(ord(c) < 32 for c in name)
        and not name.startswith("/")
        and all(part not in {"", ".", ".."} for part in name.split("/"))
        and str(PurePosixPath(name)) == name
    )


def _production_rows_with_mesh_default(rows: Any) -> Any:
    """Expand only the new false default in historical production declarations."""
    if not isinstance(rows, list):
        return rows
    normalized = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("production"), dict):
            raise ValueError("Invalid production declaration")
        metadata = dict(row["production"])
        if "deliver_stl" in metadata and type(metadata["deliver_stl"]) is not bool:
            raise ValueError("Invalid production mesh delivery declaration")
        metadata.setdefault("deliver_stl", False)
        normalized.append({**row, "production": metadata})
    return normalized


def handoff_context(
    session: dict[str, Any], revision: dict[str, Any], export_engine: str
) -> dict[str, Any]:
    from forge_mind.workspace.document import DesignDocument

    rid = revision["id"]
    selected = session.get("selected_revision")
    accepted = session.get("accepted_revision")
    proposal = (session.get("proposal") or {}).get("revision_id")
    return {
        "status": "captured",
        "session_id": session["id"],
        "revision_id": rid,
        "document_hash": digest(DesignDocument.model_validate(revision["document"]).model_dump()),
        "document_representation": "Validated current schema; omitted defaults are expanded",
        "stored_document_hash": digest(revision["document"]),
        "measurement_engine_fingerprint": revision["measurement"].get("engine_fingerprint"),
        "export_engine_fingerprint": export_engine,
        "input_version": session["engineering"]["version"],
        "engineering_inputs_hash": digest(session["engineering"]),
        "source_fingerprint": session["source_fingerprint"],
        "inspection_plan_hash": digest(session.get("design_plan")),
        "selected_revision": selected,
        "accepted_revision": accepted,
        "proposal_revision": proposal,
        "matches_accepted_revision": rid == accepted,
        "matches_current_design": rid == (proposal or selected),
        "scope": "Project selection and current engineering inputs at export, not approval.",
    }


def _verify_instruction_context(value: dict[str, Any]) -> None:
    """Recompute optional instruction/history projections; legacy absence stays readable."""
    from forge_mind.workspace.delivery_guidance import input_authority_status
    from forge_mind.workspace.guidance_history import verify_guidance_history_context
    from forge_mind.workspace.instruction_plan import (
        InstructionPlan,
        instruction_catalog,
        resolve_instruction_plan,
    )

    record = value.get("record")
    record = record if isinstance(record, dict) else {}
    fields = {
        "instruction_catalog",
        "instruction_binding",
        "history",
        "history_status",
        "input_authority",
    }
    if not (fields | {"record_provenance"}).intersection(value) and not {
        "instruction_plan",
        "instruction_plan_hash",
        "input_authority_hash",
    }.intersection(record):
        return
    if not fields <= set(value):
        raise ValueError("Customer guidance instruction context is incomplete")
    authority = value["input_authority"]
    if (
        not isinstance(authority, dict)
        or authority.get("status") not in {"clear", "blocked"}
        or authority.get("status") != input_authority_status(authority)
        or not isinstance(authority.get("identity"), dict)
        or authority["identity"].get("revision_id") != value["basis"]["revision_id"]
        or (value.get("basis") or {}).get("input_authority_hash") != digest(authority["identity"])
    ):
        raise ValueError("Customer guidance input authority binding changed")
    verify_guidance_history_context(value)
    if "instruction_plan" in record or "instruction_plan_hash" in record:
        plan = InstructionPlan.model_validate(record.get("instruction_plan"))
        if record.get("instruction_plan") != plan.model_dump(mode="json") or record.get(
            "instruction_plan_hash"
        ) != digest(record["instruction_plan"]):
            raise ValueError("Customer guidance instruction plan hash changed")
    if value["instruction_catalog"] != instruction_catalog(value):
        raise ValueError("Customer guidance instruction catalog changed")
    if value["instruction_binding"] != resolve_instruction_plan(value):
        raise ValueError("Customer guidance instruction binding changed")


def _verify_guidance(
    value: Any,
    context: dict[str, Any],
    files: dict[str, Any],
    document: dict[str, Any],
) -> str:
    """Check captured guidance relationships; unsigned packages cannot attest origin."""
    from forge_mind.workspace.delivery_guidance import SCOPE as GUIDANCE_SCOPE
    from forge_mind.workspace.delivery_guidance import ReviewedDeliveryGuidance

    if not isinstance(value, dict) or value.get("status") not in {"current", "stale", "unbound"}:
        raise ValueError("Malformed customer guidance status")
    basis = value.get("basis")
    if (
        not isinstance(basis, dict)
        or value.get("basis_hash") != digest(basis)
        or context.get("guidance_basis") != basis
        or context.get("guidance_basis_hash") != value["basis_hash"]
        or context.get("guidance_context_hash") != digest(value)
        or value.get("physical_readiness") != "unverified"
        or value.get("scope") != GUIDANCE_SCOPE
        or basis.get("facts_hash") != digest(value.get("facts"))
        or context.get("status") != "captured"
    ):
        raise ValueError("Customer guidance captured basis changed")
    for key, captured_key in (
        ("session_id", "session_id"),
        ("revision_id", "revision_id"),
        ("document_hash", "stored_document_hash"),
        ("engineering_inputs_hash", "engineering_inputs_hash"),
        ("source_fingerprint", "source_fingerprint"),
        ("input_version", "input_version"),
        ("engine_fingerprint", "export_engine_fingerprint"),
        ("measured_engine_fingerprint", "measurement_engine_fingerprint"),
    ):
        if basis.get(key) != context.get(captured_key):
            raise ValueError("Customer guidance identity or input binding changed: " + key)
    artifacts = basis.get("artifact_hashes")
    if "print_setup_record_hash" in basis:
        setup = (value.get("facts") or {}).get("print_setup")
        if (
            not isinstance(setup, dict)
            or setup.get("record_hash") != basis["print_setup_record_hash"]
            or context.get("print_setup_record_hash") != basis["print_setup_record_hash"]
            or (value["status"] == "current" and setup.get("status") != "current")
        ):
            raise ValueError("Customer guidance print setup binding changed")
    if not isinstance(artifacts, dict):
        raise ValueError("Customer guidance saved artifact bindings missing")
    for name, metadata in files.items():
        if name == "assembly/part.step" or name.startswith("make/"):
            artifact_name = name.removeprefix("assembly/").removeprefix("make/")
            if artifacts.get(artifact_name) != metadata["sha256"]:
                raise ValueError("Customer guidance saved geometry binding changed")
    legacy = document.get("delivery_guidance")
    if value.get("legacy_guidance") != legacy or value.get("legacy_guidance_status") != "unbound":
        raise ValueError("Legacy guidance cannot claim a current reference state")
    selected = context.get("selected_revision")
    design_head = context.get("proposal_revision") or selected
    target = basis["revision_id"]
    target_metadata = {
        "selected_revision": selected,
        "accepted_revision": context.get("accepted_revision"),
        "design_revision": design_head,
        "target_kind": "pending_candidate"
        if target == design_head and target != selected
        else "selected_candidate"
        if target == selected
        else "historical",
    }
    if any(key in value for key in target_metadata) and any(
        value.get(key) != expected for key, expected in target_metadata.items()
    ):
        raise ValueError("Customer guidance target differs from captured selection")
    _verify_instruction_context(value)
    record = value.get("record")
    if record is None:
        if value["status"] != "unbound":
            raise ValueError("Customer guidance status requires an acknowledged record")
        return "unbound"
    acknowledged = False
    if isinstance(record, dict):
        try:
            prose = ReviewedDeliveryGuidance.model_validate(record.get("guidance"))
            acknowledged = (
                record.get("basis") == basis
                and record.get("basis_hash") == digest(basis)
                and record.get("reviewed_current_facts") is True
                and record.get("guidance_hash") == digest(prose.model_dump(mode="json"))
                and record.get("scope") == GUIDANCE_SCOPE
                and target in {selected, design_head}
                and basis["measured_engine_fingerprint"] == basis["engine_fingerprint"]
                and (
                    "input_authority" not in value
                    or value["input_authority"].get("status") == "clear"
                )
                and (
                    "input_authority_hash" not in record
                    or (
                        value.get("input_authority", {}).get("status") == "clear"
                        and record["input_authority_hash"] == basis.get("input_authority_hash")
                    )
                )
            )
        except ValueError:
            pass  # Invalid historical declarations remain stale, never current.
    expected = "current" if acknowledged else "stale"
    if value["status"] != expected:
        raise ValueError("Customer guidance falsely claims its reference state")
    return expected


def verify_manufacturing_handoff(path: Path) -> dict[str, Any]:
    """Reject ambiguous archives and verify all members against their unsigned manifest."""
    try:
        with ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if (
                len(set(names)) != len(names)
                or not all(_member_name(n) for n in names)
                or any(info.orig_filename != info.filename for info in infos)
            ):
                raise ValueError("Duplicate or unsafe package member")
            if any(
                info.flag_bits & 1 or (info.external_attr >> 16) & 0o170000 == 0o120000
                for info in infos
            ):
                raise ValueError("Encrypted or linked package member")
            from forge_mind.workspace.processing import MAX_JOB_BYTES

            if archive.getinfo("manifest.json").file_size > MAX_JOB_BYTES:
                raise ValueError("Manufacturing manifest exceeds the worker artifact budget")
            with archive.open("manifest.json") as member:
                manifest_bytes = member.read(MAX_JOB_BYTES + 1)
            if len(manifest_bytes) > MAX_JOB_BYTES:
                raise ValueError("Manufacturing manifest exceeds the worker artifact budget")
            manifest = _json(manifest_bytes)
            if not isinstance(manifest, dict) or (
                manifest.get("schema_version") != 1
                or type(manifest.get("schema_version")) is not int
                or manifest.get("bundle_kind") != "manufacturing_review"
            ):
                raise ValueError("Unsupported manufacturing manifest")
            include_stl = manifest.get("include_stl", False)
            if type(include_stl) is not bool:
                raise ValueError("Invalid manufacturing mesh option")
            include_assembly_step = manifest.get("include_assembly_step", False)
            if type(include_assembly_step) is not bool:
                raise ValueError("Invalid assembly STEP option")
            include_import_assets = manifest.get("include_import_assets", False)
            if type(include_import_assets) is not bool:
                raise ValueError("Invalid imported source option")
            if (include_stl or include_assembly_step or include_import_assets) and sum(
                info.file_size for info in infos
            ) > MAX_JOB_BYTES:
                raise ValueError("Manufacturing mesh package exceeds the worker artifact budget")
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != set(names) - {"manifest.json"}:
                raise ValueError("Manifest does not cover exactly every package member")
            assembly_members = {name for name in files if name.startswith("assembly/")}
            if include_assembly_step:
                assembly = manifest.get("assembly")
                if assembly_members != {"assembly/part.step"} or not isinstance(assembly, dict):
                    raise ValueError("Project package requires exactly its saved assembly STEP")
                if (
                    assembly.get("file") != "assembly/part.step"
                    or assembly.get("frame") != ASSEMBLY_FRAME
                    or assembly.get("units") != "mm"
                    or assembly.get("scope") != ASSEMBLY_SCOPE
                    or assembly.get("sha256") != files["assembly/part.step"].get("sha256")
                ):
                    raise ValueError("Assembly STEP binding or frame changed")
            elif assembly_members or "assembly" in manifest:
                raise ValueError("Assembly STEP requires include_assembly_step=true")
            for name, record in sorted(files.items()):
                if (
                    not isinstance(record, dict)
                    or not isinstance(record.get("sha256"), str)
                    or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                    or type(record.get("size_bytes")) is not int
                    or record["size_bytes"] < 0
                ):
                    raise ValueError("Malformed manifest entry: " + name)
                checksum = hashlib.sha256()
                size = 0
                with archive.open(name) as member:
                    while chunk := member.read(1024 * 1024):
                        checksum.update(chunk)
                        size += len(chunk)
                if checksum.hexdigest() != record["sha256"] or size != record["size_bytes"]:
                    raise ValueError("Package member changed: " + name)
            required = {
                "document.json",
                "production.json",
                "purchasing.json",
                "sources.json",
                "engineering-evidence.json",
            }
            if not required <= set(files):
                raise ValueError("Required manufacturing review member missing")
            context = manifest.get("context")
            if not isinstance(context, dict) or context.get("status") not in {
                "captured",
                "unavailable",
            }:
                raise ValueError("Manufacturing context missing")
            if (
                manifest.get("manufacturing_ready") is not False
                or manifest.get("physical_readiness") != "unverified"
            ):
                raise ValueError("Integrity cannot confer manufacturing or physical approval")
            production = _json(archive.read("production.json"))
            if production.get("manufacturing_ready") is not False:
                raise ValueError("Production setup cannot confer manufacturing approval")
            document_bytes = archive.read("document.json")
            document = _json(document_bytes)
            from forge_mind.workspace.document import DesignDocument
            from forge_mind.workspace.production import production_report

            typed_document = DesignDocument.model_validate(document)
            document_hash = digest(
                typed_document.model_dump() if include_stl or include_import_assets else document
            )
            if manifest.get("document_hash") != document_hash:
                raise ValueError("Document binding changed")
            if include_import_assets:
                _verify_import_assets(archive, files, document, document_bytes)
            elif "import-assets.json" in files or any(name.startswith("inputs/") for name in files):
                raise ValueError("Imported source inputs require include_import_assets=true")
            declarations = production_report(typed_document)
            if any(
                _production_rows_with_mesh_default(production.get(key))
                != _production_rows_with_mesh_default(declarations[key])
                for key in ("lines", "make", "buy", "reference", "unresolved")
            ):
                raise ValueError("Production declarations do not match the editable document")
            expected_make = {
                "make/definition_" + part["part_id"] + ".step"
                if document.get("assembly")
                else "make/part.step"
                for part in production["make"]
            }
            if include_stl and not typed_document.assembly:
                if not production["make"]:
                    raise ValueError("STL package requires an explicitly declared flat make part")
                if "definition-meshes.json" in files:
                    raise ValueError("Named mesh manifest is not valid for a flat STL package")
                expected_make.add("make/part.stl")
            elif include_stl:
                from forge_mind.workspace.definition_meshes import (
                    MANIFEST_NAME,
                    validate_definition_mesh,
                )

                if not typed_document.assembly or not production["make"]:
                    raise ValueError("Named STL package requires assembly make definitions")
                if MANIFEST_NAME not in files:
                    raise ValueError("Manufacturing mesh manifest missing")
                mesh_manifest = archive.read(MANIFEST_NAME)
                mesh_revision = _json(mesh_manifest).get("revision_id")
                if context["status"] == "captured" and context.get("revision_id") != mesh_revision:
                    raise ValueError("Manufacturing mesh revision differs from captured revision")
                mesh_hashes = {
                    name.removeprefix("make/"): record["sha256"] for name, record in files.items()
                }
                for part in production["make"]:
                    part_id = part["part_id"]
                    step_name = f"make/definition_{part_id}.step"
                    stl_name = f"make/definition_{part_id}.stl"
                    if stl_name not in files or step_name not in files:
                        raise ValueError("Manufacturing mesh definition missing: " + part_id)
                    validate_definition_mesh(
                        document_bytes,
                        mesh_manifest,
                        mesh_revision,
                        part_id,
                        mesh_hashes,
                        archive.read(step_name),
                        archive.read(stl_name),
                    )
                    expected_make.add(stl_name)
            elif "definition-meshes.json" in files:
                raise ValueError("Manufacturing mesh manifest requires include_stl=true")
            make_files = {
                name: record["sha256"] for name, record in files.items() if name.startswith("make/")
            }
            if set(make_files) != expected_make or production.get("files") != make_files:
                raise ValueError("Manufacturing definitions do not match production declarations")
            if _json(archive.read("purchasing.json")).get("lines") != production["buy"]:
                raise ValueError("Purchasing lines do not match production declarations")
            evidence = _json(archive.read("engineering-evidence.json"))
            if evidence.get("physical_readiness") != "unverified":
                raise ValueError("Engineering evidence cannot confer physical approval")
            if context["status"] == "captured":
                required_context = {
                    "session_id",
                    "revision_id",
                    "document_hash",
                    "measurement_engine_fingerprint",
                    "export_engine_fingerprint",
                    "input_version",
                    "engineering_inputs_hash",
                    "source_fingerprint",
                    "selected_revision",
                    "accepted_revision",
                    "proposal_revision",
                    "matches_accepted_revision",
                    "matches_current_design",
                }
                if (
                    not required_context <= set(context)
                    or any(
                        not isinstance(context[key], str)
                        or re.fullmatch(r"[a-f0-9]{32}", context[key]) is None
                        for key in ("session_id", "revision_id")
                    )
                    or not isinstance(context["export_engine_fingerprint"], str)
                    or not context["export_engine_fingerprint"]
                ):
                    raise ValueError("Captured context is incomplete")
                if type(context["input_version"]) is not int or context["input_version"] < 0:
                    raise ValueError("Captured input version is invalid")
                inputs = _json(archive.read("engineering-inputs.json"))
                sources = _json(archive.read("sources.json"))
                if (
                    context.get("document_hash") != document_hash
                    or context.get("engineering_inputs_hash") != digest(inputs)
                    or context.get("input_version") != inputs.get("version")
                    or context.get("source_fingerprint") != sources.get("fingerprint")
                    or context.get("revision_id") != evidence.get("revision_id")
                    or not context.get("revision_id")
                    or context.get("matches_accepted_revision")
                    is not (context["revision_id"] == context.get("accepted_revision"))
                    or context.get("matches_current_design")
                    is not (
                        context["revision_id"]
                        == (context.get("proposal_revision") or context.get("selected_revision"))
                    )
                ):
                    raise ValueError("Manufacturing context does not match packaged evidence")
            if (
                "inspection_plan_hash" in context
                or any(name.startswith("engineering/") for name in files)
                or "inspection-status.json" in files
                or "inspection-intent.json" in files
            ):
                if not {"inspection-status.json", "inspection-intent.json"} <= set(files):
                    raise ValueError("Inspection handoff intent and status are required")
                intent = _json(archive.read("inspection-intent.json"))
                inspection_status = _json(archive.read("inspection-status.json"))
                if (
                    inspection_status.get("status") not in {"included", "unavailable"}
                    or inspection_status.get("physical_readiness") != "unverified"
                    or context.get("inspection_plan_hash") != digest(intent)
                ):
                    raise ValueError("Inspection handoff intent or status binding changed")
                inspection_files = {name for name in files if name.startswith("engineering/")}
                if inspection_status["status"] == "unavailable":
                    if inspection_files:
                        raise ValueError(
                            "Unavailable inspection must not include stale report artifacts"
                        )
                else:
                    if inspection_files != {
                        "engineering/report.json",
                        "engineering/drawing.svg",
                        "engineering/inspection.csv",
                    }:
                        raise ValueError("Inspection handoff artifacts are incomplete")
                    report = _json(archive.read("engineering/report.json"))
                    inspection = report.get("inspection_plan") or {}
                    plan_hash = hashlib.sha256(
                        json.dumps(intent, sort_keys=True, allow_nan=False).encode()
                    ).hexdigest()
                    if (
                        report.get("id") != inspection_status.get("report_id")
                        or inspection.get("plan_hash") != plan_hash
                        or inspection.get("revision_id") != context.get("revision_id")
                        or inspection.get("source_fingerprint") != context.get("source_fingerprint")
                        or inspection.get("engine_fingerprint")
                        != context.get("export_engine_fingerprint")
                        or inspection.get("input_version") != context.get("input_version")
                    ):
                        raise ValueError("Inspection report does not match handoff context")
            guidance = {}
            if "customer-guidance.json" in files:
                guidance["guidance_status"] = _verify_guidance(
                    _json(archive.read("customer-guidance.json")), context, files, document
                )
            elif any(
                key in context
                for key in ("guidance_basis", "guidance_basis_hash", "guidance_context_hash")
            ):
                raise ValueError("Captured customer guidance sidecar is missing")
            return {
                "integrity": "verified",
                "members_verified": len(files),
                "context": context,
                "manufacturing_ready": False,
                "physical_readiness": "unverified",
                "scope": SCOPE,
                **guidance,
            }
    except (
        BadZipFile,
        zlib.error,
        KeyError,
        OSError,
        RuntimeError,
        NotImplementedError,
        TypeError,
        AttributeError,
    ) as exc:
        raise ValueError("Invalid manufacturing handoff: " + str(exc)) from exc
