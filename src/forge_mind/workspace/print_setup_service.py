"""Owned measured print setup and immutable all-made oriented mesh packages."""

from __future__ import annotations

import json
import tempfile
import uuid
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from forge_mind.workspace.completion import _hash
from forge_mind.workspace.definition_meshes import (
    MANIFEST_NAME,
    _json_object,
    _selected_counts,
    validate_definition_mesh,
)
from forge_mind.workspace.document import digest
from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.handoff import json_bytes
from forge_mind.workspace.measure import verify_artifacts
from forge_mind.workspace.models import Measurement
from forge_mind.workspace.print_setup import sha, validate_transform
from forge_mind.workspace.print_setup_models import (
    ALGORITHM,
    SCOPE,
    PrintSetupRequest,
    PrintSetupResult,
)
from forge_mind.workspace.processing import MAX_JOB_BYTES
from forge_mind.workspace.product_sources import read_sources, source_fingerprint
from forge_mind.workspace.project_delivery import (
    MAX_PACKAGE_BYTES,
    MAX_RECEIPT_BYTES,
    RETENTION,
    _id,
    _read,
    _safe_directory,
)
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3

    from forge_mind.workspace.agent import WorkspaceService

WORKER_MODULE = "forge_mind.workspace.print_setup"
WORKER_TIMEOUT = 180.0


def _locked_basis(
    service: WorkspaceService, con: sqlite3.Connection, sid: str, owner: str, rid: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _id(sid)
    _id(rid)
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    session = json.loads(row[0])
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    if row is None:
        raise WorkspaceError("Print setup requires a saved structured revision")
    revision = json.loads(row[0])
    measurement = Measurement.model_validate(revision.get("measurement"))
    if not revision.get("document") or not measurement.geometry_valid:
        raise WorkspaceError("Print setup requires valid saved structured geometry")
    directory = service.store.directory(sid, rid)
    _safe_directory(directory)
    if not verify_artifacts(directory, measurement):
        raise WorkspaceError("Print source artifacts changed or are missing")
    row = con.execute(
        "SELECT version,document FROM engineering_inputs WHERE session=?", (sid,)
    ).fetchone()
    engineering = (
        {"version": row[0], "inputs": json.loads(row[1])} if row else {"version": 0, "inputs": None}
    )
    effective_contract = session.get("contract")
    for name in ("proposal", "resume_candidate"):
        candidate = session.get(name) or {}
        if candidate.get("revision_id") == rid:
            effective_contract = candidate.get("contract", effective_contract)
    proposal = session.get("proposal") or {}
    plan = session.get("design_plan")
    captured = (revision.get("operation_record") or {}).get("plan_snapshot")
    if proposal.get("revision_id") == rid and rid != session.get("selected_revision"):
        pending = proposal.get("operation_plan_snapshot")
        if session.get("authoring_mode") == "native_operations" and pending is None:
            raise WorkspaceError("Pending native print setup requires its captured operation plan")
        if pending is not None:
            if not isinstance(pending, dict) or pending != captured:
                raise WorkspaceError("Pending print setup plan differs from its saved revision")
            plan = pending.get("plan")
    basis = {
        "session_id": sid,
        "revision_id": rid,
        "algorithm": ALGORITHM,
        "document_hash": digest(revision["document"]),
        "artifact_hashes": measurement.artifact_hashes,
        "engine_fingerprint": engine_fingerprint(),
        "measured_engine_fingerprint": measurement.engine_fingerprint,
        "source_fingerprint": source_fingerprint(con, sid),
        "contract_hash": _hash(revision.get("contract", session.get("contract"))),
        "effective_contract_hash": _hash(effective_contract),
        "engineering_inputs_hash": digest(engineering),
        "plan_hash": _hash(plan),
    }
    return session, revision, basis


def _capture(service: WorkspaceService, sid: str, owner: str, rid: str) -> dict[str, Any]:
    read_sources(service.store, sid, owner)  # Initialize before the read transaction.
    with service.store.connect() as con:
        con.execute("BEGIN")
        session, revision, basis = _locked_basis(service, con, sid, owner, rid)
    return {"session": session, "revision": revision, "basis": basis}


def _required(document_bytes: bytes) -> dict[str, int]:
    document = _json_object(document_bytes, "Print source document")
    assembly = document.get("assembly")
    if not isinstance(assembly, dict):
        raise WorkspaceError("Print setup requires saved made definition meshes in an assembly")
    required = deepcopy(document)
    for part in required["assembly"].get("parts", []):
        production = part.get("production", {})
        production["deliver_stl"] = production.get("role") == "make"
        part["production"] = production
    counts = _selected_counts(required)
    return {part_id: quantity for part_id, quantity in counts.items() if quantity > 0}


def _stage(
    service: WorkspaceService,
    sid: str,
    rid: str,
    captured: dict[str, Any],
    directory: Path,
    part_id: str | None,
) -> PrintSetupRequest:
    source = service.store.directory(sid, rid)
    document = _read(source / "document.json", MAX_JOB_BYTES)
    manifest = _read(source / MANIFEST_NAME, MAX_JOB_BYTES)
    required = _required(document)
    if not required or (part_id is not None and part_id not in required):
        raise WorkspaceError("Print setup requires an occurring made definition")
    selected = sorted(required) if part_id is None else [part_id]
    hashes = captured["basis"]["artifact_hashes"]
    staged = {"document.json": document, MANIFEST_NAME: manifest}
    for name in selected:
        step_name, stl_name = f"definition_{name}.step", f"definition_{name}.stl"
        step, stl = (
            _read(source / step_name, MAX_JOB_BYTES),
            _read(source / stl_name, MAX_JOB_BYTES),
        )
        entry = validate_definition_mesh(document, manifest, rid, name, hashes, step, stl)
        if entry["quantity"] != required[name]:
            raise WorkspaceError("Print mesh quantity differs from the made occurrence count")
        staged[step_name], staged[stl_name] = step, stl
    if sum(map(len, staged.values())) > MAX_JOB_BYTES:
        raise WorkspaceError("Print staged sources exceed their aggregate byte budget")
    for name, data in staged.items():
        with (directory / name).open("xb") as output:
            output.write(data)
    return PrintSetupRequest(
        operation="inspect" if part_id is not None else "prepare",
        revision_id=rid,
        part_ids=selected,
        artifact_hashes=hashes,
    )


def _worker_result(directory: Path, request: PrintSetupRequest) -> dict[str, Any]:
    result = PrintSetupResult.model_validate(
        json.loads(_read(directory / "report.json", MAX_RECEIPT_BYTES))
    ).model_dump(mode="json")
    if (
        result["revision_id"] != request.revision_id
        or result["operation"] != request.operation
        or set(result["parts"]) != set(request.part_ids)
        or result["scope"] != SCOPE
    ):
        raise WorkspaceError("Print worker result does not match the captured request")
    for part_id, part in result["parts"].items():
        source = part["source"]
        entry = validate_definition_mesh(
            _read(directory / "document.json", MAX_JOB_BYTES),
            _read(directory / MANIFEST_NAME, MAX_JOB_BYTES),
            request.revision_id,
            part_id,
            request.artifact_hashes,
            _read(directory / f"definition_{part_id}.step", MAX_JOB_BYTES),
            _read(directory / f"definition_{part_id}.stl", MAX_JOB_BYTES),
        )
        expected = {
            "revision_id": request.revision_id,
            "part_id": part_id,
            "quantity": entry["quantity"],
            "body_count": entry["body_count"],
            "document_sha256": request.artifact_hashes["document.json"],
            "manifest_sha256": request.artifact_hashes[MANIFEST_NAME],
            "step_file": entry["step_file"],
            "step_sha256": entry["step_sha256"],
            "stl_file": entry["stl_file"],
            "stl_sha256": entry["stl_sha256"],
            "stl_bytes": entry["stl_bytes"],
            "frame": "definition_local",
            "units": "mm",
        }
        if source != expected:
            raise WorkspaceError("Print worker changed source mesh binding")
        candidates = part["candidates"]
        seen = set()
        for candidate in candidates:
            candidate_hash = candidate["candidate_hash"]
            if candidate_hash in seen:
                raise WorkspaceError("Print candidate identities must be unique")
            seen.add(candidate_hash)
            validate_transform(candidate["transform"])
            record = {
                name: candidate[name]
                for name in ("plane", "transform", "status", "reason", "observations")
            }
            if digest({"algorithm": ALGORITHM, "source": source, **record}) != candidate_hash:
                raise WorkspaceError("Print candidate hash does not match its observations")
        selected = [
            candidate
            for candidate in candidates
            if candidate["candidate_hash"] == part["recommended_candidate_hash"]
        ]
        if part["status"] == "measured" and (
            len(selected) != 1 or selected[0]["status"] != "measured"
        ):
            raise WorkspaceError("Measured print setup needs one explicit measured candidate")
    expected_files = (
        {f"print_{name}.stl" for name in request.part_ids}
        if result["status"] == "prepared"
        else set()
    )
    if set(result["files"]) != expected_files:
        raise WorkspaceError("Print worker did not produce the complete required mesh set")
    if result["status"] == "prepared" and any(
        part["status"] != "measured" for part in result["parts"].values()
    ):
        raise WorkspaceError("Prepared print setup has an unresolved made definition")
    for name, metadata in result["files"].items():
        data = _read(directory / name, MAX_JOB_BYTES)
        part_id = name.removeprefix("print_").removesuffix(".stl")
        part = result["parts"][part_id]
        if (
            metadata.get("sha256") != sha(data)
            or metadata.get("bytes") != len(data)
            or metadata.get("part_id") != part_id
            or metadata.get("quantity") != part["source"]["quantity"]
            or metadata.get("source_stl_sha256") != part["source"]["stl_sha256"]
            or metadata.get("candidate_hash") != part["recommended_candidate_hash"]
            or metadata.get("frame") != "plate_z_up"
            or metadata.get("units") != "mm"
        ):
            raise WorkspaceError("Derived print file differs from its measured receipt")
    return result


def _run(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    captured: dict[str, Any],
    directory: Path,
    part_id: str | None,
    timeout: float,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    basis = captured["basis"]
    if basis["engine_fingerprint"] != basis["measured_engine_fingerprint"]:
        raise WorkspaceError("Saved geometry uses a different engine; rebuild before print setup")
    if basis["contract_hash"] != basis["effective_contract_hash"]:
        raise WorkspaceError("Saved print geometry differs from its active requirements")
    request = _stage(service, sid, rid, captured, directory, part_id)
    service.executor.run_typed_tool(
        directory,
        request.model_dump(mode="json"),
        module=WORKER_MODULE,
        timeout=timeout,
        cancelled=cancelled,
    )
    if cancelled():
        raise WorkspaceError("Print setup cancelled before publishing")
    result = _worker_result(directory, request)
    if _capture(service, sid, owner, rid)["basis"] != basis:
        raise WorkspaceError("Print inputs changed during measurement; inspect again")
    return result


def inspect_print_setup(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    part_id: str,
    *,
    timeout: float = WORKER_TIMEOUT,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Read-only inspection: no revision, guidance, acceptance or setup-record writes."""
    caller_cancelled = cancelled

    def stopped() -> bool:
        return caller_cancelled() or service.stop.is_set()

    cancelled = stopped
    if cancelled():
        raise WorkspaceError("Print setup cancelled before inspection")
    captured = _capture(service, sid, owner, rid)
    with tempfile.TemporaryDirectory(prefix="forge-print-inspect-") as temporary:
        result = _run(
            service, sid, owner, rid, captured, Path(temporary), part_id, timeout, cancelled
        )
    return {**result, "basis": captured["basis"], "basis_hash": digest(captured["basis"])}


def _location(service: WorkspaceService, sid: str, package_id: str) -> Path:
    _id(sid)
    _id(package_id)
    directory = service.store.root / "print-exports" / sid / package_id
    _safe_directory(directory)
    return directory


def _zip_member(archive: ZipFile, name: str, data: bytes) -> None:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    archive.writestr(info, data)


def prepare_print_files(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    *,
    timeout: float = WORKER_TIMEOUT,
    cancelled: Callable[[], bool] = lambda: False,
) -> dict[str, Any]:
    """Publish a complete immutable made mesh set and atomically bind the current head."""
    caller_cancelled = cancelled

    def stopped() -> bool:
        return caller_cancelled() or service.stop.is_set()

    cancelled = stopped
    if cancelled():
        raise WorkspaceError("Print setup cancelled before preparation")
    captured = _capture(service, sid, owner, rid)
    head = (captured["session"].get("proposal") or {}).get("revision_id") or captured[
        "session"
    ].get("selected_revision")
    if head != rid:
        raise WorkspaceError("Only the current design head can prepare print files")
    with service.store.connect() as con:
        con.execute("BEGIN")
        existing = _locked_print_setup(service, con, sid, owner, rid)
        if existing["status"] == "current":
            if cancelled():
                raise WorkspaceError("Print setup cancelled before returning retained package")
            return dict(existing["record"]["package"])
    package_id = uuid.uuid4().hex
    directory = _location(service, sid, package_id)
    directory.mkdir(parents=True, exist_ok=False)
    package = directory / "print-setup.zip"
    descriptor = directory / "receipt.json"
    try:
        with tempfile.TemporaryDirectory(prefix="forge-print-prepare-") as temporary:
            scratch = Path(temporary)
            result = _run(service, sid, owner, rid, captured, scratch, None, timeout, cancelled)
            if result["status"] != "prepared":
                directory.rmdir()
                return {
                    **result,
                    "project_id": sid,
                    "basis": captured["basis"],
                    "basis_hash": digest(captured["basis"]),
                }
            files = {
                f"print/definition_{metadata['part_id']}.stl": deepcopy(metadata)
                for metadata in result["files"].values()
            }
            selected = {
                part_id: part["recommended_candidate_hash"]
                for part_id, part in result["parts"].items()
            }
            result_data = json_bytes(result)
            readme = (
                "Forge-Mind measured print setup\n\n"
                "All print/*.stl files use millimeters and plate +Z. Import each definition separately.\n"
                "Print the stated quantity. These derived files preserve the original geometry.\n"
                "Original source meshes and STEP remain separate, unchanged project files.\n"
                "Printer model and process settings are unresolved. No G-code is supplied.\n"
                + SCOPE
                + "\n\n"
                + "\n".join(
                    f"{name}: quantity {metadata['quantity']}" for name, metadata in files.items()
                )
                + "\n"
            ).encode("utf-8")
            members = {
                name: {"sha256": value["sha256"], "bytes": value["bytes"]}
                for name, value in files.items()
            }
            members.update(
                {
                    "setup.json": {"sha256": sha(result_data), "bytes": len(result_data)},
                    "README.txt": {"sha256": sha(readme), "bytes": len(readme)},
                }
            )
            manifest = {
                "schema_version": 1,
                "project_id": sid,
                "revision_id": rid,
                "status": "prepared",
                "basis": captured["basis"],
                "basis_hash": digest(captured["basis"]),
                "selected_candidates": selected,
                "files": files,
                "members": members,
                "result_sha256": sha(result_data),
                "scope": SCOPE,
                "manufacturing_ready": False,
                "physical_readiness": "unverified",
            }
            manifest_data = json_bytes(manifest)
            if (
                sum(value["bytes"] for value in members.values()) + len(manifest_data)
                > MAX_JOB_BYTES
            ):
                raise WorkspaceError("Print package exceeds its uncompressed byte budget")
            with ZipFile(package, "x") as archive:
                _zip_member(archive, "manifest.json", manifest_data)
                _zip_member(archive, "setup.json", result_data)
                _zip_member(archive, "README.txt", readme)
                for name, metadata in files.items():
                    _zip_member(
                        archive,
                        name,
                        _read(scratch / f"print_{metadata['part_id']}.stl", MAX_JOB_BYTES),
                    )
        data = _read(package, MAX_PACKAGE_BYTES)
        receipt = {
            **manifest,
            "package_id": package_id,
            "sha256": sha(data),
            "bytes": len(data),
            "manifest_sha256": sha(manifest_data),
            "retention": RETENTION,
        }
        encoded = json_bytes(receipt)
        if len(encoded) > MAX_RECEIPT_BYTES:
            raise WorkspaceError("Print receipt exceeds its metadata byte budget")
        with descriptor.open("xb") as output:
            output.write(encoded)
        record = {
            "basis": captured["basis"],
            "basis_hash": digest(captured["basis"]),
            "selected_candidates": selected,
            "files": files,
            "package": receipt,
            "result": result,
            "result_sha256": sha(result_data),
            "scope": SCOPE,
        }
        with service.store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            session, _, basis = _locked_basis(service, con, sid, owner, rid)
            current = (session.get("proposal") or {}).get("revision_id") or session.get(
                "selected_revision"
            )
            if current != rid or basis != captured["basis"] or cancelled():
                raise WorkspaceError("Print design or inputs changed before publication")
            records = session.get("print_setup_records", {})
            if not isinstance(records, dict):
                raise WorkspaceError("Print setup revision records are invalid")
            session["print_setup_records"] = {**records, rid: record}
            con.execute(
                "UPDATE sessions SET document=? WHERE id=? AND owner=?",
                (json.dumps(session, allow_nan=False), sid, owner),
            )
        return receipt
    except BaseException:
        descriptor.unlink(missing_ok=True)
        package.unlink(missing_ok=True)
        if directory.exists():
            directory.rmdir()
        raise


def read_print_export(
    service: WorkspaceService,
    sid: str,
    owner: str,
    package_id: str,
    sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Owned immutable download; it remains a captured file after later design changes."""
    service.store.session_record(sid, owner)
    directory = _location(service, sid, package_id)
    receipt = _json_object(_read(directory / "receipt.json", MAX_RECEIPT_BYTES), "Print receipt")
    if (
        receipt.get("project_id") != sid
        or receipt.get("package_id") != package_id
        or receipt.get("sha256") != sha256
    ):
        raise WorkspaceError("Print package identity differs from its requested resource")
    data = _read(directory / "print-setup.zip", MAX_PACKAGE_BYTES)
    if sha(data) != sha256 or len(data) != receipt.get("bytes"):
        raise WorkspaceError("Print package bytes differ from their immutable receipt")
    import io

    with ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if (
            len({entry.filename for entry in entries}) != len(entries)
            or sum(entry.file_size for entry in entries) > MAX_JOB_BYTES
        ):
            raise WorkspaceError("Print package has duplicate members or exceeds its byte budget")
        manifest_data = archive.read("manifest.json")
        manifest = _json_object(manifest_data, "Print manifest")
        if receipt.get("manifest_sha256") != sha(manifest_data) or any(
            receipt.get(key) != value for key, value in manifest.items()
        ):
            raise WorkspaceError("Print package receipt differs from its captured manifest")
        if (
            manifest.get("status") != "prepared"
            or manifest.get("physical_readiness") != "unverified"
            or manifest.get("manufacturing_ready") is not False
            or manifest.get("scope") != SCOPE
        ):
            raise WorkspaceError("Print package cannot claim physical or manufacturing approval")
        if digest(manifest.get("basis")) != manifest.get("basis_hash"):
            raise WorkspaceError("Print package basis hash changed")
        members = manifest.get("members")
        if not isinstance(members, dict) or set(archive.namelist()) != set(members) | {
            "manifest.json"
        }:
            raise WorkspaceError("Print package does not contain its complete exact member set")
        for name, metadata in members.items():
            content = archive.read(name)
            if len(content) != metadata.get("bytes") or sha(content) != metadata.get("sha256"):
                raise WorkspaceError("Print package member differs from its receipt")
        result = PrintSetupResult.model_validate(
            _json_object(archive.read("setup.json"), "Print setup")
        ).model_dump(mode="json")
        if (
            sha(archive.read("setup.json")) != manifest.get("result_sha256")
            or result["status"] != "prepared"
            or result["revision_id"] != manifest.get("revision_id")
        ):
            raise WorkspaceError("Print setup evidence differs from its package")
        expected_files = {f"print/definition_{part_id}.stl" for part_id in result["parts"]}
        if set(manifest.get("files", {})) != expected_files or set(members) != expected_files | {
            "setup.json",
            "README.txt",
        }:
            raise WorkspaceError("Print package is missing a required made definition")
        if manifest.get("selected_candidates") != {
            part_id: part["recommended_candidate_hash"] for part_id, part in result["parts"].items()
        }:
            raise WorkspaceError("Print package candidate selection changed")
        for name, metadata in manifest["files"].items():
            part_id = metadata.get("part_id")
            if metadata != result["files"].get(f"print_{part_id}.stl") or members[name] != {
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
            }:
                raise WorkspaceError("Print package derived mesh bindings changed")
    return data, receipt


def _locked_print_setup(
    service: WorkspaceService,
    con: sqlite3.Connection,
    sid: str,
    owner: str,
    rid: str,
) -> dict[str, Any]:
    """Read exact setup evidence inside the caller's owner transaction; no DB writes."""
    row = con.execute(
        "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
    ).fetchone()
    if row is None:
        raise WorkspaceError("workspace not found")
    session = json.loads(row[0])
    records = session.get("print_setup_records", {})
    if not isinstance(records, dict):
        raise WorkspaceError("Print setup records are invalid")
    record = records.get(rid)
    if record is None:
        return {
            "status": "unbound",
            "record": None,
            "record_hash": None,
            "physical_readiness": "unverified",
        }
    if not isinstance(record, dict) or not isinstance(record.get("package"), dict):
        raise WorkspaceError("Print setup record is invalid")
    package = record["package"]
    if package.get("revision_id") != rid:
        raise WorkspaceError("Print setup record belongs to a different revision")
    _, receipt = read_print_export(
        service, sid, owner, package.get("package_id"), package.get("sha256")
    )
    if (
        receipt != package
        or record.get("scope") != SCOPE
        or any(
            record.get(key) != receipt.get(key)
            for key in ("basis", "basis_hash", "selected_candidates", "files", "result_sha256")
        )
        or sha(json_bytes(record.get("result"))) != record.get("result_sha256")
    ):
        raise WorkspaceError("Print setup record differs from its immutable package")
    _, _, basis = _locked_basis(service, con, sid, owner, rid)
    head = (session.get("proposal") or {}).get("revision_id") or session.get("selected_revision")
    current = (
        head == rid and record.get("basis") == basis and record.get("basis_hash") == digest(basis)
    )
    return {
        "status": "current" if current else "stale",
        "record": record,
        "record_hash": digest(record),
        "physical_readiness": "unverified",
    }


def print_setup_context(
    service: WorkspaceService, sid: str, owner: str, rid: str
) -> dict[str, Any]:
    read_sources(service.store, sid, owner)
    with service.store.connect() as con:
        con.execute("BEGIN")
        return _locked_print_setup(service, con, sid, owner, rid)
