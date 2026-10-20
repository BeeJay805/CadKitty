"""Owned immutable review packages; saved-file delivery never starts generation."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zipfile import ZipFile

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.engineering_service import inspection_handoff
from forge_mind.workspace.handoff import (
    handoff_context,
    json_bytes,
    referenced_import_asset_ids,
    verify_manufacturing_handoff,
)
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.processing import MAX_JOB_BYTES, regular_file
from forge_mind.workspace.product_service import product_review
from forge_mind.workspace.production import manufacture_bundle
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

MAX_PACKAGE_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
RETENTION = "Captured private review artifact; no automatic expiry. Hosted retention and deletion are not qualified."


def _id(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise WorkspaceError("Invalid project, revision or package identifier")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_directory(path: Path) -> None:
    for parent in (path, *path.parents):
        if parent.is_symlink() or (
            parent.exists()
            and getattr(parent.lstat(), "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        ):
            raise WorkspaceError("Package directory contains a link or reparse point")


def _read(path: Path, maximum: int) -> bytes:
    if not regular_file(path, maximum):
        raise WorkspaceError("Package file is missing, linked or oversized")
    with path.open("rb") as stream:
        data = stream.read(maximum + 1)
    if len(data) > maximum:
        raise WorkspaceError("Package file exceeds its transfer budget")
    return data


def _capture(service: WorkspaceService, sid: str, owner: str, rid: str) -> dict[str, Any]:
    from forge_mind.workspace.delivery_guidance import guidance_context

    session = service.store.get(sid, owner)
    revision = service._load_revision(sid, rid)
    if not revision.get("document") or not revision["measurement"]["geometry_valid"]:
        raise WorkspaceError("Project export requires a valid structured document")
    guidance = guidance_context(service, sid, owner, rid)
    context = handoff_context(session, revision, engine_fingerprint())
    context["guidance_basis"] = guidance["basis"]
    context["guidance_basis_hash"] = guidance["basis_hash"]
    context["guidance_context_hash"] = digest(guidance)
    if "print_setup_record_hash" in guidance["basis"]:
        context["print_setup_record_hash"] = guidance["basis"]["print_setup_record_hash"]
    return {
        "session": session,
        "revision": revision,
        "context": context,
        "guidance": guidance,
        "review": product_review(service, sid, owner, rid),
        "inspection": inspection_handoff(
            service, sid, owner, revision, max_artifact_bytes=MAX_JOB_BYTES
        ),
    }


def _snapshot(captured: dict[str, Any]) -> dict[str, Any]:
    inspection = captured["inspection"]
    return {
        "context": captured["context"],
        "engineering_inputs": captured["session"]["engineering"],
        "review": captured["review"],
        "guidance": captured["guidance"],
        "inspection_plan": inspection["plan"],
        "inspection_status": inspection["status"],
        "inspection_artifact_hashes": {
            name: _sha(data) for name, data in inspection["artifacts"].items()
        },
    }


def build_manufacturing_handoff(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    target: Path,
    *,
    include_stl: bool = False,
    include_assembly_step: bool = False,
    include_import_assets: bool = False,
) -> None:
    """Share owned snapshot orchestration with the historical HTTP route."""
    _id(sid)
    _id(rid)
    if target.exists() or target.is_symlink():
        raise WorkspaceError("Project export must not overwrite an existing file")
    captured = _capture(service, sid, owner, rid)
    revision = captured["revision"]
    try:
        manufacture_bundle(
            DesignDocument.model_validate(revision["document"]),
            service.store.directory(sid, rid),
            target,
            revision["measurement"]["artifact_hashes"],
            revision["measurement"].get("facts", {}).get("assembly"),
            captured["review"]["sources"],
            captured["review"],
            captured["context"],
            captured["session"]["engineering"],
            captured["inspection"],
            guidance_context=captured["guidance"],
            include_stl=include_stl,
            include_assembly_step=include_assembly_step,
            include_import_assets=include_import_assets,
        )
        if _snapshot(captured) != _snapshot(_capture(service, sid, owner, rid)):
            raise WorkspaceError(
                "Project inputs or review evidence changed during handoff; export again"
            )
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _package_summary(path: Path) -> dict[str, Any]:
    # Bound decompression before the existing verifier reads any member.
    if not regular_file(path, MAX_PACKAGE_BYTES):
        raise WorkspaceError("Project package is missing, linked or oversized")
    with ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_JOB_BYTES:
            raise WorkspaceError("Project package exceeds its uncompressed budget")
    verified = verify_manufacturing_handoff(path)
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("include_assembly_step") is not True:
            raise WorkspaceError("Project package is missing its overall STEP")
        context = verified["context"]
        if context.get("status") != "captured":
            raise WorkspaceError("Project package needs captured project context")
        document = json.loads(archive.read("document.json"))
        production = json.loads(archive.read("production.json"))
        imported = referenced_import_asset_ids(document)
        include_stl = manifest.get("include_stl", False)
        include_import_assets = manifest.get("include_import_assets", False)
        made = []
        for row in production["make"]:
            part_id = row["part_id"]
            stem = f"definition_{part_id}" if document.get("assembly") else "part"
            made.append(
                {
                    "part_id": part_id,
                    "quantity": row["quantity"],
                    "files": [f"make/{stem}.step"] + ([f"make/{stem}.stl"] if include_stl else []),
                    "frame": "definition_local" if document.get("assembly") else "document_local",
                }
            )
        source_scope = {
            "file": "document.json",
            "sha256": manifest["files"]["document.json"]["sha256"],
            "representation": "saved_document_bytes"
            if include_stl or include_import_assets
            else "validated_schema_defaults_expanded",
            "omitted_import_asset_ids": imported,
            "independent_regeneration": "not_verified",
            "scope": "Editable declarations only; imported geometry inputs are not included. Requires a compatible CAD runtime. Not a full project backup.",
        }
        if include_import_assets:
            source_scope.update(
                omitted_import_asset_ids=[],
                provided_import_asset_ids=imported,
                input_map={
                    "file": "import-assets.json",
                    "sha256": manifest["files"]["import-assets.json"]["sha256"],
                },
                scope="Exact editable declarations and their referenced STEP source inputs are provided. Requires a compatible CAD runtime. Independent regeneration is not verified. Not a full project backup.",
            )
        snapshot = {
            "context": context,
            "files": manifest["files"],
            "include_stl": include_stl,
            "include_assembly_step": True,
            **({"include_import_assets": True} if include_import_assets else {}),
        }
        return {
            "project_id": context["session_id"],
            "revision_id": context["revision_id"],
            "include_stl": include_stl,
            **({"include_import_assets": True} if include_import_assets else {}),
            "snapshot_sha256": _sha(json_bytes(snapshot)),
            "context": context,
            "source_scope": source_scope,
            "made_parts": made,
            "assembly": manifest["assembly"],
            "status": "draft_candidate",
            "manufacturing_ready": False,
            "physical_readiness": "unverified",
            **(
                {"guidance_status": verified["guidance_status"]}
                if "guidance_status" in verified
                else {}
            ),
            "retention": RETENTION,
        }


def validate_project_export(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    """Validate delivered ZIP bytes and captured metadata, without extracting or rebuilding."""
    if not isinstance(receipt, dict):
        raise WorkspaceError("Malformed project package receipt")
    for key in ("project_id", "revision_id", "package_id"):
        _id(receipt.get(key))
    if (
        type(receipt.get("include_stl")) is not bool
        or type(receipt.get("include_import_assets", False)) is not bool
        or type(receipt.get("bytes")) is not int
        or receipt.get("manufacturing_ready") is not False
        or receipt.get("physical_readiness") != "unverified"
        or receipt.get("status") != "draft_candidate"
    ):
        raise WorkspaceError("Malformed project package receipt")
    data = _read(path, MAX_PACKAGE_BYTES)
    if len(data) != receipt["bytes"] or _sha(data) != receipt.get("sha256"):
        raise WorkspaceError("Project package bytes do not match its receipt")
    summary = _package_summary(path)
    if receipt.get("include_import_assets", False) != summary.get("include_import_assets", False):
        raise WorkspaceError("Project package input option does not match its receipt")
    if any(receipt.get(name) != value for name, value in summary.items()):
        raise WorkspaceError("Project package contents do not match its receipt")
    return dict(receipt)


def _location(service: WorkspaceService, sid: str, package_id: str) -> Path:
    _id(sid)
    _id(package_id)
    directory = service.store.root / "project-exports" / sid / package_id
    _safe_directory(directory)
    return directory


def create_project_export(
    service: WorkspaceService,
    project_id: str,
    principal: str,
    revision_id: str,
    *,
    include_stl: bool = False,
    include_import_assets: bool = False,
) -> dict[str, Any]:
    """Publish a new immutable captured review package without changing the revision."""
    _id(project_id)
    _id(revision_id)
    if type(include_stl) is not bool:
        raise WorkspaceError("include_stl must be a boolean")
    if type(include_import_assets) is not bool:
        raise WorkspaceError("include_import_assets must be a boolean")
    service.store.session_record(project_id, principal)
    package_id = uuid.uuid4().hex
    directory = _location(service, project_id, package_id)
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "project.zip"
    descriptor = directory / "receipt.json"
    try:
        build_manufacturing_handoff(
            service,
            project_id,
            principal,
            revision_id,
            path,
            include_stl=include_stl,
            include_assembly_step=True,
            include_import_assets=include_import_assets,
        )
        content = _read(path, MAX_PACKAGE_BYTES)
        receipt = {
            **_package_summary(path),
            "package_id": package_id,
            "sha256": _sha(content),
            "bytes": len(content),
        }
        if receipt["project_id"] != project_id or receipt["revision_id"] != revision_id:
            raise WorkspaceError("Project package context does not match its request")
        data = json_bytes(receipt)
        if len(data) > MAX_RECEIPT_BYTES:
            raise WorkspaceError("Project package receipt exceeds its metadata budget")
        with descriptor.open("xb") as output:
            output.write(data)
        return receipt
    except BaseException:
        descriptor.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
        directory.rmdir()
        raise


def read_project_export(
    service: WorkspaceService,
    project_id: str,
    principal: str,
    package_id: str,
    sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Read the exact captured package, even after later edits; recheck owner every time."""
    service.store.session_record(project_id, principal)
    directory = _location(service, project_id, package_id)
    if not isinstance(sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", sha256):
        raise WorkspaceError("Invalid package checksum")
    receipt = json.loads(_read(directory / "receipt.json", MAX_RECEIPT_BYTES))
    if not isinstance(receipt, dict) or (
        receipt.get("package_id") != package_id
        or receipt.get("project_id") != project_id
        or receipt.get("sha256") != sha256
    ):
        raise WorkspaceError("Project package identity does not match its resource")
    path = directory / "project.zip"
    validate_project_export(path, receipt)
    data = _read(path, MAX_PACKAGE_BYTES)
    if len(data) != receipt["bytes"] or _sha(data) != sha256:
        raise WorkspaceError("Project package changed during readback")
    return data, receipt


def discard_project_export(
    service: WorkspaceService,
    project_id: str,
    principal: str,
    package_id: str,
) -> None:
    """Discard only an unpublished completed export after its producer has stopped."""
    service.store.session_record(project_id, principal)
    directory = _location(service, project_id, package_id)
    for name in ("receipt.json", "project.zip"):
        path = directory / name
        if path.exists() and not regular_file(path):
            raise WorkspaceError("Refusing cleanup of a linked or nonregular export")
    for name in ("receipt.json", "project.zip"):
        (directory / name).unlink(missing_ok=True)
    directory.rmdir()
