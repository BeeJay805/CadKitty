"""Owned durable photographs of exact made STL members in captured exports."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.document import digest
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.product_photo import _read as read_photo_bytes
from forge_mind.workspace.product_photo import render_mesh, validate_image
from forge_mind.workspace.product_photo_contract import (
    ASSEMBLY_REQUEST_KEYS,
    CAPTURED_REQUEST_KEYS,
    MAX_IMAGE_BYTES,
    MAX_RECEIPT_BYTES,
    MAX_RESPONSE_BYTES,
    OPPOSED_VIEW,
    PRESETS,
    VERSION,
    PhotoUnavailableError,
    legacy_request,
    opposed_request,
    validate_receipt,
    validated_request,
)
from forge_mind.workspace.project_delivery import _safe_directory, read_project_export
from forge_mind.workspace.project_members import (
    captured_cad_files,
    captured_photo_context,
    read_captured_cad_file,
)
from forge_mind.workspace.project_photo_transport import (
    alternate_unavailable,
    validate_project_photo_pair,
)
from forge_mind.workspace.section_render import finite_number, read_bytes, read_object

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _id(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid project photo identifier")


def _hash(value: object) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Invalid project photo checksum")


def _directory(service: WorkspaceService, project_id: str, package_id: str, photo_id: str) -> Path:
    for value in (project_id, package_id, photo_id):
        _id(value)
    path = service.store.root / "sessions" / project_id / "project-photos" / package_id / photo_id
    _safe_directory(path)
    return path


def _unavailable(code: str, reason: str, candidates: list[str]) -> dict[str, Any]:
    return {"status": "unavailable", "code": code, "reason": reason[:500], "candidates": candidates}


def _request(package: dict[str, Any], row: dict[str, Any], preset: str) -> dict[str, Any]:
    return validated_request(
        {
            "project_id": package["project_id"],
            "revision_id": package["revision_id"],
            "package_id": package["package_id"],
            "package_sha256": package["sha256"],
            "part_id": row["part_id"],
            "file_id": row["file_id"],
            "quantity": row["quantity"],
            "mesh_sha256": row["sha256"],
            "frame": row["frame"],
            "preset": preset,
            "version": VERSION,
        }
    )


def _target(
    rows: list[dict[str, Any]], part_id: str | None
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    candidates = sorted({row["part_id"] for row in rows if row["role"] == "make"})
    if part_id is not None and part_id not in candidates:
        raise ValueError(
            "Project photo target is not a used made definition in the captured package"
        )
    if not candidates:
        return None, _unavailable(
            "no_made_parts", "Captured package has no used made definitions", candidates
        )
    if part_id is None and len(candidates) != 1:
        return None, _unavailable(
            "ambiguous_parts", "Select a made definition for this multipart package", candidates
        )
    chosen = part_id if part_id is not None else candidates[0]
    selected = [
        row
        for row in rows
        if row["part_id"] == chosen and row["format"] == "stl" and row["role"] == "make"
    ]
    if not selected:
        return None, _unavailable(
            "missing_stl",
            "The selected made definition has no captured STL; no mesh was rebuilt",
            candidates,
        )
    if len(selected) != 1:
        raise ValueError("Project photo target has duplicate captured STL members")
    return selected[0], None


def _assembly_source(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    package: dict[str, Any],
    context: dict[str, Any],
    preset: str,
) -> tuple[dict[str, Any], bytes, Any]:
    """Bind saved native correspondence; never import STEP or expand placements."""
    service.store.session_record(project_id, owner)
    rid = package["revision_id"]
    record = deepcopy(service.store.revision(project_id, rid))
    measurement = record["measurement"]
    if (
        record.get("id") != rid
        or measurement.get("revision_id") != rid
        or measurement.get("geometry_valid") is not True
    ):
        raise ValueError("Project photo requires matching valid saved revision evidence")
    if digest(record.get("document")) != package.get("context", {}).get("stored_document_hash"):
        raise ValueError("Project photo saved document differs from captured source context")
    hashes = measurement.get("artifact_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("Project photo has malformed saved artifact evidence")
    assembly = measurement.get("facts", {}).get("assembly")
    names = ("document.json", "assembly.json", "topology.json", "part.step")
    if assembly is not None and (
        not isinstance(assembly, dict) or not isinstance(assembly.get("occurrences"), list)
    ):
        raise ValueError("Project photo has malformed saved assembly correspondence")
    directory = service.store.directory(project_id, rid)

    def sources() -> dict[str, bytes]:
        _safe_directory(directory)
        result: dict[str, bytes] = {}
        for name in names:
            if name not in hashes:
                continue
            _hash(hashes[name])
            # Bound large source verification separately from the worker display budget.
            data = read_photo_bytes(directory / name, 256 * 1024 * 1024, "saved assembly source")
            if hashlib.sha256(data).hexdigest() != hashes[name]:
                raise ValueError("Project photo saved source changed: " + name)
            result[name] = data
        _safe_directory(directory)
        return result

    bound = sources()
    # Validate every recorded file even when other legacy records are absent.
    if assembly is None or any(name not in hashes for name in names):
        raise PhotoUnavailableError(
            "source_unavailable", "Saved revision has no complete recorded assembly display source"
        )
    if not isinstance(assembly, dict) or not isinstance(assembly.get("occurrences"), list):
        raise ValueError("Project photo has malformed saved assembly correspondence")
    if read_object(bound["assembly.json"]) != assembly:
        raise ValueError("Project photo saved assembly facts differ from measurement")
    if hashes["part.step"] != context["assembly_step_sha256"]:
        raise ValueError("Project photo saved assembly STEP differs from captured STEP")
    for key, wanted in (
        ("revision_id", rid),
        ("step_sha256", hashes["part.step"]),
        ("engine_fingerprint", measurement.get("engine_fingerprint")),
    ):
        if key in assembly and assembly[key] != wanted:
            raise ValueError("Project photo saved assembly provenance is stale")
    occurrence_rows = assembly["occurrences"]
    if not occurrence_rows:
        raise ValueError("Project photo has empty saved assembly correspondence")
    by_path: dict[str, dict[str, Any]] = {}
    indices: set[int] = set()
    for row in occurrence_rows:
        if not isinstance(row, dict):
            raise ValueError("Project photo saved occurrence row is malformed")
        path, index = row.get("path"), row.get("body_index")
        if (
            not isinstance(path, str)
            or path in by_path
            or type(index) is not int
            or not 0 <= index < len(occurrence_rows)
            or index in indices
        ):
            raise ValueError("Project photo saved body correspondence is ambiguous")
        by_path[path] = row
        indices.add(index)
    expected = context["selected"] + context["omitted"]
    if set(by_path) != {row["occurrence_path"] for row in expected}:
        raise ValueError("Project photo captured occurrences differ from saved correspondence")
    selection: dict[str, Any] = {"total_bodies": len(occurrence_rows)}
    for group in ("selected", "omitted"):
        joined = []
        for row in context[group]:
            saved = by_path[row["occurrence_path"]]
            if saved.get("definition") != row["part_id"]:
                raise ValueError(
                    "Project photo saved occurrence definition contradicts captured source"
                )
            joined.append({**row, "body_index": saved["body_index"]})
        selection[group] = joined

    def verify() -> None:
        service.store.session_record(project_id, owner)
        current = service.store.revision(project_id, rid)
        if current["measurement"] != measurement or current.get("document") != record.get(
            "document"
        ):
            raise ValueError("Project photo saved revision changed during observation")
        if sources() != bound:
            raise ValueError("Project photo saved sources changed during observation")
        if service.stop.is_set():
            raise ValueError("Project photo cancelled")

    verify()
    if len(bound["topology.json"]) > 64 * 1024 * 1024 or len(occurrence_rows) > 4096:
        raise PhotoUnavailableError(
            "render_limit", "Saved assembly display source exceeds product-photo limits"
        )
    request = validated_request(
        {
            "version": VERSION,
            "project_id": project_id,
            "revision_id": rid,
            "package_id": package["package_id"],
            "package_sha256": package["sha256"],
            "preset": preset,
            "frame": "cad_z_up_mm_world",
            "source_kind": "saved_topology_made_assembly",
            "source": {
                "saved_document_sha256": hashes["document.json"],
                "captured_document_sha256": context["captured_document_sha256"],
                "assembly_step_sha256": hashes["part.step"],
                "assembly_facts_sha256": hashes["assembly.json"],
                "topology_sha256": hashes["topology.json"],
            },
            "selection": selection,
        }
    )
    if len(json.dumps(request, allow_nan=False).encode("utf-8")) > MAX_RECEIPT_BYTES:
        raise PhotoUnavailableError(
            "render_limit", "Saved assembly occurrence request exceeds product-photo limits"
        )
    return request, bound["topology.json"], verify


def _remove_created(directory: Path) -> None:
    """Remove only this call's exclusive outputs, never a package or revision."""
    _safe_directory(directory)
    for name in ("photo.png", "receipt.json"):
        path = directory / name
        if (path.exists() or path.is_symlink()) and not regular_file(path):
            raise ValueError("Refusing cleanup of a linked or nonregular project photo")
    for name in ("photo.png", "receipt.json"):
        _safe_directory(directory)
        (directory / name).unlink(missing_ok=True)
    directory.rmdir()


def create_project_photo(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    package_id: str,
    package_sha256: str,
    *,
    preset: str = "plastic",
    part_id: str | None = None,
    timeout: float = 15,
    view: str | None = None,
) -> dict[str, Any]:
    """Photograph all saved made occurrences, or one explicit captured make STL."""
    _id(project_id)
    _id(package_id)
    _hash(package_sha256)
    if view not in (None, OPPOSED_VIEW):
        raise ValueError("Unsupported project photo view")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("Unsupported project photo appearance preset")
    if part_id is not None and (
        not isinstance(part_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", part_id)
    ):
        raise ValueError("Invalid project photo made definition identifier")
    deadline = finite_number(timeout)
    if not 0 < deadline <= 60:
        raise ValueError("Invalid project photo deadline")
    started = time.monotonic()

    def check() -> float:
        if service.stop.is_set():
            raise ValueError("Project photo cancelled")
        remaining = deadline - (time.monotonic() - started)
        if remaining <= 0:
            raise PhotoUnavailableError("render_failed", "Project photo deadline exhausted")
        return remaining

    if service.stop.is_set():
        raise ValueError("Project photo cancelled")
    content, package = read_project_export(service, project_id, owner, package_id, package_sha256)
    rows = captured_cad_files(content, package)
    context = captured_photo_context(content, package)
    candidates = sorted({item["part_id"] for item in rows if item["role"] == "make"})
    assembly_default = part_id is None and context["structured_assembly"] and bool(candidates)
    row, unavailable = (None, None) if assembly_default else _target(rows, part_id)
    initial_revision = (
        deepcopy(service.store.revision(project_id, package["revision_id"]))
        if assembly_default
        else None
    )

    def verify_source() -> None:
        # Captured STL bytes need no additional saved-assembly lookup.
        pass

    def optional_failure(exc: PhotoUnavailableError) -> dict[str, Any]:
        current, current_package = read_project_export(
            service, project_id, owner, package_id, package_sha256
        )
        if current != content or current_package != package:
            raise ValueError("Project photo captured package changed")
        if assembly_default:
            if service.store.revision(project_id, package["revision_id"]) != initial_revision:
                raise ValueError("Project photo saved revision changed")
            try:
                _, _, current_verify = _assembly_source(
                    service, project_id, owner, package, context, preset
                )
                current_verify()
            except PhotoUnavailableError:
                pass  # Only typed legacy/resource absence; integrity/cancellation still refuse.
        if service.stop.is_set():
            raise ValueError("Project photo cancelled") from exc
        return _unavailable(exc.code, exc.reason, candidates)

    if unavailable is not None:
        service.store.session_record(project_id, owner)
        if service.stop.is_set():
            raise ValueError("Project photo cancelled")
        return unavailable
    try:
        if assembly_default:
            request, mesh, verify_source = _assembly_source(
                service, project_id, owner, package, context, preset
            )
        else:
            assert row is not None
            request = _request(package, row, preset)
            mesh = read_captured_cad_file(content, package, row["file_id"])
        if view is not None:
            request = opposed_request(request)
        result = render_mesh(service, mesh, request, timeout=check())
    except PhotoUnavailableError as exc:
        return optional_failure(exc)
    if (
        not isinstance(result, dict)
        or set(result) != {"receipt", "image_base64", "mime_type"}
        or result["mime_type"] != "image/png"
        or not isinstance(result["image_base64"], str)
        or len(result["image_base64"]) > 4 * ((MAX_IMAGE_BYTES + 2) // 3)
    ):
        raise ValueError("Malformed project photo render response")
    receipt = validate_receipt(result["receipt"], request)
    image = base64.b64decode(result["image_base64"], validate=True)
    validate_image(image, receipt)
    current, current_package = read_project_export(
        service, project_id, owner, package_id, package_sha256
    )
    if current != content or current_package != package:
        raise ValueError("Project photo captured package changed")
    verify_source()
    try:
        check()
    except PhotoUnavailableError as exc:
        return optional_failure(exc)
    photo_id = uuid.uuid4().hex
    directory = _directory(service, project_id, package_id, photo_id)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        descriptor = json.dumps({"photo_id": photo_id, "receipt": receipt}, allow_nan=False).encode(
            "utf-8"
        )
        if len(descriptor) > MAX_RECEIPT_BYTES + 256:
            raise ValueError("Project photo descriptor exceeds its byte budget")
        for name, data in (("photo.png", image), ("receipt.json", descriptor)):
            _safe_directory(directory)
            check()
            with (directory / name).open("xb") as output:
                output.write(data)
        service.store.session_record(project_id, owner)
        current, current_package = read_project_export(
            service, project_id, owner, package_id, package_sha256
        )
        if current != content or current_package != package:
            raise ValueError("Project photo captured package changed during publication")
        verify_source()
        _safe_directory(directory)
        if read_bytes(directory / "receipt.json", MAX_RECEIPT_BYTES + 256) != descriptor:
            raise ValueError("Project photo descriptor changed during publication")
        validate_image(read_bytes(directory / "photo.png", MAX_IMAGE_BYTES), receipt)
        check()
        response = {"status": "available", "photo_id": photo_id, **result}
        if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Project photo delivery exceeds its response byte budget")
        return response
    except PhotoUnavailableError as exc:
        _remove_created(directory)
        return optional_failure(exc)
    except BaseException:
        _remove_created(directory)
        raise


def read_project_photo(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    package_id: str,
    package_sha256: str,
    photo_id: str,
    image_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    """Read an owned immutable photo and bind it to its exact captured member."""
    _hash(image_sha256)
    content, package = read_project_export(service, project_id, owner, package_id, package_sha256)
    directory = _directory(service, project_id, package_id, photo_id)
    descriptor = read_object(read_bytes(directory / "receipt.json", MAX_RECEIPT_BYTES + 256))
    if (
        set(descriptor) != {"photo_id", "receipt"}
        or descriptor["photo_id"] != photo_id
        or not isinstance(descriptor["receipt"], dict)
    ):
        raise ValueError("Project photo descriptor does not match its resource")
    recorded = descriptor["receipt"]
    request_keys = ASSEMBLY_REQUEST_KEYS if "source_kind" in recorded else CAPTURED_REQUEST_KEYS
    if "view" in recorded:
        request_keys = request_keys | {"view"}
    recorded_request = {key: recorded[key] for key in request_keys if key in recorded}
    normalized = legacy_request(recorded_request)
    preset = recorded.get("preset")
    if not isinstance(preset, str):
        raise ValueError("Project photo appearance preset is missing or malformed")

    def verify_source() -> None:
        # Captured STL bytes need no additional saved-assembly lookup.
        pass

    if "source_kind" in normalized:
        context = captured_photo_context(content, package)
        if not context["structured_assembly"]:
            raise ValueError("Project photo assembly receipt is bound to a flat source")
        try:
            request, _, verify_source = _assembly_source(
                service, project_id, owner, package, context, preset
            )
        except PhotoUnavailableError as exc:
            raise ValueError("Project photo saved assembly source is no longer available") from exc
    else:
        row, unavailable = _target(captured_cad_files(content, package), recorded.get("part_id"))
        if unavailable is not None or row is None:
            raise ValueError("Project photo no longer has a bound captured STL")
        request = _request(package, row, preset)
    if "view" in recorded_request:
        request = opposed_request(request)
    receipt = validate_receipt(recorded, request)
    if receipt["image"]["sha256"] != image_sha256:
        raise ValueError("Project photo checksum does not match its resource")
    image = read_bytes(directory / "photo.png", MAX_IMAGE_BYTES)
    validate_image(image, receipt)
    _safe_directory(directory)
    current, current_package = read_project_export(
        service, project_id, owner, package_id, package_sha256
    )
    if current != content or current_package != package:
        raise ValueError("Project photo package changed during read")
    verify_source()
    if service.stop.is_set():
        raise ValueError("Project photo cancelled")
    return image, receipt


def discard_project_photo(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    package_id: str,
    photo_id: str,
) -> None:
    """Discard only the caller-owned completed photo, leaving CAD exports intact."""
    service.store.session_record(project_id, owner)
    directory = _directory(service, project_id, package_id, photo_id)
    descriptor = read_object(read_bytes(directory / "receipt.json", MAX_RECEIPT_BYTES + 256))
    receipt = descriptor.get("receipt")
    if (
        set(descriptor) != {"photo_id", "receipt"}
        or descriptor["photo_id"] != photo_id
        or not isinstance(receipt, dict)
        or set(receipt)
        not in (
            CAPTURED_REQUEST_KEYS | {"scope", "image", "rendering"},
            ASSEMBLY_REQUEST_KEYS | {"scope", "image", "rendering"},
            CAPTURED_REQUEST_KEYS | {"view", "scope", "image", "rendering"},
            ASSEMBLY_REQUEST_KEYS | {"view", "scope", "image", "rendering"},
        )
        or receipt.get("project_id") != project_id
        or receipt.get("package_id") != package_id
    ):
        raise ValueError("Project photo cleanup identity is malformed")
    keys = ASSEMBLY_REQUEST_KEYS if "source_kind" in receipt else CAPTURED_REQUEST_KEYS
    if "view" in receipt:
        keys = keys | {"view"}
    validate_receipt(receipt, {key: receipt[key] for key in keys})
    _remove_created(directory)


def create_project_photo_pair(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    package_id: str,
    package_sha256: str,
    *,
    preset: str = "plastic",
    part_id: str | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    """Create two bounded views of one capture using one total time allowance."""
    _id(project_id)
    _id(package_id)
    _hash(package_sha256)
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("Unsupported project photo appearance preset")
    if part_id is not None and (
        not isinstance(part_id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", part_id)
    ):
        raise ValueError("Invalid project photo made definition identifier")
    allowance = finite_number(timeout)
    if not 0 < allowance <= 15:
        raise ValueError("Invalid project photo pair deadline")
    started = time.monotonic()
    content, package = read_project_export(service, project_id, owner, package_id, package_sha256)
    members = captured_cad_files(content, package)
    context = captured_photo_context(content, package)
    candidates = sorted({row["part_id"] for row in members if row["role"] == "make"})
    assembly = part_id is None and context["structured_assembly"] and bool(candidates)
    initial_revision = (
        deepcopy(service.store.revision(project_id, package["revision_id"])) if assembly else None
    )
    created: list[tuple[str, dict[str, Any]]] = []

    def remaining() -> float:
        return allowance - (time.monotonic() - started)

    def verify() -> None:
        current, current_package = read_project_export(
            service, project_id, owner, package_id, package_sha256
        )
        if current != content or current_package != package:
            raise ValueError("Project photo pair captured package changed")
        if assembly:
            if service.store.revision(project_id, package["revision_id"]) != initial_revision:
                raise ValueError("Project photo pair saved revision changed")
            try:
                _, _, verify_source = _assembly_source(
                    service, project_id, owner, package, context, preset
                )
                verify_source()
            except PhotoUnavailableError:
                pass  # Optional source absence never masks owner, tamper or cancellation.
        if service.stop.is_set():
            raise ValueError("Project photo pair cancelled")

    def identity(photo_id: str) -> dict[str, Any]:
        directory = _directory(service, project_id, package_id, photo_id)
        result: dict[str, Any] = {}
        for name in (None, "receipt.json", "photo.png"):
            path = directory if name is None else directory / name
            _safe_directory(directory)
            if name is not None and not regular_file(path):
                raise ValueError("Project photo pair resource is not a regular file")
            stat = path.stat()
            result[str(name)] = (stat.st_dev, stat.st_ino)
            if name is not None:
                cap = MAX_IMAGE_BYTES if name == "photo.png" else MAX_RECEIPT_BYTES + 256
                result[name + "-sha256"] = hashlib.sha256(read_bytes(path, cap)).hexdigest()
        return result

    def remember(result: dict[str, Any]) -> None:
        if result.get("status") == "available":
            photo_id = result["photo_id"]
            created.append((photo_id, identity(photo_id)))

    def discard(photo_id: str) -> None:
        recorded = next(item for item in created if item[0] == photo_id)
        if identity(photo_id) != recorded[1]:
            raise ValueError("Refusing cleanup of a replaced project photo pair resource")
        discard_project_photo(service, project_id, owner, package_id, photo_id)
        created.remove(recorded)

    def absent(code: str, reason: str) -> dict[str, Any]:
        return alternate_unavailable(code, reason, candidates)

    def verify_photos(result: dict[str, Any]) -> None:
        for item in (result["primary"], result["alternate"]):
            if item["status"] != "available":
                continue
            photo_id = item["photo_id"]
            recorded = next(value for key, value in created if key == photo_id)
            if identity(photo_id) != recorded:
                raise ValueError("Project photo pair resource changed before delivery")
            image, receipt = read_project_photo(
                service,
                project_id,
                owner,
                package_id,
                package_sha256,
                photo_id,
                item["receipt"]["image"]["sha256"],
            )
            if (
                receipt != item["receipt"]
                or base64.b64encode(image).decode("ascii") != item["image_base64"]
                or identity(photo_id) != recorded
            ):
                raise ValueError("Project photo pair resource changed before delivery")

    try:
        verify()
        primary_remaining = remaining()
        if primary_remaining <= 0:
            _, structural_absence = (None, None) if assembly else _target(members, part_id)
            primary = structural_absence or _unavailable(
                "render_failed", "Project photo pair deadline exhausted", candidates
            )
        else:
            primary = create_project_photo(
                service,
                project_id,
                owner,
                package_id,
                package_sha256,
                preset=preset,
                part_id=part_id,
                timeout=primary_remaining,
            )
        remember(primary)
        alternate_remaining = remaining()
        if primary["status"] != "available":
            alternate = absent(
                "primary_unavailable", "Primary photo is unavailable; alternate was skipped"
            )
        elif alternate_remaining <= 0:
            alternate = absent("render_failed", "Project photo pair deadline exhausted")
        else:
            second = create_project_photo(
                service,
                project_id,
                owner,
                package_id,
                package_sha256,
                preset=preset,
                part_id=part_id,
                timeout=alternate_remaining,
                view=OPPOSED_VIEW,
            )
            remember(second)
            alternate = {"view": OPPOSED_VIEW, **second}
        result = {"primary": primary, "alternate": alternate}
        validate_project_photo_pair(
            result, package, members, preset, part_id, capture_context=context
        )
        verify()
        if alternate["status"] == "available" and remaining() <= 0:
            discard(alternate["photo_id"])
            result["alternate"] = absent("render_failed", "Project photo pair deadline exhausted")
        if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
            if result["alternate"]["status"] == "available":
                discard(result["alternate"]["photo_id"])
                result["alternate"] = absent(
                    "delivery_budget_exceeded",
                    "Alternate photo exceeds the shared delivery byte allowance",
                )
            if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise ValueError("Primary photo delivery exceeds its response byte budget")
        verify()
        verify_photos(result)
        if result["alternate"]["status"] == "available" and remaining() <= 0:
            discard(result["alternate"]["photo_id"])
            result["alternate"] = absent("render_failed", "Project photo pair deadline exhausted")
        if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
            raise ValueError("Primary photo delivery exceeds its response byte budget")
        return result
    except BaseException as exc:
        failures = []
        for photo_id, _ in list(reversed(created)):
            try:
                discard(photo_id)
            except Exception as cleanup_error:
                failures.append(str(cleanup_error))
        if failures:
            raise ValueError("Project photo pair cleanup failed: " + "; ".join(failures)) from exc
        raise
