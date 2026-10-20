"""Owned ephemeral product photos from complete immutable saved STL geometry."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import shutil
import sys
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.processing import bounded_process
from forge_mind.workspace.product_photo_contract import (
    ASSEMBLY_REQUEST_KEYS,
    FRAME,
    MAX_IMAGE_BYTES,
    MAX_RECEIPT_BYTES,
    MAX_RESPONSE_BYTES,
    PRESETS,
    VERSION,
    PhotoUnavailableError,
    legacy_request,
    validate_receipt,
    validated_request,
)
from forge_mind.workspace.section_render import (
    MAX_INPUT_BYTES,
    PNG_SIGNATURE,
    finite_number,
    read_bytes,
    read_object,
)

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _read(path: Path, maximum: int, purpose: str) -> bytes:
    try:
        return read_bytes(path, maximum)
    except (ValueError, OSError) as exc:
        raise ValueError(
            f"Product photo {purpose} is missing, linked, oversized or unreadable"
        ) from exc


def _artifact(directory: Path, expected: Any) -> bytes:
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("Product photo source has no saved mesh checksum")
    content = _read(directory / "part.stl", MAX_INPUT_BYTES, "source")
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Product photo source changed; revision evidence is stale")
    return content


def render_mesh(
    service: WorkspaceService, source: bytes, request: dict[str, Any], *, timeout: float
) -> dict[str, Any]:
    """Render exact already-bound bytes; callers retain ownership and source checks."""
    request = validated_request(request)
    assembly = set(legacy_request(request)) == ASSEMBLY_REQUEST_KEYS
    source_name = "topology.json" if assembly else "part.stl"
    source_hash = request["source"]["topology_sha256"] if assembly else request["mesh_sha256"]
    if hashlib.sha256(source).hexdigest() != source_hash:
        raise ValueError("Product photo source bytes do not match their request")
    deadline = finite_number(timeout)
    if not 0 < deadline <= 60:
        raise ValueError("Product photo deadline must be at most 60 seconds")
    started = time.monotonic()
    photo_deadline_reached = False

    def cancelled() -> bool:
        nonlocal photo_deadline_reached
        if service.stop.is_set():
            return True
        if time.monotonic() - started >= deadline:
            photo_deadline_reached = True
            return True
        return False

    def verify_staged(scratch: Path, request_path: Path) -> None:
        if (
            hashlib.sha256(
                _read(scratch / source_name, MAX_INPUT_BYTES, "staged source")
            ).hexdigest()
            != source_hash
            or validated_request(
                read_object(_read(request_path, MAX_RECEIPT_BYTES, "staged request"))
            )
            != request
        ):
            raise ValueError("Product photo staged inputs changed during rendering")

    def remaining() -> float:
        if service.stop.is_set():
            raise ValueError("Product photo cancelled")
        value = deadline - (time.monotonic() - started)
        if value <= 0:
            raise PhotoUnavailableError("render_failed", "Product photo render deadline exhausted")
        return value

    executor = service.executor
    remaining()
    if executor.mode == "docker" and not executor.typed_container_verified:
        raise PhotoUnavailableError(
            "execution_unqualified", "Product photos require qualified isolated execution"
        )
    if executor.mode not in ("docker", "trusted_local"):
        raise PhotoUnavailableError(
            "execution_unqualified", "Product photo execution mode is unavailable"
        )
    with tempfile.TemporaryDirectory(prefix="forge-product-photo-") as temporary:
        scratch = Path(temporary)
        (scratch / source_name).write_bytes(source)
        request_path = scratch / "request.json"
        request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        module = "forge_mind.workspace.product_photo_render"
        try:
            if executor.mode == "docker":
                docker = executor._docker_path(shutil.which)
                if not docker:
                    raise PhotoUnavailableError(
                        "renderer_unavailable", "Docker is unavailable for isolated product photos"
                    )
                executor._run_container(
                    docker,
                    scratch,
                    module,
                    "request.json",
                    remaining(),
                    cancelled,
                    image=executor.typed_container_image,
                )
            else:
                bounded_process(
                    [sys.executable, "-B", "-m", module, str(request_path)],
                    scratch,
                    remaining(),
                    cancelled,
                    cpu_limit=2,
                )
        except ValueError as exc:
            # The executor chains cleanup failures to the active watchdog error.
            # Only an unwrapped deadline refusal can become optional absence.
            if exc.__cause__ is not None or exc.__context__ is not None:
                raise
            if service.stop.is_set():
                raise ValueError("Product photo cancelled") from exc
            if not photo_deadline_reached:
                raise
            verify_staged(scratch, request_path)
            raise PhotoUnavailableError(
                "render_failed", "Product photo render deadline exhausted"
            ) from exc
        verify_staged(scratch, request_path)
        remaining()
        unavailable_path = scratch / "unavailable.json"
        if unavailable_path.exists() or unavailable_path.is_symlink():
            if any(
                (scratch / name).exists() or (scratch / name).is_symlink()
                for name in ("photo.png", "receipt.json")
            ):
                raise ValueError("Product photo worker returned contradictory outputs")
            error = read_object(_read(unavailable_path, MAX_RECEIPT_BYTES, "optional failure"))
            if (
                set(error) != {"request", "code", "reason"}
                or validated_request(error["request"]) != request
                or not isinstance(error["reason"], str)
                or not 0 < len(error["reason"]) <= 500
            ):
                raise ValueError("Product photo optional failure is malformed or unbound")
            if error["code"] in ("source_unavailable", "render_limit") and not assembly:
                raise ValueError("Assembly photo absence cannot replace a captured STL render")
            raise PhotoUnavailableError(error["code"], error["reason"])
        receipt = validate_receipt(
            read_object(_read(scratch / "receipt.json", MAX_RECEIPT_BYTES, "output receipt")),
            request,
        )
        image = _read(scratch / "photo.png", MAX_IMAGE_BYTES, "output PNG")
        validate_image(image, receipt)
        if (
            hashlib.sha256(
                _read(scratch / source_name, MAX_INPUT_BYTES, "staged source")
            ).hexdigest()
            != source_hash
            or read_object(_read(request_path, MAX_RECEIPT_BYTES, "staged request")) != request
        ):
            raise ValueError("Product photo staged inputs changed during rendering")
        result = {
            "receipt": receipt,
            "image_base64": base64.b64encode(image).decode("ascii"),
            "mime_type": "image/png",
        }
    if len(json.dumps(result, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Product photo exceeds its response byte budget")
    return result


def validate_image(image: bytes, receipt: dict[str, Any]) -> None:
    """Bind bounded PNG bytes to the validated receipt."""
    if (
        not isinstance(image, bytes)
        or not 33 <= len(image) <= MAX_IMAGE_BYTES
        or not image.startswith(PNG_SIGNATURE)
        or image[8:12] != b"\x00\x00\x00\x0d"
        or image[12:16] != b"IHDR"
        or int.from_bytes(image[16:20], "big") != receipt["image"]["width"]
        or int.from_bytes(image[20:24], "big") != receipt["image"]["height"]
        or hashlib.sha256(image).hexdigest() != receipt["image"]["sha256"]
    ):
        raise ValueError("Product photo PNG does not match its source receipt")


def get_product_photo(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    revision_id: str,
    preset: str,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    """Observe saved mesh appearance without document reads or persistent mutation."""
    for value in (project_id, revision_id):
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{32}", value) is None:
            raise ValueError("Invalid product photo project or revision identifier")
    if preset not in PRESETS:
        raise ValueError("Unsupported product photo preset")
    try:
        deadline = finite_number(timeout)
    except ValueError as exc:
        raise ValueError("Product photo requires a finite numeric deadline") from exc
    if not 0 < deadline <= 60:
        raise ValueError("Product photo requires a finite deadline of at most 60 seconds")
    started = time.monotonic()

    def remaining() -> float:
        result = deadline - (time.monotonic() - started)
        if result <= 0 or service.stop.is_set():
            raise ValueError("Product photo cancelled or deadline exhausted")
        return result

    service.store.session_record(project_id, owner)
    remaining()
    record = service.store.revision(project_id, revision_id)
    measurement = deepcopy(record["measurement"])
    if record.get("id") != revision_id or measurement.get("revision_id") != revision_id:
        raise ValueError("Product photo measurement belongs to another revision")
    executor = service.executor
    if executor.mode == "docker" and not executor.typed_container_verified:
        raise ValueError("Product photos require qualified isolated worker execution")
    if executor.mode not in ("docker", "trusted_local"):
        raise ValueError("Unsupported product photo execution mode")
    hashes = measurement.get("artifact_hashes", {})
    directory = service.store.directory(project_id, revision_id)
    source = _artifact(directory, hashes.get("part.stl"))
    request = validated_request(
        {
            "project_id": project_id,
            "revision_id": revision_id,
            "mesh_sha256": hashes["part.stl"],
            "preset": preset,
            "frame": FRAME,
            "version": VERSION,
        }
    )
    remaining()
    try:
        response = render_mesh(service, source, request, timeout=remaining())
    except PhotoUnavailableError as exc:
        remaining()  # Preserve the standalone operation's existing deadline refusal.
        raise ValueError(str(exc)) from exc
    # No concurrent deletion, owner/source mutation or cancellation may publish a photo.
    service.store.session_record(project_id, owner)
    if service.store.revision(project_id, revision_id)["measurement"] != measurement:
        raise ValueError("Product photo revision changed during observation")
    remaining()
    _artifact(directory, hashes.get("part.stl"))
    remaining()
    if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Product photo exceeds its response byte budget")
    return response
