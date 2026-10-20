"""Owned ephemeral world-plane mesh sections of immutable saved revisions."""

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
from forge_mind.workspace.section_render import (
    FRAME,
    MAX_IMAGE_BYTES,
    MAX_INPUT_BYTES,
    MAX_RECEIPT_BYTES,
    PNG_SIGNATURE,
    SCOPE,
    finite_number,
    read_bytes,
    read_object,
    validated_request,
)

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _artifact(directory: Path, name: str, expected: Any) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Section source has no saved checksum")
    content = read_bytes(directory / name, MAX_INPUT_BYTES)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Section source changed; revision evidence is stale")
    return content


def inspect_section(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    revision_id: str,
    axis: str,
    offset_mm: float,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    """Render saved STL only; no CAD rebuild, acceptance change or persistent output."""
    for value in (project_id, revision_id):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("Invalid section project or revision identifier")
    if axis not in ("x", "y", "z"):
        raise ValueError("Section axis must be x, y or z")
    offset = finite_number(offset_mm)
    deadline = finite_number(timeout)
    if not 0 < deadline <= 60:
        raise ValueError("Section requires a finite deadline of at most 60 seconds")
    started = time.monotonic()

    def remaining() -> float:
        result = deadline - (time.monotonic() - started)
        if result <= 0 or service.stop.is_set():
            raise ValueError("Section inspection cancelled or deadline exhausted")
        return result

    service.store.session_record(project_id, owner)
    remaining()
    record = service.store.revision(project_id, revision_id)
    measurement = deepcopy(record["measurement"])
    if record.get("id") != revision_id or measurement.get("revision_id") != revision_id:
        raise ValueError("Section measurement belongs to another revision")
    executor = service.executor
    if executor.mode == "docker" and not executor.typed_container_verified:
        raise ValueError("Section images require qualified isolated worker execution")
    if executor.mode not in ("docker", "trusted_local"):
        raise ValueError("Unsupported section execution mode")
    hashes = measurement.get("artifact_hashes", {})
    directory = service.store.directory(project_id, revision_id)
    source: dict[str, bytes] = {}
    for name in ("part.stl", "document.json"):
        remaining()
        source[name] = _artifact(directory, name, hashes.get(name))
    request = validated_request(
        {
            "project_id": project_id,
            "revision_id": revision_id,
            "axis": axis,
            "offset_mm": offset,
            "frame": FRAME,
            "mesh_sha256": hashes["part.stl"],
            "document_sha256": hashes["document.json"],
        }
    )
    with tempfile.TemporaryDirectory(prefix="forge-section-") as temporary:
        scratch = Path(temporary)
        for name, content in source.items():
            (scratch / name).write_bytes(content)
        request_path = scratch / "request.json"
        request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
        module = "forge_mind.workspace.section_render"
        if executor.mode == "docker":
            docker = executor._docker_path(shutil.which)
            if not docker:
                raise ValueError("Docker is required for isolated section images")
            executor._run_container(
                docker,
                scratch,
                module,
                "request.json",
                remaining(),
                service.stop.is_set,
                image=executor.typed_container_image,
            )
        else:
            bounded_process(
                [sys.executable, "-B", "-m", module, str(request_path)],
                scratch,
                remaining(),
                service.stop.is_set,
                cpu_limit=2,
            )
        remaining()
        receipt = read_object(read_bytes(scratch / "receipt.json", MAX_RECEIPT_BYTES))
        count = receipt.get("polyline_count")
        receipt_offset = finite_number(receipt.get("offset_mm"))
        if (
            set(receipt) != set(request) | {"section_status", "polyline_count", "image", "scope"}
            or any(receipt.get(key) != value for key, value in request.items())
            or receipt_offset != request["offset_mm"]
            or receipt.get("scope") != SCOPE
            or type(count) is not int
            or count < 0
        ):
            raise ValueError("Section output does not match its source receipt")
        response: dict[str, Any] = {
            "receipt": receipt,
            "image_base64": None,
            "mime_type": None,
        }
        image_path = scratch / "section.png"
        if count == 0:
            if (
                receipt.get("section_status") != "empty"
                or receipt.get("image") is not None
                or image_path.exists()
            ):
                raise ValueError("Empty section must have zero polylines and no image")
        else:
            image = read_bytes(image_path, MAX_IMAGE_BYTES)
            if (
                receipt.get("section_status") != "intersected"
                or not image.startswith(PNG_SIGNATURE)
                or receipt.get("image")
                != {"file": "section.png", "sha256": hashlib.sha256(image).hexdigest()}
            ):
                raise ValueError("Section image does not match its source receipt")
            response.update(
                image_base64=base64.b64encode(image).decode("ascii"),
                mime_type="image/png",
            )
    service.store.session_record(project_id, owner)
    if service.store.revision(project_id, revision_id)["measurement"] != measurement:
        raise ValueError("Section revision changed during observation")
    for name in source:
        remaining()
        _artifact(directory, name, hashes.get(name))
    remaining()
    if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Section exceeds its response byte budget")
    return response
