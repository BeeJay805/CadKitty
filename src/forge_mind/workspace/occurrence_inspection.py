"""Owned, ephemeral observations of one immutable assembly occurrence."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.occurrence_facts import occurrence_facts
from forge_mind.workspace.occurrence_focus import (
    OccurrenceFocus,
    focus_view_metadata,
    resolve_focus,
    validate_focus_frame,
)
from forge_mind.workspace.processing import bounded_process, regular_file

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _artifact(directory: Path, name: str, expected: str | None) -> bytes:
    path = directory / name
    if not expected or not regular_file(path, MAX_INPUT_BYTES):
        raise ValueError("Inspection source is missing, linked, or oversized")
    data = path.read_bytes()
    if len(data) > MAX_INPUT_BYTES or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError("Inspection source changed; revision evidence is stale")
    return data


def inspect_occurrence(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    revision_id: str,
    occurrence_path: str,
    *,
    surface_offset: int = 0,
    surface_limit: int = 20,
    include_image: bool = True,
    focus: OccurrenceFocus | None = None,
    timeout: float = 30,
) -> dict[str, Any]:
    """Read saved native facts and optionally render a verified selection mesh.

    Hosted images use the qualified immutable container image. Private local
    images use a bounded child process. Neither path imports STEP in the host
    or changes the candidate, contract, measurement, or acceptance state.
    """
    if not all(re.fullmatch(r"[a-f0-9]{32}", value) for value in (project_id, revision_id)):
        raise ValueError("Invalid project or revision identifier")
    service.store.session_record(project_id, owner)
    if (
        type(timeout) not in (float, int)
        or not math.isfinite(timeout)
        or not 0 < timeout <= 60
        or type(include_image) is not bool
    ):
        raise ValueError("Inspection requires a finite deadline of at most 60 seconds")
    if focus is not None:
        focus = OccurrenceFocus.model_validate(focus)
        if not include_image:
            raise ValueError("Occurrence focus requires include_image=True")
    started = time.monotonic()
    record = service.store.revision(project_id, revision_id)
    measurement = record["measurement"]
    if measurement.get("revision_id") != revision_id:
        raise ValueError("Inspection measurement belongs to another revision")
    facts = occurrence_facts(
        measurement, occurrence_path, surface_offset=surface_offset, surface_limit=surface_limit
    )
    assembly = measurement.get("facts", {}).get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("Occurrence inspection requires a saved assembly body mapping")
    rows = assembly.get("occurrences", [])
    selected = [row for row in rows if row.get("path") == occurrence_path]
    if len(selected) != 1 or type(selected[0].get("body_index")) is not int:
        raise ValueError("Occurrence body identity is unavailable or ambiguous")
    body_index = selected[0]["body_index"]
    if body_index < 0 or sum(row.get("body_index") == body_index for row in rows) != 1:
        raise ValueError("Occurrence body identity is unavailable or ambiguous")
    directory = service.store.directory(project_id, revision_id)
    hashes = measurement.get("artifact_hashes", {})
    for key, expected in (
        ("revision_id", revision_id),
        ("step_sha256", hashes.get("part.step")),
        ("engine_fingerprint", measurement.get("engine_fingerprint")),
    ):
        if key in assembly and assembly[key] != expected:
            raise ValueError("Saved occurrence facts do not match their revision provenance")
    names = ["document.json", "part.step"] + (["topology.json"] if include_image else [])
    source: dict[str, bytes] = {}
    for name in names:
        source[name] = _artifact(directory, name, hashes.get(name))
    provenance = {
        "revision_id": revision_id,
        "occurrence_path": occurrence_path,
        "body_index": body_index,
        "step_sha256": hashes["part.step"],
        "document_sha256": hashes["document.json"],
        "engine_fingerprint": measurement.get("engine_fingerprint"),
        "frame": "cad_z_up_mm_world",
        "scope": "Saved native cylinder observations; not a new acceptance or functional check",
    }
    facts = {**facts, "provenance": provenance}
    response: dict[str, Any] = {
        "facts": facts,
        "receipt": None,
        "image_base64": None,
        "mime_type": None,
    }
    if include_image:
        executor = service.executor
        if executor.mode == "docker" and not executor.typed_container_verified:
            raise ValueError("Occurrence images require qualified isolated worker execution")
        request: dict[str, Any] = {
            "revision_id": revision_id,
            "occurrence_path": occurrence_path,
            "body_index": body_index,
            "topology_sha256": hashes["topology.json"],
            "step_sha256": hashes["part.step"],
            "document_sha256": hashes["document.json"],
        }
        resolved_focus = None
        if focus is not None:
            topology = json.loads(source["topology.json"])
            resolved_focus = resolve_focus(topology, body_index, focus)
            request["focus"] = resolved_focus
        with tempfile.TemporaryDirectory(prefix="forge-occurrence-") as temporary:
            scratch = Path(temporary)
            (scratch / "topology.json").write_bytes(source["topology.json"])
            request_path = scratch / "request.json"
            request_path.write_text(json.dumps(request, allow_nan=False), encoding="utf-8")
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0 or service.stop.is_set():
                raise ValueError("Inspection cancelled or deadline exhausted")
            module = "forge_mind.workspace.occurrence_render"
            if executor.mode == "docker":
                docker = executor._docker_path(shutil.which)
                if not docker:
                    raise ValueError("Docker is required for isolated occurrence images")
                executor._run_container(
                    docker,
                    scratch,
                    module,
                    "request.json",
                    remaining,
                    service.stop.is_set,
                    image=executor.typed_container_image,
                )
            elif executor.mode == "trusted_local":
                bounded_process(
                    [sys.executable, "-B", "-m", module, str(request_path)],
                    scratch,
                    remaining,
                    service.stop.is_set,
                    cpu_limit=2,
                )
            else:
                raise ValueError("Unsupported occurrence image execution mode")
            receipt_path, image_path = scratch / "receipt.json", scratch / "inspection.png"
            if not regular_file(receipt_path, 64 * 1024) or not regular_file(
                image_path, MAX_IMAGE_BYTES
            ):
                raise ValueError(
                    "Occurrence renderer returned missing, linked, or oversized output"
                )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            image = image_path.read_bytes()
            if (
                not isinstance(receipt, dict)
                or any(receipt.get(key) != value for key, value in request.items())
                or len(image) > MAX_IMAGE_BYTES
                or not image.startswith(b"\x89PNG\r\n\x1a\n")
                or receipt.get("image")
                != {"file": "inspection.png", "sha256": hashlib.sha256(image).hexdigest()}
            ):
                raise ValueError("Occurrence image does not match its source receipt")
            if resolved_focus is not None:
                returned_focus = validate_focus_frame(receipt.get("focus"))
                expected_view = focus_view_metadata(resolved_focus)
                if json.dumps(returned_focus, sort_keys=True, allow_nan=False) != json.dumps(
                    resolved_focus, sort_keys=True, allow_nan=False
                ) or json.dumps(
                    receipt.get("focus_view"), sort_keys=True, allow_nan=False
                ) != json.dumps(expected_view, sort_keys=True, allow_nan=False):
                    raise ValueError(
                        "Occurrence focus image does not match its resolved frame and view"
                    )
            elif "focus" in receipt or "focus_view" in receipt:
                raise ValueError("Occurrence renderer returned unrequested focus metadata")
            response.update(
                receipt=receipt,
                image_base64=base64.b64encode(image).decode("ascii"),
                mime_type="image/png",
            )
    # A concurrent deletion or source mutation cannot publish an observation.
    service.store.session_record(project_id, owner)
    if service.store.revision(project_id, revision_id)["measurement"] != measurement:
        raise ValueError("Inspection revision changed during observation")
    for name in names:
        _artifact(directory, name, hashes.get(name))
    if service.stop.is_set() or time.monotonic() - started > timeout:
        raise ValueError("Inspection cancelled or deadline exhausted")
    if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Inspection exceeds its response limit; request a smaller surface page")
    return response
