"""Owner-scoped ephemeral cylindrical access over immutable saved STEP bytes."""

from __future__ import annotations

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

from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.geometry_access_contract import (
    MAX_INPUT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_TIMEOUT,
    MAX_TOTAL_INPUT_BYTES,
    SOURCE_NAMES,
    AccessBodySelector,
    AccessContext,
    AccessQuery,
    AccessResult,
    ResolvedAccessBody,
    evaluate_query,
)
from forge_mind.workspace.processing import bounded_process
from forge_mind.workspace.section_render import finite_number, read_bytes, read_object

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _artifact(directory: Path, name: str, expected: Any) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Access source has no saved checksum")
    content = read_bytes(directory / name, MAX_INPUT_BYTES)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Access source changed; saved evidence is stale")
    return content


def _inventory(topology: dict[str, Any]) -> dict[int, str]:
    entities = topology.get("entities")
    if topology.get("frame") != "cad_z_up_mm" or not isinstance(entities, list):
        raise ValueError("Access requires a saved CAD Z-up topology inventory")
    if any(not isinstance(row, dict) for row in entities):
        raise ValueError("Malformed saved topology inventory")
    bodies: dict[int, str] = {}
    for row in entities:
        if row.get("kind") != "body":
            continue
        index, signature = row.get("body_index"), row.get("signature")
        if (
            type(index) is not int
            or index < 0
            or index in bodies
            or not isinstance(signature, str)
            or re.fullmatch(r"[a-f0-9]{64}", signature) is None
            or row.get("body_signature") != signature
        ):
            raise ValueError("Access body inventory is missing or ambiguous")
        bodies[index] = signature
    if not bodies or set(bodies) != set(range(len(bodies))):
        raise ValueError("Access requires complete contiguous saved body inventory")
    return bodies


def _resolve(
    requested: AccessBodySelector,
    bodies: dict[int, str],
    measurement: dict[str, Any],
    revision_id: str,
) -> ResolvedAccessBody:
    index = requested.body_index
    if requested.occurrence_path is not None:
        facts = measurement.get("facts")
        assembly = facts.get("assembly") if isinstance(facts, dict) else None
        if not isinstance(assembly, dict) or not isinstance(assembly.get("occurrences"), list):
            raise ValueError("Access occurrence requires a saved assembly mapping")
        rows = assembly["occurrences"]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Malformed saved occurrence mapping")
        selected = [row for row in rows if row.get("path") == requested.occurrence_path]
        if len(selected) != 1 or type(selected[0].get("body_index")) is not int:
            raise ValueError("Access occurrence identity is missing or ambiguous")
        index = selected[0]["body_index"]
        if index < 0 or sum(row.get("body_index") == index for row in rows) != 1:
            raise ValueError("Access occurrence body is missing or ambiguous")
        for key, expected in (
            ("revision_id", revision_id),
            ("step_sha256", measurement["artifact_hashes"]["part.step"]),
            ("engine_fingerprint", measurement.get("engine_fingerprint")),
        ):
            if key in assembly and assembly[key] != expected:
                raise ValueError("Saved occurrence mapping has different source provenance")
    if type(index) is not int or index not in bodies:
        raise ValueError("Access selected body is missing from saved inventory")
    return ResolvedAccessBody(requested=requested, body_index=index, body_signature=bodies[index])


def inspect_cylindrical_access(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    revision_id: str,
    query: AccessQuery,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    """Observe one declared cylinder path without building CAD or storing evidence."""
    for value in (project_id, revision_id):
        if not isinstance(value, str) or re.fullmatch(r"[a-f0-9]{32}", value) is None:
            raise ValueError("Invalid access project or revision identifier")
    query = AccessQuery.model_validate(query)
    evaluated_cylinder, evaluated_motion = evaluate_query(query)
    deadline = finite_number(timeout)
    if not 0 < deadline <= MAX_TIMEOUT:
        raise ValueError("Access requires a finite deadline of at most 60 seconds")
    started = time.monotonic()

    def remaining() -> float:
        value = deadline - (time.monotonic() - started)
        if value <= 0 or service.stop.is_set():
            raise ValueError("Access cancelled or deadline exhausted")
        return value

    service.store.session_record(project_id, owner)
    remaining()
    record = deepcopy(service.store.revision(project_id, revision_id))
    measurement = record["measurement"]
    if record.get("id") != revision_id or measurement.get("revision_id") != revision_id:
        raise ValueError("Access measurement belongs to another revision")
    executor = service.executor
    if executor.mode == "docker" and not executor.typed_container_verified:
        raise ValueError("Access requires qualified isolated native worker execution")
    if executor.mode not in ("docker", "trusted_local"):
        raise ValueError("Unsupported access execution mode")
    directory = service.store.directory(project_id, revision_id)
    hashes = measurement.get("artifact_hashes", {})
    source: dict[str, bytes] = {}
    for name in sorted(SOURCE_NAMES):
        remaining()
        source[name] = _artifact(directory, name, hashes.get(name))
    if sum(map(len, source.values())) > MAX_TOTAL_INPUT_BYTES:
        raise ValueError("Access source exceeds its aggregate byte limit")
    topology = read_object(source["topology.json"])
    read_object(source["document.json"])
    bodies = _inventory(topology)
    obstacles = [_resolve(item, bodies, measurement, revision_id) for item in query.obstacles]
    selected = [body.body_index for body in obstacles]
    if len(set(selected)) != len(selected):
        raise ValueError("Access refuses duplicate resolved obstacle bodies")
    context = AccessContext(
        project_id=project_id,
        revision_id=revision_id,
        query=query,
        evaluated_cylinder=evaluated_cylinder,
        evaluated_motion=evaluated_motion,
        obstacles=obstacles,
        inventory_body_count=len(bodies),
        omitted_body_indices=sorted(set(bodies) - set(selected)),
        source_hashes={name: hashes[name] for name in SOURCE_NAMES},
        saved_engine_fingerprint=measurement.get("engine_fingerprint"),
        query_engine_fingerprint=engine_fingerprint(),
    )
    encoded_request = context.model_dump_json().encode("utf-8")
    if len(encoded_request) > MAX_REQUEST_BYTES:
        raise ValueError("Access request exceeds its byte limit")
    with tempfile.TemporaryDirectory(prefix="forge-access-") as temporary:
        scratch = Path(temporary)
        for name, content in source.items():
            (scratch / name).write_bytes(content)
        request_path = scratch / "request.json"
        request_path.write_bytes(encoded_request)
        module = "forge_mind.workspace.geometry_access_worker"
        if executor.mode == "docker":
            docker = executor._docker_path(shutil.which)
            if not docker:
                raise ValueError("Docker is required for isolated native access queries")
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
        result = AccessResult.model_validate(
            read_object(read_bytes(scratch / "result.json", MAX_RESPONSE_BYTES))
        )
        response = result.model_dump(mode="json")
        returned_context = AccessContext.model_validate(
            {key: response[key] for key in AccessContext.model_fields}
        )
        if returned_context != context:
            raise ValueError("Access result does not match staged source and selection")
    service.store.session_record(project_id, owner)
    if service.store.revision(project_id, revision_id) != record:
        raise ValueError("Access revision changed during observation")
    for name in source:
        remaining()
        _artifact(directory, name, hashes[name])
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Access query engine changed during observation")
    remaining()
    if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Access result exceeds its response byte budget")
    return response
