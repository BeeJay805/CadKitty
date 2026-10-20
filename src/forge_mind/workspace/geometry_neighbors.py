"""Owner-scoped ephemeral neighborhoods over immutable saved STEP bytes."""

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
from forge_mind.workspace.geometry_neighbors_contract import (
    MAX_INPUT_BYTES,
    MAX_RESPONSE_BYTES,
    SOURCE_NAMES,
    NeighborContext,
    NeighborQuery,
    NeighborResult,
    NeighborSelector,
    ResolvedNeighborSelector,
)
from forge_mind.workspace.processing import bounded_process
from forge_mind.workspace.section_render import finite_number, read_bytes, read_object

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _artifact(directory: Path, name: str, expected: Any) -> bytes:
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("Neighbors source has no saved checksum")
    content = read_bytes(directory / name, MAX_INPUT_BYTES)
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError("Neighbors source changed; saved evidence is stale")
    return content


def _resolve(
    requested: NeighborSelector,
    topology: dict[str, Any],
    measurement: dict[str, Any],
    revision_id: str,
) -> ResolvedNeighborSelector:
    index = requested.body_index
    if requested.occurrence_path is not None:
        assembly = measurement.get("facts", {}).get("assembly")
        if not isinstance(assembly, dict) or not isinstance(assembly.get("occurrences"), list):
            raise ValueError("Neighbors occurrence requires a saved assembly mapping")
        rows = assembly["occurrences"]
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("Malformed saved occurrence mapping")
        selected = [row for row in rows if row.get("path") == requested.occurrence_path]
        if len(selected) != 1 or type(selected[0].get("body_index")) is not int:
            raise ValueError("Neighbors occurrence identity is missing or ambiguous")
        index = selected[0]["body_index"]
        if index < 0 or sum(row.get("body_index") == index for row in rows) != 1:
            raise ValueError("Neighbors occurrence body is missing or ambiguous")
        for key, expected in (
            ("revision_id", revision_id),
            ("step_sha256", measurement["artifact_hashes"]["part.step"]),
            ("engine_fingerprint", measurement.get("engine_fingerprint")),
        ):
            if key in assembly and assembly[key] != expected:
                raise ValueError("Saved occurrence mapping has different source provenance")
    entities = topology.get("entities")
    if topology.get("frame") != "cad_z_up_mm" or not isinstance(entities, list):
        raise ValueError("Neighbors require a saved CAD Z-up topology inventory")
    if any(not isinstance(row, dict) for row in entities):
        raise ValueError("Malformed saved topology inventory")
    bodies = [
        row for row in entities if row.get("kind") == "body" and row.get("body_index") == index
    ]
    if len(bodies) != 1 or type(bodies[0].get("body_index")) is not int:
        raise ValueError("Neighbors body identity is missing or ambiguous")
    body = bodies[0]
    if body.get("body_signature") != body.get("signature"):
        raise ValueError("Saved body signature does not match topology identity")
    resolved = ResolvedNeighborSelector(
        requested=requested,
        body_index=body["body_index"],
        body_signature=body["signature"],
    )
    selected = [
        row
        for row in entities
        if row.get("kind") == requested.kind
        and row.get("body_index") == index
        and row.get("signature") == requested.signature
    ]
    if (
        len(selected) != 1
        or type(selected[0].get("body_index")) is not int
        or selected[0].get("body_signature") != resolved.body_signature
    ):
        raise ValueError("Neighbors entity identity is unavailable or ambiguous")
    return resolved


def inspect_topology_neighbors(
    service: WorkspaceService,
    project_id: str,
    owner: str,
    revision_id: str,
    selector: NeighborSelector,
    offset: int = 0,
    limit: int = 40,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    """Observe one saved neighborhood without rebuilding CAD or storing acceptance evidence."""
    for value in (project_id, revision_id):
        if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
            raise ValueError("Invalid neighbors project or revision identifier")
    query = NeighborQuery(selector=selector, offset=offset, limit=limit)
    deadline = finite_number(timeout)
    if not 0 < deadline <= 60:
        raise ValueError("Neighbors require a finite deadline of at most 60 seconds")
    started = time.monotonic()

    def remaining() -> float:
        value = deadline - (time.monotonic() - started)
        if value <= 0 or service.stop.is_set():
            raise ValueError("Neighbors cancelled or deadline exhausted")
        return value

    service.store.session_record(project_id, owner)
    remaining()
    record = deepcopy(service.store.revision(project_id, revision_id))
    measurement = record["measurement"]
    if record.get("id") != revision_id or measurement.get("revision_id") != revision_id:
        raise ValueError("Neighbors measurement belongs to another revision")
    executor = service.executor
    if executor.mode == "docker" and not executor.typed_container_verified:
        raise ValueError("Neighbors require qualified isolated native worker execution")
    if executor.mode not in ("docker", "trusted_local"):
        raise ValueError("Unsupported neighbors execution mode")
    directory = service.store.directory(project_id, revision_id)
    hashes = measurement.get("artifact_hashes", {})
    source: dict[str, bytes] = {}
    for name in sorted(SOURCE_NAMES):
        remaining()
        source[name] = _artifact(directory, name, hashes.get(name))
    topology = read_object(source["topology.json"])
    # Document bytes are bound even though the query does not interpret design intent.
    read_object(source["document.json"])
    context = NeighborContext(
        project_id=project_id,
        revision_id=revision_id,
        selector=_resolve(query.selector, topology, measurement, revision_id),
        offset=query.offset,
        limit=query.limit,
        source_hashes={name: hashes[name] for name in SOURCE_NAMES},
        saved_engine_fingerprint=measurement.get("engine_fingerprint"),
        query_engine_fingerprint=engine_fingerprint(),
    )
    with tempfile.TemporaryDirectory(prefix="forge-neighbors-") as temporary:
        scratch = Path(temporary)
        for name, content in source.items():
            (scratch / name).write_bytes(content)
        request_path = scratch / "request.json"
        request_path.write_text(context.model_dump_json(), encoding="utf-8")
        module = "forge_mind.workspace.geometry_neighbors_worker"
        if executor.mode == "docker":
            docker = executor._docker_path(shutil.which)
            if not docker:
                raise ValueError("Docker is required for isolated native neighbors queries")
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
        result = NeighborResult.model_validate(
            read_object(read_bytes(scratch / "result.json", MAX_RESPONSE_BYTES))
        )
        response = result.model_dump(mode="json")
        returned_context = NeighborContext.model_validate(
            {key: response[key] for key in NeighborContext.model_fields}
        )
        if returned_context != context:
            raise ValueError("Neighbors result does not match the staged source and selection")
    service.store.session_record(project_id, owner)
    if service.store.revision(project_id, revision_id) != record:
        raise ValueError("Neighbors revision changed during observation")
    for name in source:
        remaining()
        _artifact(directory, name, hashes[name])
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Neighbors query engine changed during observation")
    remaining()
    if len(json.dumps(response, allow_nan=False).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Neighbors result exceeds its response byte budget")
    return response
