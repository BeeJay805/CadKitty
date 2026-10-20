"""Owned immutable photo observations; derived estimates never become source facts."""

from __future__ import annotations

import hashlib
import io
import json
import re
import tempfile
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.customer_references import reference_image
from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.processing import copy_artifact, process_revision, regular_file
from forge_mind.workspace.store import WorkspaceError

SCOPE = "Derived reference advisory only; not an owner fact, acceptance, hidden geometry or physical fit."
MAX_RECORD_BYTES = 4 * 1024 * 1024


def _implementation(service: Any) -> dict[str, Any]:
    mode = service.executor.mode
    return {
        "host_engine_fingerprint": engine_fingerprint(),
        "observer_mode": mode,
        "qualified_image": service.executor.typed_container_image if mode == "docker" else None,
    }


def _guard(
    service: Any, timeout: float, cancelled: Callable[[], bool] | None
) -> tuple[Callable[[], float], Callable[[], bool]]:
    if not 0 < timeout <= 60:
        raise WorkspaceError("Photo profile requires a positive deadline of at most 60 seconds")
    end = time.monotonic() + timeout
    cancel = cancelled or service.stop.is_set

    def remaining() -> float:
        left = end - time.monotonic()
        if cancel() or left <= 0:
            raise WorkspaceError("Photo profile cancelled or deadline exhausted")
        return left

    remaining()
    return remaining, cancel


def _encoded(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_encoded(value).encode()).hexdigest()


def _source(store: Any, sid: str, owner: str) -> tuple[bytes, dict[str, Any]]:
    image, metadata = reference_image(store, sid, owner)
    with Image.open(io.BytesIO(image)) as decoded:
        width, height = decoded.size
    return image, {
        "source_artifact_id": metadata["source_artifact_id"],
        "original_sha256": metadata["sha256"],
        "vision_sha256": metadata["vision_sha256"],
        "normalized_width": width,
        "normalized_height": height,
        "coordinate_space": "retained_normalized_vision_pixels",
    }


def _table(store: Any) -> None:
    with store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS photo_profile_observations (session TEXT NOT NULL, id TEXT NOT NULL, document TEXT NOT NULL, digest TEXT NOT NULL, input_digest TEXT NOT NULL, PRIMARY KEY(session,id), UNIQUE(session,input_digest), FOREIGN KEY(session) REFERENCES sessions(id))"
        )


def _verify_result(result: Any, request: dict[str, Any], source: dict[str, Any]) -> None:
    if (
        not isinstance(result, dict)
        or result.get("vision_image_sha256") != source["vision_sha256"]
        or result.get("request") != request
        or result.get("request_sha256") != _digest(request)
        or result.get("image_size_px") != [source["normalized_width"], source["normalized_height"]]
        or re.fullmatch(r"[a-f0-9]{64}", str(result.get("engine_fingerprint", ""))) is None
        or result.get("observation_sha256")
        != _digest({k: v for k, v in result.items() if k != "observation_sha256"})
    ):
        raise WorkspaceError("Photo worker observation does not match its source and request")


def read_profile(service: Any, sid: str, owner: str, observation_id: str) -> dict[str, Any]:
    _, source = _source(service.store, sid, owner)
    if re.fullmatch(r"[a-f0-9]{32}", observation_id) is None:
        raise WorkspaceError("Invalid photo observation identity")
    _table(service.store)
    with service.store.connect() as con:
        row = con.execute(
            "SELECT document,digest FROM photo_profile_observations WHERE session=? AND id=?",
            (sid, observation_id),
        ).fetchone()
    if row is None:
        raise WorkspaceError("Photo observation not found")
    if len(row[0].encode()) > MAX_RECORD_BYTES:
        raise WorkspaceError("Photo observation exceeds its record bound")
    record: dict[str, Any] = json.loads(row[0])
    if (
        _digest(record) != row[1]
        or record.get("source") != source
        or record.get("id") != observation_id
    ):
        raise WorkspaceError("Photo observation is stale or changed")
    inputs = {"source": source, "request": record.get("request")}
    if "implementation" in record:
        inputs["implementation"] = record["implementation"]
    if record.get("input_digest") != _digest(inputs):
        raise WorkspaceError("Photo observation input binding changed")
    _verify_result(record.get("result"), record["request"], source)
    return record


def _run(
    service: Any,
    directory: Path,
    payload: dict[str, Any],
    timeout: float,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    if not 0 < timeout <= 60:
        raise WorkspaceError("Photo profile requires a positive deadline of at most 60 seconds")
    if cancelled():
        raise WorkspaceError("Photo profile cancelled")
    if service.executor.mode == "trusted_local":
        return process_revision(directory, payload, timeout, cancelled)
    result: dict[str, Any] = service.executor.process_observation(
        directory, payload, timeout, cancelled
    )
    return result


def analyze_reference(
    service: Any,
    sid: str,
    owner: str,
    request: Any,
    *,
    timeout: float = 30,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    from forge_mind.workspace.photo_profile_models import PhotoProfileRequest

    remaining, cancel = _guard(service, timeout, cancelled)
    request = PhotoProfileRequest.model_validate(request).model_dump(mode="json")
    image, source = _source(service.store, sid, owner)
    implementation = _implementation(service)
    if request["vision_image_sha256"] != source["vision_sha256"]:
        raise WorkspaceError("Photo profile request refers to different image bytes")
    identity = _digest({"source": source, "request": request, "implementation": implementation})
    _table(service.store)
    with service.store.connect() as con:
        existing = con.execute(
            "SELECT id FROM photo_profile_observations WHERE session=? AND input_digest=?",
            (sid, identity),
        ).fetchone()
    if existing:
        result = read_profile(service, sid, owner, existing[0])
        if _implementation(service) != implementation:
            raise WorkspaceError("Photo analysis implementation changed during replay")
        remaining()
        return result
    with tempfile.TemporaryDirectory(prefix="forge-photo-profile-") as temporary:
        scratch = Path(temporary)
        (scratch / "vision.png").write_bytes(image)
        result = _run(
            service,
            scratch,
            {"operation": "photo_profile_analyze", "request": request},
            remaining(),
            cancel,
        )
        if (scratch / "vision.png").read_bytes() != image:
            raise WorkspaceError("Photo profile staged image changed")
    if _source(service.store, sid, owner)[1] != source:
        raise WorkspaceError("Photo source changed during observation")
    _verify_result(result, request, source)
    remaining()
    record = {
        "schema_version": 1,
        "id": uuid.uuid4().hex,
        "source": source,
        "input_digest": identity,
        "implementation": implementation,
        "request": request,
        "result": result,
        "scope": SCOPE,
    }
    encoded = _encoded(record)
    if len(encoded.encode()) > MAX_RECORD_BYTES:
        raise WorkspaceError("Photo profile observation exceeds its record limit")
    with service.store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        # Repeat ownership and source identity at publication; no session/plan mutation.
        if not con.execute(
            "SELECT 1 FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone():
            raise WorkspaceError("workspace not found")
        existing = con.execute(
            "SELECT id FROM photo_profile_observations WHERE session=? AND input_digest=?",
            (sid, identity),
        ).fetchone()
        if _implementation(service) != implementation:
            raise WorkspaceError("Photo analysis implementation changed before publication")
        if not existing:
            if _source(service.store, sid, owner)[1] != source:
                raise WorkspaceError("Photo source changed before observation publication")
            remaining()
            con.execute(
                "INSERT INTO photo_profile_observations VALUES (?,?,?,?,?)",
                (sid, record["id"], encoded, _digest(record), identity),
            )
    if existing:
        result = read_profile(service, sid, owner, existing[0])
        remaining()
        return result
    return record


def compare_reference(
    service: Any,
    sid: str,
    owner: str,
    observation_id: str,
    revision_id: str,
    selector: Any,
    *,
    timeout: float = 30,
    cancelled: Callable[[], bool] | None = None,
    part_id: str | None = None,
) -> dict[str, Any]:
    from forge_mind.workspace.photo_profile_models import PhotoProfileSelector

    remaining, cancel = _guard(service, timeout, cancelled)
    observation = read_profile(service, sid, owner, observation_id)
    selection = PhotoProfileSelector.model_validate(selector).model_dump(mode="json")
    revision = service.store.revision(sid, revision_id)
    artifact = "part.step"
    if part_id is not None:
        from forge_mind.workspace.document import DesignDocument
        from forge_mind.workspace.verification_admission import _part_artifact

        artifact = _part_artifact(DesignDocument.model_validate(revision["document"]), part_id)
    source = service.store.directory(sid, revision_id) / artifact
    expected = revision["measurement"]["artifact_hashes"].get(artifact)
    if not regular_file(source) or file_sha256(source) != expected:
        raise WorkspaceError("Photo comparison STEP is missing or stale")
    with tempfile.TemporaryDirectory(prefix="forge-photo-compare-") as temporary:
        scratch = Path(temporary)
        copy_artifact(source, scratch / "part.step")
        if file_sha256(scratch / "part.step") != expected or file_sha256(source) != expected:
            raise WorkspaceError("Photo comparison STEP changed during staging")
        result = _run(
            service,
            scratch,
            {
                "operation": "photo_profile_compare",
                "observation": observation["result"],
                "selector": selection,
            },
            remaining(),
            cancel,
        )
        if file_sha256(scratch / "part.step") != expected or file_sha256(source) != expected:
            raise WorkspaceError("Photo comparison STEP changed during observation")
    read_profile(service, sid, owner, observation_id)
    remaining()
    if (
        result.get("step_sha256") != expected
        or result.get("observation_sha256") != observation["result"]["observation_sha256"]
        or result.get("selector") != selection
        or result.get("vision_image_sha256") != observation["source"]["vision_sha256"]
        or re.fullmatch(r"[a-f0-9]{64}", str(result.get("engine_fingerprint", ""))) is None
        or result.get("comparison_sha256")
        != _digest({k: v for k, v in result.items() if k != "comparison_sha256"})
    ):
        raise WorkspaceError("Photo comparison result does not match its staged inputs")
    return {
        "schema_version": 1,
        "observation_id": observation_id,
        "observation_digest": _digest(observation),
        "revision_id": revision_id,
        "source": observation["source"],
        "engine_fingerprint": result["engine_fingerprint"],
        "revision_engine_fingerprint": revision["measurement"].get("engine_fingerprint"),
        "step_sha256": expected,
        "source_step_artifact": artifact,
        "selector": selection,
        "result": result,
        "scope": SCOPE,
    }


def profile_discovery(store: Any, sid: str, owner: str) -> dict[str, Any]:
    _, source = _source(store, sid, owner)
    session = store.session_record(sid, owner)
    return {
        "schema_query": "photo_profile",
        "action_kind": "photo_profile",
        "source": source,
        "known_dimension": session["customer_reference"].get("known_dimension"),
        "usage": "kind=photo_profile; query=analyze|read|compare; document_json is a PhotoProfileAction wrapper with matching action. schema query=photo_profile gives exact syntax. No calibration still returns useful pixel geometry and ratios. Anchors are normalized image pixels. Proposed observations do not become owner facts. Never invent calibration or thickness. Existing construction guards remain required.",
        "example": {
            "kind": "photo_profile",
            "query": "read",
            "document_json": '{"action":"read","observation_id":"<returned id>"}',
        },
        "scope": SCOPE,
    }
