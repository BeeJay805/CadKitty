"""Distance service boundaries with saved bytes and mocked execution, never CAD."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from forge_mind.workspace import geometry_distance as distance
from forge_mind.workspace.geometry_distance_contract import (
    MAX_RESPONSE_BYTES,
    SCOPE,
    DistanceSelector,
)

SID, RID = "a" * 32, "b" * 32
SAVED_ENGINE, QUERY_ENGINE = "c" * 64, "d" * 64
BODY_A, BODY_B, FACE_A, FACE_B = (letter * 64 for letter in "ef01")


def fixture_service(tmp_path, monkeypatch, *, mode="trusted_local", qualified=False):
    directory = tmp_path / SID / RID
    directory.mkdir(parents=True)
    topology = {
        "frame": "cad_z_up_mm",
        "entities": [
            {
                "kind": "body",
                "body_index": index,
                "signature": signature,
                "body_signature": signature,
            }
            for index, signature in enumerate((BODY_A, BODY_B))
        ]
        + [
            {"kind": "face", "body_index": 0, "signature": signature, "body_signature": BODY_A}
            for signature in (FACE_A, FACE_B)
        ],
    }
    contents = {
        "part.step": b"saved STEP fixture; never parsed",
        "document.json": b'{"name":"fixture"}',
        "topology.json": json.dumps(topology).encode(),
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    record = {
        "id": RID,
        "contract": {"requirements": []},
        "measurement": {
            "revision_id": RID,
            "engine_fingerprint": SAVED_ENGINE,
            "artifact_hashes": hashes,
            "facts": {
                "assembly": {
                    "revision_id": RID,
                    "step_sha256": hashes["part.step"],
                    "engine_fingerprint": SAVED_ENGINE,
                    "occurrences": [
                        {"path": "first", "body_index": 0},
                        {"path": "second", "body_index": 1},
                    ],
                }
            },
        },
    }
    calls = {"owner": [], "revision": [], "worker": [], "scratch": []}

    def owned(sid, owner):
        calls["owner"].append((sid, owner))
        if owner != "owner":
            raise ValueError("Foreign owner")
        return {"id": sid}

    def revision(sid, rid):
        calls["revision"].append((sid, rid))
        return record

    service = SimpleNamespace(
        store=SimpleNamespace(
            session_record=owned, revision=revision, directory=lambda *args: directory
        ),
        executor=SimpleNamespace(
            _docker_path=lambda discover: discover("docker"),
            mode=mode,
            typed_container_verified=qualified,
            typed_container_image="qualified@sha256:fixture",
        ),
        stop=Event(),
    )

    def worker(command, scratch, timeout, cancelled, **kwargs):
        calls["worker"].append((command, timeout, kwargs))
        calls["scratch"].append(scratch)
        assert not cancelled() and 0 < timeout <= 60
        assert {name: (scratch / name).read_bytes() for name in contents} == contents
        request = json.loads((scratch / "request.json").read_bytes())
        write_result(scratch, request)

    monkeypatch.setattr(distance, "bounded_process", worker)
    monkeypatch.setattr(distance, "engine_fingerprint", lambda: QUERY_ENGINE)
    return service, directory, contents, record, calls, worker


def write_result(scratch, request, *, mutate=None):
    result = {
        **request,
        "minimum_distance_mm": 2.0,
        "witness_a_mm": [0.0, 0.0, 0.0],
        "witness_b_mm": [2.0, 0.0, 0.0],
        "solution_count": 4,
        "native_completed": True,
        "native_inner_solution": False,
        "scope": SCOPE,
    }
    if mutate:
        mutate(result)
    (scratch / "result.json").write_text(json.dumps(result), encoding="utf-8")


def query(service, **kwargs):
    return distance.measure_distance(
        service,
        SID,
        "owner",
        RID,
        DistanceSelector(occurrence_path="first"),
        DistanceSelector(body_index=1),
        **kwargs,
    )


def test_owned_distance_preserves_old_engine_and_all_saved_bytes(tmp_path, monkeypatch):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    result = query(service)
    assert result["a"]["body_index"] == 0
    assert result["a"]["requested"]["occurrence_path"] == "first"
    assert result["b"]["body_signature"] == BODY_B
    assert result["saved_engine_fingerprint"] == SAVED_ENGINE
    assert result["query_engine_fingerprint"] == QUERY_ENGINE
    assert result["minimum_distance_mm"] == 2.0 and result["solution_count"] == 4
    assert result["scope"] == SCOPE
    assert record == before
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents
    assert calls["owner"] == [(SID, "owner"), (SID, "owner")]
    assert len(calls["worker"]) == 1 and calls["worker"][0][2] == {"cpu_limit": 2}
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_distinct_faces_on_same_body_are_allowed(tmp_path, monkeypatch):
    service, _, _, _, _, _ = fixture_service(tmp_path, monkeypatch)
    result = distance.measure_distance(
        service,
        SID,
        "owner",
        RID,
        DistanceSelector(body_index=0, face_signature=FACE_A),
        DistanceSelector(body_index=0, face_signature=FACE_B),
    )
    assert result["a"]["body_index"] == result["b"]["body_index"] == 0
    assert result["a"]["face_signature"] == FACE_A
    assert result["b"]["face_signature"] == FACE_B


def test_foreign_owner_refuses_before_revision_or_worker(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Foreign owner"):
        distance.measure_distance(
            service,
            SID,
            "other",
            RID,
            DistanceSelector(body_index=0),
            DistanceSelector(body_index=1),
        )
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("timeout", [True, 0, 61, float("nan"), float("inf"), "30"])
def test_invalid_deadline_never_reads_saved_data(tmp_path, monkeypatch, timeout):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        query(service, timeout=timeout)
    assert calls["owner"] == calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("mode", ["docker", "unsupported"])
def test_unqualified_worker_never_loads_sources(tmp_path, monkeypatch, mode):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch, mode=mode)
    monkeypatch.setattr(distance, "_artifact", lambda *args: pytest.fail("No source loading"))
    with pytest.raises(ValueError, match=r"qualified|Unsupported"):
        query(service)
    assert calls["worker"] == []


def test_qualified_container_uses_exact_image_and_staged_context(tmp_path, monkeypatch):
    service, _, _, _, calls, worker = fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )
    monkeypatch.setattr(distance.shutil, "which", lambda name: "fixture-docker")

    def container(docker, scratch, module, request, timeout, cancelled, *, image):
        assert docker == "fixture-docker"
        assert module == "forge_mind.workspace.geometry_distance_worker"
        assert request == "request.json" and image == "qualified@sha256:fixture"
        worker([module], scratch, timeout, cancelled)

    service.executor._run_container = container
    monkeypatch.setattr(
        distance, "bounded_process", lambda *args, **kwargs: pytest.fail("No host child")
    )
    assert query(service)["query_engine_fingerprint"] == QUERY_ENGINE
    assert len(calls["worker"]) == 1


@pytest.mark.parametrize(
    "change", ["mapping", "alias", "provenance", "body", "face", "frame", "identical"]
)
def test_missing_ambiguous_or_wrong_selection_refuses_before_dispatch(
    tmp_path, monkeypatch, change
):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    a, b = DistanceSelector(occurrence_path="first"), DistanceSelector(body_index=1)
    assembly = record["measurement"]["facts"]["assembly"]
    topology = json.loads(contents["topology.json"])
    if change == "mapping":
        assembly["occurrences"].append(deepcopy(assembly["occurrences"][0]))
    elif change == "alias":
        assembly["occurrences"][1]["body_index"] = 0
    elif change == "provenance":
        assembly["step_sha256"] = "0" * 64
    elif change == "body":
        topology["entities"].append(deepcopy(topology["entities"][0]))
    elif change == "face":
        a = DistanceSelector(body_index=0, face_signature="2" * 64)
    elif change == "frame":
        topology["frame"] = "definition_local"
    else:
        b = DistanceSelector(body_index=0)
    content = json.dumps(topology).encode()
    (directory / "topology.json").write_bytes(content)
    record["measurement"]["artifact_hashes"]["topology.json"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError):
        distance.measure_distance(service, SID, "owner", RID, a, b)
    assert calls["worker"] == []


@pytest.mark.parametrize("name", ["part.step", "document.json", "topology.json"])
def test_changed_source_refuses_before_worker(tmp_path, monkeypatch, name):
    service, directory, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    (directory / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        query(service)
    assert calls["worker"] == []


@pytest.mark.parametrize("change", ["source", "record", "owner", "cancel", "deadline", "engine"])
def test_concurrent_changes_never_publish_and_cleanup(tmp_path, monkeypatch, change):
    service, directory, _, record, calls, worker = fixture_service(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(distance.time, "monotonic", lambda: clock[0])

    def changed(*args, **kwargs):
        worker(*args, **kwargs)
        if change == "source":
            (directory / "part.step").write_bytes(b"changed")
        elif change == "record":
            record["contract"]["requirements"].append({"changed": True})
        elif change == "owner":

            def denied(*args):
                raise ValueError("Owner changed")

            service.store.session_record = denied
        elif change == "cancel":
            service.stop.set()
        elif change == "deadline":
            clock[0] = 31
        else:
            monkeypatch.setattr(distance, "engine_fingerprint", lambda: "0" * 64)

    monkeypatch.setattr(distance, "bounded_process", changed)
    with pytest.raises(ValueError):
        query(service)
    assert all(not scratch.exists() for scratch in calls["scratch"])


@pytest.mark.parametrize(
    "change", ["project", "hash", "selection", "incomplete", "witness", "extra", "oversize"]
)
def test_invalid_worker_output_refuses_and_cleanup(tmp_path, monkeypatch, change):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def corrupt(result):
        if change == "project":
            result["project_id"] = "0" * 32
        elif change == "hash":
            result["source_hashes"]["part.step"] = "0" * 64
        elif change == "selection":
            result["a"]["body_signature"] = "0" * 64
        elif change == "incomplete":
            result["native_completed"] = False
        elif change == "witness":
            result["witness_b_mm"] = [3.0, 0.0, 0.0]
        elif change == "extra":
            result["unexpected"] = True
        else:
            result["scope"] = "x" * (MAX_RESPONSE_BYTES + 1)

    def bad(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()), mutate=corrupt)

    monkeypatch.setattr(distance, "bounded_process", bad)
    with pytest.raises(ValueError):
        query(service)
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_worker_failure_propagates_without_durable_output(tmp_path, monkeypatch):
    service, directory, contents, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def failed(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        raise RuntimeError("Native process failed")

    monkeypatch.setattr(distance, "bounded_process", failed)
    with pytest.raises(RuntimeError, match="Native process failed"):
        query(service)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents
    assert all(not scratch.exists() for scratch in calls["scratch"])
