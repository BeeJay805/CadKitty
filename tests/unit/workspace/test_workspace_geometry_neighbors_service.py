"""Neighbors service boundaries with saved bytes and mocked execution, never CAD."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from forge_mind.workspace import geometry_neighbors as neighbors
from forge_mind.workspace.geometry_neighbors_contract import (
    MAX_RESPONSE_BYTES,
    SCOPE,
    NeighborSelector,
)
from forge_mind.workspace.store import WorkspaceError

SID, RID = "a" * 32, "b" * 32
SAVED_ENGINE, QUERY_ENGINE = "c" * 64, "d" * 64
BODY_A, BODY_B, FACE_A, EDGE_A = (letter * 64 for letter in "ef01")


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
            {"kind": kind, "body_index": 0, "signature": signature, "body_signature": BODY_A}
            for kind, signature in (("face", FACE_A), ("edge", EDGE_A))
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
            raise WorkspaceError("Foreign owner")
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

    monkeypatch.setattr(neighbors, "bounded_process", worker)
    monkeypatch.setattr(neighbors, "engine_fingerprint", lambda: QUERY_ENGINE)
    return service, directory, contents, record, calls, worker


def fact(kind, signature):
    return {
        "id": kind + "_" + signature[:20],
        "kind": kind,
        "signature": signature,
        "body_index": 0,
        "body_signature": BODY_A,
        "center": [0.5, 0.5, 0.0],
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 0.0],
        "measure": 1.0,
        "unit": "mm^2" if kind == "face" else "mm",
        **(
            {"surface": "PLANE", "normal": [0.0, 0.0, 1.0]} if kind == "face" else {"curve": "LINE"}
        ),
    }


def write_result(scratch, request, *, mutate=None):
    kind = request["selector"]["requested"]["kind"]
    other = "edge" if kind == "face" else "face"
    items = [{"entity": fact(other, EDGE_A if other == "edge" else FACE_A), "incidence_count": 2}]
    offset, limit = request["offset"], request["limit"]
    page = items[offset : offset + limit]
    result = {
        **request,
        "focal": fact(kind, request["selector"]["requested"]["signature"]),
        "relation": "face_boundary_edges" if kind == "face" else "edge_incident_faces",
        "neighbors": page,
        "total_neighbors": 1,
        "total_incidences": 2,
        "next_offset": None,
        "native_completed": True,
        "coverage": "complete_selected_neighborhood",
        "scope": SCOPE,
    }
    if mutate:
        mutate(result)
    (scratch / "result.json").write_text(json.dumps(result), encoding="utf-8")


def query(service, **kwargs):
    return neighbors.inspect_topology_neighbors(
        service,
        SID,
        "owner",
        RID,
        NeighborSelector(occurrence_path="first", kind="face", signature=FACE_A),
        **kwargs,
    )


def test_owned_neighbors_preserve_old_engine_saved_bytes_and_seam_count(tmp_path, monkeypatch):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    result = query(service)
    assert result["selector"]["body_index"] == 0
    assert result["selector"]["requested"]["occurrence_path"] == "first"
    assert result["saved_engine_fingerprint"] == SAVED_ENGINE
    assert result["query_engine_fingerprint"] == QUERY_ENGINE
    assert result["neighbors"][0]["incidence_count"] == result["total_incidences"] == 2
    assert result["total_neighbors"] == 1 and result["scope"] == SCOPE
    assert record == before
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents
    assert calls["owner"] == [(SID, "owner"), (SID, "owner")]
    command, _, options = calls["worker"][0]
    assert command[1:4] == ["-B", "-m", "forge_mind.workspace.geometry_neighbors_worker"]
    assert options == {"cpu_limit": 2} and len(calls["worker"]) == 1
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_edge_selection_and_empty_out_of_range_page_are_not_failed_queries(tmp_path, monkeypatch):
    service, _, _, _, _, _ = fixture_service(tmp_path, monkeypatch)
    result = neighbors.inspect_topology_neighbors(
        service,
        SID,
        "owner",
        RID,
        NeighborSelector(body_index=0, kind="edge", signature=EDGE_A),
        offset=4,
        limit=1,
    )
    assert result["relation"] == "edge_incident_faces"
    assert result["neighbors"] == [] and result["next_offset"] is None
    assert result["total_neighbors"] == 1 and result["total_incidences"] == 2


def test_foreign_owner_refuses_before_revision_or_worker(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(WorkspaceError, match="Foreign owner"):
        neighbors.inspect_topology_neighbors(
            service,
            SID,
            "other",
            RID,
            NeighborSelector(body_index=0, kind="face", signature=FACE_A),
        )
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize(
    "options", [{"offset": True}, {"offset": -1}, {"limit": True}, {"limit": 0}, {"limit": 101}]
)
def test_invalid_page_never_reads_saved_data(tmp_path, monkeypatch, options):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        query(service, **options)
    assert calls["owner"] == calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize(
    "change",
    [
        "mapping",
        "alias",
        "provenance",
        "body",
        "entity",
        "duplicate_entity",
        "entity_body",
        "frame",
        "revision",
    ],
)
def test_unavailable_or_ambiguous_selection_never_dispatches(tmp_path, monkeypatch, change):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
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
    elif change == "entity":
        topology["entities"] = [row for row in topology["entities"] if row.get("kind") != "face"]
    elif change == "duplicate_entity":
        topology["entities"].append(deepcopy(topology["entities"][2]))
    elif change == "entity_body":
        topology["entities"][2]["body_signature"] = BODY_B
    elif change == "frame":
        topology["frame"] = "definition_local"
    else:
        record["measurement"]["revision_id"] = "0" * 32
    content = json.dumps(topology).encode()
    (directory / "topology.json").write_bytes(content)
    record["measurement"]["artifact_hashes"]["topology.json"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError):
        query(service)
    assert calls["worker"] == []


@pytest.mark.parametrize(
    "change",
    ["project", "hash", "selection", "incomplete", "identity", "paging", "extra", "oversize"],
)
def test_invalid_worker_output_refuses_and_cleanup(tmp_path, monkeypatch, change):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def corrupt(result):
        if change == "project":
            result["project_id"] = "0" * 32
        elif change == "hash":
            result["source_hashes"]["part.step"] = "0" * 64
        elif change == "selection":
            result["selector"]["requested"]["occurrence_path"] = "second"
        elif change == "incomplete":
            result["native_completed"] = False
        elif change == "identity":
            result["neighbors"][0]["entity"]["body_signature"] = BODY_B
        elif change == "paging":
            result["next_offset"] = 1
        elif change == "extra":
            result["unexpected"] = True
        else:
            result["scope"] = "x" * (MAX_RESPONSE_BYTES + 1)

    def bad(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()), mutate=corrupt)

    monkeypatch.setattr(neighbors, "bounded_process", bad)
    with pytest.raises(ValueError):
        query(service)
    assert all(not scratch.exists() for scratch in calls["scratch"])


@pytest.mark.parametrize(
    "failure", [RuntimeError("Native process failed"), TimeoutError("Worker deadline")]
)
def test_worker_failure_propagates_without_durable_output(tmp_path, monkeypatch, failure):
    service, directory, contents, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def failed(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        raise failure

    monkeypatch.setattr(neighbors, "bounded_process", failed)
    with pytest.raises(type(failure), match=str(failure)):
        query(service)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents
    assert all(not scratch.exists() for scratch in calls["scratch"])


@pytest.mark.parametrize("timeout", [True, 0, 61, float("nan"), float("inf"), "30"])
def test_invalid_deadline_never_reads_saved_data(tmp_path, monkeypatch, timeout):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        query(service, timeout=timeout)
    assert calls["owner"] == calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("mode", ["docker", "unsupported"])
def test_unqualified_worker_never_loads_sources(tmp_path, monkeypatch, mode):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch, mode=mode)
    monkeypatch.setattr(neighbors, "_artifact", lambda *args: pytest.fail("No source loading"))
    with pytest.raises(ValueError, match=r"qualified|Unsupported"):
        query(service)
    assert calls["worker"] == []


def test_qualified_container_uses_exact_image_and_staged_context(tmp_path, monkeypatch):
    service, _, _, _, calls, worker = fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )
    monkeypatch.setattr(neighbors.shutil, "which", lambda name: "fixture-docker")

    def container(docker, scratch, module, request, timeout, cancelled, *, image):
        assert docker == "fixture-docker"
        assert module == "forge_mind.workspace.geometry_neighbors_worker"
        assert request == "request.json" and image == "qualified@sha256:fixture"
        worker([module], scratch, timeout, cancelled)

    service.executor._run_container = container
    monkeypatch.setattr(
        neighbors, "bounded_process", lambda *args, **kwargs: pytest.fail("No host child")
    )
    assert query(service)["query_engine_fingerprint"] == QUERY_ENGINE
    assert len(calls["worker"]) == 1


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
    monkeypatch.setattr(neighbors.time, "monotonic", lambda: clock[0])

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
            monkeypatch.setattr(neighbors, "engine_fingerprint", lambda: "0" * 64)

    monkeypatch.setattr(neighbors, "bounded_process", changed)
    with pytest.raises(ValueError):
        query(service)
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_cancelled_service_never_dispatches(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    service.stop.set()
    with pytest.raises(ValueError, match="cancelled"):
        query(service)
    assert calls["revision"] == calls["worker"] == []


def test_missing_saved_checksum_never_dispatches(tmp_path, monkeypatch):
    service, _, _, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    del record["measurement"]["artifact_hashes"]["part.step"]
    with pytest.raises(ValueError, match="saved checksum"):
        query(service)
    assert calls["worker"] == []


def test_native_page_context_and_total_incidences_survive_transport(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def paged(command, scratch, timeout, cancelled, **kwargs):
        calls["scratch"].append(scratch)
        request = json.loads((scratch / "request.json").read_bytes())
        assert request["offset"] == 1 and request["limit"] == 1

        def change(result):
            result.update(
                neighbors=[{"entity": fact("edge", "2" * 64), "incidence_count": 1}],
                total_neighbors=3,
                total_incidences=4,
                next_offset=2,
            )

        write_result(scratch, request, mutate=change)

    monkeypatch.setattr(neighbors, "bounded_process", paged)
    result = query(service, offset=1, limit=1)
    assert result["next_offset"] == 2 and result["total_incidences"] == 4
    assert len(result["neighbors"]) == 1
    assert all(not scratch.exists() for scratch in calls["scratch"])
