"""Service lifecycle controls with a fake worker; these do not qualify native CAD."""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest

from forge_mind.workspace import geometry_access as access
from forge_mind.workspace.geometry_access_contract import SCOPE, AccessQuery

SID, RID = "a" * 32, "b" * 32
SAVED_ENGINE, QUERY_ENGINE = "f" * 64, "e" * 64


def query_value():
    return {
        "cylinder": {
            "center_mm": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 2.0],
            "radial_reference": [3.0, 0.0, 0.0],
            "radius_mm": 2.0,
            "length_mm": 10.0,
        },
        "motion": {"mode": "static"},
        "obstacles": [{"occurrence_path": "second"}],
    }


def write_result(scratch, request, *, mutate=None):
    result = {
        **request,
        "sweep": {
            "method": "static_cylinder",
            "volume_mm3": 40.0 * math.pi,
            "bbox_min_mm": [-2.0, -2.0, -5.0],
            "bbox_max_mm": [2.0, 2.0, 5.0],
        },
        "observations": [
            {
                "obstacle": item,
                "overlap_components": [],
                "overlap_volume_mm3": 0.0,
                "minimum_distance_mm": 2.0,
                "witness_sweep_mm": [2.0, 0.0, 0.0],
                "witness_obstacle_mm": [4.0, 0.0, 0.0],
                "solution_count": 1,
                "native_inner_solution": False,
                "native_completed": True,
                "relation": "separated",
            }
            for item in request["obstacles"]
        ],
        "numerical_guards": {"distance_mm": 1e-6, "volume_mm3": 1e-6},
        "native_completed": True,
        "scope": SCOPE,
    }
    if mutate:
        mutate(result)
    (scratch / "result.json").write_text(json.dumps(result), encoding="utf-8")


def fixture_service(tmp_path, monkeypatch, *, mode="trusted_local", qualified=False):
    directory = tmp_path / "saved"
    directory.mkdir()
    topology = {
        "frame": "cad_z_up_mm",
        "entities": [
            {
                "kind": "body",
                "body_index": i,
                "signature": str(i + 1) * 64,
                "body_signature": str(i + 1) * 64,
            }
            for i in range(3)
        ],
    }
    contents = {
        "part.step": b"Immutable source fixture; never parsed by mocked worker",
        "document.json": b'{"name":"source fixture"}',
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
                        {"path": name, "body_index": i}
                        for i, name in enumerate(["first", "second", "third"])
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
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()))

    monkeypatch.setattr(access, "bounded_process", worker)
    monkeypatch.setattr(access, "engine_fingerprint", lambda: QUERY_ENGINE)
    return service, directory, contents, record, calls, worker


def inspect(service, **kwargs):
    return access.inspect_cylindrical_access(
        service, SID, "owner", RID, AccessQuery.model_validate(query_value()), **kwargs
    )


def test_owned_access_preserves_source_and_reports_complete_omissions(tmp_path, monkeypatch):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    result = inspect(service)
    assert result["obstacles"][0]["body_index"] == 1
    assert result["obstacles"][0]["requested"]["occurrence_path"] == "second"
    assert result["inventory_body_count"] == 3 and result["omitted_body_indices"] == [0, 2]
    assert result["evaluated_motion"]["translation_mm"] == [0.0, 0.0, 0.0]
    assert result["saved_engine_fingerprint"] == SAVED_ENGINE
    assert result["query_engine_fingerprint"] == QUERY_ENGINE
    assert result["observations"][0]["relation"] == "separated"
    assert record == before
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == contents
    assert calls["owner"] == [(SID, "owner"), (SID, "owner")]
    assert len(calls["worker"]) == 1 and calls["worker"][0][2] == {"cpu_limit": 2}
    assert all(not p.exists() for p in calls["scratch"])


def test_foreign_owner_refuses_before_source_or_worker(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    monkeypatch.setattr(access, "_artifact", lambda *args: pytest.fail("No source read"))
    with pytest.raises(ValueError, match="Foreign owner"):
        access.inspect_cylindrical_access(
            service, SID, "foreign", RID, AccessQuery.model_validate(query_value())
        )
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("timeout", [True, 0, 61, float("nan"), float("inf"), "30"])
def test_invalid_deadline_never_reads_saved_data(tmp_path, monkeypatch, timeout):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        inspect(service, timeout=timeout)
    assert calls["owner"] == calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("mode", ["docker", "unsupported"])
def test_unqualified_execution_never_reads_sources(tmp_path, monkeypatch, mode):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch, mode=mode)
    monkeypatch.setattr(access, "_artifact", lambda *args: pytest.fail("No source read"))
    with pytest.raises(ValueError, match=r"qualified|Unsupported"):
        inspect(service)
    assert calls["worker"] == []


def test_qualified_container_uses_exact_image_and_staged_context(tmp_path, monkeypatch):
    service, _, _, _, calls, worker = fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )
    monkeypatch.setattr(access.shutil, "which", lambda name: "fixture-docker")

    def container(docker, scratch, module, request, timeout, cancelled, *, image):
        assert (
            docker == "fixture-docker" and module == "forge_mind.workspace.geometry_access_worker"
        )
        assert request == "request.json" and image == "qualified@sha256:fixture"
        worker([module], scratch, timeout, cancelled)

    service.executor._run_container = container
    monkeypatch.setattr(
        access, "bounded_process", lambda *args, **kwargs: pytest.fail("No host child")
    )
    assert inspect(service)["query_engine_fingerprint"] == QUERY_ENGINE
    assert len(calls["worker"]) == 1


@pytest.mark.parametrize(
    "change",
    [
        "mapping",
        "alias",
        "provenance",
        "duplicate",
        "gap",
        "bool",
        "frame",
        "missing",
        "request_alias",
    ],
)
def test_incomplete_inventory_or_wrong_selection_refuses_before_worker(
    tmp_path, monkeypatch, change
):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    topology = json.loads(contents["topology.json"])
    rows = record["measurement"]["facts"]["assembly"]["occurrences"]
    value = query_value()
    if change == "mapping":
        rows.append(deepcopy(rows[1]))
    elif change == "alias":
        rows[0]["body_index"] = 1
    elif change == "provenance":
        record["measurement"]["facts"]["assembly"]["step_sha256"] = "0" * 64
    elif change == "duplicate":
        topology["entities"].append(deepcopy(topology["entities"][0]))
    elif change == "gap":
        topology["entities"][2]["body_index"] = 4
    elif change == "bool":
        topology["entities"][0]["body_index"] = False
    elif change == "frame":
        topology["frame"] = "definition_local"
    elif change == "missing":
        value["obstacles"] = [{"body_index": 9}]
    else:
        value["obstacles"].append({"body_index": 1})
    content = json.dumps(topology).encode()
    (directory / "topology.json").write_bytes(content)
    record["measurement"]["artifact_hashes"]["topology.json"] = hashlib.sha256(content).hexdigest()
    with pytest.raises(ValueError):
        access.inspect_cylindrical_access(
            service, SID, "owner", RID, AccessQuery.model_validate(value)
        )
    assert calls["worker"] == []


@pytest.mark.parametrize("name", ["part.step", "topology.json", "document.json"])
def test_changed_saved_bytes_refuse_before_worker(tmp_path, monkeypatch, name):
    service, directory, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    (directory / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="source changed"):
        inspect(service)
    assert calls["worker"] == []


@pytest.mark.parametrize("change", ["source", "record", "owner", "cancel", "deadline", "engine"])
def test_midflight_change_refuses_and_cleans_scratch(tmp_path, monkeypatch, change):
    service, directory, _, record, calls, worker = fixture_service(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(access.time, "monotonic", lambda: clock[0])

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
            monkeypatch.setattr(access, "engine_fingerprint", lambda: "0" * 64)

    monkeypatch.setattr(access, "bounded_process", changed)
    with pytest.raises(ValueError):
        inspect(service)
    assert all(not p.exists() for p in calls["scratch"])


@pytest.mark.parametrize(
    "change",
    [
        "project",
        "hash",
        "selection",
        "omissions",
        "frame",
        "classification",
        "incomplete",
        "observations",
        "extra",
        "oversize",
    ],
)
def test_mismatched_worker_result_refuses_and_cleans_scratch(tmp_path, monkeypatch, change):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def corrupt(result):
        if change == "project":
            result["project_id"] = "0" * 32
        elif change == "hash":
            result["source_hashes"]["part.step"] = "0" * 64
        elif change == "selection":
            result["obstacles"][0]["body_signature"] = "0" * 64
            result["observations"][0]["obstacle"]["body_signature"] = "0" * 64
        elif change == "omissions":
            result["omitted_body_indices"] = []
        elif change == "frame":
            result["evaluated_motion"]["translation_mm"] = [1.0, 0.0, 0.0]
        elif change == "classification":
            result["observations"][0]["relation"] = "positive_volume_interference"
        elif change == "incomplete":
            result["native_completed"] = False
        elif change == "observations":
            result["observations"] = []
        elif change == "extra":
            result["unexpected"] = True
        else:
            result["scope"] = "x" * (access.MAX_RESPONSE_BYTES + 1)

    def bad(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()), mutate=corrupt)

    monkeypatch.setattr(access, "bounded_process", bad)
    with pytest.raises(ValueError):
        inspect(service)
    assert all(not p.exists() for p in calls["scratch"])


def test_worker_failure_never_mutates_saved_records_or_sources(tmp_path, monkeypatch):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)

    def failed(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        raise RuntimeError("Native process failed")

    monkeypatch.setattr(access, "bounded_process", failed)
    with pytest.raises(RuntimeError, match="Native process failed"):
        inspect(service)
    assert record == before
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == contents
    assert all(not p.exists() for p in calls["scratch"])


@pytest.mark.parametrize("budget", ["source", "request"])
def test_aggregate_or_request_budget_refuses_without_worker(tmp_path, monkeypatch, budget):
    service, directory, contents, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    monkeypatch.setattr(
        access, "MAX_TOTAL_INPUT_BYTES" if budget == "source" else "MAX_REQUEST_BYTES", 1
    )
    with pytest.raises(ValueError, match="byte limit"):
        inspect(service)
    assert calls["worker"] == calls["scratch"] == []
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == contents
