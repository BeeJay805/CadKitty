"""Staged neighbors worker controls using a fake native evaluator, not STEP imports."""

from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType

import pytest

from forge_mind.workspace import geometry_neighbors_worker as worker
from forge_mind.workspace.geometry_neighbors_contract import (
    SCOPE,
    NeighborContext,
    NeighborSelector,
    ResolvedNeighborSelector,
)

ENGINE, SAVED_ENGINE = "e" * 64, "f" * 64


def staged(tmp_path, monkeypatch):
    source = {
        "part.step": b"hash-bound fixture, not parsed",
        "topology.json": b'{"entities":[],"frame":"cad_z_up_mm"}',
        "document.json": b'{"name":"fixture"}',
    }
    for name, content in source.items():
        (tmp_path / name).write_bytes(content)
    context = NeighborContext(
        project_id="a" * 32,
        revision_id="b" * 32,
        selector=ResolvedNeighborSelector(
            requested=NeighborSelector(body_index=0, kind="face", signature="1" * 64),
            body_index=0,
            body_signature="c" * 64,
        ),
        source_hashes={
            name: hashlib.sha256(content).hexdigest() for name, content in source.items()
        },
        saved_engine_fingerprint=SAVED_ENGINE,
        query_engine_fingerprint=ENGINE,
    )
    request = tmp_path / "request.json"
    request.write_text(context.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(worker, "engine_fingerprint", lambda: ENGINE)
    native = ModuleType("forge_mind.workspace.geometry_neighbors_native")
    calls = []

    def evaluate(step_path, topology, selector, offset, limit):
        calls.append((step_path, topology, selector, offset, limit))
        assert step_path.read_bytes() == source["part.step"]
        assert topology == json.loads(source["topology.json"])
        assert selector == context.selector and offset == context.offset and limit == context.limit
        return native_result()

    native.inspect_selected_neighbors = evaluate
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return request, source, context, native, calls


def fact(kind, signature):
    return {
        "id": kind + "_" + signature[:20],
        "kind": kind,
        "signature": signature,
        "body_index": 0,
        "body_signature": "c" * 64,
        "center": [0.5, 0.5, 0.0],
        "min": [0.0, 0.0, 0.0],
        "max": [1.0, 1.0, 0.0],
        "measure": 1.0,
        "unit": "mm^2" if kind == "face" else "mm",
    }


def native_result():
    return {
        "focal": fact("face", "1" * 64),
        "relation": "face_boundary_edges",
        "neighbors": [{"entity": fact("edge", "2" * 64), "incidence_count": 2}],
        "total_neighbors": 1,
        "total_incidences": 2,
        "next_offset": None,
        "native_completed": True,
        "coverage": "complete_selected_neighborhood",
        "scope": SCOPE,
    }


def test_worker_binds_result_and_preserves_staged_sources(tmp_path, monkeypatch):
    request, source, context, _, calls = staged(tmp_path, monkeypatch)
    result = worker.inspect_request(request)
    assert result["saved_engine_fingerprint"] == SAVED_ENGINE
    assert result["query_engine_fingerprint"] == ENGINE
    assert result["source_hashes"] == context.source_hashes
    assert result["scope"] == SCOPE and result["native_completed"] is True
    assert len(calls) == 1
    assert json.loads((tmp_path / "result.json").read_bytes()) == result
    assert {name: (tmp_path / name).read_bytes() for name in source} == source
    assert {path.name for path in tmp_path.iterdir()} == set(source) | {
        "request.json",
        "result.json",
    }


@pytest.mark.parametrize("name", ["part.step", "topology.json", "document.json"])
def test_changed_staged_source_refuses_before_native(tmp_path, monkeypatch, name):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


def test_worker_requires_own_current_engine_without_rejecting_old_saved_engine(
    tmp_path, monkeypatch
):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "engine_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError, match="worker engine differs"):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "incomplete",
        "nan",
        "wrong_kind",
        "bool_count",
        "scope",
        "missing",
        "duplicate",
        "paging",
    ],
)
def test_native_bad_result_never_becomes_success(tmp_path, monkeypatch, change):
    request, _, _, native, _ = staged(tmp_path, monkeypatch)

    def invalid(*args):
        value = native_result()
        if change == "extra":
            value["project_id"] = "0" * 32
        elif change == "incomplete":
            value["native_completed"] = False
        elif change == "nan":
            value["focal"]["center"][0] = float("nan")
        elif change == "wrong_kind":
            value["neighbors"][0]["entity"] = fact("face", "2" * 64)
        elif change == "bool_count":
            value["neighbors"][0]["incidence_count"] = True
        elif change == "scope":
            value["scope"] = "Invented acceptance claim"
        elif change == "missing":
            del value["coverage"]
        elif change == "duplicate":
            value["neighbors"].append(value["neighbors"][0])
            value["total_neighbors"], value["total_incidences"] = 2, 4
        else:
            value["next_offset"] = 1
        return value

    native.inspect_selected_neighbors = invalid
    with pytest.raises(ValueError):
        worker.inspect_request(request)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change", ["source", "engine", "raise"])
def test_native_failure_or_midflight_changes_refuse_publication(tmp_path, monkeypatch, change):
    request, _, _, native, _ = staged(tmp_path, monkeypatch)

    def changed(*args):
        if change == "source":
            (tmp_path / "part.step").write_bytes(b"mutated during evaluation")
        elif change == "engine":
            monkeypatch.setattr(worker, "engine_fingerprint", lambda: "0" * 64)
        else:
            raise RuntimeError("Native evaluation failed")
        return native_result()

    native.inspect_selected_neighbors = changed
    with pytest.raises((ValueError, RuntimeError)):
        worker.inspect_request(request)
    assert not (tmp_path / "result.json").exists()


def test_existing_result_refuses_without_overwrite_or_native(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    output = tmp_path / "result.json"
    output.write_bytes(b"prior result")
    with pytest.raises(ValueError, match="existing result"):
        worker.inspect_request(request)
    assert output.read_bytes() == b"prior result" and calls == []


def test_duplicate_request_keys_refuse_before_native(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    raw = request.read_text(encoding="utf-8")
    request.write_text(raw[:-1] + ',"operation":"topology_neighbors"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


def test_native_exception_propagates_without_fabricated_zero(tmp_path, monkeypatch):
    request, _, _, native, _ = staged(tmp_path, monkeypatch)

    def failed(*args):
        raise ValueError("Selected entity signature is ambiguous")

    native.inspect_selected_neighbors = failed
    with pytest.raises(ValueError, match="signature is ambiguous"):
        worker.inspect_request(request)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("offset", [0, 4])
def test_genuine_zero_neighborhood_and_out_of_range_page(tmp_path, monkeypatch, offset):
    request, _, context, native, _ = staged(tmp_path, monkeypatch)
    updated = context.model_copy(update={"offset": offset})
    request.write_text(updated.model_dump_json(), encoding="utf-8")

    def empty(step_path, topology, selector, actual_offset, limit):
        assert actual_offset == offset
        value = native_result()
        value.update(neighbors=[], total_neighbors=0, total_incidences=0, next_offset=None)
        return value

    native.inspect_selected_neighbors = empty
    result = worker.inspect_request(request)
    assert result["native_completed"] is True and result["total_neighbors"] == 0
    assert result["neighbors"] == [] and result["next_offset"] is None


def test_result_byte_budget_refuses_after_valid_native_result(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "MAX_RESPONSE_BYTES", request.stat().st_size + 1)
    with pytest.raises(ValueError, match="output byte budget"):
        worker.inspect_request(request)
    assert len(calls) == 1 and not (tmp_path / "result.json").exists()


def test_main_forwards_one_request_path(tmp_path, monkeypatch):
    calls = []
    request = tmp_path / "request.json"
    monkeypatch.setattr(worker.sys, "argv", ["worker", str(request)])
    monkeypatch.setattr(worker, "inspect_request", calls.append)
    worker.main()
    assert calls == [request]


def test_main_refuses_extra_arguments_before_evaluation(monkeypatch):
    monkeypatch.setattr(worker.sys, "argv", ["worker", "request.json", "extra"])
    monkeypatch.setattr(worker, "inspect_request", lambda *args: pytest.fail("No evaluation"))
    with pytest.raises(ValueError, match="one staged"):
        worker.main()
