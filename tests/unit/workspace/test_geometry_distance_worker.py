"""Staged distance worker controls using a fake native evaluator, not STEP imports."""

from __future__ import annotations

import hashlib
import json
import sys
from types import ModuleType

import pytest

from forge_mind.workspace import geometry_distance_worker as worker
from forge_mind.workspace.geometry_distance_contract import (
    SCOPE,
    DistanceContext,
    DistanceSelector,
    ResolvedDistanceSelector,
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
    context = DistanceContext(
        project_id="a" * 32,
        revision_id="b" * 32,
        a=ResolvedDistanceSelector(
            requested=DistanceSelector(body_index=0), body_index=0, body_signature="c" * 64
        ),
        b=ResolvedDistanceSelector(
            requested=DistanceSelector(body_index=1), body_index=1, body_signature="d" * 64
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
    native = ModuleType("forge_mind.workspace.geometry_distance_native")
    calls = []

    def evaluate(step_path, topology, a, b):
        calls.append((step_path, topology, a, b))
        assert step_path.read_bytes() == source["part.step"]
        assert topology == json.loads(source["topology.json"])
        assert a == context.a and b == context.b
        return native_result()

    native.measure_selected_distance = evaluate
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return request, source, context, native, calls


def native_result():
    return {
        "minimum_distance_mm": 2.0,
        "witness_a_mm": [0.0, 0.0, 0.0],
        "witness_b_mm": [2.0, 0.0, 0.0],
        "solution_count": 4,
        "native_completed": True,
        "native_inner_solution": False,
    }


def test_worker_binds_result_and_preserves_staged_sources(tmp_path, monkeypatch):
    request, source, context, _, calls = staged(tmp_path, monkeypatch)
    result = worker.measure_request(request)
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
        worker.measure_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


def test_worker_requires_own_current_engine_without_rejecting_old_saved_engine(
    tmp_path, monkeypatch
):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    monkeypatch.setattr(worker, "engine_fingerprint", lambda: "0" * 64)
    with pytest.raises(ValueError, match="worker engine differs"):
        worker.measure_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change", ["extra", "incomplete", "empty", "nan", "witness", "bool_count"])
def test_native_bad_result_never_becomes_success(tmp_path, monkeypatch, change):
    request, _, _, native, _ = staged(tmp_path, monkeypatch)

    def invalid(*args):
        value = native_result()
        if change == "extra":
            value["project_id"] = "0" * 32
        elif change == "incomplete":
            value["native_completed"] = False
        elif change == "empty":
            value["solution_count"] = 0
        elif change == "nan":
            value["minimum_distance_mm"] = float("nan")
        elif change == "witness":
            value["witness_b_mm"] = [3.0, 0.0, 0.0]
        else:
            value["solution_count"] = True
        return value

    native.measure_selected_distance = invalid
    with pytest.raises(ValueError):
        worker.measure_request(request)
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

    native.measure_selected_distance = changed
    with pytest.raises((ValueError, RuntimeError)):
        worker.measure_request(request)
    assert not (tmp_path / "result.json").exists()


def test_existing_result_refuses_without_overwrite_or_native(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    output = tmp_path / "result.json"
    output.write_bytes(b"prior result")
    with pytest.raises(ValueError, match="existing result"):
        worker.measure_request(request)
    assert output.read_bytes() == b"prior result" and calls == []


def test_duplicate_request_keys_refuse_before_native(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    raw = request.read_text(encoding="utf-8")
    request.write_text(raw[:-1] + ',"operation":"minimum_distance"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        worker.measure_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


def test_native_exception_propagates_without_fabricated_zero(tmp_path, monkeypatch):
    request, _, _, native, _ = staged(tmp_path, monkeypatch)

    def failed(*args):
        raise ValueError("Selected face signature is ambiguous")

    native.measure_selected_distance = failed
    with pytest.raises(ValueError, match="signature is ambiguous"):
        worker.measure_request(request)
    assert not (tmp_path / "result.json").exists()
