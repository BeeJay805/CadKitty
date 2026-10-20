"""Staged access worker controls with a fake native evaluator, not STEP imports."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from types import ModuleType

import pytest

from forge_mind.workspace import geometry_access_worker as worker
from forge_mind.workspace.geometry_access_contract import (
    SCOPE,
    AccessContext,
    AccessQuery,
    ResolvedAccessBody,
    evaluate_query,
)

ENGINE, SAVED_ENGINE = "e" * 64, "f" * 64


def native_result(context):
    return {
        "sweep": {
            "method": "static_cylinder",
            "volume_mm3": 40.0 * math.pi,
            "bbox_min_mm": [-2.0, -2.0, -5.0],
            "bbox_max_mm": [2.0, 2.0, 5.0],
        },
        "observations": [
            {
                "obstacle": context.obstacles[0].model_dump(mode="json"),
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
        ],
        "numerical_guards": {"distance_mm": 1e-6, "volume_mm3": 1e-6},
        "native_completed": True,
        "scope": SCOPE,
    }


def staged(tmp_path, monkeypatch):
    source = {
        "part.step": b"hash-bound fixture; never parsed",
        "topology.json": b'{"entities":[],"frame":"cad_z_up_mm"}',
        "document.json": b'{"name":"fixture"}',
    }
    for name, content in source.items():
        (tmp_path / name).write_bytes(content)
    query = AccessQuery.model_validate(
        {
            "cylinder": {
                "center_mm": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 2.0],
                "radial_reference": [3.0, 0.0, 0.0],
                "radius_mm": 2.0,
                "length_mm": 10.0,
            },
            "motion": {"mode": "static"},
            "obstacles": [{"body_index": 0}],
        }
    )
    cylinder, motion = evaluate_query(query)
    context = AccessContext(
        project_id="a" * 32,
        revision_id="b" * 32,
        query=query,
        evaluated_cylinder=cylinder,
        evaluated_motion=motion,
        obstacles=[
            ResolvedAccessBody(requested=query.obstacles[0], body_index=0, body_signature="c" * 64)
        ],
        inventory_body_count=2,
        omitted_body_indices=[1],
        source_hashes={
            name: hashlib.sha256(content).hexdigest() for name, content in source.items()
        },
        saved_engine_fingerprint=SAVED_ENGINE,
        query_engine_fingerprint=ENGINE,
    )
    request = tmp_path / "request.json"
    request.write_text(context.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(worker, "engine_fingerprint", lambda: ENGINE)
    native = ModuleType("forge_mind.workspace.geometry_access_native")
    calls = []

    def evaluate(
        step_path, topology, evaluated_cylinder, evaluated_motion, obstacles, inventory_body_count
    ):
        calls.append((step_path, topology, obstacles))
        assert step_path.read_bytes() == source["part.step"]
        assert topology == json.loads(source["topology.json"])
        assert evaluated_cylinder == context.evaluated_cylinder
        assert evaluated_motion == context.evaluated_motion
        assert obstacles == context.obstacles and inventory_body_count == 2
        return native_result(context)

    native.inspect_selected_cylindrical_access = evaluate
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return request, source, context, native, calls


def test_worker_binds_complete_result_and_preserves_staged_bytes(tmp_path, monkeypatch):
    request, source, context, _, calls = staged(tmp_path, monkeypatch)
    result = worker.inspect_request(request)
    assert result["source_hashes"] == context.source_hashes
    assert result["saved_engine_fingerprint"] == SAVED_ENGINE
    assert result["query_engine_fingerprint"] == ENGINE
    assert result["omitted_body_indices"] == [1]
    assert result["scope"] == SCOPE and result["native_completed"] is True
    assert len(calls) == 1
    assert json.loads((tmp_path / "result.json").read_bytes()) == result
    assert {name: (tmp_path / name).read_bytes() for name in source} == source
    assert {p.name for p in tmp_path.iterdir()} == set(source) | {"request.json", "result.json"}


@pytest.mark.parametrize("name", ["part.step", "topology.json", "document.json"])
def test_staged_tamper_refuses_before_native(tmp_path, monkeypatch, name):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    (tmp_path / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


def test_current_worker_engine_required_but_old_saved_engine_allowed(tmp_path, monkeypatch):
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
        "missing_scope",
        "empty",
        "nan",
        "witness",
        "bool_count",
        "classification",
        "wrong_body",
        "components",
    ],
)
def test_bad_native_output_never_becomes_success(tmp_path, monkeypatch, change):
    request, _, context, native, _ = staged(tmp_path, monkeypatch)

    def invalid(*args):
        value = native_result(context)
        row = value["observations"][0]
        if change == "extra":
            value["project_id"] = "0" * 32
        elif change == "incomplete":
            value["native_completed"] = False
        elif change == "missing_scope":
            value.pop("scope")
        elif change == "empty":
            value["observations"] = []
        elif change == "nan":
            row["minimum_distance_mm"] = float("nan")
        elif change == "witness":
            row["witness_obstacle_mm"] = [5.0, 0.0, 0.0]
        elif change == "bool_count":
            row["solution_count"] = True
        elif change == "classification":
            row["relation"] = "positive_volume_interference"
        elif change == "wrong_body":
            row["obstacle"]["body_signature"] = "d" * 64
        else:
            row["overlap_components"] = [
                {"volume_mm3": 1.0, "bbox_min_mm": [0.0, 0.0, 0.0], "bbox_max_mm": [1.0, 1.0, 1.0]}
            ] * 33
        return value

    native.inspect_selected_cylindrical_access = invalid
    with pytest.raises(ValueError):
        worker.inspect_request(request)
    assert not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("change", ["source", "engine", "raise"])
def test_native_failure_or_midflight_changes_refuse_publication(tmp_path, monkeypatch, change):
    request, _, context, native, _ = staged(tmp_path, monkeypatch)

    def changed(*args):
        if change == "source":
            (tmp_path / "part.step").write_bytes(b"changed during evaluation")
        elif change == "engine":
            monkeypatch.setattr(worker, "engine_fingerprint", lambda: "0" * 64)
        else:
            raise RuntimeError("Native evaluation failed")
        return native_result(context)

    native.inspect_selected_cylindrical_access = changed
    with pytest.raises((ValueError, RuntimeError)):
        worker.inspect_request(request)
    assert not (tmp_path / "result.json").exists()


def test_existing_result_never_overwritten_or_evaluated(tmp_path, monkeypatch):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    output = tmp_path / "result.json"
    output.write_bytes(b"prior result")
    with pytest.raises(ValueError, match="existing result"):
        worker.inspect_request(request)
    assert output.read_bytes() == b"prior result" and calls == []


@pytest.mark.parametrize("change", ["duplicate", "motion", "omission"])
def test_invalid_staged_context_refuses_before_native(tmp_path, monkeypatch, change):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    raw = request.read_text(encoding="utf-8")
    if change == "duplicate":
        raw = raw[:-1] + ',"operation":"cylindrical_access"}'
    else:
        value = json.loads(raw)
        if change == "motion":
            value["evaluated_motion"]["translation_mm"] = [1.0, 0.0, 0.0]
        else:
            value["omitted_body_indices"] = []
        raw = json.dumps(value)
    request.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()


@pytest.mark.parametrize("budget", ["source", "request"])
def test_input_budget_refuses_before_native(tmp_path, monkeypatch, budget):
    request, _, _, _, calls = staged(tmp_path, monkeypatch)
    monkeypatch.setattr(
        worker, "MAX_TOTAL_INPUT_BYTES" if budget == "source" else "MAX_REQUEST_BYTES", 1
    )
    with pytest.raises(ValueError):
        worker.inspect_request(request)
    assert calls == [] and not (tmp_path / "result.json").exists()
