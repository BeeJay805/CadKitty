"""Motion identities, bounds and adversarial collision coverage."""

import json
from pathlib import Path

import build123d as b
import numpy as np
import pytest

from forge_mind.workspace.motion_check import MotionCheckSettings, configurations, review_poses
from forge_mind.workspace.motion_rig import MotionRig, group_matrices, resolve_rig


def rig_data():
    return {
        "drivers": [
            {
                "id": "travel",
                "name": "Travel",
                "unit": "mm",
                "minimum": 0,
                "maximum": 20,
                "reference": 10,
            },
            {
                "id": "turn",
                "name": "Turn",
                "unit": "deg",
                "minimum": -90,
                "maximum": 180,
                "reference": 30,
            },
        ],
        "groups": [
            {
                "id": "carrier",
                "name": "Carrier",
                "driver": "travel",
                "kind": "slide",
                "axis": [1, 0, 0],
                "occurrences": ["carrier"],
            },
            {
                "id": "wheel",
                "name": "Wheel",
                "driver": "turn",
                "parent": "carrier",
                "kind": "rotate",
                "axis": [0, 2, 0],
                "pivot": [10, 0, 5],
                "ratio": 2,
                "occurrences": ["wheel"],
            },
        ],
    }


def test_reference_identity_and_parented_off_axis_rotation():
    rig = resolve_rig(MotionRig.model_validate(rig_data()), {})
    for m in group_matrices(rig, {}).values():
        assert np.allclose(m, np.eye(4))
    moved = group_matrices(rig, {"travel": 15, "turn": 75})
    # A point 2mm above the Y pivot rotates +90 to X; parent carries pivot +5 X.
    assert np.allclose(moved["wheel"] @ [10, 0, 7, 1], [17, 0, 5, 1])
    assert np.allclose(moved["carrier"] @ [10, 0, 7, 1], [15, 0, 7, 1])
    for values in ({"turn": 181}, {"travel": float("nan")}, {"missing": 1}):
        with pytest.raises(ValueError):
            group_matrices(rig, values)


@pytest.mark.parametrize(
    "case",
    [
        "cycle",
        "missing_parent",
        "missing_ancestor",
        "duplicate",
        "zero_axis",
        "huge_axis",
        "nan",
        "driver",
        "range",
        "reference",
        "variable",
        "empty",
        "displacement",
    ],
)
def test_invalid_rig_is_rejected(case):
    data = rig_data()
    if case == "cycle":
        data["groups"][0]["parent"] = "wheel"
    if case == "missing_parent":
        data["groups"][0]["parent"] = "absent"
    if case == "missing_ancestor":
        data["groups"][0]["parent"] = "wheel"
        data["groups"][1]["parent"] = "absent"
    if case == "duplicate":
        data["groups"][1]["occurrences"] = ["carrier"]
    if case == "zero_axis":
        data["groups"][0]["axis"] = [0, 0, 0]
    if case == "huge_axis":
        data["groups"][0]["axis"] = [1.79e308, 1.79e308, 1.79e308]
    if case == "nan":
        data["groups"][0]["ratio"] = float("nan")
    if case == "driver":
        data["groups"][0]["driver"] = "absent"
    if case == "range":
        data["drivers"][0]["maximum"] = 0
    if case == "reference":
        data["drivers"][0]["reference"] = -1
    if case == "variable":
        data["groups"][0]["pivot"] = ["missing", 0, 0]
    if case == "empty":
        for group in data["groups"]:
            group["occurrences"] = []
    if case == "displacement":
        data["groups"][0]["ratio"] = float("inf")
    with pytest.raises(ValueError):
        resolve_rig(MotionRig.model_validate(data), {})


def test_spatial_filter_and_rigid_reuse_preserve_collision_witnesses():
    solids = [b.Box(2, 2, 2), b.Pos(6, 0, 0) * b.Box(2, 2, 2), b.Pos(40, 0, 0) * b.Box(2, 2, 2)]
    poses = []
    for x in (0, 6, 12):
        move = np.eye(4)
        move[0, 3] = x
        poses.append(({"x": x}, [move, np.eye(4), np.eye(4)]))
    fast = review_poses(solids, ["moving", "obstacle", "fixed"], ["move", None, None], poses)
    exact = review_poses(
        solids, ["moving", "obstacle", "fixed"], ["move", None, None], poses, optimized=False
    )
    assert fast["collisions"] == exact["collisions"]
    assert fast["collisions"][0]["values"] == {"x": 6}
    assert fast["exact_intersections"] < exact["exact_intersections"]
    assert fast["disjoint_bounds"] == 8
    assert fast["completed_pairs"] == fast["total_pairs"] == 9
    stopped = review_poses(solids, ["a", "b", "c"], ["move", None, None], poses, seconds=0)
    assert stopped["coverage"] == "incomplete"
    assert stopped["status"] == "unverified"
    assert stopped["completed_configurations"] == []


def test_nonuniform_sampling_keeps_coupled_screw_phases_and_corners():
    data = rig_data()
    rig = resolve_rig(MotionRig.model_validate(data), {})
    poses = configurations(rig, MotionCheckSettings())
    assert {"travel": 0, "turn": 180} in poses
    assert any(abs((v["travel"] * 288) % 360) > 1 for v in poses)
    assert len(poses) <= 73


def test_idler_motion_has_complete_named_groups_and_stationary_supports():
    from scripts.idler_motion_bench import motion_document

    from forge_mind.workspace.assembly import expand
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.motion_rig import bind_rig

    raw = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "docs/benchmarks/engineering-idler/source-revision.json"
        ).read_text(encoding="utf-8")
    )
    document = motion_document(DesignDocument.model_validate(raw["document"]))
    rows = [{"path": r["path"], "body_index": i} for i, r in enumerate(expand(document))]
    motion = bind_rig(document, {"occurrences": rows})
    assert len(motion["bindings"]) == 29
    assert next(r for r in motion["bindings"] if r["path"] == "base_fixed")["group"] is None
    assert next(r for r in motion["bindings"] if r["path"] == "moving/idler")["group"] == "idler"
    m = group_matrices(motion["rig"], {"adjustment": 15, "roller": 90})
    assert np.allclose(m["idler"] @ [0, 0, 45, 1], [15, 0, 45, 1])
    broken = document.model_dump()
    broken["motion_rig"]["groups"][0]["occurrences"] = ["missing"]
    with pytest.raises(ValueError, match="missing leaf"):
        DesignDocument.model_validate(broken)


def test_shared_matrix_fixture_matches_backend():
    raw = json.loads(
        (Path(__file__).resolve().parents[2] / "fixtures/motion-rig-parity.json").read_text(
            encoding="utf-8"
        )
    )
    for sample in raw["samples"]:
        actual = group_matrices(raw["rig"], sample["values"])
        for group, matrix in sample["matrices"].items():
            assert np.allclose(actual[group], matrix, atol=1e-9)


def test_check_cache_selects_exact_receipt_retries_incomplete_and_rejects_tampering(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from forge_mind.workspace import motion_service
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.motion_service import check_path, run_motion_job
    from forge_mind.workspace.store import Store, WorkspaceError

    store = Store(tmp_path)
    sid = store.create("owner", "motion", {})["id"]
    motion = {
        "engine_fingerprint": engine_fingerprint(),
        "revision_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }
    service = SimpleNamespace(
        store=store,
        executor=SimpleNamespace(mode="trusted_local"),
        _load_revision=lambda *_: {
            "id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "measurement": {"geometry_valid": True, "facts": {"motion_rig": motion}},
        },
    )
    calls = []
    coverage = "discrete_only"

    def worker(*args):
        calls.append(args)
        return {"status": "unverified", "coverage": coverage}

    monkeypatch.setattr(motion_service, "process_revision", worker)
    job = {
        "session": sid,
        "owner": "owner",
        "revision_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "settings": {"samples_per_driver": 3},
    }
    first = run_motion_job(service, job, lambda: False)
    second = run_motion_job(service, {**job, "settings": {"samples_per_driver": 7}}, lambda: False)
    assert first != second
    assert run_motion_job(service, job, lambda: False) == first
    assert len(calls) == 2
    coverage = "incomplete"
    retry = {**job, "settings": {"samples_per_driver": 13}}
    assert run_motion_job(service, retry, lambda: False) != run_motion_job(
        service, retry, lambda: False
    )
    assert len(calls) == 4
    with pytest.raises(WorkspaceError):
        check_path(service, sid, "stranger", first)
    check_path(service, sid, "owner", first).write_text("tampered", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="changed"):
        run_motion_job(service, job, lambda: False)


def test_motion_worker_cancellation_kills_process(tmp_path):
    import sys
    import time

    from forge_mind.workspace.processing import bounded_process

    start = time.monotonic()
    with pytest.raises(ValueError, match="cancelled"):
        bounded_process(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            10,
            lambda: time.monotonic() - start > 0.15,
        )
    assert time.monotonic() - start < 4
