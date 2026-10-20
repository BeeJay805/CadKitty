from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from forge_mind.workspace import photo_profile_service as profiles
from forge_mind.workspace.photo_profile import analyze_profile
from forge_mind.workspace.photo_profile_api import PhotoProfileAction, execute_photo_profile
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def case(tmp_path, monkeypatch):
    image = Image.new("RGB", (64, 48), "white")
    ImageDraw.Draw(image).rectangle((10, 10, 50, 35), fill="black")
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "a" * 32,
            "prompt": "Copy this outline",
            "reference": {
                "filename": "outline.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(stream.getvalue()).decode(),
            },
        },
    )
    service = SimpleNamespace(
        store=store, executor=SimpleNamespace(mode="trusted_local"), stop=threading.Event()
    )
    calls = []

    def pure_worker(directory, payload, timeout, cancelled):
        assert 0 < timeout <= 30 and not cancelled()
        calls.append(payload)
        if payload["operation"] == "photo_profile_analyze":
            return analyze_profile((directory / "vision.png").read_bytes(), payload["request"])
        assert payload["operation"] == "photo_profile_compare"
        result = {
            "step_sha256": hashlib.sha256((directory / "part.step").read_bytes()).hexdigest(),
            "observation_sha256": payload["observation"]["observation_sha256"],
            "vision_image_sha256": payload["observation"]["vision_image_sha256"],
            "engine_fingerprint": "e" * 64,
            "selector": payload["selector"],
            "scope": "test fake native observer",
        }
        result["comparison_sha256"] = profiles._digest(result)
        return result

    monkeypatch.setattr(profiles, "process_revision", pure_worker)
    request = {
        "vision_image_sha256": session["customer_reference"]["vision_sha256"],
        "foreground": "dark",
        "threshold": 128,
        "fit_tolerance_mm": 0.1,
        "assumptions": {"source_origin": "model-proposed mask, not owner measurement"},
        "seed_pixel": [20, 20],
    }
    return SimpleNamespace(service=service, sid=session["id"], request=request, calls=calls)


def test_shared_action_replay_and_read_leave_owner_facts_plan_and_allowance_unchanged(case):
    before = case.service.store.session_record(case.sid, "owner")
    body = PhotoProfileAction(action="analyze", request=case.request)
    result = execute_photo_profile(case.service, case.sid, "owner", body)
    assert result["result"]["status"] == "advisory"
    assert result["result"]["components"] and result["result"]["proposal_document"] is None
    assert result["source"]["normalized_width"] == 64
    assert result["source"]["coordinate_space"] == "retained_normalized_vision_pixels"
    assert execute_photo_profile(case.service, case.sid, "owner", body) == result
    assert (
        execute_photo_profile(
            case.service, case.sid, "owner", {"action": "read", "observation_id": result["id"]}
        )
        == result
    )
    assert len(case.calls) == 1
    assert case.service.store.session_record(case.sid, "owner") == before
    assert "pass" not in result["result"]


def test_owner_tamper_source_and_request_bindings_fail_closed(case):
    with pytest.raises(WorkspaceError, match="not found"):
        profiles.analyze_reference(case.service, case.sid, "other", case.request)
    with pytest.raises(WorkspaceError, match="different image"):
        profiles.analyze_reference(
            case.service, case.sid, "owner", {**case.request, "vision_image_sha256": "0" * 64}
        )
    record = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    with case.service.store.connect() as con:
        changed = {**record, "scope": "forged acceptance"}
        con.execute(
            "UPDATE photo_profile_observations SET document=? WHERE id=?",
            (json.dumps(changed), record["id"]),
        )
    with pytest.raises(WorkspaceError, match="stale or changed"):
        profiles.read_profile(case.service, case.sid, "owner", record["id"])


def test_changed_vision_refuses_saved_observation(case):
    record = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    directory = (
        case.service.store.root
        / "sessions"
        / case.sid
        / "source-artifacts"
        / record["source"]["source_artifact_id"]
    )
    (directory / "vision.png").write_bytes(b"changed")
    with pytest.raises(WorkspaceError, match="changed"):
        profiles.read_profile(case.service, case.sid, "owner", record["id"])


def test_cancel_after_worker_publishes_no_observation(case, monkeypatch):
    original = profiles.process_revision

    def cancel_after(*args):
        result = original(*args)
        case.service.stop.set()
        return result

    monkeypatch.setattr(profiles, "process_revision", cancel_after)
    with pytest.raises(WorkspaceError, match="cancelled"):
        profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    with case.service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM photo_profile_observations").fetchone()[0] == 0


def test_compare_stages_exact_registered_step_and_does_not_change_selection(case):
    record = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    rid = "b" * 32
    directory = case.service.store.directory(case.sid, rid)
    directory.mkdir(parents=True)
    source = directory / "part.step"
    source.write_bytes(b"fake STEP for staging-only test")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    case.service.store.add_revision(
        case.sid,
        {
            "id": rid,
            "measurement": {
                "artifact_hashes": {"part.step": sha},
                "engine_fingerprint": "test revision engine",
            },
        },
    )
    before = case.service.store.session_record(case.sid, "owner")
    result = execute_photo_profile(
        case.service,
        case.sid,
        "owner",
        {
            "action": "compare",
            "observation_id": record["id"],
            "revision_id": rid,
            "selector": {"body_index": 0, "z_mm": 0},
        },
    )
    assert result["step_sha256"] == sha and result["source"] == record["source"]
    assert result["observation_digest"] == profiles._digest(record)
    assert result["engine_fingerprint"]
    assert case.service.store.session_record(case.sid, "owner") == before
    source.write_bytes(b"changed")
    with pytest.raises(WorkspaceError, match="stale"):
        execute_photo_profile(
            case.service,
            case.sid,
            "owner",
            {
                "action": "compare",
                "observation_id": record["id"],
                "revision_id": rid,
                "selector": {"body_index": 0, "z_mm": 0},
            },
        )


def test_http_uses_same_owned_contract(case):
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient

    from forge_mind.workspace.api import router

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(router(case.service.store, case.service, owner))
    client = TestClient(app)
    body = {"action": "analyze", "request": case.request}
    url = f"/workspace/sessions/{case.sid}/photo-profile"
    assert client.post(url, json=body, headers={"Authorization": "other"}).status_code == 409
    result = client.post(url, json=body, headers={"Authorization": "owner"})
    assert result.status_code == 200, result.text
    assert (
        profiles.read_profile(case.service, case.sid, "owner", result.json()["id"]) == result.json()
    )
    assert len(case.calls) == 1


def test_concurrent_identical_publication_has_one_immutable_record(case, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    original = profiles.process_revision
    barrier = threading.Barrier(2, timeout=5)

    def concurrent_worker(*args):
        result = original(*args)
        barrier.wait()
        return result

    monkeypatch.setattr(profiles, "process_revision", concurrent_worker)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = [
            workers.submit(
                profiles.analyze_reference,
                case.service,
                case.sid,
                "owner",
                case.request,
                timeout=10,
            )
            for _ in range(2)
        ]
        records = [future.result(timeout=15) for future in pending]
    assert records[0] == records[1]
    with case.service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM photo_profile_observations").fetchone()[0] == 1


def test_mcp_discovery_and_authenticated_tool_use_shared_service(case, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.unit.workspace.test_customer_mcp import ISSUER, RESOURCE, Verifier
    from tests.unit.workspace.test_remote_mcp import call, result

    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.customer_mcp import discovery
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.remote_mcp import create_app

    executor = Executor("docker")
    # Contract-only fake observer. No real container qualification is claimed.
    executor._typed_container_verified = True
    executor._customer_container_verified = True
    monkeypatch.setattr(executor, "process_observation", profiles.process_revision)
    service = WorkspaceService(case.service.store, executor)
    info = discovery()
    assert info["tools"]["photo_profile"] == "inspect_customer_photo_profile"
    assert info["photo_profile"]["schema"]["properties"]["action"]["enum"] == [
        "analyze",
        "read",
        "compare",
    ]
    app = create_app(service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE)
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        listed = result(
            client.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer owner",
                    "Accept": "application/json, text/event-stream",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
            )
        )
        assert "inspect_customer_photo_profile" in {tool["name"] for tool in listed["tools"]}
        body = {"project_id": case.sid, "request": {"action": "analyze", "request": case.request}}
        response = call(client, "owner", "inspect_customer_photo_profile", body)
        observed = result(response)["structuredContent"]
        assert observed["source"]["vision_sha256"] == case.request["vision_image_sha256"]
        full = profiles.read_profile(service, case.sid, "owner", observed["id"])
        assert observed["projection"]["full_record_sha256"] == profiles._digest(full)
        assert observed["result"]["components"][0]["loops"][0]["raw_point_counts"]["points_px"] > 0
        wide = result(
            call(
                client,
                "owner",
                "inspect_customer_photo_profile",
                {
                    "project_id": case.sid,
                    "request": {"action": "read", "observation_id": observed["id"]},
                },
            )
        )["structuredContent"]
        assert wide == full


def test_compact_profile_context_preserves_topology_proposal_and_full_read(case):
    from copy import deepcopy

    from forge_mind.workspace.photo_profile_context import photo_profile_context

    full = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    original = deepcopy(full)
    # Simulate a dense observation with exact persisted coordinates and topology.
    dense = deepcopy(full)
    loops = dense["result"]["components"][0]["loops"]
    loops[0]["points_px"] = [[i / 10, i / 20] for i in range(20000)]
    loops[0]["points_mm"] = [[i / 100, i / 200] for i in range(20000)]
    loops[0]["fit"] = {"kind": "polyline", "points_mm": [[1, 2], [3, 4], [5, 6], [1, 2]]}
    loops[0]["pixel_outline"] = {
        "points_px": [[10, 10], [50, 10], [50, 35], [10, 35], [10, 10]],
        "display_tolerance_px": 0.5,
        "scope": "Display only",
    }
    dense["result"]["proposal_document"] = {
        "features": [{"id": "profile", "points": [[1, 2], [3, 4]]}]
    }
    before = deepcopy(dense)
    summary = photo_profile_context(dense, case.sid)
    projected = summary["result"]["components"][0]["loops"][0]
    assert "points_px" not in projected and "points_mm" not in projected
    assert projected["raw_point_counts"] == {"points_px": 20000, "points_mm": 20000}
    assert projected["fit"] == loops[0]["fit"]
    assert projected["pixel_outline"] == loops[0]["pixel_outline"]
    assert summary["result"]["proposal_document"] == dense["result"]["proposal_document"]
    assert [c["id"] for c in summary["result"]["components"]] == [
        c["id"] for c in dense["result"]["components"]
    ]
    assert summary["projection"]["full_record_sha256"] == profiles._digest(dense)
    assert summary["projection"]["full_result_sha256"] == profiles._digest(dense["result"])
    assert len(json.dumps(summary)) < 10000 < len(json.dumps(dense)) / 10
    assert dense == before
    assert profiles.read_profile(case.service, case.sid, "owner", full["id"]) == original


def test_compact_comparison_keeps_bounds_repairs_and_all_loop_point_counts():
    from copy import deepcopy

    from forge_mind.workspace.photo_profile_context import photo_profile_context

    full = {
        "observation_id": "a" * 32,
        "revision_id": "b" * 32,
        "selector": {"body_index": 0, "z_mm": 5},
        "source": {"vision_sha256": "c" * 64},
        "result": {
            "comparison_sha256": "d" * 64,
            "observation_sha256": "e" * 64,
            "symmetric_residual_interval_mm": [0.1, 0.11],
            "inner_loop_count_observed": 2,
            "inner_loop_count_saved": 2,
            "repair_guidance": [{"location_plane_mm": [1, 2], "residual_interval_mm": [0.1, 0.11]}],
            "overlay_geometry": {
                "saved_boundary_points": [
                    {
                        "role": "outer",
                        "connected_polyline": False,
                        "points_px": [[i, i] for i in range(10000)],
                    },
                    {
                        "role": "inner",
                        "connected_polyline": False,
                        "points_px": [[i, i] for i in range(1000)],
                    },
                ]
            },
        },
    }
    before = deepcopy(full)
    summary = photo_profile_context(full, "f" * 32)
    assert summary["result"]["symmetric_residual_interval_mm"] == [0.1, 0.11]
    assert summary["result"]["repair_guidance"] == full["result"]["repair_guidance"]
    assert [
        row["point_count"] for row in summary["result"]["overlay_geometry"]["saved_boundary_points"]
    ] == [10000, 1000]
    assert summary["projection"]["full_read"]["body"]["action"] == "compare"
    assert summary["projection"]["comparison_sha256"] == "d" * 64
    assert len(json.dumps(summary)) < 3000
    assert full == before


def test_analysis_cache_uses_runtime_identity_and_preserves_historical_reads(case, monkeypatch):
    identity = ["1" * 64]
    monkeypatch.setattr(profiles, "engine_fingerprint", lambda: identity[0])
    old = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    assert profiles.analyze_reference(case.service, case.sid, "owner", case.request) == old
    identity[0] = "2" * 64
    new = profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    assert new["id"] != old["id"] and len(case.calls) == 2
    assert new["implementation"]["host_engine_fingerprint"] == "2" * 64
    assert new["result"]["engine_fingerprint"] != "2" * 64  # Actual worker provenance is retained.
    assert profiles.read_profile(case.service, case.sid, "owner", old["id"]) == old
    # Old explicit records remain readable without pretending they ran new code.
    legacy = {key: value for key, value in old.items() if key != "implementation"}
    legacy["input_digest"] = profiles._digest(
        {"source": legacy["source"], "request": legacy["request"]}
    )
    with case.service.store.connect() as con:
        con.execute(
            "UPDATE photo_profile_observations SET document=?,digest=?,input_digest=? WHERE id=?",
            (json.dumps(legacy), profiles._digest(legacy), legacy["input_digest"], legacy["id"]),
        )
    assert profiles.read_profile(case.service, case.sid, "owner", old["id"]) == legacy


def test_runtime_drift_after_worker_does_not_publish(case, monkeypatch):
    identity = ["1" * 64]
    monkeypatch.setattr(profiles, "engine_fingerprint", lambda: identity[0])
    original = profiles.process_revision

    def changed_worker(*args):
        result = original(*args)
        identity[0] = "2" * 64
        return result

    monkeypatch.setattr(profiles, "process_revision", changed_worker)
    with pytest.raises(WorkspaceError, match="implementation changed"):
        profiles.analyze_reference(case.service, case.sid, "owner", case.request)
    with case.service.store.connect() as con:
        assert con.execute("SELECT count(*) FROM photo_profile_observations").fetchone()[0] == 0


@pytest.mark.parametrize("package", ["Pillow", "scikit-image"])
def test_image_dependency_version_changes_engine_identity(monkeypatch, package):
    from forge_mind.workspace import engine_identity

    original = engine_identity.importlib.metadata.version
    before = engine_identity.engine_fingerprint()
    monkeypatch.setattr(
        engine_identity.importlib.metadata,
        "version",
        lambda name: "independent-test-version" if name == package else original(name),
    )
    assert engine_identity.engine_fingerprint() != before
