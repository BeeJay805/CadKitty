"""Saved-source service controls. Image child is mocked; no native/GPU execution."""

from __future__ import annotations

import hashlib
import json

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Measurement
from forge_mind.workspace.occurrence_focus import OccurrenceFocus, focus_view_metadata
from forge_mind.workspace.occurrence_inspection import inspect_occurrence
from forge_mind.workspace.store import Store, WorkspaceError

FACE = "f" * 64
PNG = b"\x89PNG\r\n\x1a\nfocus-worker-test"


@pytest.fixture
def focused_revision(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("alice", "focus read", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    topology = {
        "frame": "cad_z_up_mm",
        "selection_scope": "body",
        "meshes": [],
        "entities": [
            {
                "kind": "body",
                "id": "b0",
                "body_index": 0,
                "signature": "1" * 64,
                "body_signature": "1" * 64,
                "min": [30, -5, 0],
                "max": [50, 5, 10],
            },
            {
                "kind": "face",
                "id": "f0",
                "body_index": 0,
                "signature": FACE,
                "body_signature": "1" * 64,
                "min": [39, -1, 8],
                "max": [41, 1, 8],
            },
            {
                "kind": "body",
                "id": "b1",
                "body_index": 1,
                "signature": "2" * 64,
                "body_signature": "2" * 64,
                "min": [80, -5, 0],
                "max": [100, 5, 10],
            },
        ],
    }
    sources = {
        "part.step": b"opaque saved STEP",
        "document.json": b'{"assembly":{}}',
        "topology.json": json.dumps(topology).encode(),
    }
    hashes = {}
    for name, content in sources.items():
        (directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    occurrences = []
    for path, index, x in (("support", 0, 30), ("other", 1, 80)):
        occurrences.append(
            {
                "path": path,
                "definition": "plate",
                "name": path,
                "body_index": index,
                "volume_mm3": 100,
                "center_mm": [x + 10, 0, 5],
                "bbox_min_mm": [x, -5, 0],
                "bbox_max_mm": [x + 20, 5, 10],
                "cylindrical_surfaces": [],
            }
        )
    measurement = Measurement(
        revision_id=rid,
        contract_hash="contract",
        engine_fingerprint="historical-engine",
        artifact_hashes=hashes,
        geometry_valid=True,
        facts={"assembly": {"occurrences": occurrences, "ports": {}}},
    ).model_dump()
    service.store.add_revision(sid, {"id": rid, "measurement": measurement})
    return service, sid, rid, directory, measurement


def child(command, directory, timeout, cancelled, **kwargs):
    assert timeout > 0 and not cancelled()
    request = json.loads((directory / "request.json").read_text())
    receipt = {
        **request,
        "image": {"file": "inspection.png", "sha256": hashlib.sha256(PNG).hexdigest()},
    }
    if "focus" in request:
        receipt["focus_view"] = focus_view_metadata(request["focus"])
    (directory / "inspection.png").write_bytes(PNG)
    (directory / "receipt.json").write_text(json.dumps(receipt))


def test_focus_is_owned_resolved_with_exact_view_and_keeps_saved_bytes(
    focused_revision, monkeypatch
):
    service, sid, rid, directory, measurement = focused_revision
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    seen = []

    def render(*args, **kwargs):
        request = json.loads((args[1] / "request.json").read_text())
        seen.append(request)
        child(*args, **kwargs)

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    requested = OccurrenceFocus(kind="face", signature=FACE, view="front", padding=2.5)
    response = inspect_occurrence(service, sid, "alice", rid, "support", focus=requested)
    frame = response["receipt"]["focus"]
    assert frame["request"] == requested.model_dump(mode="json")
    assert (
        frame["body_index"] == 0 and frame["entity_id"] == "f0" and frame["center_mm"] == [40, 0, 8]
    )
    assert response["receipt"]["focus_view"] == focus_view_metadata(frame)
    assert seen[0]["focus"] == frame and response["mime_type"] == "image/png"
    assert service.store.revision(sid, rid)["measurement"] == measurement
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    with service.store.connect() as db:
        assert db.execute("SELECT count(*) FROM jobs").fetchone()[0] == 0


def test_no_focus_keeps_exact_legacy_worker_keys_and_response(focused_revision, monkeypatch):
    service, sid, rid, *_ = focused_revision
    seen = []

    def render(*args, **kwargs):
        seen.append(json.loads((args[1] / "request.json").read_text()))
        child(*args, **kwargs)

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    response = inspect_occurrence(service, sid, "alice", rid, "support")
    assert set(seen[0]) == {
        "revision_id",
        "occurrence_path",
        "body_index",
        "topology_sha256",
        "step_sha256",
        "document_sha256",
    }
    assert set(response) == {"facts", "receipt", "image_base64", "mime_type"}
    assert "focus" not in response["receipt"] and "focus_view" not in response["receipt"]


@pytest.mark.parametrize(
    "failure", ["missing_focus", "wrong_entity", "wrong_view", "false_depth", "numeric_depth"]
)
def test_worker_focus_or_view_mismatch_refuses_publication(focused_revision, monkeypatch, failure):
    service, sid, rid, *_ = focused_revision

    def render(*args, **kwargs):
        child(*args, **kwargs)
        path = args[1] / "receipt.json"
        receipt = json.loads(path.read_text())
        if failure == "missing_focus":
            receipt.pop("focus")
        elif failure == "wrong_entity":
            receipt["focus"]["entity_id"] = "other"
        elif failure == "wrong_view":
            receipt["focus_view"]["basis"][0][0] += 0.1
        else:
            receipt["focus_view"]["all_selected_body_triangles_depth_tested"] = (
                False if failure == "false_depth" else 1
            )
        path.write_text(json.dumps(receipt))

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    with pytest.raises(ValueError):
        inspect_occurrence(
            service,
            sid,
            "alice",
            rid,
            "support",
            focus=OccurrenceFocus(kind="face", signature=FACE),
        )


def test_foreign_owner_before_source_and_focus_without_image_before_worker(
    focused_revision, monkeypatch
):
    service, sid, rid, *_ = focused_revision
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection._artifact",
        lambda *a: pytest.fail("No source read"),
    )
    requested = OccurrenceFocus(kind="face", signature=FACE)
    with pytest.raises(WorkspaceError):
        inspect_occurrence(service, sid, "bob", rid, "support", focus=requested)
    with pytest.raises(ValueError, match="include_image"):
        inspect_occurrence(
            service, sid, "alice", rid, "support", focus=requested, include_image=False
        )


def test_wrong_occurrence_focus_refuses_before_worker(focused_revision, monkeypatch):
    service, sid, rid, *_ = focused_revision
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("No worker"),
    )
    with pytest.raises(ValueError, match="missing or ambiguous"):
        inspect_occurrence(
            service, sid, "alice", rid, "other", focus=OccurrenceFocus(kind="face", signature=FACE)
        )


@pytest.mark.parametrize("failure", ["source", "cancel", "child_failure"])
def test_focus_retains_post_worker_source_and_cancellation_guards(
    focused_revision, monkeypatch, failure
):
    service, sid, rid, directory, _ = focused_revision
    scratch_paths = []

    def render(*args, **kwargs):
        scratch_paths.append(args[1])
        if failure == "child_failure":
            raise ValueError("worker deadline exceeded")
        child(*args, **kwargs)
        if failure == "source":
            (directory / "topology.json").write_bytes(b"changed")
        else:
            service.stop.set()

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    with pytest.raises(ValueError):
        inspect_occurrence(
            service,
            sid,
            "alice",
            rid,
            "support",
            focus=OccurrenceFocus(kind="face", signature=FACE),
        )
    assert scratch_paths and all(not path.exists() for path in scratch_paths)


def test_unqualified_hosted_focus_never_uses_local_render(focused_revision, monkeypatch):
    service, sid, rid, *_ = focused_revision
    service.executor = Executor("docker")
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("No host fallback"),
    )
    with pytest.raises(ValueError, match="qualified isolated"):
        inspect_occurrence(
            service,
            sid,
            "alice",
            rid,
            "support",
            focus=OccurrenceFocus(kind="face", signature=FACE),
        )


def test_no_focus_refuses_unrequested_worker_focus_metadata(focused_revision, monkeypatch):
    service, sid, rid, *_ = focused_revision

    def render(*args, **kwargs):
        child(*args, **kwargs)
        path = args[1] / "receipt.json"
        receipt = json.loads(path.read_text())
        receipt.update(focus={}, focus_view={})
        path.write_text(json.dumps(receipt))

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    with pytest.raises(ValueError, match="unrequested"):
        inspect_occurrence(service, sid, "alice", rid, "support")
