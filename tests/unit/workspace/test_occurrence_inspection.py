from __future__ import annotations

import hashlib
import json

import pytest

from forge_mind.workspace.agent import WorkspaceService, assembly_measurement_context
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Measurement
from forge_mind.workspace.occurrence_inspection import inspect_occurrence
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def owned_revision(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    sid = service.store.create("alice", "occurrence inspection", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    hashes = {}
    for name, content in {
        "part.step": b"opaque immutable native artifact",
        "document.json": b'{"assembly":{}}',
        "topology.json": b'{"meshes":[]}',
    }.items():
        (directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    measurement = Measurement(
        revision_id=rid,
        contract_hash="contract",
        engine_fingerprint="engine",
        artifact_hashes=hashes,
        geometry_valid=True,
        facts={
            "assembly": {
                "occurrences": [
                    {
                        "path": "support",
                        "definition": "plate",
                        "name": "Support",
                        "body_index": 0,
                        "volume_mm3": 100,
                        "center_mm": [0, 0, 0],
                        "bbox_min_mm": [-5, -5, -0.5],
                        "bbox_max_mm": [5, 5, 0.5],
                        "cylindrical_surfaces": [],
                    }
                ],
                "ports": {},
            }
        },
    ).model_dump()
    service.store.add_revision(sid, {"id": rid, "measurement": measurement})
    return service, sid, rid, directory, measurement


def fake_render(command, directory, timeout, cancelled, **kwargs):
    assert command[-3] == "-m"
    assert timeout > 0 and not cancelled()
    request = json.loads((directory / "request.json").read_text())
    image = b"\x89PNG\r\n\x1a\nrenderer-test-bytes"
    (directory / "inspection.png").write_bytes(image)
    receipt = {
        **request,
        "image": {"file": "inspection.png", "sha256": hashlib.sha256(image).hexdigest()},
    }
    (directory / "receipt.json").write_text(json.dumps(receipt))


def test_saved_facts_do_not_start_worker_or_mutate_revision(owned_revision, monkeypatch):
    service, sid, rid, directory, measurement = owned_revision
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("Facts must not run an image worker"),
    )
    response = inspect_occurrence(service, sid, "alice", rid, "support", include_image=False)
    assert response["image_base64"] is None
    assert (
        response["facts"]["provenance"]["step_sha256"]
        == measurement["artifact_hashes"]["part.step"]
    )
    assert service.store.revision(sid, rid)["measurement"] == measurement
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before


def test_other_owner_and_other_revision_fail_before_read(owned_revision, monkeypatch):
    service, sid, rid, *_ = owned_revision
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection._artifact",
        lambda *a, **k: pytest.fail("Unowned revision must not be read"),
    )
    with pytest.raises(WorkspaceError):
        inspect_occurrence(service, sid, "bob", rid, "support")
    with pytest.raises(WorkspaceError):
        inspect_occurrence(service, sid, "alice", "b" * 32, "support")


@pytest.mark.parametrize("name", ["part.step", "document.json", "topology.json"])
def test_stale_artifact_rejected(owned_revision, monkeypatch, name):
    service, sid, rid, directory, _ = owned_revision
    (directory / name).write_bytes(b"changed")
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("Stale source must not reach a worker"),
    )
    with pytest.raises(ValueError, match="stale"):
        inspect_occurrence(service, sid, "alice", rid, "support")


def test_ephemeral_image_receipt_and_cleanup(owned_revision, monkeypatch):
    service, sid, rid, directory, measurement = owned_revision
    seen = []

    def render(*args, **kwargs):
        seen.append(args[1])
        fake_render(*args, **kwargs)

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    result = inspect_occurrence(service, sid, "alice", rid, "support")
    assert result["receipt"]["revision_id"] == rid
    assert result["receipt"]["body_index"] == 0
    assert result["receipt"]["topology_sha256"] == measurement["artifact_hashes"]["topology.json"]
    assert result["mime_type"] == "image/png" and result["image_base64"]
    assert len(list(directory.iterdir())) == 3
    assert all(not path.exists() for path in seen)


@pytest.mark.parametrize("mutation", ["receipt", "image", "source", "cancel"])
def test_no_partial_success_after_worker(owned_revision, monkeypatch, mutation):
    service, sid, rid, directory, _ = owned_revision

    def render(*args, **kwargs):
        fake_render(*args, **kwargs)
        scratch = args[1]
        if mutation == "receipt":
            receipt = json.loads((scratch / "receipt.json").read_text())
            receipt["body_index"] = 1
            (scratch / "receipt.json").write_text(json.dumps(receipt))
        elif mutation == "image":
            (scratch / "inspection.png").write_bytes(b"wrong image")
        elif mutation == "source":
            (directory / "part.step").write_bytes(b"modified during render")
        else:
            service.stop.set()

    monkeypatch.setattr("forge_mind.workspace.occurrence_inspection.bounded_process", render)
    with pytest.raises(ValueError):
        inspect_occurrence(service, sid, "alice", rid, "support")


def test_hosted_facts_work_but_unqualified_image_has_no_fallback(owned_revision, monkeypatch):
    service, sid, rid, *_ = owned_revision
    service.executor = Executor("docker")
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("Hosted inspection must not use host rendering"),
    )
    inspect_occurrence(service, sid, "alice", rid, "support", include_image=False)
    with pytest.raises(ValueError, match="qualified isolated"):
        inspect_occurrence(service, sid, "alice", rid, "support")


def test_qualified_hosted_image_uses_immutable_container(owned_revision, monkeypatch):
    service, sid, rid, *_ = owned_revision
    service.executor = Executor("docker")
    service.executor._typed_container_verified = True
    service.executor.typed_container_image = "sha256:" + "b" * 64
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.shutil.which", lambda _: "docker"
    )
    monkeypatch.setattr(
        "forge_mind.workspace.occurrence_inspection.bounded_process",
        lambda *a, **k: pytest.fail("Hosted inspection must not use host rendering"),
    )

    def container(docker, directory, module, request, timeout, cancelled, *, image):
        assert image == "sha256:" + "b" * 64
        assert module == "forge_mind.workspace.occurrence_render"
        fake_render(["python", "-m", module, request], directory, timeout, cancelled)

    monkeypatch.setattr(service.executor, "_run_container", container)
    assert inspect_occurrence(service, sid, "alice", rid, "support")["image_base64"]


@pytest.mark.parametrize("timeout", [0, -1, 61, float("nan"), float("inf"), True])
def test_invalid_deadline_rejected(owned_revision, timeout):
    service, sid, rid, *_ = owned_revision
    with pytest.raises(ValueError, match="deadline"):
        inspect_occurrence(service, sid, "alice", rid, "support", timeout=timeout)


def test_large_surface_payload_is_not_added_to_automatic_author_context(owned_revision):
    *_, measurement = owned_revision
    compact = assembly_measurement_context(measurement)
    assert "cylindrical_surfaces" not in compact["facts"]["assembly"]["occurrences"][0]
    assert "cylindrical_surfaces" in measurement["facts"]["assembly"]["occurrences"][0]


@pytest.mark.parametrize("field", ["revision_id", "step_sha256", "engine_fingerprint"])
def test_saved_surface_provenance_cannot_reference_another_revision(
    owned_revision, monkeypatch, field
):
    service, sid, rid, _, measurement = owned_revision
    measurement["facts"]["assembly"][field] = "other"
    monkeypatch.setattr(
        service.store, "revision", lambda *args: {"id": rid, "measurement": measurement}
    )
    with pytest.raises(ValueError, match="provenance"):
        inspect_occurrence(service, sid, "alice", rid, "support", include_image=False)


def test_nonassembly_candidate_has_explicit_unavailable_error(owned_revision, monkeypatch):
    service, sid, rid, _, measurement = owned_revision
    measurement["facts"]["assembly"] = None
    monkeypatch.setattr(
        service.store, "revision", lambda *args: {"id": rid, "measurement": measurement}
    )
    with pytest.raises(ValueError, match="saved assembly"):
        inspect_occurrence(service, sid, "alice", rid, "support", include_image=False)


@pytest.mark.parametrize("render_fails", [False, True])
def test_worker_qualification_requires_occurrence_render_probe(monkeypatch, render_fails):
    from types import SimpleNamespace

    import forge_mind.workspace.executor as executor_module

    executor = Executor()
    identity = "sha256:" + "f" * 64
    monkeypatch.setattr(executor_module.shutil, "which", lambda _: "docker")
    monkeypatch.setattr(
        executor_module.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=json.dumps([{"Id": identity}]).encode()),
    )

    def build(directory, payload, timeout, cancelled):
        names = ("part.step", "part.stl", "part.glb", "render.png", "topology.json")
        for name in names:
            (directory / name).write_bytes(b"qualification fixture")
        return {
            "geometry_valid": True,
            "artifact_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
            },
        }

    def container(docker, directory, module, request, timeout, cancelled, *, image):
        assert image == identity
        if module.endswith("occurrence_render"):
            if render_fails:
                raise ValueError("occurrence module unavailable")
            fake_render(["python", "-m", module, request], directory, timeout, cancelled)
        elif directory.name == "probe":
            checks = dict.fromkeys(
                (
                    "unprivileged",
                    "readonly_root",
                    "no_capabilities",
                    "no_new_privileges",
                    "no_network_route",
                    "host_paths_absent",
                ),
                True,
            )
            (directory / "probe-result.json").write_text(json.dumps(checks))
        else:
            raise ValueError("wall-clock" if directory.name == "timeout" else "cancelled")

    monkeypatch.setattr(executor, "process_document", build)
    monkeypatch.setattr(executor, "_run_container", container)
    if render_fails:
        with pytest.raises(ValueError, match="occurrence module unavailable"):
            executor.qualify_typed_container()
        assert not executor.typed_container_verified
    else:
        result = executor.qualify_typed_container()
        assert "occurrence_mesh_image" in result["checks"]
        assert result["image"] == identity
        assert executor.typed_container_verified
