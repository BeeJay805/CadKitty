"""Owned immutable print packages using synthetic mesh bytes and a native-face stub."""

from __future__ import annotations

import io
import json
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import RID, fixture_bytes
from tests.unit.workspace.test_print_setup import box, plane

from forge_mind.workspace import print_setup as core
from forge_mind.workspace import print_setup_service as setup
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.handoff import json_bytes
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.print_setup_models import PrintSetupRequest
from forge_mind.workspace.store import Store, WorkspaceError


def project(tmp_path, monkeypatch):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("trusted_local"))
    sid = service.store.create("owner", "Measured print fixture", None)["id"]
    directory = service.store.directory(sid, RID)
    document, hashes = fixture_bytes(directory)
    for name, data in {
        "program.py": b"# Immutable fixture declaration; never execute.\n",
        "part.step": b"Saved assembly STEP byte fixture; not native geometry evidence",
        "part.stl": b"Saved assembly STL byte fixture; not parsed by the observer",
    }.items():
        (directory / name).write_bytes(data)
        hashes[name] = core.sha(data)
    manifest = json.loads((directory / "definition-meshes.json").read_bytes())
    for part_id in ("base", "cap"):
        data = core.binary_stl(box())
        (directory / f"definition_{part_id}.stl").write_bytes(data)
        entry = manifest["parts"][part_id]
        entry.update(
            stl_sha256=core.sha(data), stl_bytes=len(data), bounds_mm=[[-10, -15, -5], [10, 15, 5]]
        )
        hashes[entry["stl_file"]] = entry["stl_sha256"]
    data = json_bytes(manifest)
    (directory / "definition-meshes.json").write_bytes(data)
    hashes["definition-meshes.json"] = core.sha(data)
    contract = Contract()
    measured = Measurement(
        revision_id=RID,
        geometry_valid=True,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="fixture-engine",
        artifact_hashes=hashes,
    )
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "document": document.model_dump(),
            "contract": contract.model_dump(),
            "measurement": measured.model_dump(),
        },
    )
    service.store.update(
        sid, contract=contract.model_dump(), selected_revision=RID, accepted_revision=None
    )
    monkeypatch.setattr(setup, "engine_fingerprint", lambda: "fixture-engine")
    monkeypatch.setattr(core, "native_planes", lambda path, expected: [plane()])

    def worker(scratch, payload, *, module, timeout, cancelled):
        assert module == setup.WORKER_MODULE and timeout > 0
        assert not cancelled()
        result = core.run_print_setup(scratch, PrintSetupRequest.model_validate(payload))
        (scratch / "report.json").write_bytes(json_bytes(result))

    monkeypatch.setattr(service.executor, "run_typed_tool", worker)
    return service, sid, directory, worker


def test_inspection_does_not_bind_setup_or_change_source(tmp_path, monkeypatch):
    service, sid, directory, _ = project(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    inspected = setup.inspect_print_setup(service, sid, "owner", RID, "base")
    assert inspected["status"] == "measured"
    assert inspected["parts"]["base"]["recommended_candidate_hash"]
    assert service.store.session_record(sid, "owner").get("print_setup_records") is None
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}


def test_all_made_delivery_is_bound_owned_and_retained_after_edit(tmp_path, monkeypatch):
    service, sid, directory, _ = project(tmp_path, monkeypatch)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    service.store.update(sid, state="running")  # Author preparation is allowed.
    receipt = setup.prepare_print_files(service, sid, "owner", RID)
    assert receipt["status"] == "prepared"
    assert receipt["physical_readiness"] == "unverified"
    assert receipt["manufacturing_ready"] is False
    assert {name: row["quantity"] for name, row in receipt["files"].items()} == {
        "print/definition_base.stl": 1,
        "print/definition_cap.stl": 2,
    }
    data, readback = setup.read_print_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    with ZipFile(io.BytesIO(data)) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "setup.json",
            "README.txt",
            "print/definition_base.stl",
            "print/definition_cap.stl",
        }
        assert not any("bolt" in name or "guide" in name for name in archive.namelist())
    context = setup.print_setup_context(service, sid, "owner", RID)
    assert context["status"] == "current" and context["record_hash"]
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    service.store.update(sid, selected_revision=None)
    assert setup.print_setup_context(service, sid, "owner", RID)["status"] == "stale"
    retained, _ = setup.read_print_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert retained == data


@pytest.mark.parametrize("mutation", ["foreign", "checksum", "bytes", "receipt", "hardlink"])
def test_changed_or_foreign_package_is_refused(tmp_path, monkeypatch, mutation):
    service, sid, _, _ = project(tmp_path, monkeypatch)
    receipt = setup.prepare_print_files(service, sid, "owner", RID)
    directory = service.store.root / "print-exports" / sid / receipt["package_id"]
    owner, checksum = "owner", receipt["sha256"]
    if mutation == "foreign":
        owner = "other"
    elif mutation == "checksum":
        checksum = "0" * 64
    elif mutation == "bytes":
        (directory / "print-setup.zip").write_bytes(b"changed")
    elif mutation == "hardlink":
        (tmp_path / "linked.zip").hardlink_to(directory / "print-setup.zip")
    else:
        changed = json.loads((directory / "receipt.json").read_bytes())
        changed["files"]["print/definition_base.stl"]["quantity"] = 9
        (directory / "receipt.json").write_bytes(json_bytes(changed))
    with pytest.raises((ValueError, OSError)):
        setup.read_print_export(service, sid, owner, receipt["package_id"], checksum)


@pytest.mark.parametrize("change", ["head", "plan", "cancel", "missing_file"])
def test_race_or_incomplete_worker_never_publishes_success(tmp_path, monkeypatch, change):
    service, sid, _, worker = project(tmp_path, monkeypatch)
    cancelled = False

    def mutate(scratch, payload, **kwargs):
        nonlocal cancelled
        worker(scratch, payload, **kwargs)
        if change == "head":
            service.store.update(sid, selected_revision=None)
        elif change == "plan":
            service.store.update(sid, design_plan={"changed": True})
        elif change == "cancel":
            cancelled = True
        else:
            (scratch / "print_cap.stl").unlink()

    monkeypatch.setattr(service.executor, "run_typed_tool", mutate)
    with pytest.raises((ValueError, OSError)):
        setup.prepare_print_files(service, sid, "owner", RID, cancelled=lambda: cancelled)
    assert service.store.session_record(sid, "owner").get("print_setup_records") is None
    assert list((service.store.root / "print-exports" / sid).iterdir()) == []


def test_one_unresolved_part_prevents_partial_success(tmp_path, monkeypatch):
    service, sid, _, _ = project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        core, "native_planes", lambda path, expected: [] if "cap" in path.name else [plane()]
    )
    result = setup.prepare_print_files(service, sid, "owner", RID)
    assert result["status"] == "unresolved" and result["files"] == {}
    assert result["parts"]["base"]["status"] == "measured"
    assert result["parts"]["cap"]["status"] == "unresolved"
    assert "package_id" not in result
    assert service.store.session_record(sid, "owner").get("print_setup_records") is None
    assert list((service.store.root / "print-exports" / sid).iterdir()) == []


def test_source_corruption_refuses_before_worker(tmp_path, monkeypatch):
    service, sid, directory, _ = project(tmp_path, monkeypatch)
    (directory / "definition_base.stl").write_bytes(b"corrupt")
    monkeypatch.setattr(
        service.executor,
        "run_typed_tool",
        lambda *args, **kwargs: pytest.fail("Corrupt source cannot launch worker"),
    )
    with pytest.raises(WorkspaceError, match="artifacts changed"):
        setup.prepare_print_files(service, sid, "owner", RID)


def test_changed_record_cannot_be_bound_into_customer_guidance(tmp_path, monkeypatch):
    service, sid, _, _ = project(tmp_path, monkeypatch)
    setup.prepare_print_files(service, sid, "owner", RID)
    records = service.store.session_record(sid, "owner")["print_setup_records"]
    records[RID]["selected_candidates"]["base"] = "0" * 64
    service.store.update(sid, print_setup_records=records)
    with pytest.raises(WorkspaceError, match="immutable package"):
        setup.print_setup_context(service, sid, "owner", RID)


def test_repeat_preparation_reuses_exact_current_receipt_without_worker(tmp_path, monkeypatch):
    service, sid, _, _ = project(tmp_path, monkeypatch)
    receipt = setup.prepare_print_files(service, sid, "owner", RID)
    before = setup.print_setup_context(service, sid, "owner", RID)
    monkeypatch.setattr(
        service.executor,
        "run_typed_tool",
        lambda *args, **kwargs: pytest.fail("Unchanged print setup must reuse immutable package"),
    )
    assert setup.prepare_print_files(service, sid, "owner", RID) == receipt
    assert setup.print_setup_context(service, sid, "owner", RID) == before


def test_service_shutdown_refuses_preparation_before_worker(tmp_path, monkeypatch):
    service, sid, _, _ = project(tmp_path, monkeypatch)
    service.stop.set()
    monkeypatch.setattr(
        service.executor,
        "run_typed_tool",
        lambda *args, **kwargs: pytest.fail("Stopped service cannot launch worker"),
    )
    with pytest.raises(WorkspaceError, match="cancelled"):
        setup.prepare_print_files(service, sid, "owner", RID)
