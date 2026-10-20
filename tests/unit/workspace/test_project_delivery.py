"""Owned saved-package service tests; CAD bytes are fixtures, not geometric proof."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from tests.unit.workspace.test_mesh_handoff import RID, fixture_bytes

from forge_mind.workspace import delivery_guidance, engineering_service
from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Measurement
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.store import Store, WorkspaceError


def saved_project(tmp_path, monkeypatch, *, selected=True, imported=False):
    service = WorkspaceService(Store(tmp_path / "store"), Executor("docker"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Export cannot start a worker"))
    sid = service.store.create("owner", "Saved package fixture", None)["id"]
    directory = service.store.directory(sid, RID)
    document, hashes = fixture_bytes(directory, selected=selected)
    if imported:
        raw = document.model_dump(mode="json")
        raw["assembly"]["parts"][0]["features"] = [
            {
                "id": "body",
                "name": "Imported shape",
                "kind": "import",
                "asset_id": "a" * 32,
            }
        ]
        document = type(document).model_validate(raw)
        content = delivery.json_bytes(document.model_dump())
        (directory / "document.json").write_bytes(content)
        hashes["document.json"] = hashlib.sha256(content).hexdigest()
    content = b"Saved overall STEP fixture with all roles"
    (directory / "part.step").write_bytes(content)
    hashes["part.step"] = hashlib.sha256(content).hexdigest()
    for name, content in {
        "program.py": b"# Immutable source fixture; never execute.\n",
        "part.stl": b"Saved overall mesh fixture",
    }.items():
        (directory / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    contract = Contract()
    measurement = Measurement(
        revision_id=RID,
        geometry_valid=True,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="fixture-engine",
        artifact_hashes=hashes,
    ).model_dump()
    service.store.add_revision(
        sid,
        {
            "id": RID,
            "document": document.model_dump(),
            "contract": contract.model_dump(),
            "measurement": measurement,
        },
    )
    service.store.update(sid, selected_revision=RID, accepted_revision=None)
    # Native construction is outside this service test. Freeze the reported engine while
    # independent implementation lanes edit unrelated source files during focused tests.
    monkeypatch.setattr(delivery, "engine_fingerprint", lambda: "fixture-export-engine")
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "fixture-export-engine")
    return service, sid, directory


@pytest.mark.parametrize("include_stl", [False, True])
def test_owned_package_is_exact_editable_snapshot_and_survives_restart(
    tmp_path,
    monkeypatch,
    include_stl,
):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    receipt = delivery.create_project_export(service, sid, "owner", RID, include_stl=include_stl)
    content, readback = delivery.read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    path = tmp_path / "received.zip"
    path.write_bytes(content)
    assert delivery.validate_project_export(path, receipt) == receipt
    with ZipFile(path) as archive:
        assert archive.read("assembly/part.step") == (source / "part.step").read_bytes()
        assert json.loads(archive.read("document.json")) == json.loads(
            (source / "document.json").read_bytes()
        )
        assert {row["part_id"]: row["quantity"] for row in receipt["made_parts"]} == {
            "base": 1,
            "cap": 2,
        }
        assert not any(
            "bolt" in name or "guide" in name
            for name in archive.namelist()
            if name.startswith("make/")
        )
        assert len([name for name in archive.namelist() if name.endswith(".stl")]) == (
            2 if include_stl else 0
        )
    assert receipt["source_scope"]["omitted_import_asset_ids"] == []
    assert receipt["physical_readiness"] == "unverified"
    assert receipt["context"]["matches_accepted_revision"] is False
    service.store.update(sid, selected_revision="c" * 32)
    restarted = WorkspaceService(Store(service.store.root), Executor("docker"))
    after, _ = delivery.read_project_export(
        restarted, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert after == content
    assert service.store.revisions(sid)[0]["id"] == RID and len(service.store.revisions(sid)) == 1


@pytest.mark.parametrize("acceptance", ["missing", "null", "different", "matching"])
def test_package_acceptance_requires_explicit_matching_owner_selection(
    tmp_path, monkeypatch, acceptance
):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    with service.store.connect() as con:
        row = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
        session = json.loads(row[0])
        if acceptance == "missing":
            session.pop("accepted_revision", None)
        else:
            session["accepted_revision"] = {
                "null": None,
                "different": "c" * 32,
                "matching": RID,
            }[acceptance]
        con.execute("UPDATE sessions SET document=? WHERE id=?", (json.dumps(session), sid))
    receipt = delivery.create_project_export(service, sid, "owner", RID)
    content, verified = delivery.read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert verified == receipt
    expected = session.get("accepted_revision")
    assert receipt["context"]["accepted_revision"] == expected
    assert receipt["context"]["matches_accepted_revision"] is (acceptance == "matching")
    with ZipFile(io.BytesIO(content)) as archive:
        context = json.loads(archive.read("manifest.json"))["context"]
        assert context == receipt["context"]
        assert context["matches_accepted_revision"] is (acceptance == "matching")
    assert service.store.session_record(sid, "owner").get("accepted_revision") == expected


def test_imported_source_dependency_is_explicit(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch, imported=True)
    receipt = delivery.create_project_export(service, sid, "owner", RID)
    assert receipt["source_scope"]["omitted_import_asset_ids"] == ["a" * 32]
    assert receipt["source_scope"]["independent_regeneration"] == "not_verified"


@pytest.mark.parametrize("mutation", ["hash", "archive", "receipt", "foreign", "hardlink"])
def test_readback_rejects_changed_or_foreign_package(tmp_path, monkeypatch, mutation):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    receipt = delivery.create_project_export(service, sid, "owner", RID)
    directory = service.store.root / "project-exports" / sid / receipt["package_id"]
    expected, owner = receipt["sha256"], "owner"
    if mutation == "hash":
        expected = "0" * 64
    elif mutation == "archive":
        (directory / "project.zip").write_bytes(b"changed")
    elif mutation == "receipt":
        value = json.loads((directory / "receipt.json").read_bytes())
        value["made_parts"][0]["quantity"] += 1
        (directory / "receipt.json").write_text(json.dumps(value), encoding="utf-8")
    elif mutation == "foreign":
        owner = "other"
    else:
        (tmp_path / "linked.zip").hardlink_to(directory / "project.zip")
    with pytest.raises((ValueError, OSError)):
        delivery.read_project_export(service, sid, owner, receipt["package_id"], expected)


@pytest.mark.parametrize("mutation", ["inputs", "review"])
def test_snapshot_race_cleans_unpublished_output(tmp_path, monkeypatch, mutation):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    original = delivery.product_review
    calls = 0

    def change(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls > 1:
            if mutation == "review":
                result["scope"] += " changed"
            else:
                service.store.update(sid, selected_revision="c" * 32)
        return result

    # Mutate immediately after the producer so the recaptured inputs are visible.
    producer = delivery.manufacture_bundle

    def build(*args, **kwargs):
        result = producer(*args, **kwargs)
        if mutation == "inputs":
            service.store.update(sid, selected_revision="c" * 32)
        return result

    monkeypatch.setattr(delivery, "product_review", change)
    monkeypatch.setattr(delivery, "manufacture_bundle", build)
    with pytest.raises(WorkspaceError, match="changed during handoff"):
        delivery.create_project_export(service, sid, "owner", RID)
    assert list((service.store.root / "project-exports" / sid).iterdir()) == []


@pytest.mark.parametrize("failure", ["missing_mesh", "oversize", "write_error"])
def test_failed_creation_never_publishes_package(tmp_path, monkeypatch, failure):
    service, sid, _ = saved_project(tmp_path, monkeypatch, selected=failure != "missing_mesh")
    if failure == "oversize":
        monkeypatch.setattr(delivery, "MAX_PACKAGE_BYTES", 1)
    if failure == "write_error":
        original = Path.open

        def denied(path, mode="r", *args, **kwargs):
            if path.name == "receipt.json" and mode == "xb":
                raise OSError("fixture write failure")
            return original(path, mode, *args, **kwargs)

        monkeypatch.setattr(Path, "open", denied)
    with pytest.raises((ValueError, OSError)):
        delivery.create_project_export(service, sid, "owner", RID, include_stl=True)
    assert list((service.store.root / "project-exports" / sid).iterdir()) == []


def test_http_project_create_download_and_foreign_owner(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    app = FastAPI()
    app.include_router(product_router(service.store, service, lambda: "owner"))
    with TestClient(app) as client:
        response = client.post(
            f"/workspace/sessions/{sid}/project-exports", json={"revision_id": RID}
        )
        assert response.status_code == 200, response.text
        receipt = response.json()
        uri = f"/workspace/sessions/{sid}/project-exports/{receipt['package_id']}/{receipt['sha256']}.zip"
        received = client.get(uri)
        assert received.status_code == 200
        assert hashlib.sha256(received.content).hexdigest() == receipt["sha256"]
        assert (
            client.post(
                f"/workspace/sessions/{sid}/project-exports",
                json={"revision_id": RID, "include_stl": "false"},
            ).status_code
            == 422
        )
    other = FastAPI()
    other.include_router(product_router(service.store, service, lambda: "other"))
    with TestClient(other) as client:
        assert client.get(uri).status_code == 409
        assert (
            client.post(
                f"/workspace/sessions/{sid}/project-exports", json={"revision_id": RID}
            ).status_code
            == 409
        )


def inspection_fixture(tmp_path, monkeypatch):
    """Saved report bytes exercise capture bounds without any native processing."""
    from forge_mind.workspace import inspection_plan

    plan = {"scope": "Controlled inspection fixture"}
    contents = {
        "report.json": json.dumps({"inspection_plan": plan}).encode(),
        "drawing.svg": b"<svg/>",
        "inspection.csv": b"measured,unverified\n",
    }
    report_id = "d" * 32
    directory = tmp_path / "reports" / "project" / report_id
    directory.mkdir(parents=True)
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    session = {
        "engineering": {"version": 0},
        "design_plan": None,
        "reports": [
            {
                "id": report_id,
                "revision_id": RID,
                "input_version": 0,
                "engine_fingerprint": "fixture-engine",
                "hashes": {
                    name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
                },
            }
        ],
    }
    service = SimpleNamespace(store=SimpleNamespace(root=tmp_path, get=lambda *args: session))
    monkeypatch.setattr(engineering_service, "engine_fingerprint", lambda: "fixture-engine")
    monkeypatch.setattr(inspection_plan, "resolve_inspection_plan", lambda *args: plan)
    return service, directory, contents


def test_inspection_exact_aggregate_budget_preserves_legacy_capture(tmp_path, monkeypatch):
    service, _, contents = inspection_fixture(tmp_path, monkeypatch)
    legacy = engineering_service.inspection_handoff(service, "project", "owner", {"id": RID})
    bounded = engineering_service.inspection_handoff(
        service,
        "project",
        "owner",
        {"id": RID},
        max_artifact_bytes=sum(map(len, contents.values())),
    )
    assert bounded == legacy
    assert bounded["artifacts"] == contents
    assert bounded["status"]["status"] == "included"


@pytest.mark.parametrize("target_name", ["report.json", "drawing.svg", "inspection.csv"])
def test_inspection_budget_refuses_before_open_or_hash(tmp_path, monkeypatch, target_name):
    service, directory, contents = inspection_fixture(tmp_path, monkeypatch)
    names = list(contents)
    # Each target exceeds the remaining aggregate allowance by exactly one byte.
    budget = sum(len(contents[name]) for name in names[: names.index(target_name) + 1]) - 1
    original_open = Path.open
    opened = []

    def controlled_open(path, *args, **kwargs):
        opened.append(path.name)
        if path == directory / target_name:
            pytest.fail("Oversized inspection file must not be opened for read or hashing")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", controlled_open)
    result = engineering_service.inspection_handoff(
        service, "project", "owner", {"id": RID}, max_artifact_bytes=budget
    )
    assert result["artifacts"] == {}
    assert result["status"]["status"] == "unavailable"
    assert "byte budget" in result["status"]["reason"]
    assert target_name not in opened


def test_inspection_growth_read_is_bounded_and_never_publishes_partial(tmp_path, monkeypatch):
    service, directory, contents = inspection_fixture(tmp_path, monkeypatch)
    budget = sum(map(len, contents.values()))
    original_open = Path.open
    observed_reads = []

    class GrowingFile(io.BytesIO):
        def read(self, size=-1):
            observed_reads.append(size)
            return super().read(size)

    def changed_open(path, *args, **kwargs):
        if path == directory / "drawing.svg":
            return GrowingFile(b"x" * (budget + 20))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", changed_open)
    result = engineering_service.inspection_handoff(
        service, "project", "owner", {"id": RID}, max_artifact_bytes=budget
    )
    assert observed_reads == [budget - len(contents["report.json"]) + 1]
    assert result["artifacts"] == {}
    assert result["status"]["status"] == "unavailable"
    assert "grew beyond" in result["status"]["reason"]


@pytest.mark.parametrize("budget", [True, -1, 1.0, "20"])
def test_inspection_budget_requires_nonnegative_integer(tmp_path, monkeypatch, budget):
    service, _, _ = inspection_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="nonnegative integer"):
        engineering_service.inspection_handoff(
            service, "project", "owner", {"id": RID}, max_artifact_bytes=budget
        )
