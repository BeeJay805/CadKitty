from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZipFile, ZipInfo

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import handoff_context, verify_manufacturing_handoff
from forge_mind.workspace.models import Contract
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.production import manufacture_bundle
from forge_mind.workspace.store import Store, WorkspaceError


@pytest.fixture
def package(tmp_path):
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "production": {"role": "make"},
        }
    )
    source = tmp_path / "part.step"
    source.write_bytes(b"frozen STEP fixture")
    inputs = {"version": 2, "inputs": {"loads": []}}
    session = {
        "id": "a" * 32,
        "selected_revision": "b" * 32,
        "accepted_revision": "b" * 32,
        "engineering": inputs,
        "source_fingerprint": "sources-1",
    }
    revision = {
        "id": "b" * 32,
        "document": document.model_dump(),
        "measurement": {"engine_fingerprint": "measurement-engine"},
    }
    context = handoff_context(session, revision, "export-engine")
    path = manufacture_bundle(
        document,
        tmp_path,
        tmp_path / "review.zip",
        {"part.step": hashlib.sha256(source.read_bytes()).hexdigest()},
        sources={"fingerprint": "sources-1"},
        engineering_evidence={"revision_id": "b" * 32, "physical_readiness": "unverified"},
        revision_context=context,
        engineering_inputs=inputs,
    )
    return path


def rewrite(path, mutate):
    with ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    mutate(members)
    with ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_complete_manifest_and_honest_context(package):
    result = verify_manufacturing_handoff(package)
    assert result["integrity"] == "verified"
    assert result["members_verified"] == 9
    assert result["manufacturing_ready"] is False
    assert result["physical_readiness"] == "unverified"
    assert result["context"]["measurement_engine_fingerprint"] == "measurement-engine"
    assert result["context"]["export_engine_fingerprint"] == "export-engine"
    with ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert set(manifest["files"]) == set(archive.namelist()) - {"manifest.json"}
        for name, item in manifest["files"].items():
            data = archive.read(name)
            assert item == {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


@pytest.mark.parametrize(
    "member",
    [
        "production.json",
        "purchasing.json",
        "sources.json",
        "engineering-evidence.json",
        "make/part.step",
        "document.json",
        "engineering-inputs.json",
    ],
)
def test_each_member_tamper_rejected(package, member):
    rewrite(package, lambda members: members.__setitem__(member, b"changed"))
    with pytest.raises(ValueError, match="member changed"):
        verify_manufacturing_handoff(package)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "size",
        "digest",
        "schema",
        "boolsize",
        "approval",
        "context",
        "duplicatekey",
    ],
)
def test_malformed_manifest_rejected(package, change):
    def mutate(members):
        manifest = json.loads(members["manifest.json"])
        if change == "missing":
            members.pop("production.json")
        elif change == "extra":
            members["unexpected.txt"] = b"unexpected"
        elif change == "size":
            manifest["files"]["production.json"]["size_bytes"] += 1
        elif change == "digest":
            manifest["files"]["production.json"]["sha256"] = "xyz"
        elif change == "schema":
            manifest["schema_version"] = 99
        elif change == "boolsize":
            manifest["files"]["production.json"]["size_bytes"] = True
        elif change == "approval":
            manifest["manufacturing_ready"] = True
        elif change == "context":
            manifest["context"]["revision_id"] = "c" * 32
        elif change == "duplicatekey":
            members["manifest.json"] = b'{"schema_version":1,"schema_version":1}'
            return
        members["manifest.json"] = json.dumps(manifest).encode()

    rewrite(package, mutate)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(package)


@pytest.mark.parametrize(
    "change", ["omit_make", "production_files", "purchasing", "physical", "context", "declarations"]
)
def test_rehashed_but_contradictory_package_rejected(package, change):
    def mutate(members):
        manifest = json.loads(members["manifest.json"])
        if change == "omit_make":
            members.pop("make/part.step")
            manifest["files"].pop("make/part.step")
        elif change == "context":
            manifest["context"].pop("session_id")
        else:
            name = (
                "production.json"
                if change in {"production_files", "declarations"}
                else "purchasing.json"
                if change == "purchasing"
                else "engineering-evidence.json"
            )
            data = json.loads(members[name])
            if change == "production_files":
                data["files"] = {"make/missing.step": "0" * 64}
            elif change == "declarations":
                data["make"] = []
            elif change == "purchasing":
                data["lines"] = [{"part_id": "invented"}]
            else:
                data["physical_readiness"] = "verified"
            members[name] = json.dumps(data).encode()
            manifest["files"][name] = {
                "sha256": hashlib.sha256(members[name]).hexdigest(),
                "size_bytes": len(members[name]),
            }
        members["manifest.json"] = json.dumps(manifest).encode()

    rewrite(package, mutate)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(package)


@pytest.mark.parametrize(
    "name",
    [
        "../evil",
        "/absolute",
        "a\\b",
        "C:relative",
        "a//b",
        "a/./b",
        "manifest.json",
        "production.json",
    ],
)
def test_unsafe_or_duplicate_members_rejected(package, name):
    with ZipFile(package, "a") as archive:
        info = ZipInfo()
        info.filename = name  # Preserve deliberately invalid Windows separators.
        archive.writestr(info, b"{}")
    with pytest.raises(ValueError, match="Duplicate or unsafe"):
        verify_manufacturing_handoff(package)


def test_invalid_archive_rejected(tmp_path):
    path = tmp_path / "invalid.zip"
    path.write_bytes(b"not a ZIP")
    with pytest.raises(ValueError, match="Invalid manufacturing"):
        verify_manufacturing_handoff(path)


def test_invalid_compressed_member_rejected(package):
    with ZipFile(package) as archive:
        member = archive.getinfo("make/part.step")
    data = bytearray(package.read_bytes())
    offset = member.header_offset
    name_size = int.from_bytes(data[offset + 26 : offset + 28], "little")
    extra_size = int.from_bytes(data[offset + 28 : offset + 30], "little")
    data[offset + 30 + name_size + extra_size] = 0x07
    package.write_bytes(data)
    with pytest.raises(ValueError, match="Invalid manufacturing"):
        verify_manufacturing_handoff(package)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "inputs",
        "selection",
        "sources",
        "evidence_revision",
        "revision_recheck",
        "legacy_document",
    ],
)
def test_route_preserves_historical_export_and_rejects_mixed_snapshot(
    tmp_path, monkeypatch, mutation
):
    from forge_mind.workspace.engineering_models import EngineeringInputs
    from forge_mind.workspace.product_sources import _append

    store = Store(tmp_path / "data")
    sid = store.create("owner", "Fixture", None)["id"]
    rid = "b" * 32
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "production": {"role": "make"},
        }
    )
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    (directory / "part.step").write_bytes(b"verified test STEP")
    # Synthetic saved bytes exercise snapshot integrity, not CAD validity.
    (directory / "program.py").write_bytes(b"retained test program")
    (directory / "part.stl").write_bytes(b"retained test STL")
    revision = {
        "id": rid,
        "document": document.model_dump(),
        "measurement": {
            "revision_id": rid,
            "contract_hash": Contract().fingerprint(),
            "geometry_valid": True,
            "engine_fingerprint": "old-engine",
            "artifact_hashes": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("program.py", "part.step", "part.stl")
            },
        },
    }
    if mutation == "legacy_document":
        revision["document"] = document.model_dump(exclude_defaults=True)
    store.add_revision(sid, revision)
    store.update(sid, selected_revision="c" * 32, accepted_revision="c" * 32)
    reads = 0
    review_calls = 0

    def load_revision(*args):
        nonlocal reads
        reads += 1
        if mutation == "revision_recheck" and review_calls > 0:
            raise WorkspaceError("Revision artifact changed")
        return store.revision(*args)

    service = SimpleNamespace(store=store, _load_revision=load_revision)

    def review(*args):
        nonlocal review_calls
        review_calls += 1
        captured = store.get(sid, "owner")
        if mutation == "inputs" and review_calls == 1:
            store.save_engineering(sid, "owner", 0, EngineeringInputs().model_dump())
        elif mutation == "selection" and review_calls == 1:
            store.update(sid, selected_revision=rid)
        elif mutation == "sources" and review_calls == 1:
            _append(store, sid, "owner", {"kind": "artifact", "data": {"id": "new-source"}})
        return {
            "revision_id": "d" * 32 if mutation == "evidence_revision" else rid,
            "sources": {"fingerprint": captured["source_fingerprint"]},
            "physical_readiness": "unverified",
        }

    monkeypatch.setattr("forge_mind.workspace.project_delivery.product_review", review)
    app = FastAPI()
    app.include_router(product_router(store, service, lambda: "owner"))
    client = TestClient(app)
    response = client.get(f"/workspace/sessions/{sid}/manufacturing/{rid}")
    assert review_calls >= 1, "The control must reach its intended review or mutation"
    if mutation and mutation != "legacy_document":
        assert response.status_code == 409
        assert list((store.root / "manufacturing" / sid).glob("*.zip")) == []
    else:
        assert response.status_code == 200
        path = tmp_path / "download.zip"
        path.write_bytes(response.content)
        result = verify_manufacturing_handoff(path)
        assert result["context"]["revision_id"] == rid
        assert result["context"]["matches_current_design"] is False
        assert result["context"]["matches_accepted_revision"] is False
        assert store.get(sid, "owner")["accepted_revision"] == "c" * 32
    other = FastAPI()
    other.include_router(product_router(store, service, lambda: "intruder"))
    assert (
        TestClient(other).get(f"/workspace/sessions/{sid}/manufacturing/{rid}").status_code == 409
    )


@pytest.mark.parametrize(
    "selected,accepted,proposal,current,is_accepted",
    [
        ("b", "b", None, True, True),
        ("a", "a", "b", True, False),
        ("a", "a", "c", False, False),
    ],
)
def test_accepted_proposal_and_historical_context(
    selected, accepted, proposal, current, is_accepted
):
    session = {
        "id": "session",
        "selected_revision": selected,
        "accepted_revision": accepted,
        "proposal": {"revision_id": proposal},
        "engineering": {"version": 1, "inputs": {}},
        "source_fingerprint": "sources",
    }
    revision = {"id": "b", "document": {}, "measurement": {}}
    context = handoff_context(session, revision, "engine")
    assert context["matches_current_design"] is current
    assert context["matches_accepted_revision"] is is_accepted
    assert context["engineering_inputs_hash"] == digest(session["engineering"])


@pytest.mark.parametrize("explicit_null", [False, True])
def test_selected_revision_without_explicit_acceptance_stays_unaccepted(explicit_null):
    session = {
        "id": "session",
        "selected_revision": "b",
        "engineering": {"version": 1, "inputs": {}},
        "source_fingerprint": "sources",
    }
    if explicit_null:
        session["accepted_revision"] = None
    context = handoff_context(session, {"id": "b", "document": {}, "measurement": {}}, "engine")
    assert context["accepted_revision"] is None
    assert context["matches_accepted_revision"] is False
    assert context["matches_current_design"] is True
