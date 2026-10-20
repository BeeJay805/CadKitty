"""Manufacturing delivery tests use saved bytes, never native CAD execution."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.production import manufacture_bundle
from forge_mind.workspace.store import Store

RID = "b" * 32


def sha(data):
    return hashlib.sha256(data).hexdigest()


def document_data(selected=True):
    return {
        "assembly": {
            "parts": [
                {
                    "id": name,
                    "name": name,
                    "features": [{"id": "body", "name": "Body", "kind": "box"}],
                    "output": "body",
                    "production": {"role": role, "deliver_stl": selected and role == "make"},
                }
                for name, role in (
                    ("base", "make"),
                    ("cap", "make"),
                    ("bolt", "buy"),
                    ("guide", "reference"),
                )
            ],
            "roots": [
                {"id": name, "definition": definition}
                for name, definition in (
                    ("base", "base"),
                    ("cap_a", "cap"),
                    ("cap_b", "cap"),
                    ("bolt", "bolt"),
                    ("guide", "guide"),
                )
            ],
        }
    }


def fixture_bytes(source, selected=True):
    source.mkdir(parents=True, exist_ok=True)
    raw = json_bytes(DesignDocument.model_validate(document_data(selected)).model_dump())
    files = {"document.json": raw}
    parts = {}
    for name, quantity in (("base", 1), ("cap", 2)):
        step = f"saved {name} STEP".encode()
        stl = f"saved {name} STL".encode()
        files[f"definition_{name}.step"] = step
        files[f"definition_{name}.stl"] = stl
        parts[name] = {
            "status": "available",
            "role": "make",
            "quantity": quantity,
            "step_file": f"definition_{name}.step",
            "step_sha256": sha(step),
            "stl_file": f"definition_{name}.stl",
            "stl_sha256": sha(stl),
            "stl_bytes": len(stl),
            "body_count": 1,
            "bounds_mm": [[0, 0, 0], [10, 10, 10]],
            "tessellation": {
                "linear_deflection": 0.01,
                "linear_deflection_is_relative": True,
                "angular_deflection": 0.1,
                "collapsed_triangles_removed": 0,
            },
            "export_consistency": {
                "volume_within_relative_0_001": True,
                "bounds_within_0_01_mm": True,
                "maximum_sampled_bounds_difference_mm": 0.0,
                "reference_tessellation": None,
                "scope": "Numerical export consistency only. Exact dimensions and requirement checks use STEP; sampled tessellation bounds do not define manufacturing precision.",
            },
        }
    files["definition-meshes.json"] = json_bytes(
        {
            "schema_version": 1,
            "revision_id": RID,
            "document_sha256": sha(raw),
            "frame": "definition_local",
            "units": "mm",
            "parts": parts if selected else {},
        }
    )
    for name, data in files.items():
        (source / name).write_bytes(data)
    return DesignDocument.model_validate_json(raw), {
        name: sha(data) for name, data in files.items()
    }


def rewrite(path, change):
    with ZipFile(path) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    change(files)
    manifest = json.loads(files["manifest.json"])
    manifest["files"] = {
        name: {"sha256": sha(data), "size_bytes": len(data)}
        for name, data in files.items()
        if name != "manifest.json"
    }
    files["manifest.json"] = json_bytes(manifest)
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)


def test_default_step_membership_and_explicit_complete_mesh_bundle(tmp_path):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source)
    plain = manufacture_bundle(doc, source, tmp_path / "plain.zip", hashes)
    with ZipFile(plain) as archive:
        assert {name for name in archive.namelist() if name.startswith("make/")} == {
            "make/definition_base.step",
            "make/definition_cap.step",
        }
        assert "definition-meshes.json" not in archive.namelist()
        assert "include_stl" not in json.loads(archive.read("manifest.json"))
    mesh = manufacture_bundle(doc, source, tmp_path / "mesh.zip", hashes, include_stl=True)
    result = verify_manufacturing_handoff(mesh)
    assert result["manufacturing_ready"] is False
    assert result["physical_readiness"] == "unverified"
    with ZipFile(mesh) as archive:
        assert {name for name in archive.namelist() if name.startswith("make/")} == {
            "make/definition_base.step",
            "make/definition_cap.step",
            "make/definition_base.stl",
            "make/definition_cap.stl",
        }
        assert archive.read("document.json") == (source / "document.json").read_bytes()
        report = json.loads(archive.read("production.json"))
        assert [row["quantity"] for row in report["make"]] == [1, 2]
        assert [row["part_id"] for row in report["buy"]] == ["bolt"]


@pytest.mark.parametrize(
    "failure", ["missing", "changed", "failed", "legacy", "context", "size", "malformed"]
)
def test_missing_failed_stale_mesh_refuses_without_replacing_package(
    tmp_path, monkeypatch, failure
):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source)
    context = None
    if failure == "missing":
        (source / "definition_cap.stl").unlink()
    elif failure == "changed":
        (source / "definition_cap.stl").write_bytes(b"changed")
    elif failure == "legacy":
        (source / "definition-meshes.json").unlink()
    elif failure == "context":
        context = {"revision_id": "c" * 32}
    elif failure == "size":
        monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", 1)
    elif failure == "malformed":
        (source / "definition-meshes.json").write_bytes(b"[]")
        hashes["definition-meshes.json"] = sha(b"[]")
    else:
        path = source / "definition-meshes.json"
        manifest = json.loads(path.read_bytes())
        entry = manifest["parts"]["cap"]
        manifest["parts"]["cap"] = {
            key: value
            for key, value in entry.items()
            if key in {"role", "quantity", "step_file", "step_sha256"}
        } | {"status": "failed", "error": "Deliberate mesh failure"}
        path.write_bytes(json_bytes(manifest))
        hashes[path.name] = sha(path.read_bytes())
    target = tmp_path / "existing.zip"
    target.write_bytes(b"previous package")
    with pytest.raises(ValueError):
        manufacture_bundle(doc, source, target, hashes, revision_context=context, include_stl=True)
    assert target.read_bytes() == b"previous package"


@pytest.mark.parametrize("mutation", ["stl", "quantity", "buy_mesh", "flag", "source"])
def test_outer_manifest_reseal_does_not_hide_mesh_binding_errors(tmp_path, mutation):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source)
    path = manufacture_bundle(doc, source, tmp_path / "mesh.zip", hashes, include_stl=True)

    def change(files):
        if mutation == "stl":
            files["make/definition_cap.stl"] = b"different mesh"
        elif mutation == "buy_mesh":
            files["make/definition_bolt.stl"] = b"not a made part"
        elif mutation == "flag":
            manifest = json.loads(files["manifest.json"])
            manifest.pop("include_stl")
            files["manifest.json"] = json_bytes(manifest)
        elif mutation == "source":
            files["document.json"] += b"\n"
        else:
            manifest = json.loads(files["definition-meshes.json"])
            manifest["parts"]["cap"]["quantity"] = 3
            files["definition-meshes.json"] = json_bytes(manifest)

    rewrite(path, change)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(path)


def test_pre_change_step_package_default_remains_verifiable(tmp_path):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source, selected=False)
    path = manufacture_bundle(doc, source, tmp_path / "legacy.zip", hashes)

    def remove_new_default(value):
        if isinstance(value, dict):
            value.pop("deliver_stl", None)
            for child in value.values():
                remove_new_default(child)
        elif isinstance(value, list):
            for child in value:
                remove_new_default(child)

    def change(files):
        for name in ("document.json", "production.json", "purchasing.json"):
            value = json.loads(files[name])
            remove_new_default(value)
            files[name] = json_bytes(value)
        manifest = json.loads(files["manifest.json"])
        manifest["document_hash"] = digest(json.loads(files["document.json"]))
        files["manifest.json"] = json_bytes(manifest)

    rewrite(path, change)
    assert verify_manufacturing_handoff(path)["integrity"] == "verified"


def test_raw_selected_document_with_omitted_defaults_keeps_its_hash(tmp_path):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source)
    raw = json_bytes(document_data())
    (source / "document.json").write_bytes(raw)
    hashes["document.json"] = sha(raw)
    manifest = json.loads((source / "definition-meshes.json").read_bytes())
    manifest["document_sha256"] = sha(raw)
    (source / "definition-meshes.json").write_bytes(json_bytes(manifest))
    hashes["definition-meshes.json"] = sha(json_bytes(manifest))
    path = manufacture_bundle(doc, source, tmp_path / "minimal.zip", hashes, include_stl=True)
    with ZipFile(path) as archive:
        assert archive.read("document.json") == raw
    assert verify_manufacturing_handoff(path)["integrity"] == "verified"


def test_new_package_budget_precedes_step_read_and_hash(tmp_path, monkeypatch):
    source = tmp_path / "source"
    doc, hashes = fixture_bytes(source)
    monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", 1)

    def unexpected_hash(*args):
        raise AssertionError("Oversized STEP must be refused before hashing")

    monkeypatch.setattr("forge_mind.workspace.production.file_sha256", unexpected_hash)
    with pytest.raises(ValueError, match="budget"):
        manufacture_bundle(doc, source, tmp_path / "large.zip", hashes, include_stl=True)


def test_manufacturing_http_option_uses_same_owner_and_revision_checks(tmp_path, monkeypatch):
    from forge_mind.workspace.models import Contract

    store = Store(tmp_path / "store")
    sid = store.create("owner", "Saved mesh export", None)["id"]
    doc, hashes = fixture_bytes(store.directory(sid, RID))
    # Complete synthetic saved artifacts; these bytes are not parsed as CAD.
    for name, data in {
        "program.py": b"retained fixture program",
        "part.step": b"retained fixture assembly STEP",
        "part.stl": b"retained fixture assembly STL",
    }.items():
        (store.directory(sid, RID) / name).write_bytes(data)
        hashes[name] = sha(data)
    store.add_revision(
        sid,
        {
            "id": RID,
            "document": doc.model_dump(),
            "measurement": {
                "revision_id": RID,
                "contract_hash": Contract().fingerprint(),
                "geometry_valid": True,
                "artifact_hashes": hashes,
                "engine_fingerprint": "test-engine",
            },
        },
    )
    service = SimpleNamespace(store=store, _load_revision=store.revision)

    def review(*args):
        return {
            "revision_id": RID,
            "sources": {"fingerprint": store.get(sid, "owner")["source_fingerprint"]},
            "physical_readiness": "unverified",
        }

    monkeypatch.setattr("forge_mind.workspace.project_delivery.product_review", review)
    app = FastAPI()
    app.include_router(product_router(store, service, lambda: "owner"))
    response = TestClient(app).get(
        f"/workspace/sessions/{sid}/manufacturing/{RID}?include_stl=true"
    )
    assert response.status_code == 200, response.text
    target = tmp_path / "download.zip"
    target.write_bytes(response.content)
    assert verify_manufacturing_handoff(target)["context"]["revision_id"] == RID
    other = FastAPI()
    other.include_router(product_router(store, service, lambda: "intruder"))
    assert (
        TestClient(other)
        .get(f"/workspace/sessions/{sid}/manufacturing/{RID}?include_stl=true")
        .status_code
        == 409
    )
