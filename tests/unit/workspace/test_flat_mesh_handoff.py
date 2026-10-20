"""Flat capture checks use opaque saved bytes and never evaluate native geometry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.production import manufacture_bundle


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def flat_source(tmp_path, role="make"):
    source = tmp_path / "source"
    source.mkdir()
    data = {
        "name": "Saved flat part",
        "features": [{"id": "body", "name": "Body", "kind": "box"}],
        "outputs": ["body"],
    }
    if role is not None:
        data["production"] = {"role": role}
    # Preserve historical omitted defaults and author formatting in the raw source.
    raw = json.dumps(data, indent=3).encode() + b"\n"
    saved = {
        "document.json": raw,
        "part.step": b"opaque saved STEP bytes, not a geometry fixture",
        "part.stl": b"opaque saved STL bytes, not a geometry fixture",
    }
    for name, content in saved.items():
        (source / name).write_bytes(content)
    return source, DesignDocument.model_validate(data), {k: sha(v) for k, v in saved.items()}


def members(path):
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def reseal(path, change):
    files = members(path)
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


@pytest.mark.parametrize("overall", [False, True])
def test_flat_make_capture_preserves_raw_source_and_saved_cad(tmp_path, overall):
    source, doc, hashes = flat_source(tmp_path)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    path = manufacture_bundle(
        doc,
        source,
        tmp_path / "flat.zip",
        hashes,
        include_stl=True,
        include_assembly_step=overall,
    )
    result = verify_manufacturing_handoff(path)
    files = members(path)
    assert files["document.json"] == original["document.json"]
    assert files["make/part.step"] == original["part.step"]
    assert files["make/part.stl"] == original["part.stl"]
    assert "definition-meshes.json" not in files
    assert {name for name in files if name.startswith("make/")} == {
        "make/part.step",
        "make/part.stl",
    }
    report = json.loads(files["production.json"])
    assert [(r["part_id"], r["quantity"], r["role"]) for r in report["make"]] == [
        ("part", 1, "make")
    ]
    assert report["make"][0]["occurrence_paths"] == ["part"]
    assert report["files"] == {
        "make/part.step": hashes["part.step"],
        "make/part.stl": hashes["part.stl"],
    }
    manifest = json.loads(files["manifest.json"])
    assert manifest["document_hash"] == digest(doc.model_dump())
    assert manifest["include_stl"] is True
    assert "flat_mesh" not in manifest
    assert manifest["context"]["status"] == "unavailable"
    assert result["physical_readiness"] == "unverified"
    assert result["manufacturing_ready"] is False
    if overall:
        assert files["assembly/part.step"] == original["part.step"]
        assert manifest["assembly"]["frame"] == "cad_z_up_mm_world"
        assert "README.md" in files
    else:
        assert "assembly/part.step" not in files
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original


def test_flat_default_step_only_keeps_legacy_manifest_and_normalized_source(tmp_path):
    source, doc, hashes = flat_source(tmp_path)
    (source / "part.stl").unlink()
    hashes.pop("part.stl")
    path = manufacture_bundle(doc, source, tmp_path / "plain.zip", hashes)
    files = members(path)
    assert files["document.json"] == json_bytes(doc.model_dump())
    assert "make/part.stl" not in files
    assert "include_stl" not in json.loads(files["manifest.json"])
    assert verify_manufacturing_handoff(path)["physical_readiness"] == "unverified"


@pytest.mark.parametrize("role", [None, "unspecified", "buy", "reference"])
def test_flat_mesh_requires_actual_explicit_make_role(tmp_path, role):
    source, doc, hashes = flat_source(tmp_path, role)
    target = tmp_path / "existing.zip"
    target.write_bytes(b"previous handoff")
    with pytest.raises(ValueError, match="declared make"):
        manufacture_bundle(doc, source, target, hashes, include_stl=True)
    assert target.read_bytes() == b"previous handoff"


@pytest.mark.parametrize(
    "failure", ["missing", "hash_absent", "changed", "directory", "source", "step"]
)
def test_flat_saved_source_failures_preserve_previous_package(tmp_path, failure):
    source, doc, hashes = flat_source(tmp_path)
    if failure == "missing":
        (source / "part.stl").unlink()
    elif failure == "hash_absent":
        hashes.pop("part.stl")
    elif failure == "changed":
        (source / "part.stl").write_bytes(b"changed saved bytes")
    elif failure == "directory":
        (source / "part.stl").unlink()
        (source / "part.stl").mkdir()
    elif failure == "step":
        (source / "part.step").write_bytes(b"changed STEP")
    else:
        raw = json.loads((source / "document.json").read_bytes())
        raw["name"] = "Changed source"
        content = json_bytes(raw)
        (source / "document.json").write_bytes(content)
        hashes["document.json"] = sha(content)
    target = tmp_path / "existing.zip"
    target.write_bytes(b"previous handoff")
    with pytest.raises(ValueError):
        manufacture_bundle(doc, source, target, hashes, include_stl=True)
    assert target.read_bytes() == b"previous handoff"
    assert {p.name for p in tmp_path.iterdir()} == {"source", "existing.zip"}


def test_flat_stl_size_limit_refuses_before_opening_mesh(tmp_path, monkeypatch):
    source, doc, hashes = flat_source(tmp_path)
    limit = sum((source / name).stat().st_size for name in hashes) - 1
    monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", limit)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == source / "part.stl":
            pytest.fail("Oversized STL must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    target = tmp_path / "flat.zip"
    with pytest.raises(ValueError, match="budget"):
        manufacture_bundle(doc, source, target, hashes, include_stl=True)
    assert not target.exists()


def test_flat_metadata_uses_same_aggregate_budget_without_partial_zip(tmp_path, monkeypatch):
    source, doc, hashes = flat_source(tmp_path)
    # All source reads fit; package metadata must still fit the aggregate budget.
    limit = sum((source / name).stat().st_size for name in hashes) + 64
    monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", limit)
    target = tmp_path / "flat.zip"
    with pytest.raises(ValueError, match="budget"):
        manufacture_bundle(doc, source, target, hashes, include_stl=True)
    assert not target.exists()
    assert {p.name for p in tmp_path.iterdir()} == {"source"}


@pytest.mark.parametrize(
    "mutation", ["manifest", "missing_stl", "unexpected_stl", "flag", "quantity", "role", "source"]
)
def test_flat_offline_verifier_rejects_resealed_inconsistent_inventory(tmp_path, mutation):
    source, doc, hashes = flat_source(tmp_path)
    path = manufacture_bundle(doc, source, tmp_path / "flat.zip", hashes, include_stl=True)

    def change(files):
        if mutation == "manifest":
            files["definition-meshes.json"] = b"{}"
        elif mutation == "missing_stl":
            files.pop("make/part.stl")
        elif mutation == "unexpected_stl":
            files["make/definition_other.stl"] = b"unexpected mesh"
        elif mutation == "flag":
            manifest = json.loads(files["manifest.json"])
            manifest.pop("include_stl")
            files["manifest.json"] = json_bytes(manifest)
        elif mutation in {"quantity", "role"}:
            report = json.loads(files["production.json"])
            report["make"][0][mutation] = 2 if mutation == "quantity" else "buy"
            files["production.json"] = json_bytes(report)
        else:
            raw = json.loads(files["document.json"])
            raw["production"]["role"] = "reference"
            files["document.json"] = json_bytes(raw)
            manifest = json.loads(files["manifest.json"])
            manifest["document_hash"] = digest(DesignDocument.model_validate(raw).model_dump())
            files["manifest.json"] = json_bytes(manifest)

    reseal(path, change)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(path)


def test_flat_stl_byte_change_without_new_manifest_hash_is_refused(tmp_path):
    source, doc, hashes = flat_source(tmp_path)
    path = manufacture_bundle(doc, source, tmp_path / "flat.zip", hashes, include_stl=True)
    files = members(path)
    files["make/part.stl"] = b"changed member"
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match="Package member changed"):
        verify_manufacturing_handoff(path)


def test_flat_verifier_retains_unsigned_internal_integrity_scope(tmp_path):
    source, doc, hashes = flat_source(tmp_path)
    path = manufacture_bundle(doc, source, tmp_path / "flat.zip", hashes, include_stl=True)

    def change(files):
        # Whitespace changes preserve declared semantics. A fully resealed unsigned
        # archive cannot claim authentic original source bytes or native validity.
        files["document.json"] += b"\n"

    reseal(path, change)
    assert verify_manufacturing_handoff(path)["physical_readiness"] == "unverified"


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_flat_stl_option_remains_strict_boolean(tmp_path, value):
    source, doc, hashes = flat_source(tmp_path)
    with pytest.raises(ValueError, match="boolean"):
        manufacture_bundle(doc, source, tmp_path / "flat.zip", hashes, include_stl=value)


def test_flat_named_selection_schema_stays_forbidden():
    with pytest.raises(ValueError, match="only on assembly"):
        DesignDocument.model_validate(
            {
                "features": [{"id": "body", "name": "Body", "kind": "box"}],
                "outputs": ["body"],
                "production": {"role": "make", "deliver_stl": True},
            }
        )
