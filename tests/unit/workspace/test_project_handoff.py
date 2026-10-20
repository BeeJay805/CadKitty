"""Project ZIP compatibility and byte-budget tests; no CAD generation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import fixture_bytes

from forge_mind.workspace import handoff, processing
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.handoff import json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.production import manufacture_bundle


@pytest.fixture
def saved(tmp_path):
    source = tmp_path / "source"
    document, hashes = fixture_bytes(source)
    assembly = b"saved overall STEP with made, bought and reference geometry"
    (source / "part.step").write_bytes(assembly)
    hashes["part.step"] = hashlib.sha256(assembly).hexdigest()
    return document, source, hashes


def bundle(saved, target, **options):
    document, source, hashes = saved
    return manufacture_bundle(document, source, target, hashes, **options)


def members(path):
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def rewrite(path, change):
    files = members(path)
    manifest = json.loads(files.pop("manifest.json"))
    change(files, manifest)
    manifest["files"] = {
        name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
        for name, data in files.items()
    }
    with ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
        archive.writestr("manifest.json", json_bytes(manifest))


@pytest.mark.parametrize("stl", [False, True])
def test_optional_assembly_preserves_make_files_source_roles_and_quantities(saved, tmp_path, stl):
    legacy = bundle(saved, tmp_path / "legacy.zip", include_stl=stl)
    project = bundle(saved, tmp_path / "project.zip", include_stl=stl, include_assembly_step=True)
    old, new = members(legacy), members(project)
    assert set(new) == set(old) | {"assembly/part.step", "README.md"}
    assert all(new[name] == data for name, data in old.items() if name != "manifest.json")
    manifest = json.loads(new["manifest.json"])
    assert manifest["include_assembly_step"] is True
    assert manifest["assembly"] == {
        "file": "assembly/part.step",
        "sha256": saved[2]["part.step"],
        "frame": handoff.ASSEMBLY_FRAME,
        "units": "mm",
        "scope": handoff.ASSEMBLY_SCOPE,
    }
    assert new["assembly/part.step"] == (saved[1] / "part.step").read_bytes()
    production = json.loads(new["production.json"])
    assert {row["part_id"]: row["quantity"] for row in production["make"]} == {"base": 1, "cap": 2}
    assert {row["part_id"] for row in production["buy"]} == {"bolt"}
    assert {row["part_id"] for row in production["reference"]} == {"guide"}
    assert all(
        "bolt" not in name and "guide" not in name for name in new if name.startswith("make/")
    )
    assert "local coordinates" in manifest["assembly"]["scope"]
    assert "Complete saved overall STEP" in manifest["disposition"]
    result = verify_manufacturing_handoff(project)
    assert result["manufacturing_ready"] is False
    assert result["physical_readiness"] == "unverified"


def test_false_option_has_exact_legacy_members_and_manifest(saved, tmp_path):
    omitted = members(bundle(saved, tmp_path / "omitted.zip"))
    explicit = members(bundle(saved, tmp_path / "false.zip", include_assembly_step=False))
    assert omitted == explicit
    manifest = json.loads(explicit["manifest.json"])
    assert "include_assembly_step" not in manifest and "assembly" not in manifest


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_option_is_strict_boolean(saved, tmp_path, value):
    with pytest.raises(ValueError, match="include_assembly_step must be a boolean"):
        bundle(saved, tmp_path / "invalid.zip", include_assembly_step=value)
    assert not (tmp_path / "invalid.zip").exists()


@pytest.mark.parametrize("change", ["missing", "unbound", "corrupt"])
def test_saved_assembly_missing_or_changed_refuses_without_partial_output(saved, tmp_path, change):
    source, hashes = saved[1:]
    if change == "missing":
        (source / "part.step").unlink()
    elif change == "unbound":
        hashes.pop("part.step")
    else:
        (source / "part.step").write_bytes(b"changed")
    with pytest.raises(ValueError, match=r"artifact (missing|changed)"):
        bundle(saved, tmp_path / "bad.zip", include_assembly_step=True)
    assert not (tmp_path / "bad.zip").exists()
    assert not list(tmp_path.glob("*.zip"))


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "disabled", "absent_flag", "frame", "units", "digest", "bool", "scope"],
)
def test_rehashed_but_inconsistent_assembly_metadata_is_rejected(saved, tmp_path, change):
    path = bundle(saved, tmp_path / "project.zip", include_assembly_step=True)

    def mutate(files, manifest):
        if change == "missing":
            files.pop("assembly/part.step")
        elif change == "extra":
            files["assembly/extra.step"] = b"extra"
        elif change == "disabled":
            manifest["include_assembly_step"] = False
        elif change == "absent_flag":
            manifest.pop("include_assembly_step")
        elif change == "bool":
            manifest["include_assembly_step"] = 1
        else:
            key = {"digest": "sha256"}.get(change, change)
            manifest["assembly"][key] = "wrong"

    rewrite(path, mutate)
    with pytest.raises(ValueError, match=r"assembly|Assembly|Project package"):
        verify_manufacturing_handoff(path)


def test_corrupted_assembly_member_detected_even_with_all_other_members_intact(saved, tmp_path):
    path = bundle(saved, tmp_path / "project.zip", include_assembly_step=True)
    original = members(path)
    original["assembly/part.step"] = b"tampered"
    with ZipFile(path, "w") as archive:
        for name, data in original.items():
            archive.writestr(name, data)
    with pytest.raises(ValueError, match=r"member changed: assembly/part\.step"):
        verify_manufacturing_handoff(path)


@pytest.mark.parametrize("stl", [False, True])
def test_complete_uncompressed_budget_includes_metadata_and_manifest(
    saved, tmp_path, monkeypatch, stl
):
    baseline = bundle(saved, tmp_path / "baseline.zip", include_assembly_step=True, include_stl=stl)
    with ZipFile(baseline) as archive:
        total = sum(info.file_size for info in archive.infolist())
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", total)
    bundle(saved, tmp_path / "exact.zip", include_assembly_step=True, include_stl=stl)
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", total - 1)
    with pytest.raises(ValueError, match="artifact budget"):
        bundle(saved, tmp_path / "over.zip", include_assembly_step=True, include_stl=stl)
    assert not (tmp_path / "over.zip").exists()
    assert {path.name for path in tmp_path.glob("*.zip")} == {"baseline.zip", "exact.zip"}


def test_oversize_assembly_refused_before_bulk_open(saved, tmp_path, monkeypatch):
    source = saved[1]
    made_bytes = sum(
        (source / f"definition_{name}.step").stat().st_size for name in ("base", "cap")
    )
    monkeypatch.setattr(
        processing, "MAX_JOB_BYTES", made_bytes + (source / "part.step").stat().st_size - 1
    )
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == saved[1] / "part.step":
            raise AssertionError("Oversized assembly must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="artifact budget"):
        bundle(saved, tmp_path / "over.zip", include_assembly_step=True)
    assert not (tmp_path / "over.zip").exists()


def test_linked_saved_assembly_is_refused(saved, tmp_path):
    os.link(saved[1] / "part.step", tmp_path / "another-name.step")
    with pytest.raises(ValueError, match="artifact missing"):
        bundle(saved, tmp_path / "linked.zip", include_assembly_step=True)
    assert not (tmp_path / "linked.zip").exists()


def test_flat_project_uses_saved_overall_geometry_without_named_stl(tmp_path):
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "production": {"role": "make"},
        }
    )
    step = b"saved flat geometry"
    (tmp_path / "part.step").write_bytes(step)
    path = manufacture_bundle(
        document,
        tmp_path,
        tmp_path / "flat.zip",
        {"part.step": hashlib.sha256(step).hexdigest()},
        include_assembly_step=True,
    )
    files = members(path)
    assert files["assembly/part.step"] == files["make/part.step"] == step
    assert verify_manufacturing_handoff(path)["integrity"] == "verified"


def test_verifier_refuses_combined_limit_before_reading_members(saved, tmp_path, monkeypatch):
    path = bundle(saved, tmp_path / "project.zip", include_assembly_step=True)
    with ZipFile(path) as archive:
        total = sum(info.file_size for info in archive.infolist())
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", total - 1)
    original_open = ZipFile.open

    def guarded_open(archive, name, *args, **kwargs):
        if name != "manifest.json":
            raise AssertionError("Over-budget payload must not be opened")
        return original_open(archive, name, *args, **kwargs)

    monkeypatch.setattr(ZipFile, "open", guarded_open)
    with pytest.raises(ValueError, match="artifact budget"):
        verify_manufacturing_handoff(path)


def test_project_metadata_is_bounded_before_archive_creation(saved, tmp_path, monkeypatch):
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", 12_000)
    with pytest.raises(ValueError, match="artifact budget"):
        bundle(
            saved,
            tmp_path / "over.zip",
            include_assembly_step=True,
            engineering_evidence={"physical_readiness": "unverified", "extra": "x" * 12_000},
        )
    assert not list(tmp_path.glob("*.zip"))


def test_failed_final_verification_keeps_existing_target_and_cleans_temporary(
    saved, tmp_path, monkeypatch
):
    target = tmp_path / "project.zip"
    target.write_bytes(b"previous usable export")

    def fail(_):
        raise ValueError("Rejected final archive")

    monkeypatch.setattr(handoff, "verify_manufacturing_handoff", fail)
    with pytest.raises(ValueError, match="Rejected final archive"):
        bundle(saved, target, include_assembly_step=True)
    assert target.read_bytes() == b"previous usable export"
    assert list(tmp_path.glob("*.zip")) == [target]
