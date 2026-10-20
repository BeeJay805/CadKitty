"""Referenced input delivery uses saved bytes; no native parsing or generation."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import RID, fixture_bytes
from tests.unit.workspace.test_project_delivery import saved_project
from tests.unit.workspace.test_project_handoff import members, rewrite

from forge_mind.workspace import processing
from forge_mind.workspace import production as production_module
from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import (
    IMPORT_ASSETS_SCOPE,
    json_bytes,
    verify_manufacturing_handoff,
)
from forge_mind.workspace.production import manufacture_bundle
from forge_mind.workspace.store import WorkspaceError

A, B, UNUSED = "a" * 32, "d" * 32, "e" * 32


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def bind(source, hashes, name, data):
    (source / name).write_bytes(data)
    hashes[name] = checksum(data)


@pytest.fixture
def imported(tmp_path):
    source = tmp_path / "source"
    document, hashes = fixture_bytes(source)
    raw = document.model_dump(mode="json")
    raw["assembly"]["parts"][3]["features"] = [
        {"id": "body", "name": "Owned reference", "kind": "import", "asset_id": A},
    ]
    raw["assembly"]["parts"][0]["features"].append(
        {
            "id": "unused_input",
            "name": "Suppressed dependency",
            "kind": "import",
            "asset_id": B,
            "suppressed": True,
        }
    )
    # Repeated references to the same input still produce one archive dependency.
    raw["assembly"]["parts"].append(
        {
            "id": "unused_ref",
            "name": "Unused reference definition",
            "features": [
                {
                    "id": "body",
                    "name": "Duplicate input declaration",
                    "kind": "import",
                    "asset_id": A,
                }
            ],
            "output": "body",
            "production": {"role": "reference"},
        }
    )
    document = DesignDocument.model_validate(raw)
    # Saved bytes intentionally differ from normalized JSON and omit false defaults.
    for part in raw["assembly"]["parts"]:
        if not part["production"].get("deliver_stl", False):
            part["production"].pop("deliver_stl", None)
    raw_bytes = json.dumps(raw, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
    bind(source, hashes, "document.json", raw_bytes)
    manifest = json.loads((source / "definition-meshes.json").read_bytes())
    manifest["document_sha256"] = checksum(raw_bytes)
    bind(source, hashes, "definition-meshes.json", json_bytes(manifest))
    records = []
    for asset_id, data in (
        (A, b"ISO-10303-21; owned irregular reference bytes"),
        (B, b"ISO-10303-21; suppressed reference bytes"),
    ):
        bind(source, hashes, f"input_{asset_id}.step", data)
        records.append({"id": asset_id, "name": "Owned input", "sha256": checksum(data)})
    bind(source, hashes, "assets.json", json.dumps(records, indent=1).encode())
    # Unreferenced uploads or files must not leak into delivery.
    (source / f"input_{UNUSED}.step").write_bytes(b"unrelated private input")
    bind(source, hashes, "part.step", b"saved overall world STEP fixture")
    return document, source, hashes


def bundle(saved, target, **options):
    document, source, hashes = saved
    return manufacture_bundle(document, source, target, hashes, **options)


@pytest.mark.parametrize("stl", [False, True])
@pytest.mark.parametrize("assembly", [False, True])
def test_opt_in_exact_source_dependency_closure_and_roles(imported, tmp_path, stl, assembly):
    path = bundle(
        imported,
        tmp_path / "inputs.zip",
        include_import_assets=True,
        include_stl=stl,
        include_assembly_step=assembly,
    )
    files = members(path)
    source, hashes = imported[1:]
    assert files["document.json"] == (source / "document.json").read_bytes()
    assert files["inputs/assets.json"] == (source / "assets.json").read_bytes()
    mapping = json.loads(files["import-assets.json"])
    assert mapping["scope"] == IMPORT_ASSETS_SCOPE
    assert mapping["document_sha256"] == hashes["document.json"]
    assert [row["asset_id"] for row in mapping["assets"]] == [A, B]
    for row in mapping["assets"]:
        data = (source / f"input_{row['asset_id']}.step").read_bytes()
        assert files[row["file"]] == data
        assert row["sha256"] == checksum(data) and row["size_bytes"] == len(data)
    assert {name for name in files if name.startswith("inputs/")} == {
        "inputs/assets.json",
        f"inputs/input_{A}.step",
        f"inputs/input_{B}.step",
    }
    assert not any(UNUSED in name for name in files)
    report = json.loads(files["production.json"])
    assert {line["part_id"]: line["quantity"] for line in report["make"]} == {"base": 1, "cap": 2}
    assert not any(
        "guide" in name or "bolt" in name or "unused_ref" in name
        for name in files
        if name.startswith("make/")
    )
    assert verify_manufacturing_handoff(path)["integrity"] == "verified"


def test_false_option_preserves_exact_default_members_and_receipt_shape(imported, tmp_path):
    old = members(bundle(imported, tmp_path / "default.zip"))
    explicit = members(bundle(imported, tmp_path / "false.zip", include_import_assets=False))
    assert old == explicit
    assert "include_import_assets" not in json.loads(old["manifest.json"])
    assert not any(name.startswith("inputs/") for name in old) and "import-assets.json" not in old


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_option_is_strict_boolean(imported, tmp_path, value):
    with pytest.raises(ValueError, match="include_import_assets must be a boolean"):
        bundle(imported, tmp_path / "bad.zip", include_import_assets=value)
    assert not list(tmp_path.glob("*.zip"))


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unbound",
        "changed",
        "metadata_hash",
        "metadata_duplicate",
        "metadata_extra",
        "metadata_nonfinite",
        "document_changed",
        "metadata_missing",
    ],
)
def test_missing_changed_or_ambiguous_source_inputs_refuse(imported, tmp_path, mutation):
    source, hashes = imported[1:]
    name = f"input_{A}.step"
    if mutation == "missing":
        (source / name).unlink()
    elif mutation == "unbound":
        hashes.pop(name)
    elif mutation == "changed":
        (source / name).write_bytes(b"changed")
    elif mutation == "document_changed":
        bind(source, hashes, "document.json", b"{}")
    elif mutation == "metadata_missing":
        hashes.pop("assets.json")
    else:
        records = json.loads((source / "assets.json").read_bytes())
        if mutation == "metadata_hash":
            records[0]["sha256"] = "0" * 64
        elif mutation == "metadata_duplicate":
            records.append(records[0])
        elif mutation == "metadata_extra":
            records.append({"id": UNUSED, "sha256": "0" * 64})
        else:
            records[0]["size_hint"] = float("inf")
        bind(source, hashes, "assets.json", json.dumps(records).encode())
    with pytest.raises(ValueError):
        bundle(imported, tmp_path / "bad.zip", include_import_assets=True)
    assert not list(tmp_path.glob("*.zip"))


@pytest.mark.parametrize(
    "mutation",
    [
        "path",
        "duplicate",
        "omitted",
        "extra_member",
        "disabled",
        "bool",
        "scope",
        "document_hash",
        "size_bool",
        "metadata_hash",
    ],
)
def test_rehashed_archive_with_inconsistent_dependency_map_refuses(imported, tmp_path, mutation):
    path = bundle(imported, tmp_path / "inputs.zip", include_import_assets=True)

    def change(files, manifest):
        mapping = json.loads(files["import-assets.json"])
        if mutation == "path":
            mapping["assets"][0]["file"] = "inputs/../escape.step"
        elif mutation == "duplicate":
            mapping["assets"][1] = mapping["assets"][0]
        elif mutation == "omitted":
            mapping["assets"].pop()
        elif mutation == "extra_member":
            files["inputs/extra.step"] = b"unrelated"
        elif mutation == "disabled":
            manifest["include_import_assets"] = False
            manifest["document_hash"] = digest(json.loads(files["document.json"]))
        elif mutation == "bool":
            manifest["include_import_assets"] = 1
        elif mutation == "scope":
            mapping["scope"] = "Physical mm and rebuild verified"
        elif mutation == "document_hash":
            mapping["document_sha256"] = "0" * 64
        elif mutation == "size_bool":
            mapping["assets"][0]["size_bytes"] = True
        else:
            records = json.loads(files["inputs/assets.json"])
            records[0]["sha256"] = "0" * 64
            files["inputs/assets.json"] = json_bytes(records)
        files["import-assets.json"] = json_bytes(mapping)

    rewrite(path, change)
    with pytest.raises(ValueError, match=r"[Ii]mport|dependency|referenced"):
        verify_manufacturing_handoff(path)


def test_combined_limit_includes_inputs_map_metadata_and_manifest(imported, tmp_path, monkeypatch):
    path = bundle(imported, tmp_path / "baseline.zip", include_import_assets=True)
    with ZipFile(path) as archive:
        total = sum(row.file_size for row in archive.infolist())
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", total)
    bundle(imported, tmp_path / "exact.zip", include_import_assets=True)
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", total - 1)
    with pytest.raises(ValueError, match="artifact budget"):
        bundle(imported, tmp_path / "over.zip", include_import_assets=True)
    assert not (tmp_path / "over.zip").exists()
    with pytest.raises(ValueError, match="artifact budget"):
        verify_manufacturing_handoff(path)


def test_oversized_input_refused_before_open(imported, tmp_path, monkeypatch):
    source = imported[1]
    consumed = sum(
        (source / name).stat().st_size
        for name in ("definition_base.step", "definition_cap.step", "document.json", "assets.json")
    )
    monkeypatch.setattr(
        processing, "MAX_JOB_BYTES", consumed + (source / f"input_{A}.step").stat().st_size - 1
    )
    original = Path.open

    def guard(path, *args, **kwargs):
        if path == source / f"input_{A}.step":
            pytest.fail("Oversized dependency must not be opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="artifact budget"):
        bundle(imported, tmp_path / "over.zip", include_import_assets=True)


def test_growth_after_stat_is_bounded_before_partial_package(imported, tmp_path, monkeypatch):
    source = imported[1]
    consumed = sum(
        (source / name).stat().st_size
        for name in ("definition_base.step", "definition_cap.step", "document.json", "assets.json")
    )
    remaining = 100
    monkeypatch.setattr(processing, "MAX_JOB_BYTES", consumed + remaining)
    original = Path.open

    def grown(path, *args, **kwargs):
        if path == source / f"input_{A}.step":
            return io.BytesIO(b"x" * (remaining + 1))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grown)
    with pytest.raises(ValueError, match="transfer budget"):
        bundle(imported, tmp_path / "grown.zip", include_import_assets=True)
    assert not list(tmp_path.glob("*.zip"))


def test_import_package_hashes_only_the_bounded_read(imported, tmp_path, monkeypatch):
    def unbounded_hash(path):
        pytest.fail("Opted-in packaging must not hash a made STEP with an unbounded pre-read")

    monkeypatch.setattr(production_module, "file_sha256", unbounded_hash)
    bundle(imported, tmp_path / "bounded.zip", include_import_assets=True)


def test_linked_input_is_refused(imported, tmp_path):
    os.link(imported[1] / f"input_{A}.step", tmp_path / "linked.step")
    with pytest.raises(ValueError, match="artifact missing"):
        bundle(imported, tmp_path / "linked.zip", include_import_assets=True)


def test_empty_dependency_option_is_explicit_and_valid(tmp_path):
    document, hashes = fixture_bytes(tmp_path)
    bind(tmp_path, hashes, "assets.json", b"[]\n")
    path = manufacture_bundle(
        document, tmp_path, tmp_path / "empty.zip", hashes, include_import_assets=True
    )
    assert json.loads(members(path)["import-assets.json"])["assets"] == []
    assert verify_manufacturing_handoff(path)["integrity"] == "verified"


def test_owned_service_input_receipt_and_readback_preserve_legacy(tmp_path, monkeypatch):
    service, sid, source = saved_project(tmp_path, monkeypatch, imported=True)
    record = service.store.revision(sid, RID)
    hashes = record["measurement"]["artifact_hashes"]
    data = b"ISO-10303-21; exact immutable owned input"
    bind(source, hashes, f"input_{A}.step", data)
    bind(
        source,
        hashes,
        "assets.json",
        json_bytes([{"id": A, "sha256": checksum(data), "name": "Owned reference"}]),
    )
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(record), sid, RID),
        )
    legacy = delivery.create_project_export(service, sid, "owner", RID)
    assert "include_import_assets" not in legacy
    assert legacy["source_scope"]["omitted_import_asset_ids"] == [A]
    receipt = delivery.create_project_export(service, sid, "owner", RID, include_import_assets=True)
    assert receipt["include_import_assets"] is True
    assert receipt["source_scope"]["omitted_import_asset_ids"] == []
    assert receipt["source_scope"]["provided_import_asset_ids"] == [A]
    assert receipt["source_scope"]["representation"] == "saved_document_bytes"
    assert receipt["source_scope"]["independent_regeneration"] == "not_verified"
    content, actual = delivery.read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert actual == receipt
    with ZipFile(io.BytesIO(content)) as archive:
        assert archive.read(f"inputs/input_{A}.step") == data
        assert (
            checksum(archive.read("import-assets.json"))
            == receipt["source_scope"]["input_map"]["sha256"]
        )
    path = source / "delivered.zip"
    path.write_bytes(content)
    with pytest.raises(WorkspaceError, match="input option"):
        delivery.validate_project_export(path, {**receipt, "include_import_assets": False})
    with pytest.raises(WorkspaceError):
        delivery.read_project_export(
            service, sid, "foreign", receipt["package_id"], receipt["sha256"]
        )


def test_service_missing_input_cleans_unpublished_package(tmp_path, monkeypatch):
    service, sid, _source = saved_project(tmp_path, monkeypatch, imported=True)
    with pytest.raises(ValueError, match="artifact missing"):
        delivery.create_project_export(service, sid, "owner", RID, include_import_assets=True)
    assert not list((service.store.root / "project-exports").rglob("*.zip"))
