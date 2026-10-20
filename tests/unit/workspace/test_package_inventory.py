"""Package paths come from saved fixtures; no CAD or model execution is needed."""

from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import fixture_bytes

from forge_mind.workspace.delivery_guidance import _facts
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.production import manufacture_bundle, package_file_inventory


def customer_saved_project(tmp_path, monkeypatch, printable):
    from tests.unit.workspace.test_customer_delivery_policy import customer
    from tests.unit.workspace.test_flat_project_delivery_transport import flat_saved_project

    service, original_sid, _, files = flat_saved_project(tmp_path, monkeypatch)
    sid = service.store.create("owner", "CAD delivery", {}, "native_operations")["id"]
    revision = service.store.revision(
        original_sid, service.store.get(original_sid, "owner")["selected_revision"]
    )
    revision["id"] = "f" * 32
    revision["measurement"]["revision_id"] = revision["id"]
    directory = service.store.directory(sid, revision["id"])
    directory.mkdir(parents=True)
    for name, content in files.items():
        (directory / name).write_bytes(content)
    service.store.add_revision(sid, revision)
    service.store.update(sid, selected_revision=revision["id"], **customer(printable))
    return service, sid


@pytest.mark.parametrize("printable", [False, True])
def test_customer_guidance_uses_automatic_mesh_policy(tmp_path, monkeypatch, printable):
    from forge_mind.workspace.delivery_guidance import guidance_context

    service, sid = customer_saved_project(tmp_path, monkeypatch, printable)
    facts = guidance_context(service, sid, "owner")["facts"]
    assert facts["package_files"]["options"]["include_stl"] is printable
    if printable:
        assert facts["package_files"]["geometry"]["make/part.stl"]["source_present"] is True
    else:
        assert "make/part.stl" not in facts["package_files"]["geometry"]


def test_cad_only_guidance_retains_explicit_mesh_and_legacy_inventory():
    from tests.unit.workspace.test_customer_delivery_policy import customer
    from tests.unit.workspace.test_mesh_handoff import document_data

    revision = {"document": DesignDocument.model_validate(document_data()).model_dump()}
    inventory = _facts(revision, session=customer(False))["package_files"]
    assert inventory["options"]["include_stl"] is True
    assert inventory["geometry"]["make/definition_base.stl"]["source_present"] is False
    assert _facts(revision, session={})["package_files"] == _facts(revision)["package_files"]


@pytest.mark.parametrize("printable", [False, True])
def test_absent_optional_definition_mesh_is_expected_only_when_requested(printable):
    from tests.unit.workspace.test_customer_delivery_policy import customer
    from tests.unit.workspace.test_mesh_handoff import document_data

    revision = {
        "document": DesignDocument.model_validate(document_data(selected=False)).model_dump()
    }
    inventory = _facts(revision, session=customer(printable))["package_files"]
    assert inventory["options"]["include_stl"] is printable
    if printable:
        assert inventory["geometry"]["make/definition_base.stl"]["source_present"] is False
    else:
        assert not any(name.endswith(".stl") for name in inventory["geometry"])


@pytest.mark.parametrize("printable", [None, "false"])
def test_customer_guidance_does_not_hide_malformed_print_policy(tmp_path, monkeypatch, printable):
    from forge_mind.workspace.delivery_guidance import guidance_context
    from forge_mind.workspace.store import WorkspaceError

    service, sid = customer_saved_project(tmp_path, monkeypatch, printable)
    with pytest.raises(WorkspaceError, match="strict printable"):
        guidance_context(service, sid, "owner")


@pytest.mark.parametrize("include_stl", [False, True])
def test_inventory_geometry_matches_actual_export_and_excludes_saved_references(
    tmp_path, include_stl
):
    source = tmp_path / "saved"
    document, hashes = fixture_bytes(source)
    for name in ("part.step", "part.stl", "definition_guide.step", "definition_bolt.step"):
        data = ("Saved " + name).encode()
        (source / name).write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    before = copy.deepcopy((document.model_dump(), hashes))
    inventory = package_file_inventory(
        document, hashes, include_stl=include_stl, include_assembly_step=True
    )
    target = manufacture_bundle(
        document,
        source,
        tmp_path / "project.zip",
        hashes,
        include_stl=include_stl,
        include_assembly_step=True,
    )
    with ZipFile(target) as archive:
        geometry = {name for name in archive.namelist() if name.endswith((".step", ".stl"))}
        assert set(inventory["geometry"]) == geometry
        assert inventory["geometry"]["make/definition_cap.step"]["quantity"] == 2
        assert "definition_guide.step" in inventory["excluded_saved_artifacts"]
        assert "definition_bolt.step" in inventory["excluded_saved_artifacts"]
        assert "part.stl" in inventory["excluded_saved_artifacts"]
        assert "assembly/part.step" in inventory["geometry"]
        assert set(inventory["metadata_files"]) <= set(archive.namelist())
        assert json.loads(archive.read("manifest.json"))["manufacturing_ready"] is False
    assert (document.model_dump(), hashes) == before


def test_inventory_distinguishes_missing_saved_mesh_from_a_prepared_print_file():
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "outputs": ["body"],
            "production": {"role": "make"},
        }
    )
    inventory = package_file_inventory(document, {"part.step": "a" * 64})
    assert inventory["geometry"]["make/part.step"]["source_present"] is True
    assert inventory["geometry"]["make/part.stl"]["source_present"] is False
    assert inventory["print_preparation"] == "not established by this inventory"
    assert "print/part.stl" not in inventory["geometry"]


@pytest.mark.parametrize("include_import_assets", [False, True])
def test_imported_source_paths_follow_the_selected_option(include_import_assets):
    asset_id = "a" * 32
    source = f"input_{asset_id}.step"
    document = DesignDocument.model_validate(
        {
            "features": [
                {"id": "body", "name": "Owned input", "kind": "import", "asset_id": asset_id}
            ],
            "outputs": ["body"],
            "production": {"role": "make"},
        }
    )
    inventory = package_file_inventory(
        document, {source: "b" * 64}, include_import_assets=include_import_assets
    )
    if include_import_assets:
        assert inventory["geometry"]["inputs/" + source]["source_artifact"] == source
        assert source not in inventory["excluded_saved_artifacts"]
    else:
        assert "inputs/" + source not in inventory["geometry"]
        assert source in inventory["excluded_saved_artifacts"]


@pytest.mark.parametrize("imported", [False, True])
def test_guidance_inventory_advertises_import_metadata_only_for_referenced_inputs(imported):
    feature = {"id": "body", "name": "Body", "kind": "box"}
    if imported:
        feature = {**feature, "kind": "import", "asset_id": "a" * 32}
    document = DesignDocument.model_validate(
        {"features": [feature], "outputs": ["body"], "production": {"role": "make"}}
    )
    inventory = _facts({"document": document.model_dump()})["package_files"]
    assert inventory["options"]["include_import_assets"] is imported
    optional = {"inputs/assets.json", "import-assets.json"}
    assert optional & set(inventory["metadata_files"]) == (optional if imported else set())
    assert any(name.startswith("inputs/") for name in inventory["geometry"]) is imported
