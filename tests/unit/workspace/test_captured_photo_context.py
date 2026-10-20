"""Internal captured photo evidence comes from bound ZIP declarations, never receipt guesses."""

import hashlib
import io
import json
from copy import deepcopy
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_project_members import rewrite, saved_capture

from forge_mind.workspace import project_members as members
from forge_mind.workspace.store import WorkspaceError


@pytest.mark.parametrize("include_stl", [True, False])
def test_context_uses_all_occurrences_and_raw_hashes_without_mutation(
    tmp_path, monkeypatch, include_stl
):
    content, package, _ = saved_capture(tmp_path, monkeypatch, include_stl=include_stl)
    before = deepcopy(package)
    value = members.captured_photo_context(content, package)
    assert value["structured_assembly"] is True
    assert value["make_quantities"] == [
        {"part_id": "base", "quantity": 1},
        {"part_id": "cap", "quantity": 2},
    ]
    assert [(r["occurrence_path"], r["part_id"]) for r in value["selected"]] == [
        ("base", "base"),
        ("cap_a", "cap"),
        ("cap_b", "cap"),
    ]
    assert [(r["occurrence_path"], r["role"]) for r in value["omitted"]] == [
        ("bolt", "buy"),
        ("guide", "reference"),
    ]
    with ZipFile(io.BytesIO(content)) as archive:
        assert (
            value["captured_document_sha256"]
            == hashlib.sha256(archive.read("document.json")).hexdigest()
        )
        assert (
            value["captured_production_sha256"]
            == hashlib.sha256(archive.read("production.json")).hexdigest()
        )
        assert (
            value["assembly_step_sha256"]
            == hashlib.sha256(archive.read("assembly/part.step")).hexdigest()
        )
    assert value["package_sha256"] == package["sha256"] and package == before
    assert set(value) == {
        "version",
        "project_id",
        "revision_id",
        "package_id",
        "package_sha256",
        "structured_assembly",
        "captured_document_sha256",
        "captured_production_sha256",
        "assembly_step_sha256",
        "selected",
        "omitted",
        "make_quantities",
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "bool_quantity",
        "count",
        "part",
        "role",
        "partition",
        "partition_bool",
        "missing",
    ],
)
def test_rebound_contradictory_production_refuses(tmp_path, monkeypatch, mutation):
    content, package, _ = saved_capture(tmp_path, monkeypatch)

    def change(files):
        value = json.loads(files["production.json"])
        line = value["lines"][1]
        if mutation == "duplicate":
            line["occurrence_paths"][1] = line["occurrence_paths"][0]
        elif mutation == "bool_quantity":
            value["lines"][0]["quantity"] = True
        elif mutation == "count":
            line["occurrence_paths"].pop()
        elif mutation == "part":
            line["part_id"] = "invented"
        elif mutation == "role":
            line["role"] = "reference"
        elif mutation == "partition":
            value["reference"] = []
        elif mutation == "partition_bool":
            value["reference"][0]["quantity"] = True
        else:
            value.pop("lines")
        files["production.json"] = json.dumps(value).encode()

    changed, rebound = rewrite(content, package, change, rebind_manifest=True)
    with pytest.raises(WorkspaceError):
        members.captured_photo_context(changed, rebound)


def test_legacy_catalog_buy_basis_is_preserved_without_lookup(tmp_path, monkeypatch):
    content, package, _ = saved_capture(tmp_path, monkeypatch)

    def change(files):
        document = json.loads(files["document.json"])
        next(row for row in document["assembly"]["parts"] if row["id"] == "bolt")["production"][
            "role"
        ] = "unspecified"
        files["document.json"] = json.dumps(document).encode()
        production = json.loads(files["production.json"])
        for rows in (production["lines"], production["buy"]):
            row = next(row for row in rows if row["part_id"] == "bolt")
            row.update(
                role_basis="unchanged legacy catalog geometry",
                catalog_id="recorded_catalog_identity",
            )
        files["production.json"] = json.dumps(production).encode()

    changed, rebound = rewrite(content, package, change, rebind_manifest=True)
    context = members.captured_photo_context(changed, rebound)
    assert context["omitted"][0]["role"] == "buy"


def test_changed_declared_metadata_without_new_manifest_refuses(tmp_path, monkeypatch):
    content, package, _ = saved_capture(tmp_path, monkeypatch)
    changed, rebound = rewrite(
        content, package, lambda files: files.update({"production.json": b"{}"})
    )
    with pytest.raises(WorkspaceError):
        members.captured_photo_context(changed, rebound)
