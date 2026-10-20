"""Prepared saved-byte controls; no native CAD or author execution is required."""

from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_mesh_handoff import fixture_bytes

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.handoff import verify_manufacturing_handoff
from forge_mind.workspace.production import (
    _package_readme,
    manufacture_bundle,
    package_file_inventory,
    production_report,
)


@pytest.mark.parametrize("required_stl", [False, True])
@pytest.mark.parametrize("included_stl", [False, True])
def test_readme_mesh_omission_warning_follows_resolved_inventory(required_stl, included_stl):
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "outputs": ["body"],
            "production": {"role": "make", "process": "cnc"},
        }
    )
    inventory = package_file_inventory(document, {"part.step": "a" * 64}, include_stl=required_stl)
    prose = "Saved original proposal: print make/part.stl only if a print file is supplied."
    context = {
        "status": "stale",
        "facts": {"package_files": inventory},
        "record": {"guidance": {"print": prose}},
    }
    members = set(inventory["metadata_files"]) | {"make/part.step", "assembly/part.step"}
    if included_stl:
        members.add("make/part.stl")
    before = copy.deepcopy((document.model_dump(), context, members))
    text = _package_readme(document, production_report(document), members, context).decode()
    missing_required = required_stl and not included_stl
    assert ("Project geometry omitted from this download:" in text) is missing_required
    assert (
        "Obtain the required files before following the affected steps." in text
    ) is missing_required
    assert ("No STL print file is included." in text) is not included_stl
    assert prose in text
    assert (document.model_dump(), context, members) == before


def with_overall_step(source):
    document, hashes = fixture_bytes(source)
    overall = b"Saved overall STEP, including bought and reference geometry"
    (source / "part.step").write_bytes(overall)
    hashes["part.step"] = hashlib.sha256(overall).hexdigest()
    return document, hashes


def test_complete_project_guide_uses_actual_members_quantities_and_roles(tmp_path):
    source = tmp_path / "saved"
    document, hashes = with_overall_step(source)
    target = manufacture_bundle(
        document,
        source,
        tmp_path / "project.zip",
        hashes,
        include_assembly_step=True,
        include_stl=True,
    )
    verify_manufacturing_handoff(target)
    with ZipFile(target) as archive:
        guide = archive.read("README.md")
        text = guide.decode("utf-8")
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["files"]["README.md"] == {
            "sha256": hashlib.sha256(guide).hexdigest(),
            "size_bytes": len(guide),
        }
        assert "base: make 1 copy" in text
        assert "cap: make 2 copies" in text
        for part in ("base", "cap"):
            for extension in ("step", "stl"):
                name = f"make/definition_{part}.{extension}"
                assert name in text and name in archive.namelist()
        assert "Declared bought part bolt:" in text
        assert "Declared name: bolt\nQuantity: 1" in text
        assert "Declared name: guide\nQuantity: 1" in text
        assert "Do not print these parts or the overall assembly" in text
        assert "make/definition_bolt" not in text
        assert "make/definition_guide" not in text
        assert "coordinates alone do not establish a print orientation" in text
        assert "Physical fit, printing and load capacity are not verified" in text
        assert manifest["manufacturing_ready"] is False


def test_legacy_make_only_default_has_no_new_member(tmp_path):
    source = tmp_path / "saved"
    document, hashes = with_overall_step(source)
    target = manufacture_bundle(document, source, tmp_path / "legacy.zip", hashes)
    with ZipFile(target) as archive:
        assert "README.md" not in archive.namelist()
        assert "assembly/part.step" not in archive.namelist()
    verify_manufacturing_handoff(target)


def test_step_only_project_does_not_invent_mesh_print_file(tmp_path):
    source = tmp_path / "saved"
    document, hashes = with_overall_step(source)
    target = manufacture_bundle(
        document,
        source,
        tmp_path / "project.zip",
        hashes,
        include_assembly_step=True,
    )
    with ZipFile(target) as archive:
        text = archive.read("README.md").decode("utf-8")
        assert "No STL print file is included" in text
        assert "make/definition_base.stl" not in text
        assert "not included" in text  # Required imports are not promised.


def test_notes_are_literal_and_empty_buy_does_not_imply_hardware_unnecessary():
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "holder",
                        "name": "Porte-appareil",
                        "source": "中文\n```\n<script>bad</script>",
                        "features": [{"id": "body", "name": "Body", "kind": "box"}],
                        "output": "body",
                        "production": {"role": "make", "finish": "Only `a` declaration."},
                    }
                ],
                "roots": [{"id": "holder", "definition": "holder"}],
            },
        }
    )
    production = document.assembly.parts[0].production.model_dump()
    line = {"part_id": "holder", "name": "Porte-appareil", "quantity": 1, "production": production}
    report = {
        "make": [line],
        "lines": [line],
        "buy": [],
        "reference": [],
        "unresolved": [],
        "gaps": ["Unknown source: ```\n<script>gap</script>"],
    }
    before = copy.deepcopy(report), document.model_dump()
    text = _package_readme(document, report, {"make/definition_holder.step"}).decode("utf-8")
    assert "No hardware entries are saved in the parts list." in text
    assert "Hardware is undeclared" not in text
    assert "does not mean hardware is unnecessary" in text
    assert "## Unverified saved author notes" in text
    assert "They are not current customer steps." in text
    assert "````text\n中文\n```\n<script>bad</script>\n\nOnly `a` declaration.\n````" in text
    assert "````text\nUnknown source: ```\n<script>gap</script>\n````" in text
    assert (report, document.model_dump()) == before


def test_guide_respects_package_budget_and_preserves_existing_target(tmp_path, monkeypatch):
    source = tmp_path / "saved"
    document, hashes = with_overall_step(source)
    target = tmp_path / "project.zip"
    target.write_bytes(b"prior package")
    monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", 1_000_000)
    monkeypatch.setattr(
        "forge_mind.workspace.production._package_readme", lambda *args: b"x" * 1_000_001
    )
    with pytest.raises(ValueError, match="budget"):
        manufacture_bundle(document, source, target, hashes, include_assembly_step=True)
    assert target.read_bytes() == b"prior package"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["project.zip", "saved"]


def test_guide_tampering_is_rejected_by_existing_member_hash_verifier(tmp_path):
    source = tmp_path / "saved"
    document, hashes = with_overall_step(source)
    target = manufacture_bundle(
        document,
        source,
        tmp_path / "project.zip",
        hashes,
        include_assembly_step=True,
    )
    with ZipFile(target) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    members["README.md"] += b"changed"
    with ZipFile(target, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(target)


def test_material_and_purchasing_use_readable_labels_without_raw_json():
    document = DesignDocument.model_validate(
        {"features": [{"id": "body", "name": "Body", "kind": "box"}]}
    )
    made = {
        "part_id": "part",
        "quantity": 1,
        "production": {"material": {"name": "PLA", "grade": "Example grade"}, "finish": None},
    }
    bought = {
        "part_id": "bolt",
        "name": "M4 screw",
        "quantity": 2,
        "production": {
            "material": None,
            "finish": None,
            "procurement": {
                "supplier": "Example shop",
                "part_number": "EX-4",
                "catalog_id": "example",
            },
        },
    }
    report = {
        "make": [made],
        "buy": [bought],
        "reference": [],
        "unresolved": [],
        "lines": [made, bought],
        "gaps": [],
    }
    text = _package_readme(document, report, {"make/part.step"}).decode("utf-8")
    assert "PLA\nGrade: Example grade" in text
    assert "Declared name: M4 screw\nQuantity: 2" in text
    assert "Supplier: Example shop\nPart number: EX-4\nCatalog ID: example" in text
    assert '"material"' not in text and '"procurement"' not in text
    assert "definition_local" not in text and "world CAD" not in text
