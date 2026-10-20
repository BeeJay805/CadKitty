import json
from zipfile import ZipFile

import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.authoring import AssemblyEdit, apply_assembly_edit
from forge_mind.workspace.document import DesignDocument, PartDefinition
from forge_mind.workspace.hardware import bill_of_materials
from forge_mind.workspace.production import manufacture_bundle, production_report


def part(identifier, production=None):
    value = {
        "id": identifier,
        "name": identifier,
        "features": [
            {"id": "body", "name": "Body", "kind": "box", "params": {"x": 10, "y": 10, "z": 10}}
        ],
        "output": "body",
    }
    if production is not None:
        value["production"] = production
    return value


def assembly():
    return DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    part(
                        "bracket",
                        {
                            "role": "make",
                            "process": "cnc",
                            "material": {"name": "Aluminum", "density_kg_m3": 2700},
                        },
                    ),
                    part(
                        "screw",
                        {
                            "role": "buy",
                            "procurement": {"supplier": "Supplier", "part_number": "M3"},
                            "material": {"name": "Steel", "density_kg_m3": 7800},
                        },
                    ),
                    part("envelope", {"role": "reference"}),
                    part("unused"),
                ],
                "subassemblies": [
                    {
                        "id": "module",
                        "name": "Module",
                        "children": [
                            {"id": "mount", "definition": "bracket"},
                            {"id": "bolt", "definition": "screw"},
                        ],
                    }
                ],
                "roots": [
                    {"id": "a", "definition": "module"},
                    {"id": "b", "definition": "module"},
                    {"id": "space", "definition": "envelope"},
                ],
            }
        }
    )


def test_default_and_omitted_edit_preserve_declaration():
    assert PartDefinition.model_validate(part("x")).production.role == "unspecified"
    document = assembly()
    edited = apply_assembly_edit(
        document, AssemblyEdit.model_validate({"parts": [part("bracket")]})
    )
    assert edited.assembly.parts[0].production == document.assembly.parts[0].production
    reset = apply_assembly_edit(
        document, AssemblyEdit.model_validate({"parts": [part("bracket", {"role": "unspecified"})]})
    )
    assert reset.assembly.parts[0].production.role == "unspecified"


def test_nested_quantities_and_reference_mass():
    facts = {
        "occurrences": [
            {"path": path, "volume_mm3": 1000}
            for path in ["a/mount", "b/mount", "a/bolt", "b/bolt"]
        ]
    }
    report = production_report(assembly(), facts)
    assert [p["quantity"] for p in report["make"]] == [2]
    assert [p["quantity"] for p in report["buy"]] == [2]
    assert report["mass_kg"] == pytest.approx(0.021)
    assert report["setup_status"] == "declared"
    assert report["manufacturing_ready"] is False
    assert not report["unresolved"]
    assert bill_of_materials(assembly())["make"][0]["part_id"] == "bracket"
    assert production_report(assembly())["mass_kg"] is None


def test_bundle_only_explicit_make_and_verified_bytes(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    hashes = {}
    for name in ["definition_bracket.step", "definition_screw.step", "definition_envelope.step"]:
        (source / name).write_text(name)
        hashes[name] = file_sha256(source / name)
    evidence = {"physical_readiness": "unverified", "interfaces": {"records": [{"stale": True}]}}
    target = manufacture_bundle(
        assembly(), source, tmp_path / "parts.zip", hashes, engineering_evidence=evidence
    )
    with ZipFile(target) as archive:
        assert sorted(archive.namelist()) == [
            "document.json",
            "engineering-evidence.json",
            "make/definition_bracket.step",
            "manifest.json",
            "production.json",
            "purchasing.json",
            "sources.json",
        ]
        assert json.loads(archive.read("purchasing.json"))["lines"][0]["quantity"] == 2
        assert json.loads(archive.read("engineering-evidence.json")) == evidence
    linked = assembly()
    linked.assembly.parts[0].production.source_claim_ids = ["missing_claim"]
    with ZipFile(manufacture_bundle(linked, source, tmp_path / "linked.zip", hashes)) as archive:
        report = json.loads(archive.read("production.json"))
        assert report["setup_status"] == "incomplete"
        assert any("missing_claim is missing" in gap for gap in report["gaps"])
    (source / "definition_bracket.step").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        manufacture_bundle(assembly(), source, target, hashes)


def test_missing_role_material_process_never_ready():
    doc = assembly().model_dump()
    doc["assembly"]["parts"][0]["production"] = {"role": "make"}
    doc["assembly"]["parts"][1]["production"] = {}
    report = production_report(DesignDocument.model_validate(doc))
    assert report["setup_status"] == "incomplete"
    assert len(report["gaps"]) == 3
    assert report["mass_kg"] is None


def test_legacy_catalog_is_buy_but_modified_hardware_unresolved():
    doc = DesignDocument.model_validate(
        {
            "features": [
                {"id": "screw", "name": "Screw", "kind": "hardware", "catalog_id": "catalog_screw"}
            ],
            "outputs": ["screw"],
        }
    )
    report = production_report(doc)
    assert report["buy"][0]["role_basis"] == "unchanged legacy catalog geometry"
    assert report["make"] == []
    modified = doc.model_dump()
    modified["features"].extend(
        [
            {"id": "tool", "name": "Tool", "kind": "box"},
            {"id": "cut", "name": "Modified screw", "kind": "cut", "inputs": ["screw", "tool"]},
        ]
    )
    modified["outputs"] = ["cut"]
    assert production_report(DesignDocument.model_validate(modified))["unresolved"]
