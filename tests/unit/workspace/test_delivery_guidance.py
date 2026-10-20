"""Saved declaration/package controls; never build or execute authored geometry."""

from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest
from pydantic import ValidationError
from tests.unit.workspace.test_mesh_handoff import fixture_bytes

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.handoff import json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.product_models import DeliveryGuidance
from forge_mind.workspace.production import _package_readme, manufacture_bundle, production_report

FIELDS = ("print", "installation", "use_and_removal", "unresolved")
HISTORICAL_FLAT_HASH = "b732d3c08aacd778e7bf1749510ae13a522b95774c7d21567199d436cf5b9f29"
GUIDANCE = {
    "print": "Print the stated mounting face on the bed. 中文\n```\n<script>literal</script>",
    "installation": "Use the listed bolt parts. Wall anchors are not selected until wall type is known.",
    "use_and_removal": "Lift the device out; actual hand/cable access is unverified.",
    "unresolved": "Physical fit, load and filament grade remain untested.",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def historical_flat_document():
    # Explicit pre-field default expansion: not produced by the current serializer.
    return {
        "production": None,
        "version": 1,
        "name": "Untitled design",
        "units": "mm",
        "variables": {},
        "relations": [],
        "features": [
            {
                "id": "body",
                "name": "Body",
                "kind": "box",
                "inputs": [],
                "params": {},
                "bindings": {},
                "points": [],
                "code": "",
                "sketch": None,
                "asset_id": None,
                "catalog_id": None,
                "selection": [],
                "suppressed": False,
            }
        ],
        "outputs": [],
        "mechanism": None,
        "motion_rig": None,
        "assembly": None,
    }


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize(
    "invalid", ["", "x" * 4001, True, 1, 1.5, float("nan"), float("inf"), b"text", [], {}]
)
def test_guidance_refuses_wrong_type_or_length(field, invalid):
    with pytest.raises(ValidationError):
        DeliveryGuidance.model_validate({field: invalid})


def test_optional_strict_guidance_boundaries_and_no_extra_fields():
    assert DeliveryGuidance.model_validate({}).model_dump() == dict.fromkeys(FIELDS)
    assert DeliveryGuidance.model_validate(
        dict.fromkeys(FIELDS, None)
    ).model_dump() == dict.fromkeys(FIELDS)
    assert DeliveryGuidance.model_validate(dict.fromkeys(FIELDS, "字" * 4000)).print == "字" * 4000
    with pytest.raises(ValidationError):
        DeliveryGuidance.model_validate({"hardware": []})


def test_legacy_expanded_serialization_and_historical_hash_remain_exact():
    raw = historical_flat_document()
    before = copy.deepcopy(raw)
    document = DesignDocument.model_validate(raw)
    assert document.model_dump() == raw
    assert json.loads(document.model_dump_json()) == raw
    assert digest(raw) == HISTORICAL_FLAT_HASH
    assert document.fingerprint() == HISTORICAL_FLAT_HASH
    assert "delivery_guidance" not in document.model_dump(mode="json")
    assert raw == before
    null = DesignDocument.model_validate({**raw, "delivery_guidance": None})
    assert null.fingerprint() == document.fingerprint()
    assert null.model_dump() == raw


def test_wrap_serializer_preserves_include_exclude_and_supplied_empty_object():
    document = DesignDocument.model_validate(
        {**historical_flat_document(), "delivery_guidance": {}}
    )
    assert document.model_dump(include={"delivery_guidance"}) == {
        "delivery_guidance": dict.fromkeys(FIELDS)
    }
    assert document.model_dump(exclude={"delivery_guidance"}) == historical_flat_document()
    assert document.model_dump(exclude_none=True)["delivery_guidance"] == {}
    assert document.model_dump(mode="json")["delivery_guidance"] == dict.fromkeys(FIELDS)
    assert document.fingerprint() != digest(historical_flat_document())


def saved_assembly(source, guidance=None):
    document, hashes = fixture_bytes(source)
    if guidance is not None:
        raw = json.loads((source / "document.json").read_bytes())
        raw["delivery_guidance"] = guidance
        content = json_bytes(raw)
        (source / "document.json").write_bytes(content)
        hashes["document.json"] = sha(content)
        mesh = json.loads((source / "definition-meshes.json").read_bytes())
        mesh["document_sha256"] = sha(content)
        mesh_bytes = json_bytes(mesh)
        (source / "definition-meshes.json").write_bytes(mesh_bytes)
        hashes["definition-meshes.json"] = sha(mesh_bytes)
        document = DesignDocument.model_validate(raw)
    overall = b"Immutable saved assembly STEP including reference and bought parts"
    (source / "part.step").write_bytes(overall)
    hashes["part.step"] = sha(overall)
    return document, hashes


def read_members(path):
    with ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


@pytest.mark.parametrize("assembly", [False, True])
def test_flat_and_assembly_guidance_is_literal_and_does_not_change_production(assembly, tmp_path):
    document = (
        saved_assembly(tmp_path / "source")[0]
        if assembly
        else DesignDocument.model_validate(historical_flat_document())
    )
    report = production_report(document)
    old = document.model_dump(), copy.deepcopy(report)
    legacy = _package_readme(document, report, set())
    document.delivery_guidance = DeliveryGuidance.model_validate(GUIDANCE)
    guide = _package_readme(document, report, set()).decode()
    assert "## Customer guidance" not in legacy.decode()
    for value in GUIDANCE.values():
        assert value in guide
    assert "````text\n" + GUIDANCE["print"] + "\n````" in guide
    assert production_report(document) == old[1]
    assert document.model_dump(exclude={"delivery_guidance"}) == old[0]
    assert report == old[1]
    if not report["buy"]:
        assert "No hardware entries are saved in the parts list." in guide
        assert "Hardware requirements may appear in the saved instructions." in guide
        assert "Hardware is undeclared" not in guide
    else:
        assert "Declared bought part bolt:" in guide
    assert "Physical fit, printing and load capacity are not verified." in guide


def test_supplied_empty_or_partial_object_explicitly_lists_missing_sections():
    document = DesignDocument.model_validate(
        {**historical_flat_document(), "delivery_guidance": {}}
    )
    guide = _package_readme(document, production_report(document), set()).decode()
    assert guide.count("Not supplied.") == 4
    for label in (
        "Printing:",
        "Installation:",
        "Use and removal:",
        "Unresolved choices and checks:",
    ):
        assert label in guide
    document.delivery_guidance.installation = "Wall substrate is unknown."
    guide = _package_readme(document, production_report(document), set()).decode()
    assert guide.count("Not supplied.") == 3
    assert "Wall substrate is unknown." in guide


def test_legacy_package_accepts_pre_field_default_expanded_source_and_hash(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    raw = historical_flat_document()
    content = json_bytes(raw)
    step = b"Retained pre-field overall STEP"
    (source / "part.step").write_bytes(step)
    target = manufacture_bundle(
        DesignDocument.model_validate(raw),
        source,
        tmp_path / "legacy.zip",
        {"part.step": sha(step)},
        include_assembly_step=True,
    )
    members = read_members(target)
    assert members["document.json"] == content
    assert json.loads(members["manifest.json"])["document_hash"] == HISTORICAL_FLAT_HASH
    assert b"## Customer guidance" not in members["README.md"]
    verify_manufacturing_handoff(target)


def test_guidance_package_preserves_exact_raw_source_and_geometry_with_stl(tmp_path):
    source = tmp_path / "source"
    document, hashes = saved_assembly(source, GUIDANCE)
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    target = manufacture_bundle(
        document,
        source,
        tmp_path / "project.zip",
        hashes,
        include_assembly_step=True,
        include_stl=True,
    )
    members = read_members(target)
    assert members["document.json"] == original["document.json"]
    assert json.loads(members["document.json"])["delivery_guidance"] == GUIDANCE
    for part in ("base", "cap"):
        for extension in ("step", "stl"):
            name = f"definition_{part}.{extension}"
            assert members[f"make/{name}"] == original[name]
    assert members["assembly/part.step"] == original["part.step"]
    assert json.loads(members["purchasing.json"])["lines"][0]["quantity"] == 1
    assert "cap: make 2 copies" in members["README.md"].decode()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original
    verify_manufacturing_handoff(target)
    members["README.md"] += b"changed guide"
    with ZipFile(target, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError):
        verify_manufacturing_handoff(target)


def test_guidance_budget_failure_preserves_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    document, hashes = saved_assembly(source)
    baseline = manufacture_bundle(
        document, source, tmp_path / "baseline.zip", hashes, include_assembly_step=True
    )
    ordinary_size = sum(
        len(value) for name, value in read_members(baseline).items() if name != "manifest.json"
    )
    monkeypatch.setattr("forge_mind.workspace.processing.MAX_JOB_BYTES", ordinary_size + 1000)
    document.delivery_guidance = DeliveryGuidance.model_validate(dict.fromkeys(FIELDS, "字" * 4000))
    target = tmp_path / "kept.zip"
    target.write_bytes(b"Previous usable package")
    with pytest.raises(ValueError, match="budget"):
        manufacture_bundle(document, source, target, hashes, include_assembly_step=True)
    assert target.read_bytes() == b"Previous usable package"
    assert not list(tmp_path.glob("*.tmp"))
