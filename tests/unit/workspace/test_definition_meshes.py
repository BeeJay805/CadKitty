"""Pure saved-artifact controls. No CAD geometry or native author execution."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from forge_mind.workspace.definition_meshes import (
    MANIFEST_NAME,
    DefinitionMeshesManifest,
    definition_mesh_entry,
    selected_definition_meshes,
    validate_definition_mesh,
)
from forge_mind.workspace.document import DesignDocument, PartDefinition
from forge_mind.workspace.product_models import PartProduction

REVISION = "a" * 32
STEP = b"bound STEP bytes, not parsed by delivery reads"
STL = b"bound STL bytes, not parsed by delivery reads"
SCOPE = (
    "Numerical export consistency only. Exact dimensions and requirement checks use STEP; "
    "sampled tessellation bounds do not define manufacturing precision."
)


def encoded(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def part(name: str = "body", **production: Any) -> dict[str, Any]:
    return {
        "id": name,
        "name": name,
        "features": [
            {"id": "block", "name": "Block", "kind": "box", "params": {"x": 2, "y": 3, "z": 4}}
        ],
        "output": "block",
        "production": {"role": "make", "deliver_stl": True, **production},
    }


def source() -> dict[str, Any]:
    return {
        "name": "Nominal assembly \u00d7 local millimeters",
        "assembly": {
            "parts": [part()],
            "roots": [{"id": "body_1", "definition": "body"}],
        },
    }


def available(name: str = "body", quantity: int = 1) -> dict[str, Any]:
    return {
        "status": "available",
        "role": "make",
        "quantity": quantity,
        "step_file": f"definition_{name}.step",
        "step_sha256": sha(STEP),
        "stl_file": f"definition_{name}.stl",
        "stl_sha256": sha(STL),
        "stl_bytes": len(STL),
        "body_count": 1,
        "bounds_mm": [[0, 0, 0], [2.125, 3.1234567890123, 4]],
        "tessellation": {
            "linear_deflection": 0.001,
            "linear_deflection_is_relative": True,
            "angular_deflection": 0.1,
            "collapsed_triangles_removed": 2,
        },
        "export_consistency": {
            "volume_within_relative_0_001": True,
            "bounds_within_0_01_mm": True,
            "maximum_sampled_bounds_difference_mm": 0.009876543210123,
            "reference_tessellation": None,
            "scope": SCOPE,
        },
    }


def fixture(
    document: dict[str, Any] | None = None,
    entries: dict[str, dict[str, Any]] | None = None,
) -> tuple[bytes, bytes, dict[str, str], dict[str, Any]]:
    document_bytes = encoded(source() if document is None else document)
    manifest = {
        "schema_version": 1,
        "revision_id": REVISION,
        "document_sha256": sha(document_bytes),
        "frame": "definition_local",
        "units": "mm",
        "parts": {"body": available()} if entries is None else entries,
    }
    manifest_bytes = encoded(manifest)
    hashes = {"document.json": sha(document_bytes), MANIFEST_NAME: sha(manifest_bytes)}
    for entry in manifest["parts"].values():
        if entry["step_sha256"] is not None:
            hashes[entry["step_file"]] = entry["step_sha256"]
        if entry["status"] == "available":
            hashes[entry["stl_file"]] = entry["stl_sha256"]
    return document_bytes, manifest_bytes, hashes, manifest


def changed_manifest(
    original: tuple[bytes, bytes, dict[str, str], dict[str, Any]],
    path: tuple[str, ...],
    value: Any,
) -> tuple[bytes, bytes, dict[str, str]]:
    document, _, hashes, manifest = original
    manifest = deepcopy(manifest)
    cursor = manifest
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    manifest_bytes = encoded(manifest)
    hashes = {**hashes, MANIFEST_NAME: sha(manifest_bytes)}
    return document, manifest_bytes, hashes


def test_exact_binding_and_non_mutation() -> None:
    document, manifest_bytes, hashes, manifest = fixture()
    before = deepcopy((manifest, hashes))
    result = validate_definition_mesh(document, manifest_bytes, REVISION, "body", hashes, STEP, STL)
    assert result == manifest["parts"]["body"]
    assert result["bounds_mm"][1][1] == 3.1234567890123
    assert result["export_consistency"]["maximum_sampled_bounds_difference_mm"] == 0.009876543210123
    result["bounds_mm"][0][0] = 100
    assert (manifest, hashes) == before
    assert (
        definition_mesh_entry(document, manifest_bytes, REVISION, "body", hashes)
        == before[0]["parts"]["body"]
    )


@pytest.mark.parametrize("role", ["buy", "reference", "unspecified"])
def test_opt_in_requires_explicit_make_role(role: str) -> None:
    with pytest.raises(ValueError, match=r"body.*explicitly declared make"):
        PartDefinition.model_validate(part(role=role))
    document, manifest, hashes, _ = fixture(source_with_part(part(role=role)))
    with pytest.raises(ValueError, match=r"body.*explicitly declared make"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


def source_with_part(record: dict[str, Any]) -> dict[str, Any]:
    document = source()
    document["assembly"]["parts"] = [record]
    return document


@pytest.mark.parametrize("flag", [0, 1, "true", "false", None, [], {}])
def test_selection_is_strict_boolean(flag: Any) -> None:
    with pytest.raises(ValidationError):
        PartProduction.model_validate({"role": "make", "deliver_stl": flag})
    document, manifest, hashes, _ = fixture(source_with_part(part(deliver_stl=flag)))
    with pytest.raises(ValueError, match="strict boolean"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


def test_flat_true_rejected_but_historical_default_preserved() -> None:
    historical = {
        "features": [
            {"id": "shape", "name": "Shape", "kind": "box", "params": {"x": 2, "y": 3, "z": 4}}
        ],
        "outputs": ["shape"],
        "production": {"role": "make"},
    }
    before = deepcopy(historical)
    document = DesignDocument.model_validate(historical)
    assert document.production is not None and document.production.deliver_stl is False
    assert selected_definition_meshes(document) == {}
    assert historical == before
    historical["production"]["deliver_stl"] = False
    assert DesignDocument.model_validate(historical).model_dump() == document.model_dump()
    historical["production"]["deliver_stl"] = True
    with pytest.raises(ValueError, match="only on assembly make part definitions"):
        DesignDocument.model_validate(historical)


def test_historical_assembly_missing_flag_stays_unselected(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_mind.workspace import assembly

    monkeypatch.setattr(assembly, "validate_assembly", lambda document: None)
    raw = source()
    del raw["assembly"]["parts"][0]["production"]["deliver_stl"]
    original = encoded(raw)
    model = DesignDocument.model_validate_json(original)
    assert selected_definition_meshes(model) == {}
    assert encoded(raw) == original
    document, manifest, hashes, _ = fixture(raw)
    with pytest.raises(ValueError, match="not selected"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


def test_actual_legacy_hardware_document_keeps_all_six_roles_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge_mind.workspace import assembly

    monkeypatch.setattr(assembly, "validate_assembly", lambda document: None)
    data = (Path(__file__).parent / "fixtures" / "hardware_legacy_document.json").read_bytes()
    assert sha(data) == "6d78a16f29a490211ac4396c875461553efd96b0646aa46e0014ad9ab6980b8d"
    raw = json.loads(data)
    assert all("deliver_stl" not in p["production"] for p in raw["assembly"]["parts"])
    document = DesignDocument.model_validate_json(data)
    assert selected_definition_meshes(document) == {}
    assert document.assembly is not None
    assert {p.id: p.production.role for p in document.assembly.parts} == {
        "base": "make",
        "cap": "make",
        "bearing608": "buy",
        "cap_screw": "buy",
        "washer": "buy",
        "shaft": "reference",
    }
    dump = document.model_dump(mode="json")
    for record in dump["assembly"]["parts"]:
        assert record["production"].pop("deliver_stl") is False
    assert dump == raw


def test_compact_repeated_group_counts_do_not_expand_occurrences() -> None:
    raw = source()
    groups = []
    previous = "body"
    for index in range(36):
        name = f"group_{index}"
        groups.append(
            {
                "id": name,
                "children": [
                    {"id": "left", "definition": previous},
                    {"id": "right", "definition": previous},
                ],
            }
        )
        previous = name
    raw["assembly"].update(subassemblies=groups, roots=[{"id": "root", "definition": previous}])
    document, manifest, hashes, _ = fixture(raw, {"body": available(quantity=2**36)})
    assert definition_mesh_entry(document, manifest, REVISION, "body", hashes)["quantity"] == 2**36


def test_nested_quantities_match_existing_expand_without_native_placements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forge_mind.workspace import assembly

    class Placement:
        def __mul__(self, other: object) -> Placement:
            return self

    monkeypatch.setattr(assembly.b, "Location", Placement)
    monkeypatch.setattr(assembly, "location", lambda instance, values: Placement())
    raw = source()
    raw["assembly"]["parts"].extend(
        [part("unused"), part("purchased", role="buy", deliver_stl=False)]
    )
    raw["assembly"]["subassemblies"] = [
        {
            "id": "pair",
            "name": "pair",
            "children": [
                {"id": "a", "definition": "body"},
                {"id": "b", "definition": "body"},
                {"id": "c", "definition": "purchased"},
            ],
        },
        {"id": "nested", "name": "nested", "children": [{"id": "inside", "definition": "pair"}]},
    ]
    raw["assembly"]["roots"] += [
        {"id": "group_1", "definition": "nested"},
        {"id": "group_2", "definition": "pair"},
    ]
    model = DesignDocument.model_validate(raw)
    counts = Counter(row["definition"] for row in assembly.expand(model))
    assert (
        selected_definition_meshes(model)
        == {"body": counts["body"], "unused": 0}
        == {"body": 5, "unused": 0}
    )
    entries = {"body": available(quantity=5), "unused": available("unused", quantity=0)}
    document, manifest, hashes, _ = fixture(raw, entries)
    # A package containing used definitions need not carry unused-definition artifact hashes.
    hashes.pop("definition_unused.step")
    hashes.pop("definition_unused.stl")
    assert definition_mesh_entry(document, manifest, REVISION, "body", hashes)["quantity"] == 5
    with pytest.raises(ValueError, match="stale STEP measurement hash"):
        definition_mesh_entry(document, manifest, REVISION, "unused", hashes)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "Missing assembly definition"),
        ("duplicate", "duplicate document root"),
        ("cycle", "Cyclic assembly definition"),
        ("definition_collision", "definition IDs must be unique"),
    ],
)
def test_hierarchy_guards(mutation: str, match: str) -> None:
    raw = source()
    spec = raw["assembly"]
    if mutation == "missing":
        spec["roots"][0]["definition"] = "absent"
    elif mutation == "duplicate":
        spec["roots"] *= 2
    else:
        group = "body" if mutation == "definition_collision" else "loop"
        spec["subassemblies"] = [{"id": group, "children": [{"id": "again", "definition": group}]}]
        spec["roots"] = [{"id": "first", "definition": group}]
    document, manifest, hashes, _ = fixture(raw)
    with pytest.raises(ValueError, match=match):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


@pytest.mark.parametrize(
    "reference", [None, {"triangle_inventory_matches": False, "scope": "observed"}]
)
def test_disagreeing_bounds_requires_real_matching_reference(reference: Any) -> None:
    fields = available()
    fields["export_consistency"].update(
        bounds_within_0_01_mm=False, reference_tessellation=reference
    )
    document, manifest, hashes, _ = fixture(entries={"body": fields})
    with pytest.raises(ValueError, match="passed measured export-consistency"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


def test_curved_sampled_bounds_existing_reference_fallback() -> None:
    fields = available()
    fields["export_consistency"].update(
        bounds_within_0_01_mm=False,
        maximum_sampled_bounds_difference_mm=0.071,
        reference_tessellation={
            "triangle_inventory_matches": True,
            "scope": "Original exact recorded numerical scope",
        },
    )
    document, manifest, hashes, _ = fixture(entries={"body": fields})
    assert (
        validate_definition_mesh(document, manifest, REVISION, "body", hashes, STEP, STL) == fields
    )


def test_full_reference_settings_are_preserved_and_checked() -> None:
    fields = available()
    reference = {
        "triangle_inventory_matches": True,
        "linear_deflection_setting": 0.001,
        "linear_deflection_is_relative": True,
        "angular_deflection_rad": 0.1,
        "scope": "Existing measured scope",
    }
    fields["export_consistency"].update(
        bounds_within_0_01_mm=False, reference_tessellation=reference
    )
    document, manifest, hashes, _ = fixture(entries={"body": fields})
    assert definition_mesh_entry(document, manifest, REVISION, "body", hashes) == fields
    reference["linear_deflection_setting"] = 0.002
    document, manifest, hashes, _ = fixture(entries={"body": fields})
    with pytest.raises(ValueError, match="settings differ"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("schema_version",), 2),
        (("extra",), "unexpected"),
        (("frame",), "assembly_world"),
        (("units",), "inch"),
        (("parts", "body", "quantity"), True),
        (("parts", "body", "quantity"), 2),
        (("parts", "body", "body_count"), 0),
        (("parts", "body", "stl_bytes"), 0),
        (("parts", "body", "bounds_mm"), [[2, 0, 0], [1, 3, 4]]),
        (("parts", "body", "bounds_mm"), [[0, 0], [1, 3, 4]]),
        (("parts", "body", "bounds_mm"), [[0, 0, 0], [float("nan"), 3, 4]]),
        (("parts", "body", "body_count"), 1.5),
        (("parts", "body", "role"), "buy"),
        (("parts", "body", "stl_file"), "../escape.stl"),
        (("parts", "body", "step_file"), "part.step"),
        (("parts", "body", "extra"), 1),
        (("parts", "body", "tessellation", "linear_deflection"), float("inf")),
        (("parts", "body", "tessellation", "angular_deflection"), 0),
        (("parts", "body", "tessellation", "linear_deflection_is_relative"), 1),
        (("parts", "body", "tessellation", "collapsed_triangles_removed"), -1),
        (("parts", "body", "export_consistency", "volume_within_relative_0_001"), False),
        (("parts", "body", "export_consistency", "volume_within_relative_0_001"), 1),
        (
            ("parts", "body", "export_consistency", "maximum_sampled_bounds_difference_mm"),
            float("-inf"),
        ),
        (("parts", "body", "export_consistency", "extra"), 1),
    ],
)
def test_rejects_invalid_saved_schema(path: tuple[str, ...], value: Any) -> None:
    document, manifest, hashes = changed_manifest(fixture(), path, value)
    with pytest.raises(ValueError):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)


@pytest.mark.parametrize(
    "artifact", ["document.json", MANIFEST_NAME, "definition_body.step", "definition_body.stl"]
)
def test_refuses_missing_or_stale_requested_artifact_hash(artifact: str) -> None:
    document, manifest, hashes, _ = fixture()
    for replacement in (None, "b" * 64, "not-a-hash"):
        altered = dict(hashes)
        if replacement is None:
            altered.pop(artifact)
        else:
            altered[artifact] = replacement
        with pytest.raises(ValueError):
            definition_mesh_entry(document, manifest, REVISION, "body", altered)


def test_refuses_revision_source_bytes_export_bytes_and_size_changes() -> None:
    document, manifest, hashes, _ = fixture()
    with pytest.raises(ValueError, match="revision or document binding"):
        definition_mesh_entry(document, manifest, "b" * 32, "body", hashes)
    for altered_step, altered_stl in [(STEP + b"x", STL), (STEP, STL + b"x")]:
        with pytest.raises(ValueError, match="Stale revision artifact bytes"):
            validate_definition_mesh(
                document, manifest, REVISION, "body", hashes, altered_step, altered_stl
            )
    with pytest.raises(ValueError, match="Stale revision artifact bytes"):
        definition_mesh_entry(document + b" ", manifest, REVISION, "body", hashes)
    document, manifest, hashes = changed_manifest(
        fixture(), ("parts", "body", "stl_bytes"), len(STL) + 1
    )
    with pytest.raises(ValueError, match="stale STL byte count"):
        validate_definition_mesh(document, manifest, REVISION, "body", hashes, STEP, STL)


def test_manifest_complete_and_legacy_unavailable() -> None:
    document, manifest, hashes = changed_manifest(fixture(), ("parts",), {})
    with pytest.raises(ValueError, match="every selected"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)
    with pytest.raises(ValueError, match="unavailable"):
        definition_mesh_entry(document, b"", REVISION, "body", hashes)
    document, manifest, hashes, _ = fixture()
    with pytest.raises(ValueError, match="not selected"):
        definition_mesh_entry(document, manifest, REVISION, "unknown", hashes)


def test_failed_entry_beside_available_is_explicit() -> None:
    raw = source()
    raw["assembly"]["parts"].append(part("failed"))
    raw["assembly"]["roots"].append({"id": "failed_1", "definition": "failed"})
    failed = {
        "status": "failed",
        "role": "make",
        "quantity": 1,
        "step_file": "definition_failed.step",
        "step_sha256": None,
        "error": "Actual export error \u00d7 retained",
    }
    document, manifest, hashes, saved = fixture(raw, {"body": available(), "failed": failed})
    assert DefinitionMeshesManifest.model_validate(saved).parts["failed"].status == "failed"
    assert definition_mesh_entry(document, manifest, REVISION, "body", hashes) == available()
    with pytest.raises(ValueError, match="Actual export error \u00d7 retained"):
        definition_mesh_entry(document, manifest, REVISION, "failed", hashes)
    for invalid in ("", " ", "x" * 2001):
        document, manifest, hashes = changed_manifest(
            fixture(raw, {"body": available(), "failed": failed}),
            ("parts", "failed", "error"),
            invalid,
        )
        with pytest.raises(ValueError):
            definition_mesh_entry(document, manifest, REVISION, "body", hashes)


def test_duplicate_json_and_nonfinite_source_rejected() -> None:
    document, manifest, hashes, _ = fixture()
    duplicate = b'{"assembly":{},"assembly":{}}'
    with pytest.raises(ValueError, match="Duplicate Document key"):
        definition_mesh_entry(
            duplicate, manifest, REVISION, "body", {**hashes, "document.json": sha(duplicate)}
        )
    raw = source()
    raw["variables"] = {"invalid": float("nan")}
    document, manifest, hashes, _ = fixture(raw)
    with pytest.raises(ValueError, match="nonfinite"):
        definition_mesh_entry(document, manifest, REVISION, "body", hashes)
