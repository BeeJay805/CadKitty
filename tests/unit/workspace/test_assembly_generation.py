from __future__ import annotations

import json
from pathlib import Path

import build123d as b
import pytest

from forge_mind.workspace.assembly import assembly_evidence, expand, preserved_assembly
from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
    Subassembly,
)
from forge_mind.workspace.hardware import bill_of_materials
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.mechanism_geometry import bind_exported
from forge_mind.workspace.steering_template import SteeringTemplate, steering_document
from forge_mind.workspace.topology import same_exported_solid, topology_index


def simple() -> DesignDocument:
    return DesignDocument(
        variables={"width": 10},
        assembly=AssemblySpec(
            parts=[
                PartDefinition(
                    id="block",
                    name="Shared block",
                    features=[
                        Feature(
                            id="body",
                            name="Body",
                            kind="box",
                            params={"y": 5, "z": 4},
                            bindings={"x": "width"},
                        )
                    ],
                    output="body",
                ),
            ],
            subassemblies=[
                Subassembly(
                    id="pair",
                    name="Two blocks",
                    children=[
                        Occurrence(id="a", definition="block"),
                        Occurrence(id="b", definition="block", position=(20, 0, 0)),
                    ],
                )
            ],
            roots=[
                Occurrence(id="left", definition="pair"),
                Occurrence(id="right", definition="pair", position=(0, 30, 0)),
            ],
        ),
    )


def test_reusable_nested_instances_roundtrip_and_edit(tmp_path: Path) -> None:
    doc = simple()
    source, history = compile_document(doc, {}, tmp_path)
    b.export_step(source, tmp_path / "assembly.step")
    shape = b.import_step(tmp_path / "assembly.step")
    evidence = assembly_evidence(doc, shape, history[-1])
    assert evidence["occurrence_count"] == 4
    assert evidence["unique_parts"] == 1
    assert len({r["body_index"] for r in evidence["occurrences"]}) == 4
    assert evidence["bom"][0]["quantity"] == 4
    assert bill_of_materials(doc)["part_definitions"][0]["quantity"] == 4
    doc.variables["width"] = 12
    edited, _ = compile_document(doc, {})
    assert [round(s.volume) for s in edited.solids()] == [240] * 4
    assert len(expand(doc)) == 4


def test_part_cache_reuses_only_unchanged_geometry(tmp_path: Path) -> None:
    first, second, third = (tmp_path / name for name in ("first", "second", "third"))
    for directory in (first, second, third):
        directory.mkdir()
    doc = simple()
    compile_document(doc, {}, first)
    _, history = compile_document(doc, {}, second, first)
    assert history[-1]["reused"] == 1
    doc.variables["width"] = 12
    _, history = compile_document(doc, {}, third, second)
    assert history[-1]["built"] == 1
    cached = json.loads((first / "parts.json").read_text())["block"]
    (first / cached["file"]).write_bytes(b"tampered")
    _, history = compile_document(simple(), {}, second, first)
    assert history[-1]["built"] == 1


def test_placement_change_reuses_definition(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    doc = simple()
    compile_document(doc, {}, first)
    assert doc.assembly
    doc.assembly.roots[1].position = (0, 60, 0)
    result, history = compile_document(doc, {}, second, first)
    assert history[-1]["reused"] == 1
    assert pytest.approx(62.5) == result.bounding_box().max.Y


def test_missing_cycle_and_expansion_limits_are_rejected() -> None:
    raw = simple().model_dump()
    raw["assembly"]["subassemblies"][0]["children"][0]["definition"] = "pair"
    with pytest.raises(ValueError, match="cyclic"):
        DesignDocument.model_validate(raw)
    raw = simple().model_dump()
    raw["assembly"]["roots"][0]["definition"] = "missing"
    with pytest.raises(ValueError, match="Missing"):
        DesignDocument.model_validate(raw)
    raw = simple().model_dump()
    raw["assembly"]["subassemblies"][0]["children"] *= 65
    with pytest.raises(ValueError):
        DesignDocument.model_validate(raw)


def test_geometry_replacement_and_occurrence_reassignment_rejected() -> None:
    baseline = simple().model_dump()
    assert not preserved_assembly(baseline, DesignDocument().model_dump())[0]
    changed = simple()
    assert changed.assembly
    changed.assembly.roots.pop()
    assert not preserved_assembly(baseline, changed.model_dump())[0]


def test_partner_identity_also_requires_same_placement() -> None:
    source = b.Box(1, 2, 3)
    assert same_exported_solid(source, source)
    assert not same_exported_solid(source, source.moved(b.Location((0, 0, 1))))
    assert not same_exported_solid(source, b.Box(1, 2, 3))


def test_steering_export_ports_and_every_moving_body_bound(tmp_path: Path) -> None:
    doc = steering_document(SteeringTemplate())
    shape, history = compile_document(doc, {})
    b.export_step(shape, tmp_path / "steering.step")
    imported = b.import_step(tmp_path / "steering.step")
    evidence = assembly_evidence(doc, imported, history[-1])
    assert evidence["occurrence_count"] == 31
    assert all(p["status"] == "pass" for p in evidence["ports"].values())
    assert all(m["status"] == "pass" for m in evidence["mates"])
    assert evidence["status"] == "unverified"
    motion = bind_exported(
        history[-2], topology_index(imported, assembly=True), "step", "doc", imported
    )
    assert len(motion["bindings"]) == 31
    assert len({r["body_index"] for r in motion["bindings"]}) == 31
    assert all(s["status"] == "solved" for s in motion["samples"])
    assert doc.assembly
    doc.assembly.parts[0].ports[0].diameter = 999
    invalid = assembly_evidence(doc, imported, history[-1])
    assert invalid["ports"]["left/wheel/tire.bore"]["status"] == "fail"
    assert invalid["status"] == "fail"


def test_two_independent_steering_sizes_remain_solved() -> None:
    from forge_mind.workspace.mechanism import resolve_mechanism, study_mechanism

    for track, wheel in [(480, 160), (800, 260)]:
        doc = steering_document(SteeringTemplate(track_width=track, wheel_diameter=wheel))
        assert doc.mechanism
        study = study_mechanism(resolve_mechanism(doc.mechanism, doc.values()))
        assert all(s["status"] == "solved" for s in study["samples"])


def test_assembly_constraint_weakening_and_rewiring_are_refused() -> None:
    doc = steering_document(SteeringTemplate())
    baseline = doc.model_dump()
    assert doc.assembly
    doc.assembly.mates[0].max_clearance = 10
    assert not preserved_assembly(baseline, doc.model_dump())[0]
    doc = DesignDocument.model_validate(baseline)
    assert doc.assembly
    doc.assembly.mates[0].b = "left/rod_pin.seat"
    assert not preserved_assembly(baseline, doc.model_dump())[0]


def test_definition_lock_tracks_bound_geometry_not_other_parts() -> None:
    from forge_mind.workspace.document_service import _closure

    doc = steering_document(SteeringTemplate())
    original = doc.model_dump()
    doc.variables["wheel_diameter"] = 240
    assert _closure(original, "axle") == _closure(doc.model_dump(), "axle")
    assert _closure(original, "wheel") != _closure(doc.model_dump(), "wheel")


def test_occurrence_encoding_is_injective_and_root_prefix_is_safe(tmp_path: Path) -> None:
    doc = simple()
    assert doc.assembly
    doc.assembly.roots = [
        Occurrence(id="a", definition="pair"),
        Occurrence(id="a__a", definition="block", position=(0, 40, 0)),
        Occurrence(id="ck_occ_user", definition="pair", position=(0, 80, 0)),
    ]
    shape, history = compile_document(doc, {})
    b.export_step(shape, tmp_path / "named.step")
    result = assembly_evidence(doc, b.import_step(tmp_path / "named.step"), history[-1])
    assert result["occurrence_count"] == 5


def test_eighty_instances_preserve_quantities_and_selection_budget() -> None:
    doc = simple()
    assert doc.assembly
    doc.assembly.roots = [
        Occurrence(id=f"pair_{i}", definition="pair", position=(0, i * 12, 0)) for i in range(40)
    ]
    shape, _ = compile_document(doc, {})
    assert len(shape.solids()) == 80
    topology = topology_index(shape, assembly=True)
    assert len([e for e in topology["entities"] if e["kind"] == "body"]) == 80
    assert bill_of_materials(doc)["part_definitions"][0]["quantity"] == 80


def test_all_body_interference_finds_known_overlapping_pair() -> None:
    from forge_mind.workspace.assembly_review import assembly_motion_review

    shape = b.Compound(children=[b.Box(10, 10, 10), b.Pos(5, 0, 0) * b.Box(10, 10, 10)])
    pose = {"status": "solved", "angle_deg": 0, "poses": {"frame": [0, 0, 0, 0]}}
    motion = {
        "range_deg": [-10, 10],
        "current": pose,
        "samples": [],
        "bindings": [{"body_index": i, "component": "frame"} for i in range(2)],
        "graph": {"components": []},
    }
    # No generated poses needed for a rigid pair: the current exported state is a witness.
    motion["current"] = {**pose, "status": "unsolved"}
    assert (
        assembly_motion_review(
            shape, motion, {"occurrences": [{"body_index": i, "path": str(i)} for i in range(2)]}
        )["coverage"]
        == "incomplete"
    )
    from unittest.mock import patch

    motion["current"] = pose
    with patch("forge_mind.workspace.assembly_review.contact_poses", return_value=[]):
        report = assembly_motion_review(
            shape, motion, {"occurrences": [{"body_index": i, "path": str(i)} for i in range(2)]}
        )
    assert report["status"] == "fail"
    assert report["collisions"][0]["overlap_mm3"] == pytest.approx(500)


def test_nested_volume_measurement_uses_all_exported_solids(tmp_path: Path) -> None:
    from forge_mind.workspace.measure import measure_revision
    from forge_mind.workspace.models import Contract

    doc = simple()
    shape, _ = compile_document(doc, {})
    b.export_step(shape, tmp_path / "part.step")
    b.export_stl(shape, tmp_path / "part.stl")
    measured = measure_revision(tmp_path, "a" * 32, Contract())
    assert measured.facts["volume_mm3"] == pytest.approx(800)
    assert measured.facts["export_agrees"] is True


def test_definition_lock_tracks_port_only_variable() -> None:
    from forge_mind.workspace.document_service import _closure

    doc = steering_document(SteeringTemplate())
    assert doc.assembly
    doc.variables["interface_d"] = 10
    doc.assembly.parts[0].ports[0].diameter = "interface_d"
    original = doc.model_dump()
    doc.variables["interface_d"] = 20
    assert _closure(original, "wheel") != _closure(doc.model_dump(), "wheel")


def test_assembly_context_preserves_failures_without_transform_repetition() -> None:
    from forge_mind.workspace.agent import assembly_measurement_context

    original = {
        "errors": ["failure"],
        "facts": {
            "assembly": {"status": "fail"},
            "feature_history": ["verbose"],
            "mechanism": {
                "bindings": [{"body": 0}],
                "samples": [
                    {"status": "unsolved", "angle_deg": 20, "error": "singular", "poses": {"a": []}}
                ],
            },
        },
    }
    compact = assembly_measurement_context(original)
    assert compact["errors"] == ["failure"]
    assert compact["facts"]["mechanism"]["sample_summary"][0]["error"] == "singular"
    assert "feature_history" not in compact["facts"]
    assert "feature_history" in original["facts"]
