from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import build123d as b
import pytest

from forge_mind.workspace.assembly import assembly_evidence, occurrence_label
from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
    PartPort,
)
from forge_mind.workspace.occurrence_facts import occurrence_facts


def fixture_measurement() -> dict[str, Any]:
    surface = {
        "surface_id": "cylinder_0",
        "radius_mm": 6.3,
        "diameter_mm": 12.6,
        "axis_point_mm": [0, 0, 0],
        "axis_direction": [0, 0, 1],
        "concave": True,
        "axial_extent_mm": [0.6, 17.4],
    }
    port = {
        "status": "fail",
        "kind": "bore",
        "diameter_mm": 12.6,
        "origin": [0, 0, 0],
        "axis": [0, 0, 1],
        "method": "retained matcher",
    }
    occurrence = {
        "path": "head",
        "definition": "part",
        "name": "Head half",
        "body_index": 0,
        "volume_mm3": 100,
        "center_mm": [0, 0, 9],
        "bbox_min_mm": [-10, -10, 0.6],
        "bbox_max_mm": [10, 10, 17.4],
        "cylindrical_surfaces": [surface],
    }
    return {
        "revision_id": "a" * 32,
        "facts": {
            "assembly": {
                "occurrences": [occurrence],
                "ports": {"head.bore": port},
                "mates": [],
                "status": "fail",
            }
        },
    }


def test_axial_datum_explains_failure_without_changing_status_or_input() -> None:
    measurement = fixture_measurement()
    before = deepcopy(measurement)
    view = occurrence_facts(measurement, "head")
    port = view["ports"][0]
    comparison = port["surface_comparisons"][0]
    assert comparison["axial_origin_projection_mm"] == 0
    assert comparison["axial_outside_distance_mm"] == 0.6
    assert comparison["predicate_matches"] == {
        "material_side": True,
        "diameter": True,
        "axis_parallel": True,
        "axis_radial": True,
        "axial_origin": False,
    }
    assert port["declaration"]["status"] == "fail"
    assert port["matching_surface_count"] == 0
    view["cylindrical_surfaces"][0]["axis_point_mm"][0] = 999
    assert measurement == before


def test_native_rotated_translated_surface_and_port_span(tmp_path: Path) -> None:
    part = PartDefinition(
        id="part",
        name="Ring",
        features=[Feature(id="body", name="Body", kind="box", params={"x": 1, "y": 1, "z": 1})],
        output="body",
        ports=[
            PartPort(name="bad", kind="bore", diameter=12.6),
            PartPort(name="good", kind="bore", diameter=12.6, origin=(0, 0, 2)),
        ],
    )
    doc = DesignDocument(
        assembly=AssemblySpec(
            parts=[part],
            roots=[
                Occurrence(id="head", definition="part", position=(30, 40, 50), rotation=(0, 90, 0))
            ],
        )
    )
    ring = b.Pos(0, 0, 9) * b.Cylinder(10, 16.8) - b.Pos(0, 0, 9) * b.Cylinder(6.3, 18)
    ring = ring.moved(b.Location((30, 40, 50), (0, 90, 0)))
    ring.label = occurrence_label("head")
    step = tmp_path / "ring.step"
    b.export_step(b.Compound(children=[ring]), step)
    evidence = assembly_evidence(doc, b.import_step(step), {"built": 1, "reused": 0})
    assert evidence["ports"]["head.bad"]["status"] == "fail"
    assert evidence["ports"]["head.good"]["status"] == "pass"
    view = occurrence_facts({"facts": {"assembly": evidence}}, "head")
    assert view["frame"] == "cad_z_up_mm_world"
    bore = next(row for row in view["cylindrical_surfaces"] if row["concave"] is True)
    assert bore["axis_direction"] == pytest.approx([1, 0, 0])
    assert bore["axial_extent_mm"] == pytest.approx([30.6, 47.4])
    by_port = {row["port_id"]: row for row in view["ports"]}
    bad = next(
        row
        for row in by_port["head.bad"]["surface_comparisons"]
        if row["surface_id"] == bore["surface_id"]
    )
    assert bad["axial_outside_distance_mm"] == pytest.approx(0.6)
    assert by_port["head.good"]["matching_surface_count"] == 1
    assert evidence["status"] == "fail"


def test_unknown_material_side_and_extent_are_not_invented() -> None:
    data = fixture_measurement()
    surface = data["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"][0]
    surface.update(concave=None, axial_extent_mm=None)
    view = occurrence_facts(data, "head")
    comparison = view["ports"][0]["surface_comparisons"][0]
    assert comparison["material_side_known"] is False
    assert comparison["axial_extent_known"] is False
    assert comparison["axial_outside_distance_mm"] is None
    assert comparison["all_existing_predicates_match"] is False
    assert view["cylindrical_surfaces"][0]["concave"] is None


def test_material_radius_axis_and_radial_diagnostics_are_independent() -> None:
    data = fixture_measurement()
    surface = data["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"][0]
    surface.update(
        radius_mm=7,
        diameter_mm=14,
        axis_point_mm=[0, 2, 0],
        axis_direction=[1, 0, 0],
        concave=False,
    )
    comparison = occurrence_facts(data, "head")["ports"][0]["surface_comparisons"][0]
    assert comparison["radius_delta_mm"] == pytest.approx(0.7)
    assert comparison["diameter_error_mm"] == pytest.approx(1.4)
    assert comparison["axis_cross_norm"] == 1
    assert comparison["axis_radial_error_mm"] == 2
    assert not any(
        comparison["predicate_matches"][key]
        for key in ("material_side", "diameter", "axis_parallel", "axis_radial")
    )


def test_pages_keep_all_ports_and_global_multiple_face_count() -> None:
    data = fixture_measurement()
    assembly = data["facts"]["assembly"]
    original = assembly["occurrences"][0]["cylindrical_surfaces"][0]
    assembly["occurrences"][0]["cylindrical_surfaces"] = [
        {**original, "surface_id": f"cylinder_{i}"} for i in range(3)
    ]
    assembly["ports"]["head.bore"]["origin"] = [0, 0, 1]
    assembly["ports"]["head.other"] = deepcopy(assembly["ports"]["head.bore"])
    assembly["ports"]["other.bore"] = deepcopy(assembly["ports"]["head.bore"])
    view = occurrence_facts(data, "head", surface_offset=1, surface_limit=1)
    assert (
        view["total"],
        view["returned_count"],
        view["next_offset"],
        view["omitted_surface_count"],
    ) == (3, 1, 2, 2)
    assert len(view["ports"]) == 2
    assert view["ports"][0]["matching_surface_count"] == 3
    assert view["ports"][0]["multiple_matching_surfaces"] is True
    assert view["ports"][0]["declaration"]["status"] == "fail"
    assert view["cylindrical_surfaces"][0]["surface_id"] == "cylinder_1"
    assert occurrence_facts(data, "head", surface_offset=20)["next_offset"] is None


def test_no_cylinders_distinct_from_old_missing_saved_data() -> None:
    data = fixture_measurement()
    occurrence = data["facts"]["assembly"]["occurrences"][0]
    occurrence["cylindrical_surfaces"] = []
    empty = occurrence_facts(data, "head")
    assert empty["availability"] == "available"
    assert empty["total"] == 0
    assert empty["ports"][0]["matching_surface_count"] == 0
    del occurrence["cylindrical_surfaces"]
    legacy = occurrence_facts(data, "head")
    assert legacy["availability"] == "unavailable"
    assert legacy["total"] is None
    assert legacy["ports"][0]["declaration"]["status"] == "fail"
    assert occurrence_facts({"facts": {}}, "head")["availability"] == "unavailable"


def test_comparison_budget_omissions_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    import forge_mind.workspace.occurrence_facts as module

    monkeypatch.setattr(module, "MAX_PAGE_COMPARISONS", 1)
    monkeypatch.setattr(module, "MAX_COUNT_COMPARISONS", 0)
    data = fixture_measurement()
    data["facts"]["assembly"]["ports"]["head.other"] = deepcopy(
        data["facts"]["assembly"]["ports"]["head.bore"]
    )
    view = occurrence_facts(data, "head")
    assert len(view["ports"]) == 2
    assert view["comparison_omitted_count"] == 2
    assert all(row["comparisons_complete_for_page"] is False for row in view["ports"])
    assert all(row["matching_surface_count"] is None for row in view["ports"])


@pytest.mark.parametrize("offset,limit", [(-1, 20), (True, 20), (0, 0), (0, 101), (0, True)])
def test_invalid_page_is_rejected(offset: int, limit: int) -> None:
    with pytest.raises(ValueError):
        occurrence_facts(fixture_measurement(), "head", surface_offset=offset, surface_limit=limit)


@pytest.mark.parametrize(
    "field,value",
    [
        ("radius_mm", float("nan")),
        ("axis_direction", [0, 0, 0]),
        ("axis_point_mm", [0, 0, float("inf")]),
        ("axial_extent_mm", [3, 2]),
        ("concave", "unknown"),
    ],
)
def test_malformed_surface_is_rejected(field: str, value: Any) -> None:
    data = fixture_measurement()
    data["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"][0][field] = value
    with pytest.raises(ValueError):
        occurrence_facts(data, "head")


def test_duplicate_ids_unknown_occurrence_and_nonfinite_metadata_are_rejected() -> None:
    data = fixture_measurement()
    surfaces = data["facts"]["assembly"]["occurrences"][0]["cylindrical_surfaces"]
    surfaces.append(deepcopy(surfaces[0]))
    with pytest.raises(ValueError, match="ambiguous"):
        occurrence_facts(data, "head")
    with pytest.raises(ValueError, match="exactly one"):
        occurrence_facts(fixture_measurement(), "missing")
    data = fixture_measurement()
    data["facts"]["assembly"]["occurrences"][0]["center_mm"] = [0, 0, float("nan")]
    with pytest.raises(ValueError, match="finite"):
        occurrence_facts(data, "head")
