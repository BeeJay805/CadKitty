"""Independent saved-STEP checks for curved, mating and repeated native geometry.

Fixtures are reusable by real-pipeline checks. These checks establish computational
geometry at the measured stations, not physical fit or continuous fluid performance.
"""

from math import cos, degrees, pi, radians, sin, tan
from pathlib import Path
from typing import Any

import build123d as b
import pytest
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface

from forge_mind.workspace.construction_reference import construction_reference
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def native_feature(id: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {"id": id, "name": id.replace("_", " "), "kind": kind, **fields}


def saved_step(shape: Any, directory: Path, name: str) -> Any:
    path = directory / f"{name}.step"
    b.export_step(shape, str(path))
    saved = b.import_step(str(path))
    assert saved.is_valid
    return saved


def countersunk_plate_document(included_angle: float = 90) -> DesignDocument:
    raw = construction_reference("countersunk_plate")["document"]
    raw["variables"]["included_angle"] = included_angle
    return DesignDocument.model_validate(raw)


def check_countersunk_plate(saved: Any, included_angle: float) -> None:
    assert saved.is_valid and len(saved.solids()) == 1
    cones = [face for face in saved.faces() if "CONE" in str(face.geom_type)]
    assert len(cones) == 1
    actual_angle = 2 * degrees(abs(BRepAdaptor_Surface(cones[0].wrapped).Cone().SemiAngle()))
    assert actual_angle == pytest.approx(included_angle, abs=1e-6)
    radii = sorted(
        BRepAdaptor_Curve(edge.wrapped).Circle().Radius()
        for edge in cones[0].edges()
        if "CIRCLE" in str(edge.geom_type)
    )
    assert radii == pytest.approx([2, 5], abs=1e-6)
    cylinders = [face for face in saved.faces() if "CYLINDER" in str(face.geom_type)]
    assert len(cylinders) == 1
    assert BRepAdaptor_Surface(cylinders[0].wrapped).Cylinder().Radius() == pytest.approx(2)
    assert saved.bounding_box().size.to_tuple() == pytest.approx((24, 24, 8))
    depth = 3 / tan(radians(included_angle / 2))
    expected = 24 * 24 * 8 - pi * 2**2 * 8 - pi * depth * (5**2 + 5 * 2 - 2 * 2**2) / 3
    assert saved.volume == pytest.approx(expected, rel=1e-7)
    assert not saved.solids()[0].is_inside((0, 0, -3.5))


def test_included_angle_edit_changes_exported_cone_and_preserves_both_radii(tmp_path):
    doc = countersunk_plate_document()
    identities = [feature.id for feature in doc.features]
    shape, _ = compile_document(doc, {})
    before = saved_step(shape, tmp_path, "angle90")
    check_countersunk_plate(before, 90)
    doc.variables["included_angle"] = 82
    shape, _ = compile_document(doc, {})
    after = saved_step(shape, tmp_path, "angle82")
    check_countersunk_plate(after, 82)
    assert [feature.id for feature in doc.features] == identities
    assert doc.variables["bore_radius"] == 2 and doc.variables["rim_radius"] == 5
    assert after.volume < before.volume
    with pytest.raises(AssertionError):
        check_countersunk_plate(after, 90)


DUCT_STATIONS = [(0, 0, 12), (4, 0, 24), (12, 0, 36), (16, 0, 48)]


def curved_duct_document(wall: float = 2, blocked: bool = False) -> DesignDocument:
    features = [
        native_feature("path", "curve", points=[(0, 0, 0), *DUCT_STATIONS, (16, 0, 60)]),
        native_feature("outer_start", "profile", params={"radius_x": 7, "radius_y": 7}),
        native_feature(
            "inner_start",
            "profile",
            bindings={"radius_x": "inner_radius", "radius_y": "inner_radius"},
        ),
        native_feature("outer_sweep", "sweep", inputs=["outer_start", "path"]),
        native_feature("inner_sweep", "sweep", inputs=["inner_start", "path"]),
        native_feature("neck_outer", "profile", params={"z": -8, "radius_x": 9, "radius_y": 9}),
        native_feature(
            "neck_inner",
            "profile",
            params={"z": -8},
            bindings={"radius_x": "inner_radius", "radius_y": "inner_radius"},
        ),
        native_feature(
            "outer_loft", "loft", inputs=["neck_outer", "outer_start"], params={"ruled": 1}
        ),
        native_feature(
            "inner_loft", "loft", inputs=["neck_inner", "inner_start"], params={"ruled": 1}
        ),
        native_feature("duct_outer", "union", inputs=["outer_loft", "outer_sweep"]),
        native_feature("duct", "cut", inputs=["duct_outer", "inner_loft", "inner_sweep"]),
    ]
    return DesignDocument.model_validate(
        {
            "name": "Hollow curved duct",
            "variables": {"wall": wall},
            "relations": [{"name": "inner_radius", "operation": "subtract", "a": 7, "b": "wall"}],
            "features": features,
            "outputs": ["duct_outer" if blocked else "duct"],
        }
    )


def check_curved_duct(saved: Any, inner_radius: float = 5) -> None:
    assert saved.is_valid and len(saved.solids()) == 1
    assert any("BSPLINE" in str(face.geom_type) for face in saved.faces())
    solid = saved.solids()[0]
    for x, _, z in DUCT_STATIONS:
        assert not solid.is_inside((x, 0, z))
        assert not solid.is_inside((x, inner_radius - 0.4, z))
        assert solid.is_inside((x, (inner_radius + 7) / 2, z))
        assert not solid.is_inside((x, 7.5, z))
    assert not solid.is_inside((0, 0, -4))
    assert solid.is_inside((0, 7, -4))


def test_hollow_curved_sweep_and_loft_rebuild_preserve_outer_interfaces(tmp_path):
    doc = curved_duct_document()
    shape, _ = compile_document(doc, {})
    before = saved_step(shape, tmp_path, "duct_before")
    check_curved_duct(before)
    doc.variables["wall"] = 3
    shape, _ = compile_document(doc, {})
    after = saved_step(shape, tmp_path, "duct_thicker")
    check_curved_duct(after, 4)
    assert after.volume > before.volume
    assert after.bounding_box().min.to_tuple() == pytest.approx(
        before.bounding_box().min.to_tuple(), abs=1e-6
    )
    assert after.bounding_box().max.to_tuple() == pytest.approx(
        before.bounding_box().max.to_tuple(), abs=1e-6
    )
    blocked, _ = compile_document(curved_duct_document(blocked=True), {})
    with pytest.raises(AssertionError):
        check_curved_duct(saved_step(blocked, tmp_path, "blocked_duct"))


def sleeve_and_plug_document(bore_radius: float = 6.25) -> DesignDocument:
    sleeve = [
        native_feature("blank", "cylinder", params={"radius": 12, "height": 16}),
        native_feature(
            "bore", "cylinder", params={"height": 18}, bindings={"radius": "bore_radius"}
        ),
        native_feature("groove", "torus", params={"major_radius": 6.4, "minor_radius": 0.8}),
        native_feature(
            "mouth",
            "sketch",
            sketch={
                "offset": 4,
                "points": [{"id": "center", "x": 0, "y": 0}],
                "entities": [
                    {"id": "circle", "kind": "circle", "points": ["center"], "radius": 6.25}
                ],
            },
        ),
        native_feature("entry", "extrude", inputs=["mouth"], params={"amount": 4, "taper": -15}),
        native_feature("body", "cut", inputs=["blank", "bore", "groove", "entry"]),
    ]
    plug = [native_feature("body", "cylinder", params={"radius": 6, "height": 14})]
    return DesignDocument.model_validate(
        {
            "name": "Mating grooved sleeve and plug",
            "variables": {"bore_radius": bore_radius},
            "assembly": {
                "parts": [
                    {"id": "sleeve", "name": "Sleeve", "features": sleeve, "output": "body"},
                    {"id": "plug", "name": "Plug", "features": plug, "output": "body"},
                ],
                "roots": [
                    {"id": "sleeve", "definition": "sleeve"},
                    {"id": "plug", "definition": "plug"},
                ],
            },
        }
    )


def check_sleeve_and_plug(saved: Any, radial_gap: float = 0.25) -> None:
    assert saved.is_valid and len(saved.solids()) == 2
    plug, sleeve = sorted(saved.solids(), key=lambda solid: solid.bounding_box().size.X)
    assert sleeve.distance_to(plug) == pytest.approx(radial_gap, abs=1e-6)
    assert plug.volume == pytest.approx(pi * 6**2 * 14)
    assert any("TORUS" in str(face.geom_type) for face in sleeve.faces())
    assert any("CONE" in str(face.geom_type) for face in sleeve.faces())
    assert not sleeve.is_inside((7, 0, 0))
    assert sleeve.is_inside((7, 0, 3))
    assert not sleeve.is_inside((6.7, 0, 7.5))
    assert sleeve.is_inside((8, 0, 7.5))


def test_tapered_grooved_mating_assembly_checks_actual_gap_and_detects_interference(tmp_path):
    doc = sleeve_and_plug_document()
    shape, _ = compile_document(doc, {})
    check_sleeve_and_plug(saved_step(shape, tmp_path, "mating_before"))
    doc.variables["bore_radius"] = 6.4
    shape, _ = compile_document(doc, {})
    check_sleeve_and_plug(saved_step(shape, tmp_path, "mating_gap_edit"), 0.4)
    assert [part.id for part in doc.assembly.parts] == ["sleeve", "plug"]
    interferes, _ = compile_document(sleeve_and_plug_document(5.9), {})
    with pytest.raises(AssertionError):
        check_sleeve_and_plug(saved_step(interferes, tmp_path, "mating_interference"))


def patterned_flange_document(
    count: int = 12, pitch_radius: float = 22, missing_last: bool = False
) -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "name": "Radially patterned flange",
            "variables": {
                "hole_count": count,
                "tool_count": count - 1 if missing_last else count,
                "pitch_radius": pitch_radius,
            },
            "relations": [
                {"name": "angle_step", "operation": "divide", "a": 360, "b": "hole_count"}
            ],
            "features": [
                native_feature("blank", "cylinder", params={"radius": 30, "height": 6}),
                native_feature("shaft_bore", "cylinder", params={"radius": 4, "height": 8}),
                native_feature("pilot", "cylinder", params={"radius": 1.5, "height": 8}),
                native_feature(
                    "placed_pilot", "transform", inputs=["pilot"], bindings={"x": "pitch_radius"}
                ),
                native_feature(
                    "pilots",
                    "pattern",
                    inputs=["placed_pilot"],
                    bindings={"count": "tool_count", "angle": "angle_step"},
                ),
                native_feature("body", "cut", inputs=["blank", "shaft_bore", "pilots"]),
            ],
            "outputs": ["body"],
        }
    )


def check_patterned_flange(saved: Any, count: int = 12, pitch_radius: float = 22) -> None:
    assert saved.is_valid and len(saved.solids()) == 1
    assert saved.volume == pytest.approx(pi * (30**2 - 4**2 - count * 1.5**2) * 6, rel=1e-7)
    centers = []
    for face in saved.faces():
        if "CYLINDER" not in str(face.geom_type):
            continue
        cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
        if abs(cylinder.Radius() - 1.5) < 1e-6:
            center = cylinder.Axis().Location()
            centers.append((center.X(), center.Y()))
    assert len(centers) == count
    solid = saved.solids()[0]
    for index in range(count):
        angle = 2 * pi * index / count
        expected = (pitch_radius * cos(angle), pitch_radius * sin(angle))
        assert min((x - expected[0]) ** 2 + (y - expected[1]) ** 2 for x, y in centers) < 1e-10
        for z in (-2.5, 0, 2.5):
            assert not solid.is_inside((*expected, z))
        assert solid.is_inside(
            ((pitch_radius + 2) * cos(angle), (pitch_radius + 2) * sin(angle), 0)
        )


def test_rotational_pattern_preserves_all_bores_after_pitch_edit_and_count_growth(tmp_path):
    doc = patterned_flange_document()
    shape, _ = compile_document(doc, {})
    before = saved_step(shape, tmp_path, "flange_before")
    check_patterned_flange(before)
    doc.variables.update(hole_count=18, tool_count=18, pitch_radius=20)
    shape, _ = compile_document(doc, {})
    after = saved_step(shape, tmp_path, "flange_edit")
    check_patterned_flange(after, 18, 20)
    assert after.bounding_box().size.to_tuple() == pytest.approx(
        before.bounding_box().size.to_tuple()
    )
    missing, _ = compile_document(patterned_flange_document(missing_last=True), {})
    with pytest.raises(AssertionError):
        check_patterned_flange(saved_step(missing, tmp_path, "flange_missing_hole"))
