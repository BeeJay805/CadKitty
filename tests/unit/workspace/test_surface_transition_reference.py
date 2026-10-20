from __future__ import annotations

from math import isfinite, pi, sqrt

import build123d as b
import pytest

from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def volume(shape):
    if isinstance(shape, (list, b.ShapeList)):
        return sum(volume(part) for part in shape)
    assert shape.is_valid, "Invalid Boolean output cannot establish material volume"
    return sum(float(solid.volume) for solid in shape.solids())


def area(shape):
    if isinstance(shape, (list, b.ShapeList)):
        return sum(area(part) for part in shape)
    assert shape.is_valid, "Invalid Boolean output cannot establish face area"
    return sum(float(face.area) for face in shape.faces())


def compound(shape):
    return b.Compound(children=list(shape)) if isinstance(shape, (list, b.ShapeList)) else shape


def equal_material(actual, expected):
    actual, expected = compound(actual), compound(expected)
    assert actual.is_valid and expected.is_valid
    assert len(actual.solids()) == len(expected.solids()) == 1
    assert volume(actual.cut(expected)) == pytest.approx(0, abs=1e-6)
    assert volume(expected.cut(actual)) == pytest.approx(0, abs=1e-6)


def saved_elbow(tmp_path, bend):
    raw = authoring_schema("construction:surface_transitions")["document"]
    doc = DesignDocument.model_validate(raw)
    doc.variables["bend_radius"] = bend
    shape, history = compile_document(doc, {})
    assert shape.is_valid and len(shape.solids()) == 1
    assert history[-1]["state"] == "built"
    assert len(history[-1]["source_sha256"]) == 64
    path = tmp_path / f"elbow-{bend}.step"
    b.export_step(shape, str(path))
    saved = b.import_step(str(path))
    repeated = b.import_step(str(path))
    equal_material(saved, repeated)
    assert volume(saved.intersect(repeated)) == pytest.approx(saved.volume, rel=1e-7)
    equal_material(saved, shape)
    return saved, history


def expected_collars():
    inlet = b.Solid.make_cylinder(6, 8).cut(b.Solid.make_cylinder(4, 8))
    outlet_plane = b.Plane(origin=(34, 0, 36), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
    outlet = b.Solid.make_cylinder(6, 8, outlet_plane).cut(
        b.Solid.make_cylinder(4, 8, outlet_plane)
    )
    return compound(inlet), compound(outlet)


def clipped_collars(shape):
    return (
        compound(shape.intersect(b.Pos(0, 0, 4) * b.Box(30, 30, 8))),
        compound(shape.intersect(b.Pos(38, 0, 36) * b.Box(8, 30, 30))),
    )


def assert_annular_end_faces(shape):
    # Compare complete native trimmed end faces, not individual arc segmentation.
    inlet = [
        face
        for face in shape.faces()
        if "PLANE" in str(face.geom_type)
        and abs(face.bounding_box().min.Z) < 1e-6
        and abs(face.bounding_box().max.Z) < 1e-6
    ]
    outlet = [
        face
        for face in shape.faces()
        if "PLANE" in str(face.geom_type)
        and abs(face.bounding_box().min.X - 42) < 1e-6
        and abs(face.bounding_box().max.X - 42) < 1e-6
    ]
    end_plane = b.Plane(origin=(42, 0, 36), x_dir=(0, 1, 0), z_dir=(1, 0, 0))
    expected = (b.Circle(6) - b.Circle(4)).faces()[0]
    for candidates, target in ((inlet, expected), (outlet, end_plane * expected)):
        assert len(candidates) == 1
        face = candidates[0]
        assert face.is_valid
        assert face.area == pytest.approx(pi * (6**2 - 4**2), rel=1e-7)
        assert area(face.cut(target)) == pytest.approx(0, abs=1e-6)
        assert area(target.cut(face)) == pytest.approx(0, abs=1e-6)


def passage_pieces(bend):
    # Independent analytic cylinders and quarter torus, not a replay of recipe sweep.
    incoming = b.Solid.make_cylinder(4, 36 - bend)
    turn_plane = b.Plane(origin=(bend, 0, 36 - bend), x_dir=(-1, 0, 0), z_dir=(0, 1, 0))
    turn = b.Solid.make_torus(bend, 4, plane=turn_plane, major_angle=90)
    outgoing = b.Solid.make_cylinder(
        4,
        42 - bend,
        b.Plane(origin=(bend, 0, 36), x_dir=(0, 1, 0), z_dir=(1, 0, 0)),
    )
    expected_volumes = (pi * 4**2 * (36 - bend), pi * 4**2 * bend * pi / 2, pi * 4**2 * (42 - bend))
    for body, expected in zip((incoming, turn, outgoing), expected_volumes, strict=True):
        assert body.is_valid and len(body.solids()) == 1
        assert body.volume == pytest.approx(expected, rel=1e-7)
    assert tuple(turn.bounding_box().min) == pytest.approx((-4, -4, 36 - bend), abs=1e-6)
    assert tuple(turn.bounding_box().max) == pytest.approx((bend, 4, 40), abs=1e-6)
    return incoming, turn, outgoing


@pytest.mark.parametrize("bend", [16, 22])
def test_saved_elbow_has_open_annular_ends_and_full_nominal_passage(tmp_path, bend):
    saved, _ = saved_elbow(tmp_path, bend)
    assert_annular_end_faces(saved)
    # Analytic path length changes while endpoints stay fixed. Full-radius probes
    # meet at the tangent section disks, covering both straight runs and the bend.
    length = 42 + 36 + (pi / 2 - 2) * bend
    assert saved.volume == pytest.approx(pi * (6**2 - 4**2) * length, rel=1e-6)
    probes = passage_pieces(bend)
    for probe in probes:
        collision = volume(saved.intersect(probe))
        assert isfinite(collision) and collision >= 0
        assert collision == pytest.approx(0, abs=1e-6)
    # A real transverse plug is detected by the same passage instrument.
    plug = b.Pos(0, 0, 4) * b.Cylinder(4.5, 1)
    blocked = saved.fuse(plug)
    assert blocked.is_valid and len(blocked.solids()) == 1
    assert volume(blocked.intersect(probes[0])) > 40
    # Torus orientation/containment is challenged by an independent solid obstacle.
    center = (bend - bend / sqrt(2), 0, 36 - bend + bend / sqrt(2))
    obstacle = b.Pos(*center) * b.Sphere(1)
    assert volume(probes[1].intersect(obstacle)) == pytest.approx(4 * pi / 3, rel=1e-6)


def test_bend_edit_changes_actual_shape_preserving_complete_collars_and_envelope(tmp_path):
    before, old_history = saved_elbow(tmp_path, 16)
    after, new_history = saved_elbow(tmp_path, 22)
    assert old_history[-1]["source_sha256"] == new_history[-1]["source_sha256"]
    for shape in (before, after):
        bounds = shape.bounding_box()
        assert tuple(bounds.min) == pytest.approx((-6, -6, 0), abs=1e-5)
        assert tuple(bounds.max) == pytest.approx((42, 6, 42), abs=1e-5)
        for actual, expected in zip(clipped_collars(shape), expected_collars(), strict=True):
            equal_material(actual, expected)
    for old, new in zip(clipped_collars(before), clipped_collars(after), strict=True):
        equal_material(old, new)
    assert volume(before.cut(after)) > 1
    assert volume(after.cut(before)) > 1
    # An enlarged inlet hole must fail preservation despite unchanged external bounds.
    changed_collar = expected_collars()[0].cut(b.Solid.make_cylinder(4.5, 8))
    assert volume(expected_collars()[0].cut(changed_collar)) > 100


@pytest.mark.parametrize(
    "name,value",
    [
        ("wall", 0),
        ("wall", 6),
        ("outer_radius", 0),
        ("bend_radius", 6),
        ("bend_radius", 28),
        ("bend_radius", -1),
        ("collar_length", 0),
        ("reach", 20),
        ("rise", 20),
    ],
)
def test_invalid_dimensions_refuse_without_hidden_fallback(name, value):
    doc = DesignDocument.model_validate(
        authoring_schema("construction:surface_transitions")["document"]
    )
    doc.variables[name] = value
    with pytest.raises(ValueError, match=r"Feature body .*Use 0 < wall"):
        compile_document(doc, {})


def test_reference_is_optional_discoverable_and_independently_editable():
    index = authoring_schema("construction")
    assert "construction:surface_transitions" in {item["query"] for item in index["recipes"]}
    assert len(str(index)) < 6000
    focused = authoring_schema("construction:surface_transitions")
    assert "parametric_design" not in focused
    assert "G1" in " ".join(focused["notes"])
    focused["document"]["variables"]["bend_radius"] = 22
    assert (
        authoring_schema("construction:surface_transitions")["document"]["variables"]["bend_radius"]
        == 16
    )
    assert authoring_schema("construction:swept_tube")["document"]["variables"]["radius"] == 7
