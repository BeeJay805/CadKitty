"""Measure native primitives and cuts independently of their dispatch code."""

from math import pi, radians, tan
from typing import Any

import build123d as b
import pytest

from forge_mind.workspace.construction_reference import feature_reference
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document


def feature(kind: str, params: dict[str, float], id: str = "body", **extra: Any) -> dict[str, Any]:
    return {"id": id, "name": id, "kind": kind, "params": params, **extra}


def document(features: list[dict[str, Any]], **extra: Any) -> DesignDocument:
    return DesignDocument.model_validate(
        {"features": features, "outputs": [features[-1]["id"]], **extra}
    )


@pytest.mark.parametrize(("bottom", "top"), [(5, 2), (0, 3), (3, 0), (3, 3)])
def test_cone_has_centered_stations_and_analytic_volume(bottom: float, top: float) -> None:
    shape, history = compile_document(
        document([feature("cone", {"bottom_radius": bottom, "top_radius": top, "height": 6})]), {}
    )
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(2 * pi * (bottom**2 + bottom * top + top**2))
    bounds = shape.bounding_box()
    assert pytest.approx(-3) == bounds.min.Z
    assert pytest.approx(3) == bounds.max.Z
    assert pytest.approx(2 * max(bottom, top)) == bounds.size.X
    assert history[0]["history_scope"] == "Native feature"


def test_cone_countersink_cut_and_bound_rebuild_export_actual_solid(tmp_path) -> None:
    doc = document(
        [
            feature("box", {"x": 20, "y": 20, "z": 6}, "plate"),
            feature("cylinder", {"radius": 2, "height": 8}, "bore"),
            feature(
                "cone",
                {"bottom_radius": 2, "top_radius": 5, "height": 3},
                "sink",
                bindings={"top_radius": "rim"},
            ),
            feature("transform", {"z": 2.5}, "placed", inputs=["sink"]),
            feature("cut", {}, inputs=["plate", "bore", "placed"]),
        ],
        variables={"rim": 5},
    )
    shape, _ = compile_document(doc, {})
    assert shape.is_valid and len(shape.solids()) == 1
    # The plate clips the tool to a 2 mm frustum: radii 2 and 4.
    assert shape.volume == pytest.approx(2400 - 104 * pi / 3)
    solid = shape.solids()[0]
    assert not solid.is_inside((0, 0, -2.5))
    assert not solid.is_inside((3, 0, 2.5))
    assert solid.is_inside((3, 0, 0))
    exported_path = tmp_path / "countersink.step"
    b.export_step(shape, str(exported_path))
    exported = b.import_step(str(exported_path))
    assert exported.is_valid and len(exported.solids()) == 1
    assert exported.volume == pytest.approx(shape.volume)
    doc.variables["rim"] = 6.5
    changed, _ = compile_document(doc, {})
    assert changed.volume == pytest.approx(2400 - 42 * pi)
    assert changed.solids()[0].is_inside((3, 0, 0))
    assert not changed.solids()[0].is_inside((4, 0, 2.5))


def test_torus_has_analytic_ring_volume_and_bound_minor_radius() -> None:
    doc = document(
        [feature("torus", {"major_radius": 8}, bindings={"minor_radius": "tube"})],
        variables={"tube": 2},
    )
    shape, _ = compile_document(doc, {})
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(64 * pi**2)
    assert pytest.approx(20) == shape.bounding_box().size.X
    assert pytest.approx(4) == shape.bounding_box().size.Z
    assert shape.solids()[0].is_inside((8, 0, 0))
    assert not shape.solids()[0].is_inside((0, 0, 0))
    doc.variables["tube"] = 1
    rebuilt, _ = compile_document(doc, {})
    assert rebuilt.volume == pytest.approx(16 * pi**2)


def test_torus_cut_makes_rounded_top_groove() -> None:
    shape, _ = compile_document(
        document(
            [
                feature("cylinder", {"radius": 12, "height": 4}, "plate"),
                feature("torus", {"major_radius": 8, "minor_radius": 1}, "ring"),
                feature("transform", {"z": 2}, "placed", inputs=["ring"]),
                feature("cut", {}, inputs=["plate", "placed"]),
            ]
        ),
        {},
    )
    assert shape.is_valid and len(shape.solids()) == 1
    # Only the bottom half of the torus intersects the plate.
    assert shape.volume == pytest.approx(576 * pi - 8 * pi**2)
    assert not shape.solids()[0].is_inside((8, 0, 1.5))
    assert shape.solids()[0].is_inside((8, 0, 0))
    assert shape.solids()[0].is_inside((0, 0, 1.5))


def test_three_profile_ruled_loft_preserves_intermediate_station() -> None:
    shape, history = compile_document(
        document(
            [
                feature("profile", {"radius_x": radius, "radius_y": radius, "z": z}, f"p{i}")
                for i, (radius, z) in enumerate([(5, 0), (2, 3), (4, 7)])
            ]
            + [feature("loft", {"ruled": 1}, inputs=["p0", "p1", "p2"])]
        ),
        {},
    )
    assert shape.is_valid and len(shape.solids()) == 1
    # Two independently calculated frusta, not a direct end-to-end loft.
    assert shape.volume == pytest.approx(pi * (39 + 112 / 3))
    assert pytest.approx(7) == shape.bounding_box().size.Z
    assert shape.solids()[0].is_inside((1, 0, 3))
    assert not shape.solids()[0].is_inside((3, 0, 3))
    assert len(history) == 4


def circular_extrusion(amount: float, taper: float | None) -> DesignDocument:
    params = {"amount": amount}
    if taper is not None:
        params["taper"] = taper
    return document(
        [
            feature(
                "sketch",
                {},
                "section",
                sketch={
                    "points": [{"id": "center", "x": 0, "y": 0}],
                    "entities": [
                        {"id": "circle", "kind": "circle", "points": ["center"], "radius": 5}
                    ],
                },
            ),
            feature("extrude", params, inputs=["section"]),
        ]
    )


@pytest.mark.parametrize("amount", [4, -4])
@pytest.mark.parametrize("taper", [10, -10])
def test_extrude_taper_has_independent_frustum_volume(amount: float, taper: float) -> None:
    shape, _ = compile_document(circular_extrusion(amount, taper), {})
    end_radius = 5 - abs(amount) * tan(radians(taper))
    assert shape.is_valid and len(shape.solids()) == 1
    assert shape.volume == pytest.approx(
        abs(amount) * pi * (25 + 5 * end_radius + end_radius**2) / 3
    )
    assert pytest.approx(min(0, amount), abs=1e-6) == shape.bounding_box().min.Z
    assert pytest.approx(max(0, amount), abs=1e-6) == shape.bounding_box().max.Z


def test_legacy_extrude_default_and_document_fingerprint_stay_unchanged() -> None:
    default, default_history = compile_document(circular_extrusion(4, None), {})
    explicit, explicit_history = compile_document(circular_extrusion(4, 0), {})
    assert default.volume == pytest.approx(100 * pi)
    assert default_history[-1]["signature"] == explicit_history[-1]["signature"]
    assert default.volume == pytest.approx(explicit.volume)
    legacy = document([feature("box", {"x": 10, "y": 8, "z": 2}, "base", name="Base")])
    assert (
        legacy.fingerprint() == "8e45b606263eae679e54a8cdef93b18ef3d59c2f3774e9d97e3153896f39aaf7"
    )


@pytest.mark.parametrize(
    ("kind", "params", "message"),
    [
        ("cone", {"bottom_radius": -1, "top_radius": 2, "height": 3}, "Cone radii"),
        ("cone", {"bottom_radius": 2, "top_radius": -1, "height": 3}, "Cone radii"),
        ("cone", {"bottom_radius": 0, "top_radius": 0, "height": 3}, "Cone radii"),
        ("cone", {"bottom_radius": 1, "top_radius": 2, "height": 0}, "height requires a positive"),
        ("cone", {"bottom_radius": 1, "height": 3}, "top_radius requires an explicit"),
        ("torus", {"major_radius": 2, "minor_radius": 2}, "minor_radius < major_radius"),
        ("torus", {"major_radius": 2, "minor_radius": 3}, "minor_radius < major_radius"),
        ("torus", {"major_radius": 0, "minor_radius": 1}, "major_radius requires a positive"),
        ("torus", {"major_radius": 2, "minor_radius": -1}, "minor_radius requires a positive"),
        ("torus", {"major_radius": 2}, "minor_radius requires a positive"),
    ],
)
def test_invalid_primitive_parameters_retain_feature_error(
    kind: str, params: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=f"Feature body .*{message}"):
        compile_document(document([feature(kind, params)]), {})


@pytest.mark.parametrize("taper", [-90, 90, 100])
def test_invalid_taper_has_feature_error(taper: float) -> None:
    with pytest.raises(ValueError, match=r"Feature body .*taper must be between"):
        compile_document(circular_extrusion(4, taper), {})


@pytest.mark.parametrize("kind", ["cone", "torus"])
def test_extension_schema_rejects_unknown_params_and_primitive_inputs(kind: str) -> None:
    with pytest.raises(ValueError, match=f"Unknown parameter for {kind}"):
        document([feature(kind, {"invented": 1})])
    with pytest.raises(ValueError, match=f"{kind} requires 0 inputs"):
        document([feature("sphere", {"radius": 1}, "seed"), feature(kind, {}, inputs=["seed"])])


def test_focused_extension_references_publish_exact_parameters() -> None:
    assert feature_reference("cone")["required"] == ["bottom_radius", "top_radius", "height"]
    assert feature_reference("torus")["required"] == ["major_radius", "minor_radius"]
    assert feature_reference("extrude")["optional"] == {"taper": 0}
    assert feature_reference("loft")["inputs"].startswith("two or more")
