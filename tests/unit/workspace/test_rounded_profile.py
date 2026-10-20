from __future__ import annotations

from copy import deepcopy
from math import pi, sqrt

import build123d as b
import pytest

from forge_mind.workspace import kernel
from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.document import DesignDocument, Feature


def feature(kind, feature_id="section", params=None, **kwargs):
    return {"id": feature_id, "name": feature_id, "kind": kind, "params": params or {}, **kwargs}


def document(features, outputs=None, **kwargs):
    return DesignDocument.model_validate(
        {
            "name": "Section control",
            "features": features,
            "outputs": outputs or [features[-1]["id"]],
            **kwargs,
        }
    )


def positive_volume(shape):
    if shape is None:
        return 0.0
    if isinstance(shape, (list, b.ShapeList)):
        return sum(positive_volume(item) for item in shape)
    return sum(float(solid.volume) for solid in shape.solids())


def same_solid(actual, expected):
    if isinstance(actual, (list, b.ShapeList)):
        actual = b.Compound(children=list(actual))
    if isinstance(expected, (list, b.ShapeList)):
        expected = b.Compound(children=list(expected))
    assert actual.is_valid and expected.is_valid
    assert len(actual.solids()) == len(expected.solids()) == 1
    assert positive_volume(actual - expected) == pytest.approx(0, abs=1e-6)
    assert positive_volume(expected - actual) == pytest.approx(0, abs=1e-6)


def capture_section(monkeypatch, params, *, bindings=None, variables=None):
    # Observe the real intermediate shape without changing validation or the solid-output gate.
    seen = []
    original_signature = kernel.signature

    def observe(shape):
        seen.append(shape)
        return original_signature(shape)

    monkeypatch.setattr(kernel, "signature", observe)
    doc = document(
        [
            feature("rounded_profile", params=params, bindings=bindings or {}),
            feature("box", "output", {"x": 1, "y": 1, "z": 1}),
        ],
        variables=variables or {},
    )
    _, history = kernel.compile_document(doc, {})
    assert history[0]["solid_count"] == 0
    assert history[0]["state"] == "built"
    return seen[0]


def test_discovery_and_reference_match_actual_graph_boundary():
    index = authoring_schema("construction")
    assert "rounded_profile" in index["operations"]
    assert "construction:rounded_transition" in {entry["query"] for entry in index["recipes"]}
    reference = authoring_schema("feature:rounded_profile")
    assert reference["required"] == ["width", "height", "radius"]
    assert reference["optional"] == {"x": 0, "y": 0, "z": 0, "rx": 0, "ry": 0, "rz": 0}
    assert "full outside dimensions" in reference["meaning"]
    assert "radius < min(width, height) / 2" in reference["meaning"]
    assert "after bindings resolve" in reference["meaning"]
    assert authoring_schema("feature:extrude")["inputs"] == "one sketch"
    assert authoring_schema("feature:revolve")["inputs"] == "one sketch"


def test_legacy_ellipse_serialization_and_geometry_are_unchanged():
    legacy = feature("profile", params={"radius_x": 7, "radius_y": 4})
    assert Feature.model_validate(legacy).model_dump(mode="json") == {
        "id": "section",
        "name": "section",
        "kind": "profile",
        "inputs": [],
        "params": {"radius_x": 7.0, "radius_y": 4.0},
        "bindings": {},
        "points": [],
        "code": "",
        "sketch": None,
        "asset_id": None,
        "catalog_id": None,
        "selection": [],
        "suppressed": False,
    }
    doc = document(
        [
            legacy,
            feature("profile", "end", {"radius_x": 7, "radius_y": 4, "z": 12}),
            feature("loft", "body", {"ruled": 1}, inputs=["section", "end"]),
        ]
    )
    shape, history = kernel.compile_document(doc, {})
    assert shape.volume == pytest.approx(pi * 7 * 4 * 12, rel=1e-7)
    assert tuple(shape.bounding_box().size) == pytest.approx((14, 8, 12), abs=1e-6)
    assert [row["solid_count"] for row in history] == [0, 0, 1]
    assert "width" not in doc.features[0].params


@pytest.mark.parametrize("width,height,radius", [(30, 14, 3), (12, 28, 2), (18, 10, 1)])
def test_intermediate_face_has_analytic_area_and_full_dimensions(
    monkeypatch, width, height, radius
):
    shape = capture_section(monkeypatch, {"width": width, "height": height, "radius": radius})
    assert shape.is_valid and len(shape.faces()) == 1 and not shape.solids()
    assert shape.area == pytest.approx(width * height - (4 - pi) * radius**2, rel=1e-8)
    assert tuple(shape.bounding_box().size) == pytest.approx((width, height, 0), abs=1e-6)
    assert tuple(shape.center()) == pytest.approx((0, 0, 0), abs=1e-6)
    prism = b.extrude(shape.faces()[0], amount=2).solids()[0]
    assert prism.is_inside((width / 2 - 0.1, 0, 1))
    assert not prism.is_inside((width / 2 - 0.1, height / 2 - 0.1, 1))
    corner = (width / 2 - radius, height / 2 - radius)
    assert prism.is_inside(
        (corner[0] + 0.9 * radius / sqrt(2), corner[1] + 0.9 * radius / sqrt(2), 1)
    )
    assert not prism.is_inside(
        (corner[0] + 1.1 * radius / sqrt(2), corner[1] + 1.1 * radius / sqrt(2), 1)
    )


def test_pose_and_dimension_bindings_use_resolved_values(monkeypatch):
    section = capture_section(
        monkeypatch,
        {"width": 1, "height": 14, "radius": 3},
        bindings={"width": "width", "x": "offset", "rz": "angle"},
        variables={"width": 30, "offset": 8, "angle": 90},
    )
    assert tuple(section.center()) == pytest.approx((8, 0, 0), abs=1e-6)
    assert tuple(section.bounding_box().size) == pytest.approx((14, 30, 0), abs=1e-6)


@pytest.mark.parametrize(
    "rx,ry,rz,normal,delta,expected_size",
    [
        (0, 0, 90, (0, 0, 1), (0, 0, 12), (14, 30, 12)),
        (90, 0, 0, (0, -1, 0), (0, -12, 0), (30, 12, 14)),
        (0, 90, 0, (1, 0, 0), (12, 0, 0), (12, 14, 30)),
    ],
)
def test_posed_loft_matches_analytic_volume_and_python_prism(
    monkeypatch, rx, ry, rz, normal, delta, expected_size
):
    pose = {"x": 5, "y": -3, "z": 7, "rx": rx, "ry": ry, "rz": rz}
    params = {"width": 30, "height": 14, "radius": 3, **pose}
    section = capture_section(monkeypatch, params)
    assert tuple(section.center()) == pytest.approx((5, -3, 7), abs=1e-6)
    assert tuple(section.faces()[0].normal_at()) == pytest.approx(normal, abs=1e-6)
    end = {
        **params,
        **{axis: pose[axis] + d for axis, d in zip(("x", "y", "z"), delta, strict=True)},
    }
    native, history = kernel.compile_document(
        document(
            [
                feature("rounded_profile", params=params),
                feature("rounded_profile", "end", end),
                feature("loft", "body", {"ruled": 1}, inputs=["section", "end"]),
            ]
        ),
        {},
    )
    python, _ = kernel.compile_document(
        document(
            [
                feature(
                    "python",
                    "body",
                    {**params, "length": 12},
                    code="section = b.RectangleRounded(params['width'], params['height'], params['radius'])\n"
                    "prism = b.extrude(section, amount=params['length'])\n"
                    "result = b.Pos(params['x'], params['y'], params['z']) * "
                    "b.Rot(params['rx'], params['ry'], params['rz']) * prism",
                ),
            ]
        ),
        {},
    )
    same_solid(native, python)
    assert native.volume == pytest.approx((30 * 14 - (4 - pi) * 3**2) * 12, rel=1e-7)
    assert tuple(native.bounding_box().size) == pytest.approx(expected_size, abs=1e-6)
    assert all(row["source_sha256"] is None for row in history)


@pytest.mark.parametrize(
    "points", [[(0, 0, 0), (0, 0, 12), (0, 0, 24)], [(0, 0, 0), (0, 0, 12), (5, 0, 24)]]
)
def test_rounded_section_sweeps_real_curve_with_python_parity(points):
    params = {"width": 8, "height": 4, "radius": 1}
    native, history = kernel.compile_document(
        document(
            [
                feature("rounded_profile", params=params),
                feature("curve", "path", points=points),
                feature("sweep", "body", inputs=["section", "path"]),
            ]
        ),
        {},
    )
    python, _ = kernel.compile_document(
        document(
            [
                feature(
                    "python",
                    "body",
                    params,
                    code=f"path = b.Wire([b.Spline(*{points!r})])\n"
                    "result = b.sweep(b.RectangleRounded(params['width'], params['height'], "
                    "params['radius']), path=path)",
                ),
            ]
        ),
        {},
    )
    same_solid(native, python)
    assert [row["solid_count"] for row in history] == [0, 0, 1]
    if points[-1][0] == 0:
        assert native.volume == pytest.approx((8 * 4 - (4 - pi)) * 24, rel=1e-7)


def test_composed_recipe_edits_inlet_and_preserves_circular_neck(tmp_path):
    raw = authoring_schema("construction:rounded_transition")["document"]
    doc = DesignDocument.model_validate(raw)
    before, history = kernel.compile_document(doc, {})
    assert before.is_valid and len(before.solids()) == 1
    assert tuple(before.bounding_box().size) == pytest.approx((36, 20, 44), abs=1e-5)
    assert all(row["source_sha256"] is None for row in history)
    doc.variables["inlet_width"] = 44
    after, _ = kernel.compile_document(doc, {})
    assert after.volume > before.volume
    assert pytest.approx(44, abs=1e-5) == after.bounding_box().size.X
    # This region excludes the changing loft and isolates the upper circular neck.
    region = b.Pos(0, 0, 38) * b.Box(100, 100, 10)
    expected = b.Pos(0, 0, 38) * b.Cylinder(9, 10)
    same_solid(before.intersect(region), expected)
    same_solid(after.intersect(region), expected)
    path = tmp_path / "transition.step"
    b.export_step(after, str(path))
    same_solid(b.import_step(str(path)), after)
    assert raw["variables"]["inlet_width"] == 36


def test_bindings_override_literal_params_and_radius_is_checked_after_resolution():
    features = [
        feature(
            "rounded_profile",
            params={"width": 30, "height": 14, "radius": 0},
            bindings={"radius": "corner"},
        ),
        feature(
            "rounded_profile",
            "end",
            {"width": 30, "height": 14, "radius": 0, "z": 12},
            bindings={"radius": "corner"},
        ),
        feature("loft", "body", inputs=["section", "end"]),
    ]
    doc = document(features, variables={"corner": 3})
    shape, _ = kernel.compile_document(doc, {})
    assert shape.volume > 0
    doc.variables["corner"] = 7
    with pytest.raises(ValueError, match=r"radius < min\(width, height\) / 2"):
        kernel.compile_document(doc, {})


@pytest.mark.parametrize(
    "changes",
    [
        {"radius": 0},
        {"radius": -1},
        {"radius": 7},
        {"radius": 8},
        {"width": 0},
        {"height": -1},
        {"width": None},
        {"height": None},
        {"radius": None},
    ],
)
def test_invalid_or_missing_dimensions_refuse_without_clamping(changes):
    params = {"width": 30, "height": 14, "radius": 3}
    for key, value in changes.items():
        if value is None:
            params.pop(key)
        else:
            params[key] = value
    doc = document(
        [
            feature("rounded_profile", params=params),
            feature("box", "body", {"x": 1, "y": 1, "z": 1}),
        ]
    )
    with pytest.raises(ValueError, match=r"Feature section"):
        kernel.compile_document(doc, {})


@pytest.mark.parametrize(
    "change",
    [
        {"params": {"width": 30, "height": 14, "radius": 3, "amount": 10}},
        {"bindings": {"radius": "missing"}},
        {"bindings": {"amount": "known"}},
        {"inputs": ["base"]},
        {"params": {"width": float("inf"), "height": 14, "radius": 3}},
    ],
)
def test_graph_refuses_unknown_keys_bindings_nonfinite_values_and_nonzero_arity(change):
    section = feature("rounded_profile", params={"width": 30, "height": 14, "radius": 3})
    section.update(deepcopy(change))
    with pytest.raises(ValueError):
        document(
            [feature("box", "base", {"x": 1, "y": 1, "z": 1}), section], variables={"known": 2}
        )


@pytest.mark.parametrize("kind", ["loft", "sweep", "extrude", "revolve"])
def test_solid_cannot_impersonate_section(kind):
    source = feature("box", "base", {"x": 10, "y": 10, "z": 10})
    other = feature("curve", "path", points=[(0, 0, 0), (0, 0, 10)])
    operation = feature(
        kind,
        "body",
        {"amount": 10} if kind == "extrude" else {},
        inputs=["base", "path"] if kind in {"loft", "sweep"} else ["base"],
    )
    with pytest.raises(ValueError):
        document([source, other, operation])


@pytest.mark.parametrize(
    "section_kind,params",
    [
        ("profile", {"radius_x": 7, "radius_y": 4}),
        ("rounded_profile", {"width": 30, "height": 14, "radius": 3}),
    ],
)
@pytest.mark.parametrize("consumer", ["extrude", "revolve"])
def test_section_does_not_expand_existing_sketch_only_operations(section_kind, params, consumer):
    with pytest.raises(ValueError, match="explicit sketch profile"):
        document(
            [
                feature(section_kind, params=params),
                feature(
                    consumer,
                    "body",
                    {"amount": 10} if consumer == "extrude" else {},
                    inputs=["section"],
                ),
            ]
        )


@pytest.mark.parametrize(
    "kind,params",
    [
        ("profile", {"radius_x": 7, "radius_y": 4}),
        ("rounded_profile", {"width": 30, "height": 14, "radius": 3}),
    ],
)
def test_intermediate_face_still_refuses_final_solid_output(kind, params):
    with pytest.raises(ValueError, match="at least one solid"):
        kernel.compile_document(document([feature(kind, params=params)]), {})
