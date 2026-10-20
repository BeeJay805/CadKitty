"""Adversarial product tool receipts from supplied intervals and exported STEP."""

import math

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.product_tools import (
    _ADAPTER,
    _geometry,
    available_product_tools,
    product_tool_schema,
    run_product_tool,
)


def fastener():
    return {
        "tool": "fastener_stack",
        "units": "mm",
        "screw_length": {"low": 9.8, "high": 10.2},
        "grip_layers": [{"low": 4.8, "high": 5.2}],
        "required_engagement": 4.0,
        "available_depth": {"low": 6.0, "high": 6.2},
    }


def test_fastener_worst_case_and_washer(tmp_path):
    req = fastener()
    result = run_product_tool(req, tmp_path)
    assert result["status"] == "pass"
    assert result["results"]["engagement_mm"] == pytest.approx([4.6, 5.4])
    req["grip_layers"].append({"low": 1.0, "high": 1.2})
    assert run_product_tool(req, tmp_path)["status"] == "fail"


@pytest.mark.parametrize(
    "field,value", [("required_engagement", 5.0), ("available_depth", {"low": 5.0, "high": 5.5})]
)
def test_short_engagement_and_bottoming(tmp_path, field, value):
    req = fastener()
    req[field] = value
    assert run_product_tool(req, tmp_path)["status"] == "fail"


@pytest.mark.parametrize(
    "field,value",
    [
        ("units", "in"),
        ("required_engagement", math.inf),
        ("required_engagement", math.nan),
        ("required_engagement", True),
        ("required_engagement", "4"),
        ("unexpected", 1),
        ("screw_length", {"low": 12.0, "high": 10.0}),
        ("grip_layers", [{"low": -1.0, "high": 1.0}]),
    ],
)
def test_invalid_inputs_rejected(tmp_path, field, value):
    req = fastener()
    req[field] = value
    with pytest.raises(ValidationError):
        run_product_tool(req, tmp_path)


def test_signed_tolerances_and_cable(tmp_path):
    result = run_product_tool(
        {
            "tool": "tolerance_stack",
            "units": "mm",
            "terms": [{"low": -2.0, "high": -1.0}, {"low": 5.0, "high": 6.0}],
            "allowable": {"low": 3.0, "high": 5.0},
        },
        tmp_path,
    )
    assert result["status"] == "pass"
    assert result["results"]["sum_mm"] == [3.0, 5.0]
    result = run_product_tool(
        {
            "tool": "cable_bend",
            "units": "mm",
            "centerline_radii": [20.0],
            "bundle_radius": 3.0,
            "minimum_inside_radius": 18.0,
        },
        tmp_path,
    )
    assert result["status"] == "fail"
    assert result["qualification_status"] == "unverified"


def hole_request():
    return {
        "tool": "hole_pattern",
        "units": "mm",
        "body": {"body_index": 0},
        "axis": [0.0, 0.0, 1.0],
        "holes": [{"center": [0.0, 0.0, 0.0], "diameter": 4.0}],
    }


def export_panel(tmp_path, *, cap=0.0, shift=0.0, diameter=4.0):
    panel = b.Box(20, 20, 4, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    cutter = b.Solid.make_cylinder(diameter / 2, 6, b.Plane(origin=(shift, 0, cap)))
    path = tmp_path / "part.step"
    b.export_step(panel - cutter, path)
    return path


def inspect(req, path):
    return _geometry(_ADAPTER.validate_python(req), path, file_sha256(path))


def test_exported_through_hole(tmp_path):
    path = export_panel(tmp_path)
    receipt = inspect(hole_request(), path)
    assert receipt["status"] == "pass"
    assert receipt["artifact_hashes"]["part.step"] == file_sha256(path)


@pytest.mark.parametrize("diameter", [3.0, 4.000005])
def test_hole_observations_measure_actual_diameter_separately_from_target(tmp_path, diameter):
    path = export_panel(tmp_path, diameter=diameter)
    receipt = inspect(hole_request(), path)
    hole = receipt["results"]["holes"][0]
    assert hole["diameter_mm"] == 4.0  # legacy field is explicitly a target
    assert hole["cylindrical_walls"]
    for wall in hole["cylindrical_walls"]:
        assert wall["diameter_mm"] == pytest.approx(diameter, abs=1e-8)
        assert wall["diameter_error_mm"] == pytest.approx(diameter - 4, abs=1e-8)
    assert receipt["status"] == ("fail" if diameter == 3 else "pass")
    assert receipt["algorithm_version"] == 2
    assert receipt["qualification_status"] == "unverified"


def test_hole_observations_keep_counterbore_diameters_separate(tmp_path):
    panel = b.Box(20, 20, 8, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    panel -= b.Solid.make_cylinder(2, 10, b.Plane(origin=(0, 0, -1)))
    panel -= b.Solid.make_cylinder(4, 5, b.Plane(origin=(0, 0, 4)))
    path = tmp_path / "part.step"
    b.export_step(panel, path)
    receipt = inspect(hole_request(), path)
    walls = receipt["results"]["holes"][0]["cylindrical_walls"]
    assert sorted({round(wall["diameter_mm"], 6) for wall in walls}) == [4, 8]
    assert receipt["status"] == "pass"  # full-depth 4 mm opening only


def test_wrong_axis_has_no_substituted_target_measurement(tmp_path):
    path = export_panel(tmp_path, shift=1)
    hole = inspect(hole_request(), path)["results"]["holes"][0]
    assert hole["cylindrical_walls"] == []
    assert hole["status"] == "fail"


@pytest.mark.parametrize(
    "mutation", [{"cap": 0.05}, {"shift": 1.0}, {"diameter": 3.0}, {"diameter": 5.0}]
)
def test_blind_shifted_wrong_diameter_fail(tmp_path, mutation):
    path = export_panel(tmp_path, **mutation)
    assert inspect(hole_request(), path)["status"] == "fail"


def test_wrong_pattern_pitch(tmp_path):
    panel = b.Box(20, 20, 4, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    for x in [-5, 5]:
        panel -= b.Solid.make_cylinder(2, 6, b.Plane(origin=(x, 0, -1)))
    path = tmp_path / "part.step"
    b.export_step(panel, path)
    req = hole_request()
    req["holes"] = [{"center": [x, 0.0, 0.0], "diameter": 4.0} for x in [-5.0, 5.0]]
    assert inspect(req, path)["status"] == "pass"
    req["holes"][1]["center"][0] = 6.0
    assert inspect(req, path)["status"] == "fail"


def test_arbitrary_axis(tmp_path):
    path = export_panel(tmp_path)
    shape = b.import_step(path).rotate(b.Axis.Y, 31)
    b.export_step(shape, path)
    req = hole_request()
    req["axis"] = [math.sin(math.radians(31)), 0.0, math.cos(math.radians(31))]
    assert inspect(req, path)["status"] == "pass"


def test_vent_cap_never_passes(tmp_path):
    req = {
        **hole_request(),
        "tool": "vent_open_fraction",
        "region_origin": [-10.0, -10.0, 0.0],
        "region_x_axis": [1.0, 0.0, 0.0],
        "region_width": 20.0,
        "region_height": 20.0,
        "minimum_open_fraction": 0.03,
    }
    result = inspect(req, export_panel(tmp_path))
    assert result["status"] == "pass"
    assert result["results"]["verified_open_fraction_lower_bound"] == pytest.approx(math.pi / 100)
    assert inspect(req, export_panel(tmp_path, cap=0.05))["status"] == "fail"
    req["holes"].append(req["holes"][0])
    with pytest.raises(ValueError, match="overlaps"):
        inspect(req, export_panel(tmp_path))


def test_static_reservation(tmp_path):
    path = tmp_path / "part.step"
    b.export_step(b.Compound(children=[b.Box(2, 2, 2), b.Box(2, 2, 2).translate((5, 0, 0))]), path)
    req = {
        "tool": "clearance_volume",
        "units": "mm",
        "reservation": {"body_index": 0},
        "obstacles": [{"body_index": 1}],
    }
    assert inspect(req, path)["status"] == "pass"
    req["obstacles"] = [{"body_index": 0}]
    assert inspect(req, path)["status"] == "fail"


def test_public_worker_and_missing_artifact(tmp_path):
    assert run_product_tool(hole_request(), tmp_path)["status"] == "unverified"
    export_panel(tmp_path)
    assert run_product_tool(hole_request(), tmp_path)["status"] == "pass"


def test_schema_and_bounds():
    tools = available_product_tools()
    assert len(tools) == 11
    assert {tool["tool"] for tool in tools} == {
        "fastener_stack",
        "tolerance_stack",
        "cable_bend",
        "hole_pattern",
        "vent_open_fraction",
        "mount_grid",
        "clearance_volume",
        "projected_pair",
        "planar_face_pair",
        "axial_cavity",
        "open_rectangular_shell",
    }
    assert "discriminator" in product_tool_schema()
    assert product_tool_schema("hole_pattern")["properties"]["holes"]["maxItems"] == 64
    req = hole_request()
    req["holes"] *= 65
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(req)


def test_middle_membrane_and_missing_body_remain_nonpassing(tmp_path):
    panel = b.Box(20, 20, 4, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    for origin, height in [(-1, 2.975), (2.025, 3)]:
        panel -= b.Solid.make_cylinder(2, height, b.Plane(origin=(0, 0, origin)))
    path = tmp_path / "part.step"
    b.export_step(panel, path)
    assert inspect(hole_request(), path)["status"] == "fail"
    req = hole_request()
    req["body"] = {"body_index": 127}
    assert run_product_tool(req, tmp_path)["status"] == "unverified"


@pytest.mark.parametrize(
    "update",
    [
        {"axis": [0.0, 0.0, 0.0]},
        {"body": {}},
        {"body": {"body_index": True}},
        {"body": {"body_index": 0, "occurrence_path": "part"}},
    ],
)
def test_geometry_selector_abuse(update):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({**hole_request(), **update})


def test_occurrence_identity_from_step_names(tmp_path):
    from forge_mind.workspace.assembly import occurrence_label

    path = export_panel(tmp_path)
    panel = b.import_step(path)
    panel.label = occurrence_label("panel")
    b.export_step(b.Compound(children=[panel]), path)
    req = hole_request()
    req["body"] = {"occurrence_path": "panel"}
    assert inspect(req, path)["status"] == "pass"


def test_mount_grid_105_pitch_and_wrong_pitch(tmp_path):
    panel = b.Box(120, 120, 4)
    for x in [-52.5, 52.5]:
        for y in [-52.5, 52.5]:
            panel -= b.Solid.make_cylinder(2, 6, b.Plane(origin=(x, y, -3)))
    path = tmp_path / "part.step"
    b.export_step(panel, path)
    req = {
        "tool": "mount_grid",
        "units": "mm",
        "body": {"body_index": 0},
        "origin": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "x_axis": [1.0, 0.0, 0.0],
        "pitch_x": 105.0,
        "pitch_y": 105.0,
        "columns": 2,
        "rows": 2,
        "diameter": 4.0,
    }
    result = inspect(req, path)
    assert result["status"] == "pass"
    assert len(result["results"]["holes"]) == 4
    assert {h["center_mm"][0] for h in result["results"]["holes"]} == {-52.5, 52.5}
    req["pitch_x"] = 104.0
    assert inspect(req, path)["status"] == "fail"
    req["x_axis"] = [1.0, 0.0, 0.1]
    with pytest.raises(ValidationError, match="perpendicular"):
        _ADAPTER.validate_python(req)
