"""Known finite voids must not become through-hole, thread or physical proof."""

from __future__ import annotations

import copy

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.assembly import occurrence_label
from forge_mind.workspace.completion_models import PreparedCheck
from forge_mind.workspace.product_tools import (
    _ADAPTER,
    _geometry,
    available_product_tools,
    product_tool_schema,
    run_product_tool,
)


def cavity_request(**changes):
    return {
        "tool": "axial_cavity",
        "units": "mm",
        "body": {"body_index": 0},
        "entry_point": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "diameter": 4.0,
        "depth_interval": {"low": 0.0, "high": 4.0},
        **changes,
    }


def panel(depth=4.0):
    body = b.Box(30, 30, 12, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    return body - b.Solid.make_cylinder(2, depth + 1, b.Plane(origin=(0, 0, -1)))


def inspect(tmp_path, body, request=None):
    path = tmp_path / "part.step"
    b.export_step(body, path)
    return _geometry(_ADAPTER.validate_python(request or cavity_request()), path, file_sha256(path))


@pytest.mark.parametrize("depth", [4.0, 13.0])
def test_finite_interval_in_blind_or_through_void_has_scoped_result(tmp_path, depth):
    receipt = inspect(tmp_path, panel(depth))
    result = receipt["results"]
    assert receipt["status"] == "pass"
    assert receipt["qualification_status"] == "unverified"
    assert result["entry_status"] == "supported_planar_mouth"
    assert result["terminal_depth_status"] == "not_measured"
    assert result["through_status"] == "unverified"
    assert result["requested_envelope_diameter_mm"] == 4
    assert result["continuous_corridor"]["obstruction_mm3"] <= 1e-6
    assert result["exterior_approach"]["obstruction_mm3"] <= 1e-6
    assert result["exterior_approach"]["interval_mm"][0] < result["body_axis_support_mm"][0]
    assert receipt["artifact_hashes"]["part.step"] == file_sha256(tmp_path / "part.step")


def test_unchanged_through_predicate_still_fails_blind_cavity(tmp_path):
    body = panel()
    finite = inspect(tmp_path, body)
    old = {
        "tool": "hole_pattern",
        "units": "mm",
        "body": {"body_index": 0},
        "axis": [0.0, 0.0, 1.0],
        "holes": [{"center": [0.0, 0.0, 0.0], "diameter": 4.0}],
    }
    through = inspect(tmp_path, body, old)
    assert finite["status"] == "pass"
    assert through["status"] == "fail"
    assert through["algorithm_version"] == 2
    assert through["results"]["holes"][0]["wall_matched"]
    assert through["results"]["holes"][0]["obstruction_mm3"] > 0


@pytest.mark.parametrize(
    "change",
    [
        {"entry_point": [0.0, 0.0, -1.0]},
        {"entry_point": [0.0, 0.0, 1.0]},
        {"entry_point": [1.0, 0.0, 0.0]},
        {"axis": [0.0, 0.0, -1.0]},
        {"axis": [1.0, 0.0, 0.0]},
    ],
)
def test_wrong_entry_or_axis_never_supplies_accessible_depth(tmp_path, change):
    receipt = inspect(tmp_path, panel(), cavity_request(**change))
    assert receipt["status"] == "unverified"
    assert receipt["results"]["entry_status"] == "unverified"


@pytest.mark.parametrize(
    "change",
    [
        {"diameter": 4.2},
        {"depth_interval": {"low": 0.0, "high": 4.2}},
    ],
)
def test_insufficient_diameter_or_blind_depth_fails(tmp_path, change):
    receipt = inspect(tmp_path, panel(), cavity_request(**change))
    assert receipt["status"] == "fail"
    assert receipt["results"]["continuous_corridor"]["obstruction_mm3"] > 1e-6


def test_rotated_reversed_and_translated_mouth_uses_requested_frame(tmp_path):
    body = panel().rotate(b.Axis.X, 180).translate((6, -7, 9))
    request = cavity_request(entry_point=[6.0, -7.0, 9.0], axis=[0.0, 0.0, -2.0])
    receipt = inspect(tmp_path, body, request)
    assert receipt["status"] == "pass"
    assert receipt["results"]["axis"] == pytest.approx([0, 0, -1])
    assert receipt["results"]["entry_point_mm"] == pytest.approx([6, -7, 9])


def counterbore():
    body = panel(6)
    return body - b.Solid.make_cylinder(4, 3, b.Plane(origin=(0, 0, -1)))


def test_counterbore_retains_diameters_and_measures_continuous_envelope(tmp_path):
    receipt = inspect(
        tmp_path, counterbore(), cavity_request(depth_interval={"low": 0.0, "high": 5.0})
    )
    assert receipt["status"] == "pass"
    assert receipt["results"]["mouth_candidates"][0]["diameter_mm"] == pytest.approx(8)
    assert sorted(
        {round(w["diameter_mm"], 6) for w in receipt["results"]["cylindrical_walls"]}
    ) == [
        4,
        8,
    ]
    large = inspect(
        tmp_path,
        counterbore(),
        cavity_request(diameter=8.0, depth_interval={"low": 0.0, "high": 5.0}),
    )
    assert large["status"] == "fail"


def test_internal_counterbore_shoulder_is_not_a_new_outer_mouth(tmp_path):
    receipt = inspect(tmp_path, counterbore(), cavity_request(entry_point=[0.0, 0.0, 2.0]))
    assert receipt["status"] == "unverified"
    assert "inside an existing coaxial cavity" in receipt["results"]["reason"]


def test_intermediate_plug_blocks_even_when_requested_interval_skips_it(tmp_path):
    plug = b.Solid.make_cylinder(2.1, 0.2, b.Plane(origin=(0, 0, 1)))
    body = panel(6).fuse(plug)
    receipt = inspect(tmp_path, body, cavity_request(depth_interval={"low": 2.0, "high": 4.0}))
    assert receipt["status"] == "fail"
    corridor = receipt["results"]["continuous_corridor"]
    assert corridor["interval_mm"] == [0.0, 4.0]
    assert corridor["obstruction_mm3"] > 1e-6
    assert corridor["components"]
    assert len(receipt["results"]["cylindrical_walls"]) >= 2


def test_exterior_approach_detects_a_cover_before_the_mouth(tmp_path):
    cap = b.Box(30, 30, 1, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (0, 0, -2)
    )
    bridge = b.Box(2, 2, 3, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (10, 0, -2)
    )
    body = panel().fuse(cap, bridge)
    receipt = inspect(tmp_path, body)
    assert receipt["status"] == "fail"
    assert receipt["results"]["exterior_approach"]["obstruction_mm3"] > 1e-6
    assert receipt["results"]["entry_status"] == "unverified"


def test_sealed_pocket_has_no_external_mouth(tmp_path):
    body = b.Box(30, 30, 12, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    body -= b.Solid.make_cylinder(2, 4, b.Plane(origin=(0, 0, 2)))
    receipt = inspect(tmp_path, body, cavity_request(entry_point=[0.0, 0.0, 2.0]))
    assert receipt["status"] != "pass"


def test_chamfered_or_nonround_mouth_is_explicitly_unsupported(tmp_path):
    chamfered = panel() - b.Solid.make_cone(3, 2, 1, b.Plane(origin=(0, 0, 0)))
    assert inspect(tmp_path, chamfered)["status"] == "unverified"
    square = b.Box(30, 30, 12, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    square -= b.Box(5, 5, 5, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (0, 0, -1)
    )
    assert inspect(tmp_path, square)["status"] == "unverified"


@pytest.mark.parametrize("reverse", [False, True])
def test_named_body_identity_survives_reordered_exports(tmp_path, reverse):
    target = panel()
    target.label = occurrence_label("receiver")
    other = b.Box(3, 4, 5).translate((40, 0, 0))
    other.label = occurrence_label("other")
    bodies = [target, other]
    if reverse:
        bodies.reverse()
    receipt = inspect(
        tmp_path,
        b.Compound(children=bodies),
        cavity_request(body={"occurrence_path": "receiver"}),
    )
    assert receipt["status"] == "pass"
    assert receipt["results"]["selector"] == {"occurrence_path": "receiver"}


@pytest.mark.parametrize(
    "change",
    [
        {"depth_interval": {"low": -1.0, "high": 4.0}},
        {"depth_interval": {"low": 4.0, "high": 4.0}},
        {"depth_interval": {"low": 4.0, "high": 3.0}},
        {"depth_interval": {"low": 0.0, "high": float("inf")}},
        {"axis": [0.0, 0.0, 0.0]},
        {"diameter": 0.0},
        {"terminal_depth": 4.0},
    ],
)
def test_cavity_schema_rejects_unbounded_or_unsupported_claims(change):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(cavity_request(**change))


def test_cavity_is_discoverable_and_roundtrips_as_prepared_request():
    request = cavity_request()
    prepared = PreparedCheck(id="blind_interval", request=copy.deepcopy(request))
    assert prepared.model_dump()["request"]["tool"] == "axial_cavity"
    assert any(tool["tool"] == "axial_cavity" for tool in available_product_tools())
    assert product_tool_schema("axial_cavity")["additionalProperties"] is False


def test_missing_saved_step_does_not_supply_a_measurement(tmp_path):
    receipt = run_product_tool(cavity_request(), tmp_path)
    assert receipt["status"] == "unverified"
    assert receipt["qualification_status"] == "unverified"
