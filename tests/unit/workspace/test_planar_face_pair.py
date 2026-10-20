"""Exact exported face identity and scoped plane measurements, never contact proof."""

import math

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.assembly import named_solids, occurrence_label
from forge_mind.workspace.product_tools import _ADAPTER, run_product_tool
from forge_mind.workspace.topology import signature


def fixture(tmp_path, *, holes=False, offset=0, rotation=0, b_rotation=0, outer=False):
    pad = b.Box(20, 12, 8)
    if holes:
        pad -= b.Cylinder(2, 30).rotate(b.Axis.Y, 90)
    shapes = []
    for path, position, tilt in [("left", (-50, 0, 0), 0), ("right", (50, offset, 0), b_rotation)]:
        body = pad.rotate(b.Axis.Z, tilt).translate(position).rotate(b.Axis.Z, rotation)
        body.label = occurrence_label(path)
        shapes.append(body)
    step = tmp_path / "part.step"
    b.export_step(b.Compound(children=shapes), step)
    imported = b.import_step(step)
    named = named_solids(imported)
    selectors = []
    for i, path in enumerate(["left", "right"]):
        angle = math.radians(rotation + (b_rotation if i else 0))
        sign = -1 if i else 1
        if outer:
            sign *= -1
        expected = (sign * math.cos(angle), sign * math.sin(angle), 0)
        face = next(
            f
            for f in named[occurrence_label(path)].faces()
            if f.geom_type == b.GeomType.PLANE
            and sum((u - v) ** 2 for u, v in zip(f.normal_at(), expected, strict=True)) < 1e-12
        )
        selectors.append({"body": {"occurrence_path": path}, "face_signature": signature(face)})
    return {
        "tool": "planar_face_pair",
        "units": "mm",
        "a": selectors[0],
        "b": selectors[1],
        "datum": [0, 0, 0],
    }, imported


def test_plane_identity_hole_trim_area_and_targets(tmp_path):
    request, _ = fixture(tmp_path, holes=True)
    receipt = run_product_tool(request, tmp_path)
    assert receipt["status"] == receipt["qualification_status"] == "unverified"
    assert receipt["artifact_hashes"]["part.step"] == file_sha256(tmp_path / "part.step")
    measured = receipt["results"]
    assert measured["ordered_separation_mm"] == pytest.approx(80)
    assert measured["boundary_midpoint_mm"] == pytest.approx(0)
    for face in (measured["a"], measured["b"]):
        assert face["area_mm2"] == pytest.approx(12 * 8 - math.pi * 2**2)
        assert face["face_signature"] == face["selector"]["face_signature"]
        assert len(face["body_signature"]) == 64
    assert measured["a"]["normal"] == pytest.approx([1, 0, 0])
    assert measured["b"]["normal"] == pytest.approx([-1, 0, 0])
    assert measured["a"]["plane_dimension_point_mm"] == pytest.approx([-40, 0, 0])
    # Endpoints are in the holes: they locate infinite planes, never material/contact points.
    assert "not necessarily inside trimmed faces" in receipt["algorithm_scope"]
    request.update(expected_separation_mm=80, expected_midpoint_mm=0, tolerance_mm=1e-5)
    assert run_product_tool(request, tmp_path)["status"] == "pass"
    request["expected_separation_mm"] = 79
    assert run_product_tool(request, tmp_path)["status"] == "fail"


def test_offset_faces_rotation_and_datum_do_not_invent_overlap(tmp_path):
    request, _ = fixture(tmp_path, offset=200, rotation=37)
    angle = math.radians(37)
    axis = [math.cos(angle), math.sin(angle), 0]
    request.update(
        datum=[7 * v for v in axis],
        expected_separation_mm=80,
        expected_midpoint_mm=-7,
        tolerance_mm=1e-5,
    )
    receipt = run_product_tool(request, tmp_path)
    assert receipt["status"] == "pass"
    assert receipt["results"]["axis"] == pytest.approx(axis)
    assert (
        receipt["results"]["a"]["center_mm"] != receipt["results"]["a"]["plane_dimension_point_mm"]
    )
    assert "no overlap/contact area" in receipt["algorithm_scope"]


def test_opposite_outer_faces_produce_signed_plane_separation(tmp_path):
    request, _ = fixture(tmp_path, outer=True)
    result = run_product_tool(request, tmp_path)["results"]
    assert result["axis"] == pytest.approx([-1, 0, 0])
    assert result["ordered_separation_mm"] == pytest.approx(-120)


@pytest.mark.parametrize("rotation, measured", [(1e-6, True), (1e-4, False), (180, False)])
def test_parallel_opposition_instrument_bound(tmp_path, rotation, measured):
    request, _ = fixture(tmp_path, b_rotation=rotation)
    receipt = run_product_tool(request, tmp_path)
    assert (receipt["results"].get("measurement_status") == "measured") is measured
    if not measured:
        assert receipt["status"] == "unverified"


def test_nonplanar_face_and_same_body_alias_remain_unverified(tmp_path):
    request, shape = fixture(tmp_path, holes=True)
    bodies = named_solids(shape)
    curved = next(
        f for f in bodies[occurrence_label("left")].faces() if f.geom_type == b.GeomType.CYLINDER
    )
    original = request["a"]["face_signature"]
    request["a"]["face_signature"] = signature(curved)
    receipt = run_product_tool(request, tmp_path)
    assert receipt["status"] == "unverified"
    assert "not an analytic plane" in receipt["results"]["reason"]
    request["a"]["face_signature"] = original
    outward = next(
        f
        for f in bodies[occurrence_label("left")].faces()
        if f.geom_type == b.GeomType.PLANE and f.normal_at().X < -0.999
    )
    request["b"] = {"body": {"body_index": 0}, "face_signature": signature(outward)}
    assert run_product_tool(request, tmp_path)["status"] == "unverified"


def test_missing_or_ambiguous_face_signature_cannot_measure(tmp_path, monkeypatch):
    request, _ = fixture(tmp_path)
    request["a"]["face_signature"] = "0" * 64
    assert run_product_tool(request, tmp_path)["status"] == "unverified"
    monkeypatch.setattr("forge_mind.workspace.topology.signature", lambda _: "0" * 64)
    receipt = run_product_tool(request, tmp_path)
    assert receipt["status"] == "unverified"
    assert "ambiguous" in receipt["results"]["reason"]


@pytest.mark.parametrize(
    "update",
    [
        {"datum": [math.nan, 0, 0]},
        {"datum": [math.inf, 0, 0]},
        {"expected_separation_mm": math.inf},
        {"expected_midpoint_mm": True},
        {"expected_separation_mm": 80},
        {"expected_midpoint_mm": 0},
        {"tolerance_mm": 1e-6},
        {"tolerance_mm": -1},
        {"tolerance_mm": math.nan},
    ],
)
def test_invalid_or_unstated_comparison_precision_rejected(update):
    selected = {"body": {"body_index": 0}, "face_signature": "a" * 64}
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(
            {
                "tool": "planar_face_pair",
                "units": "mm",
                "a": selected,
                "b": selected,
                "datum": [0, 0, 0],
                **update,
            }
        )
