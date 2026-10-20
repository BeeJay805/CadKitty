"""Independent analytic geometry for frozen nominal through-pattern criteria."""

from __future__ import annotations

import copy

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.nominal_through_pattern import (
    COORDINATE_EPSILON_MM,
    PROTOCOL,
    evaluate_nominal_through_pattern,
)
from forge_mind.workspace.processing import process_revision

CENTERS = [(-8.0, 0.0), (8.0, 0.0)]


def request(
    centers=CENTERS, diameter=4.0, diameter_tolerance=0.0, position_tolerance=0.0, axis="z"
):
    return {
        "protocol": PROTOCOL,
        "criterion_hash": "a" * 64,
        "subject_id": "plate",
        "part_id": "plate_body",
        "frame": "definition_local",
        "axis": axis,
        "count_policy": "exact",
        "holes": [
            {
                "center_mm": list(center),
                "diameter_mm": diameter,
                "diameter_tolerance_mm": diameter_tolerance,
                "position_tolerance_mm": position_tolerance,
            }
            for center in centers
        ],
    }


def plate(centers=CENTERS, diameters=(4.0, 4.0), depth=8.0):
    body = b.Box(30, 20, 6, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN))
    for center, diameter in zip(centers, diameters, strict=True):
        body -= b.Solid.make_cylinder(diameter / 2, depth, b.Plane(origin=(*center, -1)))
    return body


def evaluate(tmp_path, body, req=None):
    path = tmp_path / "part.step"
    b.export_step(body, path)
    return evaluate_nominal_through_pattern(path, req or request())


def test_actual_nominal_pattern_has_exact_scope_and_artifact_binding(tmp_path):
    result = evaluate(tmp_path, plate())
    assert result["status"] == "pass", result
    assert result["method"] == PROTOCOL
    assert result["criterion_hash"] == "a" * 64
    facts = result["facts"]
    assert facts["qualification_status"] == "unverified"
    assert facts["step_sha256"] == file_sha256(tmp_path / "part.step")
    assert result["evidence_refs"] == [{"artifact": "part.step", "sha256": facts["step_sha256"]}]
    assert len(facts["observed_holes"]) == 2
    assert [row["diameter_interval_mm"] for row in facts["observed_holes"]] == [[4.0, 4.0]] * 2
    for hole in facts["observed_holes"]:
        assert hole["mouth_witness_counts"] == [1, 1]
        assert hole["through_status"] == "pass"
        assert hole["obstruction_mm3"] <= 1e-6
    assert "No threads" in facts["scope"]


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_axis_permutation_preserves_increasing_transverse_coordinate_order(tmp_path, axis):
    centers = [(-8.0, -3.0), (7.0, 4.0)]
    body = plate(centers)
    if axis == "x":
        body = body.rotate(b.Axis.Y, 90)
        expected = [(-3.0, 8.0), (4.0, -7.0)]
    elif axis == "y":
        body = body.rotate(b.Axis.X, 90)
        expected = centers
    else:
        expected = centers
    result = evaluate(tmp_path, body, request(expected, axis=axis))
    assert result["status"] == "pass", result


def test_rigid_translation_is_measured_in_definition_frame(tmp_path):
    body = plate().translate((3, 4, 7))
    result = evaluate(tmp_path, body, request([(-5.0, 4.0), (11.0, 4.0)]))
    assert result["status"] == "pass", result
    assert evaluate(tmp_path, body)["status"] == "fail"


@pytest.mark.parametrize("diameter,expected", [(3.8, "fail"), (4.05, "pass"), (4.2, "fail")])
def test_diameter_conformance_does_not_filter_the_associated_axis(tmp_path, diameter, expected):
    result = evaluate(tmp_path, plate(diameters=(diameter, 4.0)), request(diameter_tolerance=0.1))
    assert result["status"] == expected, result
    facts = result["facts"]
    assert len(facts["observed_holes"]) == 2
    comparison = next(row for row in facts["comparisons"] if row["target_center_mm"] == [-8.0, 0.0])
    assert comparison["diameter_status"] == expected
    observed = facts["observed_holes"][comparison["observation_index"]]
    assert observed["diameter_interval_mm"] == pytest.approx([diameter, diameter])
    assert observed["through_status"] == "pass"


@pytest.mark.parametrize("diameter", [3.9, 4.1, 4.1 + COORDINATE_EPSILON_MM / 2])
def test_nonzero_diameter_allowance_boundary_is_unverified(tmp_path, diameter):
    result = evaluate(tmp_path, plate(diameters=(diameter, 4.0)), request(diameter_tolerance=0.1))
    assert result["status"] == "unverified", result
    assert result["facts"]["comparisons"][0]["diameter_status"] == "unverified"


@pytest.mark.parametrize("offset,expected", [(0.05, "pass"), (0.1, "unverified"), (0.2, "fail")])
def test_position_allowance_is_frozen_and_numerical_boundary_explicit(tmp_path, offset, expected):
    result = evaluate(
        tmp_path,
        plate([(-8.0 + offset, 0.0), (8.0, 0.0)]),
        request(position_tolerance=0.1),
    )
    assert result["status"] == expected, result


def test_zero_allowance_is_only_computational_nominal_equality(tmp_path):
    result = evaluate(tmp_path, plate(diameters=(4 + COORDINATE_EPSILON_MM / 2, 4.0)))
    assert result["status"] == "pass", result
    policy = result["facts"]["numerical_limits"]["comparison_policy"]
    assert "computational nominal equality" in policy
    assert "No manufacturing tolerance" in policy
    assert (
        evaluate(tmp_path, plate(diameters=(4 + 4 * COORDINATE_EPSILON_MM, 4.0)))["status"]
        == "fail"
    )


@pytest.mark.parametrize("centers", [[(-8.0, 0.0)], [(-8.0, 0.0), (8.0, 0.0), (0.0, 5.0)]])
def test_missing_or_extra_cylindrical_hole_fails_exact_count(tmp_path, centers):
    result = evaluate(tmp_path, plate(centers, (4.0,) * len(centers)))
    assert result["status"] == "fail", result
    assert "count differs" in result["reason"]
    assert len(result["facts"]["observed_holes"]) == len(centers)


def test_square_extra_aperture_is_not_silently_omitted(tmp_path):
    body = plate() - b.Box(3, 3, 8, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (0, 5, -1)
    )
    result = evaluate(tmp_path, body)
    assert result["status"] == "unverified", result
    assert any("aperture loop" in row["reason"] for row in result["facts"]["unsupported_surfaces"])


def test_blind_floor_and_thin_midspan_membrane_fail_passage(tmp_path):
    blind = evaluate(tmp_path, plate(depth=5.0))
    assert blind["status"] == "fail", blind
    body = plate()
    body = body.fuse(b.Solid.make_cylinder(2.1, 0.05, b.Plane(origin=(-8, 0, 2.0))))
    membrane = evaluate(tmp_path, body)
    assert membrane["status"] == "fail", membrane
    blocked = next(
        hole for hole in membrane["facts"]["observed_holes"] if hole["center_uv_mm"][0] < 0
    )
    assert blocked["obstruction_mm3"] > 1e-6
    assert blocked["obstruction_components"]
    assert len(blocked["walls"]) >= 2


def test_counterbore_preserves_both_diameters_without_claiming_uniformity(tmp_path):
    body = plate() - b.Solid.make_cylinder(4, 3, b.Plane(origin=(-8, 0, -1)))
    result = evaluate(tmp_path, body)
    assert result["status"] != "pass"
    counterbore = next(
        hole for hole in result["facts"]["observed_holes"] if hole["center_uv_mm"][0] < 0
    )
    assert counterbore["diameter_interval_mm"] == pytest.approx([4, 8])
    assert counterbore["through_status"] == "unverified"
    assert "nonuniform" in counterbore["reason"]


def test_separated_coaxial_spans_do_not_merge_across_submillimeter_gap(tmp_path):
    bottom = plate(depth=8.0)
    top = plate(depth=8.0).translate((0, 0, 6.2))
    bridge = b.Box(2, 2, 13, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (12, 7, 0)
    )
    body = bottom.fuse(top, bridge)
    result = evaluate(tmp_path, body)
    assert result["status"] == "unverified", result
    assert all(hole["through_status"] == "unverified" for hole in result["facts"]["observed_holes"])
    assert any("Separated" in hole["reason"] for hole in result["facts"]["observed_holes"])


def test_partial_cylindrical_slot_end_prevents_complete_inventory(tmp_path):
    body = plate()
    body -= b.Box(8, 4, 8, align=(b.Align.CENTER, b.Align.CENTER, b.Align.MIN)).translate(
        (-4, 0, -1)
    )
    result = evaluate(tmp_path, body)
    assert result["status"] == "unverified", result
    assert result["facts"]["unsupported_surfaces"]


def test_position_only_matching_never_uses_diameter_to_choose_a_pass(tmp_path):
    body = plate([(-3.0, 0.0), (3.0, 0.0)], (4.0, 5.0))
    req = request([(-1.0, 0.0), (1.0, 0.0)], position_tolerance=5.0)
    req["holes"][1]["diameter_mm"] = 5.0
    result = evaluate(tmp_path, body, req)
    assert result["status"] == "unverified", result
    assert "association is ambiguous" in result["reason"]


def test_wrong_axis_multiple_bodies_and_mirrored_frame_never_pass(tmp_path):
    assert evaluate(tmp_path, plate(), request(axis="x"))["status"] == "unverified"
    compound = b.Compound(children=[plate(), b.Box(2, 2, 2).translate((40, 0, 0))])
    assert evaluate(tmp_path, compound)["status"] == "unverified"
    centers = [(-8.0, -3.0), (7.0, 4.0)]
    mirrored = plate(centers).rotate(b.Axis.Z, 180)
    assert evaluate(tmp_path, mirrored, request(centers))["status"] == "fail"


def test_missing_step_and_changed_step_have_no_evidence_authority(tmp_path, monkeypatch):
    path = tmp_path / "part.step"
    assert evaluate_nominal_through_pattern(path, request())["status"] == "unverified"
    b.export_step(plate(), path)
    real_hash = file_sha256(path)
    calls = []

    def changed(_):
        calls.append(True)
        return real_hash if len(calls) == 1 else "0" * 64

    monkeypatch.setattr("forge_mind.workspace.nominal_through_pattern.file_sha256", changed)
    result = evaluate_nominal_through_pattern(path, request())
    assert result["status"] == "unverified"
    assert result["evidence_refs"] == []
    assert "step_sha256" not in result["facts"]


def test_kernel_failure_never_turns_into_zero_obstruction(tmp_path, monkeypatch):
    def broken(_):
        raise ValueError("Controlled invalid Boolean")

    monkeypatch.setattr("forge_mind.workspace.geometry_results.boolean_volume", broken)
    result = evaluate(tmp_path, plate())
    assert result["status"] == "unverified"
    assert result["evidence_refs"] == []


def test_positive_subresolution_obstruction_can_never_prove_full_passage(tmp_path, monkeypatch):
    # Control only the numerical instrument response. A detected positive
    # amount cannot be rounded away even below the reliable failure guard.
    monkeypatch.setattr("forge_mind.workspace.geometry_results.boolean_volume", lambda _: 0.5e-6)
    monkeypatch.setattr(
        "forge_mind.workspace.geometry_results.boolean_overlap_components",
        lambda _: [
            {
                "volume_mm3": 0.5e-6,
                "bbox_min_mm": [-8.0, 0.0, 0.0],
                "bbox_max_mm": [-7.999, 0.001, 0.5],
                "frame": "cad_z_up_mm",
            }
        ],
    )
    result = evaluate(tmp_path, plate())
    assert result["status"] == "unverified", result
    assert all(hole["through_status"] == "unverified" for hole in result["facts"]["observed_holes"])
    assert all(hole["obstruction_mm3"] > 0 for hole in result["facts"]["observed_holes"])


def test_incomplete_native_common_operation_never_proves_zero_obstruction(tmp_path, monkeypatch):
    class Incomplete:
        def __init__(self, *_):
            pass

        def Build(self):  # noqa: N802 - exact OCC boundary method
            pass

        def IsDone(self):  # noqa: N802 - exact OCC boundary method
            return False

        def Shape(self):  # noqa: N802 - exact OCC boundary method
            raise AssertionError("Incomplete operation must not supply a shape")

    monkeypatch.setattr("OCP.BRepAlgoAPI.BRepAlgoAPI_Common", Incomplete)
    result = evaluate(tmp_path, plate())
    assert result["status"] == "unverified", result
    assert result["evidence_refs"] == []


def test_invalid_zero_volume_common_result_never_proves_passage(tmp_path, monkeypatch):
    from OCP.BRep import BRep_Builder
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS_Solid

    wrapped = TopoDS_Solid()
    BRep_Builder().MakeSolid(wrapped)
    invalid = b.Solid(wrapped)
    # OCC accepts this topologically empty solid as valid. It is nevertheless
    # a solid result, not a genuinely empty Boolean compound.
    assert invalid.is_valid
    assert invalid.wrapped.ShapeType() == TopAbs_SOLID
    assert len(invalid.solids()) == 1
    monkeypatch.setattr("forge_mind.workspace.nominal_boolean.checked_boolean", lambda *_: invalid)
    result = evaluate(tmp_path, plate())
    assert result["status"] == "unverified", result
    assert all(hole["through_status"] == "unverified" for hole in result["facts"]["observed_holes"])
    assert all(
        hole["obstruction_classification"]["status"] == "unverified"
        for hole in result["facts"]["observed_holes"]
    )
    assert all(
        hole["obstruction_classification"]["components"] == []
        for hole in result["facts"]["observed_holes"]
    )


def test_finite_worker_targets_have_no_pattern_specific_size_cap(tmp_path):
    req = request(centers=[(200001.0, -200001.0)], diameter=200001.0)
    result = evaluate_nominal_through_pattern(tmp_path / "missing.step", req)
    assert result["status"] == "unverified"
    assert "missing" in result["reason"]


@pytest.mark.parametrize(
    "change",
    [
        {"frame": "world"},
        {"axis": "q"},
        {"count_policy": "at_least"},
        {"criterion_hash": "x" * 64},
        {"status": "pass"},
    ],
)
def test_worker_request_is_closed_and_fixed_scope(tmp_path, change):
    req = {**request(), **change}
    with pytest.raises(ValidationError):
        evaluate_nominal_through_pattern(tmp_path / "part.step", req)


def test_actual_bounded_worker_returns_same_frozen_hash_and_saved_step_evidence(tmp_path):
    b.export_step(plate(), tmp_path / "part.step")
    req = request()
    before = copy.deepcopy(req)
    result = process_revision(
        tmp_path,
        {"operation": "nominal_through_pattern", "request": req},
        30,
        lambda: False,
    )
    assert result["status"] == "pass", result
    assert result["criterion_hash"] == req["criterion_hash"]
    assert result["facts"]["step_sha256"] == file_sha256(tmp_path / "part.step")
    assert req == before


def test_bounded_worker_cancellation_does_not_create_a_result(tmp_path):
    b.export_step(plate(), tmp_path / "part.step")
    with pytest.raises(ValueError, match="cancelled"):
        process_revision(
            tmp_path,
            {"operation": "nominal_through_pattern", "request": request()},
            30,
            lambda: True,
        )
    assert not (tmp_path / "processed.json").exists()
