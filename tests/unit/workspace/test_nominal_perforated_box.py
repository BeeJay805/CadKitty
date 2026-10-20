"""Saved analytic solids challenge the complete rectangular material predicate."""

from __future__ import annotations

import copy

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.nominal_perforated_box import (
    PROTOCOL,
    _residual,
    evaluate_nominal_perforated_box,
)
from forge_mind.workspace.processing import process_revision


def request(axis="z", center=(0.0, 0.0, 0.0), size=(30.0, 20.0, 6.0)):
    return {
        "protocol": PROTOCOL,
        "criterion_hash": "b" * 64,
        "subject_id": "plate",
        "part_id": "plate_body",
        "frame": "definition_local",
        "axis": axis,
        "count_policy": "exact",
        "size_mm": list(size),
        "center_mm": list(center),
        "holes": [
            {
                "center_mm": [-8.0, 0.0],
                "diameter_mm": 4.0,
                "diameter_tolerance_mm": 0.1,
                "position_tolerance_mm": 0.1,
            }
        ],
    }


def plate(center=(-8.0, 0.0), diameter=4.0):
    return b.Box(30, 20, 6) - b.Solid.make_cylinder(diameter / 2, 8, b.Plane(origin=(*center, -4)))


def evaluate(tmp_path, body, req=None):
    path = tmp_path / "part.step"
    b.export_step(body, path)
    return evaluate_nominal_perforated_box(path, req or request())


def test_centered_nominal_body_binds_saved_step_and_not_mass_centroid(tmp_path, monkeypatch):
    body = plate()
    assert abs(body.center(b.CenterOf.MASS).X) > 0.01
    b.export_step(body, tmp_path / "part.step")
    original = b.import_step
    calls = []

    def counted(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(b, "import_step", counted)
    result = evaluate_nominal_perforated_box(tmp_path / "part.step", request())
    assert result["status"] == "pass", result
    assert len(calls) == 1
    assert result["method"] == PROTOCOL
    assert result["criterion_hash"] == "b" * 64
    assert result["facts"]["step_sha256"] == file_sha256(tmp_path / "part.step")
    assert result["facts"]["qualification_status"] == "unverified"
    facts = result["facts"]["body"]
    assert facts["actual_bbox_min_mm"] == pytest.approx([-15, -10, -3])
    assert facts["actual_bbox_max_mm"] == pytest.approx([15, 10, 3])
    for key in ("excess_material", "missing_material"):
        assert facts[key]["status"] == "pass"
        assert facts[key]["volume_mm3"] == 0
        assert facts[key]["components"] == []


@pytest.mark.parametrize("diameter,offset", [(4.05, 0.0), (4.0, 0.05), (3.95, -0.05)])
def test_verified_actual_cutters_preserve_source_hole_allowances(tmp_path, diameter, offset):
    result = evaluate(tmp_path, plate((-8 + offset, 0), diameter))
    assert result["status"] == "pass", result
    cutter = result["facts"]["body"]["actual_cutters"][0]
    assert cutter["radius_mm"] == pytest.approx(diameter / 2)
    assert cutter["origin_mm"][0] == pytest.approx(-8 + offset)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_thickness_axis_and_explicit_nonzero_outer_center(tmp_path, axis):
    body = plate()
    if axis == "x":
        body = body.rotate(b.Axis.Y, 90)
        req = request(axis="x", size=(6.0, 20.0, 30.0))
        req["holes"][0]["center_mm"] = [0.0, 8.0]
    elif axis == "y":
        body = body.rotate(b.Axis.X, 90)
        req = request(axis="y", size=(30.0, 6.0, 20.0))
    else:
        req = request()
    translation = [3.0, 4.0, 7.0]
    req["center_mm"] = translation
    transverse = [index for index in range(3) if index != "xyz".index(axis)]
    req["holes"][0]["center_mm"] = [
        value + translation[index]
        for value, index in zip(req["holes"][0]["center_mm"], transverse, strict=True)
    ]
    result = evaluate(tmp_path, body.translate(tuple(translation)), req)
    assert result["status"] == "pass", result


def test_outer_translation_cannot_borrow_hole_position_tolerance(tmp_path):
    result = evaluate(tmp_path, plate().translate((0.02, 0, 0)))
    assert result["facts"]["pattern"]["evaluation_status"] == "pass"
    assert result["status"] == "fail", result
    assert "outer stock bounds" in result["reason"]


@pytest.mark.parametrize("defect", ["pocket", "hidden_void", "notch", "thin_groove"])
def test_same_outer_bounds_do_not_hide_missing_material(tmp_path, defect):
    body = plate()
    if defect == "pocket":
        cutter = b.Box(3, 2, 2).translate((5, 0, 3))
    elif defect == "hidden_void":
        cutter = b.Box(2, 2, 1).translate((5, 0, 0))
    elif defect == "notch":
        cutter = b.Box(4, 4, 8).translate((15, 10, 0))
    else:
        cutter = b.Box(20, 0.001, 0.2).translate((1, 5, 3))
    result = evaluate(tmp_path, body - cutter)
    assert result["status"] != "pass", result
    facts = result["facts"]["body"]
    assert facts["actual_bbox_min_mm"] == pytest.approx([-15, -10, -3])
    assert facts["actual_bbox_max_mm"] == pytest.approx([15, 10, 3])
    if defect in {"pocket", "thin_groove"}:
        assert result["status"] == "unverified"
        assert "missing_material" not in facts
        assert "aperture inventory" in result["reason"]
        assert any(
            "aperture loop" in row["reason"]
            for row in result["facts"]["pattern"]["unsupported_surfaces"]
        )
    else:
        assert result["status"] == "fail", result
        missing = facts["missing_material"]
        assert missing["volume_mm3"] > 1e-6
        assert missing["components"]
        assert all(row["frame"] == "definition_local" for row in missing["components"])


def test_extra_boss_and_wrong_dimensions_cannot_pass(tmp_path):
    boss = plate().fuse(b.Box(2, 2, 2).translate((5, 0, 3)))
    assert evaluate(tmp_path, boss)["status"] == "fail"
    req = request(size=(30.0, 20.0, 7.0))
    assert evaluate(tmp_path, plate(), req)["status"] == "fail"


@pytest.mark.parametrize("diameter,offset", [(4.1, 0.0), (4.2, 0.0), (4.0, 0.1)])
def test_body_proof_cannot_promote_failed_or_ambiguous_hole_bands(tmp_path, diameter, offset):
    result = evaluate(tmp_path, plate((-8 + offset, 0), diameter))
    assert result["status"] != "pass", result
    assert "actual_cutters" not in result["facts"].get("body", {})


def test_blind_and_extra_holes_multiple_solids_and_wrong_axis_never_pass(tmp_path):
    blind = b.Box(30, 20, 6) - b.Solid.make_cylinder(2, 3, b.Plane(origin=(-8, 0, 1)))
    extra = plate() - b.Solid.make_cylinder(2, 8, b.Plane(origin=(8, 0, -4)))
    multiple = b.Compound(children=[plate(), b.Box(1, 1, 1).translate((40, 0, 0))])
    for body in (blind, extra, multiple):
        assert evaluate(tmp_path, body)["status"] != "pass"
    assert evaluate(tmp_path, plate(), request(axis="x"))["status"] != "pass"


@pytest.mark.parametrize("side,status", [(0.005, "unverified"), (0.02, "fail")])
def test_native_positive_residual_is_never_rounded_to_empty(side, status):
    facts = _residual(b.Box(side, side, side))
    assert facts["status"] == status
    assert facts["volume_mm3"] > 0
    assert len(facts["components"]) == 1


def test_zero_volume_face_debris_cannot_prove_empty_material():
    facts = _residual(b.Rectangle(3, 2).faces()[0])
    assert facts["status"] == "unverified"
    assert facts["debris"] is True


def test_empty_native_solid_is_not_an_empty_material_result():
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Solid

    wrapped = TopoDS_Solid()
    BRep_Builder().MakeSolid(wrapped)
    facts = _residual(b.Solid(wrapped))
    assert facts["status"] == "unverified"


def test_open_native_shell_is_invalid_material_topology():
    from OCP.BRep import BRep_Builder
    from OCP.TopoDS import TopoDS_Shell, TopoDS_Solid

    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)
    builder.Add(shell, b.Rectangle(3, 2).faces()[0].wrapped)
    wrapped = TopoDS_Solid()
    builder.MakeSolid(wrapped)
    builder.Add(wrapped, shell)
    facts = _residual(b.Solid(wrapped))
    assert facts["status"] == "unverified"
    assert facts["invalid_topology"] is True


def test_native_boolean_error_and_mutating_step_remain_unverified(tmp_path, monkeypatch):
    import forge_mind.workspace.nominal_perforated_box as module

    def broken(*_):
        raise ValueError("Controlled Boolean failure")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_native_cut", broken)
        result = evaluate(tmp_path, plate())
    assert result["status"] == "unverified"
    assert result["evidence_refs"] == []
    calls = []
    real_hash = file_sha256(tmp_path / "part.step")

    def changed(_):
        calls.append(True)
        return real_hash if len(calls) == 1 else "0" * 64

    monkeypatch.setattr(module, "file_sha256", changed)
    result = evaluate_nominal_perforated_box(tmp_path / "part.step", request())
    assert result["status"] == "unverified"
    assert result["evidence_refs"] == []
    assert "step_sha256" not in result["facts"]


@pytest.mark.parametrize(
    "change",
    [
        {"size_mm": [30.0, 20.0, 0.0]},
        {"center_mm": [0.0, 0.0, float("nan")]},
        {"holes": []},
        {"pattern_requirement_id": "arbitrary"},
        {"protocol": "nominal_through_hole_pattern_v1"},
    ],
)
def test_closed_body_request_rejects_unsupported_authority(tmp_path, change):
    with pytest.raises(ValidationError):
        evaluate_nominal_perforated_box(tmp_path / "part.step", {**request(), **change})


def test_actual_bounded_body_worker_and_cancellation(tmp_path):
    b.export_step(plate(), tmp_path / "part.step")
    req = request()
    before = copy.deepcopy(req)
    result = process_revision(
        tmp_path, {"operation": "nominal_perforated_box", "request": req}, 30, lambda: False
    )
    assert result["status"] == "pass", result
    assert result["method"] == PROTOCOL
    assert req == before
    with pytest.raises(ValueError, match="cancelled"):
        process_revision(
            tmp_path, {"operation": "nominal_perforated_box", "request": req}, 30, lambda: True
        )


def test_finite_body_targets_have_no_arbitrary_worker_size_cap(tmp_path):
    req = request(center=(200001.0, -200001.0, 0.0), size=(200001.0, 20.0, 6.0))
    result = evaluate_nominal_perforated_box(tmp_path / "missing.step", req)
    assert result["status"] == "unverified"
    assert "missing" in result["reason"]
