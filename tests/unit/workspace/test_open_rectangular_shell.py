"""Authored shell expectations stay separate from original source acceptance."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.completion_models import PreparedCheck
from forge_mind.workspace.nominal_open_rectangular_shell import (
    OpenRectangularShellSpec,
    shell_boxes,
)
from forge_mind.workspace.product_tools import (
    _ADAPTER,
    OpenRectangularShell,
    available_product_tools,
    product_tool_schema,
    run_product_tool,
)


def request():
    return {
        "tool": "open_rectangular_shell",
        "units": "mm",
        "body": {"body_index": 0},
        "size_mm": [80.0, 50.0, 20.0],
        "wall_thickness_mm": 3.0,
        "floor_thickness_mm": 3.0,
        "opening_axis": "z",
        "opening_direction": "positive",
    }


def spec(**changes):
    return OpenRectangularShellSpec.model_validate(
        {
            key: value
            for key, value in {**request(), **changes}.items()
            if key not in {"tool", "units", "body"}
        }
    )


def test_discoverable_request_and_existing_prepared_check_transport():
    req = request()
    parsed = _ADAPTER.validate_python(req)
    assert isinstance(parsed, OpenRectangularShell)
    assert parsed.model_dump(mode="json", exclude_none=True) == req
    check = PreparedCheck.model_validate(
        {"id": "shell", "request": req, "reference_revision_id": "a" * 32}
    )
    assert check.request == parsed
    assert (
        product_tool_schema("open_rectangular_shell")["properties"]["tool"]["const"]
        == "open_rectangular_shell"
    )
    scope = next(
        row["scope"] for row in available_product_tools() if row["tool"] == "open_rectangular_shell"
    )
    assert "matches_nominal_plan" in scope and "source completeness" in scope


@pytest.mark.parametrize(
    "field,value",
    [
        ("size_mm", [0, 50, 20]),
        ("size_mm", [80, float("inf"), 20]),
        ("wall_thickness_mm", 25),
        ("floor_thickness_mm", 20),
        ("wall_thickness_mm", "3"),
        ("floor_thickness_mm", True),
        ("floor_thickness_mm", float("nan")),
        ("opening_axis", "top"),
        ("opening_direction", "auto"),
        ("units", "inch"),
        ("body", {"body_index": 0, "occurrence_path": "tray"}),
        ("body", {}),
        ("body", {"body_index": -1}),
        ("source_quote", "This forged source must not create authority"),
        ("verification", {"status": "pass"}),
        ("status", "pass"),
        ("center_mm", [0, 0, 0]),
        ("acceptance_scope", "contract_satisfied"),
    ],
)
def test_invalid_or_authority_forging_requests_are_rejected(field, value):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python({**request(), field: value})


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_box_arithmetic_keeps_fixed_dimensions_and_explicit_open_side(axis, direction):
    nominal = spec(opening_axis=axis, opening_direction=direction)
    before = nominal.model_dump()
    center = (10.0, -4.0, 25.0)
    boxes = shell_boxes(nominal, center)
    depth = "xyz".index(axis)
    assert boxes["outer_size_mm"] == [80, 50, 20]
    lower = [c - s / 2 for c, s in zip(center, nominal.size_mm, strict=True)]
    assert boxes["outer_origin_mm"] == lower
    assert boxes["cavity_size_mm"][depth] == nominal.size_mm[depth] - 3 + 1
    for index in range(3):
        if index != depth:
            assert boxes["cavity_origin_mm"][index] == lower[index] + 3
            assert boxes["cavity_size_mm"][index] == nominal.size_mm[index] - 6
    if direction == "positive":
        assert boxes["cavity_origin_mm"][depth] == lower[depth] + 3
    else:
        assert boxes["cavity_origin_mm"][depth] == lower[depth] - 1
    assert nominal.model_dump() == before


def test_missing_saved_step_cannot_supply_a_nominal_pass(tmp_path):
    before = deepcopy(request())
    result = run_product_tool(before, tmp_path)
    assert result["status"] == "unverified"
    assert result["qualification_status"] == "unverified"
    assert before == request()


@pytest.mark.parametrize("center", [(0.0, 0.0, 1e17), (1e17, 0.0, 0.0), (float("inf"), 0.0, 0.0)])
def test_unrepresentable_fixed_boundaries_are_refused(center):
    with pytest.raises(ValueError, match=r"[Bb]ound|[Rr]esol|[Ff]inite"):
        shell_boxes(spec(), center)


@pytest.mark.parametrize(
    "changes",
    [
        {"floor_thickness_mm": 1e-9},
        {"wall_thickness_mm": 1e-9},
        {"floor_thickness_mm": 20.0 - 1e-9},
        {"wall_thickness_mm": 25.0 - 1e-9},
    ],
)
def test_kernel_resolution_refuses_indistinguishable_nominal_intervals(changes):
    from forge_mind.workspace.nominal_open_rectangular_shell import check_boundary_resolution

    nominal = spec(**changes)
    with pytest.raises(ValueError, match=r"[Bb]ound|[Rr]esol"):
        check_boundary_resolution(nominal, shell_boxes(nominal, (0.0, 0.0, 0.0)), 1e-7)


def test_kernel_resolution_refuses_rounding_that_changes_fixed_thickness():
    from forge_mind.workspace.nominal_open_rectangular_shell import check_boundary_resolution

    nominal = spec(wall_thickness_mm=3.1)
    boxes = shell_boxes(nominal, (1e12, 0.0, 0.0))
    with pytest.raises(ValueError, match=r"[Bb]ound|[Rr]esol"):
        check_boundary_resolution(nominal, boxes, 1e-7)


@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_ordinary_fixed_intervals_survive_resolution_check(axis, direction):
    from forge_mind.workspace.nominal_open_rectangular_shell import check_boundary_resolution

    nominal = spec(opening_axis=axis, opening_direction=direction)
    check_boundary_resolution(nominal, shell_boxes(nominal, (10.0, -4.0, 25.0)), 1e-7)


@pytest.mark.parametrize("resolution", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_kernel_resolution_cannot_authorize_comparison(resolution):
    from forge_mind.workspace.nominal_open_rectangular_shell import check_boundary_resolution

    nominal = spec()
    with pytest.raises(ValueError, match="resolution"):
        check_boundary_resolution(nominal, shell_boxes(nominal, (0.0, 0.0, 0.0)), resolution)


def _shape(*, wall=3.0, floor=3.0):
    import build123d as b

    return b.Box(80, 50, 20) - b.Box(80 - 2 * wall, 50 - 2 * wall, 21 - floor).translate(
        (0, 0, (floor + 1) / 2)
    )


@pytest.mark.slow
@pytest.mark.parametrize("axis", "xyz")
@pytest.mark.parametrize("direction", ["positive", "negative"])
def test_native_exact_shell_all_open_sides_and_translation(tmp_path, axis, direction):
    import build123d as b

    shape = _shape()
    req = request()
    if axis == "x":
        shape = shape.rotate(b.Axis.Y, 90)
        req["size_mm"] = [20.0, 50.0, 80.0]
    elif axis == "y":
        shape = shape.rotate(b.Axis.X, -90)
        req["size_mm"] = [80.0, 20.0, 50.0]
    if direction == "negative":
        shape = shape.rotate(b.Axis.Y if axis != "y" else b.Axis.X, 180)
    req.update(opening_axis=axis, opening_direction=direction)
    shape = shape.translate((10, -4, 25))
    b.export_step(shape, tmp_path / "part.step")
    result = run_product_tool(req, tmp_path)
    assert result["status"] == "pass", result
    observed = result["results"]
    assert observed["result_scope"] == "matches_nominal_plan"
    assert observed["frame"] == "cad_z_up_mm_world"
    assert observed["registration"]["center_mm"] == pytest.approx([10, -4, 25])
    assert observed["registration"]["rotation_fitted"] is False
    assert observed["expected"]["size_mm"] == req["size_mm"]
    assert result["qualification_status"] == "unverified"
    assert observed["excess_material"]["volume_mm3"] == 0
    assert observed["missing_material"]["volume_mm3"] == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "defect", ["wall", "floor", "lid", "hole", "rounds", "equal_volume", "wrong_axis", "wrong_size"]
)
def test_native_material_defects_cannot_match_the_nominal_plan(tmp_path, defect):
    import build123d as b

    shape = _shape(wall=4 if defect == "wall" else 3, floor=4 if defect == "floor" else 3)
    req = request()
    if defect == "lid":
        shape = shape.fuse(b.Box(80, 50, 1).translate((0, 0, 9.5)))
    elif defect == "hole":
        shape = shape - b.Solid.make_cylinder(1, 5, b.Plane(origin=(0, 0, -11)))
    elif defect == "rounds":
        shape = shape.fillet(0.5, shape.edges().filter_by(b.Axis.Z))
    elif defect == "equal_volume":
        shape = (shape - b.Box(1, 1, 1).translate((0, 0, -9.5))).fuse(
            b.Box(1, 1, 1).translate((0, 0, -6.5))
        )
        assert shape.volume == pytest.approx(_shape().volume)
    elif defect == "wrong_axis":
        req["opening_axis"] = "x"
    elif defect == "wrong_size":
        req["size_mm"] = [81.0, 50.0, 20.0]
    b.export_step(shape, tmp_path / "part.step")
    result = run_product_tool(req, tmp_path)
    assert result["status"] == "fail", result
    assert result["results"]["expected"]["wall_thickness_mm"] == 3
    assert result["results"]["expected"]["floor_thickness_mm"] == 3


@pytest.mark.slow
def test_native_worker_failure_and_mid_read_step_mutation_never_pass(tmp_path, monkeypatch):
    import build123d as b

    import forge_mind.workspace.nominal_open_rectangular_shell as module

    b.export_step(_shape(), tmp_path / "part.step")
    original = module.checked_boolean

    def broken(*args):
        raise ValueError("Controlled incomplete Boolean")

    monkeypatch.setattr(module, "checked_boolean", broken)
    assert run_product_tool(request(), tmp_path)["status"] == "unverified"

    def changed(*args):
        result = original(*args)
        (tmp_path / "part.step").write_bytes(b"Changed immutable input")
        return result

    monkeypatch.setattr(module, "checked_boolean", changed)
    assert run_product_tool(request(), tmp_path)["status"] == "unverified"


@pytest.mark.slow
@pytest.mark.parametrize("feature", ["floor", "wall", "cavity_depth", "cavity_width"])
def test_native_unresolvable_nominal_feature_cannot_match_saved_material(tmp_path, feature):
    import build123d as b

    # A tube must not earn a match when the requested tiny floor vanishes in the kernel.
    b.export_step(_shape(floor=0 if feature == "floor" else 3), tmp_path / "part.step")
    changes = {
        "floor": {"floor_thickness_mm": 1e-9},
        "wall": {"wall_thickness_mm": 1e-9},
        "cavity_depth": {"floor_thickness_mm": 20 - 1e-9},
        "cavity_width": {"wall_thickness_mm": 25 - 1e-9},
    }
    result = run_product_tool({**request(), **changes[feature]}, tmp_path)
    assert result["status"] == "unverified", result
    assert result["qualification_status"] == "unverified"


@pytest.mark.slow
def test_native_large_translation_cannot_hide_fixed_boundary_loss():
    from forge_mind.workspace.nominal_open_rectangular_shell import inspect_open_rectangular_shell

    translated = _shape().translate((0, 0, 1e17))
    try:
        result = inspect_open_rectangular_shell(translated, spec())
    except ValueError as exc:
        assert "bound" in str(exc).lower() or "resol" in str(exc).lower()
    else:
        assert result["status"] == "unverified", result


@pytest.mark.slow
def test_native_missing_expected_boundary_witness_refuses_before_residuals(tmp_path, monkeypatch):
    import build123d as b

    import forge_mind.workspace.nominal_open_rectangular_shell as module

    b.export_step(_shape(), tmp_path / "part.step")
    real_boolean = module.checked_boolean
    calls = []

    def lost_floor(left, right, kind):
        calls.append(kind)
        # Simulate a completed, positive, valid target that silently lost its floor.
        return _shape(floor=0) if len(calls) == 1 else real_boolean(left, right, kind)

    monkeypatch.setattr(module, "checked_boolean", lost_floor)
    result = run_product_tool(request(), tmp_path)
    assert result["status"] == "unverified", result
    assert calls == ["cut"]  # No residual comparison may follow a missing target witness.
