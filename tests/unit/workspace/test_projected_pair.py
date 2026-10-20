"""Real STEP projection probes are ordered measurements, never frozen acceptance."""

import math
import shutil
from copy import deepcopy

import build123d as b
import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.assembly import occurrence_label
from forge_mind.workspace.product_tools import _ADAPTER, run_product_tool


def request(**updates):
    return {
        "tool": "projected_pair",
        "units": "mm",
        "a": {"occurrence_path": "left/pad"},
        "b": {"occurrence_path": "right/pad"},
        "axis": [1.0, 0.0, 0.0],
        "datum": [0.0, 0.0, 0.0],
        **updates,
    }


def export_pair(tmp_path, *, rotation=0, tilt=0, translation=(0, 0, 0), sideways=0, protrusion=0):
    left = b.Box(20, 8, 6).translate((-50, 0, 0))
    if protrusion:
        left += b.Box(protrusion + 1, 1, 1).translate((-40 + (protrusion - 1) / 2, 0, 0))
    right = b.Box(20, 8, 6).translate((50, sideways, 0))
    children = []
    for name, shape in [("left/pad", left), ("right/pad", right)]:
        shape = shape.rotate(b.Axis.Z, rotation).rotate(b.Axis.Y, tilt).translate(translation)
        shape.label = occurrence_label(name)
        children.append(shape)
    path = tmp_path / "part.step"
    b.export_step(b.Compound(children=children), path)
    return path


def test_measured_intervals_and_explicit_targets(tmp_path):
    path = export_pair(tmp_path)
    receipt = run_product_tool(request(), tmp_path)
    assert receipt["status"] == "unverified"
    assert receipt["qualification_status"] == "unverified"
    result = receipt["results"]
    assert result["measurement_status"] == "measured"
    assert result["a"]["interval_mm"] == pytest.approx([-60, -40], abs=1e-6)
    assert result["b"]["interval_mm"] == pytest.approx([40, 60], abs=1e-6)
    assert result["ordered_separation_mm"] == pytest.approx(80, abs=1e-6)
    assert result["boundary_midpoint_mm"] == pytest.approx(0, abs=1e-6)
    assert receipt["artifact_hashes"]["part.step"] == file_sha256(path)
    targeted = request(expected_separation_mm=80, expected_midpoint_mm=0, tolerance_mm=1e-5)
    assert run_product_tool(targeted, tmp_path)["status"] == "pass"
    targeted["expected_separation_mm"] = 79.99998
    assert run_product_tool(targeted, tmp_path)["status"] == "fail"
    targeted["expected_separation_mm"] = 79.999995
    assert run_product_tool(targeted, tmp_path)["status"] == "pass"
    assert (
        run_product_tool(request(expected_midpoint_mm=1, tolerance_mm=1e-5), tmp_path)["status"]
        == "fail"
    )


def test_rotation_uses_brep_not_projected_world_aabb_and_datum_translation(tmp_path):
    export_pair(tmp_path, rotation=37, tilt=23, translation=(12, -9, 21))
    angle, tilt = math.radians(37), math.radians(23)
    unit = [math.cos(angle) * math.cos(tilt), math.sin(angle), -math.cos(angle) * math.sin(tilt)]
    req = request(
        axis=[5 * value for value in unit],
        datum=[12, -9, 21],
        expected_separation_mm=80,
        expected_midpoint_mm=0,
        tolerance_mm=1e-5,
    )
    receipt = run_product_tool(req, tmp_path)
    assert receipt["status"] == "pass"
    assert receipt["results"]["axis"] == pytest.approx(unit)
    req["datum"] = [
        value + 7 * direction for value, direction in zip([12, -9, 21], unit, strict=True)
    ]
    req["expected_midpoint_mm"] = -7
    assert run_product_tool(req, tmp_path)["status"] == "pass"


def test_reversing_axis_alone_is_not_negating_the_separation(tmp_path):
    export_pair(tmp_path)
    req = request(axis=[-1, 0, 0])
    result = run_product_tool(req, tmp_path)["results"]
    assert result["ordered_separation_mm"] == pytest.approx(-120, abs=1e-6)
    req["a"], req["b"] = req["b"], req["a"]
    assert run_product_tool(req, tmp_path)["results"]["ordered_separation_mm"] == pytest.approx(
        80, abs=1e-6
    )


def test_protrusion_changes_support_and_sideways_displacement_does_not_prove_alignment(tmp_path):
    export_pair(tmp_path, protrusion=0.2, sideways=200)
    receipt = run_product_tool(request(expected_separation_mm=79.8, tolerance_mm=1e-5), tmp_path)
    assert receipt["status"] == "pass"
    assert receipt["qualification_status"] == "unverified"
    assert "alignment" in receipt["algorithm_scope"]
    assert receipt["results"]["boundary_midpoint_mm"] == pytest.approx(0.1, abs=1e-6)


def test_overlapping_projections_do_not_claim_collision(tmp_path):
    left, right = b.Box(20, 4, 4), b.Box(20, 4, 4).translate((10, 50, 0))
    left.label, right.label = occurrence_label("left/pad"), occurrence_label("right/pad")
    b.export_step(b.Compound(children=[left, right]), tmp_path / "part.step")
    receipt = run_product_tool(request(), tmp_path)
    assert receipt["results"]["ordered_separation_mm"] == pytest.approx(-10, abs=1e-6)
    assert receipt["results"]["boundary_midpoint_mm"] == pytest.approx(5, abs=1e-6)
    assert receipt["status"] == "unverified"
    assert "collision" in receipt["algorithm_scope"]


def test_distinct_selectors_for_same_exported_body_never_pass(tmp_path):
    export_pair(tmp_path)
    result = run_product_tool(request(), tmp_path)["results"]
    alias = request(
        b={"body_index": result["a"]["body_index"]}, expected_separation_mm=-20, tolerance_mm=1e-5
    )
    receipt = run_product_tool(alias, tmp_path)
    assert receipt["status"] == "unverified"
    assert "distinct" in receipt["results"]["reason"]
    assert (
        run_product_tool(request(b={"occurrence_path": "missing"}), tmp_path)["status"]
        == "unverified"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"axis": [0, 0, 0]},
        {"axis": [1e-12, 0, 0]},
        {"axis": [math.nan, 0, 1]},
        {"datum": [math.inf, 0, 0]},
        {"expected_separation_mm": True},
        {"expected_separation_mm": 80},
        {"expected_midpoint_mm": 0},
        {"tolerance_mm": 1e-6},
        {"tolerance_mm": -1},
        {"tolerance_mm": math.inf},
        {"a": {"body_index": 0, "occurrence_path": "left/pad"}},
    ],
)
def test_invalid_or_unstated_precision_rejected(updates):
    with pytest.raises(ValidationError):
        _ADAPTER.validate_python(request(**updates))


def test_missing_or_ambiguous_step_remains_unverified(tmp_path):
    assert run_product_tool(request(), tmp_path)["status"] == "unverified"
    a, other = b.Box(2, 2, 2), b.Box(2, 2, 2).translate((5, 0, 0))
    a.label = other.label = occurrence_label("left/pad")
    b.export_step(b.Compound(children=[a, other]), tmp_path / "part.step")
    assert run_product_tool(request(), tmp_path)["status"] == "unverified"


def test_real_run_tool_candidate_receipt_preserves_frozen_contract_and_findings(tmp_path):
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.measure import engine_fingerprint
    from forge_mind.workspace.models import Measurement
    from forge_mind.workspace.product_service import run_tool
    from forge_mind.workspace.store import Store

    service = WorkspaceService(Store(tmp_path / "s"), Executor("trusted_local"))
    contract = {"requirements": [{"id": "gap", "text": "Pad opening is 80 mm"}]}
    sid = service.store.create("owner", "Pad opening", contract)["id"]
    rid = "b" * 32
    source = export_pair(tmp_path)
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    shutil.copyfile(source, directory / "part.step")
    b.export_stl(b.import_step(source), tmp_path / "part.stl")
    shutil.copyfile(tmp_path / "part.stl", directory / "part.stl")
    (directory / "program.py").write_text("# STEP fixture exported by export_pair\n")
    measurement = Measurement(
        revision_id=rid,
        contract_hash="frozen",
        geometry_valid=True,
        engine_fingerprint=engine_fingerprint(),
        artifact_hashes={
            name: file_sha256(directory / name) for name in ("part.step", "part.stl", "program.py")
        },
        findings=[
            {
                "requirement_id": "gap",
                "status": "unverified",
                "method": "unsupported",
                "reason": "Frozen requirement needs its own verification",
            }
        ],
    ).model_dump()
    revision = {"id": rid, "contract": contract, "measurement": measurement}
    service.store.add_revision(sid, revision)
    service.store.update(
        sid,
        state="needs_review",
        accepted_revision=None,
        selected_revision=None,
        proposal={"revision_id": rid, "eligible": False},
    )
    before = deepcopy(service.store.get(sid, "owner"))
    receipt = run_tool(
        service,
        sid,
        "owner",
        rid,
        request(expected_separation_mm=80, expected_midpoint_mm=0, tolerance_mm=1e-5),
        timeout=30,
    )
    assert receipt["result"]["status"] == "pass"
    assert receipt["result"]["qualification_status"] == "unverified"
    assert receipt["revision_id"] == rid
    assert receipt["contract"] == contract
    assert receipt["artifact_hashes"] == measurement["artifact_hashes"]
    assert service.store.revision(sid, rid) == revision
    assert service.store.get(sid, "owner") == before
