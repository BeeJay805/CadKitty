"""Inspection proposals preserve measured provenance without promoting acceptance."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from tests.unit.workspace.test_completion import RID
from tests.unit.workspace.test_completion import harness as original_harness

from forge_mind.workspace.completion import _snapshot
from forge_mind.workspace.completion_inline import _binding
from forge_mind.workspace.drawings import write_drawings
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.inspection_models import InspectionCharacteristic
from forge_mind.workspace.inspection_plan import resolve_inspection_plan, validate_inspection_plan


def characteristic(**changes: Any) -> dict[str, Any]:
    return {
        "id": "jaw_gap",
        "label": "Jaw opening",
        "requirement_ids": ["fit"],
        "occurrence_paths": ["frame", "cover"],
        "characteristic": "Ordered support separation",
        "datum": "X direction from declared datum origin",
        "nominal_mm": 80,
        "minus_mm": 0.02,
        "plus_mm": 0.04,
        "basis": "Author proposal; process capability unknown",
        "physical_method": "Inspect with calibrated gauge; method requires review",
        "missing_evidence": ["Physical repeatability"],
        "probe": {"check_id": "frame_check", "result": "ordered_separation_mm"},
        **changes,
    }


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    h = original_harness.__wrapped__(monkeypatch)
    plan = h.service.store.session["design_plan"]
    plan["subsystems"][0]["checks"][0]["request"] = {
        "tool": "projected_pair",
        "units": "mm",
        "a": {"occurrence_path": "frame"},
        "b": {"occurrence_path": "cover"},
        "axis": [1, 0, 0],
        "datum": [0, 0, 0],
        "expected_separation_mm": 80,
        "tolerance_mm": 100,
    }
    plan["inspection"] = [characteristic()]
    for index, occurrence in enumerate(
        h.revision["measurement"]["facts"]["assembly"]["occurrences"]
    ):
        occurrence["body_index"] = index
    snapshot = _snapshot(h.service, "s", "o", RID)
    definition = plan["subsystems"][0]["checks"][0]
    receipt = {
        **{
            k: snapshot[k]
            for k in ("revision_id", "engine_fingerprint", "source_fingerprint", "artifact_hashes")
        },
        "id": "receipt",
        "contract": h.revision["contract"],
        "request": definition["request"],
        "result": {
            "tool": "projected_pair",
            "status": "pass",
            "algorithm_scope": "Whole-body support intervals; not contact or physical readiness",
            "results": {
                "units": "mm",
                "measurement_status": "measured",
                "ordered_separation_mm": 79.0,
                "a": {"selector": {"occurrence_path": "frame"}, "body_index": 0},
                "b": {"selector": {"occurrence_path": "cover"}, "body_index": 1},
            },
        },
    }
    encoded = json.dumps(receipt)
    h.service.store.connection.execute(
        "INSERT INTO product_checks VALUES (?,?,?,?)",
        ("receipt", "s", encoded, hashlib.sha256(encoded.encode()).hexdigest()),
    )
    h.service.store.session["prepared_check_cache"] = {
        "frame_check": {"binding": _binding(snapshot, definition), "receipt_id": "receipt"}
    }
    return h


def resolve(h: SimpleNamespace) -> dict[str, Any]:
    return resolve_inspection_plan(h.service, "s", "o", h.revision)


def test_loose_probe_pass_remains_separate_from_proposed_limits_and_acceptance(harness):
    before = copy.deepcopy(harness.service.store.session)
    result = resolve(harness)
    row = result["rows"][0]
    assert row["cad"]["value_mm"] == 79
    assert row["cad"]["probe_status"] == "pass"
    assert row["intent"]["nominal_mm"] == 80
    assert row["intent"]["minus_mm"] == 0.02
    assert row["physical"]["status"] == "unverified"
    assert row["physical"]["measured_mm"] is None
    assert row["missing_evidence"] == ["Physical repeatability"]
    assert result["receipts"]["receipt"]["request"]["tolerance_mm"] == 100
    assert harness.service.store.session == before
    assert not harness.calls
    assert row["cad"]["measured_occurrence_paths"] == ["frame", "cover"]
    assert row["cad"]["measured_body_indices"] == [0, 1]


@pytest.mark.parametrize(
    "declared",
    [
        ["unrelated_bushing"],
        ["frame"],
        ["frame", "unrelated_bushing"],
        ["frame", "cover", "unrelated_bushing"],
        ["frame", "frame"],
    ],
)
def test_existing_unrelated_occurrence_cannot_receive_pair_measurement(harness, declared):
    harness.revision["measurement"]["facts"]["assembly"]["occurrences"].append(
        {"path": "unrelated_bushing", "definition": "bushing", "body_index": 2}
    )
    harness.service.store.session["design_plan"]["inspection"][0]["occurrence_paths"] = declared
    row = resolve(harness)["rows"][0]
    assert row["cad"]["status"] == "unavailable"
    assert row["cad"]["value_mm"] is None
    assert "attribution mismatch" in row["cad"]["reason"]
    assert row["cad"]["measured_occurrence_paths"] == ["frame", "cover"]
    assert row["cad"]["measured_body_indices"] == [0, 1]
    assert "dimension_points_mm" not in row["cad"]


@pytest.mark.parametrize("declared", [[], ["frame", "cover"], ["cover", "frame"]])
def test_empty_or_correct_authored_pair_exposes_actual_ordered_identities(harness, declared):
    harness.service.store.session["design_plan"]["inspection"][0]["occurrence_paths"] = declared
    row = resolve(harness)["rows"][0]
    assert row["cad"]["value_mm"] == 79
    assert row["cad"]["measured_occurrence_paths"] == ["frame", "cover"]
    assert row["intent"]["occurrence_paths"] == declared


@pytest.mark.parametrize(
    "change",
    [
        "index",
        "selector",
        "missing_selector",
        "alias",
        "missing_index",
        "bool_index",
        "missing_map",
    ],
)
def test_pair_identity_contradictions_withhold_numeric_result(harness, change):
    con = harness.service.store.connection
    receipt = json.loads(con.execute("SELECT document FROM product_checks").fetchone()[0])
    actual = receipt["result"]["results"]
    if change == "index":
        actual["a"]["body_index"] = 1
    elif change == "selector":
        actual["a"]["selector"] = {"occurrence_path": "cover"}
    elif change == "missing_selector":
        actual["a"].pop("selector")
    elif change == "missing_index":
        actual["a"].pop("body_index")
    elif change == "bool_index":
        actual["a"]["body_index"] = False
    elif change == "alias":
        actual["b"]["body_index"] = 0
    else:
        harness.revision["measurement"]["facts"]["assembly"]["occurrences"][0].pop("body_index")
    encoded = json.dumps(receipt)
    con.execute(
        "UPDATE product_checks SET document=?,digest=?",
        (encoded, hashlib.sha256(encoded.encode()).hexdigest()),
    )
    assert resolve(harness)["rows"][0]["cad"]["value_mm"] is None


@pytest.mark.parametrize("tool", ["projected_pair", "planar_face_pair"])
def test_numeric_request_body_indices_resolve_through_current_occurrence_inventory(harness, tool):
    from forge_mind.workspace.inspection_plan import measured_pair_identity

    a, b = {"body_index": 0}, {"body_index": 1}
    if tool == "planar_face_pair":
        a, b = {"body": a, "face_signature": "a" * 64}, {"body": b, "face_signature": "b" * 64}
    request = {"tool": tool, "a": a, "b": b}
    result = {"a": {"selector": a, "body_index": 0}, "b": {"selector": b, "body_index": 1}}
    if tool == "planar_face_pair":
        result["a"]["face_signature"] = "a" * 64
        result["b"]["face_signature"] = "b" * 64
    assert measured_pair_identity(request, result, harness.revision) == (["frame", "cover"], [0, 1])
    result["a"]["body_index"] = 1
    with pytest.raises(ValueError, match="index disagrees"):
        measured_pair_identity(request, result, harness.revision)


@pytest.mark.parametrize(
    "change",
    [
        "request",
        "artifact",
        "source",
        "engine",
        "input",
        "contract",
        "receipt",
        "other_session",
        "revision",
    ],
)
def test_changed_binding_cannot_retain_current_numeric_measurement(harness, monkeypatch, change):
    assert resolve(harness)["rows"][0]["cad"]["status"] == "measured"
    if change == "request":
        harness.service.store.session["design_plan"]["subsystems"][0]["checks"][0]["request"][
            "tolerance_mm"
        ] = 1
    elif change == "artifact":
        harness.revision["measurement"]["artifact_hashes"]["model.step"] = "changed"
    elif change in {"source", "engine"}:
        if change == "source":
            monkeypatch.setattr(
                "forge_mind.workspace.completion.read_sources",
                lambda *a: {"fingerprint": "changed"},
            )
        else:
            monkeypatch.setattr(
                "forge_mind.workspace.completion.engine_fingerprint", lambda: "changed"
            )
    elif change == "input":
        harness.service.store.session["engineering"]["version"] = 1
    elif change == "contract":
        harness.revision["contract"]["requirements"][0]["text"] = "changed"
    elif change == "receipt":
        harness.service.store.connection.execute("UPDATE product_checks SET document='{}'")
    elif change == "other_session":
        harness.service.store.connection.execute("UPDATE product_checks SET session='other'")
    else:
        harness.revision["id"] = "b" * 32
    result = resolve(harness)
    assert result["rows"][0]["cad"]["value_mm"] is None
    assert not result["receipts"]


def test_inspection_amendment_reuses_same_receipt_but_changes_snapshot(harness):
    original = resolve(harness)
    harness.service.store.session["design_plan"]["inspection"][0]["physical_method"] = (
        "Revised proposed method"
    )
    amended = resolve(harness)
    assert amended["plan_hash"] != original["plan_hash"]
    assert amended["receipts"] == original["receipts"]
    assert amended["rows"][0]["cad"]["value_mm"] == 79


def test_unknown_limits_and_requirement_obligations_remain_visible(harness):
    plan = harness.service.store.session["design_plan"]
    plan["inspection"] = [characteristic(nominal_mm=None, minus_mm=None, plus_mm=None, probe=None)]
    plan["requirement_dispositions"] = [
        {
            "requirement_id": "fit",
            "missing_inputs": ["Supplier applicability"],
            "physical_validation": ["Drilled specimen inspection"],
        }
    ]
    row = resolve(harness)["rows"][0]
    assert row["cad"]["value_mm"] is None
    assert "Proposed nominal is unspecified" in row["missing_evidence"]
    assert "fit: Supplier applicability" in row["missing_evidence"]
    assert "fit: Drilled specimen inspection" in row["missing_evidence"]


@pytest.mark.parametrize(
    "patch",
    [
        {"nominal_mm": float("nan")},
        {"minus_mm": -1},
        {"plus_mm": None},
        {"nominal_mm": None},
        {"nominal_mm": True},
        {"units": "inch"},
        {"physical_measured_mm": 80},
        {"verified": True},
        {"probe": {"check_id": "frame_check", "result": "results.any.path"}},
    ],
)
def test_schema_rejects_invalid_limits_or_authored_measurement_authority(patch):
    with pytest.raises(ValueError):
        InspectionCharacteristic.model_validate(characteristic(**patch))


@pytest.mark.parametrize(
    "change", ["unknown_check", "unknown_requirement", "duplicate", "wrong_tool"]
)
def test_plan_references_and_numeric_extractor_are_typed(harness, change):
    plan = harness.service.store.session["design_plan"]
    if change == "unknown_check":
        plan["inspection"][0]["probe"]["check_id"] = "missing"
    elif change == "unknown_requirement":
        plan["inspection"][0]["requirement_ids"] = ["missing"]
    elif change == "duplicate":
        plan["inspection"].append(copy.deepcopy(plan["inspection"][0]))
    else:
        plan["subsystems"][0]["checks"][0]["request"]["tool"] = "hole_pattern"
    with pytest.raises(ValueError):
        validate_inspection_plan(plan, harness.revision["contract"])


def test_probe_only_never_exposes_echoed_request_as_measured(harness):
    harness.service.store.session["design_plan"]["inspection"][0]["probe"]["result"] = "probe"
    row = resolve(harness)["rows"][0]
    assert row["cad"]["status"] == "probe_only"
    assert row["cad"]["value_mm"] is None


def test_schedule_and_csv_keep_proposals_cad_and_empty_physical_fields_separate(
    harness, tmp_path: Path
):
    plan = harness.service.store.session["design_plan"]
    plan["inspection"][0]["physical_method"] = '=HYPERLINK("untrusted")'
    plan["inspection"][0]["basis"] = "<script>untrusted</script>"
    result = resolve(harness)
    result["rows"] *= 12
    revision = {
        "id": RID,
        "contract": {
            "requirements": [{"id": "fit", "text": "Frozen fit", "value": None, "tolerance": 0}]
        },
        "measurement": {"facts": {"bbox_min": [0, 0, 0], "bbox_max": [80, 10, 10]}, "findings": []},
    }
    report = write_drawings(
        tmp_path, {"entities": []}, revision, EngineeringInputs(), inspection_plan=result
    )
    assert report["inspection_characteristics"] == 12
    svg = (tmp_path / "drawing.svg").read_text()
    assert svg.count("jaw_gap |") == 12
    assert "&lt;script&gt;" in svg and "<script>" not in svg
    assert "Proposed nominal 80 mm, -0.02/+0.04 mm" in svg
    with (tmp_path / "inspection.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 13
    assert rows[0]["CAD status"] == "unverified"
    assert rows[1]["CAD measured mm"] == "79.0"
    assert rows[1]["Target"] == "80.0"
    assert rows[1]["Physical measured"] == rows[1]["Accepted by"] == ""
    assert rows[1]["Receipt ID"] == "receipt"
    assert rows[1]["Unit / method"] == "mm"
    assert rows[1]["Proposed physical method"].startswith("'=HYPERLINK")
    assert rows[1]["Proposed lower mm"] == "79.98"
    assert rows[1]["Proposed upper mm"] == "80.04"
    assert rows[0]["Row kind"] == "Frozen requirement"


def test_callout_endpoints_come_only_from_measured_planes(harness, tmp_path):
    from forge_mind.workspace.inspection_plan import dimension_points

    values = {
        "axis": [1, 0, 0],
        "datum": [0, 0, 0],
        "a": {"interval_mm": [-10, 0]},
        "b": {"interval_mm": [79, 89]},
    }
    points = dimension_points("projected_pair", values)
    assert points == [[0, 0, 0], [79, 0, 0]]
    assert (
        dimension_points(
            "planar_face_pair",
            {
                "a": {"plane_dimension_point_mm": [0, 0, 0]},
                "b": {"plane_dimension_point_mm": [79, 0, 0]},
            },
        )
        == points
    )
    assert dimension_points("projected_pair", {**values, "axis": [2, 0, 0]}) is None
    result = resolve(harness)
    row = result["rows"][0]
    row["cad"].update(
        dimension_points_mm=points, dimension_scope="Datum-axis planes, not face contact"
    )
    revision = {
        "id": RID,
        "contract": {"requirements": []},
        "measurement": {
            "facts": {"bbox_min": [-10, -5, -5], "bbox_max": [89, 5, 5]},
            "findings": [],
        },
    }
    report = write_drawings(
        tmp_path, {"entities": []}, revision, EngineeringInputs(), inspection_plan=result
    )
    assert {c["view"] for c in report["inspection_callouts"]} == {"Top (XY)", "Front (XZ)"}
    svg = (tmp_path / "drawing.svg").read_text()
    assert svg.count('data-inspection="jaw_gap"') == 2
    assert "#1: 79 mm" in svg
    assert "#1 jaw_gap" in svg
    small = copy.deepcopy(result)
    small["rows"][0]["cad"].update(value_mm=0.3, dimension_points_mm=[[0, 0, 0], [0.3, 0, 0]])
    # In a 300mm frame, 0.3mm is subpixel. Mark location, never fake a larger gap.
    revision["measurement"]["facts"]["bbox_max"][0] = 300
    small["rows"] *= 10
    report = write_drawings(
        tmp_path, {"entities": []}, revision, EngineeringInputs(), inspection_plan=small
    )
    assert len(report["inspection_callouts"]) == 16
    assert all(c["display"] == "small-gap leader" for c in report["inspection_callouts"])
    assert len(report["omitted_inspection_callouts"]) == 14
    assert "#10 jaw_gap" in (tmp_path / "drawing.svg").read_text()


def test_review_rerun_plan_binding_rejects_changed_schedule(tmp_path, monkeypatch):
    from tests.unit.workspace.test_operation_recovery import interrupted, workspace

    from forge_mind.workspace.store import WorkspaceError

    w = workspace.__wrapped__(tmp_path, monkeypatch)
    store, sid, _ = w
    store.update(sid, design_plan={"inspection": [characteristic()]})
    jid = interrupted(w)
    assert store.get(sid, "owner")["operation_recovery"]["available"]
    store.update(sid, design_plan={"inspection": [characteristic(nominal_mm=81)]})
    with pytest.raises(WorkspaceError, match="changed"):
        store.enqueue(sid, "owner", {"retry_operation": jid})


def test_legacy_review_binding_does_not_adopt_new_inspection(tmp_path, monkeypatch):
    from tests.unit.workspace.test_operation_recovery import interrupted, workspace

    from forge_mind.workspace.store import WorkspaceError

    w = workspace.__wrapped__(tmp_path, monkeypatch)
    store, sid, _ = w
    jid = interrupted(w)
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
        payload["operation_binding"].pop("inspection_plan_digest")
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
    assert store.get(sid, "owner")["operation_recovery"]["available"]
    store.update(sid, design_plan={"inspection": [characteristic()]})
    with pytest.raises(WorkspaceError, match="Historical review"):
        store.enqueue(sid, "owner", {"retry_operation": jid})
