"""Saved-byte service/authority controls; the fake worker supplies no geometry proof."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_native_handoff_controller import ENGINE, RID, controller_case
from tests.unit.workspace.test_open_rectangular_shell import request

from forge_mind.workspace import completion, product_service
from forge_mind.workspace.completion_coverage import functional_coverage
from forge_mind.workspace.completion_inline import execute_prepared
from forge_mind.workspace.completion_models import PreparedCheck
from forge_mind.workspace.completion_work import remaining_work
from forge_mind.workspace.product_sources import _append
from forge_mind.workspace.product_tools import _SCOPES
from forge_mind.workspace.store import WorkspaceError


@pytest.fixture
def case(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    monkeypatch.setattr(completion, "engine_fingerprint", lambda: ENGINE)
    monkeypatch.setattr(product_service, "engine_fingerprint", lambda: ENGINE)
    case.workers = []
    case.after_worker = lambda: None

    def worker(directory, payload, timeout, cancelled):
        case.workers.append(
            {"payload": deepcopy(payload), "timeout": timeout, "cancelled": cancelled()}
        )
        case.after_worker()
        return {
            "tool": "open_rectangular_shell",
            "status": "pass",
            "algorithm_version": 1,
            "algorithm_scope": _SCOPES["open_rectangular_shell"],
            "qualification_status": "unverified",
            "artifact_hashes": {
                "part.step": case.service.store.revision(case.sid, RID)["measurement"][
                    "artifact_hashes"
                ]["part.step"]
            },
            "results": {
                "result_scope": "matches_nominal_plan",
                "scope": "Controlled receipt boundary, not native proof",
            },
        }

    monkeypatch.setattr(product_service, "process_revision", worker)
    case.definition = PreparedCheck.model_validate(
        {"id": "shell", "request": request(), "reference_revision_id": RID}
    ).model_dump(mode="json")
    return case


def test_existing_prepared_path_reuses_only_the_exact_current_request(case):
    snapshot = completion._snapshot(case.service, case.sid, "owner", RID)
    first, reused = execute_prepared(
        case.service, case.sid, "owner", snapshot, case.definition, 17, lambda: False
    )
    assert reused is False
    second, reused = execute_prepared(
        case.service, case.sid, "owner", snapshot, case.definition, 13, lambda: False
    )
    assert reused is True and second["id"] == first["id"]
    assert len(case.workers) == 1
    changed = deepcopy(case.definition)
    changed["request"]["floor_thickness_mm"] = 4.0
    third, reused = execute_prepared(
        case.service, case.sid, "owner", snapshot, changed, 11, lambda: False
    )
    assert reused is False and third["id"] != first["id"]
    assert len(case.workers) == 2
    assert case.workers[0]["payload"]["operation"] == "product_tool"
    assert case.workers[0]["timeout"] == 17
    assert case.workers[0]["payload"]["request"] == case.definition["request"]
    assert first["request"]["floor_thickness_mm"] == 3
    assert third["request"]["floor_thickness_mm"] == 4
    assert first["result"]["results"]["result_scope"] == "matches_nominal_plan"
    assert "do not replace frozen acceptance" in first["acceptance_scope"]


def test_receipt_scope_does_not_mutate_source_contract_plan_selection_or_acceptance(case):
    before = case.service.store.get(case.sid, "owner")
    contract = deepcopy(before["contract"])
    receipt = product_service.run_tool(case.service, case.sid, "owner", RID, request(), timeout=17)
    saved = case.service.store.get(case.sid, "owner")
    for field in ("contract", "selected_revision", "accepted_revision", "design_plan", "plan_hash"):
        assert saved.get(field) == before.get(field)
    assert receipt["result"]["qualification_status"] == "unverified"
    assert receipt["contract"] == contract
    assert (
        product_service.read_tool_receipt(case.service, case.sid, "owner", receipt["id"])["stale"]
        is False
    )
    with pytest.raises(WorkspaceError, match="not found"):
        product_service.read_tool_receipt(case.service, case.sid, "other", receipt["id"])


def test_changed_source_and_artifact_bytes_cannot_supply_current_proof(case):
    receipt = product_service.run_tool(case.service, case.sid, "owner", RID, request())
    snapshot = completion._snapshot(case.service, case.sid, "owner", RID)
    _append(
        case.service.store,
        case.sid,
        "owner",
        {"kind": "artifact", "data": {"id": "manual", "text": "Changed immutable input"}},
    )
    assert (
        product_service.read_tool_receipt(case.service, case.sid, "owner", receipt["id"])["stale"]
        is True
    )
    with pytest.raises(WorkspaceError, match="inputs changed"):
        execute_prepared(
            case.service, case.sid, "owner", snapshot, case.definition, 17, lambda: False
        )
    (case.service.store.directory(case.sid, RID) / "part.step").write_bytes(b"Changed STEP")
    with pytest.raises(WorkspaceError, match="artifacts changed"):
        product_service.read_tool_receipt(case.service, case.sid, "owner", receipt["id"])


def test_receipt_tampering_cannot_mint_a_current_nominal_pass(case):
    receipt = product_service.run_tool(case.service, case.sid, "owner", RID, request())
    with case.service.store.connect() as con:
        con.execute("UPDATE product_checks SET document='{}' WHERE id=?", (receipt["id"],))
    with pytest.raises(WorkspaceError, match="integrity"):
        product_service.read_tool_receipt(case.service, case.sid, "owner", receipt["id"])


def test_cancellation_after_worker_saves_no_receipt(case):
    cancelled = []
    case.after_worker = lambda: cancelled.append(True)
    with pytest.raises(WorkspaceError, match="cancelled"):
        product_service.run_tool(
            case.service,
            case.sid,
            "owner",
            RID,
            request(),
            timeout=17,
            cancelled=lambda: bool(cancelled),
        )
    with case.service.store.connect() as con:
        exists = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='product_checks'"
        ).fetchone()
        count = (
            con.execute(
                "SELECT COUNT(*) FROM product_checks WHERE session=?", (case.sid,)
            ).fetchone()[0]
            if exists
            else 0
        )
    assert count == 0


def test_passing_authored_nominal_check_does_not_cover_original_source_or_missing_parts():
    contract = {"requirements": [{"id": "function", "metric": "unmeasured"}]}
    assert functional_coverage({}, contract)["missing_requirement_ids"] == ["function"]
    row = {
        "id": "engine",
        "requirement_ids": ["function"],
        "dependencies": [],
        "missing_part_ids": ["crankshaft"],
        "missing_occurrence_paths": [],
        "checks": [{"id": "shell", "status": "pass"}],
        "status": "missing",
        "geometry_declared": True,
        "unresolved": [],
    }
    work = remaining_work([row], [])
    assert work["next_work_category"] == "construction"
    assert work["next_target"] == "engine"
