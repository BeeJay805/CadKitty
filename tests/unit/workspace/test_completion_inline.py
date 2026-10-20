from __future__ import annotations

import copy
import json
import time
from typing import Any

import pytest
from tests.unit.workspace.test_completion import RID, finish
from tests.unit.workspace.test_completion import harness as original_harness

from forge_mind.workspace.completion import completion_context, validate_plan
from forge_mind.workspace.completion_inline import author_checkpoint


@pytest.fixture
def harness(monkeypatch):
    return original_harness.__wrapped__(monkeypatch)


def checkpoint(harness: Any, **kwargs: Any) -> dict[str, Any]:
    return author_checkpoint(
        harness.service,
        "s",
        "o",
        RID,
        kwargs.get("job", "job"),
        kwargs.get("deadline", time.monotonic() + 60),
        kwargs.get("cancelled", lambda: False),
    )


def test_early_measurement_reaches_context_before_finalization_and_reuses(harness):
    frozen = copy.deepcopy(harness.service.store.session["contract"])
    first = checkpoint(harness)
    assert first["measured_subsystem"] == "frame"
    assert first["subsystems"][0]["checks"][0]["status"] == "pass"
    context = completion_context(harness.service, "s", "o", RID)
    assert context["next_target"] == "cover"
    assert context["status"] == "unverified"
    second = checkpoint(harness)
    assert second["measured_subsystem"] == "cover"
    assert checkpoint(harness)["measured_subsystem"] is None
    assert len(harness.calls) == 2
    report = finish(harness)
    assert len([c for c in harness.calls if c[0] == "tool"]) == 2
    assert report["physical_readiness"] == "unverified"
    assert harness.service.store.session["contract"] == frozen


def test_failed_probe_is_visible_and_reused_then_repaired_input_is_rechecked(harness, monkeypatch):
    import forge_mind.workspace.product_service as product_service

    original = product_service.run_tool

    def tool(*args):
        receipt = original(*args)
        if args[4]["allowable"]["high"] == 3:
            receipt["result"] = {"status": "fail", "results": {"reason": "misfit"}}
            import hashlib

            encoded = json.dumps(receipt)
            harness.service.store.connection.execute(
                "UPDATE product_checks SET document=?,digest=? WHERE id=?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest(), receipt["id"]),
            )
        return receipt

    monkeypatch.setattr(product_service, "run_tool", tool)
    first = checkpoint(harness)
    assert first["subsystems"][0]["status"] == "checked_fail"
    assert completion_context(harness.service, "s", "o", RID)["next_target"] == "frame"
    checkpoint(harness)
    assert len(harness.calls) == 2  # independent cover still measured
    assert checkpoint(harness)["measured_subsystem"] is None
    harness.service.store.session["design_plan"]["subsystems"][0]["checks"][0]["request"][
        "allowable"
    ]["high"] = 4
    repaired = checkpoint(harness)
    assert repaired["subsystems"][0]["checks"][0]["status"] == "pass"
    assert len(harness.calls) == 3
    assert completion_context(harness.service, "s", "o", RID)["status"] == "unverified"


@pytest.mark.parametrize(
    "change", ["artifact", "source", "engine", "input", "contract", "request", "receipt"]
)
def test_binding_changes_do_not_reuse_old_measurements(harness, monkeypatch, change):
    checkpoint(harness)
    if change == "artifact":
        harness.revision["measurement"]["artifact_hashes"]["model.step"] = "new"
    elif change == "source":
        monkeypatch.setattr(
            "forge_mind.workspace.completion.read_sources", lambda *a: {"fingerprint": "new"}
        )
    elif change == "engine":
        monkeypatch.setattr("forge_mind.workspace.completion.engine_fingerprint", lambda: "new")
    elif change == "input":
        harness.service.store.session["engineering"]["version"] += 1
    elif change == "contract":
        harness.revision["contract"]["requirements"][0]["text"] = "new"
    elif change == "request":
        harness.service.store.session["design_plan"]["subsystems"][0]["checks"][0]["request"][
            "allowable"
        ]["high"] = 4
    else:
        harness.service.store.connection.execute("UPDATE product_checks SET document='{}'")
    result = checkpoint(harness)
    # Corrupt receipts must be retried even within the same job.
    assert result["subsystems"][0]["checks"][0].get("receipt_id") != "1"
    assert result["subsystems"][0]["checks"][0]["status"] != "pass" or len(harness.calls) == 2


def test_unavailable_worker_does_not_loop_and_resumed_job_retries(harness, monkeypatch):
    calls = []

    def unavailable(*args):
        calls.append(args)
        raise RuntimeError("Missing instrument")

    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", unavailable)
    checkpoint(harness)
    checkpoint(harness)
    result = checkpoint(harness)
    assert len(calls) == 2
    assert "Missing instrument" in result["subsystems"][0]["checks"][0]["reason"]
    checkpoint(harness, job="resumed")
    assert len(calls) == 3


@pytest.mark.parametrize("resume", [True, False])
def test_unavailable_receipt_retries_on_resume_or_finalization(harness, monkeypatch, resume):
    import hashlib

    import forge_mind.workspace.product_service as product_service

    original = product_service.run_tool

    def tool(*args):
        receipt = original(*args)
        if receipt["id"] == "1":
            receipt["result"] = {
                "status": "unverified",
                "results": {"reason": "Geometry inspection unavailable"},
            }
            encoded = json.dumps(receipt)
            harness.service.store.connection.execute(
                "UPDATE product_checks SET document=?,digest=? WHERE id=?",
                (encoded, hashlib.sha256(encoded.encode()).hexdigest(), receipt["id"]),
            )
        return receipt

    monkeypatch.setattr(product_service, "run_tool", tool)
    checkpoint(harness)
    checkpoint(harness)
    assert checkpoint(harness)["measured_subsystem"] is None
    assert len(harness.calls) == 2
    context = completion_context(harness.service, "s", "o", RID)
    assert "Geometry inspection unavailable" in context["subsystems"][0]["checks"][0]["reason"]
    assert context["subsystems"][0]["checks"][0]["status"] == "unverified"
    result = checkpoint(harness, job="resume") if resume else finish(harness)
    assert result["subsystems"][0]["checks"][0]["status"] == "pass"
    assert len([c for c in harness.calls if c[0] == "tool"]) == 3


@pytest.mark.parametrize("cancel", [True, False])
def test_early_work_preserves_cancel_and_deadline(harness, cancel):
    result = checkpoint(
        harness, cancelled=lambda: cancel, deadline=time.monotonic() - 1 if not cancel else None
    )
    assert not harness.calls
    assert result["stale"]
    assert all(c["status"] == "unverified" for s in result["subsystems"] for c in s["checks"])


def test_unrelated_plan_change_reuses_same_request_but_refreshes_context(harness):
    checkpoint(harness)
    harness.service.store.session["design_plan"]["subsystems"][0]["name"] = "Renamed"
    result = checkpoint(harness)
    assert result["subsystems"][0]["name"] == "Renamed"
    assert result["subsystems"][0]["checks"][0]["receipt_id"] == "1"
    assert len(harness.calls) == 2  # only second subsystem ran


def test_context_reconstructs_coverage_and_results_from_verified_receipts(harness):
    checkpoint(harness)
    early = harness.service.store.session["author_checkpoint"]
    early["subsystems"][0]["checks"][0].update(
        status="fail", result={"status": "fail", "made_up_measurement": -1000}
    )
    early["subsystems"][0]["unresolved"] = []
    early["subsystems"][1]["checks"] = [{"id": "invented", "status": "pass", "receipt_id": "1"}]
    early["subsystems"][1]["status"] = "checked_pass"
    context = completion_context(harness.service, "s", "o", RID)
    first, second = context["subsystems"]
    assert first["checks"][0]["status"] == "pass"
    assert "made_up_measurement" not in first["checks"][0]["result"]
    assert second["checks"][0]["id"] == "cover_check"
    assert second["checks"][0]["status"] == "unverified"
    assert context["next_target"] == "cover"


@pytest.mark.parametrize("tamper", ["missing_row", "missing_check", "invented_coverage", "request"])
def test_final_context_does_not_trust_mutable_coverage_or_request_hashes(harness, tamper):
    finish(harness)
    report = harness.service.store.session["completion_report"]
    if tamper == "missing_row":
        report["subsystems"].pop()
    elif tamper == "missing_check":
        report["subsystems"][0]["checks"] = []
    elif tamper == "invented_coverage":
        report["subsystems"][0]["requirement_ids"].append("invented")
    else:
        report["subsystems"][0]["checks"][0]["request_hash"] = "invented"
    context = completion_context(harness.service, "s", "o", RID)
    assert context["stale"]
    assert context["status"] == "unverified"
    assert len(context["subsystems"]) == 2


def test_dispositions_cannot_supply_acceptance_or_unknown_check(harness):
    plan = copy.deepcopy(harness.service.store.session["design_plan"])
    plan["requirement_dispositions"] = [
        {
            "requirement_id": "fit",
            "check_ids": ["frame_check"],
            "missing_inputs": ["Spindle runout"],
            "physical_validation": ["Measure finished hole"],
        }
    ]
    validate_plan(plan, harness.revision["contract"])
    harness.service.store.session["design_plan"] = plan
    checkpoint(harness)
    item = completion_context(harness.service, "s", "o", RID)["requirement_dispositions"][0]
    assert item["status"] == "unverified" and item["checks"][0]["status"] == "pass"
    assert item["missing_inputs"] == ["Spindle runout"]
    plan["requirement_dispositions"][0]["status"] = "verified"
    with pytest.raises(ValueError):
        validate_plan(plan, harness.revision["contract"])
    del plan["requirement_dispositions"][0]["status"]
    plan["requirement_dispositions"][0]["check_ids"] = ["missing"]
    with pytest.raises(ValueError, match="unknown"):
        validate_plan(plan, harness.revision["contract"])
