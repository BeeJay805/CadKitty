from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
import time
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.workspace.completion import completion_context, finalize_completion, validate_plan
from forge_mind.workspace.completion_models import PreparedCheck
from forge_mind.workspace.motion_check import MotionCheckSettings, configurations

RID = "a" * 32
REQUEST = {
    "tool": "tolerance_stack",
    "units": "mm",
    "terms": [{"low": 1, "high": 2}],
    "allowable": {"low": 0, "high": 3},
}


def test_prepared_request_has_closed_codex_wire_objects_and_roundtrips() -> None:
    from forge_mind.llm.codex_client import _restore_maps, _strict_schema
    from forge_mind.workspace.models import Action

    schema = Action.model_json_schema()
    wire = _strict_schema(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            assert "oneOf" not in value
            if value.get("type") == "object":
                assert value.get("additionalProperties") is False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(wire)
    payload = {
        "kind": "plan",
        "explanation": "Save prepared fit probe",
        "plan": {"objective": "Fit", "subsystems": [subsystem("frame")]},
    }
    action = Action.model_validate(_restore_maps(payload, schema, schema))
    assert action.plan is not None
    assert action.plan.model_dump()["subsystems"][0]["checks"][0]["request"] == REQUEST


def test_expanded_completion_runs_all_checks_without_changing_acceptance(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("COMPLETION_SUBSYSTEM_LIMIT", "COMPLETION_CHECK_LIMIT", "COMPLETION_PLAN_BYTES"):
        monkeypatch.delenv("FORGE_WORKSPACE_" + name, raising=False)
    session = harness.service.store.session
    frozen_contract = copy.deepcopy(session["contract"])
    rows = [subsystem(f"s_{i}", [f"s_{i - 1}"] if i else []) for i in range(65)]
    for row in rows:
        row["part_ids"] = ["frame"]
    session["design_plan"] = {"subsystems": list(reversed(rows))}
    report = finish(harness)
    assert len([call for call in harness.calls if call[0] == "tool"]) == 65
    assert len(set(report["receipt_ids"])) == 65
    assert report["status"] == "checked_pass"
    assert report["physical_readiness"] == "unverified"
    assert session["contract"] == frozen_contract
    assert completion_context(harness.service, "s", "o", RID)["status"] == "checked_pass"
    retained_plan = copy.deepcopy(session["design_plan"])
    retained_report = copy.deepcopy(session["completion_report"])
    monkeypatch.setenv("FORGE_WORKSPACE_COMPLETION_CHECK_LIMIT", "64")
    context = completion_context(harness.service, "s", "o", RID)
    assert context["status"] == "unverified"
    assert "Operator resource policy exceeded" in context["reason"]
    assert session["design_plan"] == retained_plan
    assert session["completion_report"] == retained_report


def subsystem(name: str, dependencies: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": name,
        "name": name,
        "requirement_ids": ["fit"],
        "part_ids": [name],
        "dependencies": dependencies or [],
        "checks": [{"id": name + "_check", "request": copy.deepcopy(REQUEST)}],
    }


class FakeStore:
    def __init__(self, session: dict[str, Any]) -> None:
        self.session = session
        self.connection = sqlite3.connect(":memory:")
        self.connection.execute(
            "CREATE TABLE product_checks (id TEXT, session TEXT, document TEXT, digest TEXT)"
        )

    def connect(self) -> sqlite3.Connection:
        return self.connection

    def get(self, sid: str, owner: str) -> dict[str, Any]:
        return copy.deepcopy(self.session)

    def update(self, sid: str, **values: Any) -> None:
        self.session.update(copy.deepcopy(values))


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    contract = {"requirements": [{"id": "fit"}]}
    session = {
        "selected_revision": RID,
        "contract": contract,
        "engineering": {"version": 0},
        "design_plan": {"subsystems": [subsystem("frame"), subsystem("cover")]},
    }
    revision = {
        "id": RID,
        "contract": contract,
        "document": {"assembly": {"parts": [{"id": "frame"}, {"id": "cover"}]}},
        "measurement": {
            "geometry_valid": True,
            "artifact_hashes": {"model.step": "hash"},
            "facts": {
                "assembly": {
                    "occurrences": [
                        {"path": "frame", "definition": "frame"},
                        {"path": "cover", "definition": "cover"},
                    ]
                }
            },
        },
    }
    service = SimpleNamespace(store=FakeStore(session), _load_revision=lambda sid, rid: revision)
    calls: list[tuple[str, float]] = []

    def tool(*args: Any) -> dict[str, Any]:
        calls.append(("tool", args[5]))
        receipt = {
            "id": str(len(calls)),
            "result": {"status": "pass"},
            "revision_id": RID,
            "source_fingerprint": "s",
            "engine_fingerprint": "engine",
            "artifact_hashes": revision["measurement"]["artifact_hashes"],
            "contract": contract,
            "request": args[4],
        }
        encoded = json.dumps(receipt)
        service.store.connection.execute(
            "INSERT INTO product_checks VALUES (?,?,?,?)",
            (
                receipt["id"],
                "s",
                encoded,
                hashlib.sha256(encoded.encode()).hexdigest(),
            ),
        )
        return receipt

    def interfaces(*args: Any) -> list[Any]:
        calls.append(("interfaces", args[4]))
        return []

    def report(*args: Any) -> None:
        calls.append(("report", args[1]["max_seconds"]))
        service.store.update("s", reports=[{"id": "report"}])

    monkeypatch.setattr(
        "forge_mind.workspace.completion.read_sources", lambda *a: {"fingerprint": "s"}
    )
    monkeypatch.setattr("forge_mind.workspace.completion.engine_fingerprint", lambda: "engine")
    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", tool)
    monkeypatch.setattr("forge_mind.workspace.product_interfaces.validate_interfaces", interfaces)
    monkeypatch.setattr("forge_mind.workspace.engineering_service.run_report_job", report)
    return SimpleNamespace(
        service=service, calls=calls, revision=revision, job={"session": "s", "owner": "o"}
    )


def finish(harness: SimpleNamespace, **kwargs: Any) -> dict[str, Any]:
    return finalize_completion(
        harness.service,
        harness.job,
        kwargs.get("deadline", time.monotonic() + 60),
        kwargs.get("cancelled", lambda: False),
        "budget_exhausted",
    )


def test_strict_request_and_graph_validation() -> None:
    with pytest.raises(ValueError):
        PreparedCheck(id="fake", request={"tool": "invented"})
    contract = {"requirements": [{"id": "fit"}]}
    with pytest.raises(ValueError, match="cycle"):
        validate_plan({"subsystems": [subsystem("a", ["b"]), subsystem("b", ["a"])]}, contract)
    with pytest.raises(ValueError, match="unknown frozen"):
        validate_plan({"subsystems": [subsystem("a")]}, {"requirements": []})
    prior = {"subsystems": [subsystem("a")]}
    with pytest.raises(ValueError, match="cannot be removed"):
        validate_plan({"subsystems": []}, contract, prior)
    changed = copy.deepcopy(prior)
    changed["subsystems"][0]["checks"] = []
    with pytest.raises(ValueError, match="checks cannot be removed"):
        validate_plan(changed, contract, prior)
    changed = copy.deepcopy(prior)
    changed["subsystems"][0]["checks"][0]["request"]["allowable"]["high"] = 4
    with pytest.raises(ValueError, match="explicit criterion_amendment"):
        validate_plan(changed, contract, prior)
    from forge_mind.workspace.check_amendment import request_hash

    changed["subsystems"][0]["checks"][0]["criterion_amendment"] = {
        "previous_request_hash": request_hash(prior["subsystems"][0]["checks"][0]["request"]),
        "reason": "Correct the exploratory allowable interval",
        "basis": "Explicit supplied interval; frozen fit requirement remains unchanged",
    }
    validate_plan(changed, contract, prior)


def test_finalizer_runs_all_phases_and_preserves_acceptance(harness: SimpleNamespace) -> None:
    report = finish(harness)
    assert [c[0] for c in harness.calls] == ["tool", "tool", "interfaces", "report"]
    assert all(0 < c[1] <= 120 for c in harness.calls)
    assert report["status"] == "checked_pass"
    assert report["physical_readiness"] == "unverified"
    assert report["author_state"] == "budget_exhausted"
    assert harness.service.store.session["contract"] == {"requirements": [{"id": "fit"}]}
    assert completion_context(harness.service, "s", "o", RID)["status"] == "checked_pass"


def test_unlimited_validation_propagates_absent_deadline_to_every_phase(harness):
    result = finish(harness, deadline=None)
    assert [name for name, _ in harness.calls] == ["tool", "tool", "interfaces", "report"]
    assert all(seconds is None for _, seconds in harness.calls)
    assert result["status"] == "checked_pass"
    assert result["physical_readiness"] == "unverified"
    json.dumps(harness.service.store.session, allow_nan=False)


@pytest.mark.parametrize("cancelled", [False, True])
def test_exhausted_or_cancelled_never_starts_worker(
    harness: SimpleNamespace, cancelled: bool
) -> None:
    result = finish(harness, deadline=time.monotonic() - 1, cancelled=lambda: cancelled)
    assert harness.calls == []
    assert result["status"] == "unverified"
    assert result["phase"] == ("cancelled" if cancelled else "complete")
    assert all(c["reason"] != "Not run" for s in result["subsystems"] for c in s["checks"])


def test_failed_dependency_does_not_suppress_independent_probe(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = harness.service.store.session["design_plan"]
    plan["subsystems"].append(subsystem("dependent", ["frame"]))
    harness.revision["document"]["assembly"]["parts"].append({"id": "dependent"})
    harness.revision["measurement"]["facts"]["assembly"]["occurrences"].append(
        {"path": "dependent", "definition": "dependent"}
    )

    def fail(*args: Any) -> dict[str, Any]:
        harness.calls.append(("tool", args[5]))
        return {"id": str(len(harness.calls)), "result": {"status": "fail"}}

    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", fail)
    result = finish(harness)
    assert [s["status"] for s in result["subsystems"]] == [
        "checked_fail",
        "checked_fail",
        "checked_fail",
    ]
    assert len([c for c in harness.calls if c[0] == "tool"]) == 3


def test_unresolved_qualification_does_not_block_downstream_measurements(harness):
    plan = harness.service.store.session["design_plan"]
    plan["subsystems"][0]["unresolved"] = ["Torque and tool access require qualification"]
    plan["subsystems"][1]["dependencies"] = ["frame"]
    result = finish(harness)
    assert [s["status"] for s in result["subsystems"]] == ["unverified", "checked_pass"]
    assert len([c for c in harness.calls if c[0] == "tool"]) == 2
    assert result["status"] == "unverified"


def test_missing_dependency_geometry_blocks_its_dependents(harness):
    plan = harness.service.store.session["design_plan"]
    plan["subsystems"][0]["part_ids"] = ["not_instantiated"]
    plan["subsystems"][1]["dependencies"] = ["frame"]
    result = finish(harness)
    assert [s["status"] for s in result["subsystems"]] == ["missing", "blocked"]
    assert not [c for c in harness.calls if c[0] == "tool"]


def test_no_checks_or_missing_parts_cannot_pass(harness: SimpleNamespace) -> None:
    plan = harness.service.store.session["design_plan"]
    plan["subsystems"][0]["checks"] = []
    plan["subsystems"][1]["part_ids"] = ["absent"]
    result = finish(harness)
    assert result["status"] == "unverified"
    assert [s["status"] for s in result["subsystems"]] == ["unverified", "missing"]


def test_numeric_selector_requires_current_reference(harness: SimpleNamespace) -> None:
    check = harness.service.store.session["design_plan"]["subsystems"][0]["checks"][0]
    check["request"] = {
        "tool": "hole_pattern",
        "units": "mm",
        "body": {"body_index": 0},
        "axis": [0, 0, 1],
        "holes": [{"center": [0, 0, 0], "diameter": 2}],
    }
    result = finish(harness)
    assert "exact final" in result["subsystems"][0]["checks"][0]["reason"]
    assert len([c for c in harness.calls if c[0] == "tool"]) == 1


def test_mutation_during_tool_invalidates_pass(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def mutate(*args: Any) -> dict[str, Any]:
        harness.service.store.session["engineering"]["version"] = 1
        return {"id": "probe", "result": {"status": "pass"}}

    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", mutate)
    result = finish(harness)
    assert result["stale"]
    assert result["status"] == "unverified"
    assert all(s["status"] != "checked_pass" for s in result["subsystems"])


def test_read_invalidates_old_results_after_source_or_plan_change(harness: SimpleNamespace) -> None:
    finish(harness)
    harness.service.store.session["design_plan"]["subsystems"][0]["name"] = "Changed"
    context = completion_context(harness.service, "s", "o", RID)
    assert context["stale"] and context["status"] == "unverified"
    assert context["next_target"] == "frame"  # stale probes become actionable work


def test_tampered_receipt_cannot_retain_green_context(harness: SimpleNamespace) -> None:
    finish(harness)
    harness.service.store.connection.execute("UPDATE product_checks SET document='{}' WHERE id='1'")
    context = completion_context(harness.service, "s", "o", RID)
    assert context["stale"] and context["status"] == "unverified"


def test_unused_definition_is_missing_and_target_waits_for_dependencies(
    harness: SimpleNamespace,
) -> None:
    harness.service.store.session["design_plan"]["subsystems"] = [
        subsystem("cover", ["frame"]),
        subsystem("frame"),
    ]
    harness.revision["measurement"]["facts"]["assembly"]["occurrences"] = []
    context = completion_context(harness.service, "s", "o", RID)
    assert context["next_target"] == "frame"
    assert all(s["status"] == "missing" for s in context["subsystems"])


def test_preview_baseline_restore_does_not_invalidate_candidate(harness: SimpleNamespace) -> None:
    finish(harness)
    session = harness.service.store.session
    session["proposal"] = {"revision_id": RID, "contract": session["contract"]}
    session["contract"] = {"requirements": []}
    assert not completion_context(harness.service, "s", "o", RID)["stale"]


def test_no_geometry_still_persists_explicit_report(harness: SimpleNamespace) -> None:
    harness.service.store.session["selected_revision"] = None
    result = finish(harness)
    assert result["status"] == "unverified"
    assert "No valid saved geometry" in result["reason"]
    assert harness.calls == []
    assert harness.service.store.session["completion_report"]["phase"] == "complete"


def test_scalar_probe_without_declared_geometry_cannot_complete(harness: SimpleNamespace) -> None:
    for item in harness.service.store.session["design_plan"]["subsystems"]:
        item["part_ids"] = []
    result = finish(harness)
    assert result["status"] == "unverified"
    assert all(s["status"] == "unverified" for s in result["subsystems"])


def test_resume_reuses_only_integrity_checked_current_receipts(harness: SimpleNamespace) -> None:
    finish(harness)
    harness.calls.clear()
    finish(harness)
    assert [c[0] for c in harness.calls] == ["interfaces", "report"]
    harness.service.store.connection.execute("UPDATE product_checks SET document='{}' WHERE id='1'")
    harness.calls.clear()
    finish(harness)
    assert [c[0] for c in harness.calls] == ["tool", "interfaces", "report"]


def test_round_robin_prevents_first_subsystem_consuming_all_checks(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    subs = harness.service.store.session["design_plan"]["subsystems"]
    for index, item in enumerate(subs):
        item["checks"].append({"id": f"extra_{index}", "request": copy.deepcopy(REQUEST)})
        for offset, check in enumerate(item["checks"]):
            check["request"]["allowable"]["high"] = 10 * (index + 1) + offset
    seen = []

    def tool(*args: Any) -> dict[str, Any]:
        seen.append(args[4]["allowable"]["high"])
        return {"id": str(len(seen)), "result": {"status": "pass"}}

    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", tool)
    finish(harness)
    assert seen == [10, 20, 11, 21]


def test_artifact_verification_error_invalidates_context(harness: SimpleNamespace) -> None:
    finish(harness)

    def changed(*args: Any) -> Any:
        raise ValueError("Revision artifacts changed")

    harness.service._load_revision = changed
    context = completion_context(harness.service, "s", "o", RID)
    assert context["stale"] and context["status"] == "unverified"
    assert "artifacts changed" in context["reason"]


def test_check_exception_retains_other_checks(
    harness: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> dict[str, Any]:
        harness.calls.append(("tool", args[5]))
        if len(harness.calls) == 1:
            raise TimeoutError("worker deadline")
        return {"id": "second", "result": {"status": "pass"}}

    monkeypatch.setattr("forge_mind.workspace.product_service.run_tool", fail)
    result = finish(harness)
    assert result["subsystems"][0]["checks"][0]["status"] == "unverified"
    assert result["subsystems"][1]["status"] == "checked_pass"


@pytest.fixture
def motion_harness(harness, monkeypatch, tmp_path):
    from forge_mind.validation_provenance import file_sha256

    harness.service.store.root = tmp_path
    rig = {
        "drivers": [{"id": "opening", "minimum": 20, "maximum": 100, "reference": 80}],
        "groups": [],
    }
    harness.revision["document"]["motion_rig"] = rig
    harness.revision["measurement"]["artifact_hashes"].update(
        {"part.step": "step", "document.json": "doc"}
    )
    harness.revision["measurement"]["facts"]["motion_rig"] = {
        "rig": rig,
        "bindings": [{"body_index": 0}, {"body_index": 1}],
        "revision_id": RID,
        "engine_fingerprint": "engine",
        "step_sha256": "step",
        "document_sha256": "doc",
    }
    configs = configurations(rig, MotionCheckSettings())
    harness.motion_result = {
        "status": "unverified",
        "coverage": "discrete_only",
        "collisions": [],
        "collision_count": 0,
        "errors": [],
        "requested_configurations": configs,
        "completed_configurations": configs,
        "completed_pairs": len(configs),
        "total_pairs": len(configs),
        "exact_intersections": 0,
        "disjoint_bounds": len(configs),
        "rigid_pair_reuses": 0,
        "revision_id": RID,
        "engine_fingerprint": "engine",
        "step_sha256": "step",
        "document_sha256": "doc",
        "settings": MotionCheckSettings().model_dump(),
        "scope": "Discrete configurations only; physical readiness unverified",
    }
    harness.motion_after = lambda: None
    harness.motion_error = None
    harness.motion_path = tmp_path / "motion-checks" / "s" / "motion" / "check.json"

    def motion(service, job, cancelled):
        harness.calls.append(("motion", job["max_seconds"]))
        assert job["revision_id"] == RID
        assert job["settings"] == MotionCheckSettings().model_dump()
        assert not cancelled()
        if harness.motion_error:
            raise harness.motion_error
        harness.motion_path.parent.mkdir(parents=True, exist_ok=True)
        harness.motion_path.write_text(json.dumps(harness.motion_result), encoding="utf-8")
        service.store.update(
            "s", motion_checks=[{"id": "motion", "sha256": file_sha256(harness.motion_path)}]
        )
        harness.motion_after()
        return "motion"

    monkeypatch.setattr("forge_mind.workspace.motion_service.run_motion_job", motion)
    return harness


@pytest.mark.parametrize("unlimited", [False, True])
def test_declared_rig_automatically_checks_motion_and_preserves_budget(
    motion_harness, monkeypatch, unlimited
):
    shares = []
    from forge_mind.workspace.budget import share_time

    def capture(deadline, parts):
        shares.append(parts)
        return share_time(deadline, parts)

    monkeypatch.setattr("forge_mind.workspace.completion.share_time", capture)
    # Test phase allocation independently of machine contention; expiry has separate controls.
    monkeypatch.setattr(
        "forge_mind.workspace.budget.time", SimpleNamespace(monotonic=lambda: 100.0)
    )
    h = motion_harness
    result = finish(h, deadline=None if unlimited else 160.0)
    assert [name for name, _ in h.calls] == ["tool", "tool", "interfaces", "motion", "report"]
    assert shares == [5, 4, 3, 2, 1]
    assert [seconds for _, seconds in h.calls] == (
        [None] * 5 if unlimited else [12.0, 15.0, 20.0, 30.0, 60.0]
    ), h.calls
    assert result["motion"]["status"] == "checked"
    assert result["motion"]["coverage"] == "discrete_only"
    assert result["motion"]["check_id"] == "motion"
    assert result["status"] == "checked_pass"
    assert result["physical_readiness"] == "unverified"
    assert completion_context(h.service, "s", "o", RID)["status"] == "checked_pass"


def test_no_rig_is_explicitly_not_applicable_without_worker(harness):
    result = finish(harness)
    assert result["motion"]["status"] == "not_applicable"
    assert "motion" not in [name for name, _ in harness.calls]


@pytest.mark.parametrize("measured_only", [False, True])
def test_declared_without_measured_rig_is_unverified_but_measured_only_rig_is_checked(
    motion_harness, measured_only
):
    h = motion_harness
    if measured_only:
        h.revision["document"].pop("motion_rig")
    else:
        h.revision["document"]["motion_rig"] = {}
        h.revision["measurement"]["facts"].pop("motion_rig")
    result = finish(h)
    assert result["motion"]["status"] == ("checked" if measured_only else "unverified")
    assert ("motion" in [name for name, _ in h.calls]) is measured_only
    assert h.calls[-1][0] == "report"
    if not measured_only:
        assert result["status"] == "unverified"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            {
                "collision_count": 1,
                "collisions": [
                    {"a": "jaw", "b": "guard", "overlap_mm3": 0.2, "values": {"opening": 20}}
                ],
                "status": "fail",
            },
            "checked_fail",
        ),
        ({"errors": ["Kernel failed"]}, "unverified"),
        ({"coverage": "incomplete"}, "unverified"),
        ({"completed_pairs": 0}, "unverified"),
        (
            {
                "total_pairs": 0,
                "completed_pairs": 0,
                "requested_configurations": [],
                "completed_configurations": [],
            },
            "unverified",
        ),
        ({"completed_configurations": []}, "unverified"),
        ({"revision_id": "b" * 32}, "unverified"),
        ({"step_sha256": "different"}, "unverified"),
    ],
)
def test_failed_incomplete_or_mismatched_motion_cannot_green_completion(
    motion_harness, mutation, expected
):
    h = motion_harness
    h.motion_result.update(mutation)
    result = finish(h)
    assert result["motion"]["status"] == expected
    assert result["status"] == expected
    assert h.calls[-1][0] == "report"
    if expected == "checked_fail":
        assert result["motion"]["collisions"][0]["a"] == "jaw"


def test_motion_failure_does_not_suppress_downstream_engineering_report(motion_harness):
    h = motion_harness
    h.motion_error = TimeoutError("Motion worker deadline")
    result = finish(h)
    assert result["status"] == "unverified"
    assert "Motion worker deadline" in result["motion"]["reason"]
    assert h.calls[-1][0] == "report"


@pytest.mark.parametrize("cancel", [False, True])
def test_motion_reply_cannot_green_after_cancellation_or_snapshot_change(motion_harness, cancel):
    h = motion_harness
    stopped = False

    def after():
        nonlocal stopped
        if cancel:
            stopped = True
        else:
            h.service.store.session["engineering"]["version"] += 1

    h.motion_after = after
    result = finish(h, cancelled=lambda: stopped)
    assert result["motion"]["status"] == "unverified"
    assert result["status"] == "unverified"
    assert "report" not in [name for name, _ in h.calls]


def test_tampered_motion_receipt_invalidates_cached_completion(motion_harness):
    h = motion_harness
    finish(h)
    h.motion_path.write_text("{}", encoding="utf-8")
    context = completion_context(h.service, "s", "o", RID)
    assert context["stale"]
    assert context["status"] == "unverified"


def test_rig_without_any_prepared_checks_still_receives_motion_validation(motion_harness):
    h = motion_harness
    for row in h.service.store.session["design_plan"]["subsystems"]:
        row["checks"] = []
    result = finish(h)
    assert [name for name, _ in h.calls] == ["interfaces", "motion", "report"]
    assert result["motion"]["status"] == "checked"
    assert result["physical_readiness"] == "unverified"


def test_legacy_completion_missing_motion_phase_cannot_stay_green(motion_harness):
    h = motion_harness
    finish(h)
    h.service.store.session["completion_report"].pop("motion")
    result = completion_context(h.service, "s", "o", RID)
    assert result["stale"] and result["status"] == "unverified"
    assert result["motion"]["status"] == "unverified"


def test_explicit_deadline_expiring_during_motion_discards_completion_pass(
    motion_harness, monkeypatch
):
    h = motion_harness
    now = 0
    monkeypatch.setattr("forge_mind.workspace.budget.time.monotonic", lambda: now)

    def expire():
        nonlocal now
        now = 61

    h.motion_after = expire
    result = finish(h, deadline=60)
    assert result["motion"]["status"] == "unverified"
    assert result["status"] == "unverified"
    assert "report" not in [name for name, _ in h.calls]
    assert "budget exhausted" in result["engineering_report"]["reason"]
