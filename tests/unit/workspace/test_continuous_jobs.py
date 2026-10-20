from __future__ import annotations

import json

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent, completion
from forge_mind.workspace.budget import end_time, expired, share_time, time_left
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, DesignPlan, Measurement
from forge_mind.workspace.store import Store


def test_unlimited_author_exceeds_old_time_and_action_limits(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    store = Store(tmp_path)
    sid = store.create("owner", "Review a design", Contract().model_dump())["id"]
    calls = []

    class Client:
        timeout_s = None

        def generate_structured(self, **kwargs):
            assert self.timeout_s is None
            context = json.loads(kwargs["user"])
            assert context["seconds_left"] is context["actions_left"] is None
            calls.append(context)
            clock[0] += 120
            return Action(
                kind="schema" if len(calls) <= 40 else "finish",
                query="plan",
                explanation="Inspect schema" if len(calls) <= 40 else "Review complete",
            ), LLMUsage("test", 0, 0)

    def factory(**kwargs):
        assert kwargs["timeout_s"] is None
        return Client()

    service = agent.WorkspaceService(store, Executor("trusted_local"), factory)
    observed = []

    def finalizer(service, job, deadline, cancelled, state):
        assert deadline is None and not cancelled()
        assert clock[0] > 3600
        observed.append(state)
        clock[0] += 900

    monkeypatch.setattr(completion, "finalize_completion", finalizer)
    store.enqueue(sid, "owner", {})
    service.run_job(store.claim())
    assert len(calls) == 41
    assert observed == ["needs_review"]
    session = store.get(sid, "owner")
    assert session["state"] == "needs_review"
    progress = session["completion_progress"]
    assert progress["reserved_seconds"] is progress["remaining_seconds"] is None
    json.dumps(session, allow_nan=False)


@pytest.mark.parametrize("limits", [{"max_actions": 2}, {"max_seconds": 60}])
def test_explicit_independent_budgets_still_stop(tmp_path, monkeypatch, limits):
    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    store = Store(tmp_path)
    sid = store.create("owner", "Review", Contract().model_dump())["id"]

    class Client:
        timeout_s = None

        def generate_structured(self, **kwargs):
            clock[0] += 25
            return Action(kind="schema", query="plan", explanation="Inspect"), LLMUsage(
                "test", 0, 0
            )

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(completion, "finalize_completion", lambda *args: {})
    store.enqueue(sid, "owner", limits)
    service.run_job(store.claim())
    assert store.get(sid, "owner")["state"] == "budget_exhausted"


def test_cancel_stops_unlimited_author_and_prevents_new_work(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("owner", "Review", Contract().model_dump())["id"]

    class Client:
        def generate_structured(self, **kwargs):
            store.cancel(sid, "owner")
            return Action(kind="build", code="result = 1", explanation="Too late"), LLMUsage(
                "test", 0, 0
            )

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "build", lambda *a, **kw: pytest.fail("Started after cancel"))
    store.enqueue(sid, "owner", {"preview_only": True})
    service.run_job(store.claim())
    assert store.get(sid, "owner")["state"] == "cancelled"


def test_unexpected_job_error_does_not_strand_queue(tmp_path, monkeypatch):
    store = Store(tmp_path)
    first = store.create("owner", "First", None)["id"]
    second = store.create("owner", "Second", None)["id"]
    store.enqueue(first, "owner", {})
    store.enqueue(second, "owner", {})
    service = agent.WorkspaceService(store, Executor("trusted_local"))

    def run(job):
        if job["session"] == first:
            raise RuntimeError("Unexpected failure")
        store.finish(job["id"], second, "needs_review")
        service.stop.set()

    monkeypatch.setattr(service, "run_job", run)
    service._worker()
    assert store.get(first, "owner")["state"] == "error"
    assert store.get(second, "owner")["state"] == "needs_review"


def test_optional_deadlines_are_json_safe_and_do_not_refresh_expiry(monkeypatch):
    clock = [5.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    assert end_time(None) is time_left(None) is share_time(None) is None
    assert not expired(None)
    deadline = end_time(10)
    clock[0] = 20
    assert time_left(deadline) == share_time(deadline) == 0
    assert expired(deadline)
    for invalid in [-1, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            end_time(invalid)


def test_premature_plan_only_finish_retains_preview_resume(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Build a bracket", Contract().model_dump())["id"]
    calls = []
    plan = DesignPlan(
        objective="Build a bracket",
        subsystems=[{"id": "bracket", "name": "Bracket", "part_ids": ["bracket"]}],
    )

    class Client:
        def generate_structured(self, **kwargs):
            calls.append(json.loads(kwargs["user"]))
            return Action(kind="finish", explanation="Stopped early", plan=plan), LLMUsage(
                "test", 0, 0
            )

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    first = store.enqueue(sid, "owner", {"preview_only": True})
    service.run_job(store.claim())
    session = store.get(sid, "owner")
    assert len(calls) == 2
    assert session["state"] == "needs_review"
    assert session["resume_candidate"]["revision_id"] is None
    assert session["proposal"] is None
    assert "incomplete_finish_job_id" not in session
    second = store.enqueue(sid, "owner", {"preview_only": True, "resume": True})
    assert second != first
    service.run_job(store.claim())
    assert len(calls) == 4
    assert store.get(sid, "owner")["state"] == "needs_review"
    assert store.get(sid, "owner")["resume_candidate"] is not None


def test_numeric_pass_cannot_override_incomplete_finish_resume(tmp_path, monkeypatch):
    store = Store(tmp_path)
    contract = Contract.model_validate(
        {
            "requirements": [
                {
                    "id": "width",
                    "text": "Width 10 mm",
                    "source_quote": "10 mm",
                    "source": "explicit",
                    "metric": "extent_x",
                    "value": 10,
                }
            ]
        }
    )
    sid = store.create("owner", "Complete the assembly", contract.model_dump())["id"]
    rid = "a" * 32
    measurement = Measurement(
        revision_id=rid,
        contract_hash=contract.fingerprint(),
        artifact_hashes={},
        engine_fingerprint=agent.engine_fingerprint(),
        geometry_valid=True,
        requirements_satisfied=True,
        facts={"extent_x": 10.0},
    )
    revision = {
        "id": rid,
        "contract": contract.model_dump(),
        "measurement": measurement.model_dump(),
    }
    store.add_revision(sid, revision)
    store.update(sid, selected_revision=rid, best_revision=rid)
    plan = DesignPlan(
        objective="Complete the assembly",
        subsystems=[{"id": "cover", "name": "Missing cover", "part_ids": ["cover"]}],
    )

    class Client:
        def generate_structured(self, **kwargs):
            return Action(kind="finish", explanation="Incomplete", plan=plan), LLMUsage(
                "test", 0, 0
            )

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda *args, **kwargs: revision)
    store.enqueue(sid, "owner", {"preview_only": True})
    service.run_job(store.claim())
    session = store.get(sid, "owner")
    assert session["state"] == "needs_review"
    assert session["completion_report"]["author_state"] == "needs_review"
    assert session["resume_candidate"]["revision_id"] == rid
    store.enqueue(sid, "owner", {"preview_only": True, "resume": True})
    resumed = store.claim()
    assert resumed["revision_id"] == rid


@pytest.mark.parametrize("marker", [None, "old-job", "current-job"])
def test_incomplete_finish_checkpoint_is_job_scoped_and_keeps_proposal(tmp_path, marker):
    store = Store(tmp_path)
    sid = store.create("owner", "Review a bracket", Contract().model_dump())["id"]
    jid = store.enqueue(sid, "owner", {"preview_only": True})
    store.claim()
    store.update(
        sid,
        selected_revision="candidate",
        incomplete_finish_job_id=jid if marker == "current-job" else marker,
    )
    proposal = {"revision_id": "candidate", "eligible": False}
    store.finish(jid, sid, "needs_review")
    store.complete_preview(sid, jid, None, proposal)
    session = store.get(sid, "owner")
    assert session["proposal"] == proposal
    assert bool(session["resume_candidate"]) == (marker == "current-job")
    assert "incomplete_finish_job_id" not in session
