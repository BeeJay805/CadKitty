from __future__ import annotations

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent, completion
from forge_mind.workspace.agent_memory import cache_schema
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, DesignPlan
from forge_mind.workspace.store import Store


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("exit_kind", ["finish", "timeout", "late_build", "actions"])
def test_every_author_exit_keeps_lease_and_reserved_validation(
    tmp_path, monkeypatch, preview, exit_kind
):
    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    store = Store(tmp_path)
    sid = store.create("owner", "Design an assembly", Contract().model_dump())["id"]

    class Client:
        timeout_s = 0.0

        def generate_structured(self, **kwargs):
            assert self.timeout_s <= 45
            if exit_kind == "timeout":
                clock[0] += self.timeout_s
                raise LLMError("Codex call cancelled or exceeded its wall-clock limit")
            if exit_kind == "late_build":
                clock[0] = 46
                return Action(kind="build", explanation="Too late", code="result = 1"), LLMUsage(
                    "test", 0, 0
                )
            return Action(
                kind="finish" if exit_kind == "finish" else "schema",
                explanation="Done",
                query="plan",
            ), LLMUsage("test", 0, 0)

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "build", lambda *a, **kw: pytest.fail("Late worker started"))
    observed = []

    def finalizer(service, job, deadline, cancelled, state):
        assert not cancelled()
        assert store.get(sid, "owner")["state"] == "running"
        assert deadline == 60
        assert deadline - clock[0] >= 14
        observed.append(state)

    monkeypatch.setattr(completion, "finalize_completion", finalizer)
    store.enqueue(sid, "owner", {"max_seconds": 60, "max_actions": 1, "preview_only": preview})
    job = store.claim()
    service.run_job(job)
    assert observed == ["needs_review" if exit_kind == "finish" else "budget_exhausted"]
    assert store.get(sid, "owner")["completion_progress"]["phase"] == "finished"


def test_contract_extraction_uses_author_deadline(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    store = Store(tmp_path)
    sid = store.create("owner", "A new mechanism", None)["id"]
    observed = []

    class Client:
        timeout_s = 0.0

        def generate_structured(self, **kw):
            assert kw["output_model"] is Contract
            assert self.timeout_s == 75
            clock[0] = 75
            raise LLMError("timed out")

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(
        completion,
        "finalize_completion",
        lambda s, j, d, c, state: observed.append((d - clock[0], state)),
    )
    store.enqueue(sid, "owner", {"max_seconds": 100, "max_actions": 2})
    service.run_job(store.claim())
    assert observed == [(25, "budget_exhausted")]


def test_finish_attached_plan_is_saved_before_validation(tmp_path, monkeypatch):
    store = Store(tmp_path)
    sid = store.create("owner", "A bracket", Contract().model_dump())["id"]
    plan = DesignPlan(
        objective="A bracket",
        subsystems=[{"id": "bracket", "name": "Bracket", "unresolved": ["Load unspecified"]}],
    )

    class Client:
        def generate_structured(self, **kw):
            return Action(kind="finish", explanation="Missing loads", plan=plan), LLMUsage(
                "test", 0, 0
            )

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    observed = []
    monkeypatch.setattr(
        completion,
        "finalize_completion",
        lambda *a: observed.append(store.get(sid, "owner")["design_plan"]),
    )
    store.enqueue(sid, "owner", {"max_seconds": 60, "max_actions": 1})
    service.run_job(store.claim())
    assert observed == [plan.model_dump()]


def test_schema_cache_contains_actual_content_and_has_a_ceiling():
    cached = cache_schema({}, "plan", {"schema": {"properties": {"objective": "string"}}})
    assert cached["plan"]["schema"]["properties"]["objective"] == "string"
    for index in range(10):
        cached = cache_schema(cached, str(index), {"data": "x" * 40000})
    assert len(cached) <= 3
    assert "plan" not in cached


def test_first_resume_model_failure_retains_candidate_for_validation(tmp_path, monkeypatch):
    from forge_mind.workspace.models import Measurement

    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "A bracket", contract.model_dump())["id"]
    rid = "a" * 32
    revision = {
        "id": rid,
        "contract": contract.model_dump(),
        "measurement": Measurement(
            revision_id=rid,
            contract_hash=contract.fingerprint(),
            artifact_hashes={},
            engine_fingerprint="historical",
            geometry_valid=False,
        ).model_dump(),
    }
    store.add_revision(sid, revision)
    original = store.enqueue(sid, "owner", {"preview_only": True})
    assert store.claim()
    store.update(sid, selected_revision=rid)
    store.finish(original, sid, "budget_exhausted")
    store.complete_preview(sid, original, None, None)

    class Client:
        def generate_structured(self, **kw):
            raise LLMError("invalid_json_schema")

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda *a, **kw: revision)
    observed = []
    monkeypatch.setattr(
        completion,
        "finalize_completion",
        lambda *a: observed.append(store.get(sid, "owner")["selected_revision"]),
    )
    store.enqueue(sid, "owner", {"resume": True, "preview_only": True, "max_seconds": 60})
    service.run_job(store.claim())
    assert observed == [rid]
    assert store.get(sid, "owner")["resume_candidate"]["revision_id"] == rid
