"""An invalid public schema name returns a recoverable author tool rejection."""

import json

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.lab_api import lab_schema
from forge_mind.workspace.models import Action, Contract, DesignPlan
from forge_mind.workspace.store import Store


@pytest.mark.parametrize("query", ["lab:prediciton", "lab:"])
def test_unknown_lab_schema_can_be_corrected_in_the_same_author_job(tmp_path, monkeypatch, query):
    store = Store(tmp_path)
    sid = store.create("owner", "Read experiment syntax", Contract().model_dump())["id"]
    actions = iter(
        [
            Action(kind="schema", query=query, explanation="Read a schema with a typo"),
            Action(kind="schema", query="lab:prediction", explanation="Read the valid schema"),
            Action(kind="finish", explanation="Schema reviewed; geometry remains unbuilt"),
        ]
    )
    contexts = []

    class Client:
        model = "scripted-schema-test"

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return next(actions), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    job = store.claim()
    assert job is not None
    state = service._run_agent_job(job)
    assert state == "needs_review"
    assert len(contexts) == 3
    rejection = contexts[1]["history"][-1]
    assert query in rejection["tool_error"]
    assert "prediction" in rejection["tool_error"]
    assert rejection["action_result"]["retryable"] is True
    assert rejection["action_result"]["applied"] is False
    assert query not in contexts[1]["schemas_already_read"]
    assert query not in contexts[1]["authoritative_schemas"]
    assert contexts[2]["authoritative_schemas"]["lab:prediction"] == lab_schema()["prediction"]
    assert store.get(sid, "owner")["revisions"] == []


@pytest.mark.parametrize("query", ["lab:prediciton", "tool:unknown_tool"])
def test_schema_name_is_validated_before_attached_plan_can_mutate_state(
    tmp_path, monkeypatch, query
):
    store = Store(tmp_path)
    sid = store.create("owner", "Keep schema failures atomic", Contract().model_dump())["id"]
    plan = DesignPlan(objective="Do not admit this plan with an invalid tool")
    actions = iter(
        [
            Action(kind="schema", query=query, plan=plan, explanation="Invalid schema with plan"),
            Action(kind="schema", query="lab:prediction", explanation="Read valid schema"),
            Action(kind="finish", explanation="No geometry was built"),
        ]
    )
    contexts = []

    class Client:
        model = "scripted-schema-test"

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return next(actions), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    state = service._run_agent_job(store.claim())
    assert state == "needs_review"
    assert len(contexts) == 3
    assert contexts[1]["design_plan"] is None
    rejection = contexts[1]["history"][-1]["action_result"]
    assert rejection["applied"] is False
    assert rejection["retryable"] is True
    assert store.get(sid, "owner").get("design_plan") is None
    assert query not in contexts[1]["authoritative_schemas"]


@pytest.mark.parametrize(
    "query",
    [
        "lab:prediction",
        "feature:box",
        "tool:hole_pattern",
        "feature:unknown_feature",
        "construction:unknown_topic",
        "unknown_schema",
    ],
)
def test_valid_schema_with_attached_plan_keeps_existing_success_behavior(
    tmp_path, monkeypatch, query
):
    store = Store(tmp_path)
    sid = store.create("owner", "Read syntax with a useful plan", Contract().model_dump())["id"]
    plan = DesignPlan(objective="Review exact tool syntax")
    actions = iter(
        [
            Action(kind="schema", query=query, plan=plan, explanation="Read valid syntax and plan"),
            Action(kind="finish", explanation="Geometry remains unbuilt"),
        ]
    )
    contexts = []

    class Client:
        model = "scripted-schema-test"

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            return next(actions), LLMUsage(self.model, 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 2, "max_seconds": 120})
    state = service._run_agent_job(store.claim())
    assert state == "needs_review"
    assert len(contexts) == 2
    assert contexts[1]["design_plan"] == plan.model_dump()
    assert query in contexts[1]["authoritative_schemas"]
    assert store.get(sid, "owner")["design_plan"] == plan.model_dump()
    schema = contexts[1]["authoritative_schemas"][query]
    if query == "feature:unknown_feature":
        assert schema["unknown_kind"] == "unknown_feature" and "box" in schema["available"]
    elif query == "construction:unknown_topic":
        assert schema["unknown_topic"] == "unknown_topic" and schema["recipes"]
    elif query == "unknown_schema":
        assert "document" in schema["available"]


def test_internal_schema_keyerror_is_not_disguised_as_a_recoverable_name_error(
    tmp_path, monkeypatch
):
    store = Store(tmp_path)
    sid = store.create("owner", "Expose internal schema defects", Contract().model_dump())["id"]

    class Client:
        model = "scripted-schema-test"

        def generate_structured(self, **kwargs):
            return Action(
                kind="schema", query="lab:prediction", explanation="Read schema"
            ), LLMUsage(self.model, 0, 0)

    def broken_schema():
        raise KeyError("internal schema invariant")

    monkeypatch.setattr("forge_mind.workspace.lab_api.lab_schema", broken_schema)
    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    store.enqueue(sid, "owner", {"max_actions": 2, "max_seconds": 120})
    assert service._run_agent_job(store.claim()) == "error"
    assert "internal schema invariant" in str(store.get(sid, "owner")["events"])
