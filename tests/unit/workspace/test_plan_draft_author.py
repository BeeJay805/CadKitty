"""Scripted plan-only author actions exercise recovery without CAD or a model provider."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from forge_mind.llm.client import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.author_guidance import build_author_guidance
from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, Requirement
from forge_mind.workspace.native_authoring import NativeAction, native_bootstrap
from forge_mind.workspace.plan_draft import PlanDraftRead, read_draft, summary
from forge_mind.workspace.store import Store

PROMPT = "Make a native mechanism"


def plan_payload(*, large: bool = False, bad_frame: bool = False) -> dict[str, Any]:
    return {
        "objective": PROMPT,
        "operation_plan": {
            "part_ids": ["mount", "pin"],
            "frames": [{"id": "pivot", "description": "Shared pivot", "part_id": "mount"}],
            "steps": [
                {
                    "id": "make_pin",
                    "part_id": "pin",
                    "frame_id": "pivot" if bad_frame else None,
                    "feature_ids": ["body"],
                    "requirement_ids": ["function"],
                    "expected_effect": "The pin supports the connection",
                }
            ],
            "dimensions": [
                {
                    "id": f"fact_{index}",
                    "value": 1.0,
                    "source": "assumption",
                    "source_quote": "Unverified design assumption. " * 20,
                }
                for index in range(75 if large else 0)
            ],
        },
    }


def action(query: str, value: dict[str, Any]) -> Action:
    return Action(
        kind="plan",
        query=query,
        explanation="Prepare or repair intent",
        document_json=json.dumps(value),
    )


def scripted(
    tmp_path,
    monkeypatch,
    steps: list[Action | Callable[[dict[str, Any]], Action]],
    *,
    native: bool = True,
):
    contract = Contract(requirements=[Requirement(id="function", text=PROMPT, source_quote=PROMPT)])
    store = Store(tmp_path)
    sid = store.create(
        "owner",
        PROMPT,
        contract.model_dump(),
        authoring_mode="native_operations" if native else None,
    )["id"]
    contexts = []
    iterator = iter(steps)

    class Client:
        model = "scripted-test"

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            step = next(iterator)
            result = step(context) if callable(step) else step
            return kwargs["output_model"].model_validate(result.model_dump()), LLMUsage(
                self.model, 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("No worker may start"))
    store.enqueue(sid, "owner", {"max_actions": len(steps)})
    job = store.claim()
    assert job is not None
    state = service._run_agent_job(job)
    assert state != "error", store.get(sid, "owner")["events"]
    assert len(contexts) == len(steps)
    return service, sid, contexts, state


def test_large_rejected_plan_is_recoverable_in_scoped_author_context(tmp_path, monkeypatch):
    raw = plan_payload(large=True, bad_frame=True)
    assert len(json.dumps(raw).encode()) > 32000
    service, sid, contexts, _ = scripted(
        tmp_path,
        monkeypatch,
        [
            action("", raw),
            action("draft_read", {"collection": "operation_steps", "record_id": "make_pin"}),
        ],
    )
    failure = contexts[1]["failed_submission"]
    assert "submitted_action" not in failure
    assert failure["plan_draft"]["source_bytes"] > 32000
    assert contexts[1]["plan_draft"]["status"] == "unadmitted_draft"
    assert "operation_plan" not in contexts[1]["plan_draft"]
    draft = read_draft(service.store, sid, "owner", PlanDraftRead(collection="operation_steps"))
    assert draft["records"] == raw["operation_plan"]["steps"]
    session = service.store.get(sid, "owner")
    assert session.get("design_plan") is None
    assert session.get("plan_version", 0) == 0
    assert session["revisions"] == []
    assert service.store.get(sid, "owner")["failed_submission"]["geometry_produced"] is False


def test_scoped_repair_and_explicit_commit_admit_the_retained_plan(tmp_path, monkeypatch):
    raw = plan_payload(large=True, bad_frame=True)

    def repair(context):
        draft = context["plan_draft"]
        return action(
            "draft_patch",
            {
                "basis": context["edit_basis"],
                "draft_version": draft["draft_version"],
                "draft_hash": draft["draft_hash"],
                "changes": {
                    "operation_frames": [
                        {"id": "pivot", "description": "Shared pivot", "part_id": None}
                    ]
                },
            },
        )

    def commit(context):
        draft = context["plan_draft"]
        assert draft["draft_version"] == 2
        return action(
            "draft_commit",
            {
                "basis": context["edit_basis"],
                "draft_version": draft["draft_version"],
                "draft_hash": draft["draft_hash"],
            },
        )

    service, sid, contexts, _ = scripted(tmp_path, monkeypatch, [action("", raw), repair, commit])
    session = service.store.get(sid, "owner")
    assert session["plan_version"] == 1
    assert session["design_plan"]["operation_plan"]["frames"][0]["part_id"] is None
    assert len(session["design_plan"]["operation_plan"]["dimensions"]) == 75
    assert session["failed_submission"] is None
    assert summary(service.store, sid, "owner") is None
    assert session["revisions"] == []
    assert contexts[2]["plan_draft"]["cad_applied"] is False


def test_failed_draft_commit_retains_identity_and_refusal_for_next_call(tmp_path, monkeypatch):
    raw = plan_payload(bad_frame=True)

    def commit(context):
        draft = context["plan_draft"]
        return action(
            "draft_commit",
            {
                "basis": context["edit_basis"],
                "draft_version": draft["draft_version"],
                "draft_hash": draft["draft_hash"],
            },
        )

    service, sid, contexts, _ = scripted(
        tmp_path,
        monkeypatch,
        [action("", raw), commit, action("draft_read", {"collection": "operation_frames"})],
    )
    assert contexts[2]["plan_draft"]["draft_version"] == contexts[1]["plan_draft"]["draft_version"]
    assert contexts[2]["plan_draft"]["draft_hash"] == contexts[1]["plan_draft"]["draft_hash"]
    assert contexts[2]["failed_submission"]["plan_draft"] is not None
    assert service.store.get(sid, "owner").get("plan_version", 0) == 0
    assert service.store.get(sid, "owner")["revisions"] == []


def test_initial_typed_fragment_applies_nothing_before_commit(tmp_path, monkeypatch):
    raw = plan_payload()

    def begin(context):
        assert context["plan_draft"] is None
        return action(
            "draft_patch",
            {
                "basis": context["edit_basis"],
                "draft_version": 0,
                "draft_hash": None,
                "changes": {
                    "objective": PROMPT,
                    "part_ids": ["mount", "pin"],
                    "operation_steps": raw["operation_plan"]["steps"],
                },
            },
        )

    service, sid, contexts, _ = scripted(
        tmp_path, monkeypatch, [begin, action("draft_read", {"collection": "operation_steps"})]
    )
    assert contexts[1]["plan_draft"]["draft_version"] == 1
    assert contexts[1]["design_plan"] is None
    assert service.store.get(sid, "owner")["revisions"] == []


@pytest.mark.parametrize("raw", ['{"objective":', '{"objective":"one","objective":"two"}'])
def test_malformed_or_ambiguous_full_plan_is_not_saved_as_a_draft(tmp_path, monkeypatch, raw):
    bad = Action(kind="plan", explanation="Rejected input", document_json=raw)
    service, sid, _, _ = scripted(
        tmp_path, monkeypatch, [bad, Action(kind="finish", explanation="Input remains unresolved")]
    )
    assert summary(service.store, sid, "owner") is None
    assert service.store.get(sid, "owner")["revisions"] == []


def test_draft_queries_do_not_run_through_hybrid_authoring(tmp_path, monkeypatch):
    service, sid, _, _ = scripted(
        tmp_path, monkeypatch, [action("draft_read", {"collection": "metadata"})], native=False
    )
    assert (
        "require native mode"
        in service.store.get(sid, "owner")["failed_submission"]["validation_error"]
    )
    assert summary(service.store, sid, "owner") is None


def test_draft_schema_discovery_does_not_expand_native_action_output():
    for name in ("plan_draft_read", "plan_draft_patch", "plan_draft_commit"):
        assert authoring_schema(name)["name"] == name
        assert name in authoring_schema("index")["available"]
    assert "plan_draft" in native_bootstrap()["discovery"]
    assert "$defs" not in NativeAction.model_json_schema()
    simple = build_author_guidance({"request": "A plain block"}, "trusted_local")
    assert len(simple) < 7000
    assert "draft_commit" in build_author_guidance(
        {"authoring_mode": "native_operations"}, "trusted_local"
    )
