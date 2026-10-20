"""Staged intent uses real owned plan admission, without CAD or a provider."""

from __future__ import annotations

import json

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.author_guidance import build_author_guidance
from forge_mind.workspace.authoring import authoring_schema
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.generation_policy import native_call_allowance
from forge_mind.workspace.models import Action, Contract, Requirement
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.plan_intent import PlanIntent, lower_intent
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import Store


def intent():
    return PlanIntent.model_validate(
        {
            "objective": "Make a useful connection",
            "parts": [
                {
                    "id": "base",
                    "role": "make",
                    "purpose": "Mount the connection",
                    "requirement_ids": ["fit"],
                },
                {"id": "pin", "role": "buy", "purpose": "Retain the joint"},
            ],
            "first_part_id": "base",
            "unresolved": ["The load and retention proof remain unknown"],
        }
    )


def run_author(tmp_path, monkeypatch, steps, *, mutate=None, delivery=False):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    contract = Contract(
        requirements=[
            Requirement(
                id="fit", text="Make a useful connection", source_quote="Make a useful connection"
            )
        ]
    )
    store = Store(tmp_path)
    sid = store.create(
        "owner",
        "Make a useful connection",
        contract.model_dump(),
        authoring_mode="native_operations",
    )["id"]
    contexts, schemas, timeouts = [], [], []
    pending = iter(steps)

    class Client:
        model = "scripted-staged-test"
        timeout_s = None

        def generate_structured(self, **kwargs):
            contexts.append(json.loads(kwargs["user"]))
            schemas.append(kwargs["output_model"])
            timeouts.append(self.timeout_s)
            value = next(pending)
            if callable(value):
                value = value(contexts[-1])
            if mutate:
                mutate(store, sid, len(contexts))
            return kwargs["output_model"].model_validate(value.model_dump()), LLMUsage(
                self.model, 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(
        sid,
        "owner",
        {"max_actions": len(steps), "max_seconds": 1500, "delivery_requested": delivery},
    )
    job = store.claim()
    assert job is not None
    state = service._run_agent_job(job)
    store.finish(job["id"], sid, state)
    return service, sid, contexts, schemas, timeouts, state


def test_startup_uses_small_typed_input_then_existing_native_actions(tmp_path, monkeypatch):
    service, sid, contexts, schemas, timeouts, state = run_author(
        tmp_path,
        monkeypatch,
        [intent(), Action(kind="schema", query="construction", explanation="Read construction")],
    )
    assert state != "error"
    assert schemas == [PlanIntent, NativeAction]
    assert contexts[0]["phase"] == "startup_intent"
    assert "native_authoring" not in contexts[0]
    assert "authoritative_schemas" not in contexts[0]
    assert contexts[1]["design_plan"]["operation_plan"]["steps"][0]["feature_ids"] == []
    assert contexts[0]["frozen_contract"] == contexts[1]["frozen_contract"]
    assert all(0 < allowance <= 300 for allowance in timeouts)
    session = service.store.get(sid, "owner")
    assert session["plan_version"] == 1
    assert session["revisions"] == []
    assert not session.get("accepted_revision")
    assert authoring_schema("plan_intent")["schema"] == PlanIntent.model_json_schema()


def test_cached_schema_survives_an_intervening_non_schema_action(tmp_path, monkeypatch):
    service, sid, contexts, _, _, state = run_author(
        tmp_path,
        monkeypatch,
        [
            intent(),
            Action(kind="schema", query="plan_patch", explanation="Read exact plan edit syntax"),
            Action(kind="read_part", part_id="base", explanation="Inspect the planned part"),
            Action(kind="schema", query="feature:box", explanation="Read feature syntax"),
        ],
    )
    assert state != "error"
    exact = authoring_schema("plan_patch")
    assert contexts[2]["authoritative_schemas"]["plan_patch"] == exact
    assert contexts[3]["authoritative_schemas"]["plan_patch"] == exact
    # No revision exists: the real read path returns an error without CAD.
    assert "tool_error" in contexts[3]["history"][-1]
    assert contexts[3]["history"][-1]["action_result"]["applied"] is False
    for context in contexts[2:]:
        assert context["schemas_already_read"] == ["plan_patch"]
        catalog = {row["name"]: row for row in context["schema_catalog"]}
        assert set(catalog) == {"plan_patch"}
        assert catalog["plan_patch"]["retrieve"] == "schema query=plan_patch"
        assert "is_summary" not in context["authoritative_schemas"]["plan_patch"]
        assert all("authoring_schema" not in row for row in context["history"])
    assert service.store.get(sid, "owner")["schema_cache"]["plan_patch"] == exact
    assert service.store.get(sid, "owner")["revisions"] == []


def test_invalid_startup_returns_to_normal_repair_with_exact_refusal(tmp_path, monkeypatch):
    bad = intent().model_copy(deep=True)
    bad.parts[0].requirement_ids = ["unknown"]
    service, sid, contexts, schemas, _, state = run_author(
        tmp_path,
        monkeypatch,
        [bad, Action(kind="schema", query="plan_intent", explanation="Read exact intent syntax")],
    )
    assert state != "error"
    assert schemas == [PlanIntent, NativeAction]
    assert contexts[1]["failed_submission"]["startup_intent_rejected"] is True
    assert (
        json.loads(contexts[1]["failed_submission"]["submitted_action"]["document_json"])
        == bad.model_dump()
    )
    assert service.store.get(sid, "owner").get("design_plan") is None
    assert service.store.get(sid, "owner")["revisions"] == []
    with service.store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM plan_drafts").fetchone()[0] == 0


def test_late_intent_cannot_overwrite_concurrently_admitted_plan(tmp_path, monkeypatch):
    def mutate(store, sid, call):
        if call == 1:
            contract = Contract.model_validate(store.get(sid, "owner")["contract"])
            # A competing authority update is simulated without starting another worker.
            plan = lower_intent(intent(), contract)
            store.update(sid, design_plan=plan.model_dump(), plan_version=7, plan_hash="f" * 64)

    service, sid, _, _, _, _ = run_author(tmp_path, monkeypatch, [intent()], mutate=mutate)
    assert service.store.get(sid, "owner")["plan_version"] == 7
    assert service.store.get(sid, "owner")["revisions"] == []


def test_empty_intent_does_not_finish_requested_delivery(tmp_path, monkeypatch):
    service, sid, _, _, _, state = run_author(
        tmp_path,
        monkeypatch,
        [intent(), Action(kind="finish", explanation="Attempt early handoff")],
        delivery=True,
    )
    assert state != "complete"
    session = service.store.get(sid, "owner")
    assert session["revisions"] == []
    assert session.get("native_review", {}).get("status") != "ready_for_validation"
    assert session.get("native_delivery", {}).get("status") != "captured"


def test_additive_part_inventory_preserves_initial_scope():
    from forge_mind.workspace.edit_contracts import EditBasis

    contract = Contract(requirements=[Requirement(id="fit", text="Fit", source_quote="Fit")])
    original = lower_intent(intent(), contract)
    binding = snapshot(original, 1, contract)
    basis = EditBasis.model_validate(
        {
            "revision_id": None,
            "document_hash": None,
            "contract_hash": contract.fingerprint(),
            "plan_version": 1,
            "plan_hash": binding["plan_hash"],
            "input_version": 0,
            "source_fingerprint": "e" * 64,
        }
    )
    from forge_mind.workspace.models import OperationStep

    patch = PlanPatch(
        basis=basis,
        part_ids=["bushing"],
        operation_steps=[
            OperationStep(id="bushing", part_id="bushing", expected_effect="Make the new bushing")
        ],
    )
    updated = apply_plan_patch(original, patch, contract)
    assert updated.operation_plan.part_ids == ["base", "pin", "bushing"]
    assert original.operation_plan.part_ids == ["base", "pin"]
    assert updated.subsystems == original.subsystems
    assert updated.operation_plan.steps[:2] == original.operation_plan.steps
    assert (
        apply_plan_patch(original, PlanPatch(basis=basis, part_ids=["base"]), contract) == original
    )


def test_present_first_part_does_not_hide_remaining_parts_or_empty_ownership():
    from forge_mind.workspace.completion import _structure
    from forge_mind.workspace.document import DesignDocument
    from forge_mind.workspace.operation_plan import coverage

    contract = Contract(requirements=[Requirement(id="fit", text="Fit", source_quote="Fit")])
    plan = lower_intent(intent(), contract)
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "base",
                        "name": "Base",
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 20, "y": 10, "z": 4},
                            }
                        ],
                        "output": "body",
                    }
                ],
                "roots": [{"id": "base", "definition": "base"}],
            }
        }
    )
    revision = {
        "document": document.model_dump(),
        "measurement": {
            "facts": {"assembly": {"occurrences": [{"path": "base", "definition": "base"}]}}
        },
    }
    rows = {row["id"]: row for row in _structure(plan.model_dump(), revision)}
    assert rows["startup_part_0"]["status"] == "ready"
    assert rows["startup_part_1"]["missing_part_ids"] == ["pin"]
    assert rows["request_scope"]["missing_part_ids"] == ["pin"]
    report = coverage(snapshot(plan, 1, contract), document)
    assert report["graph_complete"] is False
    assert all(not row["effect_verified"] for row in report["steps"])


@pytest.mark.parametrize("remaining,expected", [(None, 300), (1500, 300), (17.5, 17.5)])
def test_per_call_allowance_retains_episode_limit(remaining, expected):
    assert native_call_allowance(remaining) == expected


def test_staged_guidance_requires_exact_part_identity_and_full_closure():
    text = build_author_guidance(
        {"authoring_mode": "native_operations", "staged_native_planning": True}, "trusted_local"
    )
    assert "first save a detailed" not in text
    assert "exact planned part ID" in text
    assert "Fill required functional interfaces" in text
    assert "physical qualification" in text
