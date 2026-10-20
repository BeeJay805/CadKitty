"""Executable intent amendments retain normal admission and never create proof."""

from copy import deepcopy

import pytest
from tests.unit.workspace.test_plan_patch import contract, document, plan

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.completion_coverage import functional_coverage
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import DesignPlan
from forge_mind.workspace.operation_plan import coverage, snapshot
from forge_mind.workspace.plan_patch import PlanPatch
from forge_mind.workspace.store import Store


def case(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    frozen = contract()
    sid = service.store.create(
        "owner", "Fit the plate", frozen.model_dump(), authoring_mode="native_operations"
    )["id"]
    intent = plan().model_copy(update={"functional_interfaces": [], "requirement_dispositions": []})
    service.store.set_design_plan(sid, "owner", intent, expected_plan_version=0)
    mapping = functional_coverage(intent, frozen)
    context = {
        "authoring_mode": "native_operations",
        "frozen_contract": frozen.model_dump(),
        "design_plan": intent.model_dump(),
        "edit_basis": service.store.edit_basis(sid, "owner").model_dump(),
        "completion": {**mapping, "functional_coverage": mapping},
    }
    return service, sid, context


def amendment(context, workflow=None):
    from forge_mind.workspace.completion_amendment import completion_amendment

    return completion_amendment(context, workflow)


def authored_patch(route, **changes):
    return PlanPatch.model_validate({"basis": route["basis"], **changes})


def test_current_missing_mapping_has_exact_executable_schema_without_authored_intent(tmp_path):
    _, _, context = case(tmp_path)
    original = deepcopy(context)
    route = amendment(context)
    assert route["requirement_ids"] == ["fit"]
    assert route["basis"] == context["edit_basis"]
    assert route["action"] == {"kind": "plan", "query": "patch", "payload_field": "document_json"}
    assert set(route["schema"]["properties"]) == {
        "basis",
        "requirement_dispositions",
        "functional_interfaces",
    }
    assert route["retained_requirement_dispositions"] == []
    assert "unsupported_checks" not in route
    assert context == original
    route["basis"]["plan_version"] = 999
    assert context == original


def test_real_admission_applies_authored_disposition_without_geometry_or_source_pass(tmp_path):
    service, sid, context = case(tmp_path)
    route = amendment(context)
    before = service.store.get(sid, "owner")
    result = patch_plan(
        service,
        sid,
        "owner",
        authored_patch(
            route,
            requirement_dispositions=[
                {
                    "requirement_id": "fit",
                    "physical_validation": ["Physical fit remains unverified"],
                }
            ],
        ),
    )
    saved = service.store.get(sid, "owner")
    assert result["plan_applied"] is True and result["cad_applied"] is False
    assert saved["contract"] == before["contract"]
    assert saved.get("selected_revision") == before.get("selected_revision")
    assert service.store.revisions(sid) == []
    mapped = functional_coverage(saved["design_plan"], saved["contract"])
    assert mapped["status"] == "unverified"
    assert (
        coverage(
            snapshot(saved["design_plan"], saved["plan_version"], saved["contract"]), document()
        )["steps"][0]["graph_present"]
        is True
    )
    missing = document().model_copy(update={"features": [], "outputs": []})
    assert (
        coverage(snapshot(saved["design_plan"], saved["plan_version"], saved["contract"]), missing)[
            "steps"
        ][0]["graph_present"]
        is False
    )


@pytest.mark.parametrize("invalid", ["stale", "foreign", "unknown", "dropped"])
def test_existing_patch_guards_reject_invalid_amendment_without_mutation(tmp_path, invalid):
    service, sid, context = case(tmp_path)
    route = amendment(context)
    changes = {
        "requirement_dispositions": [
            {"requirement_id": "fit", "unsupported_checks": ["Physical qualification unavailable"]}
        ]
    }
    if invalid in {"stale", "dropped"}:
        patch_plan(service, sid, "owner", authored_patch(route, **changes))
        if invalid == "dropped":
            route["basis"] = service.store.edit_basis(sid, "owner").model_dump()
            changes["requirement_dispositions"][0]["unsupported_checks"] = []
    if invalid == "unknown":
        changes["requirement_dispositions"][0]["requirement_id"] = "unknown"
    before = service.store.session_record(sid, "owner")
    with pytest.raises(ValueError):
        patch_plan(
            service,
            sid,
            "foreign" if invalid == "foreign" else "owner",
            authored_patch(route, **changes),
        )
    assert service.store.session_record(sid, "owner") == before


@pytest.mark.parametrize("status", ["published", "stale", "inactive", "approved", "not_prepared"])
def test_approved_or_unavailable_workflow_never_offers_plan_mutation(tmp_path, status):
    _, _, context = case(tmp_path)
    assert amendment(context, {"status": status, "approval_required": True}) is None


def test_review_subsystem_offers_only_actual_missing_dispositions(tmp_path):
    _, _, context = case(tmp_path)
    context["completion"] = {
        "next_target": "mount",
        "next_work_category": "review",
        "subsystems": [{"id": "mount", "requirement_ids": ["fit"], "geometry_declared": True}],
        "requirement_dispositions": [{"requirement_id": "fit", "declared": False}],
    }
    assert amendment(context)["requirement_ids"] == ["fit"]
    context["completion"]["requirement_dispositions"][0]["declared"] = True
    assert amendment(context) is None


@pytest.mark.parametrize(
    "change", ["startup", "other_target", "construction", "unknown", "invalid_basis"]
)
def test_unrelated_or_invalid_context_never_synthesizes_amendment(tmp_path, change):
    _, _, context = case(tmp_path)
    if change == "startup":
        context.pop("design_plan")
    elif change == "other_target":
        context["completion"]["next_target"] = "verification_bindings"
    elif change == "construction":
        context["completion"]["next_work_category"] = "construction"
    elif change == "unknown":
        context["completion"]["functional_coverage"]["missing_requirement_ids"] = ["unknown"]
    else:
        context["edit_basis"]["plan_version"] = -1
    assert amendment(context) is None


def test_existing_obligations_are_visible_and_schema_is_original_subset(tmp_path):
    _, _, context = case(tmp_path)
    retained = {"requirement_id": "fit", "unsupported_checks": ["Retain this original limit"]}
    context["design_plan"]["requirement_dispositions"] = [retained]
    route = amendment(context)
    assert route["retained_requirement_dispositions"] == [retained]
    full = PlanPatch.model_json_schema()
    for name, schema in route["schema"]["properties"].items():
        assert schema == full["properties"][name]
    assert DesignPlan.model_validate(context["design_plan"])


def test_shared_agent_call_exposes_amendment_and_uses_normal_plan_admission(tmp_path, monkeypatch):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace import completion, native_delivery
    from forge_mind.workspace.native_authoring import NativeAction

    service, sid, context = case(tmp_path)
    calls = []
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    # This controlled completion observation isolates dispatch from CAD workers.
    monkeypatch.setattr(completion, "completion_context", lambda *args: context["completion"])
    monkeypatch.setattr(completion, "finalize_completion", lambda *args: None)
    monkeypatch.setattr(native_delivery, "capture_native_delivery", lambda *args: {})

    def forbidden(*args, **kwargs):
        pytest.fail("Intent amendment must not build or replay CAD")

    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "process_observation", forbidden)

    class Author:
        model = "scripted-amendment"
        timeout_s = None

        def generate_structured(self, **arguments):
            import json

            observed = json.loads(arguments["user"])
            calls.append(observed)
            route = observed["completion_amendment"]
            patch = authored_patch(
                route,
                requirement_dispositions=[
                    {
                        "requirement_id": "fit",
                        "physical_validation": ["Physical fit remains unknown"],
                    }
                ],
            )
            return NativeAction(
                kind="plan",
                query="patch",
                document_json=patch.model_dump_json(exclude_unset=True),
                explanation="Declare the actual physical validation obligation",
            ), LLMUsage(self.model, 1, 1)

    service.client_factory = lambda **kwargs: Author()
    jid = service.store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": None})
    job = service.store.claim()
    assert job["id"] == jid
    service.run_job(job)
    saved = service.store.get(sid, "owner")
    assert len(calls) == 1
    errors = [row["message"] for row in saved["events"] if row["kind"] in {"error", "tool_error"}]
    if errors:
        pytest.fail("\n".join(errors))
    assert saved["plan_version"] == context["edit_basis"]["plan_version"] + 1, saved["events"]
    assert saved["design_plan"]["requirement_dispositions"][0]["physical_validation"] == [
        "Physical fit remains unknown"
    ]
    assert service.store.revisions(sid) == []
