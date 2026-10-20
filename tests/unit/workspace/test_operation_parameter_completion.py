"""Missing planned inputs must remain actionable through final review."""

import pytest
from tests.unit.workspace.test_operation_plan import contract, document, plan

from forge_mind.workspace.completion_work import remaining_work
from forge_mind.workspace.models import OperationStep
from forge_mind.workspace.operation_plan import coverage, snapshot


@pytest.mark.parametrize("binding", ["input", "relation", "missing"])
def test_independent_parameter_step_is_required_without_changing_cad_approval(binding):
    intent = plan()
    intent.operation_plan.steps.append(
        OperationStep(
            id="edit_width",
            part_id="main",
            kind="parameters",
            binding_names=["editable_width"],
            expected_effect="Retain an editable width input",
        )
    )
    design = document()
    if binding == "input":
        design.variables["editable_width"] = 20
    elif binding == "relation":
        from forge_mind.workspace.document import VariableRelation

        design.variables["half_width"] = 10
        design.relations.append(
            VariableRelation(name="editable_width", operation="multiply", a="half_width", b=2)
        )
    observed = coverage(snapshot(intent, 1, contract()), design)
    assert observed["graph_complete"] and observed["references_complete"]
    work = remaining_work([], [], observed["steps"])
    assert work["next_target"] == ("edit_width" if binding == "missing" else None)
    assert observed["parameters_complete"] is (binding != "missing")
    if binding == "missing":
        assert work["next_work_category"] == "parameters"
    assert all(row["effect_verified"] is False for row in observed["steps"])


def test_plans_without_parameter_steps_remain_complete():
    observed = coverage(snapshot(plan(), 1, contract()), document())
    assert observed["parameters_complete"] is True
    assert remaining_work([], [], observed["steps"])["next_target"] is None


def test_parameter_dependency_is_not_hidden_by_present_downstream_binding():
    intent = plan()
    intent.operation_plan.steps.extend(
        [
            OperationStep(
                id="missing_input",
                part_id="main",
                kind="parameters",
                binding_names=["absent"],
                expected_effect="Retain a named input",
            ),
            OperationStep(
                id="dependent",
                part_id="main",
                kind="parameters",
                depends_on=["missing_input"],
                binding_names=["present"],
                expected_effect="Retain a dependent input",
            ),
        ]
    )
    design = document()
    design.variables["present"] = 20
    observed = coverage(snapshot(intent, 1, contract()), design)
    assert observed["parameters_complete"] is False
    assert observed["steps"][-1]["missing_dependency_steps"] == ["missing_input"]
    assert remaining_work([], [], observed["steps"])["next_target"] == "missing_input"


def test_current_guidance_cannot_finish_without_a_planned_parameter(tmp_path):
    import json

    from tests.unit.workspace.test_native_process_completion import baseline, guide_payload

    from forge_mind.workspace.delivery_guidance import (
        GuidanceUpdate,
        guidance_context,
        update_guidance,
    )
    from forge_mind.workspace.document_service import build_document
    from forge_mind.workspace.models import Contract, DesignPlan
    from forge_mind.workspace.native_finish import finish_context

    service, sid, design, previous = baseline(tmp_path)
    target = DesignPlan.model_validate(service.store.get(sid, "owner")["design_plan"])
    target.operation_plan.steps.append(
        OperationStep(
            id="editable_input",
            part_id="mount",
            kind="parameters",
            binding_names=["future_input"],
            expected_effect="Retain an editable input",
        )
    )
    service.store.set_design_plan(sid, "owner", target, expected_plan_version=1)
    revision = build_document(
        service,
        sid,
        "owner",
        design,
        Contract.model_validate(previous["contract"]),
        previous["id"],
        "Capture planned inputs",
        60,
        lambda: False,
    )
    service.store.update(sid, selected_revision=revision["id"])
    payload = json.loads(
        guide_payload({"delivery_guidance": guidance_context(service, sid, "owner")})
    )
    payload["guidance"]["installation"] = (
        "Check the 40 mm mount width against the installation space."
    )
    update_guidance(service, sid, "owner", GuidanceUpdate.model_validate(payload))
    status = finish_context(service, sid, "owner", revision["id"])
    assert status["status"] == "needs_work"
    assert any("parameter" in reason.lower() for reason in status["reasons"])
    assert service.store.get(sid, "owner").get("accepted_revision") is None

    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.native_authoring import NativeAction

    class Author:
        model = "scripted-incomplete-parameter"

        def generate_structured(self, **kwargs):
            return NativeAction(
                kind="guidance",
                query="finish",
                document_json=json.dumps(payload),
                explanation="Explicitly review the saved instructions",
            ), LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service.client_factory = lambda **kw: Author()
    service.store.enqueue(
        sid, "owner", {"max_actions": 1, "max_seconds": 120, "delivery_requested": True}
    )
    service.run_job(service.store.claim())
    result = service.store.get(sid, "owner")
    assert result["state"] == "needs_review"
    assert result["resume_candidate"]["revision_id"] == revision["id"]
    assert result.get("accepted_revision") is None
