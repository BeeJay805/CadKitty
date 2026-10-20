"""Actual startup requests retain existing operation and source authority."""

from __future__ import annotations

import json

import pytest
from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.author_guidance import PRINT_PREPARATION_DETAIL
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.intent_attribution import ContractDraft, IntentAttribution, IntentIssue
from forge_mind.workspace.intent_capabilities import preparation_capability
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.plan_intent import IntentPart, PlanIntent
from forge_mind.workspace.print_setup_models import PrintSetupRequest
from forge_mind.workspace.source_delivery_contract_draft import SourceDeliveryContractDraft
from forge_mind.workspace.store import Store


@pytest.mark.parametrize(
    "duty,question",
    [
        (None, None),
        (
            "It must withstand a 10 kg load.",
            "Where and in which direction is the load applied?",
        ),
        (
            "Print it on my printer.",
            "Which printer does the physical-print requirement refer to?",
        ),
    ],
)
def test_service_startup_receives_exact_capability_without_waiving_source_duties(
    tmp_path, monkeypatch, duty, question
):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    source = "Make a bracket. Deliver a prepared print package."
    requirements = [
        Requirement(id="bracket", text="Make a bracket", source_quote="Make a bracket"),
        Requirement.model_validate(
            {
                "id": "printing",
                "text": "a prepared print package",
                "source_quote": "a prepared print package",
                "verification": {
                    "kind": "digital_delivery",
                    "scope": "digital_artifacts",
                    "coverage": "all_made",
                    "deliverable": "prepared_print_package",
                    "source_quote": "a prepared print package",
                },
            }
        ),
    ]
    issues = []
    if duty is not None:
        source += " " + duty
        requirements.append(Requirement(id="physical", text=duty, source_quote=duty))
        issues.append(
            IntentIssue(
                id="physical_input",
                question=question,
                reason="The explicit physical source duty needs this missing customer input.",
                requirement_ids=["physical"],
            )
        )
    contract = Contract(requirements=requirements)
    draft = ContractDraft(contract=contract, attribution=IntentAttribution())
    planned = PlanIntent(
        objective="Construct the requested bracket and preserve every source duty",
        parts=[
            IntentPart(
                id="bracket",
                role="make",
                purpose="The requested bracket",
                requirement_ids=[row.id for row in requirements],
            )
        ],
        first_part_id="bracket",
        unresolved=[question] if question is not None else [],
        attribution_issues=issues,
    )
    calls = []

    class Scripted:
        model = "scripted-startup-capability"
        timeout_s = None

        def generate_structured(self, **kwargs):
            calls.append(kwargs)
            if kwargs["output_model"] is SourceDeliveryContractDraft:
                wire = source_delivery_wire_fixture(draft, digital_requirement_ids=["printing"])
                response = kwargs["output_model"].model_validate(wire)
            else:
                assert kwargs["output_model"] is PlanIntent
                response = planned
            return response, LLMUsage(self.model, 20, 10)

        def generate_structured_with_image(self, **kwargs):
            pytest.fail("Startup must not launch image inference")

    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    service.client_factory = lambda **kwargs: Scripted()

    def forbidden(*args, **kwargs):
        pytest.fail("This request-boundary control cannot start CAD or a native worker")

    monkeypatch.setattr(service, "build", forbidden)
    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbidden)
    sid = service.store.create("owner", source, None, authoring_mode="native_operations")["id"]
    service.store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": 100})
    job = service.store.claim()
    assert job is not None
    state = service._run_agent_job(job)
    service.store.finish(job["id"], sid, state)

    assert [call["output_model"] for call in calls] == [SourceDeliveryContractDraft, PlanIntent]
    startup = calls[1]
    assert startup["system"].count(PRINT_PREPARATION_DETAIL) == 1
    descriptor_line = next(
        line
        for line in startup["system"].splitlines()
        if line.startswith("Actual prepared-print operation semantics: ")
    )
    descriptor = json.loads(descriptor_line.partition(": ")[2])
    assert descriptor == preparation_capability()
    assert (
        descriptor["required_saved_inputs"]
        == [
            name
            for name, field in PrintSetupRequest.model_fields.items()
            if field.is_required() and name != "operation"
        ]
        == ["revision_id", "part_ids", "artifact_hashes"]
    )
    assert descriptor["required_customer_inputs"] == []
    assert "printer_profile" in descriptor["excluded_outputs"]
    assert "physical_approval" in descriptor["excluded_outputs"]
    context = json.loads(startup["user"])
    assert context["phase"] == "startup_intent"
    assert context["request"] == source
    assert context["proposed_contract"] == contract.model_dump(mode="json")
    assert context["frozen_contract"] is None
    assert "Do not resolve customer uncertainty yourself" in startup["system"]
    assert "Unresolved prose cannot suspend a wrong target" in startup["system"]
    saved = service.store.get(sid, "owner")
    assert not saved["revisions"] and not saved.get("accepted_revision")
    if duty is None:
        assert state != "error"
        assert saved["contract"] == contract.model_dump(mode="json")
        assert saved["design_plan"]["unresolved"] == planned.unresolved
        assert saved["design_plan"]["operation_plan"]["unresolved"] == planned.unresolved
    else:
        assert state == "needs_input"
        assert saved["contract"] is None and not saved.get("design_plan")
        retained = saved["intent_attribution"]
        assert retained["status"] == "needs_input"
        assert retained["proposed_contract"] == contract.model_dump(mode="json")
        assert retained["attribution"]["issues"] == [
            item.model_dump(mode="json") for item in issues
        ]
        physical = retained["proposed_contract"]["requirements"][-1]
        assert physical == requirements[-1].model_dump(mode="json")
        assert physical["critical"] and physical["metric"] == "unmeasured"
        assert physical.get("verification") is None
