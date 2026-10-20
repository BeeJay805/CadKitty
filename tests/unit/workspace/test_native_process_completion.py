"""Real native edit, plan reconciliation and explicit review closure."""

import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_operation_authoring_flow import contract, document, plan

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.author_guidance import INSTRUCTION_REVIEW
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.input_application import baseline as input_baseline
from forge_mind.workspace.models import DesignPlan
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.native_finish import finish_context
from forge_mind.workspace.product_models import PartProduction
from forge_mind.workspace.project_delivery import read_project_export
from forge_mind.workspace.store import Store, WorkspaceError


def baseline(tmp_path):
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    sid = store.create(
        "owner",
        "Change to 44 mm width; provide files",
        contract().model_dump(),
        authoring_mode="native_operations",
    )["id"]
    raw = plan().model_dump()
    raw["operation_plan"]["dimensions"] = [
        {
            "id": "width_source",
            "binding_name": "width",
            "value": 40,
            "source": "customer",
            "source_quote": "40 mm width",
            "part_id": "mount",
        }
    ]
    store.set_design_plan(
        sid, "owner", DesignPlan.model_validate(raw), expected_plan_version=0, contract=contract()
    )
    design = document().model_copy(update={"production": PartProduction(role="make")})
    revision = build_document(
        service, sid, "owner", design, contract(), None, "Saved baseline", 60, lambda: False
    )
    store.update(sid, selected_revision=revision["id"])
    return service, sid, design, revision


def guide_payload(context):
    guide = context["delivery_guidance"]
    return json.dumps(
        {
            "revision_id": guide.get("revision_id") or guide["basis"]["revision_id"],
            "basis_hash": guide["basis_hash"],
            "reviewed_current_facts": True,
            "guidance": {
                "print": "Check the printer profile before printing.",
                "installation": "The mount width is 44 mm. Check physical fit.",
                "use_and_removal": "Keep removal access clear.",
                "unresolved": "Physical fit and strength remain unverified.",
            },
        }
    )


@pytest.mark.parametrize("retrieve_patch_schema", [False, True])
def test_plan_edit_and_guidance_finish_skip_redundant_final_model_call(
    tmp_path, retrieve_patch_schema
):
    service, sid, design, previous = baseline(tmp_path)
    calls = []

    class Author:
        model = "scripted-native-process"

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            calls.append(context)
            index = len(calls) - (2 if retrieve_patch_schema else 0)
            if retrieve_patch_schema and len(calls) == 1:
                action = NativeAction(kind="schema", query="guidance", explanation="Review syntax")
            elif retrieve_patch_schema and len(calls) == 2:
                assert context["review_context"]["is_summary"]
                assert kwargs["system"] == INSTRUCTION_REVIEW
                assert kwargs["output_model"] is NativeAction
                action = NativeAction(
                    kind="schema", query="plan_reconciliation", explanation="Patch syntax"
                )
            elif index == 1:
                if retrieve_patch_schema:
                    assert "plan_reconciliation" in context["authoritative_schemas"]
                patch = {
                    "expected_plan_version": context["plan_reconciliation"][
                        "expected_plan_version"
                    ],
                    "revision_id": previous["id"],
                    "updates": [
                        {
                            "id": "width_edit",
                            "value": 44,
                            "source": "customer",
                            "source_quote": "44 mm width",
                            "part_id": "mount",
                            "binding_name": "width",
                            "supersedes_fact_id": "width_source",
                        }
                    ],
                }
                action = NativeAction(
                    kind="plan",
                    query="reconcile",
                    document_json=json.dumps(patch),
                    explanation="Update intent",
                )
            elif index == 2:
                edited = design.model_copy(update={"variables": {"width": 44}})
                action = NativeAction(
                    kind="document",
                    document_json=edited.model_dump_json(),
                    explanation="Build edit",
                )
            elif index == 3:
                action = NativeAction(kind="schema", query="guidance", explanation="Review syntax")
            elif index == 4:
                assert context["review_context"]["is_summary"]
                assert kwargs["system"] == INSTRUCTION_REVIEW
                assert kwargs["output_model"] is NativeAction
                assert context["plan_reconciliation"]["status"] == "aligned"
                action = NativeAction(
                    kind="guidance",
                    query="finish",
                    document_json=guide_payload(context),
                    explanation="Finish reviewed instructions",
                )
            else:
                pytest.fail("The explicit instruction review must not ask for another finish call")
            return action, LLMUsage(self.model, 0, 0)

        generate_structured_with_image = generate_structured

    service.client_factory = lambda **kw: Author()
    service.store.enqueue(
        sid, "owner", {"max_actions": 8, "max_seconds": 120, "delivery_requested": True}
    )
    job = service.store.claim()
    service.run_job(job)
    result = service.store.get(sid, "owner")
    assert len(calls) == (6 if retrieve_patch_schema else 4), json.dumps(result["events"])
    assert result["native_review"]["status"] == "ready_for_validation"
    assert result["state"] == "contract_satisfied"
    assert result.get("accepted_revision") is None
    assert result["automatic_delivery"]["status"] == "delivered"
    assert result["automatic_delivery"]["guidance_status"] == "current"
    current = service.store.revision(sid, result["selected_revision"])
    facts = current["operation_record"]["plan_snapshot"]["plan"]["operation_plan"]["dimensions"]
    assert facts[0]["value"] == 40 and facts[1]["value"] == 44
    assert facts[1]["supersedes_fact_id"] == facts[0]["id"]
    assert result["plan_hash"] == current["operation_record"]["plan_hash"]
    receipt = result["automatic_delivery"]["receipt"]
    content, readback = read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    assert readback == receipt
    with ZipFile(BytesIO(content)) as archive:
        readme = archive.read("README.md").decode()
        guide = json.loads(archive.read("customer-guidance.json"))
        assert "44 mm" in guide["record"]["guidance"]["installation"]
        assert "40 mm width" not in readme
        assert guide["facts"]["measured"]["extents_mm"][0] == pytest.approx(44)
        assert guide["instruction_binding"]["status"] == "legacy_unverified"
        assert "Instruction binding status: legacy_unverified." in readme
        assert "[customer-guidance.json](customer-guidance.json)" in readme
        assert "Physical fit" in readme
    assert current["contract"] == service.store.revision(sid, previous["id"])["contract"]


@pytest.mark.parametrize("defect", ["dimension_drift", "uncaptured_plan", "stale_guidance"])
def test_fresh_finish_checks_refuse_inconsistent_inputs(tmp_path, defect):
    from forge_mind.workspace.delivery_guidance import (
        GuidanceUpdate,
        guidance_context,
        update_guidance,
    )

    service, sid, _, previous = baseline(tmp_path)
    context = {"delivery_guidance": guidance_context(service, sid, "owner")}
    update_guidance(
        service, sid, "owner", GuidanceUpdate.model_validate_json(guide_payload(context))
    )
    if defect == "dimension_drift":
        raw = service.store.get(sid, "owner")["design_plan"]
        raw["operation_plan"]["dimensions"].append(
            {
                "id": "edit",
                "binding_name": "width",
                "value": 44,
                "source": "assumption",
                "source_quote": "Planned width edit",
                "part_id": "mount",
                "supersedes_fact_id": "width_source",
            }
        )
        service.store.set_design_plan(
            sid, "owner", DesignPlan.model_validate(raw), expected_plan_version=1
        )
    elif defect == "uncaptured_plan":
        raw = service.store.get(sid, "owner")["design_plan"]
        raw["objective"] = "Updated plan still requires an immutable candidate binding"
        service.store.set_design_plan(
            sid, "owner", DesignPlan.model_validate(raw), expected_plan_version=1
        )
    else:
        service.store.update(sid, delivery_guidance_record=None, delivery_guidance_records={})
    result = finish_context(service, sid, "owner", previous["id"])
    assert result["status"] == "needs_work"
    assert result["reasons"]
    assert service.store.get(sid, "owner").get("accepted_revision") is None


@pytest.mark.parametrize("edit", ["unmeasured_goal", "instruction_provenance"])
def test_finish_cannot_review_an_old_contract_after_active_intent_changes(tmp_path, edit):
    from forge_mind.workspace.delivery_guidance import (
        GuidanceUpdate,
        guidance_context,
        update_guidance,
    )

    service, sid, _, previous = baseline(tmp_path)
    changed = service.store.get(sid, "owner")["contract"]
    if edit == "unmeasured_goal":
        changed["requirements"].append(
            {
                "id": "release",
                "text": "Release without CAD knowledge",
                "source_quote": "Simple release",
                "metric": "unmeasured",
            }
        )
    else:
        changed["requirements"][0]["text"] = "Provide installation tools and cleanup for this bore"
        changed["requirements"][0]["source_quote"] = "Installation tools and cleanup"
    service.store.update(sid, contract=changed)
    context = {"delivery_guidance": guidance_context(service, sid, "owner")}
    assert context["delivery_guidance"]["input_authority"]["status"] == "blocked"
    before = service.store.session_record(sid, "owner")
    with pytest.raises(WorkspaceError, match="Owner input"):
        update_guidance(
            service, sid, "owner", GuidanceUpdate.model_validate_json(guide_payload(context))
        )
    assert service.store.session_record(sid, "owner") == before
    result = finish_context(service, sid, "owner", previous["id"])
    assert result["status"] == "needs_work"
    assert any("active requirements" in reason for reason in result["reasons"])


@pytest.mark.parametrize("candidate", ["proposal", "resume_candidate"])
def test_finish_uses_effective_candidate_contract_instead_of_accepted_baseline(tmp_path, candidate):
    from forge_mind.workspace.delivery_guidance import (
        GuidanceUpdate,
        guidance_context,
        update_guidance,
    )

    service, sid, _, revision = baseline(tmp_path)
    accepted_baseline = {"requirements": []}
    # Seed a consumed accepted baseline, not an inconsistent raw Contract edit.
    accepted_source = {**service.store.session_record(sid, "owner"), "contract": accepted_baseline}
    service.store.update(
        sid,
        contract=accepted_baseline,
        contract_application=input_baseline(accepted_source),
        **{candidate: {"revision_id": revision["id"], "contract": revision["contract"]}},
    )
    context = {"delivery_guidance": guidance_context(service, sid, "owner")}
    update_guidance(
        service, sid, "owner", GuidanceUpdate.model_validate_json(guide_payload(context))
    )
    result = finish_context(service, sid, "owner", revision["id"])
    assert not any("active requirements" in reason for reason in result["reasons"])
