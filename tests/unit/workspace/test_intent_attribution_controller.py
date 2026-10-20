"""Scripted intent pauses use real Store authority without CAD or a provider."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import (
    author_images,
    completion,
    completion_inline,
    document_service,
    native_delivery,
)
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.edit_service import read_edit_basis
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.intent_attribution import (
    ContractDraft,
    IntentAnswer,
    IntentAttribution,
    IntentIssue,
    IntentResolution,
)
from forge_mind.workspace.intent_attribution_service import record_conflict
from forge_mind.workspace.models import Action, Contract, ContractEdit, DesignPlan, Requirement
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.plan_draft import summary as draft_summary
from forge_mind.workspace.plan_intent import PlanIntent, lower_intent
from forge_mind.workspace.source_delivery_contract_draft import SourceDeliveryContractDraft
from forge_mind.workspace.store import Store

PROMPT = "Make a bracket for the sensor. Its body is 45 mm wide."
MODEL = "scripted-intent-controller"


def target():
    return Contract(
        requirements=[
            Requirement(id="fit", text="Fit the sensor", source_quote="for the sensor"),
            Requirement(
                id="width",
                text="Retain the requested body width",
                source_quote="Its body is 45 mm wide",
            ),
        ]
    )


def issue(*, with_fact=True):
    return IntentIssue(
        id="width_subject",
        question="Does the 45 mm width describe the sensor or the bracket?",
        reason="The subject changes the fit geometry.",
        fact_ids=["body_width"] if with_fact else [],
        requirement_ids=["width"],
    )


def draft(*, ambiguous=False):
    attribution = IntentAttribution.model_validate(
        {
            "subjects": [
                {
                    "id": "bracket",
                    "label": "Bracket to make",
                    "role": "designed",
                    "source_quote": "bracket",
                },
                {
                    "id": "sensor",
                    "label": "Existing sensor",
                    "role": "reference",
                    "source_quote": "sensor",
                },
            ],
            "facts": [
                {
                    "id": "body_width",
                    "subject_id": None if ambiguous else "sensor",
                    "candidate_subject_ids": ["bracket", "sensor"] if ambiguous else [],
                    "property": "body width",
                    "value": 45,
                    "unit": "mm",
                    "source_quote": "Its body is 45 mm wide",
                    "requirement_ids": ["width"],
                }
            ],
            "requirement_subjects": {
                "fit": ["bracket", "sensor"],
                "width": ["bracket", "sensor"],
            },
            "issues": [issue().model_dump()] if ambiguous else [],
        }
    )
    return ContractDraft(contract=target(), attribution=attribution)


def startup(*, issues=()):
    return PlanIntent.model_validate(
        {
            "objective": "Make the requested bracket for the existing sensor",
            "parts": [
                {
                    "id": "bracket",
                    "role": "make",
                    "purpose": "Support the sensor",
                    "requirement_ids": ["fit", "width"],
                },
                {"id": "sensor", "role": "reference", "purpose": "Retain the existing object"},
            ],
            "first_part_id": "bracket",
            "attribution_issues": [item.model_dump() for item in issues],
        }
    )


def controller_case(tmp_path, monkeypatch, *, supplied_contract=None, preview=False):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    store = Store(tmp_path)
    service = WorkspaceService(store, Executor("trusted_local"))
    sid = store.create("owner", PROMPT, supplied_contract, authoring_mode="native_operations")["id"]
    case = SimpleNamespace(
        service=service,
        store=store,
        sid=sid,
        preview=preview,
        calls=[],
        finalize=[],
        capture=[],
        checkpoints=[],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Intent controller tests must not start CAD, workers or provider services")

    monkeypatch.setattr(service, "start", forbidden)
    monkeypatch.setattr(service, "build", forbidden)
    monkeypatch.setattr(service, "_load_revision", forbidden)
    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbidden)
    monkeypatch.setattr(document_service, "build_document", forbidden)
    monkeypatch.setattr(document_service, "protection_results", forbidden)
    monkeypatch.setattr(
        author_images,
        "prepare_author_images",
        lambda *args, **kwargs: author_images.AuthorImages((), (), None, "scripted", "none"),
    )

    def checkpoint(*args, **kwargs):
        case.checkpoints.append(True)
        return {"status": "unverified", "reason": "No CAD or probes in this control"}

    def finalize(*args, **kwargs):
        case.finalize.append(True)

    def capture(*args, **kwargs):
        case.capture.append(True)
        return {"status": "not_requested", "scope": "No delivery worker in this control"}

    monkeypatch.setattr(completion_inline, "author_checkpoint", checkpoint)
    monkeypatch.setattr(completion, "finalize_completion", finalize)
    monkeypatch.setattr(native_delivery, "capture_native_delivery", capture)
    return case


def run_script(case, responses, *, max_actions=1, instruction=None, queued_job_id=None):
    pending = iter(responses)

    class Client:
        model = MODEL
        timeout_s = None

        def generate_structured(self, **kwargs):
            output_model = kwargs["output_model"]
            context = (
                None
                if output_model
                in {SourceDeliveryContractDraft, ContractDraft, ContractEdit, Contract}
                else json.loads(kwargs["user"])
            )
            case.calls.append(
                {
                    "schema": output_model,
                    "context": context,
                    "user": kwargs["user"],
                    "system": kwargs["system"],
                }
            )
            response = next(pending)
            if callable(response):
                response = response(case, context)
            payload = (
                source_delivery_wire_fixture(response)
                if output_model is SourceDeliveryContractDraft
                else response.model_dump()
            )
            return output_model.model_validate(payload), LLMUsage(
                MODEL, 100 + len(case.calls), 10 + len(case.calls)
            )

        generate_structured_with_image = generate_structured

    case.service.client_factory = lambda **kwargs: Client()
    payload = {
        "max_actions": max_actions,
        "max_seconds": 900,
        "preview_only": case.preview,
        "delivery_requested": True,
    }
    if instruction is not None:
        payload["instruction"] = instruction
    case.jid = queued_job_id or case.store.enqueue(case.sid, "owner", payload)
    case.job = case.store.claim()
    assert case.job is not None and case.job["id"] == case.jid
    case.service.run_job(case.job)
    return case.store.get(case.sid, "owner")


def assert_paused(case, session, *, calls):
    assert session["state"] == "needs_input"
    assert session["intent_attribution"]["status"] == "needs_input"
    assert session["intent_attribution"]["attribution"]["issues"][0]["id"] == "width_subject"
    assert len(case.calls) == calls
    assert [(item["input_tokens"], item["output_tokens"]) for item in session["usage"]] == [
        (101 + index, 11 + index) for index in range(calls)
    ]
    assert session["author_model_reported"] == MODEL
    assert not session.get("design_plan")
    assert not session["revisions"]
    assert session.get("selected_revision") is None
    assert session.get("accepted_revision") is None
    assert not case.finalize and not case.capture
    assert session["automatic_delivery"]["status"] == "incomplete"
    assert session["automatic_delivery"]["receipt"] is None
    assert session["completion_progress"]["phase"] == "finished"
    with case.store.connect() as connection:
        jobs = connection.execute("SELECT id,state FROM jobs").fetchall()
        assert len(jobs) == 1 and tuple(jobs[0]) == (case.jid, "needs_input")


@pytest.mark.parametrize("preview", [False, True])
def test_material_draft_pauses_before_canonical_plan_or_cad_and_retains_usage(
    tmp_path, monkeypatch, preview
):
    case = controller_case(tmp_path, monkeypatch, preview=preview)
    session = run_script(case, [draft(ambiguous=True), startup()])
    assert_paused(case, session, calls=1)
    assert [call["schema"] for call in case.calls] == [SourceDeliveryContractDraft]
    assert session["contract"] is None
    assert not case.checkpoints
    assert session["intent_attribution"]["proposed_contract"] == target().model_dump()
    question_events = [event for event in session["events"] if event["kind"] == "intent_question"]
    assert len(question_events) == 1
    assert (
        question_events[0]["intent_attribution"]["draft_hash"]
        == session["intent_attribution"]["draft_hash"]
    )
    assert session["intent_attribution"]["resolutions"] == []


def test_clear_draft_admits_canonical_contract_and_plan_together_after_startup(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    admitted = []
    original_admit = case.store.set_design_plan_for_job
    original_update = case.store.update

    def update(sid, **values):
        if values.get("contract") and not case.store.get(sid, "owner").get("design_plan"):
            pytest.fail("Canonical contract changed before atomic startup admission")
        return original_update(sid, **values)

    def admit(*args, **kwargs):
        before = case.store.get(case.sid, "owner")
        assert before["contract"] is None and not before.get("design_plan")
        result = original_admit(*args, **kwargs)
        after = case.store.get(case.sid, "owner")
        assert after["contract"] == target().model_dump()
        assert after["design_plan"] == lower_intent(startup(), target()).model_dump()
        admitted.append(result)
        return result

    def reviewed_startup(case, context):
        assert context["frozen_contract"] is None
        assert context["proposed_contract"] == target().model_dump()
        assert context["request_attribution"]["status"] == "interpreted"
        assert case.store.get(case.sid, "owner")["contract"] is None
        return startup()

    monkeypatch.setattr(case.store, "update", update)
    monkeypatch.setattr(case.store, "set_design_plan_for_job", admit)
    session = run_script(case, [draft(), reviewed_startup])
    assert [call["schema"] for call in case.calls] == [SourceDeliveryContractDraft, PlanIntent]
    assert len(admitted) == 1 and session["plan_version"] == 1
    assert Contract.model_validate(session["contract"]).fingerprint() == target().fingerprint()
    assert session["intent_attribution"]["status"] == "interpreted"
    assert session["intent_attribution"]["resolutions"] == []
    assert not session["revisions"] and session.get("accepted_revision") is None


@pytest.mark.parametrize("preview", [False, True])
def test_startup_conflict_preserves_supplied_legacy_contract(tmp_path, monkeypatch, preview):
    canonical = target().model_dump()
    fingerprint = target().fingerprint()
    case = controller_case(
        tmp_path, monkeypatch, supplied_contract=deepcopy(canonical), preview=preview
    )
    session = run_script(case, [startup(issues=[issue(with_fact=False)])])
    assert_paused(case, session, calls=1)
    assert [call["schema"] for call in case.calls] == [PlanIntent]
    assert session["contract"] == canonical
    assert Contract.model_validate(session["contract"]).fingerprint() == fingerprint
    assert case.calls[0]["context"]["frozen_contract"] == canonical
    assert not session["intent_attribution"]["attribution"]["facts"]


def test_startup_can_report_material_issue_before_admitting_a_clear_extraction(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    session = run_script(case, [draft(), startup(issues=[issue()])])
    assert_paused(case, session, calls=2)
    assert session["contract"] is None
    assert session["intent_attribution"]["attribution"]["facts"][0]["id"] == "body_width"
    assert session["intent_attribution_history"][0]["status"] == "interpreted"


@pytest.mark.parametrize("preview", [False, True])
def test_clear_model_response_cannot_erase_an_issue_recorded_during_inference(
    tmp_path, monkeypatch, preview
):
    case = controller_case(tmp_path, monkeypatch, preview=preview)
    questions = []

    def conflict_then_clear(case, context):
        questions.append(
            record_conflict(
                case.store,
                case.sid,
                "owner",
                case.jid,
                [issue()],
                target(),
                PROMPT,
                context["edit_basis"],
            )
        )
        return startup()

    session = run_script(case, [draft(), conflict_then_clear])
    assert_paused(case, session, calls=2)
    assert session["contract"] is None
    assert session["intent_attribution"] == questions[0]
    assert session["intent_attribution"]["resolutions"] == []


def test_question_during_extraction_retains_instruction_source_and_accepts_bound_owner_answer(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    instruction = "Retain the sensor's button access."
    current_source = PROMPT + "\nUser revision instruction: " + instruction
    questions = []

    def conflict_then_clear(case, context):
        assert context is None
        assert case.calls[0]["user"] == current_source
        assert case.store.get(case.sid, "owner")["instructions"] == [instruction]
        questions.append(
            record_conflict(
                case.store,
                case.sid,
                "owner",
                case.jid,
                [issue(with_fact=False)],
                target(),
                current_source,
                read_edit_basis(case.service, case.sid, "owner"),
            )
        )
        return draft()

    session = run_script(case, [conflict_then_clear], instruction=instruction)
    assert_paused(case, session, calls=1)
    assert session["contract"] is None
    assert session["intent_attribution"] == questions[0]
    assert session["intent_attribution"]["source_text"] == current_source
    resolution = IntentResolution(
        expected_version=questions[0]["version"],
        draft_hash=questions[0]["draft_hash"],
        answers=[IntentAnswer(issue_id="width_subject", answer="The width describes the sensor.")],
    )
    next_job = case.store.enqueue(
        case.sid,
        "owner",
        {"intent_resolution": resolution.model_dump(), "max_actions": 1, "max_seconds": 900},
    )
    answered = case.store.get(case.sid, "owner")
    assert answered["job_id"] == next_job
    assert answered["intent_attribution"]["status"] == "answered"
    assert answered["intent_attribution"]["resolutions"] == resolution.model_dump()["answers"]
    assert answered["contract"] is None and not answered["revisions"]


def test_queued_owner_answer_reconstructs_full_fresh_contract_before_atomic_startup(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    question = "Does 45 mm describe the existing sensor body width? Untrusted model note: 999 mm."
    assert "999" not in PROMPT
    first_draft = draft(ambiguous=True)
    first_draft.attribution.issues[0].question = question
    paused = run_script(case, [first_draft])
    assert_paused(case, paused, calls=1)
    record = paused["intent_attribution"]
    raw_answer = "yes"
    resolution = IntentResolution(
        expected_version=record["version"],
        draft_hash=record["draft_hash"],
        answers=[IntentAnswer(issue_id="width_subject", answer=raw_answer)],
    )
    next_job = case.store.enqueue(
        case.sid,
        "owner",
        {"intent_resolution": resolution.model_dump(), "max_actions": 1, "max_seconds": 900},
    )
    queued = case.store.get(case.sid, "owner")
    assert queued["contract"] is None and queued.get("design_plan") is None
    assert queued["intent_attribution"]["status"] == "answered"
    assert queued["intent_attribution"]["resolutions"] == resolution.model_dump()["answers"]
    with case.store.connect() as connection:
        payload = json.loads(
            connection.execute("SELECT payload FROM jobs WHERE id=?", (next_job,)).fetchone()[0]
        )
    owner_instruction = "User clarification (width_subject): yes"
    assert payload["instruction"] == owner_instruction
    owner_source = PROMPT + "\nUser revision instruction: " + owner_instruction
    assert question not in owner_source
    before_calls = len(case.calls)
    admissions = []
    original_admit = case.store.set_design_plan_for_job

    def full_draft(case, context):
        assert context is None
        request = case.calls[-1]
        assert request["schema"] is SourceDeliveryContractDraft
        assert PROMPT in request["user"] and owner_instruction in request["user"]
        assert question in request["user"] and "width_subject" in request["user"]
        assert "return ONLY requirements explicitly changed" not in request["system"]
        assert "full" in request["system"].lower()
        assert case.store.get(case.sid, "owner")["contract"] is None
        return draft()

    def reviewed_startup(case, context):
        assert context["frozen_contract"] is None
        assert {item["id"] for item in context["proposed_contract"]["requirements"]} == {
            "fit",
            "width",
        }
        interpretation = context["request_attribution"]
        assert interpretation["status"] == "interpreted"
        assert interpretation["source_text"] == owner_source
        assert question not in interpretation["source_text"]
        assert "999" not in interpretation["source_text"]
        return startup()

    def admit(*args, **kwargs):
        before = case.store.get(case.sid, "owner")
        assert before["contract"] is None and not before.get("design_plan")
        result = original_admit(*args, **kwargs)
        after = case.store.get(case.sid, "owner")
        assert after["contract"] == target().model_dump()
        assert after["design_plan"] == lower_intent(startup(), target()).model_dump()
        admissions.append(result)
        return result

    monkeypatch.setattr(case.store, "set_design_plan_for_job", admit)
    session = run_script(case, [full_draft, reviewed_startup], queued_job_id=next_job)
    assert [call["schema"] for call in case.calls[before_calls:]] == [
        SourceDeliveryContractDraft,
        PlanIntent,
    ]
    assert len(admissions) == 1 and session["plan_version"] == 1
    assert {item["id"] for item in session["contract"]["requirements"]} == {"fit", "width"}
    assert session["contract"] == target().model_dump()
    assert session["instructions"] == [owner_instruction]
    assert session["intent_attribution"]["source_text"] == owner_source
    assert "999" not in session["intent_attribution"]["source_text"]
    assert all(item["value"] != 999 for item in session["contract"]["requirements"])
    width = next(item for item in session["contract"]["requirements"] if item["id"] == "width")
    assert width["source_quote"] == "Its body is 45 mm wide"
    assert session["intent_attribution"]["attribution"]["facts"][0]["value"] == 45
    answered_history = [
        item for item in session["intent_attribution_history"] if item["status"] == "answered"
    ]
    assert len(answered_history) == 1
    assert answered_history[0]["resolutions"] == resolution.model_dump()["answers"]
    assert answered_history[0]["attribution"]["issues"][0]["question"] == question
    assert not session["revisions"] and session.get("accepted_revision") is None


def test_supplied_canonical_contract_keeps_revision_delta_extraction(tmp_path, monkeypatch):
    canonical = target().model_dump()
    case = controller_case(tmp_path, monkeypatch, supplied_contract=deepcopy(canonical))
    instruction = "Keep the sensor body width unchanged."

    def delta(case, context):
        assert context is None
        request = case.calls[-1]
        assert request["schema"] is ContractEdit
        assert "return ONLY requirements explicitly changed" in request["system"]
        assert instruction in request["user"]
        return ContractEdit(
            requirements=[
                Requirement(
                    id="width",
                    text="Keep the sensor body width unchanged",
                    source_quote=instruction,
                )
            ]
        )

    session = run_script(case, [delta, startup()], instruction=instruction)
    assert [call["schema"] for call in case.calls] == [ContractEdit, PlanIntent]
    requirements = {item["id"]: item for item in session["contract"]["requirements"]}
    assert set(requirements) == {"fit", "width"}
    assert requirements["fit"] == canonical["requirements"][0]
    assert requirements["width"]["source_quote"] == instruction
    assert session.get("intent_attribution") is None
    assert not session["revisions"] and session.get("accepted_revision") is None


def test_rejected_startup_draft_repair_commits_host_proposed_contract_without_cad(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    invalid = startup().model_copy(deep=True)
    invalid.parts[0].requirement_ids = ["unknown"]
    expected = lower_intent(startup(), target())
    assert expected.operation_plan is not None

    def fallback(case, context):
        session = case.store.get(case.sid, "owner")
        assert session["contract"] is None and not session.get("design_plan")
        assert context["failed_submission"]["startup_intent_rejected"] is True
        assert context["plan_draft"] is None
        invalid_plan = expected.model_copy(deep=True)
        assert invalid_plan.operation_plan is not None
        invalid_plan.operation_plan.steps[0].requirement_ids = ["unknown"]
        return Action(
            kind="plan",
            explanation="Prepare a full canonical fallback plan",
            document_json=invalid_plan.model_dump_json(),
        )

    def repair(case, context):
        session = case.store.get(case.sid, "owner")
        assert session["contract"] is None and not session.get("design_plan")
        saved = context["plan_draft"]
        assert saved["status"] == "unadmitted_draft" and saved["cad_applied"] is False
        return Action(
            kind="plan",
            query="draft_patch",
            explanation="Repair the retained canonical fallback plan",
            document_json=json.dumps(
                {
                    "basis": context["edit_basis"],
                    "draft_version": saved["draft_version"],
                    "draft_hash": saved["draft_hash"],
                    "changes": {
                        "operation_steps": [
                            item.model_dump() for item in expected.operation_plan.steps
                        ],
                    },
                }
            ),
        )

    def commit(case, context):
        session = case.store.get(case.sid, "owner")
        assert session["contract"] is None and not session.get("design_plan")
        saved = context["plan_draft"]
        assert saved["draft_version"] == 2 and saved["cad_applied"] is False
        return Action(
            kind="plan",
            query="draft_commit",
            explanation="Explicitly admit the repaired plan against the host proposal",
            document_json=json.dumps(
                {
                    "basis": context["edit_basis"],
                    "draft_version": saved["draft_version"],
                    "draft_hash": saved["draft_hash"],
                }
            ),
        )

    session = run_script(case, [draft(), invalid, fallback, repair, commit], max_actions=4)
    assert [call["schema"] for call in case.calls] == [
        SourceDeliveryContractDraft,
        PlanIntent,
        NativeAction,
        NativeAction,
        NativeAction,
    ]
    assert session["contract"] == target().model_dump()
    assert Contract.model_validate(session["contract"]).fingerprint() == target().fingerprint()
    admitted = DesignPlan.model_validate(session["design_plan"])
    assert admitted.parts == expected.parts
    assert admitted.subsystems == expected.subsystems
    assert admitted.operation_plan is not None
    assert admitted.operation_plan.steps == expected.operation_plan.steps
    assert session["plan_version"] == 1 and session["failed_submission"] is None
    assert draft_summary(case.store, case.sid, "owner") is None
    assert session["intent_attribution"]["status"] == "interpreted"
    assert not session["revisions"] and session.get("accepted_revision") is None
