"""Protected author handoff with scripted inference and saved bytes, never CAD."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.unit.workspace import test_native_process_completion as process_fixture
from tests.unit.workspace.test_operation_authoring_flow import contract, plan

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import (
    agent,
    author_images,
    completion,
    completion_inline,
    delivery_guidance,
    document_service,
    measure,
    native_delivery,
    native_finish,
    print_setup_service,
)
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.delivery_guidance import guidance_context
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.input_application import baseline as input_baseline
from forge_mind.workspace.instruction_context import ProtectedGuidanceUpdate as GuidanceUpdate
from forge_mind.workspace.models import Finding, Measurement
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.operation_plan import coverage
from forge_mind.workspace.store import Store, WorkspaceError

RID = "a" * 32
ENGINE = "f" * 64


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def cross_boundary(self, context):
        seconds = context.get("phase_seconds_left", context["seconds_left"])
        assert seconds is not None
        self.advance(seconds + 1)


def reviewed_instructions(context):
    payload = json.loads(process_fixture.guide_payload(context))
    catalog = context["delivery_guidance"]["instruction_catalog"]
    payload["instruction_plan"] = {
        "version": 1,
        "fact_refs": catalog["required_refs"],
        "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
    }
    payload["guidance"]["installation"] = (
        "The saved mount width is 40 mm. Check physical fit before installation."
    )
    return GuidanceUpdate.model_validate(payload)


def controller_case(tmp_path, monkeypatch, *, partial=False, empty=False, staged=True):
    """Reuse the native baseline's intent while replacing its only CAD producer."""
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1" if staged else "0")
    for module in (agent, completion, delivery_guidance, measure):
        monkeypatch.setattr(module, "engine_fingerprint", lambda: ENGINE)

    def saved_build(service, sid, owner, design, frozen, parent, *args, **kwargs):
        if partial:
            design = type(design).model_validate(
                {
                    **design.model_dump(),
                    "features": [design.features[0].model_dump()],
                    "outputs": ["base"],
                }
            )
        binding = service.store.plan_snapshot(sid, owner)
        directory = service.store.directory(sid, RID)
        directory.mkdir(parents=True)
        files = {
            "program.py": b"# Saved fixture; never execute.\n",
            "document.json": design.model_dump_json(indent=2).encode(),
            "part.step": b"Saved STEP fixture; not geometric proof",
            "part.stl": b"Saved mesh fixture; never render",
        }
        for name, content in files.items():
            (directory / name).write_bytes(content)
        measured = Measurement(
            revision_id=RID,
            contract_hash=frozen.fingerprint(),
            engine_fingerprint=ENGINE,
            geometry_valid=True,
            artifact_hashes={
                name: hashlib.sha256(content).hexdigest() for name, content in files.items()
            },
            requirements_satisfied=not partial,
            findings=[
                Finding(
                    requirement_id=requirement.id,
                    status="unverified" if partial else "pass",
                    expected=requirement.value,
                    measured=requirement.value
                    if not partial and requirement.metric != "unmeasured"
                    else None,
                    method="Controlled saved-byte fixture",
                    reason="Controller behavior only; no native measurement ran",
                )
                for requirement in frozen.requirements
            ],
        )
        revision = {
            "id": RID,
            "parent_id": parent,
            "code": files["program.py"].decode(),
            "document": design.model_dump(),
            "contract": frozen.model_dump(),
            "measurement": measured.model_dump(),
            "parameters": dict(design.variables),
            "editable_parameters": list(design.variables),
            "operation_record": {**coverage(binding, design), "plan_snapshot": binding},
        }
        service.store.add_revision(sid, revision)
        return revision

    monkeypatch.setattr(process_fixture, "build_document", saved_build)
    if empty:
        service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
        sid = service.store.create(
            "owner",
            "Preserve the requested bore",
            contract().model_dump(),
            authoring_mode="native_operations",
        )["id"]
        service.store.set_design_plan(sid, "owner", plan(), expected_plan_version=0)
    else:
        service, sid, _, _ = process_fixture.baseline(tmp_path)
    # This is already-consumed fixture intent, not a new queued owner edit.
    accepted = service.store.session_record(sid, "owner")
    accepted["instructions"] = ["Preserve the requested bore"]
    service.store.update(
        sid,
        instructions=accepted["instructions"],
        contract_application=input_baseline(accepted),
    )

    def forbid_native(*args, **kwargs):
        pytest.fail("Controller tests must not start a model, CAD, renderer or worker")

    monkeypatch.setattr(service, "start", forbid_native)
    monkeypatch.setattr(service, "build", forbid_native)
    monkeypatch.setattr(service.executor, "build", forbid_native)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbid_native)
    monkeypatch.setattr(document_service, "build_document", forbid_native)
    monkeypatch.setattr(
        author_images,
        "prepare_author_images",
        lambda *args, **kwargs: author_images.AuthorImages((), (), None, "saved-fixture", "none"),
    )
    monkeypatch.setattr(
        completion_inline,
        "author_checkpoint",
        lambda *args, **kwargs: {"status": "unverified", "reason": "Saved-byte fixture; no probes"},
    )
    clock = Clock()
    monkeypatch.setattr(agent.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(agent.time, "sleep", clock.advance)
    case = SimpleNamespace(
        service=service,
        sid=sid,
        clock=clock,
        calls=[],
        factories=[],
        trace=[],
        validation=[],
        capture=[],
    )

    def validate(service_arg, job, deadline, cancelled, state):
        case.trace.append("validation")
        selected = service_arg.store.get(sid, "owner").get("selected_revision")
        guide = guidance_context(service_arg, sid, "owner", selected) if selected else None
        case.validation.append(
            {
                "job_id": job["id"],
                "deadline": deadline,
                "state": state,
                "guide": guide,
                "incomplete_finish_job_id": service_arg.store.get(sid, "owner").get(
                    "incomplete_finish_job_id"
                ),
            }
        )
        service_arg.store.update(
            sid, completion_report={"status": "unverified", "physical_readiness": "unverified"}
        )

    def capture(service_arg, job, deadline, cancelled, state):
        case.trace.append("capture")
        case.capture.append({"job_id": job["id"], "deadline": deadline, "state": state})
        return {"status": "not_requested", "scope": "No package worker in controller tests"}

    monkeypatch.setattr(completion, "finalize_completion", validate)
    monkeypatch.setattr(native_delivery, "capture_native_delivery", capture)
    return case


def run_script(case, script, *, max_actions=12, max_seconds=1800):
    class Author:
        model = "scripted-handoff-controller"
        timeout_s = None

        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            case.calls.append(
                {
                    "context": context,
                    "schema": kwargs["output_model"],
                    "timeout": self.timeout_s,
                    "job_id": case.service.store.get(case.sid, "owner")["job_id"],
                }
            )
            case.trace.append(
                "instructions" if kwargs["output_model"] is GuidanceUpdate else "action"
            )
            assert len(case.calls) <= max_actions, "The controller reset its action counter"
            result = script(context, kwargs["output_model"])
            return kwargs["output_model"].model_validate(result.model_dump()), LLMUsage(
                self.model, 11, 7
            )

        generate_structured_with_image = generate_structured

    def factory(**kwargs):
        author = Author()
        author.cancelled = kwargs["cancelled"]
        case.factories.append(author)
        return author

    case.service.client_factory = factory
    case.before = deepcopy(case.service.store.get(case.sid, "owner"))
    case.jid = case.service.store.enqueue(
        case.sid,
        "owner",
        {"max_actions": max_actions, "max_seconds": max_seconds, "delivery_requested": True},
    )
    job = case.service.store.claim()
    assert job is not None and job["id"] == case.jid
    case.service.run_job(job)
    return case.service.store.get(case.sid, "owner")


def syntax_action():
    return NativeAction(kind="schema", query="feature:box", explanation="Read exact syntax")


def phase_timeouts(case, *, late_document=False):
    attempts = Counter()

    def script(context, output_model):
        phase = context["handoff_phase"]
        attempts[phase] += 1
        if output_model is GuidanceUpdate:
            return reviewed_instructions(context)
        if phase == "construction" and attempts[phase] == 1 and not late_document:
            return syntax_action()
        if phase == "construction" and late_document:
            # The construction cue is now soft for inference. Cross the hard
            # instruction cutoff to retain the no-worker/no-publication refusal.
            case.clock.advance(context["seconds_left"] + 1)
            revision = case.service.store.revision(case.sid, RID)
            return NativeAction(
                kind="document",
                document_json=json.dumps(revision["document"]),
                explanation="A response after the hard cutoff must not rebuild saved CAD",
            )
        case.clock.cross_boundary(context)
        if phase == "construction":
            return syntax_action()
        raise LLMError("Model call timed out at the declared phase deadline")

    return script


def test_soft_transition_then_hard_timeout_keeps_same_job_handoff_and_validation(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    saved = run_script(case, phase_timeouts(case), max_seconds=300)
    phases = [call["context"]["handoff_phase"] for call in case.calls]
    assert list(dict.fromkeys(phases)) == ["construction", "handoff", "instructions"]
    assert case.calls[-1]["schema"] is GuidanceUpdate
    assert len(case.factories) == 1 and {call["job_id"] for call in case.calls} == {case.jid}
    assert case.trace[-3:] == ["instructions", "validation", "capture"]
    assert case.validation[0]["guide"]["status"] == "current"
    assert saved["native_review"]["status"] == "ready_for_validation"
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert saved["contract"] == case.before["contract"]
    assert saved["prompt"] == case.before["prompt"]
    assert saved["instructions"] == case.before["instructions"]
    # Crossing only the soft construction cue now returns one additional action,
    # rather than cancelling it and spending another call on the same phase.
    assert len(saved["usage"]) == 3
    assert all(item["input_tokens"] == 11 and item["output_tokens"] == 7 for item in saved["usage"])
    for call in case.calls:
        assert call["context"]["request"] == case.before["prompt"]
        assert call["context"]["frozen_contract"] == case.before["contract"]
        assert call["context"]["user_revision_history"] == case.before["instructions"]
        assert 0 < call["timeout"] <= 300
    assert case.clock.now < case.validation[0]["deadline"] <= case.capture[0]["deadline"]


def test_construction_response_crossing_hard_cutoff_is_discarded_before_any_worker_or_revision(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    revisions = deepcopy(case.service.store.revisions(case.sid))
    saved = run_script(case, phase_timeouts(case, late_document=True), max_seconds=300)
    assert case.service.store.revisions(case.sid) == revisions
    assert saved["selected_revision"] == RID
    assert case.validation[0]["guide"]["status"] == "current"
    assert saved["state"] != "error"
    assert any(
        event["kind"] == "handoff_boundary"
        and event.get("phase") == "construction"
        and event.get("active_phase") == "instructions"
        for event in saved["events"]
    )


def test_inference_crossing_soft_cue_admits_one_worker_with_current_handoff_deadline(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    workers = []
    checks = []

    def checkpoint(service, sid, owner, rid, jid, deadline, cancelled):
        checks.append(deadline)
        return {"status": "unverified", "reason": "Controlled early-check deadline"}

    monkeypatch.setattr(completion_inline, "author_checkpoint", checkpoint)

    def inspect(service, sid, owner, rid, part_id, *, timeout, cancelled):
        assert (service, sid, owner, rid) == (case.service, case.sid, "owner", RID)
        remaining = case.work_cutoff - case.clock.now
        assert timeout == min(print_setup_service.WORKER_TIMEOUT, remaining)
        assert any(
            event["kind"] == "handoff_phase" and event.get("phase") == "handoff"
            for event in service.store.get(sid, owner)["events"]
        )
        workers.append(cancelled())
        case.clock.advance(remaining + 1)
        workers.append(cancelled())
        assert case.factories[0].cancelled() is True
        raise WorkspaceError("Active handoff deadline reached before print publication")

    monkeypatch.setattr(print_setup_service, "inspect_print_setup", inspect)

    def script(context, output_model):
        if output_model is GuidanceUpdate:
            return reviewed_instructions(context)
        assert len(case.calls) == 1
        assert context["handoff_phase"] == "construction"
        case.work_cutoff = case.clock.now + context["seconds_left"]
        assert context["per_call_seconds"] == context["seconds_left"]
        assert context["seconds_left"] > context["phase_seconds_left"]
        assert case.calls[-1]["timeout"] == context["per_call_seconds"]
        assert checks == [case.clock.now + context["phase_seconds_left"]]
        case.clock.cross_boundary(context)
        assert case.factories[0].cancelled() is False
        return NativeAction(
            kind="print_setup",
            query="inspect",
            part_id="mount",
            revision_id=RID,
            explanation="Inspect the saved part after inference crosses the soft cue",
        )

    saved = run_script(case, script, max_seconds=300)
    assert workers == [False, True]  # One admission, then the unchanged hard worker cutoff.
    assert len(case.calls) == 2 and len(case.factories) == 1
    assert {call["job_id"] for call in case.calls} == {case.jid}
    assert case.calls[-1]["schema"] is GuidanceUpdate
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"
    assert not saved.get("print_setup_records")


def test_response_crossing_author_deadline_cannot_start_worker_or_instruction(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    revisions = deepcopy(case.service.store.revisions(case.sid))

    def script(context, output_model):
        # With a 300-second job the protected instruction tail is 30 seconds.
        case.clock.advance(context["seconds_left"] + 31)
        assert case.factories[0].cancelled() is True
        revision = case.service.store.revision(case.sid, RID)
        return NativeAction(kind="document", document_json=json.dumps(revision["document"]))

    saved = run_script(case, script, max_seconds=300)
    assert saved["state"] == "budget_exhausted"
    assert len(case.calls) == 1
    assert case.service.store.revisions(case.sid) == revisions
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert guidance_context(case.service, case.sid, "owner")["status"] == "unbound"


@pytest.mark.parametrize("max_actions", [1, 2, 6])
def test_finite_action_limit_contains_the_protected_instruction_call(
    tmp_path, monkeypatch, max_actions
):
    case = controller_case(tmp_path, monkeypatch)

    def script(context, output_model):
        return reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()

    saved = run_script(case, script, max_actions=max_actions)
    assert len(case.calls) <= max_actions
    assert sum(call["schema"] is GuidanceUpdate for call in case.calls) == 1
    assert case.calls[-1]["context"]["handoff_phase"] == "instructions"
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"
    assert saved.get("accepted_revision") is None


def test_current_partial_instructions_do_not_close_missing_construction(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, partial=True)
    saved = run_script(case, lambda context, model: reviewed_instructions(context), max_actions=1)
    revision = case.service.store.revision(case.sid, RID)
    assert revision["operation_record"]["graph_complete"] is False
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"
    assert saved["native_review"]["status"] == "needs_work"
    assert any("planned construction" in reason for reason in saved["native_review"]["reasons"])
    assert saved["state"] == "needs_review"
    assert case.validation[0]["incomplete_finish_job_id"] == case.jid
    assert "incomplete_finish_job_id" not in saved  # Store.finish consumes the job-local marker.
    assert saved["native_review"]["physical_readiness"] == "unverified"
    assert saved.get("accepted_revision") is None


@pytest.mark.parametrize("cross_soft", [False, True])
def test_cancellation_stops_before_reserved_instruction_or_late_action(
    tmp_path, monkeypatch, cross_soft
):
    case = controller_case(tmp_path, monkeypatch)

    def script(context, output_model):
        if cross_soft:
            case.clock.cross_boundary(context)
            assert case.factories[0].cancelled() is False
        case.service.store.cancel(case.sid, "owner")
        assert case.factories[0].cancelled() is True
        return syntax_action()

    saved = run_script(case, script, max_seconds=300)
    assert len(case.calls) == 1 and saved["state"] == "cancelled"
    assert not any(call["schema"] is GuidanceUpdate for call in case.calls)
    assert saved["selected_revision"] == RID and saved.get("accepted_revision") is None
    assert guidance_context(case.service, case.sid, "owner")["status"] == "unbound"


def test_no_saved_geometry_cannot_receive_fabricated_instructions(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, empty=True)
    saved = run_script(case, lambda context, model: syntax_action(), max_actions=3)
    assert not any(call["schema"] is GuidanceUpdate for call in case.calls)
    assert case.service.store.revisions(case.sid) == []
    assert not saved.get("selected_revision") and not saved.get("accepted_revision")
    assert not saved.get("delivery_guidance_record")
    assert saved["state"] in {"needs_review", "budget_exhausted"}


def test_unlimited_time_uses_only_declared_action_limit_for_phase_progress(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)

    def script(context, output_model):
        assert context["seconds_left"] is None
        return reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()

    run_script(case, script, max_actions=4, max_seconds=None)
    assert len(case.calls) <= 4 and case.calls[-1]["schema"] is GuidanceUpdate
    assert case.validation[0]["deadline"] is None and case.capture[0]["deadline"] is None
    assert case.clock.now == 1000.0


def test_unstaged_default_keeps_normal_action_contract(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, staged=False)

    def script(context, output_model):
        assert "handoff_phase" not in context
        assert output_model is NativeAction
        return NativeAction(
            kind="guidance",
            query="finish",
            explanation="Review saved instructions",
            document_json=reviewed_instructions(context).model_dump_json(),
        )

    saved = run_script(case, script, max_actions=1)
    assert len(case.calls) == 1
    assert saved["native_review"]["status"] == "ready_for_validation"
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"


def test_construction_worker_callback_observes_its_phase_boundary(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    callbacks = []

    def inspect(service, sid, owner, rid, part_id, *, timeout, cancelled):
        assert (service, sid, owner, rid) == (case.service, case.sid, "owner", RID)
        # Early workers retain the construction cue even though model inference
        # now has a longer allowance.
        assert 0 < timeout <= case.calls[-1]["context"]["phase_seconds_left"]
        callbacks.append(cancelled())
        case.clock.cross_boundary(case.calls[-1]["context"])
        callbacks.append(cancelled())
        raise WorkspaceError("Phase ended before print publication")

    monkeypatch.setattr(print_setup_service, "inspect_print_setup", inspect)

    def script(context, output_model):
        if output_model is GuidanceUpdate:
            return reviewed_instructions(context)
        if context["handoff_phase"] == "construction":
            return NativeAction(
                kind="print_setup",
                query="inspect",
                part_id="mount",
                revision_id=RID,
                explanation="Inspect saved printing geometry within the phase",
            )
        case.clock.cross_boundary(context)
        raise LLMError("Model call timed out at the declared phase deadline")

    saved = run_script(case, script)
    assert callbacks == [False, True]
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"
    assert not saved.get("print_setup_records")


def preparation_obligation(case, monkeypatch):
    """Use a controlled preparation service; the typed intent predicate has its own tests."""
    case.prepared = False
    case.preparations = []

    def preparation(document, guide, *, explicitly_requested=False):
        assert explicitly_requested is False
        return {
            "required": True,
            "part_ids": ["part"],
            "status": "current" if case.prepared else "missing",
        }

    def prepare(service, sid, owner, rid, *, timeout, cancelled):
        assert (service, sid, owner, rid) == (case.service, case.sid, "owner", RID)
        assert not cancelled()
        case.preparations.append(timeout)
        case.prepared = True
        return {"status": "prepared", "revision_id": rid}

    monkeypatch.setattr(native_finish, "print_preparation", preparation)
    monkeypatch.setattr(print_setup_service, "prepare_print_files", prepare)


def test_instruction_boundary_allows_missing_preparation_then_protected_refresh(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)
    preparation_obligation(case, monkeypatch)

    def script(context, output_model):
        if context["handoff_phase"] == "construction":
            case.clock.advance(context["seconds_left"] + 1)
            return syntax_action()
        if output_model is NativeAction:
            assert context["handoff_phase"] == "instructions"
            assert context["handoff_preflight"]["print_preparation"]["status"] == "missing"
            assert "handoff_correction" in context
            assert context["seconds_left"] < 100
            return NativeAction(
                kind="print_setup",
                query="prepare",
                revision_id=RID,
                explanation="Prepare current files before the final customer review",
            )
        assert case.prepared
        return reviewed_instructions(context)

    saved = run_script(case, script, max_seconds=1800)
    assert len(case.preparations) == 1
    assert case.preparations[0] < 100
    assert case.calls[-1]["schema"] is GuidanceUpdate
    assert case.trace[-3:] == ["instructions", "validation", "capture"]
    assert saved["native_review"]["status"] == "ready_for_validation"
    assert saved.get("accepted_revision") is None
    assert len(case.calls) <= 90


def test_unrepaired_work_uses_final_reserved_time_for_truthful_instructions(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    preparation_obligation(case, monkeypatch)

    def script(context, output_model):
        if context["handoff_phase"] == "construction":
            case.clock.advance(context["seconds_left"] + 1)
            return syntax_action()
        if output_model is NativeAction:
            case.clock.advance(context["seconds_left"] + 1)
            raise LLMError("Corrective inference reached its reserved cutoff")
        return reviewed_instructions(context)

    saved = run_script(case, script, max_seconds=1800)
    assert case.calls[-1]["schema"] is GuidanceUpdate
    assert not case.preparations
    assert case.calls[-1]["context"]["seconds_left"] <= 100
    assert saved["native_review"]["status"] == "needs_work"
    assert saved["state"] == "needs_review"
    assert guidance_context(case.service, case.sid, "owner")["status"] == "current"
    assert case.clock.now < case.validation[0]["deadline"]
    assert len(case.calls) <= 90


def test_last_existing_slot_cannot_be_consumed_by_corrective_preparation(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    preparation_obligation(case, monkeypatch)
    saved = run_script(
        case,
        lambda context, output_model: (
            reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()
        ),
        max_actions=2,
    )
    assert len(case.calls) == 2 and case.calls[-1]["schema"] is GuidanceUpdate
    assert not case.preparations
    assert saved["native_review"]["status"] == "needs_work"
    assert saved["state"] == "needs_review"


def test_flat_additive_preparation_stays_blocked_pending_supported_conversion(
    tmp_path, monkeypatch
):
    case = controller_case(tmp_path, monkeypatch)

    def preparation(document, guide, *, explicitly_requested=False):
        assert explicitly_requested is False
        return {
            "required": True,
            "supported": False,
            "part_ids": ["part"],
            "status": "conversion_required",
        }

    monkeypatch.setattr(
        native_finish,
        "print_preparation",
        preparation,
    )
    saved = run_script(case, lambda context, model: reviewed_instructions(context), max_actions=1)
    assert saved["native_review"]["status"] == "needs_work"
    assert saved["native_review"]["print_preparation"]["status"] == "conversion_required"
    assert any(
        "Convert flat additive CAD" in reason for reason in saved["native_review"]["reasons"]
    )
    assert saved["native_review"]["physical_readiness"] == "unverified"
