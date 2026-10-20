"""Controller lifecycle with real atomic eligibility and saved-byte CAD fixtures.

Atomic eligibility and phase admission are tested in test_native_handoff_auto.
These controls exercise the real agent loop across publication, host dispatch,
editable image review, manual recovery, and the unchanged finalizer hooks.
"""

from __future__ import annotations

import base64
import hashlib
import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_native_handoff_controller import (
    RID,
    controller_case,
    reviewed_instructions,
    run_script,
)

from forge_mind.llm.client import LLMError
from forge_mind.workspace import author_images, document_service, native_finish, native_handoff_auto
from forge_mind.workspace import native_handoff_prepare as handoff
from forge_mind.workspace.delivery_guidance import GuidanceUpdate, guidance_context
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.operation_plan import coverage
from forge_mind.workspace.store import WorkspaceError


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    case.publications = []
    case.claim_inputs = []
    case.preparations = []
    case.auto_result = "prepared_for_instruction_review"
    case.valid_geometry = True
    case.prepared_revisions = set()
    case.cross_during_claim = False
    case.actual_print_preparation = native_finish.print_preparation
    original = deepcopy(case.service.store.revision(case.sid, RID))

    def publish(service, sid, owner, design, frozen, parent, *args, **kwargs):
        rid = ("b" if not case.publications else "c") * 32
        directory = service.store.directory(sid, rid)
        directory.mkdir(parents=True)
        files = {}
        for name in original["measurement"]["artifact_hashes"]:
            content = (service.store.directory(sid, RID) / name).read_bytes()
            if name == "document.json":
                content = design.model_dump_json(indent=2).encode()
            (directory / name).write_bytes(content)
            files[name] = hashlib.sha256(content).hexdigest()
        image = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWHcAAAAASUVORK5CYII="
        )
        for name in ("render.png", "render-inspection.png"):
            (directory / name).write_bytes(image)
            files[name] = hashlib.sha256(image).hexdigest()
        binding = service.store.plan_snapshot(sid, owner)
        revision = deepcopy(original)
        revision.update(
            id=rid,
            parent_id=parent,
            document=design.model_dump(),
            contract=frozen.model_dump(),
            operation_record={**coverage(binding, design), "plan_snapshot": binding},
        )
        revision["measurement"].update(
            revision_id=rid,
            contract_hash=frozen.fingerprint(),
            geometry_valid=case.valid_geometry,
            artifact_hashes=files,
        )
        service.store.add_revision(sid, revision)
        case.publications.append(rid)
        return revision

    monkeypatch.setattr(document_service, "build_document", publish)

    actual_claim = native_handoff_auto.claim_automatic_preparation

    def claim(store, sid, owner, jid, **kwargs):
        case.claim_inputs.append(
            deepcopy({key: value for key, value in kwargs.items() if key != "schedule"})
        )
        case.schedule = kwargs["schedule"]
        marker = actual_claim(store, sid, owner, jid, **kwargs)
        if marker is not None and case.cross_during_claim:
            cutoff = kwargs["schedule"].deadline("construction")
            case.clock.advance(cutoff - case.clock.now + 0.1)
        return marker

    monkeypatch.setattr(native_handoff_auto, "claim_automatic_preparation", claim)

    def print_preparation(document, guide, *, explicitly_requested=False):
        # Controlled print facts, while finish_context and atomic admission run.
        rid = guide["basis"]["revision_id"]
        return {
            "required": True,
            "supported": True,
            "part_ids": ["part"],
            "status": "current" if rid in case.prepared_revisions else "missing",
        }

    monkeypatch.setattr(native_finish, "print_preparation", print_preparation)

    def prepare(service, sid, owner, jid, request, *, deadline, cancelled):
        assert not cancelled()
        if request.prepare_print:
            assert deadline == case.schedule.deadline("construction")
        case.clock.advance(2.0)
        case.preparations.append(request)
        if len(case.preparations) == 1 and request.prepare_print:
            if isinstance(case.auto_result, Exception):
                raise case.auto_result
            status = case.auto_result
        else:
            status = "prepared_for_instruction_review"
        if status == "prepared_for_instruction_review":
            case.prepared_revisions.add(request.revision_id)
        return {
            "status": status,
            "revision_id": request.revision_id,
            "delivery_guidance": guidance_context(service, sid, owner, request.revision_id),
            "guidance_schema": GuidanceUpdate.model_json_schema(),
            "requirements_inspection": {
                "results": [
                    {"requirement_id": "retained_unknown", "status": "unverified"},
                ]
            },
        }

    monkeypatch.setattr(handoff, "prepare_handoff", prepare)

    def images(directory, selected, **kwargs):
        names = ("render.png", "render-inspection.png")
        if not all((directory / name).is_file() for name in names):
            return author_images.AuthorImages((), (), None, "saved-fixture", "none")
        content = tuple((directory / name).read_bytes() for name in names)
        for name, data in zip(names, content, strict=True):
            assert (
                hashlib.sha256(data).hexdigest() == selected["measurement"]["artifact_hashes"][name]
            )
        return author_images.AuthorImages(
            content,
            names,
            selected["id"],
            "saved-fixture",
            "none",
        )

    monkeypatch.setattr(author_images, "prepare_author_images", images)
    original_invoke = author_images.invoke_author_images

    def invoke(client, images, arguments, **kwargs):
        def with_images(**request):
            assert request.pop("images_png") == images.images
            assert request.pop("image_labels") == images.labels
            return client.generate_structured(**request)

        client.generate_structured_with_images = with_images
        return original_invoke(client, images, arguments, **kwargs)

    monkeypatch.setattr(author_images, "invoke_author_images", invoke)
    case.document = original["document"]
    return case


def document_action(case):
    return NativeAction(
        kind="document",
        document_json=json.dumps(case.document),
        explanation="Publish current native fixture CAD",
    )


def manual_preparation(context):
    rid = context["selected"]["id"]
    return NativeAction(
        kind="inspect",
        query="prepare_handoff",
        revision_id=rid,
        explanation="Explicit manual preparation after review",
        document_json=handoff.PrepareHandoffRequest(
            revision_id=rid,
            prepare_print=False,
        ).model_dump_json(),
    )


def finish(context):
    return NativeAction(
        kind="guidance",
        query="finish",
        explanation="Review current image and facts",
        document_json=reviewed_instructions(context).model_dump_json(),
    )


def host_events(saved, kind):
    return [row for row in saved["events"] if row["kind"] == kind and row.get("origin") == "host"]


def test_same_job_publication_uses_one_host_slot_without_model_or_usage_row(lifecycle):
    case = lifecycle

    def script(context, output_model):
        if len(case.calls) == 1:
            return document_action(case)
        assert len(case.calls) == 2
        assert output_model is NativeAction  # Visual review can still edit geometry.
        assert context["image_observation"]["revision_id"] == "b" * 32
        assert context["image_observation"]["images"]
        assert "document" in output_model.model_fields["kind"].annotation.__args__
        return finish(context)

    saved = run_script(case, script)
    actions = host_events(saved, "action")
    assert len(actions) == 1 and actions[0]["action_index"] == 1
    assert actions[0]["job_id"] == case.jid
    assert actions[0]["automatic_handoff_attempt"] == saved["automatic_handoff_attempt"]
    assert len(case.calls) == len(saved["usage"]) == 2
    assert all(row["input_tokens"] == 11 and row["output_tokens"] == 7 for row in saved["usage"])
    outcome = host_events(saved, "automatic_handoff_result")
    assert len(outcome) == 1 and outcome[0]["elapsed_seconds"] == 2.0
    assert outcome[0]["outcome"] == {"status": "prepared_for_instruction_review"}
    assert len(case.preparations) == 1 and case.preparations[0].prepare_print is True
    assert len(case.validation) == len(case.capture) == 1
    assert saved["native_review"]["status"] == "ready_for_validation"
    assert saved.get("accepted_revision") is None
    assert saved["contract"] == case.before["contract"]


@pytest.mark.parametrize("automatic_result", ["blocked", WorkspaceError("Scoped check failed")])
def test_blocked_or_error_is_charged_once_then_manual_recovery_remains(lifecycle, automatic_result):
    case = lifecycle
    case.auto_result = automatic_result

    def script(context, output_model):
        if len(case.calls) == 1:
            return document_action(case)
        if len(case.calls) == 2:
            return manual_preparation(context)
        assert len(case.calls) == 3
        return finish(context)

    saved = run_script(case, script)
    assert len(host_events(saved, "action")) == 1
    assert len(case.preparations) == 2
    assert [request.prepare_print for request in case.preparations] == [True, False]
    assert len(case.calls) == len(saved["usage"]) == 3
    outcome = host_events(saved, "automatic_handoff_result")
    assert len(outcome) == 1 and outcome[0]["elapsed_seconds"] == 2.0
    expected = "error" if isinstance(automatic_result, Exception) else "blocked"
    assert outcome[0]["outcome"]["status"] == expected
    if isinstance(automatic_result, Exception):
        assert outcome[0]["outcome"]["message"] == str(automatic_result)
    assert saved["automatic_handoff_attempt"]["action_index"] == 1
    assert len(case.validation) == len(case.capture) == 1


def test_visual_edit_rebuilds_with_new_images_without_second_automatic_attempt(lifecycle):
    case = lifecycle

    def script(context, output_model):
        if len(case.calls) == 1:
            return document_action(case)
        if len(case.calls) == 2:
            assert output_model is NativeAction
            assert context["image_observation"]["revision_id"] == "b" * 32
            # Change an unbound dimension; retain the frozen width, thickness,
            # bore, feature ownership, and all captured operation-plan duties.
            case.document["features"][0]["params"]["y"] = 26.0
            return document_action(case)
        assert context["image_observation"]["revision_id"] == "c" * 32
        if len(case.calls) == 3:
            return manual_preparation(context)
        assert len(case.calls) == 4
        return finish(context)

    saved = run_script(case, script)
    assert case.publications == ["b" * 32, "c" * 32]
    previous = case.service.store.revision(case.sid, "b" * 32)["document"]
    edited = case.service.store.revision(case.sid, "c" * 32)["document"]
    assert previous["features"][0]["params"]["y"] == 24.0
    assert edited["features"][0]["params"]["y"] == 26.0
    assert previous["variables"] == edited["variables"]
    assert len(host_events(saved, "action")) == 1
    assert saved["automatic_handoff_attempt"]["revision_id"] == "b" * 32
    assert [request.revision_id for request in case.preparations] == ["b" * 32, "c" * 32]
    assert saved["selected_revision"] == "c" * 32
    assert len(case.calls) == len(saved["usage"]) == 4
    assert len(case.validation) == len(case.capture) == 1


def test_historical_candidate_keeps_manual_preparation_without_automatic_publication(lifecycle):
    case = lifecycle

    def script(context, output_model):
        return manual_preparation(context) if len(case.calls) == 1 else finish(context)

    saved = run_script(case, script)
    assert not case.publications and len(case.preparations) == 1
    assert not host_events(saved, "action")
    assert all(row["published_revision_id"] is None for row in case.claim_inputs)
    assert "automatic_handoff_attempt" not in saved
    assert len(case.calls) == len(saved["usage"]) == 2


def test_invalid_native_candidate_never_supplies_automatic_publication(lifecycle):
    case = lifecycle
    case.valid_geometry = False

    def script(context, output_model):
        return document_action(case) if len(case.calls) == 1 else finish(context)

    saved = run_script(case, script)
    assert case.publications == ["b" * 32]
    assert saved["selected_revision"] == RID
    assert not case.preparations and not host_events(saved, "action")
    assert all(row["published_revision_id"] is None for row in case.claim_inputs)
    assert len(case.validation) == len(case.capture) == 1


def test_claim_crossing_construction_cutoff_keeps_claim_but_never_runs_host_work(lifecycle):
    case = lifecycle
    case.cross_during_claim = True

    def script(context, output_model):
        if len(case.calls) == 1:
            return document_action(case)
        if len(case.calls) == 2:
            assert context["handoff_phase"] == "handoff"
            assert not case.preparations
            return manual_preparation(context)
        return finish(context)

    saved = run_script(case, script)
    assert len(host_events(saved, "action")) == 1
    assert len(case.preparations) == 1 and case.preparations[0].prepare_print is False
    outcome = host_events(saved, "automatic_handoff_result")
    assert len(outcome) == 1
    assert outcome[0]["outcome"] == {
        "status": "not_executed",
        "reason": "cancelled_or_deadline_exhausted",
    }
    assert saved["automatic_handoff_attempt"]["action_index"] == 1
    assert len(case.calls) == len(saved["usage"]) == 3
    assert len(case.validation) == len(case.capture) == 1


def test_real_print_preflight_and_atomic_claim_compose_after_native_publication(
    lifecycle, monkeypatch
):
    case = lifecycle
    monkeypatch.setattr(native_finish, "print_preparation", case.actual_print_preparation)
    source = deepcopy(case.document)
    case.document.update(
        production=None,
        features=[],
        outputs=[],
        assembly={
            "parts": [
                {
                    "id": "mount",
                    "name": "Mount",
                    "features": source["features"],
                    "output": source["outputs"][0],
                    "production": {"role": "make", "process": "additive"},
                }
            ],
            "roots": [{"id": "installed", "definition": "mount"}],
        },
    )
    case.auto_result = "blocked"

    def script(context, output_model):
        if len(case.calls) == 1:
            return document_action(case)
        assert len(case.preparations) == 1
        preflight = context["handoff_preflight"]
        assert preflight["print_preparation"]["required"] is True
        assert preflight["print_preparation"]["supported"] is True
        assert preflight["print_preparation"]["status"] == "missing"
        assert len(preflight["work_reasons"]) == 1
        assert output_model is NativeAction
        raise LLMError("Stop the controlled lifecycle after real eligibility was observed")

    saved = run_script(case, script)
    published = case.service.store.revision(case.sid, "b" * 32)
    assert published["document"]["production"] is None
    assert published["document"]["assembly"]["parts"][0]["production"]["process"] == "additive"
    assert all(
        published["operation_record"][key]
        for key in ("graph_complete", "references_complete", "parameters_complete")
    )
    assert len(host_events(saved, "action")) == 1
    assert saved["automatic_handoff_attempt"]["revision_id"] == "b" * 32
    assert len(host_events(saved, "automatic_handoff_result")) == 1
    assert len(case.validation) == len(case.capture) == 1
