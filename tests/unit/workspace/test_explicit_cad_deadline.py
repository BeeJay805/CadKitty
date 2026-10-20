"""Explicit author CAD keeps its request deadline; no CAD producer runs here."""

from __future__ import annotations

import json

import pytest
from tests.unit.workspace.test_native_handoff_controller import (
    GuidanceUpdate,
    controller_case,
    preparation_obligation,
    reviewed_instructions,
    run_script,
    syntax_action,
)
from tests.unit.workspace.test_operation_authoring_flow import document
from tests.unit.workspace.test_revision_bound_edits import saved_revision

from forge_mind.workspace import document_service
from forge_mind.workspace.authoring import AssemblyEdit
from forge_mind.workspace.native_authoring import NativeAction
from forge_mind.workspace.store import WorkspaceError


def cad_action(kind, context):
    design = document(41)
    if kind == "assembly_edit":
        payload = AssemblyEdit.model_validate(
            {
                "variables": design.variables,
                "parts": [
                    {
                        "id": "mount",
                        "name": "Mount",
                        "features": [feature.model_dump() for feature in design.features],
                        "output": "bored",
                    }
                ],
                "roots": [{"id": "mount", "definition": "mount"}],
            }
        ).model_dump_json()
    elif kind == "feature_edit":
        feature = design.features[0].model_dump()
        feature["params"]["y"] = 25
        payload = json.dumps({"basis": context["edit_basis"], "changes": [{"update": [feature]}]})
    else:
        payload = design.model_dump_json()
    return NativeAction(kind=kind, document_json=payload, explanation="Build the authored edit")


@pytest.mark.parametrize("kind", ["document", "assembly_edit", "feature_edit"])
@pytest.mark.parametrize("reply_at", [80, 81, 82, 116, 117, 118])
def test_explicit_cad_deadline_has_no_construction_cue_cliff(tmp_path, monkeypatch, kind, reply_at):
    case = controller_case(tmp_path, monkeypatch, empty=kind != "feature_edit")
    workers = []
    cancellations = []
    revisions = case.service.store.revisions(case.sid)

    def build(*args):
        timeout, cancelled = args[-2:]
        workers.append(timeout)
        cancellations.append(cancelled())
        case.clock.now = 1116.999
        cancellations.append(cancelled())
        case.clock.now = 1117
        cancellations.append(cancelled())
        raise WorkspaceError("Controlled cancellation before publication")

    monkeypatch.setattr(document_service, "build_document", build)

    def script(context, output_model):
        if len(case.calls) == 1:
            assert context["handoff_phase"] == "construction"
            assert context["seconds_left"] == 117
            assert context["phase_seconds_left"] == 81
            case.clock.now = 1000 + reply_at
            return cad_action(kind, context)
        return reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()

    saved = run_script(case, script, max_actions=4, max_seconds=180)
    assert workers == ([pytest.approx(117 - reply_at)] if reply_at < 117 else [])
    assert cancellations == ([False, False, True] if reply_at < 117 else [])
    assert case.service.store.revisions(case.sid) == revisions
    assert saved.get("accepted_revision") is None
    assert len(case.calls) <= 4
    assert case.validation[0]["deadline"] == 1171
    assert case.capture[0]["deadline"] == 1180


@pytest.mark.parametrize("cancel_when", ["response", "worker"])
def test_explicit_cad_still_obeys_original_cancellation(tmp_path, monkeypatch, cancel_when):
    case = controller_case(tmp_path, monkeypatch, empty=True)
    workers = []
    cancellations = []

    def build(*args):
        workers.append(True)
        cancellations.append(args[-1]())
        case.service.stop.set()
        cancellations.append(args[-1]())
        raise WorkspaceError("Owner cancelled before publication")

    monkeypatch.setattr(document_service, "build_document", build)

    def script(context, output_model):
        case.clock.now = 1080
        if cancel_when == "response":
            case.service.stop.set()
        return cad_action("document", context)

    saved = run_script(case, script, max_actions=4, max_seconds=180)
    assert workers == ([True] if cancel_when == "worker" else [])
    assert cancellations == ([False, True] if cancel_when == "worker" else [])
    assert saved["state"] == "cancelled"
    assert case.service.store.revisions(case.sid) == []


def test_explicit_cad_unbounded_time_creates_no_deadline(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, empty=True)
    workers = []
    cancellations = []

    def build(*args):
        workers.append(args[-2])
        case.clock.advance(100000)
        cancellations.append(args[-1]())
        raise WorkspaceError("Controlled unbounded worker exit")

    monkeypatch.setattr(document_service, "build_document", build)

    def script(context, output_model):
        return cad_action("document", context) if len(case.calls) == 1 else syntax_action()

    run_script(case, script, max_actions=4, max_seconds=None)
    assert workers == [None]
    assert cancellations == [False]
    assert len(case.calls) == 4


def test_explicit_cad_correction_keeps_protected_review_cutoff(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch)
    preparation_obligation(case, monkeypatch)
    workers = []
    cancellations = []

    def build(*args):
        workers.append(args[-2])
        case.clock.now = 1125.999
        cancellations.append(args[-1]())
        case.clock.now = 1126
        cancellations.append(args[-1]())
        raise WorkspaceError("Protected correction ended before publication")

    monkeypatch.setattr(document_service, "build_document", build)

    def script(context, output_model):
        if len(case.calls) == 1:
            case.clock.now = 1118
            return syntax_action()
        if "handoff_correction" in context and not workers:
            case.clock.now = 1120
            return cad_action("document", context)
        return reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()

    run_script(case, script, max_actions=4, max_seconds=180)
    assert workers == [6]
    assert cancellations == [False, True]
    assert case.calls[-1]["schema"] is GuidanceUpdate
    assert case.validation[0]["deadline"] == 1171
    assert case.capture[0]["deadline"] == 1180


@pytest.mark.parametrize("kind", ["document", "assembly_edit", "feature_edit"])
@pytest.mark.parametrize("stop", ["deadline", "cancelled"])
def test_late_valid_cad_candidate_is_retained_without_publication(
    tmp_path, monkeypatch, kind, stop
):
    case = controller_case(tmp_path, monkeypatch, empty=kind != "feature_edit")
    original = case.service.store.get(case.sid, "owner").get("selected_revision")
    candidates = []
    candidate_bytes = []
    cancellations = []

    def build(service, sid, owner, design, contract, parent, explanation, timeout, cancelled):
        candidate = saved_revision(
            service.store,
            sid,
            "b" * 32,
            design,
            contract,
            service.store.plan_snapshot(sid, owner),
            parent,
        )
        candidates.append(candidate)
        candidate_bytes.append(
            (service.store.directory(sid, "b" * 32) / "document.json").read_bytes()
        )
        if stop == "deadline":
            case.clock.now = 1117
        else:
            case.service.stop.set()
        cancellations.append(cancelled())
        return candidate

    monkeypatch.setattr(document_service, "build_document", build)

    def script(context, output_model):
        if len(case.calls) == 1:
            case.clock.now = 1080
            return cad_action(kind, context)
        return reviewed_instructions(context) if output_model is GuidanceUpdate else syntax_action()

    saved = run_script(case, script, max_actions=4, max_seconds=180)
    assert len(candidates) == 1
    assert cancellations == [True]
    assert saved.get("selected_revision") == original
    assert saved.get("accepted_revision") is None
    assert saved.get("best_revision") != "b" * 32
    retained = case.service.store.revision(case.sid, "b" * 32)
    assert retained["document"] == json.loads(json.dumps(candidates[0]["document"]))
    assert (
        case.service.store.directory(case.sid, "b" * 32) / "document.json"
    ).read_bytes() == candidate_bytes[0]
    assert retained["measurement"]["geometry_valid"] is True
    assert not retained["measurement"]["errors"]
    assert case.service.store.feature_publication(case.sid, "owner", case.jid) is None
