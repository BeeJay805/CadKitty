"""Native transport delegates atomic publication, with no CAD or model execution."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from tests.unit.workspace.test_native_handoff_controller import (
    RID,
    controller_case,
    reviewed_instructions,
    run_script,
)

from forge_mind.workspace import construction_transaction
from forge_mind.workspace.native_authoring import NativeAction


def finish_action(context):
    return NativeAction(
        kind="guidance",
        query="finish",
        explanation="Review exact current files",
        document_json=reviewed_instructions(context).model_dump_json(),
    )


@pytest.mark.parametrize("kind", ["document", "feature_edit"])
def test_optional_patch_reaches_transaction_service_without_independent_publication(
    tmp_path, monkeypatch, kind
):
    case = controller_case(tmp_path, monkeypatch, staged=False)
    calls = []
    selections = []
    original_update = case.service.store.update

    def observe_update(sid, **values):
        if "selected_revision" in values:
            selections.append(set(values))
        return original_update(sid, **values)

    monkeypatch.setattr(case.service.store, "update", observe_update)
    prepared = SimpleNamespace(successor_snapshot={"plan_hash": "c" * 64})

    def prepare(service, sid, owner, jid, document, patch, *, author_intent):
        assert service is case.service and (sid, owner, jid) == (case.sid, "owner", case.jid)
        assert document.model_dump() == case.service.store.revision(case.sid, RID)["document"]
        assert patch.objective == "Bind the authored construction"
        assert patch.basis == case.service.store.edit_basis(case.sid, "owner")
        assert author_intent.startswith(case.before["prompt"])
        assert case.before["instructions"][0] in author_intent
        calls.append(("prepare", document, patch))
        return prepared

    def build(service, sid, owner, jid, value, *, explanation, timeout, cancelled):
        assert value is prepared and not cancelled()
        assert timeout is not None and timeout > 0
        calls.append(("build",))
        # This mock stands in for atomic Store publication. Real transaction and
        # recovery behavior have independent service tests.
        return {
            "action": "construction",
            "applied": True,
            "plan_applied": True,
            "cad_applied": True,
            "candidate_revision_id": RID,
            "revision_id": RID,
        }

    monkeypatch.setattr(construction_transaction, "prepare_construction", prepare)
    monkeypatch.setattr(construction_transaction, "build_construction", build)

    def script(context, output_model):
        if len(case.calls) > 1:
            return finish_action(context)
        basis = context["edit_basis"]
        document = case.service.store.revision(case.sid, RID)["document"]
        payload = (
            document
            if kind == "document"
            else {
                "basis": basis,
                "changes": [{"update": [document["features"][0]]}],
            }
        )
        return NativeAction(
            kind=kind,
            explanation="Submit explicit CAD and operation ownership together",
            document_json=json.dumps(payload),
            plan_patch_json=json.dumps(
                {"basis": basis, "objective": "Bind the authored construction"}
            ),
        )

    saved = run_script(case, script, max_actions=3)
    assert [call[0] for call in calls] == ["prepare", "build"]
    # Only the unchanged end-of-loop finalizer writes selection. The adapter
    # must never add its ordinary document/feature publication after the service.
    assert selections == [{"selected_revision", "usage"}]
    assert saved["selected_revision"] == RID
    assert saved.get("accepted_revision") is None
    assert saved["native_review"]["status"] == "ready_for_validation"


def test_non_cad_sidecar_is_recoverable_and_retains_exact_rejected_payload(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, staged=False)
    payload = '{"basis":"intentionally invalid non-CAD payload"}'

    def script(context, output_model):
        if len(case.calls) == 1:
            return NativeAction(
                kind="schema",
                query="feature:box",
                explanation="Invalid sidecar scope",
                plan_patch_json=payload,
            )
        failure = context["failed_submission"]
        assert failure["submitted_action"]["plan_patch_json"] == payload
        assert "only on native document" in failure["validation_error"]
        return finish_action(context)

    saved = run_script(case, script, max_actions=3)
    assert len(case.calls) == 2
    assert saved["state"] != "error"
    assert saved["native_review"]["status"] == "ready_for_validation"


def test_feature_and_patch_basis_mismatch_never_reaches_builder(tmp_path, monkeypatch):
    case = controller_case(tmp_path, monkeypatch, staged=False)
    before = deepcopy(case.service.store.revisions(case.sid))
    prepared = []
    monkeypatch.setattr(
        construction_transaction,
        "prepare_construction",
        lambda *args, **kwargs: prepared.append(args),
    )

    def script(context, output_model):
        if len(case.calls) > 1:
            assert "same EditBasis" in context["failed_submission"]["validation_error"]
            return finish_action(context)
        basis = context["edit_basis"]
        mismatch = {**basis, "plan_version": basis["plan_version"] + 1}
        document = case.service.store.revision(case.sid, RID)["document"]
        return NativeAction(
            kind="feature_edit",
            explanation="Reject conflicting original identities",
            document_json=json.dumps(
                {"basis": basis, "changes": [{"update": [document["features"][0]]}]}
            ),
            plan_patch_json=json.dumps({"basis": mismatch, "objective": "Invalid combined basis"}),
        )

    saved = run_script(case, script, max_actions=3)
    assert not prepared
    assert case.service.store.revisions(case.sid) == before
    assert saved["selected_revision"] == RID
