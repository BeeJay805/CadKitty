"""Proposed pure regression tests; not installed or executed during the freeze.

After the root applies the prepared source change, copy this file to
tests/unit/workspace/test_intent_attribution_epoch.py and run its focused gate.
"""

from __future__ import annotations

from copy import deepcopy

import pytest

from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution import (
    IntentAttribution,
    IntentIssue,
)
from forge_mind.workspace.intent_attribution_service import (
    read_attribution,
    record_conflict,
    record_proposal,
    request_text,
)
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.store import Store, WorkspaceError

OWNER = "owner"
SOURCE = "Create a carrier for a sample. The sample is 45 mm wide and 20 mm tall."
RAW_ANSWER = "  The width belongs to the sample.\nRetain 45 mm. \u03b1  "


def contract(*, keep_width=True):
    requirements = [Requirement(id="height", text="Height intent", source_quote="20 mm tall")]
    if keep_width:
        requirements.insert(
            0, Requirement(id="width", text="Width intent", source_quote="45 mm wide")
        )
    return Contract(requirements=requirements)


def mapped(*, ambiguous=False):
    data = {
        "subjects": [
            {"id": "carrier", "label": "Carrier", "role": "designed", "source_quote": "carrier"},
            {"id": "sample", "label": "Sample", "role": "reference", "source_quote": "sample"},
        ],
        "facts": [
            {
                "id": "input_width",
                "subject_id": None if ambiguous else "sample",
                "candidate_subject_ids": ["carrier", "sample"] if ambiguous else [],
                "property": "width",
                "value": 45.0,
                "unit": "mm",
                "source_quote": "45 mm wide",
                "requirement_ids": ["width"],
            }
        ],
        "requirement_subjects": {} if ambiguous else {"width": ["sample"]},
        "issues": [width_issue().model_dump()] if ambiguous else [],
    }
    return IntentAttribution.model_validate(data)


def width_issue():
    return IntentIssue(
        id="width_subject",
        question="Which subject has this width?",
        reason="A material ownership conflict was reported.",
        fact_ids=["input_width"],
        requirement_ids=["width"],
    )


def height_issue():
    return IntentIssue(
        id="height_subject",
        question="Which subject has this height?",
        reason="A later material conflict needs an owner answer.",
        requirement_ids=["height"],
    )


def initial(tmp_path, *, ambiguous=False):
    store = Store(tmp_path)
    sid = store.create(OWNER, SOURCE, contract().model_dump(), authoring_mode="native_operations")[
        "id"
    ]
    jid = store.enqueue(sid, OWNER, {})
    assert store.claim()
    original = record_proposal(
        store,
        sid,
        OWNER,
        jid,
        mapped(ambiguous=ambiguous),
        contract(),
        SOURCE,
        store.edit_basis(sid, OWNER),
    )
    return store, sid, jid, original


def report_height(store, sid, jid, current_contract):
    return record_conflict(
        store,
        sid,
        OWNER,
        jid,
        [height_issue()],
        current_contract,
        request_text(store.session_record(sid, OWNER)),
        store.edit_basis(sid, OWNER),
    )


def assert_requirement_only(record):
    assert record["status"] == "needs_input"
    assert record["attribution"]["subjects"] == []
    assert record["attribution"]["facts"] == []
    assert record["attribution"]["requirement_subjects"] == {}
    assert record["attribution"]["issues"] == [height_issue().model_dump()]
    assert record["resolutions"] == []


@pytest.mark.parametrize("changed", ["contract", "source", "both"])
def test_new_epoch_conflict_does_not_validate_old_map_as_current(tmp_path, changed):
    store, sid, jid, original = initial(tmp_path)
    preserved = deepcopy(original)
    store.finish(jid, sid, "needs_review")
    # Simulate the existing host's explicit owner-edit reconciliation, not a
    # model-selected relaxation or a mutation performed by the conflict service.
    instruction = (
        "Remove the width requirement; retain height."
        if changed != "source"
        else "Review height ownership."
    )
    next_jid = store.enqueue(sid, OWNER, {"instruction": instruction})
    assert store.claim()
    current = contract(keep_width=changed == "source")
    store.update(
        sid,
        contract=current.model_dump(),
        instructions=[] if changed == "contract" else [instruction],
    )
    before = store.session_record(sid, OWNER)
    reported = report_height(store, sid, next_jid, current)
    assert_requirement_only(reported)
    after = store.session_record(sid, OWNER)
    assert after["intent_attribution_history"] == [preserved]
    assert original == preserved
    assert preserved["attribution"]["facts"][0]["requirement_ids"] == ["width"]
    assert preserved["draft_hash"] == digest(
        {key: value for key, value in preserved.items() if key != "draft_hash"}
    )
    assert reported["contract_hash"] == current.fingerprint()
    assert reported["source_text"] == request_text(after)
    for field in ("contract", "selected_revision", "accepted_revision", "design_plan"):
        assert after.get(field) == before.get(field)
    assert not store.revisions(sid)


@pytest.mark.parametrize("apply_answer_to_source", [False, True])
def test_answered_unresolved_facts_do_not_reopen_or_break_a_later_question(
    tmp_path, apply_answer_to_source
):
    store, sid, jid, original = initial(tmp_path, ambiguous=True)
    store.finish(jid, sid, "needs_input")
    next_jid = store.enqueue(
        sid,
        OWNER,
        {
            "intent_resolution": {
                "expected_version": original["version"],
                "draft_hash": original["draft_hash"],
                "answers": [{"issue_id": "width_subject", "answer": RAW_ANSWER}],
            }
        },
    )
    queued = store.claim()
    assert queued and queued["id"] == next_jid
    answered = deepcopy(read_attribution(store, sid, OWNER)["record"])
    if apply_answer_to_source:
        store.update(sid, instructions=[queued["instruction"]])
    reported = report_height(store, sid, next_jid, contract())
    assert_requirement_only(reported)
    saved = store.session_record(sid, OWNER)
    assert saved["intent_attribution_history"] == [original, answered]
    assert answered["status"] == "answered"
    assert answered["resolutions"] == [{"issue_id": "width_subject", "answer": RAW_ANSWER}]
    assert answered["attribution"] == original["attribution"]
    assert original["attribution"]["facts"][0]["subject_id"] is None
    assert not any("confirm" in key or "approv" in key for key in reported)
    assert saved["contract"] == contract().model_dump()


def test_same_epoch_conflict_retains_current_input_facts(tmp_path):
    store, sid, jid, original = initial(tmp_path)
    reported = record_conflict(
        store,
        sid,
        OWNER,
        jid,
        [width_issue()],
        contract(),
        SOURCE,
        store.edit_basis(sid, OWNER),
    )
    for field in ("subjects", "facts", "requirement_subjects"):
        assert reported["attribution"][field] == original["attribution"][field]
    assert reported["attribution"]["issues"] == [width_issue().model_dump()]


def test_same_epoch_open_questions_remain_open_when_another_issue_is_added(tmp_path):
    store, sid, jid, original = initial(tmp_path, ambiguous=True)
    reported = report_height(store, sid, jid, contract())
    assert reported["attribution"]["facts"] == original["attribution"]["facts"]
    assert reported["attribution"]["issues"] == [
        width_issue().model_dump(),
        height_issue().model_dump(),
    ]
    assert reported["resolutions"] == []


def test_new_epoch_does_not_guess_replacements_for_stale_incoming_fact_links(tmp_path):
    store, sid, jid, original = initial(tmp_path)
    store.update(sid, instructions=["Review the current height requirement."])
    incoming = height_issue().model_copy(update={"fact_ids": ["input_width"]})
    before = deepcopy(store.session_record(sid, OWNER))
    with pytest.raises(WorkspaceError, match="current requirement IDs"):
        record_conflict(
            store,
            sid,
            OWNER,
            jid,
            [incoming],
            contract(),
            request_text(before),
            store.edit_basis(sid, OWNER),
        )
    assert store.session_record(sid, OWNER) == before
    assert read_attribution(store, sid, OWNER)["record"] == original
