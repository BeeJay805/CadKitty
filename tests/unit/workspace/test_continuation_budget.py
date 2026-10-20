"""Owner answers and resume retain their actual finite job policy, without new defaults."""

import json

import pytest
from tests.unit.workspace.test_intent_attribution_service import OWNER, resolution, terminal_pending

from forge_mind.workspace.store import Store


def finite_question(tmp_path):
    store, sid, job, record = terminal_pending(tmp_path, delivery_requested=True)
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
        )
        payload.update(max_seconds=600, max_actions=80)
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
    return store, sid, record


@pytest.mark.parametrize(
    "override",
    [{}, {"max_seconds": None, "max_actions": None}, {"max_seconds": 45, "max_actions": 3}],
)
def test_answer_and_resume_keep_actual_policy_when_omitted_or_null(tmp_path, override):
    store, sid, record = finite_question(tmp_path)
    expected = {"max_seconds": 600, "max_actions": 80}
    expected.update({key: value for key, value in override.items() if value is not None})
    store.enqueue(sid, OWNER, {"intent_resolution": resolution(record), **override})
    answered = store.claim()
    assert {key: answered[key] for key in expected} == expected
    original = store.get(sid, OWNER)["contract_application"]["pending"]
    store.finish(answered["id"], sid, "needs_review")
    store.enqueue(sid, OWNER, {"resume": True, "max_seconds": None, "max_actions": None})
    resumed = store.claim()
    assert {key: resumed[key] for key in expected} == expected
    assert store.get(sid, OWNER)["contract_application"]["pending"] == original


def test_answer_and_resume_overrides_still_obey_current_server_caps(tmp_path):
    store, sid, record = finite_question(tmp_path)
    store.update(sid, customer_hosted_limits={"max_seconds": 200, "max_actions": 20})
    store.enqueue(
        sid,
        OWNER,
        {"intent_resolution": resolution(record), "max_seconds": 1000, "max_actions": 999},
    )
    answered = store.claim()
    assert (answered["max_seconds"], answered["max_actions"]) == (200, 20)
    store.finish(answered["id"], sid, "needs_review")
    store.update(sid, customer_hosted_limits={"max_seconds": 100, "max_actions": 10})
    store.enqueue(sid, OWNER, {"resume": True})
    resumed = store.claim()
    assert (resumed["max_seconds"], resumed["max_actions"]) == (100, 10)


def test_uncapped_prior_policy_does_not_gain_a_new_default(tmp_path):
    store, sid, _, record = terminal_pending(tmp_path)
    store.enqueue(sid, OWNER, {"intent_resolution": resolution(record)})
    answered = store.claim()
    assert answered.get("max_seconds") is None and answered.get("max_actions") is None


def test_new_explicit_edit_does_not_inherit_previous_work_policy(tmp_path):
    store = Store(tmp_path)
    sid = store.create(OWNER, "A plate", {"requirements": []})["id"]
    jid = store.enqueue(sid, OWNER, {"max_seconds": 600, "max_actions": 80})
    assert store.claim()["id"] == jid
    store.finish(jid, sid, "needs_review")
    store.enqueue(
        sid,
        OWNER,
        {"instruction": "Make the plate wider", "max_seconds": None, "max_actions": None},
    )
    edited = store.claim()
    assert edited.get("max_seconds") is None and edited.get("max_actions") is None
