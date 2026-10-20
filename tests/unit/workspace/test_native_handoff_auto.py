"""Real Store claim boundaries with synthetic preflight facts, never CAD proof."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.native_handoff_auto import claim_automatic_preparation
from forge_mind.workspace.native_handoff_budget import HandoffBudget
from forge_mind.workspace.store import Store

RID = "a" * 32


def document(store, sid):
    with store.connect() as con:
        row = con.execute("SELECT document FROM sessions WHERE id=?", (sid,)).fetchone()
    return json.loads(row[0])


@pytest.fixture
def case(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Retained source duty", None, "native_operations")["id"]
    store.enqueue(sid, "owner", {"max_seconds": 300})
    job = store.claim()
    assert job is not None
    # Consume the initial owner input before simulating this job's CAD publication.
    # Completing a job alone does not apply its pending source or permit a new edit.
    store.apply_input_contract(
        sid,
        "owner",
        job["id"],
        job["input_application_id"],
        {
            "requirements": [
                {"id": "r1", "text": "Retained source duty", "source_quote": "Retained source duty"}
            ]
        },
    )
    assert document(store, sid)["contract_application"]["pending"] is None
    store.update(sid, selected_revision=RID, plan_hash="plan", usage=[{"original": True}])
    context = {
        "selected": {"id": RID},
        "handoff_preflight": {
            "revision_id": RID,
            "plan_hash": "plan",
            "work_reasons": ["Existing missing preparation reason"],
            "print_preparation": {"required": True, "supported": True, "status": "missing"},
        },
        "delivery_guidance": {
            "status": "missing",
            "basis_hash": "basis",
            "basis": {"revision_id": RID, "plan_hash": "plan"},
            "facts": {"physical_readiness": "unverified"},
        },
    }
    return store, sid, job["id"], context


def claim(case, **overrides):
    store, sid, jid, context = case
    arguments = {
        "context": context,
        "published_revision_id": RID,
        "index": 1,
        "schedule": HandoffBudget(100, 100, 8),
        "phase": "construction",
        "now": 10.0,
        "enabled": True,
    }
    arguments.update(overrides)
    return claim_automatic_preparation(store, sid, "owner", jid, **arguments)


def test_claim_changes_only_durable_marker_and_returns_detached_provenance(case):
    store, sid, jid, context = case
    original_context = deepcopy(context)
    before = document(store, sid)
    with store.connect() as con:
        original_job = dict(con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())
    marker = claim(case)
    assert marker == {
        "job_id": jid,
        "sid": sid,
        "revision_id": RID,
        "basis_hash": "basis",
        "plan_hash": "plan",
        "action_index": 1,
        "phase": "construction",
        "origin": "host",
        "claimed_at": 10.0,
    }
    after = document(store, sid)
    assert after.pop("automatic_handoff_attempt") == marker
    assert after == before and context == original_context
    with store.connect() as con:
        assert dict(con.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()) == original_job
    marker["basis_hash"] = "caller mutation"
    assert document(store, sid)["automatic_handoff_attempt"]["basis_hash"] == "basis"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("selected", "id"), "other"),
        (("handoff_preflight", "revision_id"), "other"),
        (("handoff_preflight", "plan_hash"), None),
        (("handoff_preflight", "work_reasons"), []),
        (("handoff_preflight", "work_reasons"), ["print", "incomplete part"]),
        (("handoff_preflight", "print_preparation", "required"), False),
        (("handoff_preflight", "print_preparation", "required"), 1),
        (("handoff_preflight", "print_preparation", "supported"), False),
        (("handoff_preflight", "print_preparation", "supported"), None),
        (("handoff_preflight", "print_preparation", "status"), "current"),
        (("handoff_preflight", "print_preparation", "status"), "conversion_required"),
        (("delivery_guidance", "status"), "unavailable"),
        (("delivery_guidance", "basis_hash"), None),
        (("delivery_guidance", "basis", "revision_id"), "other"),
        (("delivery_guidance", "basis", "plan_hash"), "other"),
        (("failed_submission",), {"error": "repair source"}),
        (("failed_attempt",), {"measurement": {"geometry_valid": False}}),
        (("plan_draft",), {"status": "stale_draft"}),
        (("proposal",), {"revision_id": RID}),
        (("proposed_contract",), {"pending": True}),
        (("request_attribution",), {"status": "needs_input"}),
    ],
)
def test_ineligible_author_state_never_claims(case, path, value):
    store, sid, _, context = case
    target = context
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    before = document(store, sid)
    assert claim(case) is None
    assert document(store, sid) == before


@pytest.mark.parametrize(
    "overrides",
    [
        {"enabled": False},
        {"schedule": None},
        {"phase": "handoff"},
        {"phase": "instructions"},
        {"published_revision_id": None},
        {"published_revision_id": "other"},
        {"index": -1},
        {"index": True},
        {"now": float("nan")},
        {"now": 60.0},
        {"now": 80.0},
        {"schedule": HandoffBudget(None, None, 2), "index": 0},
        {"schedule": HandoffBudget(None, None, 3), "index": 1},
        {"schedule": HandoffBudget(None, None, 8), "index": 6},
        # No construction reserve means the next worst-case slot is instructions.
        {"schedule": HandoffBudget(100, None, 8)},
        {"schedule": HandoffBudget(100, 0, 8)},
    ],
)
def test_original_phase_action_and_time_bounds_are_not_extended(case, overrides):
    store, sid, _, _ = case
    assert claim(case, **overrides) is None
    assert "automatic_handoff_attempt" not in document(store, sid)


def test_exact_three_slots_and_unbounded_declared_time_remain_usable(case):
    assert claim(case, index=0, schedule=HandoffBudget(None, None, 3)) is not None


def test_absent_action_and_wall_limits_are_not_invented(case):
    assert claim(case, schedule=HandoffBudget(None, None, None)) is not None


@pytest.mark.parametrize(
    "fields",
    [
        {"selected_revision": "other"},
        {"plan_hash": "changed"},
        {"job_id": "other"},
        {"proposal": {"revision_id": RID}},
        {"intent_attribution": {"status": "needs_input"}},
    ],
)
def test_live_authority_changes_after_preflight_cannot_claim(case, fields):
    store, sid, _, _ = case
    store.update(sid, **fields)
    before = document(store, sid)
    assert claim(case) is None
    assert document(store, sid) == before


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("cancel", 1),
        ("state", "queued"),
        ("state", "complete"),
        ("worker_pid", -1),
        ("worker_created", -1),
        ("worker_host", "another worker host"),
    ],
)
def test_existing_owned_running_writer_guard_remains_authoritative(case, column, value):
    store, sid, jid, _ = case
    with store.connect() as con:
        con.execute(f"UPDATE jobs SET {column}=? WHERE id=?", (value, jid))
    assert claim(case) is None
    assert "automatic_handoff_attempt" not in document(store, sid)


def test_wrong_owner_cannot_claim_another_session(case):
    store, sid, jid, context = case
    assert (
        claim_automatic_preparation(
            store,
            sid,
            "other owner",
            jid,
            context=context,
            published_revision_id=RID,
            index=0,
            schedule=HandoffBudget(None, None, 3),
            phase="construction",
            now=1.0,
            enabled=True,
        )
        is None
    )
    assert "automatic_handoff_attempt" not in document(store, sid)


def test_live_unadmitted_draft_is_checked_inside_claim_transaction(case):
    store, sid, _, _ = case
    with store.connect() as con:
        con.execute(
            "INSERT INTO plan_drafts(session,version,basis,content,raw_json,validation,content_hash) "
            "VALUES (?,1,'{}','{}','{}','{}',?)",
            (sid, "d" * 64),
        )
    assert claim(case) is None
    assert "automatic_handoff_attempt" not in document(store, sid)


def test_changed_acceptance_uses_existing_writer_guard(case):
    store, sid, jid, _ = case
    with store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        )
        payload["preview_acceptance_identity"] = store._preview_acceptance_identity(
            document(store, sid)
        )
        con.execute("UPDATE jobs SET payload=? WHERE id=?", (json.dumps(payload), jid))
    store.update(sid, accepted_revision="changed")
    assert claim(case) is None


def test_recorded_current_job_authority_conflict_never_claims(case):
    store, sid, jid, _ = case
    store.update(sid, authority_conflict_job_id=jid)
    assert claim(case) is None
    assert "automatic_handoff_attempt" not in document(store, sid)


def test_one_attempt_survives_event_truncation_edits_and_new_service(case):
    store, sid, jid, context = case
    first = claim(case)
    assert first is not None
    # A blocked/error outcome never resets the marker. Later edits get the manual path.
    store.update(sid, events=[], selected_revision="b" * 32, plan_hash="changed")
    context["selected"]["id"] = "b" * 32
    context["handoff_preflight"].update(revision_id="b" * 32, plan_hash="changed")
    context["delivery_guidance"].update(
        basis_hash="new print/guidance basis",
        basis={"revision_id": "b" * 32, "plan_hash": "changed"},
    )
    reopened = WorkspaceService(Store(store.root), Executor("trusted_local"))
    assert claim((reopened.store, sid, jid, context), published_revision_id="b" * 32) is None
    assert document(reopened.store, sid)["automatic_handoff_attempt"] == first


def test_fresh_job_can_replace_the_single_current_job_marker(case):
    store, sid, jid, context = case
    first = claim(case)
    with store.connect() as con:
        con.execute("UPDATE jobs SET state='complete' WHERE id=?", (jid,))
    store.enqueue(sid, "owner", {"max_seconds": 300})
    next_job = store.claim()
    assert next_job is not None
    second = claim((store, sid, next_job["id"], context))
    assert second is not None and first is not None
    assert second["job_id"] != first["job_id"]
    assert document(store, sid)["automatic_handoff_attempt"] == second


def test_concurrent_service_claims_do_not_duplicate_automatic_work(case):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: claim(case), range(2)))
    assert sum(result is not None for result in results) == 1


def test_database_failure_is_not_disguised_as_a_claim_or_success(case, monkeypatch):
    store, _, _, _ = case

    def unavailable():
        raise sqlite3.OperationalError("database unavailable")

    monkeypatch.setattr(store, "connect", unavailable)
    with pytest.raises(sqlite3.OperationalError, match="database unavailable"):
        claim(case)
