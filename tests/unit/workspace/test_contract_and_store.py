from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from forge_mind.workspace.measure import ground_contract, merge_contract_edit
from forge_mind.workspace.models import Contract, ContractEdit, Removal, Requirement
from forge_mind.workspace.store import Store, WorkspaceError


def req(**changes: object) -> Requirement:
    values = {
        "id": "width",
        "text": "Width 20 mm",
        "source_quote": "Width 20 mm",
        "metric": "extent_x",
        "value": 20.0,
    }
    values.update(changes)
    return Requirement.model_validate(values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_nonfinite_or_negative_targets_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        req(value=value)


def test_duplicate_requirement_identity_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Contract(requirements=[req(), req()])


def test_model_cannot_invent_a_tolerance_or_target() -> None:
    for wrong in (
        req(value=21),
        req(tolerance=1),
        req(source_quote="fictional source"),
        req(source="assumption"),
    ):
        grounded = ground_contract(Contract(requirements=[wrong]), "Width 20 mm")
        assert grounded.requirements[0].metric == "unmeasured"
    assert (
        ground_contract(Contract(requirements=[req()]), "Width 20 mm").requirements[0].metric
        == "extent_x"
    )


def test_explicit_contract_preserves_a_user_supplied_target() -> None:
    target = req(source="explicit", value=21)
    assert (
        ground_contract(Contract(requirements=[target]), "Width 20 mm").requirements[0].metric
        == "unmeasured"
    )
    assert (
        ground_contract(Contract(requirements=[target]), "Width 20 mm", allow_explicit=True)
        .requirements[0]
        .value
        == 21
    )


def test_owner_scope_and_revision_scope_survive_reopen(tmp_path: Path) -> None:
    store = Store(tmp_path)
    a = store.create("alice", "plate", None)
    b = store.create("bob", "ring", None)
    store.add_revision(a["id"], {"id": "r1", "measurement": {}})
    reopened = Store(tmp_path)
    assert len(reopened.list_sessions("alice")) == 1
    with pytest.raises(WorkspaceError):
        reopened.get(a["id"], "bob")
    with pytest.raises(WorkspaceError):
        reopened.revision(b["id"], "r1")
    with pytest.raises(WorkspaceError):
        reopened.enqueue(a["id"], "bob", {})
    with pytest.raises(sqlite3.IntegrityError):
        reopened.add_revision(a["id"], {"id": "r1", "measurement": {"changed": True}})
    assert reopened.revision(a["id"], "r1")["measurement"] == {}


def test_admission_is_atomic_across_independent_workers(tmp_path: Path) -> None:
    store = Store(tmp_path)
    for i in range(4):
        s = store.create(str(i), "part", None)
        store.enqueue(s["id"], str(i), {"max_seconds": 60})
    with ThreadPoolExecutor(max_workers=4) as pool:
        claims = list(pool.map(lambda _: Store(tmp_path).claim(), range(4)))
    assert sum(c is not None for c in claims) == 1


def test_busy_session_cannot_be_overwritten(tmp_path: Path) -> None:
    store = Store(tmp_path)
    s = store.create("alice", "part", None)
    store.enqueue(s["id"], "alice", {})
    with pytest.raises(WorkspaceError, match="active job"):
        store.enqueue(s["id"], "alice", {})


def test_expired_legacy_job_is_interrupted_not_reauthored(tmp_path: Path) -> None:
    store = Store(tmp_path)
    s = store.create("alice", "part", None)
    store.enqueue(s["id"], "alice", {})
    job = store.claim()
    assert job
    with store.connect() as con:
        con.execute(
            "UPDATE jobs SET deadline=0,worker_pid=NULL,worker_created=NULL,worker_host=NULL "
            "WHERE id=?",
            (job["id"],),
        )
    assert store.claim() is None
    assert store.get(s["id"], "alice")["state"] == "interrupted"


@pytest.mark.parametrize(
    "quote,tolerance,expected",
    [
        ("Width 20 mm", 20, "unmeasured"),
        ("Width 20 +/- 0.1 mm", 0.1, "extent_x"),
        ("Width 20 mm with tolerance 0.1 mm", 0.1, "extent_x"),
        ("Width 20 mm, bores at X +/-30 mm", 30, "unmeasured"),
    ],
)
def test_tolerance_must_be_an_explicit_allowance(
    quote: str, tolerance: float, expected: str
) -> None:
    c = Contract(requirements=[req(source_quote=quote, tolerance=tolerance)])
    assert ground_contract(c, quote).requirements[0].metric == expected


def test_author_cannot_turn_exact_target_into_lower_bound() -> None:
    c = Contract(requirements=[req(comparator="ge", critical=False)])
    result = ground_contract(c, "Width 20 mm").requirements[0]
    assert result.metric == "unmeasured"
    assert result.critical


def test_word_counts_are_grounded() -> None:
    c = Contract(requirements=[req(metric="hole_count", value=4, source_quote="four bores")])
    assert ground_contract(c, "four bores").requirements[0].metric == "hole_count"


def test_inspection_cannot_erase_explicit_edited_targets() -> None:
    previous = Contract(requirements=[req(source="explicit", value=21)])
    proposed = Contract(requirements=[req(value=20)], coverage_notes=["Inspection only"])
    merged = merge_contract_edit(previous, proposed, "Show a cross section")
    assert merged.requirements == previous.requirements
    assert merged.fingerprint() == previous.fingerprint()
    changed = Contract(requirements=[req(value=25, source_quote="Make width 25 mm")])
    assert merge_contract_edit(previous, changed, "Make width 25 mm").requirements[0].value == 25


def test_edit_reconciles_unique_instrument_even_if_model_renames_id() -> None:
    previous = Contract(requirements=[req()])
    proposed = Contract(
        requirements=[req(id="overall_width", value=25, source_quote="Make width 25 mm")]
    )
    merged = merge_contract_edit(previous, proposed, "Make width 25 mm")
    assert len(merged.requirements) == 1
    assert merged.requirements[0].id == "width"
    assert merged.requirements[0].value == 25


def test_explicit_feature_removal_requires_a_new_user_quote() -> None:
    previous = Contract(requirements=[req()])
    proposed = ContractEdit(
        removals=[Removal(requirement_id="width", source_quote="Remove the width constraint")]
    )
    assert not merge_contract_edit(previous, proposed, "Remove the width constraint").requirements
    assert merge_contract_edit(previous, proposed, "Inspect the width constraint").requirements


def test_adding_an_upper_bound_does_not_erase_the_lower_bound() -> None:
    from forge_mind.workspace.measure import _finding

    previous = Contract(requirements=[req(id="minimum_width", comparator="ge", source="explicit")])
    proposed = Contract(
        requirements=[
            req(
                id="maximum_width",
                comparator="le",
                value=30,
                source_quote="Also require width at most 30 mm",
            )
        ]
    )
    merged = merge_contract_edit(previous, proposed, "Also require width at most 30 mm")
    assert [(r.comparator, r.value) for r in merged.requirements] == [("ge", 20), ("le", 30)]
    assert any(_finding(r, 10, "test", "test").status == "fail" for r in merged.requirements)
    assert all(_finding(r, 25, "test", "test").status == "pass" for r in merged.requirements)


def test_queued_cancellation_finishes_and_allows_another_job(tmp_path: Path) -> None:
    store = Store(tmp_path)
    sid = store.create("owner", "part", None)["id"]
    store.enqueue(sid, "owner", {})
    store.cancel(sid, "owner")
    assert store.get(sid, "owner")["state"] == "cancelled"
    assert store.claim() is None
    pending = store.session_record(sid, "owner")["resume_candidate"]["contract_application"][
        "pending"
    ]
    assert pending["mode"] == "full"
    with pytest.raises(WorkspaceError, match="Resume the pending owner input"):
        store.enqueue(sid, "owner", {})
    store.enqueue(sid, "owner", {"resume": True})
    job = store.claim()
    assert job
    assert job["input_application_id"] == pending["input_id"]
    store.cancel(sid, "owner")
    assert store.get(sid, "owner")["state"] == "cancelling"
    store.finish(job["id"], sid, "contract_satisfied")
    assert store.get(sid, "owner")["state"] == "cancelled"
    store.cancel(sid, "owner")
    assert store.get(sid, "owner")["state"] == "cancelled"
