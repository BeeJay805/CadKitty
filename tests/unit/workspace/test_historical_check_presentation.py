"""Archived failed predicates stay readable without becoming current evidence."""

import hashlib
import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_customer_proposals import MEMBERS, proposal_context, readme, seal

from forge_mind.workspace.customer_delivery import customer_delivery_view
from forge_mind.workspace.plan_history import (
    historical_check_context,
    plan_history_entry,
    request_hash,
)
from forge_mind.workspace.store import Store


def archived_case(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Retain the old failed scope", {})["id"]
    old_request = {"tool": "hole_pattern", "units": "mm", "body": {"body_index": 0}}
    old_check = {"id": "camera_bore", "request": old_request}
    revision = {
        "id": "a" * 32,
        "contract": {},
        "measurement": {"artifact_hashes": {"part.step": "f" * 64}, "engine_fingerprint": "e" * 64},
    }
    store.add_revision(sid, revision)
    receipt = {
        "id": "b" * 32,
        "request": old_request,
        "revision_id": revision["id"],
        "contract": {},
        "artifact_hashes": revision["measurement"]["artifact_hashes"],
        "engine_fingerprint": "e" * 64,
        "result": {
            "status": "fail",
            "algorithm_scope": "Full-body-span cylindrical clearance only; no threads.",
            "results": {"obstruction_mm3": 50.0},
        },
    }
    encoded = json.dumps(receipt)
    with store.connect() as con:
        con.execute("CREATE TABLE product_checks(id TEXT,session TEXT,document TEXT,digest TEXT)")
        con.execute(
            "INSERT INTO product_checks VALUES(?,?,?,?)",
            (
                receipt["id"],
                sid,
                encoded,
                hashlib.sha256(encoded.encode()).hexdigest(),
            ),
        )
        old_state = {
            "design_plan": {"subsystems": [{"id": "camera", "checks": [old_check]}]},
            "plan_version": 1,
            "plan_hash": "1" * 64,
            "prepared_check_cache": {"camera_bore": {"receipt_id": receipt["id"]}},
        }
        history = plan_history_entry(con, sid, old_state, head_revision_id=revision["id"])
    current = {
        "id": "camera_bore",
        "request": {
            "tool": "axial_cavity",
            "units": "mm",
            "depth_interval": {"low": 0.0, "high": 4.0},
        },
        "criterion_amendment": {
            "previous_request_hash": request_hash(old_request),
            "reason": "The old predicate asks for through clearance; the supplied hole is blind.",
            "basis": "Owner supplied 4 mm usable blind depth.",
        },
    }
    state = {
        "design_plan": {"subsystems": [{"id": "camera", "checks": [current]}]},
        "design_plan_history": [history],
    }
    return store, sid, state


@pytest.mark.parametrize("damage", [None, "pointer_digest", "request", "source", "receipt_digest"])
def test_historical_projection_rechecks_receipt_and_source_identity(tmp_path, damage):
    store, sid, state = archived_case(tmp_path)
    if damage == "pointer_digest":
        state["design_plan_history"][0]["prior_check_receipts"][0]["receipt_digest"] = "0" * 64
    elif damage == "request":
        state["design_plan_history"][0]["plan"]["subsystems"][0]["checks"][0]["request"]["tool"] = (
            "changed"
        )
    with store.connect() as con:
        if damage == "source":
            source = json.loads(
                con.execute("SELECT document FROM revisions WHERE session=?", (sid,)).fetchone()[0]
            )
            source["measurement"]["engine_fingerprint"] = "d" * 64
            con.execute(
                "UPDATE revisions SET document=? WHERE session=?", (json.dumps(source), sid)
            )
        elif damage == "receipt_digest":
            con.execute("UPDATE product_checks SET digest=? WHERE session=?", ("0" * 64, sid))
        rows = historical_check_context(con, sid, state)
    assert len(rows) == 1
    assert rows[0]["historical_status"] == ("fail" if damage is None else "unverified")
    assert rows[0]["verification"] == (
        "verified_historical" if damage is None else "unverified_historical"
    )
    assert "never current completion proof" in rows[0]["scope"]
    assert "result" not in rows[0]


def test_readable_old_failure_does_not_change_current_corrected_nominal_pass(tmp_path):
    store, sid, state = archived_case(tmp_path)
    with store.connect() as con:
        historical = historical_check_context(con, sid, state)
    value = proposal_context()
    for finding in value["facts"]["requirement_findings"]:
        finding["status"] = "pass"
    value["facts"]["functional_interfaces"] = {"rows": []}
    value["facts"]["historical_check_amendments"] = historical
    seal(value)
    before = deepcopy(value)
    view = customer_delivery_view(value, members=MEMBERS)
    text = readme(value, MEMBERS)
    assert view["checks"]["failed"] == []
    assert "Historical check camera_bore (hole_pattern): fail; verified_historical." in text
    assert "Full-body-span cylindrical clearance only; no threads." in text
    assert "Owner supplied 4 mm usable blind depth." in text
    assert "They do not change current check results" in text
    assert "**Saved checks failed." not in text
    assert view["physical_readiness"] == "unverified"
    assert value == before


def test_unamended_current_check_does_not_change_legacy_guidance_facts(tmp_path):
    store, sid, state = archived_case(tmp_path)
    state["design_plan"]["subsystems"][0]["checks"][0].pop("criterion_amendment")
    with store.connect() as con:
        assert historical_check_context(con, sid, state) == []


def test_multiple_amendments_keep_each_historical_correction_with_its_own_predecessor(tmp_path):
    store, sid, state = archived_case(tmp_path)
    intermediate = deepcopy(state["design_plan"]["subsystems"][0]["checks"][0])
    first_reason = intermediate["criterion_amendment"]["reason"]
    state["design_plan_history"].append(
        {
            "plan": {"subsystems": [{"id": "camera", "checks": [intermediate]}]},
        }
    )
    current = state["design_plan"]["subsystems"][0]["checks"][0]
    current["request"]["depth_interval"]["high"] = 3.0
    current["criterion_amendment"] = {
        "previous_request_hash": request_hash(intermediate["request"]),
        "reason": "Measure the explicitly specified shorter engagement interval.",
        "basis": "Owner supplied 3 mm nominal screw engagement.",
    }
    with store.connect() as con:
        rows = historical_check_context(con, sid, state)
    original = next(row for row in rows if row["tool"] == "hole_pattern")
    second = next(row for row in rows if row["tool"] == "axial_cavity")
    assert original["historical_status"] == "fail"
    assert original["amendment"]["reason"] == first_reason
    assert original["amendment"]["previous_request_hash"] == original["request_hash"]
    assert original["successor_request_hash"] == request_hash(intermediate["request"])
    assert second["amendment"] == current["criterion_amendment"]
    assert second["amendment"]["previous_request_hash"] == second["request_hash"]
