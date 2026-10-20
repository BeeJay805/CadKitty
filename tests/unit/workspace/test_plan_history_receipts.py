"""Old exploratory failures remain attributable after a plan amendment."""

import hashlib
import json

import pytest

from forge_mind.workspace.plan_history import plan_history_entry, request_hash
from forge_mind.workspace.store import Store


@pytest.mark.parametrize("pointer", ["cache", "checkpoint", "report"])
@pytest.mark.parametrize("damage", [None, "digest", "request", "revision", "engine"])
def test_only_verified_old_request_receipts_enter_history(tmp_path, pointer, damage):
    store = Store(tmp_path)
    sid = store.create("owner", "Keep early failure", {})["id"]
    request = {"tool": "hole_pattern", "body": {"body_index": 0}}
    definition = {"id": "bore", "request": request}
    revision = {
        "id": "a" * 32,
        "contract": {},
        "measurement": {"artifact_hashes": {"part.step": "f" * 64}, "engine_fingerprint": "e" * 64},
    }
    store.add_revision(sid, revision)
    receipt = {
        "id": "b" * 32,
        "request": request,
        "revision_id": revision["id"],
        "contract": {},
        "artifact_hashes": revision["measurement"]["artifact_hashes"],
        "engine_fingerprint": "e" * 64,
        "result": {"status": "fail", "results": {"obstruction_mm3": 50}},
    }
    if damage == "request":
        receipt["request"] = {"tool": "changed"}
    elif damage == "revision":
        receipt["revision_id"] = "c" * 32
    elif damage == "engine":
        receipt["engine_fingerprint"] = "d" * 64
    encoded = json.dumps(receipt)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    with store.connect() as con:
        con.execute("CREATE TABLE product_checks(id TEXT,session TEXT,document TEXT,digest TEXT)")
        con.execute(
            "INSERT INTO product_checks VALUES(?,?,?,?)",
            (
                receipt["id"],
                sid,
                encoded,
                "0" * 64 if damage == "digest" else digest,
            ),
        )
    state = {
        "design_plan": {"subsystems": [{"id": "camera", "checks": [definition]}]},
        "plan_hash": "1" * 64,
        "plan_version": 3,
    }
    if pointer == "cache":
        state["prepared_check_cache"] = {"bore": {"receipt_id": receipt["id"], "binding": "old"}}
    else:
        state["author_checkpoint" if pointer == "checkpoint" else "completion_report"] = {
            "subsystems": [
                {"id": "camera", "checks": [{"id": "bore", "receipt_id": receipt["id"]}]}
            ],
        }
    with store.connect() as con:
        entry = plan_history_entry(con, sid, state, head_revision_id=revision["id"])
    assert entry["plan"] == state["design_plan"]
    if damage:
        assert "prior_check_receipts" not in entry
    else:
        proof = entry["prior_check_receipts"][0]
        assert proof["receipt_id"] == receipt["id"] and proof["receipt_digest"] == digest
        assert proof["request_hash"] == request_hash(request)
        assert "not current completion proof" in proof["verification_scope"]
        assert receipt["id"] in entry["receipt_ids"]


def test_legacy_history_needs_no_receipt_table_or_new_fields(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Legacy", {})["id"]
    state = {"design_plan": {"objective": "Existing intent"}, "plan_version": 1}
    with store.connect() as con:
        entry = plan_history_entry(con, sid, state, head_revision_id=None)
    assert "prior_check_receipts" not in entry
    assert entry["receipt_ids"] == [] and entry["status"] == "unverified"
