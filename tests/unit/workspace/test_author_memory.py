"""Context compaction preserves actionable reads and exact failure diagnostics."""

import copy
import json

from forge_mind.workspace.agent_memory import (
    MAX_FAILED_SUBMISSION_BYTES,
    active_submission,
    recent_context,
    rejected_submission,
)


def test_recent_context_does_not_replay_cached_schema_or_dense_geometry():
    schema = {"schema": {"description": "x" * 50000}}
    failure = {"status": "fail", "measured": 79, "expected": 80, "reason": "Gap too small"}
    history = [
        {"schema_read": "document", "authoring_schema": schema},
        {
            "revision": "failed",
            "measurement": {
                "revision_id": "failed",
                "geometry_valid": False,
                "findings": [failure],
                "errors": ["Cut produced an empty solid"],
                "facts": {"geometry": "x" * 100000},
                "artifact_hashes": {"part.step": "hash"},
            },
            "regressions": ["gap"],
        },
    ]
    before = copy.deepcopy(history)
    rows = recent_context(history, {"document": schema})
    assert len(json.dumps(rows)) < 1000
    assert rows[0] == {"schema_read": "document"}
    assert rows[1]["measurement"]["findings"] == [failure]
    assert rows[1]["measurement"]["errors"] == ["Cut produced an empty solid"]
    assert rows[1]["regressions"] == ["gap"]
    assert history == before


def test_evicted_schema_and_explicit_part_source_reads_remain_available():
    history = [
        {"schema_read": "part", "authoring_schema": {"schema": "needed"}},
        {"part": {"id": "housing", "features": [{"code": "result=b.Box(1,2,3)"}]}},
        {"source_text": {"id": "supplier", "text": "Exact supplier dimensions"}},
    ]
    assert recent_context(history, {}) == history


def test_changed_cached_schema_does_not_replace_actual_observation():
    history = [{"schema_read": "part", "authoring_schema": {"schema": "observed"}}]
    assert recent_context(history, {"part": {"schema": "different"}}) == history


def test_rejected_input_stays_exact_and_scoped_to_current_contract_and_source():
    action = {"kind": "document", "document_json": '{"source": "exact whitespace ☃"}'}
    failure = rejected_submission(action, "candidate", "contract", "invalid feature")
    assert failure["submitted_action"] == action
    assert active_submission(failure, "candidate", "contract") == failure
    assert active_submission(failure, "other candidate", "contract") is None
    assert active_submission(failure, "candidate", "other contract") is None
    assert "artifact_hashes" not in failure and "measurement" not in failure
    assert not failure["geometry_produced"] and not failure["evidence_produced"]


def test_oversized_rejected_input_is_explicitly_unavailable_without_truncated_source():
    action = {"kind": "document", "document_json": "☃" * MAX_FAILED_SUBMISSION_BYTES}
    failure = rejected_submission(action, None, "contract", "invalid feature")
    assert "submitted_action" not in failure
    assert "source was not retained" in failure["unavailable"]
    assert failure["submitted_action_bytes"] > MAX_FAILED_SUBMISSION_BYTES
    assert len(json.dumps(failure)) < 1000
    assert "hash" not in failure and "measurement" not in failure
