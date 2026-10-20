"""Durable observations survive intervening reads without relaxing memory limits."""

import copy
import json

from forge_mind.workspace.agent_memory import remember


def test_discarded_geometry_reads_do_not_evict_actionable_receipts():
    receipt = {"engineering_report_receipt": {"id": "report", "revision_id": "revision"}}
    history = [receipt, *({"part": {"id": str(index)}} for index in range(40))]
    original = copy.deepcopy(history)
    assert remember(history) == [receipt]
    assert history == original


def test_retention_window_counts_only_observations_that_survive_projection():
    history = []
    for index in range(40):
        history.extend([{"receipt": {"id": index}}, {"measurement": {"facts": "dense"}}])
    rows = remember(history)
    assert [row["receipt"]["id"] for row in rows] == list(range(8, 40))
    assert remember(rows) == rows


def test_retained_observations_keep_the_existing_total_serialized_limit():
    history = [{"receipt": {"id": index, "detail": "x" * 14000}} for index in range(40)]
    rows = remember(history)
    assert len(json.dumps(rows).encode("utf-8")) <= 64000
    assert rows[-1] == history[-1]
    assert len(rows) <= 32


def test_oversized_result_does_not_erase_independent_receipt():
    receipt = {"id": "report", "revision_id": "revision"}
    rows = remember([{"engineering_review": {"dense": "x" * 90000}, "receipt": receipt}])
    assert rows[0]["receipt"] == receipt
    assert rows[0]["omitted_keys"] == ["engineering_review"]
    assert len(json.dumps(rows).encode("utf-8")) <= 16000
