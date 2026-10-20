"""Automatic instruction copies retain exact anchors, receipts and explicit reads."""

import hashlib
import json
from copy import deepcopy

import pytest

from forge_mind.workspace.native_stages import deduplicate_context


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def expand(context):
    """Independent pointer walker validates actual final request targets and hashes."""

    def visit(value, active=()):
        if isinstance(value, dict) and set(value) == {"context_reference"}:
            receipt = value["context_reference"]
            path = receipt["json_pointer"]
            assert path not in active, "Context reference cycle"
            target = context
            for token in path.split("/")[1:]:
                key = token.replace("~1", "/").replace("~0", "~")
                target = target[int(key)] if isinstance(target, list) else target[key]
            assert hashlib.sha256(encoded(target).encode()).hexdigest() == receipt["sha256"]
            return visit(target, (*active, path))
        if isinstance(value, dict):
            return {key: visit(item, active) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item, active) for item in value]
        return value

    return visit(context)


def context():
    fact = {
        "id": "part:cover/☃~seat",
        "status": "declared",
        "value": {"text": "Exact saved fact ☃ " * 100},
    }
    record = {
        "basis": {"revision_id": "historical", "artifact_hashes": {"part.step": "saved-hash"}},
        "guidance": {"installation": "Literal saved author words " * 100},
        "guidance_hash": "literal-record-hash",
    }
    guidance = {
        "status": "stale",
        "basis_hash": "current-basis",
        "instruction_catalog": {
            "facts": {fact["id"]: deepcopy(fact)},
            "required_refs": [fact["id"]],
        },
        "instruction_binding": {
            "status": "unbound",
            "facts": [deepcopy(fact)],
            "errors": ["Existing exact binding failure"],
        },
        "record": record,
        "record_provenance": {
            "record_hash": "source-record-receipt",
            "history_status": {"status": "verified"},
            "history": [
                {"record": deepcopy(record), "predecessor_hash": None, "head_hash": "history-head"}
            ],
        },
    }
    return {"delivery_guidance": guidance}


def assert_lossless(before, after):
    assert encoded(expand(after)) == encoded(expand(before))


def test_automatic_internal_copies_reference_full_values_and_keep_stale_metadata():
    original = context()
    frozen = encoded(original)
    result = deduplicate_context(original)
    guidance = result["delivery_guidance"]
    fact = guidance["instruction_binding"]["facts"][0]
    assert (
        fact["context_reference"]["json_pointer"]
        == "/delivery_guidance/instruction_catalog/facts/part:cover~1☃~0seat"
    )
    historical = guidance["record_provenance"]["history"][0]
    assert historical["record"]["context_reference"]["json_pointer"] == "/delivery_guidance/record"
    assert guidance["instruction_catalog"] == original["delivery_guidance"]["instruction_catalog"]
    assert guidance["record"] == original["delivery_guidance"]["record"]
    assert guidance["status"] == "stale" and guidance["basis_hash"] == "current-basis"
    assert guidance["instruction_binding"]["errors"] == ["Existing exact binding failure"]
    assert historical["head_hash"] == "history-head" and historical["predecessor_hash"] is None
    assert guidance["record_provenance"]["record_hash"] == "source-record-receipt"
    assert encoded(original) == frozen
    assert_lossless(original, result)
    assert len(json.dumps(result).encode()) < len(json.dumps(original).encode())
    assert_lossless(original, deduplicate_context(result))


@pytest.mark.parametrize("difference", ["bool_integer", "different_text", "different_basis"])
def test_unequal_instruction_and_historical_values_remain_complete(difference):
    original = context()
    guidance = original["delivery_guidance"]
    catalog = next(iter(guidance["instruction_catalog"]["facts"].values()))
    bound = guidance["instruction_binding"]["facts"][0]
    record = guidance["record"]
    historical = guidance["record_provenance"]["history"][0]["record"]
    if difference == "bool_integer":
        catalog["value"]["flag"] = True
        bound["value"]["flag"] = 1
        record["flag"], historical["flag"] = True, 1
    elif difference == "different_text":
        bound["value"]["text"] += "changed"
        historical["guidance"]["installation"] += "changed"
    else:
        bound["basis_hash"] = "older"
        historical["basis"]["revision_id"] = "different-revision"
    result = deduplicate_context(original)
    actual = result["delivery_guidance"]
    assert actual["instruction_binding"]["facts"] == guidance["instruction_binding"]["facts"]
    assert actual["record_provenance"]["history"] == guidance["record_provenance"]["history"]
    assert_lossless(original, result)


@pytest.mark.parametrize(
    "failure",
    [
        {"status": "fail"},
        {"errors": ["Exact failed witness"]},
        {"nested": {"status": "needs_review"}},
    ],
)
def test_exact_failed_fact_and_record_copies_stay_full(failure):
    original = context()
    guidance = original["delivery_guidance"]
    next(iter(guidance["instruction_catalog"]["facts"].values())).update(deepcopy(failure))
    guidance["instruction_binding"]["facts"][0].update(deepcopy(failure))
    guidance["record"].update(deepcopy(failure))
    guidance["record_provenance"]["history"][0]["record"].update(deepcopy(failure))
    result = deduplicate_context(original)
    assert result == original
    assert_lossless(original, result)


@pytest.mark.parametrize(
    "failure", [{"status": "failed"}, {"errors": ["Failed saved history receipt"]}]
)
def test_failure_marked_history_row_keeps_its_identical_safe_record_complete(failure):
    original = context()
    row = original["delivery_guidance"]["record_provenance"]["history"][0]
    row.update(deepcopy(failure))
    result = deduplicate_context(original)
    assert result["delivery_guidance"]["record_provenance"]["history"][0] == row
    assert "context_reference" in result["delivery_guidance"]["instruction_binding"]["facts"][0]
    assert_lossless(original, result)


def test_explicit_guidance_source_geometry_and_cached_schemas_are_not_rewritten():
    original = context()
    original.update(
        history=[
            {
                "explicit_guidance_read": True,
                "delivery_guidance": deepcopy(original["delivery_guidance"]),
            },
            {"source_document": {"features": [{"source": "Complete explicit source " * 100}]}},
        ],
        selected={
            "document": {
                "features": [{"kind": "python", "source": "Complete selected geometry " * 100}]
            }
        },
        authoritative_schemas={
            "assembly_edit": {"properties": {"exact": "Complete fetched schema " * 100}}
        },
    )
    result = deduplicate_context(original)
    for name in ("history", "selected", "authoritative_schemas"):
        assert result[name] == original[name]
    assert_lossless(original, result)
    original["delivery_guidance"]["explicit_guidance_read"] = True
    assert deduplicate_context(original)["delivery_guidance"] == original["delivery_guidance"]


def test_existing_reference_targets_are_not_used_for_new_indirection():
    original = context()
    original["frozen_contract"] = {"text": "Full immutable authority " * 100}
    existing = {
        "context_reference": {
            "json_pointer": "/frozen_contract",
            "sha256": hashlib.sha256(encoded(original["frozen_contract"]).encode()).hexdigest(),
        }
    }
    guidance = original["delivery_guidance"]
    next(iter(guidance["instruction_catalog"]["facts"].values()))["nested"] = deepcopy(existing)
    guidance["instruction_binding"]["facts"][0]["nested"] = deepcopy(existing)
    guidance["record"]["nested"] = deepcopy(existing)
    guidance["record_provenance"]["history"][0]["record"]["nested"] = deepcopy(existing)
    result = deduplicate_context(original)
    assert "context_reference" not in result["delivery_guidance"]["instruction_binding"]["facts"][0]
    assert (
        "context_reference"
        not in result["delivery_guidance"]["record_provenance"]["history"][0]["record"]
    )
    assert_lossless(original, result)


@pytest.mark.parametrize(
    "guidance",
    [
        None,
        {},
        {"instruction_catalog": {"facts": []}, "instruction_binding": {"facts": {}}},
        {"record": {}, "record_provenance": {"history": [None, {}, {"record": {}}]}},
    ],
)
def test_missing_small_or_malformed_automatic_fields_do_not_invent_references(guidance):
    original = {"delivery_guidance": guidance}
    assert deduplicate_context(original) == original
