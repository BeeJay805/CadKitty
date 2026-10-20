"""Criterion edits retain exact predecessors; explanations confer no proof."""

from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.check_amendment import (
    CriterionAmendment,
    request_hash,
    same_criterion,
    validate_check_amendments,
)
from forge_mind.workspace.completion import validate_plan
from forge_mind.workspace.completion_models import PreparedCheck
from forge_mind.workspace.completion_readonly import _hash, prepared_binding
from forge_mind.workspace.product_tools import Holes

REQUEST = {
    "tool": "hole_pattern",
    "units": "mm",
    "body": {"occurrence_path": "receiver"},
    "axis": [0.0, 0.0, 1.0],
    "holes": [{"center": [0.0, 0.0, 0.0], "diameter": 4.0}],
}
CAVITY = {
    "tool": "axial_cavity",
    "units": "mm",
    "body": {"occurrence_path": "receiver"},
    "entry_point": [0.0, 0.0, 0.0],
    "axis": [0.0, 0.0, 1.0],
    "diameter": 4.0,
    "depth_interval": {"low": 0.0, "high": 4.0},
}
CONTRACT = {"requirements": [{"id": "receiver_depth"}]}


def plan():
    return {
        "subsystems": [
            {
                "id": "attachment",
                "name": "Attachment",
                "requirement_ids": ["receiver_depth"],
                "occurrence_paths": ["receiver"],
                "checks": [{"id": "receiver_check", "request": copy.deepcopy(REQUEST)}],
            }
        ],
        "requirement_dispositions": [
            {
                "requirement_id": "receiver_depth",
                "check_ids": ["receiver_check"],
                "physical_validation": ["Actual threads and attachment strength remain unverified"],
            }
        ],
    }


def check(value):
    return value["subsystems"][0]["checks"][0]


def correct(previous):
    current = copy.deepcopy(previous)
    check(current)["request"] = copy.deepcopy(CAVITY)
    check(current)["criterion_amendment"] = {
        "previous_request_hash": request_hash(check(previous)["request"]),
        "reason": "The earlier probe required a through hole; this stated criterion is finite depth",
        "basis": "Receiver depth requirement: check the explicit 4 mm usable axial interval",
    }
    return current


def test_legacy_prepared_check_serialization_keeps_amendment_absent():
    prepared = PreparedCheck(id="receiver_check", request=copy.deepcopy(REQUEST))
    old = {
        "id": "receiver_check",
        "request": Holes.model_validate(REQUEST).model_dump(),
        "reference_revision_id": None,
    }
    assert prepared.model_dump() == old
    assert request_hash(prepared.model_dump()) == request_hash(old)
    assert "criterion_amendment" not in prepared.model_dump_json()


def test_hash_matches_existing_exact_prepared_request_binding():
    assert request_hash(REQUEST) == _hash(REQUEST)
    assert request_hash(CAVITY) == _hash(CAVITY)


def fastener_without_optional_default():
    return {
        "tool": "fastener_stack",
        "units": "mm",
        "screw_length": {"low": 12.0, "high": 12.0},
        "grip_layers": [{"low": 5.0, "high": 5.0}],
        "required_engagement": 3.0,
        "available_depth": {"low": 8.0, "high": 8.0},
    }


def test_numeric_default_roundtrip_is_same_criterion_with_different_exact_hash():
    original = PreparedCheck(id="stack", request=fastener_without_optional_default())
    before = original.model_dump()["request"]
    after = PreparedCheck.model_validate(original.model_dump()).model_dump()["request"]
    assert isinstance(before["minimum_tip_clearance"], int)
    assert isinstance(after["minimum_tip_clearance"], float)
    assert same_criterion(before, after)
    assert request_hash(before) != request_hash(after)
    assert request_hash(before) == _hash(before)
    assert request_hash(after) == _hash(after)
    old = plan()
    check(old)["request"] = before
    new = copy.deepcopy(old)
    check(new)["request"] = after
    new["subsystems"][0]["name"] = "Unrelated prose correction"
    validate_plan(new, CONTRACT, old)


def test_omitted_and_explicit_defaults_have_same_meaning_without_rewriting_receipts():
    absent = fastener_without_optional_default()
    explicit = {**absent, "minimum_tip_clearance": 0.0}
    assert same_criterion(absent, explicit)
    assert request_hash(absent) != request_hash(explicit)
    old = plan()
    check(old)["request"] = absent
    new = copy.deepcopy(old)
    check(new)["request"] = explicit
    validate_plan(new, CONTRACT, old)


def test_real_numeric_change_requires_exact_original_hash_despite_default_normalization():
    original = PreparedCheck(id="receiver_check", request=fastener_without_optional_default())
    old = plan()
    before = original.model_dump()["request"]
    check(old)["request"] = before
    new = copy.deepcopy(old)
    normalized = PreparedCheck.model_validate(original.model_dump()).model_dump()["request"]
    check(new)["request"] = copy.deepcopy(normalized)
    check(new)["request"]["required_engagement"] += 1e-10
    assert not same_criterion(before, check(new)["request"])
    with pytest.raises(ValueError, match="explicit criterion_amendment"):
        validate_plan(new, CONTRACT, old)
    check(new)["criterion_amendment"] = {
        "previous_request_hash": request_hash(normalized),
        "reason": "Correct the declared engagement",
        "basis": "Explicit nominal target correction; physical fit remains unverified",
    }
    with pytest.raises(ValueError, match="predecessor request hash"):
        validate_plan(new, CONTRACT, old)
    check(new)["criterion_amendment"]["previous_request_hash"] = request_hash(before)
    validate_plan(new, CONTRACT, old)


def test_unchanged_old_plan_needs_no_migration():
    old = plan()
    validate_plan(old, CONTRACT)
    validate_plan(copy.deepcopy(old), CONTRACT, old)
    validate_check_amendments(old, None)


def test_explicit_correction_preserves_frozen_claim_and_invalidates_old_cache_binding():
    previous = plan()
    current = correct(previous)
    prior_bytes = copy.deepcopy(previous)
    validate_plan(current, CONTRACT, previous)
    assert previous == prior_bytes
    assert current["requirement_dispositions"] == previous["requirement_dispositions"]
    assert current["subsystems"][0]["requirement_ids"] == ["receiver_depth"]
    snapshot = {"revision_id": "a" * 32, "plan_hash": "plan"}
    assert prepared_binding(snapshot, check(previous)) != prepared_binding(snapshot, check(current))
    # An authored explanation has no place to supply a passing verdict.
    assert "status" not in check(current)


def test_silent_request_replacement_is_rejected_at_shared_plan_validation():
    previous = plan()
    current = correct(previous)
    del check(current)["criterion_amendment"]
    with pytest.raises(ValueError, match="explicit criterion_amendment"):
        validate_plan(current, CONTRACT, previous)


def test_wrong_predecessor_is_rejected():
    previous = plan()
    current = correct(previous)
    check(current)["criterion_amendment"]["previous_request_hash"] = "0" * 64
    with pytest.raises(ValueError, match="predecessor request hash"):
        validate_plan(current, CONTRACT, previous)


@pytest.mark.parametrize("field", ["reason", "basis"])
def test_empty_reason_or_basis_is_rejected(field):
    previous = plan()
    current = correct(previous)
    check(current)["criterion_amendment"][field] = " \n "
    with pytest.raises(ValidationError, match="non-whitespace"):
        validate_plan(current, CONTRACT, previous)


@pytest.mark.parametrize("change", ["remove", "reason", "basis", "hash"])
def test_same_request_cannot_rewrite_its_prior_amendment(change):
    previous = correct(plan())
    current = copy.deepcopy(previous)
    if change == "remove":
        del check(current)["criterion_amendment"]
    elif change == "hash":
        check(current)["criterion_amendment"]["previous_request_hash"] = "b" * 64
    else:
        check(current)["criterion_amendment"][change] = "New unsupported historical explanation"
    with pytest.raises(ValueError, match="Retain criterion amendment provenance"):
        validate_plan(current, CONTRACT, previous)


def test_next_correction_must_bind_the_current_request_and_preserve_prior_source():
    first = correct(plan())
    second = copy.deepcopy(first)
    check(second)["request"]["depth_interval"]["high"] = 5.0
    with pytest.raises(ValueError, match="predecessor request hash"):
        validate_plan(second, CONTRACT, first)
    check(second)["criterion_amendment"] = {
        "previous_request_hash": request_hash(check(first)["request"]),
        "reason": "Correct the earlier finite interval target",
        "basis": "Explicit updated depth fact; whole requirement remains unverified",
    }
    validate_plan(second, CONTRACT, first)
    assert check(first)["request"]["depth_interval"]["high"] == 4
    assert check(first)["criterion_amendment"]["previous_request_hash"] == request_hash(REQUEST)


def test_revision_rebinding_is_remeasurement_not_a_criterion_change():
    previous = plan()
    check(previous)["reference_revision_id"] = "a" * 32
    current = copy.deepcopy(previous)
    check(current)["reference_revision_id"] = "b" * 32
    validate_plan(current, CONTRACT, previous)


@pytest.mark.parametrize("previous", [None, {}])
def test_new_check_cannot_claim_unknown_same_id_predecessor(previous):
    current = correct(plan())
    with pytest.raises(ValueError, match="unknown same-ID"):
        validate_check_amendments(current, previous)


def test_saved_amended_plan_still_loads_and_reads_without_migration():
    current = correct(plan())
    PreparedCheck.model_validate(check(current))
    validate_plan(current, CONTRACT)
    with pytest.raises(ValueError, match="unknown same-ID"):
        validate_plan(current, CONTRACT, {})


def test_amendment_cannot_remove_assigned_requirement_or_prior_check():
    previous = plan()
    current = correct(previous)
    current["subsystems"][0]["requirement_ids"] = []
    with pytest.raises(ValueError, match="assigned frozen requirements"):
        validate_plan(current, CONTRACT, previous)
    current = correct(previous)
    current["subsystems"][0]["checks"] = []
    with pytest.raises(ValueError):
        validate_plan(current, CONTRACT, previous)


def test_amendment_is_closed_and_cannot_assert_waiver_or_result():
    payload = correct(plan())["subsystems"][0]["checks"][0]
    for field in ("status", "waived", "superseded_pass"):
        modified = copy.deepcopy(payload)
        modified["criterion_amendment"][field] = True
        with pytest.raises(ValidationError, match="Extra inputs"):
            PreparedCheck.model_validate(modified)
    amendment = CriterionAmendment.model_validate(payload["criterion_amendment"])
    assert set(amendment.model_dump()) == {"previous_request_hash", "reason", "basis"}
