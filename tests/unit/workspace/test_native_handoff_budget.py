"""Scheduling keeps supplied budgets, phase boundaries, and immutable policy data."""

from dataclasses import FrozenInstanceError

import pytest

from forge_mind.workspace.native_handoff_budget import HandoffBudget, capture_deadline


@pytest.mark.parametrize(
    "deadline,seconds,expected",
    [
        (2000.0, 1500.0, (1400.0, 1800.0, 2000.0)),
        (1000.0, 600.0, (760.0, 920.0, 1000.0)),
        (500.0, 300.0, (380.0, 460.0, 500.0)),
        (10.0, 0.0, (10.0, 10.0, 10.0)),
        (None, None, (None, None, None)),
        (None, 1500.0, (None, None, None)),
        (1000.0, None, (1000.0, 1000.0, 1000.0)),
    ],
)
def test_phase_deadlines_stay_inside_supplied_author_budget(deadline, seconds, expected):
    budget = HandoffBudget(deadline, seconds, None)
    actual = tuple(budget.deadline(phase) for phase in ("construction", "handoff", "instructions"))
    assert actual == expected
    # Only inference borrows the handoff allowance across the construction cue.
    assert tuple(
        budget.inference_deadline(phase) for phase in ("construction", "handoff", "instructions")
    ) == (expected[1], expected[1], expected[2])


@pytest.mark.parametrize(
    "now,expected",
    [
        (1399.999, "construction"),
        (1400.0, "handoff"),
        (1799.999, "handoff"),
        (1800.0, "instructions"),
        (2000.0, "instructions"),
        (2001.0, "instructions"),
    ],
)
def test_time_advances_phase_at_exact_boundaries(now, expected):
    assert HandoffBudget(2000.0, 1500.0, None).phase(0, now) == expected


@pytest.mark.parametrize(
    "limit,handoff_index,instructions_index",
    [(3, 1, 2), (4, 2, 3), (8, 6, 7), (12, 9, 11), (64, 48, 63), (120, 104, 119)],
)
def test_action_only_budget_reserves_slots_within_existing_limit(
    limit, handoff_index, instructions_index
):
    budget = HandoffBudget(None, None, limit)
    assert budget.phase(handoff_index - 1, 1000000.0) == "construction"
    assert budget.phase(handoff_index, 1000000.0) == "handoff"
    assert budget.phase(instructions_index - 1, 1000000.0) == "handoff"
    assert budget.phase(instructions_index, 1000000.0) == "instructions"
    assert budget.action_limit == limit
    assert budget.deadline("construction") is None
    assert budget.deadline("handoff") is None
    assert budget.deadline("instructions") is None


@pytest.mark.parametrize(
    "limit,expected",
    [(1, ["instructions"]), (2, ["handoff", "instructions"])],
)
def test_tiny_action_limits_do_not_require_extra_iterations(limit, expected):
    budget = HandoffBudget(2000.0, 1500.0, limit)
    assert [budget.phase(index, 0.0) for index in range(limit)] == expected
    assert budget.action_limit == limit


def test_zero_action_budget_does_not_add_an_action():
    budget = HandoffBudget(None, None, 0)
    assert list(range(budget.action_limit)) == []
    assert budget.phase(0, 0.0) == "instructions"


@pytest.mark.parametrize(
    "index,now,expected",
    [(0, 1400.0, "handoff"), (104, 0.0, "handoff"), (0, 1800.0, "instructions")],
)
def test_either_finite_budget_can_advance_phase(index, now, expected):
    assert HandoffBudget(2000.0, 1500.0, 120).phase(index, now) == expected


def test_action_instruction_boundary_overrides_earlier_time_phase():
    assert HandoffBudget(2000.0, 1500.0, 120).phase(119, 1400.0) == "instructions"


@pytest.mark.parametrize(
    "previous,expected", [("handoff", "handoff"), ("instructions", "instructions")]
)
def test_previous_phase_never_moves_back_when_time_or_index_moves_back(previous, expected):
    budget = HandoffBudget(2000.0, 1500.0, 120)
    assert budget.phase(0, 0.0, previous) == expected


def test_prior_handoff_still_advances_to_instructions():
    budget = HandoffBudget(2000.0, 1500.0, None)
    assert budget.phase(0, 1800.0, "handoff") == "instructions"


def test_unlimited_budgets_create_neither_time_nor_action_cutoff():
    budget = HandoffBudget(None, None, None)
    assert budget.phase(1000000, 1000000000.0) == "construction"
    assert budget.phase(1000000, 1000000000.0, "handoff") == "handoff"
    assert budget.phase(1000000, 1000000000.0, "instructions") == "instructions"
    assert budget.action_limit is None


def test_no_author_duration_does_not_invent_an_earlier_time_boundary():
    budget = HandoffBudget(1000.0, None, None)
    assert budget.phase(0, 999.999) == "construction"
    assert budget.phase(0, 1000.0) == "instructions"


@pytest.mark.parametrize("unknown", ["review", "finish", "", "HANDOFF"])
def test_unknown_phase_rejects_even_without_any_finite_budget(unknown):
    budget = HandoffBudget(None, None, None)
    with pytest.raises(ValueError, match="Unknown handoff phase"):
        budget.deadline(unknown)
    with pytest.raises(ValueError, match="Unknown handoff phase"):
        budget.inference_deadline(unknown)
    with pytest.raises(ValueError, match="Unknown handoff phase"):
        budget.phase(0, 0.0, unknown)


def test_policy_is_immutable():
    budget = HandoffBudget(2000.0, 1500.0, 120)
    with pytest.raises(FrozenInstanceError):
        budget.action_limit = 121


@pytest.mark.parametrize(
    "overall,reserve,expected",
    [
        (1800.0, 300.0, 1740.0),
        (1800.0, 600.0, 1740.0),
        (100.0, 25.0, 95.0),
        (100.0, 0.0, 100.0),
        (100.0, None, 100.0),
        (None, 300.0, None),
        (None, None, None),
    ],
)
def test_capture_time_is_reserved_only_within_an_existing_overall_deadline(
    overall, reserve, expected
):
    assert capture_deadline(overall, reserve) == expected


@pytest.mark.parametrize("seconds", [-1.0, float("inf"), float("nan")])
def test_invalid_author_duration_rejects(seconds):
    with pytest.raises(ValueError, match="author_seconds"):
        HandoffBudget(2000.0, seconds, None)


@pytest.mark.parametrize("deadline", [float("inf"), float("nan")])
def test_nonfinite_author_deadline_rejects(deadline):
    with pytest.raises(ValueError, match="author_deadline"):
        HandoffBudget(deadline, 1500.0, None)


def test_negative_action_limit_rejects():
    with pytest.raises(ValueError, match="action_limit"):
        HandoffBudget(None, None, -1)


@pytest.mark.parametrize("reserve", [-1.0, float("inf"), float("nan")])
def test_capture_reserve_cannot_extend_or_corrupt_the_overall_budget(reserve):
    with pytest.raises(ValueError, match="reserved_seconds"):
        capture_deadline(1800.0, reserve)


@pytest.mark.parametrize("deadline", [float("inf"), float("nan")])
def test_nonfinite_overall_deadline_rejects(deadline):
    with pytest.raises(ValueError, match="overall_deadline"):
        capture_deadline(deadline, 300.0)


def test_correction_retains_half_instruction_time_and_last_existing_slot():
    budget = HandoffBudget(2000.0, 1500.0, 90)
    assert budget.correction_deadline() == 1900.0
    assert budget.deadline("handoff") == 1800.0
    assert budget.can_correct(88, 1899.999)
    assert not budget.can_correct(88, 1900.0)
    assert not budget.can_correct(89, 1800.0)
    assert budget.deadline("instructions") == 2000.0
    assert budget.action_limit == 90


@pytest.mark.parametrize("limit", [0, 1])
def test_tiny_limits_cannot_add_a_corrective_action(limit):
    assert not HandoffBudget(2000.0, 1500.0, limit).can_correct(0, 0.0)


def test_correction_reserve_does_not_invent_unlimited_job_cutoff():
    budget = HandoffBudget(None, None, None)
    assert budget.correction_deadline() is None
    assert budget.can_correct(1000000, 1000000000.0)
    assert not HandoffBudget(None, None, 2).can_correct(1, 1000000000.0)
