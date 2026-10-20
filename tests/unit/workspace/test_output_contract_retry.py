"""Output correction shares existing attempt/time limits and never repeats an action."""

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.llm.output_contract import OutputContractError
from forge_mind.workspace.model_retry import CAPACITY_ERROR, run_model_call


def rejected(stage="json_decode", *, retryable=True):
    return OutputContractError(stage=stage, schema={"type": "object"}, retryable=retryable)


def exercise(
    errors,
    *,
    native=True,
    timeout=True,
    limit=90,
    factory=None,
    cancel_on_factory=False,
    factory_delay=0,
):
    clock = [0.0]
    cancelled = [False]
    calls, factories, retries, configured = [], [], [], []
    outcomes = iter(errors)

    def operation(label):
        calls.append(label)
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        return value

    def correct(error):
        factories.append(error)
        clock[0] += factory_delay
        if cancel_on_factory:
            cancelled[0] = True
        if factory is not None:
            return factory(error)
        return lambda: operation("corrected")

    result, caught = None, None
    try:
        result = run_model_call(
            lambda: operation("original"),
            native=native,
            allow_timeout_retry=timeout,
            remaining=lambda: None if limit is None else limit - clock[0],
            cancelled=lambda: cancelled[0],
            configure_timeout=configured.append,
            on_retry=lambda *args: retries.append(args),
            correction_factory=correct,
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            clock=lambda: clock[0],
        )
    except LLMError as error:
        caught = error
    return result, caught, calls, factories, retries, configured


def test_valid_response_has_one_call_and_no_correction():
    result, error, calls, factories, retries, _ = exercise(["success"])
    assert (result, error, calls, factories, retries) == ("success", None, ["original"], [], [])


@pytest.mark.parametrize("stage", ["json_decode", "pydantic_validation"])
def test_one_eligible_failure_gets_one_corrected_operation(stage):
    result, error, calls, factories, retries, configured = exercise([rejected(stage), "success"])
    assert result == "success" and error is None
    assert calls == ["original", "corrected"]
    assert len(factories) == 1 and configured == [90, 90]
    assert retries == [("output_contract", 2, 0.0)]


def test_second_invalid_output_stops_even_with_an_unused_attempt():
    first, second = rejected(), rejected("pydantic_validation")
    _, error, calls, factories, retries, _ = exercise([first, second, "must not run"])
    assert error is second
    assert calls == ["original", "corrected"] and factories == [first]
    assert retries == [("output_contract", 2, 0.0)]


@pytest.mark.parametrize(
    "stage", ["map_restore", "utf8_decode", "missing_response", "output_overflow", "unknown"]
)
def test_other_typed_stages_do_not_retry_even_when_requested(stage):
    failure = rejected(stage)
    _, error, calls, factories, retries, _ = exercise([failure])
    assert error is failure and calls == ["original"]
    assert factories == retries == []


def test_nonretryable_failure_and_untyped_permanent_error_do_not_retry():
    for failure in (rejected(retryable=False), LLMError("HTTP 400: invalid request")):
        _, error, calls, factories, retries, _ = exercise([failure])
        assert error is failure and calls == ["original"]
        assert factories == retries == []


@pytest.mark.parametrize("limit", [44.999, 1])
def test_correction_does_not_start_below_the_remaining_time_floor(limit):
    failure = rejected()
    _, error, calls, factories, retries, _ = exercise([failure], limit=limit)
    assert error is failure and calls == ["original"]
    assert factories == retries == []


@pytest.mark.parametrize("limit", [45, None])
def test_boundary_and_unbounded_deadline_keep_existing_bounded_attempt_policy(limit):
    result, error, calls, factories, _retries, _ = exercise([rejected(), "success"], limit=limit)
    assert result == "success" and error is None
    assert calls == ["original", "corrected"] and len(factories) == 1


def test_existing_one_attempt_call_cannot_gain_a_correction():
    failure = rejected()
    _, error, calls, factories, retries, _ = exercise([failure], native=False, timeout=False)
    assert error is failure and calls == ["original"]
    assert factories == retries == []


def test_factory_can_decline_correction_without_an_extra_attempt():
    failure = rejected()
    _, error, calls, factories, retries, _ = exercise([failure], factory=lambda error: None)
    assert error is failure and calls == ["original"]
    assert factories == [failure] and retries == []


def test_cancellation_after_factory_prevents_corrected_dispatch():
    _, error, calls, factories, _, _ = exercise(
        [rejected(), "must not run"], cancel_on_factory=True
    )
    assert isinstance(error, LLMError)
    assert calls == ["original"] and len(factories) == 1


def test_factory_cannot_consume_time_below_the_correction_floor():
    failure = rejected()
    _, error, calls, factories, retries, _ = exercise(
        [failure, "must not run"], limit=50, factory_delay=6
    )
    assert error is failure and calls == ["original"]
    assert factories == [failure] and retries == []


@pytest.mark.parametrize("transient", [CAPACITY_ERROR, "timed out"])
def test_corrected_operation_is_reused_for_remaining_transient_retry(transient):
    result, error, calls, factories, retries, _ = exercise(
        [rejected(), LLMError(transient), "success"]
    )
    assert result == "success" and error is None
    assert calls == ["original", "corrected", "corrected"]
    assert len(factories) == 1 and len(retries) == 2


def test_capacity_then_correction_consumes_the_same_three_attempt_cap():
    failure = rejected()
    result, error, calls, factories, retries, _ = exercise(
        [LLMError(CAPACITY_ERROR), failure, "success"]
    )
    assert result == "success" and error is None
    assert calls == ["original", "original", "corrected"]
    assert factories == [failure]
    assert [row[0] for row in retries] == ["capacity", "output_contract"]


def test_exhausted_capacity_slots_cannot_create_a_fourth_attempt():
    failure = rejected()
    _, error, calls, factories, retries, _ = exercise(
        [LLMError(CAPACITY_ERROR), LLMError(CAPACITY_ERROR), failure]
    )
    assert error is failure and calls == ["original"] * 3
    assert factories == [] and len(retries) == 2
