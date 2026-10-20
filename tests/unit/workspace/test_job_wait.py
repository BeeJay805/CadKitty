from __future__ import annotations

import asyncio
import copy

import pytest

from forge_mind.workspace.job_wait import summary_text, wait_for_job

JOB = "a" * 32
OTHER_JOB = "b" * 32


def observed(state: str = "running", job_id: str | None = JOB) -> dict:
    return {
        "id": "c" * 32,
        "job_id": job_id,
        "state": state,
        "design_revision": "d" * 32,
        "selected_revision": "e" * 32,
        "accepted_revision": None,
        "input_version": 3,
        "prompt": "Must not appear in a wait response",
        "recent_events": [{"message": "Large event history"}],
        "candidate": {"eligible": True},
    }


@pytest.mark.parametrize(
    "state",
    ["contract_satisfied", "needs_review", "error", "cancelled", "interrupted", "budget_exhausted"],
)
def test_terminal_requires_matching_job_and_preserves_exact_state(state: str) -> None:
    async def read(remaining: float) -> dict:
        assert 0 < remaining <= 20
        return observed(state)

    result = asyncio.run(wait_for_job(read, JOB))
    assert result.outcome == "terminal"
    assert result.state == state
    assert result.target_job_id == result.current_job_id == JOB
    assert "Inspect evidence" in summary_text(result)
    assert result.snapshot.model_dump_json() in summary_text(result)
    assert '"design_revision":"' + "d" * 32 + '"' in summary_text(result)
    assert "recent_events" not in summary_text(result)
    assert "prompt" not in summary_text(result)
    assert "acceptance" not in result.model_dump()


@pytest.mark.parametrize("current_job", [OTHER_JOB, None])
def test_replacement_does_not_infer_target_completion(current_job: str | None) -> None:
    async def read(remaining: float) -> dict:
        return observed("contract_satisfied", current_job)

    result = asyncio.run(wait_for_job(read, JOB))
    assert result.outcome == "superseded"
    assert result.current_job_id == current_job
    assert result.snapshot.design_revision == "d" * 32
    assert "Completion of the requested job is unknown" in summary_text(result)


@pytest.mark.parametrize("state", ["new_future_state", "idle", "accepted"])
def test_unknown_state_does_not_mean_terminal(state: str) -> None:
    async def read(remaining: float) -> dict:
        return observed(state)

    result = asyncio.run(wait_for_job(read, JOB))
    assert result.outcome == "unknown"
    assert result.state == state


@pytest.mark.parametrize("state", ["queued", "running", "cancelling"])
def test_active_states_wait_with_sleep_and_without_mutation(state: str) -> None:
    source = observed(state)
    original = copy.deepcopy(source)
    calls = []

    async def read(remaining: float) -> dict:
        calls.append(asyncio.get_running_loop().time())
        return source if len(calls) == 1 else observed("needs_review")

    result = asyncio.run(wait_for_job(read, JOB, timeout_seconds=1))
    assert result.outcome == "terminal"
    assert len(calls) == 2
    assert calls[1] - calls[0] >= 0.20
    assert source == original
    assert set(result.snapshot.model_dump()) == {
        "id",
        "design_revision",
        "selected_revision",
        "accepted_revision",
        "input_version",
    }
    assert "prompt" not in result.model_dump_json()
    assert "recent_events" not in result.model_dump_json()


def test_deadline_includes_initial_slow_read_and_has_no_unauthorized_snapshot() -> None:
    cancelled = []

    async def read(remaining: float) -> dict:
        assert 0 < remaining <= 0.04
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        return observed()

    async def scenario() -> None:
        started = asyncio.get_running_loop().time()
        with pytest.raises(TimeoutError):
            await wait_for_job(read, JOB, timeout_seconds=0.04)
        assert asyncio.get_running_loop().time() - started < 1

    asyncio.run(scenario())
    assert cancelled == [True]


def test_deadline_includes_later_slow_read_and_returns_last_snapshot() -> None:
    calls = []
    cancelled = []

    async def read(remaining: float) -> dict:
        calls.append(remaining)
        if len(calls) == 2:
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)
        return observed()

    result = asyncio.run(wait_for_job(read, JOB, timeout_seconds=0.35))
    assert result.outcome == "timeout"
    assert result.state == "running"
    assert len(calls) == 2
    assert 0 < calls[1] < calls[0] <= 0.35
    assert cancelled == [True]
    assert result.elapsed_seconds < 1
    assert "last observed" in summary_text(result)


def test_poll_backoff_is_positive_and_bounded(monkeypatch) -> None:
    delays = []
    real_sleep = asyncio.sleep

    async def sleep(delay: float) -> None:
        delays.append(delay)
        await real_sleep(0)

    async def read(remaining: float) -> dict:
        return observed("needs_review" if len(delays) == 4 else "running")

    monkeypatch.setattr(asyncio, "sleep", sleep)
    result = asyncio.run(wait_for_job(read, JOB))
    assert result.outcome == "terminal"
    assert delays == [0.25, 0.5, 1.0, 1.0]


def test_deadline_during_sleep_is_read_only_and_does_not_busy_poll() -> None:
    calls = []

    async def read(remaining: float) -> dict:
        calls.append(remaining)
        return observed()

    result = asyncio.run(wait_for_job(read, JOB, timeout_seconds=0.04))
    assert result.outcome == "timeout"
    assert len(calls) == 1


def test_request_cancellation_propagates_without_job_cancellation() -> None:
    calls = []

    async def scenario() -> None:
        entered = asyncio.Event()

        async def read(remaining: float) -> dict:
            calls.append("read")
            entered.set()
            await asyncio.sleep(10)
            return observed()

        task = asyncio.create_task(wait_for_job(read, JOB))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert calls == ["read"]


@pytest.mark.parametrize(
    "error", [PermissionError("owner"), RuntimeError("transport"), TimeoutError("request")]
)
@pytest.mark.parametrize("after_snapshot", [False, True])
def test_callback_errors_propagate_unchanged(error: Exception, after_snapshot: bool) -> None:
    calls = []

    async def read(remaining: float) -> dict:
        calls.append(remaining)
        if after_snapshot and len(calls) == 1:
            return observed()
        raise error

    with pytest.raises(type(error)) as raised:
        asyncio.run(wait_for_job(read, JOB))
    assert raised.value is error


@pytest.mark.parametrize("timeout", [0, -1, 20.01, float("inf"), float("nan"), True, "1"])
def test_invalid_deadline_never_calls_read(timeout) -> None:
    async def read(remaining: float) -> dict:
        pytest.fail("Invalid arguments must not read a project")

    with pytest.raises(ValueError, match="timeout_seconds"):
        asyncio.run(wait_for_job(read, JOB, timeout_seconds=timeout))


@pytest.mark.parametrize("job", ["", "A" * 32, "../job", None])
def test_invalid_job_never_calls_read(job) -> None:
    async def read(remaining: float) -> dict:
        pytest.fail("Invalid arguments must not read a project")

    with pytest.raises(ValueError, match="job_id"):
        asyncio.run(wait_for_job(read, job))
