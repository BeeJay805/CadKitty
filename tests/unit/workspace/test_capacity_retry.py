"""Temporary native inference refusal must not replay actions or lose diagnostics."""

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace import agent, author_images
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, Requirement
from forge_mind.workspace.store import Store

CAPACITY = (
    "Codex structured call failed (exit=1): Selected model is at capacity. "
    "Please try a different model."
)


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("native", [False, True])
def test_capacity_retry_keeps_request_images_and_diagnostics_then_dispatches_once(
    tmp_path, monkeypatch, native, image_count
):
    clock = [0.0]
    monkeypatch.setattr(agent.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        agent.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds)
    )
    images = author_images.AuthorImages(
        images=tuple(bytes([i + 1]) for i in range(image_count)),
        labels=tuple(f"View {i}" for i in range(image_count)),
        revision_id=None,
        mode="controlled-test",
        supplemental_status="none",
    )
    monkeypatch.setattr(author_images, "prepare_author_images", lambda *args, **kwargs: images)
    seen = []

    class Author:
        model = "scripted-provider"
        timeout_s = None
        last_call_diagnostics = None

        def generate_structured(self, **kwargs):
            seen.append(dict(kwargs))
            index = len(seen)
            self.last_call_diagnostics = {
                "call_id": f"capacity-attempt-{index}",
                "outcome": "nonzero_exit" if index < 3 else "success",
            }
            if index < 3:
                raise LLMError(CAPACITY)
            return Action(kind="schema", query="feature:box", explanation="Read syntax"), LLMUsage(
                self.model, 0, 0
            )

        generate_structured_with_image = generate_structured
        generate_structured_with_images = generate_structured

    store = Store(tmp_path)
    contract = Contract(requirements=[Requirement(id="part", text="A part", source_quote="A part")])
    sid = store.create(
        "owner", "A part", contract.model_dump(), "native_operations" if native else "hybrid"
    )["id"]
    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Author())
    store.enqueue(sid, "owner", {"max_seconds": 120, "max_actions": 1})
    service.run_job(store.claim())
    saved = store.get(sid, "owner")
    assert len(seen) == (3 if native else 1)
    assert all(request == seen[0] for request in seen)
    assert len([event for event in saved["events"] if event["kind"] == "action"]) == int(native)
    diagnostics = [
        event["diagnostics"] for event in saved["events"] if event["kind"] == "model_call"
    ]
    assert [record["call_id"] for record in diagnostics] == [
        f"capacity-attempt-{i}" for i in range(1, len(seen) + 1)
    ]
    assert store.revisions(sid) == []
    assert saved.get("accepted_revision") is None
    if image_count:
        observations = [event for event in saved["events"] if event["kind"] == "image_observation"]
        assert len(observations) == len(seen)
        assert len({event["invocation_id"] for event in observations}) == len(seen)
        assert all(event["images"] == images.records() for event in observations)
        assert [event["response_status"] for event in observations] == (
            ["failed_or_unknown", "failed_or_unknown", "complete"]
            if native
            else ["failed_or_unknown"]
        )


def exercise_call(errors, *, native=True, timeout_retry=True, limit=90, abort_after=None):
    from forge_mind.workspace.model_retry import run_model_call

    clock = [0.0]
    calls, configured, retries = [], [], []

    def operation():
        calls.append(clock[0])
        error = errors[len(calls) - 1] if len(calls) <= len(errors) else None
        if error is not None:
            raise error
        return "success"

    result = None
    caught = None
    try:
        result = run_model_call(
            operation,
            native=native,
            allow_timeout_retry=timeout_retry,
            remaining=lambda: None if limit is None else limit - clock[0],
            cancelled=lambda: abort_after is not None and clock[0] >= abort_after,
            configure_timeout=configured.append,
            on_retry=lambda *args: retries.append(args),
            sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            clock=lambda: clock[0],
        )
    except Exception as error:
        caught = error
    return result, caught, calls, configured, retries, clock[0]


def test_capacity_exhaustion_has_three_total_attempts_and_one_shared_clock():
    errors = [LLMError(CAPACITY) for _ in range(4)]
    result, error, calls, configured, retries, elapsed = exercise_call(errors)
    assert result is None and error is errors[2]
    assert len(calls) == 3
    assert elapsed == pytest.approx(6)
    assert configured == pytest.approx([90, 88, 84])
    assert retries == [("capacity", 2, 2.0), ("capacity", 3, 4.0)]


@pytest.mark.parametrize(
    "error",
    [
        LLMError("Authentication required"),
        LLMError("Codex response did not satisfy the output contract"),
        LLMError("Codex logs exceeded their output-byte limit"),
        LLMError("Selected model is at capacity"),
        LLMError(CAPACITY.replace("exit=1", "exit=2")),
        LLMError("Unknown error: " + CAPACITY),
        ValueError(CAPACITY),
    ],
)
def test_other_failures_are_not_retried(error):
    _, actual, calls, _, retries, _ = exercise_call([error])
    assert actual is error
    assert len(calls) == 1 and retries == []


def test_cancellation_during_backoff_stops_before_second_call():
    _, error, calls, _, _, elapsed = exercise_call([LLMError(CAPACITY)], abort_after=0.2)
    assert isinstance(error, LLMError)
    assert "cancelled" in str(error)
    assert len(calls) == 1
    assert elapsed <= 0.3


@pytest.mark.parametrize("limit", [0, -1])
def test_expired_author_deadline_never_starts_a_call(limit):
    _, error, calls, configured, _, _ = exercise_call([], limit=limit)
    assert isinstance(error, LLMError)
    assert "deadline" in str(error)
    assert calls == [] and configured == []


def test_short_remaining_budget_does_not_start_capacity_backoff():
    refusal = LLMError(CAPACITY)
    _, error, calls, _, retries, elapsed = exercise_call([refusal], limit=2)
    assert error is refusal and len(calls) == 1
    assert retries == [] and elapsed == 0


def test_deadline_expires_during_backoff_before_another_inference():
    from forge_mind.workspace.model_retry import run_model_call

    clock = [0.0]
    calls = []

    def operation():
        calls.append(clock[0])
        raise LLMError(CAPACITY)

    with pytest.raises(LLMError, match="deadline"):
        run_model_call(
            operation,
            native=True,
            allow_timeout_retry=True,
            remaining=lambda: 90 - clock[0],
            cancelled=lambda: False,
            configure_timeout=lambda seconds: None,
            on_retry=lambda *args: None,
            sleep=lambda seconds: clock.__setitem__(0, 91),
            clock=lambda: clock[0],
        )
    assert len(calls) == 1


def test_capacity_and_timeout_share_one_attempt_limit():
    errors = [LLMError(CAPACITY), LLMError("timed out"), LLMError(CAPACITY)]
    _, error, calls, _, retries, _ = exercise_call(errors)
    assert error is errors[2] and len(calls) == 3
    assert [row[0] for row in retries] == ["capacity", "timeout"]


def test_existing_timeout_retries_at_most_once():
    errors = [LLMError("timed out"), LLMError("timed out")]
    _, error, calls, _, retries, _ = exercise_call(errors)
    assert error is errors[1] and len(calls) == 2
    assert retries == [("timeout", 2, 0.0)]


@pytest.mark.parametrize("native", [False, True])
def test_existing_timeout_keeps_remaining_budget_threshold(native):
    timeout = LLMError("timed out")
    _, error, calls, _, retries, _ = exercise_call([timeout], native=native, limit=44)
    assert error is timeout and len(calls) == 1 and retries == []


def test_image_path_does_not_gain_timeout_retries():
    timeout = LLMError("timed out")
    _, error, calls, _, retries, _ = exercise_call([timeout], timeout_retry=False)
    assert error is timeout and len(calls) == 1 and retries == []


def test_unbounded_native_job_still_has_bounded_capacity_attempts():
    result, error, calls, configured, _, elapsed = exercise_call(
        [LLMError(CAPACITY), LLMError(CAPACITY)], limit=None
    )
    assert result == "success" and error is None
    assert len(calls) == 3 and configured == [None, None, None]
    assert elapsed == pytest.approx(6)
