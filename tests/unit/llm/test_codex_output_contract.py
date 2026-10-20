"""Rejected local output keeps safe cause metadata without making provider calls."""

import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, Field, model_validator

from forge_mind.llm import codex_client as transport
from forge_mind.llm.client import LLMError

SECRET = "PRIVATE_RESPONSE_CANARY_7af3"


class Response(BaseModel):
    count: int = Field(gt=0)


@pytest.fixture
def controlled(monkeypatch):
    state = SimpleNamespace(
        answer=b'{"count":1}',
        events=b'{"type":"turn.completed"}\n',
        exit_code=0,
        launches=0,
        roots=[],
    )
    monkeypatch.setattr(transport, "_command", lambda: ["output-contract-process-mock"])

    def launch(command, **kwargs):
        state.launches += 1
        root = Path(kwargs["cwd"])
        state.roots.append(root)
        if state.answer is not None:
            (root / "answer.json").write_bytes(state.answer)
        kwargs["stdout"].write(state.events)
        kwargs["stdout"].flush()
        return SimpleNamespace(
            pid=999999,
            stdin=io.BytesIO(),
            returncode=state.exit_code,
            poll=lambda: state.exit_code,
            wait=lambda **_: state.exit_code,
        )

    monkeypatch.setattr(transport.subprocess, "Popen", launch)
    return transport.CodexClient(timeout_s=30), state


def call(client, output_model=Response):
    return client.generate_structured(
        system="Original system",
        user="Original owner input",
        output_model=output_model,
        model="gpt-6-astra",
        max_tokens=100,
    )


@pytest.mark.parametrize(
    ("answer", "stage"),
    [
        (SECRET.encode(), "json_decode"),
        (json.dumps({"count": SECRET}).encode(), "pydantic_validation"),
    ],
)
def test_clean_completed_output_rejection_is_typed_and_safe(controlled, answer, stage):
    client, state = controlled
    state.answer = answer
    with pytest.raises(LLMError) as failure:
        call(client)
    error = failure.value
    assert error.stage == stage
    assert error.retryable is True
    assert error.__cause__ is not None
    diagnostics = client.last_call_diagnostics
    assert diagnostics["output_contract"]["stage"] == stage
    assert diagnostics["output_contract"]["retryable"] is True
    assert diagnostics["answer_bytes"] == len(answer)
    assert SECRET not in str(error)
    assert SECRET not in json.dumps(diagnostics)
    assert SECRET not in json.dumps(error.feedback)
    assert "response_sha256" not in json.dumps(diagnostics)
    assert state.launches == 1
    assert all(not root.exists() for root in state.roots)


@pytest.mark.parametrize(
    "events",
    [
        b"",
        b"not JSON\n",
        b'{"type":"turn.completed"}\n{"type":"error"}\n',
        b'{"type":"turn.completed"}\n{"type":"turn.failed"}\n',
    ],
)
def test_uncertain_or_failed_lifecycle_never_enables_output_correction(controlled, events):
    client, state = controlled
    state.answer = SECRET.encode()
    state.events = events
    with pytest.raises(LLMError) as failure:
        call(client)
    assert failure.value.retryable is False
    assert client.last_call_diagnostics["output_contract"]["retryable"] is False


def test_utf8_rejection_is_observable_but_not_correctable(controlled):
    client, state = controlled
    state.answer = b"\xff"
    with pytest.raises(LLMError) as failure:
        call(client)
    assert failure.value.stage == "utf8_decode"
    assert failure.value.retryable is False


def test_custom_validator_messages_and_dynamic_keys_do_not_escape(controlled):
    class CustomResponse(BaseModel):
        values: dict[str, int]

        @model_validator(mode="after")
        def reject(self):
            raise ValueError(SECRET + str(self.values))

    client, state = controlled
    for value in ({SECRET: "invalid"}, {SECRET: 1}):
        state.answer = json.dumps({"values": value}).encode()
        with pytest.raises(LLMError) as failure:
            call(client, CustomResponse)
        assert failure.value.stage == "pydantic_validation"
        assert SECRET not in json.dumps(failure.value.feedback)
        assert SECRET not in json.dumps(client.last_call_diagnostics)
        assert SECRET not in str(failure.value)


def test_valid_response_is_unchanged_and_has_no_contract_failure(controlled):
    client, state = controlled
    result, _ = call(client)
    assert result.count == 1
    assert state.launches == 1
    assert client.last_call_diagnostics["output_contract"] is None


def test_cleanup_failure_retains_the_rejection_stage_but_never_enables_correction(
    controlled, monkeypatch
):
    client, state = controlled
    state.answer = SECRET.encode()
    temporary = transport.tempfile.TemporaryDirectory

    @contextmanager
    def failed_cleanup(*args, **kwargs):
        try:
            with temporary(*args, **kwargs) as root:
                yield root
        finally:
            raise OSError(SECRET)

    monkeypatch.setattr(transport.tempfile, "TemporaryDirectory", failed_cleanup)
    with pytest.raises(OSError):
        call(client)
    diagnostics = client.last_call_diagnostics
    assert diagnostics["cleanup_error"] is True
    assert diagnostics["output_contract"]["stage"] == "json_decode"
    assert diagnostics["output_contract"]["retryable"] is False
    assert SECRET not in json.dumps(diagnostics)


@pytest.mark.parametrize(
    "mode", ["empty", "missing", "overflow", "nonzero", "cancelled", "unknown_event"]
)
def test_permanent_or_cancelled_failures_cannot_enable_correction(controlled, mode):
    client, state = controlled
    state.answer = SECRET.encode()
    if mode == "empty":
        state.answer = b""
    elif mode == "missing":
        state.answer = None
    elif mode == "overflow":
        state.answer = b"x" * 512_001
    elif mode == "nonzero":
        state.exit_code = 1
    elif mode == "cancelled":
        client.cancelled = lambda: state.launches > 0
    else:
        state.events += b'{"type":"unknown"}\n'
    with pytest.raises(LLMError) as failure:
        call(client)
    assert getattr(failure.value, "retryable", False) is False
    record = client.last_call_diagnostics["output_contract"]
    assert record is None or record["retryable"] is False
    assert state.launches == 1


def test_map_restoration_errors_remain_noncorrectable(controlled):
    class MapResponse(BaseModel):
        values: dict[str, int]

    client, state = controlled
    state.answer = b'{"values":[{"key":"x","value":1},{"key":"x","value":2}]}'
    with pytest.raises(LLMError) as failure:
        call(client, MapResponse)
    assert failure.value.stage == "map_restore"
    assert failure.value.retryable is False
