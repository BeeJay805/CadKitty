"""The structured-output chokepoint must survive a MALFORMED response (the fan-shroud crash): the
SDK raises pydantic's ValidationError from inside ``messages.parse`` when a response truncates
mid-JSON, which used to propagate raw and kill the whole generation. The client now retries ONCE
(malformation is stochastic) and converts a second failure into a clean LLMError."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from forge_mind.llm.client import LLMClient, LLMError


class _Out(BaseModel):
    x: int


def _validation_error() -> ValidationError:
    """A REAL pydantic ValidationError, produced the same way the SDK produces it: by parsing
    truncated JSON (no hand-built error data, so it matches the production exception exactly)."""
    try:
        TypeAdapter(_Out).validate_json('{"x": 1')  # truncated mid-object
    except ValidationError as exc:
        return exc
    raise AssertionError("expected truncated JSON to raise")


def _good_response() -> Any:
    return SimpleNamespace(
        stop_reason="end_turn",
        parsed_output=_Out(x=1),
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        ),
    )


def _client_with_parse(side_effects: list[Any]) -> LLMClient:
    """An LLMClient whose messages.parse pops scripted outcomes (exception instances raise)."""
    client = LLMClient(api_key="test-key-never-used")

    def _parse(**_kwargs: Any) -> Any:
        effect = side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    client._client = SimpleNamespace(messages=SimpleNamespace(parse=_parse))  # type: ignore[assignment]
    return client


def test_one_malformed_response_is_retried_and_recovers() -> None:
    client = _client_with_parse([_validation_error(), _good_response()])
    parsed, usage = client.generate_structured(
        system="s", user="u", output_model=_Out, model="claude-sonnet-4-6", max_tokens=100
    )
    assert parsed.x == 1
    assert usage.input_tokens == 10  # the SUCCESSFUL call's usage


def test_two_malformed_responses_raise_a_clean_llmerror() -> None:
    client = _client_with_parse([_validation_error(), _validation_error()])
    with pytest.raises(LLMError, match="failed to parse twice"):
        client.generate_structured(
            system="s", user="u", output_model=_Out, model="claude-sonnet-4-6", max_tokens=100
        )


class _FakeStream:
    """Context manager mimicking the SDK's MessageStreamManager -> MessageStream."""

    def __init__(self, final: Any) -> None:
        self._final = final

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def get_final_message(self) -> Any:
        return self._final


def test_large_budgets_take_the_streaming_path() -> None:
    """Above _STREAM_OVER the client must STREAM (the SDK refuses non-streaming ~21.3k+),
    returning the same parsed final message (the i4 fable-plan truncation fix)."""
    client = LLMClient(api_key="test-key-never-used")
    calls = {"stream": 0, "parse": 0}

    def _stream(**_kwargs: Any) -> Any:
        calls["stream"] += 1
        return _FakeStream(_good_response())

    def _parse(**_kwargs: Any) -> Any:
        calls["parse"] += 1
        return _good_response()

    client._client = SimpleNamespace(  # type: ignore[assignment]
        messages=SimpleNamespace(parse=_parse, stream=_stream)
    )
    parsed, _usage = client.generate_structured(
        system="s", user="u", output_model=_Out, model="claude-fable-5", max_tokens=32000
    )
    assert parsed.x == 1
    assert calls == {"stream": 1, "parse": 0}
