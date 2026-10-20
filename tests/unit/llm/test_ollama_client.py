"""ADR-0111 local-GPU provider: OllamaClient duck-types LLMClientLike against a local Ollama
server. httpx is mocked (no live server in CI) -- these cover the parse + fail-closed contract; the
live qwen3-coder round-trip is exercised out-of-band (TEST_LOG)."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from forge_mind.llm.client import LLMError
from forge_mind.llm.ollama_client import OllamaClient


class _Brief(BaseModel):
    object_name: str
    n_features: int


class _FakeResp:
    def __init__(self, payload: dict[str, Any], exc: Exception | None = None) -> None:
        self._payload = payload
        self._exc = exc

    def raise_for_status(self) -> None:
        if self._exc is not None:
            raise self._exc

    def json(self) -> dict[str, Any]:
        return self._payload


def _client_returning(monkeypatch: pytest.MonkeyPatch, resp: _FakeResp) -> OllamaClient:
    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", lambda *a, **k: resp)
    return c


def test_parses_structured_output_and_is_free(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResp(
        {
            "message": {"content": '{"object_name": "bracket", "n_features": 3}'},
            "prompt_eval_count": 42,
            "eval_count": 17,
        }
    )
    c = _client_returning(monkeypatch, resp)
    out, usage = c.generate_structured(
        system="s", user="u", output_model=_Brief, model="ignored", max_tokens=256
    )
    assert out.object_name == "bracket" and out.n_features == 3
    assert usage.cost_usd == 0.0  # LOCAL is free -- the billing invariant
    assert usage.input_tokens == 42 and usage.output_tokens == 17


def test_vision_sibling_degrades_to_text(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResp({"message": {"content": '{"object_name": "x", "n_features": 1}'}})
    c = _client_returning(monkeypatch, resp)
    out, _ = c.generate_structured_with_image(
        system="s", user="u", image_png=b"PNG", output_model=_Brief, model="m", max_tokens=64
    )
    assert out.object_name == "x"  # image ignored, text-only path still yields a valid model


def test_empty_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client_returning(monkeypatch, _FakeResp({"message": {"content": ""}}))
    with pytest.raises(LLMError, match="empty content"):
        c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=64)


def test_unparseable_content_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    c = _client_returning(monkeypatch, _FakeResp({"message": {"content": "not json at all"}}))
    with pytest.raises(LLMError, match="failed to validate"):
        c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=64)


def test_temperature_env_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    # Default 0 (deterministic contract); FORGE_OLLAMA_TEMPERATURE opts a run into sampling
    # (measured 2026-07-18: greedy re-rolls are near-identical, defeating best-of-N locally).
    posted: dict[str, Any] = {}
    resp = _FakeResp({"message": {"content": '{"object_name": "x", "n_features": 1}'}})

    def _post(url: str, json: dict[str, Any]) -> _FakeResp:
        posted.update(json)
        return resp

    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", _post)
    c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=64)
    assert posted["options"]["temperature"] == 0.0

    monkeypatch.setenv("FORGE_OLLAMA_TEMPERATURE", "0.4")
    c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=64)
    assert posted["options"]["temperature"] == 0.4


def test_http_error_raises_llmerror(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = _FakeResp({}, exc=httpx.ConnectError("connection refused"))
    c = _client_returning(monkeypatch, resp)
    with pytest.raises(LLMError, match="ollama request failed"):
        c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=64)


def test_runaway_retry_fires_once_with_temp_and_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measured 2026-07-18: greedy decoding filled the ENTIRE budget mid-JSON (invalid JSON +
    eval_count at the cap). The client retries exactly once with a mild temperature and a
    be-compact system note; a valid second answer is returned."""
    calls: list[dict[str, Any]] = []
    responses = [
        _FakeResp(
            {
                "message": {"content": '{"object_name": "runaway...'},  # invalid JSON
                "prompt_eval_count": 500,
                "eval_count": 8192,
            }
        ),  # eval at/above any budget
        _FakeResp(
            {
                "message": {"content": '{"object_name": "ok", "n_features": 1}'},
                "prompt_eval_count": 500,
                "eval_count": 90,
            }
        ),
    ]

    def _post(url: str, json: dict[str, Any]) -> _FakeResp:
        calls.append(json)
        return responses[len(calls) - 1]

    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", _post)
    out, _usage = c.generate_structured(
        system="s", user="u", output_model=_Brief, model="m", max_tokens=8192
    )
    assert out.object_name == "ok"
    assert len(calls) == 2
    assert calls[0]["options"]["temperature"] == 0.0
    assert calls[1]["options"]["temperature"] == 0.35  # the retry samples
    assert "overflowed the available space" in calls[1]["messages"][0]["content"]


def test_non_runaway_validation_failure_still_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # invalid JSON but the output did NOT fill the budget -> no retry, fail loud.
    resp = _FakeResp(
        {"message": {"content": "not json"}, "prompt_eval_count": 50, "eval_count": 20}
    )
    c = OllamaClient()
    monkeypatch.setattr(c._http, "post", lambda *a, **k: resp)
    with pytest.raises(LLMError, match="failed to validate"):
        c.generate_structured(system="s", user="u", output_model=_Brief, model="m", max_tokens=256)
