"""CadCoderClient (ADR-0112): raw-CadQuery transport for the local CAD-Coder model. Mocks the HTTP
layer -- no server, $0. Covers the model-specific handling discovered by testing the real model:
raw ChatML (no system), imperative-prompt normalization, and retry-on-degenerate."""

from __future__ import annotations

from typing import Any

import pytest

from forge_mind.llm.cadcoder_client import CadCoderClient
from forge_mind.llm.client import LLMError

_GOOD = "import cadquery as cq\nr = cq.Workplane('XY').box(1, 1, 1)\n"


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _client(
    monkeypatch: pytest.MonkeyPatch, responses: list[str], capture: dict[str, Any] | None = None
) -> CadCoderClient:
    """Mock /api/generate to return `responses` in order (one per attempt)."""
    client = CadCoderClient(model="test-model", host="http://x")
    seq = iter(responses)

    def _post(url: str, json: dict[str, Any]) -> _FakeResp:
        if capture is not None:
            capture["url"] = url
            capture["json"] = json
        return _FakeResp({"response": next(seq), "prompt_eval_count": 12, "eval_count": 34})

    monkeypatch.setattr(client._http, "post", _post)
    return client


def test_returns_code_and_zero_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    code, usage = _client(monkeypatch, [_GOOD]).generate_cadquery("a cube")
    assert "cq.Workplane" in code
    assert usage.cost_usd == 0.0  # local inference: the subscription-only invariant holds
    assert usage.input_tokens == 12 and usage.output_tokens == 34


def test_sends_raw_chatml_with_no_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    _client(monkeypatch, [_GOOD], capture).generate_cadquery("Create a cube 10mm.")
    body = capture["json"]
    assert body["raw"] is True  # we own the ChatML; ollama must not re-template
    assert body["prompt"].startswith("<|im_start|>user\n")
    assert "system" not in body["prompt"]  # a system prompt derails this fine-tune


def test_normalizes_non_imperative_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    _client(monkeypatch, [_GOOD], capture).generate_cadquery("a rectangular block 40x20x10mm")
    # a noun-phrase prompt is nudged to the imperative training distribution.
    assert "Create a rectangular block" in capture["json"]["prompt"]


def test_leaves_imperative_prompt_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    _client(monkeypatch, [_GOOD], capture).generate_cadquery("Design a cylinder r5 h10")
    assert "<|im_start|>user\nDesign a cylinder" in capture["json"]["prompt"]


def test_extracts_code_from_a_fenced_block(monkeypatch: pytest.MonkeyPatch) -> None:
    fenced = "Here is the part:\n```python\nimport cadquery as cq\nr = cq.Workplane('XY').box(2,2,2)\n```"
    code, _ = _client(monkeypatch, [fenced]).generate_cadquery("a cube")
    assert code.startswith("import cadquery as cq")
    assert "```" not in code and "Here is" not in code


def test_retries_past_a_degenerate_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    # first draw degenerates to a bare fence (the observed failure); the retry succeeds.
    code, _ = _client(monkeypatch, ["```", _GOOD]).generate_cadquery("a cube")
    assert "cq.Workplane" in code


def test_raises_when_every_attempt_degenerates(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(LLMError, match="degenerated"):
        _client(monkeypatch, ["```", "", "  "]).generate_cadquery("a cube", attempts=3)
