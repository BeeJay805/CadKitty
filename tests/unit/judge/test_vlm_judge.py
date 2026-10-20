"""Offline tests for the ADVISORY VLM intent-match judge.

No real LLM / paid calls: the LLM client is faked or its internal
``messages.parse`` is monkeypatched. A real STL is produced by compiling a
trivial one-Box PartProgram so the render step exercises the real renderer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.judge.vlm_judge import (
    IntentJudgment,
    _VlmVerdict,
    judge_intent_match,
)
from forge_mind.llm.cost import LLMUsage
from forge_mind.types import Vec3


def _tiny_stl(tmp_path: Path) -> Path:
    """Compile a trivial one-Box part to a real STL on disk."""
    program = PartProgram(
        metadata=Metadata(name="tiny_box"),
        operations=[Box(op_id="body", width=20.0, depth=20.0, height=4.0, position=Vec3(z=2.0))],
    )
    model = compile_program(program, tmp_path)
    return model.stl_path


class _FakeClient:
    """Stand-in LLMClient: returns a canned verdict, records the call."""

    def __init__(self, verdict: _VlmVerdict, usage: LLMUsage) -> None:
        self._verdict = verdict
        self._usage = usage
        self.calls: list[dict[str, Any]] = []

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[_VlmVerdict],
        model: str,
        max_tokens: int,
    ) -> tuple[_VlmVerdict, LLMUsage]:
        self.calls.append(
            {
                "system": system,
                "user": user,
                "image_png": image_png,
                "model": model,
                "max_tokens": max_tokens,
            }
        )
        return self._verdict, self._usage


class _RaisingClient:
    """Stand-in whose vision call always raises (simulates an LLM/parse error)."""

    def generate_structured_with_image(self, **_: Any) -> tuple[_VlmVerdict, LLMUsage]:
        raise RuntimeError("boom from the model")


def test_maps_fake_verdict_into_judgment_with_cost(tmp_path: Path) -> None:
    stl = _tiny_stl(tmp_path)
    verdict = _VlmVerdict(
        match=False,
        missing_or_wrong=["central bore (no circular opening visible)"],
        reason="The top view shows a solid plate with no hole.",
    )
    # cost_usd is set by the client (via cost_of); the fake supplies it directly.
    usage = LLMUsage(
        model="claude-sonnet-4-6", input_tokens=1500, output_tokens=120, cost_usd=0.0063
    )
    fake = _FakeClient(verdict, usage)

    result = judge_intent_match(stl, "a plate with a central bore", work_dir=tmp_path, client=fake)

    assert isinstance(result, IntentJudgment)
    assert result.match is False
    assert result.missing_or_wrong == ["central bore (no circular opening visible)"]
    assert result.reason == "The top view shows a solid plate with no hole."
    assert result.cost_usd == usage.cost_usd
    assert result.cost_usd > 0.0  # propagated from the usage

    # The render happened and the image bytes were forwarded (PNG magic).
    assert len(fake.calls) == 1
    assert fake.calls[0]["image_png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert "a plate with a central bore" in fake.calls[0]["user"]


def test_match_true_passthrough(tmp_path: Path) -> None:
    stl = _tiny_stl(tmp_path)
    verdict = _VlmVerdict(match=True, missing_or_wrong=[], reason="All named features present.")
    fake = _FakeClient(
        verdict, LLMUsage(model="claude-sonnet-4-6", input_tokens=10, output_tokens=5)
    )

    result = judge_intent_match(stl, "a flat plate", work_dir=tmp_path, client=fake)

    assert result.match is True
    assert result.missing_or_wrong == []


def test_llm_failure_never_raises_returns_none(tmp_path: Path) -> None:
    stl = _tiny_stl(tmp_path)

    # Must NOT raise — advisory tool can never propagate an exception into a gate.
    result = judge_intent_match(stl, "a plate", work_dir=tmp_path, client=_RaisingClient())

    assert result.match is None
    assert result.missing_or_wrong == []
    assert "could not judge" in result.reason
    assert "boom from the model" in result.reason
    assert result.cost_usd == 0.0


def test_render_failure_never_raises_returns_none(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.stl"
    verdict = _VlmVerdict(match=True, missing_or_wrong=[], reason="n/a")
    fake = _FakeClient(
        verdict, LLMUsage(model="claude-sonnet-4-6", input_tokens=1, output_tokens=1)
    )

    # Render will fail on the missing STL; judge must swallow it and return None.
    result = judge_intent_match(bogus, "a plate", work_dir=tmp_path, client=fake)

    assert result.match is None
    assert result.missing_or_wrong == []
    assert "could not judge" in result.reason
    assert result.cost_usd == 0.0
    # The LLM was never reached because render failed first.
    assert fake.calls == []


def test_client_builds_image_and_text_content_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The new client method must send [image-block, text-block] without a network call."""
    from forge_mind.llm.client import LLMClient

    captured: dict[str, Any] = {}

    class _StubResponse:
        stop_reason = "end_turn"
        parsed_output = _VlmVerdict(match=True, missing_or_wrong=[], reason="ok")

        class usage:  # noqa: N801 - mimic SDK attribute shape
            input_tokens = 100
            output_tokens = 20
            cache_read_input_tokens = 0
            cache_creation_input_tokens = 0

    def _fake_parse(**kwargs: Any) -> _StubResponse:
        captured.update(kwargs)
        return _StubResponse()

    client = LLMClient.__new__(LLMClient)  # bypass __init__ (no API key needed)

    class _Messages:
        parse = staticmethod(_fake_parse)

    class _Inner:
        messages = _Messages()

    client._client = _Inner()  # type: ignore[attr-defined]

    verdict, usage = client.generate_structured_with_image(
        system="SYS",
        user="USER",
        image_png=b"\x89PNG\r\n\x1a\nDATA",
        output_model=_VlmVerdict,
        model="claude-sonnet-4-6",
        max_tokens=256,
    )

    assert verdict.match is True
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cost_usd > 0.0

    # Content blocks: image first, then text.
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["type"] == "base64"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "USER"
    # base64 of the raw bytes, decoded back, round-trips.
    import base64

    assert base64.standard_b64decode(content[0]["source"]["data"]) == b"\x89PNG\r\n\x1a\nDATA"
    # System prompt forwarded as a text block, marked for ephemeral prompt caching
    # (the vision system prompt is re-sent every draft/iteration, so caching pays off).
    assert captured["system"] == [
        {"type": "text", "text": "SYS", "cache_control": {"type": "ephemeral"}}
    ]
    assert captured["output_format"] is _VlmVerdict
