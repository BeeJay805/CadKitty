"""OllamaVisionClient (ADR-0113): real image input + structured output for the local vision role.
Mocks the HTTP layer -- no server, $0."""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest
from pydantic import BaseModel

from forge_mind.llm.ollama_vision_client import OllamaVisionClient


class _Verdict(BaseModel):
    looks_right: bool
    note: str


class _FakeResp:
    def __init__(self, payload: dict[str, Any], capture: dict[str, Any]) -> None:
        self._payload = payload
        self._capture = capture

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _client(monkeypatch: pytest.MonkeyPatch, capture: dict[str, Any]) -> OllamaVisionClient:
    client = OllamaVisionClient(model="qwen2.5vl:7b", host="http://x")
    payload = {"message": {"content": json.dumps({"looks_right": True, "note": "clean box"})}}

    def _post(url: str, json: dict[str, Any]) -> _FakeResp:
        capture["url"] = url
        capture["body"] = json
        return _FakeResp(payload, capture)

    monkeypatch.setattr(client._http, "post", _post)
    return client


def test_image_is_base64_encoded_into_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    png = b"\x89PNG\r\n\x1a\n fake bytes"
    verdict, usage = client.generate_structured_with_image(
        system="you inspect parts",
        user="does this look right?",
        image_png=png,
        output_model=_Verdict,
        model="ignored",
        max_tokens=256,
    )
    assert verdict.looks_right is True and verdict.note == "clean box"
    assert usage.cost_usd == 0.0  # local vision: $0
    # the PNG rode along as a base64 image on the user message.
    user_msg = capture["body"]["messages"][1]
    assert user_msg["images"] == [base64.b64encode(png).decode("ascii")]
    # structured output was requested via the schema.
    assert "format" in capture["body"]


def test_text_only_path_sends_no_image(monkeypatch: pytest.MonkeyPatch) -> None:
    capture: dict[str, Any] = {}
    client = _client(monkeypatch, capture)
    client.generate_structured(
        system="s", user="u", output_model=_Verdict, model="ignored", max_tokens=64
    )
    assert "images" not in capture["body"]["messages"][1]
