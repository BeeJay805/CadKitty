"""Local VISION provider (ADR-0113): Qwen2.5-VL over Ollama, with REAL image input.

The critic/inspection role needs a model that can actually SEE the rendered part (unlike the code
model, which degrades vision to text-only). This client duck-types ``LLMClientLike`` so the vision
critic threads the same structured-output path as every other stage, but its
``generate_structured_with_image`` sends the PNG to a vision model (Ollama ``/api/chat`` ``images``
field, base64) instead of dropping it. $0 (local): ``cost_usd`` is always 0.0.
"""

from __future__ import annotations

import base64
import os
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from forge_mind.llm.client import LLMError
from forge_mind.llm.context_guard import check_not_truncated, plan_window
from forge_mind.llm.cost import LLMUsage

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "qwen2.5vl:7b"
_DEFAULT_HOST = "http://localhost:11434"
# Cap context so the KV cache stays small enough for VRAM co-residence with the other roster models.
_NUM_CTX = int(os.environ.get("FORGE_OLLAMA_NUM_CTX", "8192"))


class OllamaVisionClient:
    """Structured generation against a local Qwen2.5-VL, with optional image input. $0 (local)."""

    def __init__(
        self, model: str | None = None, host: str | None = None, timeout_s: float = 600.0
    ) -> None:
        self._model = model or os.environ.get("FORGE_VISION_MODEL", _DEFAULT_MODEL)
        self._host = (host or os.environ.get("FORGE_VISION_HOST", _DEFAULT_HOST)).rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,  # accepted for the Protocol; the LOCAL vision model is always used
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]:
        return self._chat(system, user, output_model, max_tokens, image_png=None)

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: type[T],
        model: str,
        max_tokens: int,
    ) -> tuple[T, LLMUsage]:
        return self._chat(system, user, output_model, max_tokens, image_png=image_png)

    def _chat(
        self,
        system: str,
        user: str,
        output_model: type[T],
        max_tokens: int,
        *,
        image_png: bytes | None,
    ) -> tuple[T, LLMUsage]:
        user_msg: dict[str, Any] = {"role": "user", "content": user}
        if image_png is not None:
            user_msg["images"] = [base64.b64encode(image_png).decode("ascii")]
        # char count is a LOWER bound on prompt tokens even with an image attached (the image only
        # ADDS tokens), so the guard stays sound here -- it can under-fire, never false-fire.
        prompt_chars = len(system) + len(user)
        num_ctx, num_predict = plan_window(
            prompt_chars=prompt_chars,
            requested_predict=max_tokens,
            floor=_NUM_CTX,
            label=f"ollama vision ({self._model})",
        )
        try:
            resp = self._http.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "messages": [{"role": "system", "content": system}, user_msg],
                    "format": output_model.model_json_schema(),
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": num_predict, "num_ctx": num_ctx},
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"ollama vision request failed (host {self._host}, model {self._model}); is "
                f"`ollama serve` running and the model pulled? {exc}"
            ) from exc
        data: dict[str, Any] = resp.json()
        check_not_truncated(
            prompt_chars=prompt_chars,
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            num_ctx=num_ctx,
            num_predict=num_predict,
            label=f"ollama vision ({self._model})",
        )
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise LLMError(f"ollama vision returned empty content (model {self._model})")
        try:
            parsed = output_model.model_validate_json(content)
        except ValidationError as exc:
            raise LLMError(
                f"ollama vision output failed to validate against {output_model.__name__} "
                f"(model {self._model}): {exc}"
            ) from exc
        usage = LLMUsage(
            model=self._model,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            cost_usd=0.0,
        )
        return parsed, usage


__all__ = ["OllamaVisionClient"]
