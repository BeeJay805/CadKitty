"""Local-GPU provider (ADR-0111): route the engine's structured LLM calls to a local Ollama model
(the RTX 4090 flywheel). $0, and no subscription rate-limits / 401s -- the failure mode that
truncated every breadth run this session.

``qwen3-coder:30b`` produces schema-valid structured output via Ollama's ``format`` param (measured:
a real ``output_model.model_json_schema()`` round-trips and validates). A code model cannot SEE, so
the vision sibling degrades to TEXT-ONLY -- the reference render is an enhancement (ADR-0069), not a
requirement, and the author builds from the text prompt. Duck-types ``LLMClientLike`` so every engine
stage threads the same paths as the subscription/API clients.
"""

from __future__ import annotations

import os
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from forge_mind.llm.client import LLMError
from forge_mind.llm.context_guard import check_not_truncated, plan_window
from forge_mind.llm.cost import LLMUsage

T = TypeVar("T", bound=BaseModel)

_DEFAULT_MODEL = "qwen3-coder:30b"  # the pulled roster coder; the old qwen2.5-coder:7b default
# 404'd every env-less call (model was never pulled -- 2026-07-19 roster audit)
_DEFAULT_HOST = "http://localhost:11434"
# Cap the context so the KV cache stays small enough for all three roster models to co-reside on a
# 24GB card (32k default evicts CAD-Coder). Part prompts are tiny; 8k is ample. Env-overridable.
_NUM_CTX = int(os.environ.get("FORGE_OLLAMA_NUM_CTX", "8192"))


# Appended to the system prompt on the ONE runaway retry (see _chat): greedy decoding on some
# prompts fills the ENTIRE generation budget mid-JSON-string (measured 2026-07-18: the cable-clip
# author cut at ~13K chars under a 4096 budget and ~19.7K chars under 6144 -- the budget is not
# the problem, the decode path is). A temperature bump + this note breaks the runaway.
_RUNAWAY_NOTE = (
    "\n\nIMPORTANT: your previous answer overflowed the available space. Answer COMPACTLY this "
    "time -- minimal comments, no repetition, only what the schema asks for."
)
_RUNAWAY_RETRY_TEMP = 0.35


def _temperature() -> float:
    """Sampling temperature (FORGE_OLLAMA_TEMPERATURE, default 0 = the deterministic contract).
    Measured 2026-07-18: at 0, same-prompt re-rolls return near-identical completions, so best-of-N
    style callers need either this knob or prompt-level diversification (revision.py does the
    latter). Kept env-scoped so structured extraction stays deterministic by default."""
    try:
        return max(0.0, float(os.environ.get("FORGE_OLLAMA_TEMPERATURE", "0")))
    except ValueError:
        return 0.0


class OllamaClient:
    """Structured-generation client against a local Ollama server. Local + free: ``cost_usd`` is
    always 0.0 (the subscription-only billing invariant holds -- a nonzero cost would be a leak
    tell, and there is genuinely no spend here)."""

    def __init__(
        self, model: str | None = None, host: str | None = None, timeout_s: float = 600.0
    ) -> None:
        self._model = model or os.environ.get("FORGE_OLLAMA_MODEL", _DEFAULT_MODEL)
        self._host = (host or os.environ.get("FORGE_OLLAMA_HOST", _DEFAULT_HOST)).rstrip("/")
        self._http = httpx.Client(timeout=timeout_s)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,  # accepted for the Protocol; the LOCAL model is always used
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]:
        return self._chat(system, user, output_model, max_tokens)

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
        # a code model cannot see -- degrade to TEXT-ONLY (the render is an enhancement, not required).
        return self._chat(system, user, output_model, max_tokens)

    def _chat(
        self,
        system: str,
        user: str,
        output_model: type[T],
        max_tokens: int,
        *,
        _runaway_retry: bool = True,
        _temp_override: float | None = None,
    ) -> tuple[T, LLMUsage]:
        schema = output_model.model_json_schema()
        prompt_chars = len(system) + len(user)
        # Grow the window to fit THIS prompt and clamp the generation budget to the room left
        # (ADR-0115): the fixed 8192 silently truncated every author call, and the engine's
        # cloud-sized 16K max_tokens would otherwise refuse every local author call outright.
        num_ctx, num_predict = plan_window(
            prompt_chars=prompt_chars,
            requested_predict=max_tokens,
            floor=_NUM_CTX,
            label=f"ollama structured ({self._model})",
        )
        try:
            resp = self._http.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "format": schema,  # Ollama structured output: constrain to the pydantic schema
                    "stream": False,
                    "options": {
                        "temperature": (
                            _temp_override if _temp_override is not None else _temperature()
                        ),
                        "num_predict": num_predict,
                        "num_ctx": num_ctx,
                    },
                },
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(
                f"ollama request failed (host {self._host}, model {self._model}); is `ollama serve` "
                f"running and the model pulled? {exc}"
            ) from exc
        data: dict[str, Any] = resp.json()
        check_not_truncated(
            prompt_chars=prompt_chars,
            prompt_eval_count=int(data.get("prompt_eval_count") or 0),
            num_ctx=num_ctx,
            num_predict=num_predict,
            label=f"ollama structured ({self._model})",
        )
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise LLMError(f"ollama returned empty content (model {self._model})")
        try:
            parsed = output_model.model_validate_json(content)
        except ValidationError as exc:
            out_tokens = int(data.get("eval_count") or 0)
            # Runaway retry (measured 2026-07-18): greedy decoding sometimes fills the ENTIRE
            # budget mid-JSON (invalid JSON + eval_count at the cap). One retry with a mild
            # temperature and a be-compact note breaks the runaway path; any other validation
            # failure -- and a failed retry -- still raises (fail loud, never loop).
            if _runaway_retry and out_tokens >= num_predict - 8:
                return self._chat(
                    system + _RUNAWAY_NOTE,
                    user,
                    output_model,
                    max_tokens,
                    _runaway_retry=False,
                    _temp_override=_RUNAWAY_RETRY_TEMP,
                )
            raise LLMError(
                f"ollama structured output failed to validate against {output_model.__name__} "
                f"(model {self._model}): {exc}"
            ) from exc
        usage = LLMUsage(
            model=self._model,
            input_tokens=int(data.get("prompt_eval_count") or 0),
            output_tokens=int(data.get("eval_count") or 0),
            cost_usd=0.0,  # LOCAL inference: genuinely free
        )
        return parsed, usage


__all__ = ["OllamaClient"]
