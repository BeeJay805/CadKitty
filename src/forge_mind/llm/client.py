"""Anthropic client wrapper.

One job: turn (system prompt, user prompt, output model) into a validated
Pydantic instance plus normalized usage/cost. Uses ``messages.parse`` for
structured output, caches the (large, stable) system prompt, and records tokens
and cost. The SDK already retries 429/5xx with backoff.
"""

from __future__ import annotations

import base64
import os
from typing import Any, TypeVar

import anthropic
from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
)
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from forge_mind.llm.cost import LLMUsage, cost_of

T = TypeVar("T", bound=BaseModel)

# Load .env at import. override=True so the real key in .env wins over a stale
# or empty ANTHROPIC_API_KEY that a launcher (e.g. uv) may have injected from an
# earlier empty .env.
load_dotenv(override=True)


class LLMError(RuntimeError):
    """Raised for unrecoverable LLM problems (missing key, refusal, bad output)."""


# Above this budget, structured calls use the SDK's STREAMING path (same parsed result):
# non-streaming requests are refused by the SDK above ~21.3k projected output tokens.
_STREAM_OVER = 20_000


class LLMClient:
    """Thin wrapper around anthropic.Anthropic for structured generation."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to .env (see .env.example).")
        # max_retries above the SDK default (2): under best-of-N + critic + judge load the API
        # rate-limits transiently, and a swallowed judge failure was rejecting VALID parts (the
        # benchmark's fan shroud). The SDK retries 429/5xx with exponential backoff + retry-after.
        self._client = anthropic.Anthropic(api_key=key, max_retries=8)

    def _parse_retrying(
        self,
        *,
        model: str,
        max_tokens: int,
        system: list[TextBlockParam],
        messages: list[MessageParam],
        output_format: type[T],
    ) -> Any:
        """``messages.parse`` with typed error handling + ONE retry on MALFORMED structured output.

        A response that hits ``max_tokens`` mid-JSON (or is otherwise unparseable) makes the SDK
        raise pydantic's ``ValidationError`` from inside ``parse()`` -- BEFORE our ``stop_reason``
        checks ever run. That crash propagated through the author stage and killed a whole
        generation (measured: the fan-shroud bench run died at output column 10263 of a truncated
        script). Malformation is stochastic, so ONE fresh retry usually recovers it; a second
        failure raises a clean ``LLMError`` that callers treat as a stage failure, never a crash.
        Honest accounting caveat: a failed attempt's tokens are real spend the SDK does not surface
        on a parse failure, so they are absent from usage (rare + bounded by the iteration caps).

        Budgets above ``_STREAM_OVER`` use the STREAMING path (same parsed final message): the
        SDK refuses non-streaming requests above ~21.3k projected tokens, and a Fable chief
        authoring a 14-card engine plan measurably needs >16k (the i4 probe truncation).
        """
        last: Exception | None = None
        for _attempt in range(2):
            try:
                if max_tokens > _STREAM_OVER:
                    with self._client.messages.stream(
                        model=model,
                        max_tokens=max_tokens,
                        system=system,
                        messages=messages,
                        output_format=output_format,
                    ) as stream:
                        return stream.get_final_message()
                return self._client.messages.parse(
                    model=model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                    output_format=output_format,
                )
            except anthropic.AuthenticationError as exc:
                raise LLMError(f"authentication failed: {exc}") from exc
            except anthropic.NotFoundError as exc:
                raise LLMError(f"model not found ({model}): {exc}") from exc
            except ValidationError as exc:
                last = exc
                continue
        raise LLMError(
            f"structured output failed to parse twice (model {model}) -- likely truncated at "
            f"max_tokens={max_tokens} or malformed JSON: {last}"
        ) from last

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: type[T],
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[T, LLMUsage]:
        """Generate a validated ``output_model`` instance from the prompts.

        Raises LLMError on refusal or if the SDK cannot parse the output.
        """
        system_block: TextBlockParam = {"type": "text", "text": system}
        if cache_system:
            system_block["cache_control"] = {"type": "ephemeral"}
        system_blocks: list[TextBlockParam] = [system_block]

        response = self._parse_retrying(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks,
            messages=[{"role": "user", "content": user}],
            output_format=output_model,
        )

        if response.stop_reason == "refusal":
            raise LLMError("model refused to produce the structured output")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"response truncated at max_tokens ({max_tokens}); the structured "
                "output is incomplete and must not be trusted. Raise max_tokens for "
                "this stage."
            )

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("model returned no parseable structured output")

        usage = self._usage(model, response)
        return parsed, usage

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
        """Generate a validated ``output_model`` from a system prompt + (image, text) user turn.

        Additive multimodal sibling of ``generate_structured``: base64-encodes the
        PNG, sends it as an image content block followed by the text prompt, and
        parses the structured output. Reuses ``_usage`` for normalized cost.

        Raises LLMError on refusal or if the SDK cannot parse the output.
        """
        b64 = base64.standard_b64encode(image_png).decode("ascii")
        image_source: Base64ImageSourceParam = {
            "type": "base64",
            "media_type": "image/png",
            "data": b64,
        }
        image_block: ImageBlockParam = {"type": "image", "source": image_source}
        text_block: TextBlockParam = {"type": "text", "text": user}
        message: MessageParam = {"role": "user", "content": [image_block, text_block]}
        system_block: TextBlockParam = {"type": "text", "text": system}
        # Cache the (large) vision system prompt -- the critic + judge prompts are re-sent every
        # iteration and every draft, so an ephemeral cache write pays for itself immediately.
        system_block["cache_control"] = {"type": "ephemeral"}

        response = self._parse_retrying(
            model=model,
            max_tokens=max_tokens,
            system=[system_block],
            messages=[message],
            output_format=output_model,
        )

        if response.stop_reason == "refusal":
            raise LLMError("model refused to produce the structured output")
        if response.stop_reason == "max_tokens":
            raise LLMError(
                f"response truncated at max_tokens ({max_tokens}); the structured "
                "output is incomplete and must not be trusted. Raise max_tokens for "
                "this stage."
            )

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("model returned no parseable structured output")

        usage = self._usage(model, response)
        return parsed, usage

    @staticmethod
    def _usage(model: str, response: Any) -> LLMUsage:
        u = response.usage
        input_tokens = int(getattr(u, "input_tokens", 0) or 0)
        output_tokens = int(getattr(u, "output_tokens", 0) or 0)
        cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
        return LLMUsage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            cost_usd=cost_of(model, input_tokens, output_tokens, cache_read, cache_write),
        )


def api_key_available() -> bool:
    """True if an API key is configured (used to skip LLM tests)."""
    load_dotenv(override=True)
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


__all__ = ["LLMClient", "LLMError", "api_key_available"]
