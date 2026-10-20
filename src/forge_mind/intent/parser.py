"""IntentParser — free text -> IntentSpec via the LLM.

The easy stage: a structured-extraction transform. Defaults to Haiku 4.5 (the
cheap tier); the model is overridable for cross-tier measurement.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.config import BUDGETS, INTENT_MODEL
from forge_mind.intent.schema import IntentSpec
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "intent_v1.md"


class IntentParser:
    """Stage 1: natural-language request -> structured IntentSpec."""

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or INTENT_MODEL

    def __call__(self, prompt: str) -> tuple[IntentSpec, LLMUsage]:
        user = f"Extract the IntentSpec from this part request:\n\n{prompt.strip()}\n"
        return self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=IntentSpec,
            model=self._model,
            max_tokens=BUDGETS["intent"].max_tokens,
        )


__all__ = ["IntentParser"]
