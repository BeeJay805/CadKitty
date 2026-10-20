"""BlueprintGenerator — IntentSpec -> Blueprint via Sonnet 4.6 (structured output)."""

from __future__ import annotations

from pathlib import Path

from forge_mind.blueprint.schema import Blueprint, BlueprintVec3
from forge_mind.config import BLUEPRINT_MODEL, BUDGETS
from forge_mind.intent.schema import IntentSpec
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "blueprint_v1.md"


class BlueprintGenerator:
    """Stage 2: structured intent -> structured blueprint.

    ``model`` defaults to the pinned BLUEPRINT_MODEL (Sonnet 4.6) but can be
    overridden to measure pass-rate across model tiers.
    """

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or BLUEPRINT_MODEL

    def __call__(self, intent: IntentSpec) -> tuple[Blueprint, LLMUsage]:
        user = (
            "Convert this IntentSpec into a Blueprint.\n\n"
            "IntentSpec (JSON):\n"
            f"{intent.model_dump_json(indent=2)}\n"
        )
        blueprint, usage = self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=Blueprint,
            model=self._model,
            max_tokens=BUDGETS["blueprint"].max_tokens,
        )
        # The HARD max is a USER limit, not an LLM choice: set it DETERMINISTICALLY from intent so it
        # is reliable (never hallucinated) and overrides whatever the LLM emitted. None when the user
        # stated no max -> the interpreter grows the envelope freely.
        mx = intent.constraints.max_envelope_mm
        if mx is None:
            hard = None  # the user stated no cap -> the interpreter grows the envelope freely
        elif len(mx) == 3:
            hard = BlueprintVec3(x=mx[0], y=mx[1], z=mx[2])
        else:
            # A present-but-malformed envelope (a reachable LLM-extraction state) must NOT be silently
            # dropped to "no cap": that erased the user's stated hard maximum (audit M8).
            raise ValueError(
                f"max_envelope_mm must be exactly 3 elements (x, y, z) or absent; got {mx!r}"
            )
        return blueprint.model_copy(update={"max_envelope_mm": hard}), usage


__all__ = ["BlueprintGenerator"]
