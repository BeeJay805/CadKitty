"""LayoutSpecGenerator — Blueprint -> LayoutSpec (union) via the LLM (ADR-0005, ADR-0008).

The LLM chooses the matching pattern and fills ONLY that spec's sizes +
constraints. Its output type is a wrapper of THREE optional specs (one per known
pattern); it sets exactly one. None of the specs have an operations field, so the
model still cannot emit geometry — the deterministic interpreter owns every
coordinate.

A several-optional-fields wrapper is used instead of a raw discriminated union so
the emitted JSON schema stays a plain object (no oneOf), which structured output
handles reliably. The Python return type is still the LayoutSpec union.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from forge_mind.blueprint.schema import Blueprint
from forge_mind.config import BUDGETS, LAYOUTSPEC_MODEL
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.layoutspec import LayoutSpec
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "layoutspec_v3.md"


class GeneratedLayout(BaseModel):
    """LLM output (ADR-0014): the plate family collapsed to ONE composable spec, plus the
    separate electronics tray. Fill exactly one of `plate` or `pi4_case`. Two members
    (down from three) keeps the structured-output schema under the grammar budget.

    NOTE (ADR-0019): a third `computed` member (a structural strut) was attempted and
    REVERTED — the 2-member wrapper is already at the Anthropic compiled-grammar ceiling, so
    even a trimmed 3rd strict member is rejected ("compiled grammar is too large"). The
    deterministic computed-geometry path (ComputedBodySpec -> interpret_layout -> builder) is
    in place and validated; the LLM front-door awaits a structural grammar fix (a separate
    computed wrapper / a cheap router / hoisting shared coarse intent)."""

    model_config = ConfigDict(extra="forbid")

    plate: ComposablePlateSpec | None = None
    # Field name kept as `pi4_case` for backward compatibility with saved LLM output
    # (ADR-0007 Stage E); the type is the board-generic tray spec, unchanged.
    pi4_case: ElectronicsTraySpec | None = None


class LayoutSpecGenerator:
    """Stage: Blueprint -> LayoutSpec union. The LLM fills sizes + constraints."""

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or LAYOUTSPEC_MODEL

    def __call__(self, blueprint: Blueprint) -> tuple[LayoutSpec, LLMUsage]:
        user = (
            "Choose the base body + interfaces that match this Blueprint. For a plate "
            "family part (motor mount, adapter plate, fan mount) fill `plate`; for a "
            "board enclosure fill `pi4_case`. Copy envelope, material, and min_wall from "
            "the Blueprint; leave dims/offsets null so the interpreter derives them. Set "
            "exactly one of `plate` or `pi4_case` and leave the other null.\n\n"
            "Blueprint (JSON):\n"
            f"{blueprint.model_dump_json(indent=2)}\n"
        )
        choice, usage = self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=GeneratedLayout,
            model=self._model,
            max_tokens=BUDGETS["layoutspec"].max_tokens,
        )
        filled = [s for s in (choice.plate, choice.pi4_case) if s is not None]
        if len(filled) != 1:
            raise LLMError(
                f"LayoutSpecGenerator must fill exactly one of plate/pi4_case, got {len(filled)}."
            )
        return filled[0], usage


__all__ = ["GeneratedLayout", "LayoutSpecGenerator"]
