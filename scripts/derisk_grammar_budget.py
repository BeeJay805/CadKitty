"""ADR-0019 Step-4/6 grammar-budget probe against the REAL Anthropic compiled-grammar limit.

The offline char-count proxy (model_json_schema length) is NOT faithful: the 3-member wrapper
passed it but the API rejects it ("compiled grammar is too large"). This probe tests candidate
wrappers against the real limit with a tiny generate call (a rejected grammar-compile costs
~$0). It tells us whether a TRIMMED computed member fits, or whether we must STOP.

Run: uv run python scripts/derisk_grammar_budget.py
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from forge_mind.config import LAYOUTSPEC_MODEL
from forge_mind.layout.composable_spec import ComposablePlateSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.pi4_spec import ElectronicsTraySpec
from forge_mind.layout.spec import Pattern
from forge_mind.llm.client import LLMClient
from forge_mind.types import Vec3


class W2(BaseModel):  # production baseline (known-good)
    model_config = ConfigDict(extra="forbid")
    plate: ComposablePlateSpec | None = None
    pi4_case: ElectronicsTraySpec | None = None


class W3Full(BaseModel):  # current (rejected)
    model_config = ConfigDict(extra="forbid")
    plate: ComposablePlateSpec | None = None
    pi4_case: ElectronicsTraySpec | None = None
    computed: ComputedBodySpec | None = None


class _StrutFlat(BaseModel):
    """Trimmed: op inlined (no nested ComputedOp union), material + max_overhang dropped."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    pattern: Literal[Pattern.COMPUTED_BODY] = Pattern.COMPUTED_BODY
    name: str
    envelope_mm: Vec3
    min_wall_mm: float
    op_kind: Literal["faceted_strut"] = "faceted_strut"
    cross_section: Literal["rectangular", "hexagonal"]


class W3Trim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plate: ComposablePlateSpec | None = None
    pi4_case: ElectronicsTraySpec | None = None
    computed: _StrutFlat | None = None


def _probe(client: LLMClient, name: str, model_cls: type[BaseModel]) -> None:
    try:
        client.generate_structured(
            system="Return an empty object.",
            user="{}",
            output_model=model_cls,
            model=LAYOUTSPEC_MODEL,
            max_tokens=16,
        )
        print(f"  {name:<10} FITS (grammar compiled)")
    except Exception as e:
        msg = str(e)
        verdict = (
            "OVER (grammar too large)" if "grammar" in msg.lower() else f"ERROR: {type(e).__name__}"
        )
        print(f"  {name:<10} {verdict}")


def main() -> None:
    client = LLMClient()
    print("Grammar-budget probe vs the REAL Anthropic compiled-grammar limit:")
    _probe(client, "W2", W2)  # plate + pi4_case (expect FITS)
    _probe(client, "W3Full", W3Full)  # + full computed (expect OVER)
    _probe(client, "W3Trim", W3Trim)  # + trimmed computed (the question)


if __name__ == "__main__":
    main()
