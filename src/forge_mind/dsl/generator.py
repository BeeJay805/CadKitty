"""DSLGenerator — Blueprint -> PartProgram via the LLM (highest-risk stage).

The model emits a full PartProgram (geometry + interface declarations). We then
override metadata + constraints from the blueprint, so the validator checks the
geometry against the blueprint's promises rather than whatever the model echoed.
The model's real job is the geometry and where the interfaces sit.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.blueprint.schema import Blueprint
from forge_mind.config import BUDGETS, DSL_MODEL
from forge_mind.dsl.draft import DSLDraft, draft_to_program
from forge_mind.dsl.program import PartProgram
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

_PROMPT_PATH = Path(__file__).parent / "prompts" / "dsl_v1.md"


class DSLGenerator:
    """Stage 3: structured blueprint -> compilable PartProgram.

    The model emits a DSLDraft (solids + holes + interfaces); we assemble the
    PartProgram so holes are always subtracted and metadata/constraints come
    from the blueprint.
    """

    def __init__(self, client: LLMClientLike | None = None, model: str | None = None) -> None:
        self._client = client or make_client()
        self._system = _PROMPT_PATH.read_text(encoding="utf-8")
        self._model = model or DSL_MODEL

    def __call__(self, blueprint: Blueprint) -> tuple[PartProgram, LLMUsage]:
        user = (
            "Design the part for this Blueprint. Emit a DSLDraft: the solids to "
            "add and the holes to cut, plus the interface declarations.\n\n"
            "Blueprint (JSON):\n"
            f"{blueprint.model_dump_json(indent=2)}\n"
        )
        draft, usage = self._client.generate_structured(
            system=self._system,
            user=user,
            output_model=DSLDraft,
            model=self._model,
            max_tokens=BUDGETS["dsl"].max_tokens,
        )
        return draft_to_program(draft, blueprint), usage


__all__ = ["DSLGenerator"]
