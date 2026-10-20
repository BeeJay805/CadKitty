"""Adversarial acceptance judge: the FINAL ship/refuse gate.

Distinct from the loop's critic (which DRIVES revision and is calibrated to keep
iterating): this is an independent, skeptical reviewer that runs ONCE on the final
candidate and decides whether the part convincingly IS the intended object. It can
REFUSE -- a part the judge does not accept is reported as not-shippable rather than
shipped with a false PASS. Uses Opus (the nuanced tier) and the part's own render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.brief import DesignBrief
from forge_mind.config import OPUS
from forge_mind.llm.cost import LLMUsage
from forge_mind.llm.provider import LLMClientLike, make_client

JUDGE_MAX_TOKENS = 4000  # dense multi-feature verdicts need list room (2026-07-10)


class AcceptanceVerdict(BaseModel):
    """The final gate's verdict on whether the part is shippable as the intended object."""

    model_config = ConfigDict(extra="forbid")

    is_the_object: bool = Field(
        ..., description="True ONLY if the part convincingly reads as the intended object."
    )
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence in the call.")
    present_features: list[str] = Field(
        default_factory=list, description="Key features that ARE clearly present and correct."
    )
    missing_or_wrong: list[str] = Field(
        default_factory=list, description="Key features absent or clearly wrong. Empty if accepted."
    )
    reason: str = Field(..., description="One-paragraph rationale for the verdict.")


ACCEPT_SYSTEM = """You are the FINAL acceptance gate for an AI CAD pipeline. A part ships to the \
user only if you accept it, so be SKEPTICAL and rigorous -- it is far worse to ship a part that is \
not really the requested object than to reject a borderline one.

You are shown ONE image: a six-panel render (isometric, front, right, rear, top, bottom) of a single 3D part, plus \
the intended REAL object and the key features it must have. Decide whether the part CONVINCINGLY \
reads as that object.

HOW TO JUDGE
- Look for the defining silhouette and the key features. A claw hammer MUST have a compact head \
and a long slender handle AND a split/curved claw; a paneled door MUST have recessed panels; a \
bracket MUST have the right angle and mounting holes.
- Actively try to find why it is NOT the object: wrong proportions (a dumbbell, not a hammer), a \
missing defining feature, parts that look fused wrong, or a generic blob.
- IGNORE rendering artifacts: faceting on curves, shading, color, aliasing. Judge SHAPE and \
FEATURE PRESENCE, not smoothness or exact millimeters.

DECIDE
- is_the_object = true ONLY if the silhouette and the KEY defining features clearly read as the \
intended object. If a defining feature is missing or the proportions are wrong, set false.
- confidence: high / medium / low.
- present_features: the key features clearly present.
- missing_or_wrong: the key features absent or wrong (empty if accepted).
- reason: one concise paragraph; name the views you relied on.

Hold the bar high but judge fairly: a clearly-correct part with minor imperfections still passes."""


def acceptance_judge(
    render_path: Path,
    brief: DesignBrief,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
    render_note: str | None = None,
) -> tuple[AcceptanceVerdict, LLMUsage]:
    """Run the final adversarial gate on a part's render. Uses Opus by default.

    ``render_note`` (optional): a TRUE statement about how the render was produced -- e.g.
    an articulated assembly rendered at its HOME pose (all joints at 0, links stacked).
    It removes a systematic false-negative (judging a folded-up mechanism for not striking
    the posed silhouette) WITHOUT touching the verdict criteria: the judge still requires
    every key feature present. None (every legacy caller) = byte-identical prompt."""
    client = client or make_client()
    user = _judge_user(brief)
    if render_note:
        user += f"\n\nRENDER NOTE (a fact about how the image was produced): {render_note}"
    return client.generate_structured_with_image(
        system=ACCEPT_SYSTEM,
        user=user,
        image_png=render_path.read_bytes(),
        output_model=AcceptanceVerdict,
        model=model or OPUS,
        max_tokens=JUDGE_MAX_TOKENS,
    )


def _judge_user(brief: DesignBrief) -> str:
    lines = [
        "Here is the rendered six-view image of the finished part.",
        "",
        f"INTENDED OBJECT: {brief.object_name}",
        f"It should look like: {brief.visual_description}",
        "",
        "KEY FEATURES it must have:",
    ]
    lines.extend(f"  - {f.name}: {f.purpose}" for f in brief.features)
    lines.append("")
    lines.append(
        "Decide whether this part convincingly IS that object, following your instructions."
    )
    return "\n".join(lines)


__all__ = ["ACCEPT_SYSTEM", "AcceptanceVerdict", "acceptance_judge"]
