"""VLM intent-match judge — **ADVISORY ONLY**.

**This tool is ADVISORY. It NEVER gates, NEVER rejects, and NEVER raises into a
caller's pass/fail. Callers MUST NOT use its output to reject, fail, or block a
part.** A wrong VLM verdict on a genuinely-correct part would be a false-FAIL; to
make that impossible, the judge is structurally incapable of producing a FAIL
that any gate could read: on ANY internal error it returns ``match=None`` ("could
not judge") and it never propagates exceptions. Use it only to flag, log, or
surface a *possible* intent mismatch for a human to look at.

What it does: render a generated part to a 3-view PNG, send that image plus the
ORIGINAL natural-language request to a vision model, and return a structured,
calibrated judgment of whether the part *plausibly* contains the features the
request named.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from forge_mind.config import SONNET
from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.viz.render import render_part_to_png

# Vision-capable model. Sonnet 4.6 is multimodal; this is the default the lead
# may override by passing ``model=``.
JUDGE_MODEL = SONNET
JUDGE_MAX_TOKENS = 1024


class IntentJudgment(BaseModel):
    """Advisory judgment of whether a rendered part plausibly satisfies a request.

    ``match`` is tri-state on purpose:
      - ``True``  -> every named feature plausibly present (or uncertain-but-fine).
      - ``False`` -> at least one named feature is convincingly ABSENT or WRONG.
      - ``None``  -> could-not-judge (render/LLM/parse error). NOT a failure.

    ``None`` must be treated as "no signal", never as a fail.
    """

    match: bool | None = Field(
        ...,
        description="True=plausibly satisfies; False=clear mismatch; None=could-not-judge.",
    )
    missing_or_wrong: list[str] = Field(
        default_factory=list,
        description="Specific requested features that appear ABSENT or WRONG in the render.",
    )
    reason: str = Field(..., description="One-paragraph rationale for the verdict.")
    cost_usd: float = Field(default=0.0, description="USD cost of the vision call (0 on error).")


class _VlmVerdict(BaseModel):
    """The raw structured output we ask the vision model for (mapped into IntentJudgment)."""

    match: bool = Field(
        ...,
        description="True if EVERY feature the request names is plausibly present; else False.",
    )
    missing_or_wrong: list[str] = Field(
        default_factory=list,
        description="Each requested feature that looks clearly absent or clearly the wrong type/count.",
    )
    reason: str = Field(..., description="Concise rationale, one short paragraph.")


# ---------------------------------------------------------------------------
# The judge system prompt. MAX EFFORT — this is the deliverable. Calibrated to
# prefer match=true under uncertainty so the advisory tool never invents a
# false mismatch on a good part.
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """\
You are a meticulous mechanical-design reviewer performing a single, narrow check.

You are shown ONE image: a rendered three-panel view (isometric, top-down, and \
front) of a single 3D-printable mechanical part. You are also given the user's \
original natural-language request that the part was generated to satisfy.

YOUR ONLY JOB
Judge whether the part in the image PLAUSIBLY contains the FEATURES that the \
request explicitly NAMES. Nothing else. You are a feature-presence checker, not \
a dimensional inspector, not an aesthetics critic, and not a print-quality grader.

WHAT COUNTS AS A "NAMED FEATURE"
Read the request and extract the concrete, checkable features it asks for. Judge \
only those. Examples of how to map common requests to what you should look for:
  - "four holes" / "4 mounting holes"  -> roughly four through-holes are visible \
    (count is APPROXIMATE; see calibration below).
  - "a hole" / "a bore" / "a central bore" -> at least one circular opening, \
    typically centered.
  - "a 608 bearing seat" / "a bearing pocket" -> a circular, shouldered \
    seat/recess/boss sized to receive a bearing (a counterbored or stepped \
    circular feature), not just a plain through-hole.
  - "a fan mount" / "fan bracket" -> a roughly square bolt pattern (four corner \
    holes) around a large central circular bore (the airflow opening).
  - "a NEMA 17 mount" / "motor mount" -> a 4-bolt SQUARE pattern with a CENTRAL \
    bore (the motor's pilot boss/shaft hole), usually on a square-ish plate.
  - "a slot" / "slotted hole" -> an elongated (oblong) opening, not a round hole.
  - "a counterbore" / "countersink" -> a stepped or conical widening at a hole mouth.
  - "a bracket" / "L-bracket" / "angle bracket" -> two faces meeting at an angle \
    (an L or right-angle profile), often with holes on each face.
  - "a flat plate" / "base plate" -> a thin, broad, flat slab.
  - "a boss" / "standoff" / "pillar" -> a raised cylindrical protrusion.
Arrangement words are features too: "in a row" -> holes roughly collinear; \
"2x2 grid" / "square pattern" -> a roughly square four-point arrangement; \
"evenly spaced" / "on a circle" / "bolt circle" -> roughly regular spacing.

HOW TO USE THE THREE VIEWS
Use the TOP view to count and locate holes and to read bolt patterns / \
arrangement; it is the most reliable for hole layout. Use the FRONT view for \
side profile, steps/shoulders (bearing seats, counterbores), and overall \
proportions. Use the ISOMETRIC view for overall shape and to disambiguate. \
A feature only needs to be evident in ONE view to count as present. If a feature \
would naturally be hidden in a given view (e.g. a blind pocket seen edge-on), do \
not treat its absence in that view as evidence it is missing.

CALIBRATION — BE CONSERVATIVE ABOUT FLAGGING (this is critical)
This is an ADVISORY check. A false mismatch on a good part is the worst outcome \
and must be avoided. Therefore:
  - Default to match=true. Only set match=false when a NAMED feature is \
    CONVINCINGLY absent or is CLEARLY the wrong TYPE or wrong COUNT.
  - Counting is APPROXIMATE. Treat counts as a match within +/-1: a request for \
    "four holes" is satisfied by 3, 4, or 5 visible holes. Only flag a count if \
    it is off by more than one AND you are confident (e.g. "four holes" but the \
    part is plainly solid with zero holes, or has a dozen).
  - When you are UNCERTAIN whether a feature is present, treat it as PRESENT \
    (match stays true). Uncertainty is not evidence of absence.
  - Rendering artifacts are NOT mismatches. Ignore faceting/polygonal edges on \
    curved surfaces, color/shading, lighting, jagged silhouettes, aliasing, and \
    view framing. A faceted circle is still a circle.
  - Dimensions, exact sizes, tolerances, and proportions are OUT OF SCOPE unless \
    the request names a size that implies a TYPE (e.g. "608 bearing" implies a \
    shouldered seat, which is a type cue, not a measurement). Never flag a part \
    for being the "wrong size".
  - Aesthetics, smoothness, elegance, and "does it look professional" are OUT OF \
    SCOPE. Never flag for those.
  - Features the request did NOT name are irrelevant. Extra features are never a \
    mismatch. Do not flag a part for having MORE than was asked.

DECIDING match
  - match = true  : every feature the request names is plausibly present \
    (including the uncertain-so-assume-present cases above). missing_or_wrong \
    should be empty.
  - match = false : at least one NAMED feature is convincingly absent, or is \
    clearly the wrong type/count. List exactly those features in missing_or_wrong, \
    naming each one specifically (e.g. "central bore (no circular opening visible \
    in any view)", "4-bolt square pattern (only 2 holes visible)").

OUTPUT
Return the structured verdict:
  - match: boolean per the rule above.
  - missing_or_wrong: a list of the SPECIFIC named features that look absent or \
    wrong. Empty when match is true. Be specific and terse; one short phrase each.
  - reason: one concise paragraph explaining the verdict, citing which view(s) \
    you relied on. If you were uncertain about anything and resolved it toward \
    "present", say so briefly.

Remember: your verdict is advisory. When in doubt, say it matches."""


def judge_intent_match(
    stl_path: Path,
    prompt: str,
    *,
    work_dir: Path | None = None,
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> IntentJudgment:
    """Render ``stl_path``, ask a vision model if it plausibly satisfies ``prompt``.

    **ADVISORY ONLY.** Returns an :class:`IntentJudgment`. NEVER raises: on any
    render, LLM, or parse failure it returns ``match=None`` with a "could not
    judge" reason. Callers MUST NOT use the result to gate, reject, or fail a part.

    ``work_dir`` is where the temporary render PNG is written (a tempdir if None).
    ``client`` defaults to the provider factory's client (``make_client()``). ``model`` defaults to
    :data:`JUDGE_MODEL` (a multimodal Claude model).
    """
    used_model = model or JUDGE_MODEL

    try:
        with tempfile.TemporaryDirectory() as td:
            out_dir = work_dir if work_dir is not None else Path(td)
            out_dir.mkdir(parents=True, exist_ok=True)
            png_path = out_dir / f"{stl_path.stem}__judge.png"
            render_part_to_png(stl_path, png_path, title=stl_path.stem, caption=prompt)
            image_png = png_path.read_bytes()

            judge_client = client if client is not None else make_client()
            verdict, usage = judge_client.generate_structured_with_image(
                system=JUDGE_SYSTEM_PROMPT,
                user=_build_user_prompt(prompt),
                image_png=image_png,
                output_model=_VlmVerdict,
                model=used_model,
                max_tokens=JUDGE_MAX_TOKENS,
            )
    except Exception as exc:  # advisory: never raise, never gate.
        return IntentJudgment(
            match=None,
            missing_or_wrong=[],
            reason=f"could not judge: {exc}",
            cost_usd=0.0,
        )

    return IntentJudgment(
        match=verdict.match,
        missing_or_wrong=list(verdict.missing_or_wrong),
        reason=verdict.reason,
        cost_usd=usage.cost_usd,
    )


def _build_user_prompt(prompt: str) -> str:
    """The text turn that accompanies the image: restate the task, pin the request."""
    return (
        "Here is the rendered three-view image of the generated part.\n\n"
        "The user's original request was:\n"
        f"<request>\n{prompt}\n</request>\n\n"
        "Judge whether the part plausibly contains the features this request names, "
        "following your instructions. Be conservative about flagging; when in doubt, "
        "treat the feature as present and return match=true."
    )


__all__ = [
    "JUDGE_MODEL",
    "JUDGE_SYSTEM_PROMPT",
    "IntentJudgment",
    "judge_intent_match",
]
