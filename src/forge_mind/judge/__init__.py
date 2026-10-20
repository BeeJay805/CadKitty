"""VLM intent-match judge package — **ADVISORY ONLY**.

Renders a generated part and asks a vision model whether it plausibly satisfies
the original request. It NEVER gates, rejects, or raises into a caller's pass/fail
(see ``vlm_judge`` for the full contract). Use only to flag/log/surface.
"""

from __future__ import annotations

from forge_mind.judge.vlm_judge import (
    JUDGE_MODEL,
    JUDGE_SYSTEM_PROMPT,
    IntentJudgment,
    judge_intent_match,
)

__all__ = [
    "JUDGE_MODEL",
    "JUDGE_SYSTEM_PROMPT",
    "IntentJudgment",
    "judge_intent_match",
]
