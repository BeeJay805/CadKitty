"""Fail-closed context-window guard for the local Ollama roster (ADR-0115).

Measured failure (2026-07-15 planetary RCA): the author prompt reached the local coder as a
28-29K-character message but the server evaluated exactly 4098 prompt tokens -- Ollama silently
truncates any prompt that does not fit ``num_ctx``, so the model builds from a clipped prompt and
the pipeline burns repair iterations on output nobody can attribute. Nothing in the client layer
detected it. These helpers make that state LOUD, on both sides of the call:

- ``check_fits`` (pre-flight): a prompt whose conservative token lower bound cannot fit beside the
  generation budget is refused BEFORE the GPU call -- no wasted inference on guaranteed truncation.
- ``check_not_truncated`` (post-hoc): after the call, the server's own ``prompt_eval_count`` is the
  ground truth. If it is far below the prompt's token lower bound AND sits near the context-window
  cap (the signature of a clipped window), the call raises instead of returning silently-wrong output.

The near-the-cap condition is load-bearing: Ollama prefix-caches repeated prompts, so a re-rolled
best-of-N attempt legitimately reports a tiny ``prompt_eval_count``. A cached call sits far BELOW
the cap; a truncated one sits AT it. Only the truncation signature raises.

Token counts are estimated from characters with a deliberate LOWER bound (6 chars/token -- real
prose/code averages ~3.5-4), so the guard only fires when overflow is unambiguous. Kill switch:
``FORGE_OLLAMA_CTX_GUARD=0`` (default on -- fail-closed everywhere, per the constitution).
"""

from __future__ import annotations

import os

from forge_mind.llm.client import LLMError

# Lower bound: no realistic English/code prompt averages more than 6 characters per token, so
# chars/6 UNDERestimates the true token count -- fires only on unambiguous overflow.
_CHARS_PER_TOKEN_LOWER = 6.0
# Post-hoc slack: require the lower bound to exceed the evaluated count by 10% before raising, so
# a partially prefix-cached call (evaluated = new tokens only) cannot false-fire at the margin.
_TRUNCATION_SLACK = 1.1
# "Near the cap" = evaluated count >= 80% of the clip point. MEASURED (live probe 2026-07-18,
# reproducing the RCA number exactly): Ollama clips an over-window prompt to ~num_ctx/2 -- a
# 69,599-char prompt at num_ctx=8192 evaluated 4098 prompt tokens, the same 4098 the 2026-07-15
# RCA logged. So the clip point is min(num_ctx - num_predict, num_ctx/2), and a cached repeat
# (tiny evaluated count) still sits far below it.
_NEAR_CAP_FRACTION = 0.8

_GUARD_ENV = "FORGE_OLLAMA_CTX_GUARD"

# Upper bound for SIZING (distinct from the guard's lower bound): the author system prompt
# measured 3.52 chars/token live on qwen3-coder:30b (28,192 chars -> 8,004 tokens, 2026-07-18),
# so chars/3 OVERestimates the token count -- a window sized by it cannot truncate.
_CHARS_PER_TOKEN_UPPER = 3.0
# Grow the window in whole steps so a role's num_ctx stays stable across similar-sized calls
# (an option change reloads the Ollama runner -- flip-flopping sizes would thrash the load).
_CTX_STEP = 4096
# Measured cap: qwen3-coder:30b at num_ctx=16384 is 100% GPU-resident on the 24GB card
# (2026-07-18). Env-overridable for other cards/models.
_MAX_CTX_ENV = "FORGE_OLLAMA_MAX_CTX"
_DEFAULT_MAX_CTX = 16384


def guard_enabled() -> bool:
    return os.environ.get(_GUARD_ENV, "1").strip() != "0"


def min_prompt_tokens(prompt_chars: int) -> int:
    """Conservative LOWER bound on the token count of a prompt of ``prompt_chars`` characters."""
    return int(prompt_chars / _CHARS_PER_TOKEN_LOWER)


def max_ctx() -> int:
    try:
        return int(os.environ.get(_MAX_CTX_ENV, str(_DEFAULT_MAX_CTX)))
    except ValueError:
        return _DEFAULT_MAX_CTX


def fit_num_ctx(*, prompt_chars: int, num_predict: int, floor: int) -> int:
    """The num_ctx to request: grown (in ``_CTX_STEP`` steps) to fit an UPPER-bound token estimate
    of the prompt plus the generation budget, clamped to [floor, max_ctx()].

    The 07-15/07-18 measurements showed the fixed 8192 window silently truncated EVERY local
    build123d author call (the system prompt alone is 8,004 tokens). Growing the window only when
    the prompt needs it keeps small-prompt roles (CAD-Coder parts, vision) at the cheap floor while
    the author gets a window it actually fits in. When even max_ctx() cannot fit the prompt, the
    pre-flight guard raises -- fail closed, never silently clipped.
    """
    needed = int(prompt_chars / _CHARS_PER_TOKEN_UPPER) + num_predict + 64
    if needed <= floor:
        return floor
    stepped = ((needed + _CTX_STEP - 1) // _CTX_STEP) * _CTX_STEP
    return max(floor, min(stepped, max(max_ctx(), floor)))


# The engine's cloud-sized generation ceilings (e.g. the author's 16,000-token "big visions"
# budget, row 54) do not translate to a 16K local window: MEASURED author outputs run 471-747
# tokens typically. Locally, max_tokens is a BUDGET, not a need -- cap what we ASK the window to
# reserve, then give the model whatever room the window has left after the prompt. The cap was
# first set 4096 and MEASURED TOO TIGHT (2026-07-18 arm O): legitimate long author outputs hit it
# mid-JSON-string (cut at ~13,100 chars ~= 4K tokens) and failed schema validation twice ->
# author_failed. 6144 fits the room the 16K window leaves beside the ~9.5K-token author prompt.
_MAX_PREDICT_ENV = "FORGE_OLLAMA_MAX_PREDICT"
_DEFAULT_MAX_PREDICT = 6144
# Below this output budget a CAD authoring call cannot return a usable script -- refuse instead
# of letting the window squeeze generation into garbage.
_MIN_PREDICT = 512


def _max_predict() -> int:
    try:
        return int(os.environ.get(_MAX_PREDICT_ENV, str(_DEFAULT_MAX_PREDICT)))
    except ValueError:
        return _DEFAULT_MAX_PREDICT


def plan_window(
    *, prompt_chars: int, requested_predict: int, floor: int, label: str
) -> tuple[int, int]:
    """Return ``(num_ctx, num_predict)`` for a local call: the window grown to fit the prompt plus
    a CAPPED generation budget, and the budget clamped to the room the window actually has left
    after an upper-bound prompt estimate.

    Raises ``LLMError`` (fail closed) when even ``_MIN_PREDICT`` output tokens cannot fit beside
    the prompt at ``max_ctx()`` -- the prompt is genuinely too big for the local model.
    """
    predict = min(requested_predict, _max_predict())
    num_ctx = fit_num_ctx(prompt_chars=prompt_chars, num_predict=predict, floor=floor)
    if guard_enabled():
        upper = int(prompt_chars / _CHARS_PER_TOKEN_UPPER)
        room = num_ctx - upper - 64
        if room < _MIN_PREDICT:
            lower = min_prompt_tokens(prompt_chars)
            if lower + _MIN_PREDICT > num_ctx:
                raise LLMError(
                    f"{label}: prompt cannot fit the local context window -- ~{lower}-{upper} "
                    f"prompt tokens ({prompt_chars} chars) leave under {_MIN_PREDICT} output "
                    f"tokens at num_ctx {num_ctx}. Shrink the prompt or raise "
                    f"FORGE_OLLAMA_MAX_CTX."
                )
            # Upper bound says tight but the lower bound still fits a minimum budget: proceed
            # with the minimum -- the post-hoc truncation check remains the backstop.
            predict = _MIN_PREDICT
        else:
            predict = min(predict, room)
    return num_ctx, predict


def check_fits(*, prompt_chars: int, num_ctx: int, num_predict: int, label: str) -> None:
    """Raise ``LLMError`` when the prompt cannot possibly fit beside the generation budget.

    Uses the token LOWER bound, so a raise means certain truncation -- the call would burn GPU time
    to produce output built from a clipped prompt.
    """
    if not guard_enabled():
        return
    lower = min_prompt_tokens(prompt_chars)
    if lower + num_predict > num_ctx:
        raise LLMError(
            f"{label}: prompt cannot fit the local context window -- at least ~{lower} prompt "
            f"tokens ({prompt_chars} chars) + num_predict {num_predict} > num_ctx {num_ctx}. "
            f"Ollama would silently truncate the prompt (the 2026-07-15 RCA failure). Shrink the "
            f"prompt, lower max_tokens, or raise FORGE_OLLAMA_NUM_CTX."
        )


def check_not_truncated(
    *,
    prompt_chars: int,
    prompt_eval_count: int,
    num_ctx: int,
    num_predict: int,
    label: str,
) -> None:
    """Raise ``LLMError`` when the server's reported ``prompt_eval_count`` carries the truncation
    signature: far below the prompt's token lower bound AND pinned near the context-window cap.

    ``prompt_eval_count <= 0`` (server did not report) is not judged -- nothing to measure.
    """
    if not guard_enabled() or prompt_eval_count <= 0:
        return
    lower = min_prompt_tokens(prompt_chars)
    clip_point = max(min(num_ctx - num_predict, num_ctx // 2), 1)
    near_cap = prompt_eval_count >= _NEAR_CAP_FRACTION * clip_point
    if lower > prompt_eval_count * _TRUNCATION_SLACK and near_cap:
        raise LLMError(
            f"{label}: the local model saw a TRUNCATED prompt -- server evaluated "
            f"{prompt_eval_count} prompt tokens but the prompt is at least ~{lower} tokens "
            f"({prompt_chars} chars) against num_ctx {num_ctx} / num_predict {num_predict}. "
            f"The output was built from a clipped prompt; refusing to return it. Shrink the "
            f"prompt or raise FORGE_OLLAMA_NUM_CTX."
        )


__all__ = [
    "check_fits",
    "check_not_truncated",
    "fit_num_ctx",
    "guard_enabled",
    "max_ctx",
    "min_prompt_tokens",
    "plan_window",
]
