"""Central configuration: pinned model IDs and per-stage budgets.

Model IDs are pinned deliberately (ADR-worthy to change). Per the plan:
Sonnet 4.6 for the structured-reasoning stages (Blueprint, DSL), Haiku 4.5 for
the cheap transforms (Intent, Reviser-apply).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Pinned model IDs. Bumping these is an ADR-worthy change (re-baseline benchmarks).
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5"
# Frontier authoring model for the agentic CAD path: Opus writes the build123d code + judges the
# render. Required by agentic.author + agentic.judge (the agentic engine is the main /generate path).
OPUS = "claude-opus-4-8"
# Opus 5 (released 2026-07-24), SELECTABLE BUT NOT DEFAULT ANYWHERE (alias ``opus5``).
#
# Deliberately does NOT rebind ``OPUS``: that constant is the judge/critic model AND the default
# ``BRAIN_MODEL``, so re-pointing it would silently flip three stages at once -- and bumping a pin
# is ADR-worthy by the rule at the top of this file.
#
# Why it is wired at all: ADR-0076 chose FABLE over Opus as the author on measured one-shots
# (Opus no-built where Fable passed), but that verdict is against Opus 4.x and predates this model.
# Re-testing it is legitimate; it just needs to be an A/B arm, not a swap. Note the author's
# reported ceiling is already high (L88: 16/16 parts accepted) while the measured losses sit in the
# PLAN/architecture tier, so the planner arm (``FORGE_BRAIN_MODEL=opus5``) is the better first test.
OPUS_5 = "claude-opus-5"
# THE DEFAULT AUTHOR since 2026-07-02 (founder decision, ADR-0076): both one-shots passed where
# Opus no-built, incl. static balance 0.008 + the bladed compressor wheel -- and Fable is cheaper
# ($10/$50 vs $15/$75 per M). The judge/critic stay OPUS. Env-overridable via FORGE_AUTHOR_MODEL.
FABLE = "claude-fable-5"


def _resolve_brain_model(env_value: str) -> str:
    """Resolve the invention-lab / chief-engineer BRAIN model from ``FORGE_BRAIN_MODEL``.

    A known alias (opus/fable/sonnet/haiku) maps to its pinned id; any other non-empty value is used
    verbatim (so a future model id needs no code change); empty -> OPUS. Pure -> unit-testable without
    reimporting the module.
    """
    aliases = {
        "opus": OPUS,
        "opus5": OPUS_5,
        "fable": FABLE,
        "sonnet": SONNET,
        "haiku": HAIKU,
    }
    return aliases.get(env_value.strip().lower(), env_value.strip() or OPUS)


# The "one brain" model for the invention front door + chief-engineer planning (ADR-0046/0047/0055).
# Founder direction 2026-06-15: measure the lab LIVE on Opus via the subscription NOW rather than wait
# on Fable; keep the swap a one-line ENV flip with no code change -- set FORGE_BRAIN_MODEL=fable (or
# opus/sonnet/haiku, or a full model id) to override. Default: OPUS.
BRAIN_MODEL = _resolve_brain_model(os.environ.get("FORGE_BRAIN_MODEL", ""))

BLUEPRINT_MODEL = SONNET
DSL_MODEL = SONNET
INTENT_MODEL = HAIKU
REVISER_MODEL = HAIKU
# LayoutSpec generation (ADR-0004 A/B path): the LLM only chooses a pattern +
# dimensions + constraints (the interpreter owns geometry), so this is a
# Blueprint-like transform. Pinned to Sonnet to match DSL_MODEL for a fair A/B.
LAYOUTSPEC_MODEL = SONNET


@dataclass(frozen=True)
class StageBudget:
    """Per-stage guardrails. max_tokens caps output; max_cost_usd is advisory
    (the benchmark harness enforces the hard per-trial budget).
    """

    max_tokens: int
    max_cost_usd: float


# Agentic-path structured-output ceilings (founder 2026-07-10: "big visions" headroom).
# max_tokens is a CAP, not a spend -- billing is per ACTUAL token; a generous cap only
# lets dense multi-part briefs/plans COMPLETE. Truncation at the cap corrupts structured
# JSON DETERMINISTICALLY (TEST_LOG row 54: the engine probe died at a 2000-token brief,
# and the parse retry cannot save an under-provisioned budget). Ceiling rationale: stay
# under the SDK's ~21.3k non-streaming line (messages.parse cannot stream) and under
# every production model's output limit (Sonnet 64k, Opus 32k).
BRIEF_MAX_TOKENS = 16000
# The plan is the biggest structured object in the engine: a Fable chief writing 14 exact
# build cards MEASURABLY exceeds 16k (the i4 probe truncation). 32000 rides the client's
# streaming path (> _STREAM_OVER) and sits at Opus/Fable's output limit.
PLAN_MAX_TOKENS = 32000  # decompose + chief plan-revision (the multi-part plan JSONs)

BUDGETS = {
    "intent": StageBudget(max_tokens=2000, max_cost_usd=0.02),
    "blueprint": StageBudget(max_tokens=4000, max_cost_usd=0.08),
    "dsl": StageBudget(max_tokens=8000, max_cost_usd=0.15),
    "reviser": StageBudget(max_tokens=8000, max_cost_usd=0.10),
    "layoutspec": StageBudget(max_tokens=2000, max_cost_usd=0.05),
    # ADR-0020: the cheap path-classifier (Stage A) emits only a GenerationPath enum.
    "router": StageBudget(max_tokens=256, max_cost_usd=0.01),
    # ADR-0037: the /patch edit mapper emits one (field, op, value) pick.
    "spec_patch": StageBudget(max_tokens=512, max_cost_usd=0.01),
}

# Automated scaling engine (ADR-0016): the $-budget scale-run is BLOCKED until the
# validator (the oracle) is calibrated against physical prints. The bottleneck is oracle
# TRUST, not budget -- at scale a miscalibrated oracle banks thousands of confidently-wrong
# passes. Flip this to True ONLY after a few engine-banked families are printed + confirmed
# (validator-pass => physically-correct, assemblable part). The small-scale loop validation
# (`forge-mind scale --validate`) is NOT gated by this.
SCALING_PRINT_CALIBRATED = False


__all__ = [
    "BLUEPRINT_MODEL",
    "BRAIN_MODEL",
    "BUDGETS",
    "DSL_MODEL",
    "FABLE",
    "HAIKU",
    "INTENT_MODEL",
    "LAYOUTSPEC_MODEL",
    "OPUS",
    "REVISER_MODEL",
    "SCALING_PRINT_CALIBRATED",
    "SONNET",
    "StageBudget",
]
