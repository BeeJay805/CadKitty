"""Isolated author A/B — does FORGE_CLAUDE_EFFORT=low cut the author call WITHOUT breaking the build?

Removes front-door + path nondeterminism: research ONE fixed brief, then author it under effort=high
(the CLI default) vs effort=low, and for EACH record output tokens, wall-clock, AND whether the authored
code actually BUILDS in the sandbox (a watertight solid that exports). $0 on the Opus subscription.

Run: $env:FORGE_LLM_PROVIDER='claude_code'; uv run --no-sync python -u scripts/author_effort_ab.py
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from forge_mind.agentic.author import author_initial
from forge_mind.agentic.brief import research_brief
from forge_mind.agentic.sandbox import run_build123d_code
from forge_mind.llm.claude_code_client import ClaudeCodeClient
from forge_mind.llm.provider import make_client

PROMPT = (
    "A clamp-on desk-edge cable barrier rail: a 350 mm long constant-cross-section part that clips over "
    "a 19 mm thick desk edge with a sprung C-clamp and has an open-top channel along the top to drop 6 "
    "cables (3-6 mm) in and out without threading. PLA, 4 mm walls."
)
TRIALS = ["low", "low", "low"]  # low only: the heavy high trial timed out under throttle; high is
# documented (30962 tok / 413s baseline). These confirm low-effort author CONSISTENCY + tokens.
BASELINE_HIGH_TOK = 30962
BASELINE_HIGH_S = 413.0


def _trial(effort: str, brief: Any, client: Any, out: Path) -> dict[str, Any]:
    if effort:
        os.environ["FORGE_CLAUDE_EFFORT"] = effort
    else:
        os.environ.pop("FORGE_CLAUDE_EFFORT", None)
    label = effort or "default(high)"
    try:
        t = time.perf_counter()
        authored, usage = author_initial(brief, client=client)
        author_s = time.perf_counter() - t
        res = run_build123d_code(authored.code, out, part_name="part")
        return {
            "effort": label,
            "out_tokens": usage.output_tokens,
            "author_s": round(author_s, 1),
            "code_chars": len(authored.code),
            "built": bool(getattr(res, "ok", False)),
            "error": getattr(res, "error_type", None),
        }
    except Exception as e:  # throttle timeout / API error: record it and keep going
        return {
            "effort": label,
            "out_tokens": None,
            "author_s": None,
            "code_chars": None,
            "built": False,
            "error": repr(e)[:120],
        }


def main() -> None:
    # Raise the per-call timeout above the 600s default: under a throttled window even a ~413s
    # high-effort author can queue past 600s. Subscription-only diagnostic.
    client = (
        ClaudeCodeClient(timeout_s=900)
        if os.environ.get("FORGE_LLM_PROVIDER") == "claude_code"
        else make_client()
    )
    out = Path("benchmarks/results/author_effort_ab")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["FORGE_CLAUDE_EFFORT"] = (
        "low"  # shared brief at low effort: fast + throttle-resilient
    )
    print(f"research brief (once, effort=low): {PROMPT[:55]}...", flush=True)
    brief, bu = research_brief(PROMPT, client=client)
    print(
        f"  brief: {brief.object_name} | target={brief.target} | "
        f"{len(brief.features)} features | brief_out_tokens={bu.output_tokens}",
        flush=True,
    )
    rows: list[dict[str, Any]] = []
    for i, effort in enumerate(TRIALS):
        print(
            f"\n[{i + 1}/{len(TRIALS)}] author effort={effort or 'default(high)'} ...", flush=True
        )
        r = _trial(effort, brief, client, out / f"t{i + 1}")
        rows.append(r)
        print(
            f"  -> out_tokens={r['out_tokens']} author_s={r['author_s']} "
            f"code_chars={r['code_chars']} built={r['built']} err={r['error']}",
            flush=True,
        )

    print(
        f"\n==== AUTHOR EFFORT A/B (low vs documented baseline {BASELINE_HIGH_TOK} tok / {BASELINE_HIGH_S}s) ===="
    )
    print(f"{'effort':<14}{'out_tok':>9}{'author_s':>10}{'code_ch':>9}{'built':>7}{'err':>5}")
    for r in rows:
        tok = "-" if r["out_tokens"] is None else str(r["out_tokens"])
        sec = "-" if r["author_s"] is None else f"{r['author_s']:.1f}"
        ch = "-" if r["code_chars"] is None else str(r["code_chars"])
        print(
            f"{r['effort']:<14}{tok:>9}{sec:>10}{ch:>9}{r['built']!s:>7}{'Y' if r['error'] else '':>5}"
        )
    ok = [r for r in rows if r["out_tokens"] is not None]
    if ok:
        mlt = sum(r["out_tokens"] for r in ok) / len(ok)
        mls = sum(r["author_s"] for r in ok) / len(ok)
        built = sum(1 for r in ok if r["built"])
        print(
            f"\nlow(mean of {len(ok)}): {mlt:.0f} tok / {mls:.0f}s  vs  baseline "
            f"{BASELINE_HIGH_TOK} tok / {BASELINE_HIGH_S}s  ->  {BASELINE_HIGH_TOK / max(mlt, 1):.1f}x "
            f"fewer tokens, {BASELINE_HIGH_S / max(mls, 1):.1f}x faster | built {built}/{len(ok)}"
        )
    else:
        print("\nall trials failed (throttle?) — window not recovered; re-run later.")


if __name__ == "__main__":
    main()
