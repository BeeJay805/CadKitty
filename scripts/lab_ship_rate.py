"""Deterministic SHIP-RATE harness — compare engine configs on a FIXED concept, N trials each.

The lab_pipeline_probe runs the whole nondeterministic pipeline, so each run draws a DIFFERENT
concept/part — single-sample ship comparisons across configs are uninterpretable (the confound that
made "high shipped / low rejected / medium rejected" meaningless: three different parts). This harness
removes that confound: it FIXES the concept (hardcoded prompt + STATED-FACTS, bypassing the front
door) and runs the full author->critic->judge loop N times per config, reporting the SHIP rate (judge
acceptance, not just "did it build") with a Wilson CI + latency + a throttle rate.

Run (subscription, $0):
  $env:FORGE_LLM_PROVIDER='claude_code'
  uv run --no-sync python -u scripts/lab_ship_rate.py [n_trials] [cooldown_s]

Honesty rails baked in: the build cache is FORCE-DISABLED (else trial 2..N replay trial 1 and report a
fake 0%/100% with no variance), ship rate is reported over EFFECTIVE-N (throttle-errored trials excluded
and surfaced separately), and small-N rates carry a Wilson 95% CI so 3/5-vs-2/5 is not over-read.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# The build cache would replay a prior part for an identical fixed prompt and fake the ship rate.
os.environ.pop("FORGE_BUILD_CACHE", None)

from forge_mind.agentic.result import produce_part
from forge_mind.llm.claude_code_client import ClaudeCodeClient
from forge_mind.llm.provider import make_client

# A FIXED concept (what the front door would have produced for the cable problem) + its STATED FACTS.
# Hardcoded so the ONLY variable across trials is the config under test.
PROMPT = (
    "Clamp-on edge rail with open-top channels: a 350 mm long constant-cross-section part that clips "
    "over a 19 mm thick desk edge with a sprung C-clamp and has an open-top channel along the top to "
    "drop 6 cables (3-6 mm) in and out without threading (solving: my desk cables keep sliding off the "
    "back edge of my desk onto the floor). PLA, 4 mm walls."
)
CONTEXT = (
    "STATED FACTS (the user's answers to clarifying questions -- treat as given):\n"
    "- desk edge: 19 mm laminated particleboard, flat, no lip\n"
    "- 6 cables, 3-6 mm diameter; must pop in and out (open slots, not threaded)\n"
    "- no-damage mounting preferred (clamp-on), PLA"
)
MAX_ITERATIONS = 8
BUDGET_USD = 3.0  # inert on subscription; bounds an API run


@dataclass(frozen=True)
class Config:
    name: str
    effort: str | None = None  # FORGE_CLAUDE_EFFORT value; None = CLI default (high)
    brief_model: str | None = None
    author_model: str | None = None
    critic_model: str | None = None
    judge_model: str | None = None


@dataclass
class TrialResult:
    config: str
    trial: int
    disposition: str | None = None
    shipped: bool = False
    geometry_passed: bool | None = None
    judge_is_object: bool | None = None
    judge_confidence: str | None = None
    n_iterations: int | None = None
    build_s: float | None = None
    error: str | None = None


# The default sweep answers the original question: holding the concept fixed, does low effort lower the
# SHIP rate? Add model-swap configs (e.g. Config("author=sonnet", author_model="claude-sonnet-4-6")) later.
CONFIGS = [
    Config("effort=high(default)", effort=None),
    Config("effort=low", effort="low"),
]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def run_trial(cfg: Config, client: Any, out: Path, trial: int) -> TrialResult:
    if cfg.effort:
        os.environ["FORGE_CLAUDE_EFFORT"] = cfg.effort
    else:
        os.environ.pop("FORGE_CLAUDE_EFFORT", None)
    try:
        t = time.perf_counter()
        pr = produce_part(
            PROMPT,
            out / f"{cfg.name}_t{trial}",
            client=client,
            extra_context=CONTEXT,
            max_iterations=MAX_ITERATIONS,
            budget_usd=BUDGET_USD,
            brief_model=cfg.brief_model,
            author_model=cfg.author_model,
            critic_model=cfg.critic_model,
            judge_model=cfg.judge_model,
        )
        build_s = round(time.perf_counter() - t, 1)
        return TrialResult(
            config=cfg.name,
            trial=trial,
            disposition=pr.disposition,
            shipped=pr.disposition in ("shipped", "shipped_with_caveats"),
            geometry_passed=pr.loop.geometry_passed,
            judge_is_object=pr.verdict.is_the_object if pr.verdict else None,
            judge_confidence=pr.verdict.confidence if pr.verdict else None,
            n_iterations=len(pr.loop.iterations),
            build_s=build_s,
        )
    except Exception as e:  # throttle timeout / API error: record + keep going
        return TrialResult(config=cfg.name, trial=trial, error=repr(e)[:160])


def summarize(rows: list[TrialResult]) -> None:
    print("\n" + "=" * 78)
    print(f"SHIP-RATE SUMMARY (fixed concept, max_iter={MAX_ITERATIONS}, cache OFF)")
    print(
        f"{'config':<22}{'N(eff)':>7}{'ship':>6}{'rate':>7}{'  95% CI':>14}"
        f"{'med_s':>8}{'mean_it':>8}{'throttle':>9}"
    )
    by_cfg: dict[str, list[TrialResult]] = {}
    for r in rows:
        by_cfg.setdefault(r.config, []).append(r)
    for name, rs in by_cfg.items():
        eff = [r for r in rs if r.error is None]
        n = len(eff)
        shipped = sum(1 for r in eff if r.shipped)
        rate = shipped / n if n else 0.0
        lo, hi = _wilson(shipped, n)
        secs = sorted(r.build_s for r in eff if r.build_s is not None)
        med = secs[len(secs) // 2] if secs else 0.0
        its = [r.n_iterations for r in eff if r.n_iterations is not None]
        mean_it = sum(its) / len(its) if its else 0.0
        throttle = sum(1 for r in rs if r.error is not None)
        print(
            f"{name:<22}{n:>7}{shipped:>6}{rate:>7.0%}{f'[{lo:.0%}-{hi:.0%}]':>14}"
            f"{med:>8.0f}{mean_it:>8.1f}{f'{throttle}/{len(rs)}':>9}"
        )
    print(
        "\n(rate over EFFECTIVE-N = non-throttled trials. CIs that overlap => indistinguishable; raise N.)"
    )


def main() -> None:
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    cooldown_s = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path("benchmarks/results") / f"ship_rate_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    client = (
        ClaudeCodeClient(timeout_s=900)
        if os.environ.get("FORGE_LLM_PROVIDER") == "claude_code"
        else make_client()
    )
    rows: list[TrialResult] = []
    print(
        f"ship-rate: {len(CONFIGS)} configs x {n_trials} trials, cooldown {cooldown_s:.0f}s, out={out}"
    )
    # All trials of one config, THEN the next: a throttle episode hits one config's tail, not the comparison.
    for cfg in CONFIGS:
        for trial in range(1, n_trials + 1):
            print(f"\n[{cfg.name} {trial}/{n_trials}] ...", flush=True)
            r = run_trial(cfg, client, out, trial)
            rows.append(r)
            print(
                f"  -> disp={r.disposition} shipped={r.shipped} iters={r.n_iterations} "
                f"build_s={r.build_s} judge_obj={r.judge_is_object} err={r.error}",
                flush=True,
            )
            (out / "trials.json").write_text(  # kill-safe: rewritten after every trial
                json.dumps([asdict(x) for x in rows], indent=2), encoding="utf-8"
            )
            if cooldown_s and trial < n_trials:
                time.sleep(cooldown_s)  # let the subscription window breathe between heavy trials
    summarize(rows)


if __name__ == "__main__":
    main()
