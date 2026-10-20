"""Reject-sampling HARVEST for the 4090 flywheel (founder-directed 2026-07-14).

Runs the LOCAL engine ($0, no cloud) across the procedural prompt bank (scripts/gen_prompts.py)
and lets the training-data tap (forge_mind.train_log) append every (prompt -> code, validator
verdict) pair to ~/.forge-train/parts.jsonl. The validator's verdict IS the reward label a
reject-sampling / DPO fine-tune filters on: keep hard_ok=True as SFT targets, pair pass-vs-fail
for DPO. More diverse prompts x attempts = more labeled data, bounded only by GPU time.

CURRICULUM: process easy tiers first (the model's early positives come from simple parts; the
reject filter self-curricula the rest as its pass-rate climbs).

PLANNER (who writes the brief; the BUILDER is always qwen-coder, the model we're training):
    coder  -- qwen-coder briefs itself      (all local, no VRAM swap, fastest, LOW yield)
    local  -- qwen3:30b briefs, coder builds (all local, VRAM swap, higher yield)   [default]
    opus   -- Opus briefs, coder builds      (cloud teacher: highest yield, bills subscription)

ONE LLM AT A TIME: serial pipeline + self-healing PID lock + Ollama OLLAMA_NUM_PARALLEL=1.
WINDOWS SPAWN GUARD: FEA runs in spawned subprocesses -> all execution under __main__.

Run:  FORGE_LLM_PROVIDER=claude_code python -u scripts/harvest.py [planner] [attempts]
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SONNET = "claude-sonnet-4-6"
OPUS = "claude-opus-4-8"


def _client(planner: str) -> tuple[Any, dict[str, str]]:
    """Return (client, per-stage models). The builder we TRAIN is qwen-coder in the self-harvest
    modes; the TEACHER mode is the full production Claude pipeline (Opus author to dodge Fable's
    weekly cap; critic+judge SEE the render) -- it distills high-fidelity, non-primitive targets."""
    from forge_mind.llm.ollama_client import OllamaClient
    from plan_ab import QWEN, QWEN_THINK, LocalRoutingClient, RoutingClient  # reuse the routers

    if planner == "teacher":
        from forge_mind.llm.claude_code_client import ClaudeCodeClient

        return ClaudeCodeClient(), {
            "brief": SONNET,
            "author": OPUS,
            "critic": SONNET,
            "judge": OPUS,
        }
    if planner == "opus":
        return RoutingClient(), {"brief": OPUS, "author": QWEN, "critic": QWEN, "judge": QWEN}
    if planner == "local":
        return LocalRoutingClient(), {
            "brief": QWEN_THINK,
            "author": QWEN,
            "critic": QWEN,
            "judge": QWEN,
        }
    return OllamaClient(), {"brief": QWEN, "author": QWEN, "critic": QWEN, "judge": QWEN}


def main() -> int:
    from forge_mind.agentic.fidelity import fidelity_report
    from forge_mind.agentic.result import produce_part
    from forge_mind.fabrication import FabricationMode

    planner = sys.argv[1] if len(sys.argv) > 1 else "local"
    attempts = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    bank = Path(os.environ.get("FORGE_HARVEST_BANK", "benchmarks/harvest/prompts.jsonl"))
    raw = [json.loads(x) for x in bank.read_text(encoding="utf-8").splitlines() if x.strip()]
    # curriculum order: easy tiers first, but INTERLEAVE families within a tier (round-robin) so the
    # early dataset is DIVERSE from the start (else it grinds ~150 of one family before any variety).
    from collections import defaultdict

    buckets: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for p in raw:
        buckets[(p["tier"], p["family"])].append(p)
    for k in buckets:
        buckets[k].sort(key=lambda p: p["id"])
    prompts = []
    for tier in sorted({t for t, _ in buckets}):
        fams = [f for (t, f) in buckets if t == tier]
        i = 0
        while any(buckets[(tier, f)] for f in fams):
            f = fams[i % len(fams)]
            if buckets[(tier, f)]:
                prompts.append(buckets[(tier, f)].pop(0))
            i += 1

    out = Path("benchmarks/harvest/harvest_log.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if not r.get("infra_fail"):
                    done.add(r["key"])
    work = out.parent / "work"

    client, models = _client(planner)
    print(
        f"harvest: planner={planner} models={models} bank={bank.name} "
        f"attempts={attempts} prompts={len(prompts)} done={len(done)}",
        flush=True,
    )
    pass_by_tier: Counter[int] = Counter()
    n_by_tier: Counter[int] = Counter()

    for p in prompts:
        for a in range(attempts):
            key = f"{planner}|{p['id']}|{a}"
            if key in done:
                continue
            t0 = time.perf_counter()
            try:
                r = produce_part(
                    p["prompt"],
                    work / key.replace("|", "_"),
                    n_drafts=1,
                    max_iterations=6,
                    budget_usd=999,
                    fabrication_mode=FabricationMode.GENERAL,
                    brief_model=models["brief"],
                    author_model=models["author"],
                    critic_model=models["critic"],
                    judge_model=models["judge"],
                    client=client,
                )
                v = r.loop.validation
                hard_ok = bool(v.hard_ok) if v is not None else None
                # FIDELITY GATE: is this real construction or a primitive pile? (from the STEP + code)
                outdir = work / key.replace("|", "_")
                pcode = (
                    (outdir / "program.py").read_text(encoding="utf-8")
                    if (outdir / "program.py").exists()
                    else None
                )
                fid = fidelity_report(outdir / "part.step", pcode, p["prompt"])
                row = {
                    "key": key,
                    "planner": planner,
                    "family": p["family"],
                    "tier": p["tier"],
                    "attempt": a,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "hard_ok": hard_ok,
                    "disposition": r.disposition,
                    "fid_verdict": fid.verdict,
                    "fid_score": fid.score,
                    "fid_expected": fid.expected,
                    "fid_pile": fid.primitive_pile,
                    "infra_fail": False,
                }
                # sub-5s no-build = swallowed brief/author blip -> infra (retried, not a datum)
                if hard_ok is None and row["seconds"] < 5:
                    row["infra_fail"] = True
                else:
                    n_by_tier[p["tier"]] += 1
                    if hard_ok:
                        pass_by_tier[p["tier"]] += 1
            except Exception as e:  # fail-open: never let one blip stop the harvest
                msg = str(e)
                row = {
                    "key": key,
                    "planner": planner,
                    "family": p["family"],
                    "tier": p["tier"],
                    "attempt": a,
                    "seconds": round(time.perf_counter() - t0, 1),
                    "infra_fail": True,
                    "auth": "401" in msg or "authenticate" in msg.lower(),
                    "error": msg[:160],
                }
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            tag = "INFRA" if row["infra_fail"] else ("BUILD" if row.get("hard_ok") else "fail ")
            tr = f"T{p['tier']}"
            rate = ""
            if n_by_tier:
                rate = " | pass/tier " + " ".join(
                    f"{t}:{pass_by_tier[t]}/{n_by_tier[t]}" for t in sorted(n_by_tier)
                )
            print(
                f"[{planner:5} {tr} {p['family']:12} a{a}] {tag} {row['seconds']:5.0f}s{rate}",
                flush=True,
            )

    print(f"\nharvest done. pass/tier: {dict(pass_by_tier)} of {dict(n_by_tier)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
