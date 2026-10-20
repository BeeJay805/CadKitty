"""PLANNER A/B: does a STRONG architect + LOCAL builder beat all-local? (the founder's hypothesis)

Holds the BUILDER constant (qwen3-coder does author + critic + judge) and varies ONLY the PLANNER
(the brief/decomposition) across the same 12 parts, so any difference is the planner's:

  opus_plan   brief = Opus (subscription)  |  author/critic/judge = qwen3-coder (local)
  qwen_plan   brief = qwen3-coder (local)  |  author/critic/judge = qwen3-coder (local)

The mix is one produce_part per part via a RoutingClient that dispatches each stage by model id
(a 'claude' model -> the subscription ClaudeCodeClient; anything else -> the local OllamaClient) --
so produce_part's own per-stage model params (brief_model / author_model / ...) do the routing, full
pipeline + judge intact. Separates BUILD (hard_ok -- working code) from PLAN (features_missing +
judge disposition -- the right part). Append-only-resumable + fail-open (a subscription 401 on the
brief -> infra, retried). $0 for the local stages; only the opus_plan brief bills the subscription.

Run: FORGE_LLM_PROVIDER=claude_code python -u scripts/plan_ab.py [trials]
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

OPUS = "claude-opus-4-8"
QWEN = "qwen3-coder:30b"
QWEN_THINK = "qwen3:30b"  # local reasoning model as the ARCHITECT (not a coder)


class RoutingClient:
    """Per-stage dispatch by model id: a claude/opus/sonnet model -> subscription; else -> local."""

    def __init__(self) -> None:
        from forge_mind.llm.claude_code_client import ClaudeCodeClient
        from forge_mind.llm.ollama_client import OllamaClient

        self._claude = ClaudeCodeClient()
        self._ollama = OllamaClient()

    def _pick(self, model: str) -> Any:
        m = (model or "").lower()
        return self._claude if any(k in m for k in ("claude", "opus", "sonnet")) else self._ollama

    def generate_structured(self, *, model: str, **kw: Any) -> Any:
        return self._pick(model).generate_structured(model=model, **kw)

    def generate_structured_with_image(self, *, model: str, **kw: Any) -> Any:
        return self._pick(model).generate_structured_with_image(model=model, **kw)


class LocalRoutingClient:
    """FULLY LOCAL two-model pipeline: a reasoning model (qwen3:30b) plans, the coder builds. No
    cloud. Dispatches by model id to a per-model OllamaClient (Ollama swaps them in VRAM under
    OLLAMA_MAX_LOADED_MODELS=1). Tests whether a LOCAL architect recovers the Opus-planner lift."""

    def __init__(self) -> None:
        from forge_mind.llm.ollama_client import OllamaClient

        self._coder = OllamaClient(model=QWEN)
        self._planner = OllamaClient(model=QWEN_THINK)

    def _pick(self, model: str) -> Any:
        return self._coder if "coder" in (model or "").lower() else self._planner

    def generate_structured(self, *, model: str, **kw: Any) -> Any:
        return self._pick(model).generate_structured(model=model, **kw)

    def generate_structured_with_image(self, *, model: str, **kw: Any) -> Any:
        return self._pick(model).generate_structured_with_image(model=model, **kw)


ARMS = {
    "opus_plan": (OPUS, "routing"),  # brief Opus (cloud), build qwen-coder (local)
    "qwen_plan": (QWEN, "ollama"),  # all qwen-coder (local)
    "local_plan": (QWEN_THINK, "local_routing"),  # brief qwen3:30b, build qwen-coder -- ALL LOCAL
}


def _metrics(rec_loop: Any, disposition: str) -> dict[str, Any]:
    v = rec_loop.validation
    passed = bool(v.passed) if v is not None else bool(rec_loop.geometry_passed)
    failed = v.report.failed_check_ids() if (v is not None and v.report is not None) else []
    return {
        "passed": passed,
        "hard_ok": (bool(v.hard_ok) if v is not None else None),
        "features_missing": (bool(v.features_missing) if v is not None else None),
        "disposition": disposition,
        "failed_checks": failed,
        "iterations": len(rec_loop.iterations),
    }


def main() -> int:
    from bench_agentic import PROMPTS
    from bench_matrix import _EXTENDED
    from forge_mind.agentic.result import produce_part
    from forge_mind.fabrication import FabricationMode
    from forge_mind.llm.ollama_client import OllamaClient

    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    # 24 diverse parts (base 12 + the extended set): brackets/plates/gears/shrouds/mechanisms/
    # threads/enclosures/curves -- a WIDE array to separate planning from building across classes.
    prompts = list(PROMPTS) + [(lbl, txt) for lbl, txt, _k in _EXTENDED]
    out = Path("benchmarks/results/plan_ab.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    rows: list[dict[str, Any]] = []
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                if not r.get("infra_fail"):
                    done.add(r["key"])
    work = out.parent / "plan_ab_work"

    for arm, (brief_model, kind) in ARMS.items():
        for label, prompt in prompts:
            for t in range(trials):
                key = f"{arm}|{label}|{t}"
                if key in done:
                    continue
                if kind == "routing":
                    client: Any = RoutingClient()
                elif kind == "local_routing":
                    client = LocalRoutingClient()
                else:
                    client = OllamaClient()
                t0 = time.perf_counter()
                try:
                    r = produce_part(
                        prompt,
                        work / key.replace("|", "_"),
                        n_drafts=1,
                        max_iterations=6,
                        budget_usd=999,
                        fabrication_mode=FabricationMode.GENERAL,
                        brief_model=brief_model,
                        author_model=QWEN,
                        critic_model=QWEN,
                        judge_model=QWEN,
                        client=client,
                    )
                    row = {
                        "key": key,
                        "arm": arm,
                        "label": label,
                        "trial": t,
                        "seconds": round(time.perf_counter() - t0, 1),
                        "infra_fail": False,
                        **_metrics(r.loop, r.disposition),
                    }
                    # DEFENSIVE: a genuine produce_part takes tens of seconds; a sub-5s run that
                    # built nothing is a swallowed brief/author blip (auth 401, ollama contention),
                    # NOT a real part failure. Tag it infra so it is EXCLUDED from the rate AND
                    # retried on resume -- else a transient blip freezes as a permanent FAIL.
                    if row.get("hard_ok") is None and row["seconds"] < 5:
                        row["infra_fail"] = True
                        row["error"] = "swallowed blip: no build in <5s"
                except Exception as e:  # a subscription 401 on the Opus brief -> infra, retried
                    msg = str(e)
                    row = {
                        "key": key,
                        "arm": arm,
                        "label": label,
                        "trial": t,
                        "seconds": round(time.perf_counter() - t0, 1),
                        "infra_fail": True,
                        "auth": "401" in msg or "authenticate" in msg.lower(),
                        "error": msg[:160],
                    }
                rows.append(row)
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row) + "\n")
                tag = "INFRA" if row["infra_fail"] else ("PASS" if row.get("passed") else "FAIL")
                print(
                    f"[{arm:9} {label:18} t{t}] {tag:5} build={row.get('hard_ok')} "
                    f"disp={row.get('disposition')} {row['seconds']:.0f}s "
                    f"{row.get('failed_checks') or row.get('error', '')}",
                    flush=True,
                )

    print("\n==== PLANNER A/B ROLLUP (builder=qwen in both; only the PLANNER differs) ====")
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm and not r.get("infra_fail")]
        if not a:
            continue
        n = len(a)
        b = sum(1 for r in a if r.get("hard_ok"))
        f = sum(1 for r in a if r.get("features_missing") is False)
        acc = sum(1 for r in a if r.get("disposition") == "accepted")
        p = sum(1 for r in a if r.get("passed"))
        checks: Counter[str] = Counter()
        for r in a:
            for c in r.get("failed_checks") or []:
                checks[c] += 1
        top = ", ".join(f"{c} x{v}" for c, v in checks.most_common(4)) or "none"
        print(
            f"  {arm:9} BUILD {b}/{n} | FEATURES {f}/{n} | ACCEPTED {acc}/{n} | PASS {p}/{n}  "
            f"top-fails: {top}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
