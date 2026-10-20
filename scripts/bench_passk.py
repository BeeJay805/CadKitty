"""pass@1 -> pass@k headroom probe (training gate #1) -- how much does the validator gain
from best-of-k sampling?

Draws K INDEPENDENT full-pipeline samples of each holdout prompt (the LLM's own stochasticity
is the sampling; no paraphrases -- pass@k is repeated draws of the SAME input) and scores each
against the rubric with the SAME independent measurement the frozen bench uses. Reports:

  * pass@1  = the fraction of prompts whose FIRST sample passes (single-shot, deployed behavior)
  * pass@k  = the fraction of prompts where ANY of the K samples passes (validator-selected best)
  * GAP     = pass@k - pass@1 -- the headroom for best-of-k + self-training

Gate reading (pre-registered): gap >= 15 pts -> self-training / best-of-k has real headroom (and
sizes a studio tier); gap < 5 pts -> deprioritize the learning agenda. Every PASS is a validated
brief->code pair for the fall corpus.

Runs on the HELD-OUT set (scripts/gen_holdout_problems.py, a fresh rotated seed -- NEVER the frozen
benches, so this measures true generalization). Reuses bench_arbitrary's produce_part call and
independent artifact scoring verbatim. Kill-safe (rewritten per sample); resumable (pass a prior
results.json). $0 on the subscription (FORGE_LLM_PROVIDER=claude_code).

Args: [holdout_json] [k] [arm] [n_prompts] [resume_json] [shard_ix] [shard_n]
  holdout_json  path to an arbitrary_holdout_v1.json (default benchmarks/results/holdout_rot7351/...)
  k             samples per prompt (default 8)
  arm           sonnet|opus|fable (default fable -- the production author)
Run:  $env:FORGE_LLM_PROVIDER='claude_code'; uv run --no-sync python -u scripts/bench_passk.py
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the frozen bench's produce_part call + independent scoring verbatim (one instrument).
from bench_arbitrary import _MODELS, _run_variant  # noqa: E402
from forge_mind.fabrication import FabricationMode  # noqa: E402

_DEFAULT_HOLDOUT = "benchmarks/results/holdout_rot7351/arbitrary_holdout_v1.json"


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    import math

    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _headline(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """pass@1 (sample 0) and pass@k (any sample) over prompts with >=1 recorded sample."""
    by_prompt: dict[str, dict[int, bool]] = {}
    for r in records:
        by_prompt.setdefault(r["label"], {})[int(r["variant"])] = bool(r["passed"])
    p1 = pk = n = 0
    per_prompt = []
    for label, samples in sorted(by_prompt.items()):
        if not samples:
            continue
        n += 1
        s1 = bool(samples.get(0, False))
        sk = any(samples.values())
        p1 += int(s1)
        pk += int(sk)
        per_prompt.append(
            {
                "label": label,
                "pass@1": s1,
                "pass@k": sk,
                "passes": sum(samples.values()),
                "k_run": len(samples),
            }
        )
    pass1 = p1 / n if n else 0.0
    passk = pk / n if n else 0.0
    return {
        "n_prompts": n,
        "k": k,
        "pass@1": round(pass1, 4),
        "pass@k": round(passk, 4),
        "gap_pts": round((passk - pass1) * 100, 1),
        "pass@1_wilson95": [round(x, 3) for x in _wilson(p1, n)],
        "pass@k_wilson95": [round(x, 3) for x in _wilson(pk, n)],
        "per_prompt": per_prompt,
    }


def main() -> int:
    holdout = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / _DEFAULT_HOLDOUT
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    arm = sys.argv[3] if len(sys.argv) > 3 else "fable"
    n_prompts = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    resume = Path(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "-" else None
    shard_ix = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    shard_n = int(sys.argv[7]) if len(sys.argv) > 7 else 1

    arm_id = _MODELS.get(arm)
    if arm_id is None:
        raise SystemExit(f"unknown arm {arm!r}; use sonnet|opus|fable")
    data = json.loads(holdout.read_text(encoding="utf-8"))
    # The corpus declares its own scoring semantics (frontier_v1 is print-mode with the
    # ADR-0085 corrected semantics); absent fields keep the historical holdout defaults.
    fabrication_mode = FabricationMode(str(data.get("fabrication_mode", "general")))
    hole_semantics = str(data.get("hole_semantics", "stacks"))
    bbox_semantics = str(data.get("bbox_semantics", "per_axis"))
    prompts = data["prompts"][: n_prompts or len(data["prompts"])]
    if shard_n > 1:
        prompts = [p for i, p in enumerate(prompts) if i % shard_n == shard_ix]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_s{shard_ix}" if shard_n > 1 else ""
    out_dir = REPO / "benchmarks" / "results" / f"passk_{arm}_{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"

    records: list[dict[str, Any]] = []
    done: set[tuple[str, int]] = set()
    if resume is not None and resume.exists():
        records = list(json.loads(resume.read_text(encoding="utf-8")).get("results", []))
        done = {(str(r["label"]), int(r["variant"])) for r in records}
        print(f"-- resuming: {len(records)} samples already recorded --", flush=True)

    def _save() -> None:
        out_json.write_text(
            json.dumps(
                {
                    "benchmark": data.get("version", "holdout_v1"),
                    "holdout_seed": data.get("seed"),
                    "arm": arm,
                    "arm_model": arm_id,
                    "k": k,
                    "fabrication_mode": str(fabrication_mode),
                    "hole_semantics": hole_semantics,
                    "bbox_semantics": bbox_semantics,
                    "headline": _headline(records, k),
                    "results": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    _save()
    total = len(prompts) * k
    ix = 0
    for p in prompts:
        label = str(p["id"])
        for s in range(k):
            ix += 1
            if (label, s) in done:
                continue
            rec = _run_variant(
                label,
                s,
                str(p["prompt"]),
                p.get("expect", {}),
                out_dir / f"{label}_k{s}",
                arm_id,
                hole_semantics=hole_semantics,
                bbox_semantics=bbox_semantics,
                fabrication_mode=fabrication_mode,
            )
            records.append(rec)
            _save()
            tag = (
                "PASS"
                if rec["passed"]
                else f"fail({rec.get('stopped_reason') or rec.get('crash', '?')})"
            )
            print(f"[{ix}/{total}] {label} sample {s}: {tag}", flush=True)

    h = _headline(records, k)
    print(
        json.dumps(
            {kk: h[kk] for kk in ("n_prompts", "k", "pass@1", "pass@k", "gap_pts")}, indent=2
        ),
        flush=True,
    )
    print(f"[done] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
