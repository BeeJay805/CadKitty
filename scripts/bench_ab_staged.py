"""Staged-vs-monolithic A/B on interaction-dense parts (complex-parts arc P2/P3).

PRE-REGISTERED GATE (also declared in the corpus file): staged ships iff
  pass@1 >= monolithic +15pts, OR equal pass@1 with >=30% fewer author LLM calls
  (the declared token proxy on subscription runs). Kill honored.

Both arms run the SAME prompts, rubrics, final validation gate, author arm and caps --
the ONLY difference is FORGE_STAGED_BUILD around produce_part (the initial-construction
seam). Scoring reuses bench_arbitrary's independent artifact measurement verbatim.
Call-count proxy: len(iterations) + staged stage tries (from stages.json); output-char
proxy recorded alongside. Kill-safe (results rewritten per record); resumable; shardable
by prompt-index parity.

Args: [corpus_json] [k] [arm] [shard_ix] [shard_n] [resume_json]
Run:  $env:FORGE_LLM_PROVIDER='claude_code'
      uv run --no-sync python -u scripts/bench_ab_staged.py benchmarks/interaction_v0/interaction_v0.json 2 opus
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from bench_arbitrary import _MODELS, _run_variant  # noqa: E402
from forge_mind.fabrication import FabricationMode  # noqa: E402

_ARMS = ("mono", "staged")


def _stage_tries(job_dir: Path) -> tuple[int, int]:
    """(total stage author tries, stages completed) from stages.json; (0, 0) when absent."""
    p = job_dir / "work" / "staged" / "stages.json"
    if not p.exists():
        p = job_dir / "staged" / "stages.json"
    if not p.exists():
        # produce_part job dirs keep the loop work dir at the top level; search shallowly
        hits = list(job_dir.glob("**/stages.json"))
        if not hits:
            return 0, 0
        p = hits[0]
    try:
        recs = json.loads(p.read_text(encoding="utf-8"))
        return sum(int(r.get("tries", 0)) for r in recs), sum(1 for r in recs if r.get("built"))
    except Exception:
        return 0, 0


def main() -> int:
    corpus = Path(
        sys.argv[1] if len(sys.argv) > 1 else "benchmarks/interaction_v0/interaction_v0.json"
    )
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    arm_name = sys.argv[3] if len(sys.argv) > 3 else "opus"
    shard_ix = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    shard_n = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    resume = Path(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] != "-" else None

    model_id = _MODELS[arm_name]
    data = json.loads(corpus.read_text(encoding="utf-8"))
    mode = FabricationMode(str(data.get("fabrication_mode", "general")))
    hole_sem = str(data.get("hole_semantics", "stacks"))
    bbox_sem = str(data.get("bbox_semantics", "per_axis"))
    prompts = [p for i, p in enumerate(data["prompts"]) if i % shard_n == shard_ix]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_s{shard_ix}" if shard_n > 1 else ""
    out_dir = REPO / "benchmarks" / "results" / f"ab_staged_{arm_name}_{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"

    records: list[dict[str, Any]] = []
    done: set[tuple[str, str, int]] = set()
    if resume is not None and resume.exists():
        records = list(json.loads(resume.read_text(encoding="utf-8")).get("results", []))
        done = {(str(r["label"]), str(r["ab_arm"]), int(r["variant"])) for r in records}
        print(f"-- resuming: {len(records)} records --", flush=True)

    def _headline() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm in _ARMS:
            rs = [r for r in records if r["ab_arm"] == arm]
            first = {r["label"]: r for r in rs if int(r["variant"]) == 0}
            n = len(first)
            p1 = sum(1 for r in first.values() if r["passed"])
            calls = [int(r.get("author_calls", 0)) for r in rs if r.get("author_calls")]
            out[arm] = {
                "records": len(rs),
                "pass_all_draws": sum(1 for r in rs if r["passed"]),
                "n_prompts_drawn0": n,
                "pass@1": p1,
                "mean_author_calls": round(sum(calls) / len(calls), 2) if calls else None,
            }
        return out

    def _save() -> None:
        out_json.write_text(
            json.dumps(
                {
                    "benchmark": data.get("version"),
                    "ab_gate": data.get("ab_gate"),
                    "arm_model": model_id,
                    "k": k,
                    "fabrication_mode": str(mode),
                    "headline": _headline(),
                    "results": records,
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    _save()
    total = len(prompts) * len(_ARMS) * k
    ix = 0
    for p in prompts:
        label = str(p["id"])
        for ab_arm in _ARMS:
            os.environ["FORGE_STAGED_BUILD"] = "1" if ab_arm == "staged" else ""
            for s in range(k):
                ix += 1
                if (label, ab_arm, s) in done:
                    continue
                job = out_dir / f"{label}__{ab_arm}_k{s}"
                rec = _run_variant(
                    label,
                    s,
                    str(p["prompt"]),
                    p.get("expect", {}),
                    job,
                    model_id,
                    hole_semantics=hole_sem,
                    bbox_semantics=bbox_sem,
                    fabrication_mode=mode,
                )
                tries, built_stages = _stage_tries(job)
                rec["ab_arm"] = ab_arm
                rec["stage_tries"] = tries
                rec["stages_built"] = built_stages
                rec["author_calls"] = int(rec.get("iterations", 0)) + tries
                records.append(rec)
                _save()
                tag = "PASS" if rec["passed"] else f"fail({rec.get('stopped_reason')})"
                print(
                    f"[{ix}/{total}] {label} [{ab_arm}] k{s}: {tag} "
                    f"(iters={rec.get('iterations')}, stage_tries={tries})",
                    flush=True,
                )
    print(json.dumps(_headline(), indent=1), flush=True)
    print(f"[done] {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
