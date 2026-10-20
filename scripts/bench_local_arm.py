"""One campaign arm on the local roster (ADR-0117): a frozen-brief corpus through the full
agentic pipeline, fixed sample-major interleave, per-iteration records, incremental results,
scoreboard printed at the end. Every arm gets its own TEST_LOG row; this runner only records.

Usage (venv python, PYTHONPATH=src):
  python scripts/bench_local_arm.py --briefs benchmarks/local_feature_v2/briefs.json \
      --arm F0_baseline --out E:/ForgeMind/benchmarks/2026-07-19_campaign/F0 \
      --samples 8 [--n-drafts 1] [--temp ""] [--max-iterations 6] [--env K=V ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

_BASE_ENV = {
    "FORGE_OLLAMA_MODEL": "qwen3-coder:30b",
    "FORGE_PROVIDER_CODER": "ollama",
    "FORGE_PROVIDER_REPAIR": "ollama",
    "FORGE_PROVIDER_VISION": "ollama_vision",
    "FORGE_VISION_MODEL": "qwen2.5vl:7b",
    "FORGE_PROVIDER_PLANNER": "ollama",
    "FORGE_API_ROLES": "",
    "FORGE_FORBID_API_SPEND": "1",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--briefs", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--n-drafts", type=int, default=1)
    ap.add_argument("--temp", default="")
    ap.add_argument("--max-iterations", type=int, default=6)
    ap.add_argument("--env", action="append", default=[], help="extra K=V env overrides")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="skip (label, sample) pairs already in results.json -- a crashed sweep "
        "relaunches for only its missing runs (the FINAL segfault cost 2 runs without this)",
    )
    args = ap.parse_args()

    os.environ.update(_BASE_ENV)
    if args.temp:
        os.environ["FORGE_OLLAMA_TEMPERATURE"] = args.temp
    for kv in args.env:
        k, _, v = kv.partition("=")
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)

    import subprocess as _sp

    from failure_scoreboard import scoreboard  # scripts/ sibling
    from forge_mind.agentic.brief import DesignBrief
    from forge_mind.agentic.loop import run_agentic_pipeline
    from forge_mind.fabrication import FabricationMode

    # Attribution stamp (ADR-0119): every row records WHICH code and WHICH knobs produced
    # it, so a regression bisects by field read -- never by session archaeology.
    try:
        _sha = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=10,
        ).stdout.strip()
    except Exception:
        _sha = ""
    _config: dict[str, object] = {
        "sha": _sha,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FORGE_")},
        "n_drafts": args.n_drafts,
        "max_iterations": args.max_iterations,
    }

    corpus = [r for r in json.loads(Path(args.briefs).read_text(encoding="utf-8")) if "brief" in r]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    done: set[tuple[str, int]] = set()
    if args.resume and (out / "results.json").exists():
        loaded: list[dict[str, object]] = json.loads(
            (out / "results.json").read_text(encoding="utf-8")
        )
        rows = list(loaded)
        done = {(str(r["label"]), int(str(r["sample"]))) for r in loaded}
        print(f"resume: {len(done)} runs already recorded", flush=True)
    for s in range(args.samples):
        for rec in corpus:  # sample-major: the ONE fixed interleave for every arm
            label = rec["label"]
            if (label, s) in done:
                continue
            t0 = time.time()
            row: dict[str, object] = {
                "arm": args.arm,
                "label": label,
                "sample": s,
                "config": _config,
            }
            try:
                res = run_agentic_pipeline(
                    rec["prompt"],
                    out / f"{label}_s{s}",
                    max_iterations=args.max_iterations,
                    n_drafts=args.n_drafts,
                    fabrication_mode=FabricationMode.PRINT_3D,
                    brief=DesignBrief.model_validate(rec["brief"]),
                )
                row["stopped"] = res.stopped_reason
                row["approved"] = bool(res.critic_approved)
                row["passed_final"] = bool(res.geometry_passed)
                row["iters"] = [
                    {
                        "i": it.iteration,
                        "phase": it.phase,
                        "built": it.built,
                        "score": it.score,
                        "kept": it.kept_as_best,
                        "passed": it.validation_passed,
                        "resemblance": it.critic_resemblance,
                        "fixes": it.critic_fixes,
                        "err": (it.sandbox_error or "")[:120],
                    }
                    for it in res.iterations
                ]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {str(exc)[:140]}"
            row["t_s"] = round(time.time() - t0, 1)
            rows.append(row)
            print(
                json.dumps({k: v for k, v in row.items() if k != "iters"}),
                flush=True,
            )
            (out / "results.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    ships = sum(1 for r in rows if r.get("stopped") == "accepted")
    print(f"SHIPS: {ships}/{len(rows)}")
    print(json.dumps(scoreboard(rows, out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
