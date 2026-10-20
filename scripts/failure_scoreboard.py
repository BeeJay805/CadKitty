"""Failure scoreboard: the per-arm taxonomy report for the local-lane campaign (ADR-0117).

Reads an arm results JSON (rows with per-iteration records: built/score/passed/err) and emits
the terminal-class counts, no_build error signatures, identical-error fixed points, and --
when the run dirs are available -- code-idiom rates (the detail leading indicators). Report
only; never gates. Acceptance (tested): reproduces the 2026-07-18 hand-computed taxonomy on
breadth_results.json / breadth_results_b.json (terminal classes 4/11/5/4 and 4/10/7/3).

Usage:
  python scripts/failure_scoreboard.py <results.json> [--run-dirs <root>] [--iters-key iters]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ADDITIVE_OPS = ("PolarLocations", "revolve(", "sweep(", "loft(", "Helix")


def classify_run(row: dict[str, Any]) -> str:
    """Terminal blocking class for one run. Mirrors the 2026-07-18 hand taxonomy:
    shipped / no_build (zero built iters) / critic_block (some iteration validator-green) /
    validator_red (built, never passed; sub-classed by best score elsewhere)."""
    if row.get("stopped") == "accepted" or (row.get("approved") and row.get("passed_final")):
        return "shipped"
    iters = row.get("iters") or []
    built = [it for it in iters if it.get("built")]
    if not built:
        return "no_build"
    if any(it.get("passed") for it in built):
        return "critic_block"
    return "validator_red"


def best_score(row: dict[str, Any]) -> int | None:
    scores = [it.get("score") for it in (row.get("iters") or []) if it.get("built")]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else None


def fixed_points(row: dict[str, Any]) -> tuple[int, int]:
    """(streak_count, iters_burned) for runs of the SAME non-empty error >=3 consecutive."""
    errs = [(it.get("err") or "").strip() for it in (row.get("iters") or [])]
    streaks, burned = 0, 0
    run_len, prev = 0, None
    for e in [*errs, None]:  # sentinel flush
        if e and e == prev:
            run_len += 1
        else:
            if prev and run_len >= 3:
                streaks += 1
                burned += run_len
            run_len = 1 if e else 0
        prev = e
    return streaks, burned


def error_signature(err: str) -> str:
    e = err.removeprefix("exec: ").strip()
    e = re.sub(r"\d+(\.\d+)?", "N", e)  # fold numbers so radii/counts group
    return e[:60]


def idiom_rates(run_dir: Path) -> dict[str, Any] | None:
    """Code-level detail indicators from the run dir's iter_*/job.json files."""
    jobs = sorted(run_dir.glob("iter_*/job.json"))
    if not jobs:
        return None
    n = 0
    counts: Counter[str] = Counter()
    box_cuts = 0
    for j in jobs:
        try:
            code = json.loads(j.read_text(encoding="utf-8")).get("code") or ""
        except Exception:
            continue
        n += 1
        for op in (
            "PolarLocations",
            "GridLocations",
            "revolve(",
            "sweep(",
            "loft(",
            "chamfer(",
            "fillet(",
        ):
            if op in code:
                counts[op] += 1
        additive = any(op in code for op in _ADDITIVE_OPS)
        boxy = ("Box(" in code or "Rectangle(" in code) and "Mode.SUBTRACT" in code
        if boxy and not additive:
            box_cuts += 1
    if n == 0:
        return None
    return {"iters_with_code": n, "box_with_cuts_only": box_cuts, "op_counts": dict(counts)}


def scoreboard(rows: list[dict[str, Any]], run_root: Path | None) -> dict[str, Any]:
    classes = Counter(classify_run(r) for r in rows)
    sub_red: Counter[str] = Counter()
    for r in rows:
        if classify_run(r) == "validator_red":
            sub_red[f"score{best_score(r)}"] += 1
    sigs: Counter[str] = Counter()
    for r in rows:
        for it in r.get("iters") or []:
            if not it.get("built") and (it.get("err") or "").strip():
                sigs[error_signature(it["err"])] += 1
    fp_runs, fp_iters = 0, 0
    for r in rows:
        s, b = fixed_points(r)
        if s:
            fp_runs += 1
            fp_iters += b
    out: dict[str, Any] = {
        "n_runs": len(rows),
        "terminal_classes": dict(classes),
        "validator_red_by_best_score": dict(sub_red),
        "top_error_signatures": sigs.most_common(8),
        "fixed_point_runs": fp_runs,
        "fixed_point_iters_burned": fp_iters,
    }
    if run_root is not None and run_root.exists():
        agg_n = agg_box = 0
        agg_ops: Counter[str] = Counter()
        for r in rows:
            label = r.get("label") or r.get("brief", "").replace(" ", "_")
            sample = r.get("sample")
            candidates = [run_root / f"{label}_s{sample}"] if sample is not None else []
            candidates += [run_root / f"{label}_b", run_root / str(label)]
            for cand in candidates:
                rates = idiom_rates(cand) if cand.exists() else None
                if rates:
                    agg_n += rates["iters_with_code"]
                    agg_box += rates["box_with_cuts_only"]
                    agg_ops.update(rates["op_counts"])
                    break
        if agg_n:
            out["idioms"] = {
                "iters_with_code": agg_n,
                "box_with_cuts_only_pct": round(100 * agg_box / agg_n, 1),
                "op_usage_pct": {k: round(100 * v / agg_n, 1) for k, v in sorted(agg_ops.items())},
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--run-dirs", default=None)
    args = ap.parse_args()
    rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    board = scoreboard(rows, Path(args.run_dirs) if args.run_dirs else None)
    print(json.dumps(board, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
