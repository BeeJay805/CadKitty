"""Merge sharded arbitrary_v1 results into ONE headline (pass@1, min-across-variants).

Shards are disjoint prompt subsets of the SAME frozen run config, so merging is a
straight union of their records. Refuses to merge across different arms, benchmarks,
manifest hashes, or variant counts (those are different experiments, not shards).
Also reports the rate-limit / crash audit: crashed variants are listed so they can be
stripped + rerun via the resume arg BEFORE the headline is cited -- a rate-limited
author is not an engine failure.

Run: uv run --no-sync python scripts/merge_bench_results.py benchmarks/results/arbitrary_opus_<stamp>_s*/results.json
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from forge_mind.assembly.bench import wilson_lower_bound


def main() -> int:
    paths = [Path(p) for arg in sys.argv[1:] for p in glob.glob(arg)]
    if not paths:
        raise SystemExit("no results files matched")
    merged: list[dict] = []
    keys: set[tuple] = set()
    for p in paths:
        d = json.loads(p.read_text(encoding="utf-8"))
        keys.add((d["benchmark"], d["manifest_sha"], d["arm"], d["n_variants"]))
        merged.extend(d["results"])
    if len(keys) != 1:
        raise SystemExit(f"refusing to merge different experiments: {keys}")
    benchmark, sha, arm, n_variants = next(iter(keys))

    crashed = [r for r in merged if r.get("crash")]
    by_label: dict[str, list[dict]] = {}
    for r in merged:
        by_label.setdefault(str(r["label"]), []).append(r)
    complete = {k: v for k, v in by_label.items() if len(v) >= n_variants}
    passed = sum(1 for v in complete.values() if all(r["passed"] for r in v))
    n = len(complete)

    print(f"benchmark {benchmark} (sha {sha[:12]})  arm={arm}  variants={n_variants}")
    print(f"shards merged: {len(paths)}  records: {len(merged)}")
    if crashed:
        print(f"\nCRASHED variants ({len(crashed)}) -- audit before citing the headline:")
        for r in crashed:
            print(f"  {r['label']} v{r['variant']}: {str(r.get('crash'))[:90]}")
    print(
        f"\npass@1 (min-across-{n_variants}): {passed}/{n} = "
        f"{(passed / n * 100 if n else 0):.0f}%  "
        f"Wilson95 LB {wilson_lower_bound(passed, n) * 100 if n else 0:.0f}%"
    )
    fails = {
        k: [r for r in v if not r["passed"]]
        for k, v in complete.items()
        if not all(r["passed"] for r in v)
    }
    print(f"\nfailing prompts ({len(fails)}):")
    for k, rs in sorted(fails.items()):
        for r in rs:
            bad = [c for c, ok in (r.get("checks") or {}).items() if not ok]
            why = (
                r.get("crash")
                or ",".join(bad)
                or "engine_fail:" + ",".join(r.get("failed_checks") or [])
            )
            print(
                f"  {k} v{r['variant']}: it={r.get('iterations')} {r.get('stopped_reason')} {why}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
