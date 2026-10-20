"""Merge the composition shard manifests + report confidence (Step 2 merge + Step 3 stats).

Reads benchmarks/composition/manifest_<shard>.json, writes the merged manifest.json + rejects.json,
and reports: total library size before/after, per-base + per-interface-type Wilson 95% lower bound
(passed / attempted), and reject counts by reason.

Run: uv run python scripts/merge_composition.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from forge_mind.assembly.bench import wilson_lower_bound

REPO = Path.cwd()
OUT = REPO / "benchmarks" / "composition"
_LIBRARY_BEFORE = 138  # the deterministic library builder's geometry-distinct count


def _wilson(passed: int, attempted: int) -> float:
    return wilson_lower_bound(passed, attempted) if attempted else 0.0


def _row(label: str, parts: list[Any], rejects: list[Any]) -> None:
    p, a = len(parts), len(parts) + len(rejects)
    print(f"  {label:26} passed={p:>4}/{a:<4}  Wilson95LB={_wilson(p, a) * 100:5.1f}%")


def main() -> None:
    shard_files = sorted(f for f in OUT.glob("manifest_*.json"))
    if not shard_files:
        print("no shard manifests found -- run scripts/run_composition.py first")
        return
    parts: list[Any] = []
    rejects: list[Any] = []
    for f in shard_files:
        d = json.loads(f.read_text())
        parts.extend(d["parts"])
        rejects.extend(d["rejects"])

    (OUT / "manifest.json").write_text(
        json.dumps({"total_parts": len(parts), "parts": parts}, indent=2), encoding="utf-8"
    )
    (OUT / "rejects.json").write_text(
        json.dumps({"total_rejects": len(rejects), "rejects": rejects}, indent=2), encoding="utf-8"
    )

    print(f"=== merged {len(shard_files)} shards: {len(parts)} parts, {len(rejects)} rejects ===")
    print("\n=== per base-body ===")
    for b in sorted({x["base"] for x in parts + rejects}):
        _row(b, [p for p in parts if p["base"] == b], [r for r in rejects if r["base"] == b])

    print("\n=== per interface-type ===")
    refs = sorted({f["ref"] for x in parts + rejects for f in x["features"]})
    for ref in refs:
        pp = [p for p in parts if any(f["ref"] == ref for f in p["features"])]
        rr = [r for r in rejects if any(f["ref"] == ref for f in r["features"])]
        _row(ref, pp, rr)

    print("\n=== rejects by reason ===", dict(sorted(Counter(r["stage"] for r in rejects).items())))
    geom = len(parts)
    print(
        f"\n=== LIBRARY SIZE: before={_LIBRARY_BEFORE} (deterministic builder) -> "
        f"after={_LIBRARY_BEFORE + geom} geometry-distinct (+{geom} composed) ==="
    )
    print(f"  composed parts x 3 materials (PLA/PETG/ABS) = {geom * 3} catalog entries")
    print(
        f"  manifest -> {(OUT / 'manifest.json').relative_to(REPO)}  rejects -> {(OUT / 'rejects.json').relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
