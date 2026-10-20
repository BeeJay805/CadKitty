"""Run the composition engine for a shard (or all) and write its manifest shard JSON.

Deterministic + re-runnable. Sharded so base-body families / interface groups run in parallel and
merge (Step 2). STLs -> build/composition/ (gitignored, regenerable); manifest -> benchmarks/.

Run: uv run python scripts/run_composition.py --shard flat_singles
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_mind.composition import SHARDS, build_compositions

REPO = Path.cwd()
STL_DIR = REPO / "build" / "composition" / "stl"
OUT_DIR = REPO / "benchmarks" / "composition"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", choices=[*SHARDS, "all"], default="all")
    args = ap.parse_args()
    shards = tuple(SHARDS) if args.shard == "all" else (args.shard,)

    STL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_compositions(STL_DIR, shards=shards)
    out = OUT_DIR / f"manifest_{args.shard}.json"
    out.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    print(
        f"shard={args.shard}  parts={len(manifest.parts)}  rejects={len(manifest.rejects)}  -> {out.relative_to(REPO)}"
    )


if __name__ == "__main__":
    main()
