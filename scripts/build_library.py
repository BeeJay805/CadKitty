"""Run the deterministic autonomous library builder ($0, offline, re-runnable).

Writes each kept part's STL under build/library/stl/<family>/<id>/ (gitignored, regenerable) and
the indexed manifest + reject log to benchmarks/library/ (committed). Prints per-family counts +
Wilson 95% lower bound + the reject-tier breakdown.

Run: uv run python scripts/build_library.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from forge_mind.library import build_library

REPO = Path.cwd()
STL_DIR = REPO / "build" / "library" / "stl"
OUT_DIR = REPO / "benchmarks" / "library"


def main() -> None:
    STL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_library(STL_DIR)

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2), encoding="utf-8"
    )
    (OUT_DIR / "rejects.json").write_text(
        json.dumps(manifest.rejects_to_dict(), indent=2), encoding="utf-8"
    )

    print(f"=== LIBRARY: {manifest.total} parts, {len(manifest.rejects)} rejects ===")
    for s in manifest.families:
        print(
            f"  {s.family:18} passed={s.passed:>3}/{s.attempted:<3} rejected={s.rejected:<3} "
            f"Wilson95LB={s.wilson95_lb * 100:.1f}%"
        )
    by_tier = Counter(r.tier for r in manifest.rejects)
    print(f"  reject tiers: {dict(sorted(by_tier.items()))}")
    print(f"  manifest -> {(OUT_DIR / 'manifest.json').relative_to(REPO)}")
    print(f"  rejects  -> {(OUT_DIR / 'rejects.json').relative_to(REPO)}")
    print(f"  STLs     -> {STL_DIR.relative_to(REPO)}/ (gitignored, regenerable)")


if __name__ == "__main__":
    main()
