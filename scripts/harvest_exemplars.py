"""Seed the exemplar library: run DISJOINT-from-benchmark seed prompts through the
general author and harvest the validator-PASSING builds (lever e, ADR-0045).

The library is the production story in miniature: validated builds accumulate;
frozen-benchmark prompts can never enter (the deny guard in agentic/exemplars.py).
Seeds live in scripts/exemplar_seeds.txt (one prompt per line, '#' comments). A
disjointness test pins that no seed matches a bench prompt.

Args: [lib_dir] [max_prompts]
Run:  $env:FORGE_LLM_PROVIDER='claude_code'; uv run --no-sync python -u scripts/harvest_exemplars.py exemplar_lib
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

os.environ["FORGE_BUILD_CACHE"] = ""  # a replayed build is not a fresh harvest source

from forge_mind.agentic.exemplars import (
    ExemplarRecord,
    _is_denied,  # the harvester IS the guard's call site
    benchmark_deny_list,
    store,
)
from forge_mind.agentic.result import produce_part
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.claude_code_client import subscription_client_if_selected

SEEDS_FILE = Path("scripts/exemplar_seeds.txt")


def main() -> int:
    lib_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exemplar_lib")
    max_prompts = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    client = subscription_client_if_selected()
    deny = benchmark_deny_list()
    seeds = [
        line.strip()
        for line in SEEDS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if max_prompts:
        seeds = seeds[:max_prompts]
    denied = [s for s in seeds if _is_denied(s, deny)]
    if denied:
        raise SystemExit(f"seed prompts collide with the frozen benchmark: {denied}")

    stored = 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, prompt in enumerate(seeds):
            out = Path(tmp) / f"seed_{i}"
            try:
                result = produce_part(
                    prompt,
                    out,
                    n_drafts=1,
                    max_iterations=8,
                    budget_usd=1.0,
                    fabrication_mode=FabricationMode.GENERAL,
                    client=client,
                )
            except Exception as exc:
                print(f"[{i + 1}/{len(seeds)}] CRASH {type(exc).__name__}: {prompt[:60]}")
                continue
            val = result.loop.validation
            passed = bool(val.passed) if val is not None else False
            code_path = out / "program.py"
            if passed and code_path.exists():
                brief = result.loop.brief
                store(
                    ExemplarRecord(
                        object_name=(brief.object_name if brief else prompt[:40]),
                        prompt=prompt,
                        code=code_path.read_text(encoding="utf-8", errors="replace"),
                        source=f"seed_harvest:{i}",
                        harvested_at=datetime.now(UTC).isoformat(),
                    ),
                    lib_dir,
                )
                stored += 1
            print(
                f"[{i + 1}/{len(seeds)}] {'PASS+stored' if passed else 'FAIL':12} {prompt[:60]}",
                flush=True,
            )
    print(f"\nlibrary {lib_dir}: {stored} exemplars stored from {len(seeds)} seeds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
