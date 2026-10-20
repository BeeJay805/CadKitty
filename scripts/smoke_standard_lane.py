"""Paid smoke for the standard lane (ADR-0035): grammar fit + routed prompts, LANE ONLY.

Runs each prompt through ``api.core._generate_standard`` directly -- the deterministic lane
WITHOUT the agentic fallback -- so a lane miss costs its classified outcome (~$0.02-0.10), never
the $3 agentic budget. Also re-runs the real-API grammar smoke (ADR-0020), which matters this
pass because ``ElectronicsTraySpec`` grew a ``max_envelope_mm`` field.

Usage:
    uv run python scripts/smoke_standard_lane.py [--grammar] [--prompts]

Spend: --grammar ~ $0 (tiny max_tokens); --prompts ~ $0.02-0.10 per prompt.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

# Prompts chosen to prove ADR-0035: the un-killed Pi-4 family at production defaults
# (min_wall UNSTATED -> blueprint default 4mm), plus families A1 newly routes.
PROMPTS = [
    "a Raspberry Pi 4 case",  # A2: was 100% dead (LayoutError) before this pass
    "a raspberry pi 4 tray with vent slots",  # A2, second form
    "a 60mm fan mount",  # A1: bench-006 30/30, never routed before
    "a NEMA 23 motor mounting plate",  # A1: nema23 spelling, never routed before
    "a mounting plate with a 608 bearing seat and four M4 holes",  # A1: bearing family
    "a 2040 extrusion joining plate",  # A1: 2040 profile, never routed before
]


def run_grammar() -> bool:
    from forge_mind.layout.router import check_grammar

    ok = True
    for label, fits, detail in check_grammar():
        print(f"  grammar {label}: {'FITS' if fits else 'OVER'} ({detail})")
        ok = ok and fits
    return ok


def run_prompts() -> bool:
    from forge_mind.api.core import _generate_standard
    from forge_mind.layout.standard_match import standard_match

    all_shipped = True
    for prompt in PROMPTS:
        if not standard_match(prompt):
            print(f"  ROUTING BUG: {prompt!r} does not match the lane")
            all_shipped = False
            continue
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory() as d:
            res, lane = _generate_standard(prompt, Path(d), lambda k, s: None)
            row = {
                "prompt": prompt,
                "outcome": lane.outcome,
                "gen_path": lane.gen_path,
                "cost_usd": lane.cost_usd,
                "duration_s": round(time.monotonic() - t0, 1),
                "fallback_reason": lane.fallback_reason,
                "refusal": lane.refusal,
                "error": lane.error,
            }
            print("  " + json.dumps({k: v for k, v in row.items() if v is not None}))
            if res is None or lane.outcome != "shipped":
                all_shipped = False
    return all_shipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grammar", action="store_true", help="run the grammar fit check")
    parser.add_argument("--prompts", action="store_true", help="run the routed-prompt smoke")
    args = parser.parse_args()
    if not (args.grammar or args.prompts):
        parser.error("pick at least one of --grammar / --prompts")

    ok = True
    if args.grammar:
        print("== grammar fit (all router schemas) ==")
        ok = run_grammar() and ok
    if args.prompts:
        print("== standard lane, routed prompts (no agentic fallback) ==")
        ok = run_prompts() and ok
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
