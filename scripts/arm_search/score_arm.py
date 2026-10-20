"""Thin CLI over ``forge_mind.structures.objectives`` (the code moved into the engine; ADR-0097).

Usage:
  uv run --no-sync python scripts/arm_search/score_arm.py --stl part.stl [--canonical-link]
  uv run --no-sync python scripts/arm_search/score_arm.py --dir ref_dir/ [--reference] [--out F]
  uv run --no-sync python scripts/arm_search/score_arm.py --run <run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from forge_mind.materials import DEFAULT_MATERIAL_ID
from forge_mind.structures.loadcase import SMOKE_CANTILEVER_5N, canonical_link_case

# Re-exports for callers that still import this script as a module.
from forge_mind.structures.objectives import (  # noqa: F401
    ALLOWED_DISPOSITIONS,
    ArmObjectives,
    PartObjectives,
    fea_deflection_within,
    run_disposition,
    run_is_scorable,
    score_mesh,
    score_run_dir,
    score_stl,
    score_stl_dir,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Arm objective-vector measurement.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stl", metavar="PATH")
    g.add_argument("--dir", metavar="PATH")
    g.add_argument("--run", metavar="PATH")
    ap.add_argument("--material", default=DEFAULT_MATERIAL_ID)
    ap.add_argument("--no-fea", action="store_true")
    ap.add_argument("--canonical-link", action="store_true")
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--out", metavar="FILE")
    args = ap.parse_args(argv)

    if args.no_fea:
        load_case: Any | None = None
    elif args.canonical_link:
        load_case = canonical_link_case
    else:
        load_case = SMOKE_CANTILEVER_5N

    if args.stl:
        payload: dict[str, Any] = asdict(
            score_stl(args.stl, material=args.material, load_case=load_case)
        )
    elif args.dir:
        arm = score_stl_dir(args.dir, material=args.material, load_case=load_case)
        payload = arm.to_dict()
        if args.reference:
            (Path(args.dir) / "reference_objectives.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
    else:
        payload = score_run_dir(args.run, material=args.material).to_dict()

    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
