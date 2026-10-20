"""Thin CLI over ``forge_mind.structures.sections`` (the code moved into the engine; ADR-0097).

The deterministic physics section-author, print gates included -- rerunning THIS reproduces the
recorded claim (Fable's reproducibility fix).

Usage:
  FORGE_FEA_TIMEOUT_S=400 uv run --no-sync python scripts/arm_search/analytic_link.py \
    --ref-mass-g 145.49 --ref-deflection-mm 0.0457 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Re-exports for callers that still import this script as a module.
from forge_mind.structures.sections import (  # noqa: F401
    author_isection_link,
    buckling_margins,
    build_link,
    i_isection,
    i_required,
    print_gates,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic analytical M1 section-author.")
    ap.add_argument("--ref-mass-g", type=float, required=True)
    ap.add_argument("--ref-deflection-mm", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=float, default=142.0, help="link span (mm)")
    ap.add_argument("--hy", type=float, default=24.5, help="thin/load-axis section height (mm)")
    ap.add_argument("--wx", type=float, default=67.3, help="wide-axis section width (mm)")
    args = ap.parse_args(argv)
    rec = author_isection_link(
        ref_mass_g=args.ref_mass_g,
        ref_deflection_mm=args.ref_deflection_mm,
        length=args.length,
        hy=args.hy,
        wx=args.wx,
        out_dir=Path(args.out),
    )
    # The tamper-proof acceptance checker reads a fixed filename from a run dir; the author's
    # record IS this benchmark's report, so it is written under both names (same bytes).
    (Path(args.out) / "functional_report.json").write_text(
        json.dumps(rec, indent=2), encoding="utf-8"
    )
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
