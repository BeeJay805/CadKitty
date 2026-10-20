"""M1 N-idiom search driver (plan Ticket 5, single-link scope).

Proposes K structural idioms for the same link (frozen interface + envelope), runs each through
the M1 recipe (`m1_probe.run_m1`: author with the mass lever -> measure mass + validated FEA),
then SELECTS deterministically: keep only candidates that BEAT the reference (built, geometry
passed, deflection <= reference, lighter than reference), rank by mass, report the winner + the
field, and append every candidate to `archive.jsonl` (the M4 flywheel tap).

Diversity is prompt-seeded, not temperature-hoped (plan §4): each idiom NAMES its section, and all
are SINGLE-SHELL / open (I-beam / ribs / channels / pockets) -- `fit_loss` rejects enclosed voids
(body_count 2) and open sections are the FDM-printable ones anyway (2026-07-13 finding, ADR-0096).

The selection is pure + unit-tested; the LLM authoring loop is lazy-imported (subscription only).
Usage-gated per the subscription discipline. NOT started until a valid `claude login` -- the first
run 401'd (TEST_LOG row 74).

Usage:
  FORGE_PARAM_OPT=1 FORGE_FEA_TIMEOUT_S=400 \
  uv run --no-sync python scripts/arm_search/m1_search.py \
    --span 142 --width 24 --height 60 \
    --ref-mass-g 145.49 --ref-deflection-mm 0.0457 --target-mass-g 123.7 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forge_mind.structures.selection import select_winner as select_m1_winner

# Open-section structural idioms (single external shell). Prompt-seeded diversity, plan §4.
IDIOMS: list[tuple[str, str]] = [
    ("i_beam", "an I-beam section: top and bottom flanges joined by one central vertical web"),
    (
        "ribbed_channel",
        "an open U/C-channel with external cross ribs (ribs outside, no enclosed cavity)",
    ),
    (
        "twin_plate_truss",
        "two thin side plates joined by an open diagonal cross-brace (a truss web)",
    ),
    (
        "hogged_pockets",
        "a solid beam with open milled pockets on each face (open to the surface, never sealed)",
    ),
    (
        "tapered_web",
        "a single tapered web thickest at the fixed end, thinning toward the far end, with edge rails",
    ),
]


def build_prompt(idiom_clause: str, *, span: float, width: float, height: float) -> str:
    """Compose one idiom's link brief. States the interface (pinned dims) + envelope + the idiom +
    the single-shell constraint. The mass lever does the lightening; this fixes what must not move."""
    return (
        f"A structural link for a small robot arm, 3D printed in PLA. It spans {span:.0f} mm end to "
        f"end. At EACH end is a flat {width:.0f} mm by 40 mm mounting pad with two 3.4 mm bolt holes "
        f"spaced 20 mm apart (M3 bolts to a servo bracket) -- these pads and holes are fixed. Between "
        f"the pads the body is {idiom_clause}, about {width:.0f} mm wide and {height:.0f} mm tall, "
        f"sized to resist bending under a load at the far end. It MUST be a single solid shell with NO "
        f"enclosed internal cavities (open sections only). Use as little material as possible while "
        f"keeping the beam stiff."
    )


# Selection now lives in forge_mind.structures.selection (imported above); these names remain
# re-exported here for existing callers/tests of this script.


def run_search(
    *,
    span: float,
    width: float,
    height: float,
    ref_mass_g: float,
    ref_deflection_mm: float,
    target_mass_g: float,
    material_id: str,
    out_dir: Path,
    idioms: list[tuple[str, str]],
    max_iterations: int | None,
) -> dict[str, Any]:
    import importlib

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    m1_probe = importlib.import_module("m1_probe")  # sibling CLI (engine logic lives in structures)

    archive = out_dir / "archive.jsonl"
    results: list[dict[str, Any]] = []
    for name, clause in idioms:
        prompt = build_prompt(clause, span=span, width=width, height=height)
        rec = m1_probe.run_m1(
            prompt,
            target_mass_g=target_mass_g,
            ref_mass_g=ref_mass_g,
            ref_deflection_mm=ref_deflection_mm,
            material_id=material_id,
            out_dir=out_dir / name,
            max_iterations=max_iterations,
        )
        rec["idiom"] = name
        results.append(rec)
        with archive.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    selection = select_m1_winner(results)
    (out_dir / "m1_search_report.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return selection


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M1 N-idiom search (plan Ticket 5, single link).")
    ap.add_argument("--span", type=float, required=True)
    ap.add_argument("--width", type=float, default=24.0)
    ap.add_argument("--height", type=float, default=60.0)
    ap.add_argument("--ref-mass-g", type=float, required=True)
    ap.add_argument("--ref-deflection-mm", type=float, required=True)
    ap.add_argument("--target-mass-g", type=float, required=True)
    ap.add_argument("--material", default="pla")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-iterations", type=int, default=None)
    ap.add_argument("--idioms", help="comma-separated idiom names (default: all)")
    args = ap.parse_args(argv)

    chosen = IDIOMS
    if args.idioms:
        want = {s.strip() for s in args.idioms.split(",")}
        chosen = [it for it in IDIOMS if it[0] in want]
        if not chosen:
            print(
                json.dumps(
                    {"error": f"no idioms matched {sorted(want)}; known: {[i[0] for i in IDIOMS]}"}
                )
            )
            return 2

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = run_search(
        span=args.span,
        width=args.width,
        height=args.height,
        ref_mass_g=args.ref_mass_g,
        ref_deflection_mm=args.ref_deflection_mm,
        target_mass_g=args.target_mass_g,
        material_id=args.material,
        out_dir=out_dir,
        idioms=chosen,
        max_iterations=args.max_iterations,
    )
    print(json.dumps(selection, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
