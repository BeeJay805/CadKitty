"""M2 scorer: measure a completed functional-arm run against the reference axes (plan §2-M2).

Reads a ``produce_functional_part`` run dir and emits the M2 objective row: disposition (the §4
hard filter), part count, per-part + total printed mass (fail-closed: any unmeasurable part ->
no total), and the run's own fit/motion verdicts VERBATIM (equal-function is claimed by the
pipeline's gates, never by this scorer). With ``--ref-json`` (a ``reference_objectives.json``)
it prints the honest comparison; without one it reports absolute numbers only.

Usage:
  uv run --no-sync python scripts/arm_search/m2_score.py --run C:/Users/brenn/work/probe_m2_arm \
    [--ref-json C:/Users/brenn/work/reference_so101/reference_objectives.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from forge_mind.structures.objectives import run_disposition, run_is_scorable, score_stl


def _current_part_ids(run_dir: Path) -> set[str] | None:
    """The CURRENT plan's part-dir ids (slugged the same way the pipeline names dirs), or None
    when no plan.json exists. A run dir accumulates part dirs across replans (measured: run 1's
    'turntable'/'gripper_plate' beside run 3's 'yaw_bracket'/'gripper_mount'), so scoring must
    filter to the current generation or it double-counts stale parts."""
    plan = run_dir / "plan.json"
    if not plan.exists():
        return None
    try:
        from forge_mind.agentic.loop import _disambiguate_slugs

        names = [p["name"] for p in json.loads(plan.read_text(encoding="utf-8"))["parts"]]
        return set(_disambiguate_slugs(names))
    except Exception:
        return None


def find_part_stls(run_dir: Path) -> list[tuple[str, Path]]:
    """(part_name, final_stl) per CURRENT-plan part, from each part dir's
    ``part_checkpoint.json`` -- the AUTHORITATIVE pointer the pipeline itself resumes from
    (verified against a real run: artifacts are slug-named inside iter_NN/, several per
    iteration, so globbing lies). A part whose checkpoint has no existing final_stl
    contributes nothing (fail-closed); stale part dirs from earlier replans are excluded."""
    current = _current_part_ids(run_dir)
    out: list[tuple[str, Path]] = []
    for sub in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        if current is not None and sub.name not in current:
            continue
        ck = sub / "part_checkpoint.json"
        if not ck.exists():
            continue
        try:
            payload = json.loads(ck.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        stl = payload.get("final_stl")
        if stl and Path(stl).exists():
            out.append((sub.name, Path(stl)))
    return out


def score_m2_run(run_dir: Path, *, material: str = "pla") -> dict[str, Any]:
    report_path = run_dir / "functional_report.json"
    fr: dict[str, Any] = {}
    if report_path.exists():
        fr = json.loads(report_path.read_text(encoding="utf-8"))
    named = find_part_stls(run_dir)
    parts = [(name, score_stl(p, material=material, load_case=None)) for name, p in named]
    masses = [po.mass_g for _n, po in parts]
    total = None if (not masses or any(m is None for m in masses)) else round(sum(masses), 1)
    return {
        "run": str(run_dir),
        "disposition": run_disposition(run_dir),
        "scorable": run_is_scorable(run_dir),  # the §4 hard filter
        "part_count_report": len(fr.get("parts", [])) or None,
        "part_count_stl": len(named),
        "stls": [str(p) for _n, p in named],
        "per_part_mass_g": {name: po.mass_g for name, po in parts},
        "total_printed_mass_g": total,
        "any_non_watertight": any(not po.watertight for _n, po in parts),
        # equal-function evidence comes from the pipeline's OWN gates, quoted verbatim:
        "fit": fr.get("fit"),
        "motion": fr.get("motion"),
        "judge": fr.get("judge"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="M2 objective scorer for a functional-arm run.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--ref-json", default=None)
    ap.add_argument("--material", default="pla")
    args = ap.parse_args(argv)
    row = score_m2_run(Path(args.run), material=args.material)
    if args.ref_json:
        ref = json.loads(Path(args.ref_json).read_text(encoding="utf-8"))
        ref_mass = ref.get("total_mass_g")
        ref_count = ref.get("part_count")
        cmp: dict[str, Any] = {"ref_total_mass_g": ref_mass, "ref_part_count": ref_count}
        if row["total_printed_mass_g"] is not None and ref_mass:
            cmp["mass_vs_ref_pct"] = round(100 * (row["total_printed_mass_g"] / ref_mass - 1), 1)
        row["reference_comparison"] = cmp
        row["comparison_caveat"] = (
            "equal-function NOT asserted by this scorer: the candidate's function evidence is "
            "its own fit/motion/judge gates above; the reference's function is its shipped design."
        )
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
