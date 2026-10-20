"""Measure the print-lane slicer gate (ADR-0012 SS3) on real geometry -- $0, no LLM.

Slices every known-good corpus part (and the 3 row-38 goldens) with the committed profile and
reports the gcode facts, so we can see (a) the goldens get slicer verdicts and (b) the false-FAIL
SURFACE on known-goods -- how many trip each fact if it were promoted to a website-lane gate.

Honesty note: SUPPORT volume > 0 is expected non-zero on general parts (many legitimately need
support) -- it is a CELL-lane self-supporting requirement, NOT a website-lane false-FAIL. The
website-lane promotion candidates are the DEFECT facts (slicer WARNINGS and silently-DROPPED
layers); those are the numbers that must stay ~zero on known-goods. All three are reported apart.

Usage:
  uv run --no-sync python scripts/measure_slicer_gate.py [corpus_glob] [golden_run_dir]
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.audit.corpus import load_corpus  # noqa: E402
from forge_mind.slicing.gcode_facts import parse_gcode_facts  # noqa: E402
from forge_mind.slicing.slicer import find_slicer_exe, slice_part  # noqa: E402

_DEFAULT_GLOB = (
    "benchmarks/results/arbitrary_*,benchmarks/results/audit_fixtures_v1,"
    "benchmarks/results/problems_v1_*"
)
_DEFAULT_GOLDEN_RUN = "benchmarks/results/problems_v1_20260707T051100Z"  # the row-38 goldens


def _slice_one(label: str, stl: Path, workdir: Path, *, is_golden: bool) -> dict[str, Any]:
    rec: dict[str, Any] = {"label": label, "is_golden": is_golden, "stl": str(stl)}
    if not stl.exists():
        rec["error"] = "stl missing"
        return rec
    safe = label.replace("/", "_").replace(":", "_").replace("\\", "_")
    outcome = slice_part(stl, workdir / safe)
    if not outcome.ok:
        rec["error"] = outcome.error
        return rec
    try:
        mesh = trimesh.load(str(stl), force="mesh")
    except Exception as exc:
        mesh = None
        rec["mesh_error"] = f"{type(exc).__name__}: {exc}"
    facts = parse_gcode_facts(
        outcome.gcode_path.read_text(encoding="utf-8", errors="replace"),  # type: ignore[union-attr]
        warnings=outcome.warnings,
        mesh=mesh,
    )
    rec.update(
        ok=True,
        n_warnings=len(facts.warnings),
        warnings=facts.warnings[:5],
        support_mm3=facts.support_volume_mm3,
        first_layer_contact_mm2=facts.first_layer_contact_mm2,
        layer_count=facts.layer_count,
        n_dropped=len(facts.dropped_layers) if facts.dropped_layers is not None else None,
        min_area_ratio=facts.min_area_ratio,
    )
    return rec


def main() -> None:
    glob_pat = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_GLOB
    golden_run = REPO / (sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_GOLDEN_RUN)

    exe = find_slicer_exe()
    print(f"[slicer] {exe or 'NONE FOUND'}", flush=True)
    if exe is None:
        print(
            "no slicer binary -- install PrusaSlicer/OrcaSlicer or set FORGE_SLICER_EXE", flush=True
        )
        raise SystemExit(1)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "benchmarks" / "results" / f"slicer_gate_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "gcode"

    records: list[dict[str, Any]] = []

    corpus = load_corpus(REPO, glob_pat)
    print(f"[corpus] {len(corpus)} known-good parts", flush=True)
    for part in corpus:
        rec = _slice_one(part["label"], part["job_dir"] / "part.stl", work, is_golden=False)
        records.append(rec)
        print(f"  good {part['label']}: {_line(rec)}", flush=True)

    for golden in ("nr200_ssd", "nr200_hdd35", "meshify_duct"):
        stl = golden_run / f"golden_{golden}" / "part.stl"
        rec = _slice_one(f"golden::{golden}", stl, work, is_golden=True)
        records.append(rec)
        print(f"  GOLD {golden}: {_line(rec)}", flush=True)

    goods = [r for r in records if not r["is_golden"] and r.get("ok")]
    good_fail = [r for r in records if not r["is_golden"] and not r.get("ok")]
    warn_hits = [r for r in goods if r["n_warnings"] > 0]
    drop_hits = [r for r in goods if r.get("n_dropped")]
    support_hits = [r for r in goods if (r.get("support_mm3") or 0) > 0]

    summary = {
        "stamp": stamp,
        "glob": glob_pat,
        "slicer": str(exe),
        "known_goods_sliced": len(goods),
        "known_goods_slice_failed": len(good_fail),
        "false_fail_surface_warnings": len(warn_hits),  # website-lane promotion candidate
        "false_fail_surface_dropped": len(drop_hits),  # website-lane promotion candidate
        "support_needed_count": len(support_hits),  # cell-lane only -- expected non-zero
        "warn_labels": [r["label"] for r in warn_hits],
        "drop_labels": [r["label"] for r in drop_hits],
        "slice_failed_labels": [(r["label"], r.get("error")) for r in good_fail],
    }
    (out_dir / "records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print("\n==== SLICER GATE MEASUREMENT ====", flush=True)
    for k, v in summary.items():
        print(f"  {k}: {v}", flush=True)
    print(f"\nartifacts -> {out_dir}", flush=True)


def _line(rec: dict[str, Any]) -> str:
    if not rec.get("ok"):
        return f"SLICE-FAIL ({rec.get('error')})"
    support = rec.get("support_mm3")
    support_s = f"{support:.0f}mm3" if support is not None else "?"
    ratio = rec.get("min_area_ratio")
    ratio_s = f"{ratio:.2f}" if ratio is not None else "?"
    return (
        f"warns={rec['n_warnings']} support={support_s} "
        f"dropped={rec['n_dropped']} min_ratio={ratio_s}"
    )


if __name__ == "__main__":
    main()
