"""Re-render a committed failure dataset under the CURRENT taxonomy.

No LLM trials are run. Each record's primary_label is re-derived from its own
recorded ``primary_check_id`` via ``remap_validate_label`` (only validator-derived
labels change; upstream root-cause labels are preserved). The raw observations
(failed_check_ids, messages, artifacts, cost, latency) are untouched.

    uv run python scripts/reaggregate_analysis.py [analysis_dir] [run_id]

Defaults to benchmarks/analysis/bench-001-nema17-mount-v1 / n30.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from forge_mind.analysis.records import TrialRecord, aggregate
from forge_mind.analysis.report import render_markdown
from forge_mind.analysis.taxonomy import remap_validate_label

REPO = Path(__file__).resolve().parent.parent


def main() -> int:
    analysis_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO / "benchmarks" / "analysis" / "bench-001-nema17-mount-v1"
    )
    run_id = sys.argv[2] if len(sys.argv) > 2 else "n30"
    bench_id = analysis_dir.name

    records_path = analysis_dir / f"{run_id}-records.jsonl"
    if not records_path.exists():
        print(f"No records at {records_path}", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]

    changed = 0
    records: list[TrialRecord] = []
    for row in rows:
        old = row["primary_label"]
        new = remap_validate_label(old, row.get("primary_check_id")).value
        if new != old:
            changed += 1
        row["primary_label"] = new
        records.append(TrialRecord(**row))

    dataset = aggregate(bench_id, run_id, records)

    # Re-write the committed artifacts in place.
    (analysis_dir / f"{run_id}-report.md").write_text(
        render_markdown(dataset) + "\n", encoding="utf-8"
    )
    payload = asdict(dataset)
    payload.pop("records", None)
    (analysis_dir / f"{run_id}-report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (analysis_dir / f"{run_id}-records.jsonl").open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec)) + "\n")

    print(f"Re-labelled {changed}/{len(records)} records under the current taxonomy.")
    print("New label counts:")
    for label, count in dataset.label_counts.items():
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
