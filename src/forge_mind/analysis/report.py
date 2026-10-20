"""Render a FailureDataset to markdown + JSON, and a per-trial records.jsonl."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from forge_mind.analysis.records import MANUAL_LABELS, FailureDataset, dataset_to_dict


def render_markdown(dataset: FailureDataset) -> str:
    d = dataset
    lines: list[str] = []
    lines.append(f"# Failure analysis: {d.bench_id}")
    lines.append("")
    lines.append(f"- **Run:** {d.run_id}")
    lines.append(f"- **Pass rate:** {d.n_passed}/{d.n_trials} ({d.pass_rate * 100:.0f}%)")
    lines.append(f"- **Mean cost:** ${d.mean_cost_usd:.4f}/trial")
    lines.append(f"- **Mean latency:** {d.mean_duration_ms / 1000:.1f}s/trial")
    lines.append("")

    lines.append("## Failure counts by label")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|-------|-------|")
    for label, count in d.label_counts.items():
        lines.append(f"| {label} | {count} |")
    lines.append("")
    if MANUAL_LABELS:
        lines.append(
            "_Manual-only labels (never auto-assigned; add by reviewing artifacts): "
            + ", ".join(MANUAL_LABELS)
            + "._"
        )
        lines.append("")

    lines.append("## Failure counts by first-failing stage")
    lines.append("")
    lines.append("| stage | count |")
    lines.append("|-------|-------|")
    for stage, count in d.stage_counts.items():
        lines.append(f"| {stage} | {count} |")
    lines.append("")

    lines.append("## Trials by generation path")
    lines.append("")
    lines.append("| gen_path | count |")
    lines.append("|----------|-------|")
    for path, count in d.gen_path_counts.items():
        lines.append(f"| {path} | {count} |")
    lines.append("")

    lines.append("## Top failed check_ids")
    lines.append("")
    lines.append("| check_id | count |")
    lines.append("|----------|-------|")
    for cid, count in d.top_check_ids:
        lines.append(f"| {cid} | {count} |")
    lines.append("")

    lines.append("## Representative failed artifacts")
    lines.append("")
    if d.representative_artifacts:
        for label, path in d.representative_artifacts.items():
            lines.append(f"- **{label}**: `{path}`")
    else:
        lines.append("- (no failures)")
    lines.append("")

    return "\n".join(lines)


def write_dataset(dataset: FailureDataset, out_dir: Path) -> dict[str, Path]:
    """Write report.md, report.json, and records.jsonl into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "report.md"
    json_path = out_dir / "report.json"
    jsonl_path = out_dir / "records.jsonl"

    md_path.write_text(render_markdown(dataset) + "\n", encoding="utf-8")

    # Keep the json report compact: drop the heavy per-record list (it lives in
    # records.jsonl) but keep the aggregates.
    payload = dataset_to_dict(dataset)
    payload.pop("records", None)
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in dataset.records:
            fh.write(json.dumps(asdict(record)) + "\n")

    return {"markdown": md_path, "json": json_path, "jsonl": jsonl_path}


__all__ = ["render_markdown", "write_dataset"]
