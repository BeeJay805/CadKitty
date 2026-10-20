"""Export allowlisted public evidence from retained customer outcome journals."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from scripts.customer_outcome_bench import (
    DEFAULT_CORPUS,
    MODEL,
    SCOPE,
    encoded,
    load_corpus,
    read_events,
    save_new,
    sha,
    strict_json,
)


def number(value: Any) -> float | None:
    if type(value) not in {float, int} or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def public_result(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """An allowlist, not a recursive blacklist. Raw prompts, errors and receipts stay private."""
    observed = row.get("observation", {})
    assessed = row.get("assessment", {})
    actual = observed.get("data_origin") == "actual_customer_pipeline"
    eligible = actual and row.get("source_unchanged") is True and row.get("error") is None
    valid_record = row.get("source_unchanged") is True and row.get("error") is None
    return {
        "case_id": case["id"],
        "family": case["family"],
        "split": case["split"],
        "attempt_id": row["attempt_id"],
        "condition": row.get("condition")
        if row.get("condition") in {"cold", "warm"}
        else "unverified",
        "attempt_status": row.get("attempt_status", "finished"),
        "data_origin": "actual_customer_pipeline" if actual else "synthetic_or_missing",
        "state": observed.get("state")
        if observed.get("state")
        in {
            "complete",
            "completed",
            "ready",
            "done",
            "failed",
            "needs_input",
            "needs_approval",
            "budget_exhausted",
            "cancelled",
            "queued",
            "running",
            "review_required",
            "contract_satisfied",
            "needs_review",
            "error",
            "interrupted",
            "cancelling",
        }
        else "other_or_missing",
        "actual_model": MODEL if observed.get("actual_model") == MODEL else "unverified",
        "identity_verified": assessed.get("actual_identity_verified") is True,
        "nominal_digital_pass": assessed.get("nominal_digital_pass") is True and eligible,
        "clarification_pass": assessed.get("clarification_pass") is True and eligible,
        "nominal_status": assessed.get("nominal_status")
        if valid_record and assessed.get("nominal_status") in {"pass", "fail", "unassessed"}
        else "unassessed",
        "digital_delivery": assessed.get("digital_delivery") is True and valid_record,
        "automatic_delivery_status": observed.get("automatic_delivery_status")
        if observed.get("automatic_delivery_status")
        in {
            "delivered",
            "captured",
            "incomplete",
            "failed",
            "not_requested",
            "unavailable",
            "cancelled",
            "budget_exhausted",
        }
        else "unavailable",
        "source_unchanged": row.get("source_unchanged") is True,
        "trial_error": row.get("error") is not None,
        "first_cad_seconds": number(observed.get("first_cad_seconds")),
        "total_seconds": number(observed.get("total_seconds")),
        "owner_wait_seconds": number(observed.get("owner_wait_seconds")),
        **{
            name: number(observed.get(name))
            for name in (
                "model_calls",
                "failed_calls",
                "build_retries",
                "questions",
                "unresolved",
                "required_failed",
            )
        },
        "geometry_file_sha256": observed.get("nominal", {}).get("step_sha256")
        if isinstance(observed.get("nominal", {}).get("step_sha256"), str)
        and len(observed["nominal"]["step_sha256"]) == 64
        and all(c in "0123456789abcdef" for c in observed["nominal"]["step_sha256"])
        else None,
        "result_sha256": row["result_sha256"],
        "requirement_statuses": [
            {"requirement_id": duty["requirement_id"], "status": duty["status"]}
            for duty in observed.get("requirement_statuses", [])
            if isinstance(duty, dict)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", str(duty.get("requirement_id", "")))
            and duty.get("status") in {"pass", "fail", "unverified"}
        ],
        "worker_mode": observed.get("executor")
        if observed.get("executor") in {"docker", "trusted_local"}
        else "unverified",
        "worker_image": observed.get("worker", {}).get("image")
        if re.fullmatch(r"sha256:[a-f0-9]{64}", str(observed.get("worker", {}).get("image", "")))
        else None,
        "full_task_success": None,
        "physical_readiness": "unverified",
    }


def build_report(run: Path, corpus: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    rows = read_events(run / "events.jsonl")
    if not rows or rows[0]["kind"] != "run":
        raise ValueError("Missing run identity")
    cases, corpus_hash = load_corpus(corpus)
    identity = rows[0]["identity"]
    if identity["corpus_sha256"] != corpus_hash:
        raise ValueError("Report corpus differs from frozen run")
    known = {case["id"]: case for case in cases}
    public = []
    finished = {row["attempt_id"]: row for row in rows if row["kind"] == "attempt_finished"}
    abandoned_ids = {row["attempt_id"] for row in rows if row["kind"] == "attempt_abandoned"}
    for row in rows:
        if row["kind"] != "attempt_started":
            continue
        if not re.fullmatch(r"[a-f0-9]{32}", row["attempt_id"]):
            raise ValueError("Invalid attempt identity")
        if row["attempt_id"] in finished:
            result = finished[row["attempt_id"]]
            raw = (run / row["attempt_id"] / "result.json").read_bytes()
            if sha(raw) != result["result_sha256"]:
                raise ValueError("Retained result changed")
            retained = strict_json(raw)
            journal_result = {
                key: value
                for key, value in result.items()
                if key
                not in {"sequence", "previous_hash", "kind", "at", "event_hash", "result_sha256"}
            }
            if retained != journal_result:
                raise ValueError("Journal result differs from immutable result file")
        else:
            result = {
                **row,
                "condition": identity["condition"],
                "attempt_status": "abandoned"
                if row["attempt_id"] in abandoned_ids
                else "unfinished",
                "result_sha256": None,
                "error": "Incomplete attempt",
                "observation": {
                    "data_origin": "actual_customer_pipeline"
                    if identity["adapter"] == "customer_local"
                    else "synthetic_fixture"
                },
            }
        public.append(public_result(result, known[row["case_id"]]))
    terminal = {
        row["attempt_id"]
        for row in rows
        if row["kind"] in {"attempt_finished", "attempt_abandoned"}
    }
    abandoned = sum(row["kind"] == "attempt_abandoned" for row in rows)
    pending = sum(
        row["kind"] == "attempt_started" and row["attempt_id"] not in terminal for row in rows
    )
    groups = []
    for split in ("development", "heldout"):
        for condition in ("cold", "warm"):
            group = [
                row
                for row in public
                if row["split"] == split
                and row["condition"] == condition
                and row["data_origin"] == "actual_customer_pipeline"
            ]
            if not group:
                continue
            first: dict[str, dict[str, Any]] = {}
            for row in group:
                first.setdefault(row["case_id"], row)
            groups.append(
                {
                    "split": split,
                    "condition": condition,
                    "actual_attempts": len(group),
                    "actual_distinct_cases": len(first),
                    "first_attempt_nominal_digital_passes": sum(
                        row["nominal_digital_pass"] for row in first.values()
                    ),
                    "first_attempt_denominator": len(first),
                    "all_attempt_nominal_digital_passes": sum(
                        row["nominal_digital_pass"] for row in group
                    ),
                    "clarification_passes": sum(row["clarification_pass"] for row in group),
                    "unassessed_nominal_attempts": sum(
                        row["nominal_status"] == "unassessed" for row in group
                    ),
                    "latency": {
                        name: {
                            "observed": len(values),
                            "missing": len(group) - len(values),
                            "median_seconds": statistics.median(values) if values else None,
                        }
                        for name in ("first_cad_seconds", "total_seconds", "owner_wait_seconds")
                        for values in [[row[name] for row in group if row[name] is not None]]
                    },
                }
            )
    return {
        "schema_version": 1,
        "title": "CadKitty customer outcome development evidence",
        "scope": SCOPE,
        "corpus_sha256": corpus_hash,
        "journal_head_sha256": rows[-1]["event_hash"],
        "source_inventory_sha256": sha(encoded(identity["source_files"])),
        "source_files": {
            name: digest
            for name, digest in identity["source_files"].items()
            if not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and "\\" not in name
            and name.startswith(("engine/", "harness/"))
        },
        "runtime": identity.get("runtime"),
        "configured_model": MODEL,
        "generation_policy": identity.get("generation_policy"),
        "delivery_protocol": identity.get("delivery_protocol"),
        "harness_version": identity.get("harness_version"),
        "oracle_protocol": identity.get("oracle_protocol"),
        "staged_native_planning_flag": identity.get("staged_native_planning_flag"),
        "model_identity_scope": "CLI invocation and returned usage; no independent provider attestation",
        "condition_scope": "Operator-declared cold/warm host/cache condition; not independently established",
        "owner_wait_scope": "Automated script wait only; human labor and labor savings unmeasured",
        "corpus_cases": len(cases),
        "selected_cases": len(identity["case_ids"]),
        "completed_attempts": len(finished),
        "abandoned_attempts": abandoned,
        "unfinished_attempts": pending,
        "synthetic_attempts": sum(
            row["data_origin"] != "actual_customer_pipeline" for row in public
        ),
        "groups": groups,
        "attempts": public,
        "heldout_scope": "Fixed prospective split. Once a case informs repairs, it is development evidence for that source lineage.",
        "privacy": "Allowlisted facts only. Authentication, local paths, raw model text, worker diagnostics and error details are excluded.",
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# " + report["title"],
        "",
        report["scope"],
        "",
        f"Completed attempts: {report['completed_attempts']}. Synthetic attempts: {report['synthetic_attempts']}.",
        f"Abandoned attempts: {report['abandoned_attempts']}. Unfinished attempts: {report['unfinished_attempts']}.",
        "",
        "Synthetic attempts are excluded from actual outcome rates.",
        "",
        "| Case | Split | Condition | Origin | Attempt | Nominal | Files + guide | Nominal digital pass | First CAD s | Total s | Owner wait s | Calls | Unresolved |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["attempts"]:
        columns = [
            row[name]
            for name in (
                "case_id",
                "split",
                "condition",
                "data_origin",
                "attempt_status",
                "nominal_status",
                "digital_delivery",
                "nominal_digital_pass",
                "first_cad_seconds",
                "total_seconds",
                "owner_wait_seconds",
                "model_calls",
                "unresolved",
            )
        ]
        lines.append(
            "| "
            + " | ".join("unmeasured" if value is None else str(value) for value in columns)
            + " |"
        )
    lines += [
        "",
        report["owner_wait_scope"],
        "",
        report["condition_scope"],
        "",
        report["model_identity_scope"],
        "",
        report["heldout_scope"],
        "",
        "Corpus SHA-256: `" + report["corpus_sha256"] + "`",
        "",
        "Journal head SHA-256: `" + report["journal_head_sha256"] + "`",
        "",
    ]
    return "\n".join(lines)


def html_report(report: dict[str, Any]) -> str:
    fields = (
        "case_id",
        "split",
        "condition",
        "data_origin",
        "attempt_status",
        "nominal_status",
        "digital_delivery",
        "nominal_digital_pass",
        "first_cad_seconds",
        "total_seconds",
        "owner_wait_seconds",
        "model_calls",
        "unresolved",
    )
    header = "".join("<th>" + html.escape(name.replace("_", " ")) + "</th>" for name in fields)
    body = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape("unmeasured" if row[name] is None else str(row[name])) + "</td>"
            for name in fields
        )
        + "</tr>"
        for row in report["attempts"]
    )
    return (
        """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Customer outcome evidence</title>
<style>body{font:18px system-ui;max-width:1500px;margin:40px auto;padding:20px;color:#182027;background:#f5f7f8}p{max-width:1000px;line-height:1.6}table{border-collapse:collapse;background:white}th,td{padding:12px;border:1px solid #bdc9d0;text-align:left}th{background:#e0e9ef}.scroll{overflow:auto}code{overflow-wrap:anywhere}</style>
<h1>"""
        + html.escape(report["title"])
        + "</h1><p>"
        + html.escape(report["scope"])
        + "</p><p>"
        + html.escape(
            f"{report['completed_attempts']} completed; {report['synthetic_attempts']} synthetic; {report['abandoned_attempts']} abandoned; {report['unfinished_attempts']} unfinished. Synthetic attempts never count as actual passes."
        )
        + "</p><div class=scroll><table><thead><tr>"
        + header
        + "</tr></thead><tbody>"
        + body
        + "</tbody></table></div>"
        + "".join(
            "<p>" + html.escape(report[key]) + "</p>"
            for key in (
                "owner_wait_scope",
                "condition_scope",
                "model_identity_scope",
                "heldout_scope",
                "privacy",
            )
        )
        + "<p>Corpus SHA-256: <code>"
        + report["corpus_sha256"]
        + "</code></p><p>Journal head: <code>"
        + report["journal_head_sha256"]
        + "</code></p></html>"
    )


def export_report(run: Path, output: Path, corpus: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    report = build_report(run, corpus)
    output.mkdir(parents=True, exist_ok=False)
    save_new(output / "evidence.json", report)
    (output / "evidence.md").write_text(markdown(report), encoding="utf-8")
    (output / "index.html").write_text(html_report(report), encoding="utf-8")
    save_new(
        output / "SHA256.json",
        {path.name: sha(path.read_bytes()) for path in sorted(output.iterdir())},
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args(argv)
    report = export_report(args.run, args.output, args.corpus)
    print(
        json.dumps(
            {
                "completed_attempts": report["completed_attempts"],
                "synthetic_attempts": report["synthetic_attempts"],
            }
        )
    )


if __name__ == "__main__":
    main()
