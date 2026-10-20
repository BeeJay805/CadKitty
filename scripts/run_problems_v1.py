"""Run the problems_v1 intake bench (ADR-0087 S7) live and score it deterministically.

Usage (subscription: set FORGE_LLM_PROVIDER=claude_code; metered: ANTHROPIC_API_KEY):

    uv run python scripts/run_problems_v1.py                  # all prompts, intake only
    uv run python scripts/run_problems_v1.py --only nr200_ssd --only spec_plate
    uv run python scripts/run_problems_v1.py --build-goldens  # + the 3 real golden builds

Results land in benchmarks/results/problems_v1_<UTC-stamp>/ (gitignored): one outcome json per
prompt, report.json, and summary.md. Every real run gets a docs/TEST_LOG.md row the day it runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.config import BRAIN_MODEL  # noqa: E402
from forge_mind.invention.bench_problems import (  # noqa: E402
    aggregate,
    bench_dir,
    load_bench,
    run_prompt,
    score_prompt,
)
from forge_mind.invention.problem_intake import confirm_and_build  # noqa: E402
from forge_mind.llm.provider import make_client  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="problems_v1 intake bench (ADR-0087 S7)")
    ap.add_argument("--only", action="append", default=[], help="run only these prompt ids")
    ap.add_argument("--model", default=None, help=f"brain model (default: {BRAIN_MODEL})")
    ap.add_argument(
        "--build-goldens",
        action="store_true",
        help="ALSO run the real engine build + env checks on golden prompts that reach spec_ready",
    )
    args = ap.parse_args()

    bench = load_bench()
    prompts = [p for p in bench.prompts if not args.only or p.id in args.only]
    if not prompts:
        print(f"no prompts matched --only {args.only}", file=sys.stderr)
        return 2

    model = args.model or BRAIN_MODEL
    client = make_client()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "benchmarks" / "results" / f"problems_v1_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = []
    for prompt in prompts:
        t0 = time.monotonic()
        outcome = run_prompt(prompt, client=client, model=model)
        score = score_prompt(prompt, outcome)
        elapsed = time.monotonic() - t0
        scores.append(score)
        (out_dir / f"{prompt.id}.json").write_text(
            json.dumps(
                {
                    "outcome": outcome.model_dump(mode="json"),
                    "score": score.model_dump(mode="json"),
                    "elapsed_s": round(elapsed, 1),
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        flag = "OK" if (score.refusal_ok and not score.invented and score.route_ok) else "LOOK"
        print(
            f"[{flag}] {prompt.id}: stage={score.final_stage or 'spec'} "
            f"recall={score.recall} q={score.question_count} invented={len(score.invented)} "
            f"({elapsed:.0f}s)"
        )

        if args.build_goldens and prompt.golden_build and outcome.state is not None:
            state = outcome.state
            if state.stage == "spec_ready":
                from forge_mind.api.core import produce_for_invention
                from forge_mind.fabrication import FabricationMode

                job = out_dir / f"golden_{prompt.id}"
                job.mkdir(parents=True, exist_ok=True)

                def produce(p: str, ctx: str, material: str = "PLA", _job: Path = job) -> object:
                    # the goldens measure the PRINTABLE lane — same mode as the lab adapter
                    return produce_for_invention(
                        p,
                        _job,
                        extra_context=ctx or None,
                        material=material,
                        fabrication_mode=FabricationMode.PRINT_3D,
                    )

                built = confirm_and_build(state, produce=produce)  # type: ignore[arg-type]
                # The BuildRecord carries the disposition but not WHY — pull the engine's own
                # validation report from the job dir so a rejected golden is self-documenting
                # (the first golden run forced a spelunk; never again).
                record = built.build.model_dump(mode="json") if built.build else {}
                report_path = job / "report.json"
                if report_path.is_file():
                    rep = json.loads(report_path.read_text(encoding="utf-8"))
                    val = rep.get("validation", {}) or {}
                    record["reject_reason"] = {
                        "hard_ok": val.get("hard_ok"),
                        "validation_passed": val.get("passed"),
                        "stopped_reason": rep.get("stopped_reason"),
                        "issues": list(val.get("issues", []))[:6],
                    }
                (out_dir / f"golden_{prompt.id}.json").write_text(
                    json.dumps(record, indent=1), encoding="utf-8"
                )
                envs = built.build.env_results if built.build else ()
                rr = record.get("reject_reason", {})
                print(
                    f"    golden build: {built.build.disposition if built.build else '?'} "
                    f"(hard_ok={rr.get('hard_ok')}, {len(rr.get('issues', []))} issue(s)); "
                    f"env checks: {sum(1 for r in envs if r.status == 'pass')}/{len(envs)} pass"
                )
            else:
                print(f"    golden build skipped: stage={state.stage}")

    report = aggregate(bench.version, scores)
    (out_dir / "report.json").write_text(report.model_dump_json(indent=1), encoding="utf-8")
    lines = [
        f"# problems_v1 run {stamp}",
        f"model: {model}  prompts: {report.prompts}",
        f"GATE: invented_fact_total={report.invented_fact_total} "
        f"refusal_correctness={report.refusal_correctness:.0%} -> "
        f"{'PASS' if report.gate_pass else 'FAIL'}",
        f"route_accuracy={report.route_accuracy:.0%} "
        f"missing_branch_rate={report.missing_branch_rate:.0%}",
        f"mean_recall={report.mean_recall:.0%} mean_precision={report.mean_precision:.0%}",
        f"question_discipline={report.question_discipline:.0%} "
        f"thermal_ceiling={report.thermal_ceiling_rate:.0%} "
        f"spurious_refusals={report.spurious_refusals} errors={report.errors}",
        "",
        "| prompt | stage | recall | q | invented | refusal_ok | missing_domains |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in report.scores:
        lines.append(
            f"| {s.prompt_id} | {s.final_stage or 'spec'} | {s.recall} | {s.question_count} "
            f"| {len(s.invented)} | {s.refusal_ok} | {', '.join(s.missing_domains) or '-'} |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nresults: {out_dir}")
    print(f"bench dir: {bench_dir()}")
    return 0 if report.gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
