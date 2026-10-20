"""Lab pipeline probe -- the WHOLE invention pipeline, end-to-end, heavily instrumented.

Runs:  problem -> clarifying QUESTIONS -> ANSWERS (simulated maker) -> CONCEPTS (front door) ->
BUILD PART (the agentic engine) -> INVENTION PACKAGE, LIVE on the Opus subscription ($0), recording
per-stage wall-clock + tokens + the part-build per-iteration breakdown so the BOTTLENECKS are obvious.

Run (subscription, $0):
  $env:FORGE_LLM_PROVIDER='claude_code'; $env:FORGE_INVENTION='1'
  uv run --no-sync python -u scripts/lab_pipeline_probe.py [n_problems] [minutes_budget]

Writes benchmarks/results/lab_probe_<stamp>/report.json (rewritten after every problem -> kill-safe).
The ANSWERS are SIMULATED (LLM-as-user) ONLY so the run is unattended; a real run takes human answers.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge_mind.agentic.result import produce_part
from forge_mind.api.core import _compose_generated_part
from forge_mind.config import OPUS
from forge_mind.fabrication import FabricationMode
from forge_mind.invention.clarify import (
    answer_clarifying_questions,
    clarifications_to_context,
    generate_clarifying_questions,
)
from forge_mind.invention.frontdoor import run_front_door
from forge_mind.invention.lifecycle import concept_to_prompt
from forge_mind.invention.package import build_invention_package
from forge_mind.llm.provider import make_client

PROBLEMS = [
    "my desk cables keep sliding off the back edge of my desk onto the floor",
    "I need to stop a hinged door from slamming shut when a breeze catches it",
    "my phone won't stay propped at a good angle on my nightstand for watching videos",
    "the lid on my filament dry-box doesn't seal so the desiccant stops working",
    "my water bottle rattles out of its bike cage on rough trails",
]

MAX_ITERATIONS = 8
BUDGET_USD = 3.0  # inert on the subscription (cost records 0); bounds an API run


def _t() -> float:
    return time.perf_counter()


def _usage_row(stage: str, usage: Any, dur: float) -> dict[str, Any]:
    return {
        "stage": stage,
        "duration_s": round(dur, 2),
        "model": getattr(usage, "model", None),
        "tokens_in": getattr(usage, "input_tokens", None),
        "tokens_out": getattr(usage, "output_tokens", None),
        "cost_usd": round(getattr(usage, "cost_usd", 0.0) or 0.0, 4),
    }


class TimingClient:
    """Wraps an LLM client and records every call (method, model, wall-clock, tokens) so a stage's
    wall-clock can be split into LLM time (text = author/brief/concepts, vision = critic/judge) vs
    NON-LLM time (the build123d/OCP CAD kernel + PNG rendering + Python overhead)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.calls: list[dict[str, Any]] = []

    def _record(self, method: str, kw: dict[str, Any], dur: float, usage: Any) -> None:
        self.calls.append(
            {
                "method": method,
                "model": kw.get("model"),
                "dur_s": round(dur, 2),
                "tok_in": getattr(usage, "input_tokens", None),
                "tok_out": getattr(usage, "output_tokens", None),
            }
        )

    def generate_structured(self, *args: Any, **kw: Any) -> Any:
        t = _t()
        result, usage = self.inner.generate_structured(*args, **kw)
        self._record("text", kw, _t() - t, usage)
        return result, usage

    def generate_structured_with_image(self, *args: Any, **kw: Any) -> Any:
        t = _t()
        result, usage = self.inner.generate_structured_with_image(*args, **kw)
        self._record("vision", kw, _t() - t, usage)
        return result, usage


def run_one(problem: str, out_dir: Path, client: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {"problem": problem, "stages": [], "ok": False}
    t_total = _t()

    # 1. clarifying questions
    t = _t()
    questions, uq = generate_clarifying_questions(problem, client=client, model=OPUS)
    rec["stages"].append(
        {
            **_usage_row("questions", uq, _t() - t),
            "n_questions": len(questions),
            "questions": [{"q": q.question, "pins": q.pins} for q in questions],
        }
    )

    # 2. answers (SIMULATED maker -- unattended testing only)
    t = _t()
    answers, ua = answer_clarifying_questions(problem, questions, client=client, model=OPUS)
    rec["stages"].append(
        {
            **_usage_row("answers(sim)", ua, _t() - t),
            "answers": [{"q": a.question, "a": a.answer} for a in answers],
        }
    )
    context = clarifications_to_context(answers)

    # 3. concepts (front door), GROUNDED in the answers
    t = _t()
    plan, uf = run_front_door(problem, client=client, model=OPUS, context=context)
    chosen = plan.recommended()
    rec["stages"].append(
        {
            **_usage_row("concepts", uf, _t() - t),
            "n_concepts": len(plan.concepts),
            "mechanisms": [c.primary_mechanism.value for c in plan.concepts],
            "feasibility": plan.feasibility_verdict.value,
            "chosen": chosen.name,
        }
    )

    # 4. BUILD the chosen concept into a part -- the long pole. extra_context grounds the brief.
    # Snapshot the LLM-call log around the build so its wall-clock splits into LLM time (text =
    # author/brief, vision = critic/judge) vs NON-LLM time (the CAD kernel + rendering + overhead).
    n0 = len(client.calls) if hasattr(client, "calls") else 0
    t = _t()
    pr = produce_part(
        concept_to_prompt(problem, chosen),
        out_dir / "part",
        client=client,
        extra_context=context,
        max_iterations=MAX_ITERATIONS,
        budget_usd=BUDGET_USD,
    )
    build_dur = _t() - t
    build_calls = client.calls[n0:] if hasattr(client, "calls") else []
    llm_s = sum(c["dur_s"] for c in build_calls)
    vision_s = sum(c["dur_s"] for c in build_calls if c["method"] == "vision")
    text_s = sum(c["dur_s"] for c in build_calls if c["method"] == "text")
    iters = [
        {
            "i": it.iteration,
            "phase": it.phase,
            "dur_s": round(it.duration_s, 2),
            "built": it.built,
            "valid": it.validation_passed,
            "score": it.score,
            "kept_best": it.kept_as_best,
            "critic_ok": it.critic_looks_right,
            "resemblance": it.critic_resemblance,
            "err": it.sandbox_error,
        }
        for it in pr.loop.iterations
    ]
    rec["stages"].append(
        {
            "stage": "build_part",
            "duration_s": round(_t() - t, 2),
            "disposition": pr.disposition,
            "stopped_reason": pr.loop.stopped_reason,
            "geometry_passed": pr.loop.geometry_passed,
            "critic_approved": pr.loop.critic_approved,
            "n_iterations": len(iters),
            "cost_usd": round(pr.total_cost_usd, 4),
            "object_name": pr.loop.brief.object_name if pr.loop.brief else None,
            "n_features": len(pr.loop.brief.features) if pr.loop.brief else None,
            # the split: where build_part's wall-clock actually goes
            "llm_calls": len(build_calls),
            "llm_time_s": round(llm_s, 2),
            "vision_time_s": round(vision_s, 2),  # critic + judge (image calls)
            "text_llm_time_s": round(text_s, 2),  # brief + author (structured calls)
            "nonllm_time_s": round(build_dur - llm_s, 2),  # CAD kernel + render + overhead
            "calls": build_calls,
            "iterations": iters,
        }
    )

    # 5. assemble the package (deterministic, ~0 cost)
    t = _t()
    gp = _compose_generated_part(pr, "PLA", FabricationMode.GENERAL)
    pkg = build_invention_package(
        part_name=gp.part_name,
        disposition=gp.disposition,
        kit=gp.kit,
        confidence=gp.confidence,
        artifacts=gp.artifacts,
        problem_statement=problem,
        feasibility=plan.feasibility(),
        concept=chosen,
    )
    rec["stages"].append(
        {
            "stage": "package",
            "duration_s": round(_t() - t, 2),
            "part_name": pkg.part_name,
            "disposition": pkg.disposition,
            "bom_summary": pkg.bom_summary,
            "banner": pkg.physical_validation.banner,
        }
    )

    rec["total_duration_s"] = round(_t() - t_total, 2)
    rec["ok"] = pr.disposition in ("shipped", "shipped_with_caveats")
    return rec


def _print_run(rec: dict[str, Any]) -> None:
    if rec.get("error"):
        print(f"  FAILED: {rec['error']}")
        return
    print(f"  total {rec['total_duration_s']}s  ok={rec['ok']}")
    for s in rec["stages"]:
        line = f"    {s['stage']:<13} {s['duration_s']:>7.2f}s"
        if "n_iterations" in s:
            line += (
                f"  iters={s['n_iterations']} disp={s['disposition']} stop={s['stopped_reason']}"
            )
        elif s.get("tokens_out") is not None:
            line += f"  tok_out={s['tokens_out']}"
        if s["stage"] == "concepts":
            line += f"  mechs={s['mechanisms']}"
        print(line)
        if s.get("llm_calls") is not None:
            print(
                f"      -> LLM {s['llm_time_s']:.0f}s in {s['llm_calls']} calls "
                f"(vision {s['vision_time_s']:.0f}s + text {s['text_llm_time_s']:.0f}s)  |  "
                f"non-LLM (CAD kernel+render) {s['nonllm_time_s']:.0f}s"
            )


def _print_summary(runs: list[dict[str, Any]], out: Path) -> None:
    done = [r for r in runs if not r.get("error") and r.get("stages")]
    print("\n" + "=" * 72)
    print("SUMMARY / BOTTLENECKS")
    n_ok = sum(1 for r in done if r["ok"])
    print(f"problems: {len(runs)} run, {len(done)} completed, {n_ok} ok (shipped/with-caveats)")
    if not done:
        print(f"\nfull data: {out / 'report.json'}")
        return

    agg: dict[str, list[float]] = {}
    for r in done:
        for s in r["stages"]:
            agg.setdefault(s["stage"], []).append(s["duration_s"])
    total_mean = sum(sum(ds) / len(ds) for ds in agg.values())
    print("\nper-stage wall-clock (mean / max over completed runs):")
    rows = [(stage, sum(ds) / len(ds), max(ds)) for stage, ds in agg.items()]
    for stage, mean, mx in rows:
        share = 100 * mean / total_mean if total_mean else 0
        print(f"  {stage:<13} mean {mean:>7.2f}s  max {mx:>7.2f}s  ({share:4.0f}% of pipeline)")
    bottleneck = max(rows, key=lambda x: x[1])
    print(
        f"\n>>> BOTTLENECK: '{bottleneck[0]}' "
        f"({100 * bottleneck[1] / total_mean:.0f}% of mean wall-clock per problem)"
    )

    builds = [s for r in done for s in r["stages"] if s["stage"] == "build_part"]
    if builds:
        mean_iters = sum(b["n_iterations"] for b in builds) / len(builds)
        print(
            f"    build_part detail: mean {mean_iters:.1f} iterations/part | "
            f"dispositions: {', '.join(b['disposition'] for b in builds)} | "
            f"stops: {', '.join(b['stopped_reason'] for b in builds)}"
        )
        split = [b for b in builds if b.get("llm_calls") is not None]
        if split:
            mllm = sum(b["llm_time_s"] for b in split) / len(split)
            mvis = sum(b["vision_time_s"] for b in split) / len(split)
            mnon = sum(b["nonllm_time_s"] for b in split) / len(split)
            dom = "LLM calls" if mllm > mnon else "CAD kernel/render"
            print(
                f"    build_part SPLIT (mean): LLM {mllm:.0f}s (vision {mvis:.0f}s) vs "
                f"non-LLM CAD+render {mnon:.0f}s  ->  {dom} dominates"
            )
    print(f"\nfull per-stage + per-iteration data: {out / 'report.json'}")


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 55.0
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path("benchmarks/results") / f"lab_probe_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    client = TimingClient(make_client())
    deadline = _t() + minutes * 60
    n = min(n, len(PROBLEMS))
    runs: list[dict[str, Any]] = []
    print(f"lab pipeline probe: {n} problems, {minutes:.0f}m budget, model=opus, out={out}")
    for i, problem in enumerate(PROBLEMS[:n]):
        if _t() > deadline:
            print(f"\n[time budget {minutes:.0f}m hit -- stopping before problem {i + 1}]")
            break
        print(f"\n=== problem {i + 1}/{n}: {problem}")
        pdir = out / f"p{i + 1}"
        pdir.mkdir(parents=True, exist_ok=True)
        try:
            rec = run_one(problem, pdir, client)
        except (
            Exception
        ) as e:  # a probe records the failing stage; it must never crash the whole run
            rec = {"problem": problem, "error": repr(e), "ok": False, "stages": []}
            print(f"  ERROR: {e!r}")
        runs.append(rec)
        (out / "report.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")  # kill-safe
        _print_run(rec)
    _print_summary(runs, out)


if __name__ == "__main__":
    main()
