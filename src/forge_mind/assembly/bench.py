"""Assembly benchmark: measure the LLM driving the GENERAL assembly path (ADR-0010).

For each decontaminated prompt, run N trials of
    request -> AssemblyGraphGenerator (LLM) -> solve -> compile -> validate
and report the pass rate with a Wilson 95% lower bound, plus where failures land
(emit / solve / compile / validate) and the cost. The point is reliability AND
generality: a hinge AND a linear slide, both from the same catalog.

The harness enforces a hard ``max_cost_usd`` ceiling and stops cleanly if reached,
so a benchmark run can't blow a budget.
"""

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from forge_mind.assembly.generator import AssemblyGraphGenerator
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import CheckStatus


def wilson_lower_bound(passes: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion (honest small-n)."""
    if n == 0:
        return 0.0
    phat = passes / n
    denom = 1.0 + z * z / n
    center = phat + z * z / (2.0 * n)
    margin = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4.0 * n * n))
    return max(0.0, (center - margin) / denom)


@dataclass
class TrialResult:
    prompt_id: str
    passed: bool
    stage: str  # "emit" | "solve" | "compile" | "validate" | "pass"
    failed_checks: list[str]
    cost_usd: float
    error: str | None = None
    accepted_warns: list[str] = field(
        default_factory=list
    )  # known-deferred WARNs that did NOT fail the trial
    topology: str = ""  # emitted archetypes + joint types (for diagnosing dispatch)


@dataclass
class PromptResult:
    prompt_id: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.trials)

    @property
    def passed(self) -> int:
        return sum(1 for t in self.trials if t.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.n if self.n else 0.0

    @property
    def wilson_lb(self) -> float:
        return wilson_lower_bound(self.passed, self.n)

    @property
    def cost_usd(self) -> float:
        return sum(t.cost_usd for t in self.trials)


@dataclass
class BenchResult:
    prompts: list[PromptResult] = field(default_factory=list)
    stopped_on_budget: bool = False

    @property
    def n(self) -> int:
        return sum(p.n for p in self.prompts)

    @property
    def passed(self) -> int:
        return sum(p.passed for p in self.prompts)

    @property
    def cost_usd(self) -> float:
        return sum(p.cost_usd for p in self.prompts)

    @property
    def wilson_lb(self) -> float:
        return wilson_lower_bound(self.passed, self.n)


def run_one_trial(
    generator: AssemblyGraphGenerator,
    prompt_id: str,
    prompt: str,
    out_dir: Path,
    *,
    required_checks: list[str] | None = None,
    accepted_warns: list[str] | None = None,
) -> TrialResult:
    """One end-to-end trial. Failures are captured and classified, never raised.

    ``accepted_warns`` lists check ids whose WARN is a KNOWN-DEFERRED non-failure for
    this bench (e.g. ``assembly.range_of_motion`` for a static-actuated joint whose
    swept-collision needs a not-yet-built motor mesh). Any OTHER warn still fails the
    trial; ``required_checks`` must all be present-and-PASS. Default empty -> identical
    to the old ``report.passed`` gate for an all-PASS report, and any warn still fails.
    """
    try:
        graph, usage = generator(prompt)
        cost = usage.cost_usd
    except Exception as exc:
        return TrialResult(prompt_id, False, "emit", [], 0.0, f"{type(exc).__name__}: {exc}")
    # Record the emitted topology (archetypes + FULLY-wired joints incl. endpoints) so a
    # failing trial is diagnosable without a re-run -- WHAT the LLM wired, not just which
    # check failed. Endpoints matter: a mis-wired revolute (wrong a/b ports) is invisible
    # in joint-types alone.
    topology = (
        "nodes=[" + ",".join(f"{n.id}:{n.archetype}" for n in graph.nodes) + "] "
        "joints=[" + ",".join(f"{j.type.value} {j.a}->{j.b}" for j in graph.joints) + "]"
    )
    try:
        program = solve(graph)
    except Exception as exc:
        return TrialResult(
            prompt_id, False, "solve", [], cost, f"{type(exc).__name__}: {exc}", topology=topology
        )
    try:
        report = validate_assembly(compile_assembly(program, out_dir))
    except Exception as exc:
        return TrialResult(
            prompt_id, False, "compile", [], cost, f"{type(exc).__name__}: {exc}", topology=topology
        )
    accepted = accepted_warns or []
    # The accepted-warn ids that were actually warned this trial (recorded for honesty).
    warned_accepted = [
        w for w in report.warned_check_ids() if any(report._matches(w, a) for a in accepted)
    ]
    if report.passes_criteria(required_checks=required_checks, accepted_warns=accepted_warns):
        return TrialResult(
            prompt_id, True, "pass", [], cost, accepted_warns=warned_accepted, topology=topology
        )
    # On failure, name EVERY blocking reason so the failure list is never phantom-empty:
    #   - hard FAILs,
    #   - unaccepted WARNs (matched by kind, same as accepted),
    #   - required checks (by kind) with no present-and-PASS match (tagged, diagnosable).
    blocking_warns = [
        w for w in report.warned_check_ids() if not any(report._matches(w, a) for a in accepted)
    ]
    passing_ids = [c.id for c in report.checks if c.status == CheckStatus.PASS]
    missing_required = [
        f"missing_required:{rc}"
        for rc in (required_checks or [])
        if not any(report._matches(cid, rc) for cid in passing_ids)
    ]
    failed = report.failed_check_ids() + blocking_warns + missing_required
    return TrialResult(
        prompt_id,
        False,
        "validate",
        failed,
        cost,
        accepted_warns=warned_accepted,
        topology=topology,
    )


def to_detail_dict(result: BenchResult) -> dict[str, Any]:
    """Serialize a run to a per-trial detail dict (JSON-able).

    The console summary only shows stage *counts*; this keeps each trial's
    ``stage`` and ``failed_checks`` so a failing prompt can be diagnosed without
    re-running the LLM (e.g. *which* assembly check the hinge_door hinges fail).
    """
    return {
        "overall": {
            "passed": result.passed,
            "n": result.n,
            "wilson_lb": result.wilson_lb,
            "cost_usd": result.cost_usd,
            "stopped_on_budget": result.stopped_on_budget,
        },
        "prompts": [
            {
                "prompt_id": p.prompt_id,
                "passed": p.passed,
                "n": p.n,
                "wilson_lb": p.wilson_lb,
                "cost_usd": p.cost_usd,
                "trials": [
                    {
                        "passed": t.passed,
                        "stage": t.stage,
                        "failed_checks": t.failed_checks,
                        "accepted_warns": t.accepted_warns,
                        "topology": t.topology,
                        "error": t.error,
                        "cost_usd": t.cost_usd,
                    }
                    for t in p.trials
                ],
            }
            for p in result.prompts
        ],
    }


def run_assembly_benchmark(
    prompts: dict[str, str],
    n: int,
    *,
    model: str | None = None,
    max_cost_usd: float = 2.0,
    required_checks: list[str] | None = None,
    accepted_warns: list[str] | None = None,
) -> BenchResult:
    """Run ``n`` trials per prompt, honoring a hard cost ceiling.

    ``required_checks`` / ``accepted_warns`` are passed to every trial (see
    ``run_one_trial``): the honest way to score a static-actuated bench whose
    swept-collision motion check is a known-deferred WARN, without making warnings
    pass anywhere else. Default empty preserves the prior all-PASS scoring.
    """
    generator = AssemblyGraphGenerator(model=model)
    result = BenchResult()
    for prompt_id, prompt in prompts.items():
        pr = PromptResult(prompt_id=prompt_id)
        result.prompts.append(pr)
        for _ in range(n):
            if result.cost_usd >= max_cost_usd:
                result.stopped_on_budget = True
                return result
            with tempfile.TemporaryDirectory() as d:
                pr.trials.append(
                    run_one_trial(
                        generator,
                        prompt_id,
                        prompt,
                        Path(d),
                        required_checks=required_checks,
                        accepted_warns=accepted_warns,
                    )
                )
    return result


__all__ = [
    "BenchResult",
    "PromptResult",
    "TrialResult",
    "run_assembly_benchmark",
    "run_one_trial",
    "to_detail_dict",
    "wilson_lower_bound",
]
