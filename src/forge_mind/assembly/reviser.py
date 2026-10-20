"""Assembly reviser — the closing stage of Path 4 (ADR-0010).

Deterministic, validator-guided repair of a failing AssemblyGraph. It reads the
failed checks, applies an ALLOW-LISTED graph patch (loosen the fit class; shorten an
over-long travel), re-solves and re-validates, and adopts the change only if it is
STRICTLY better. It never weakens the validator -- it edits the design, not the
checks -- and stops as soon as it stops improving. This mirrors the single-part
patch-only reviser's philosophy, lifted to the assembly graph. Some defects (a
missing DOF -- a welded joint) have no allow-listed repair: the reviser stops
honestly and returns the still-failing best.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge_mind.assembly.joints import AssemblyGraph
from forge_mind.assembly.solver import solve
from forge_mind.compiler.compile_assembly import compile_assembly
from forge_mind.validator.assembly_checks import validate_assembly
from forge_mind.validator.report import ValidationReport

# Fit-class ladder: a too-tight fit is loosened one rung at a time.
_LOOSER = {"interference": "running_fit", "running_fit": "loose_fit"}


@dataclass
class RevisionResult:
    graph: AssemblyGraph
    report: ValidationReport
    applied: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def iterations(self) -> int:
        return len(self.applied)


def _validate(graph: AssemblyGraph, out_dir: Path) -> ValidationReport:
    return validate_assembly(compile_assembly(solve(graph), out_dir))


def _with_joint(graph: AssemblyGraph, idx: int, update: dict[str, object]) -> AssemblyGraph:
    joints = list(graph.joints)
    joints[idx] = joints[idx].model_copy(update=update)
    return graph.model_copy(update={"joints": joints})


def _propose(graph: AssemblyGraph, report: ValidationReport) -> tuple[AssemblyGraph, str] | None:
    """A single allow-listed patch for the worst failure, or None if unrepairable."""
    failed = set(report.failed_check_ids())

    # 1) The fit is too tight (pin/piston fatter than the bore) -> loosen the fit class.
    if "assembly.pin_clearance" in failed:
        for i, j in enumerate(graph.joints):
            looser = _LOOSER.get(j.clearance_class)
            if looser is not None:
                return _with_joint(
                    graph, i, {"clearance_class": looser}
                ), f"loosen fit {j.clearance_class}->{looser}"

    # 2) A slide that escapes its rail -> halve the declared travel.
    motion = report.check("assembly.range_of_motion")
    if (
        "assembly.range_of_motion" in failed
        and motion is not None
        and "slides out" in (motion.actual or "")
    ):
        for i, j in enumerate(graph.joints):
            if j.range_mm is not None:
                lo, hi = j.range_mm
                return _with_joint(
                    graph, i, {"range_mm": (lo, (lo + hi) / 2.0)}
                ), f"shorten travel {hi:.0f}->{(lo + hi) / 2.0:.0f}mm"

    return None  # e.g. a FIXED joint (missing DOF) -- no allow-listed repair


def revise_assembly(graph: AssemblyGraph, out_dir: Path, max_iters: int = 3) -> RevisionResult:
    """Repair a failing assembly with allow-listed patches; adopt only strict improvements."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report = _validate(graph, out_dir / "iter0")
    applied: list[str] = []
    for k in range(1, max_iters + 1):
        if report.passed:
            break
        proposal = _propose(graph, report)
        if proposal is None:
            break
        cand_graph, desc = proposal
        cand_report = _validate(cand_graph, out_dir / f"iter{k}")
        improved = cand_report.passed or len(cand_report.failed_check_ids()) < len(
            report.failed_check_ids()
        )
        if not improved:
            break  # the patch didn't help -- stop rather than thrash
        graph, report = cand_graph, cand_report
        applied.append(desc)
    return RevisionResult(graph=graph, report=report, applied=applied)


__all__ = ["RevisionResult", "revise_assembly"]
