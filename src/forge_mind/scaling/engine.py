"""The scaling loop (ADR-0016 Q2/Q4/Q5): robustness battery -> Wilson-LB gate ->
bank/flag, under a hard budget cap with stop-on-cap.

The trial runner is INJECTABLE (a ``TrialRunner``): the loop's classification + budget
logic is unit-tested deterministically with a fake runner ($0); the real LLM runner
(``trial.py``) wires in for the small-scale validation. A "pass" requires the part to
validate AND every declared interface to be INDEPENDENTLY certified (ADR-0015) -- the
oracle-trust half. Banking is conservative: only high-confidence, every-perturbation,
clear-of-known-bounds families are banked; everything else is FLAGGED for human/print
audit, never silently banked.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.layout.composable_spec import InterfaceRef
from forge_mind.scaling.enumerate import FamilyShape
from forge_mind.types import Vec3

# Gate (project bench standard) + the stricter bank threshold (ADR-0016 Q4).
GATE_WILSON_LB = 0.80
BANK_WILSON_LB = 0.90
# Perturbation: vary the free Blueprint parameters across a SAFE range. min_wall stays
# >= 3 mm so the localization stays clear of the ADR-0015 sub-~2.2 mm fan-interleave bound.
_BOUND_MIN_WALL_MM = 2.5


@dataclass(frozen=True)
class Perturbation:
    min_wall_mm: float
    envelope_mm: Vec3
    label: str


def default_perturbations() -> list[Perturbation]:
    """Three bound-safe size points (the bench-001/003 perturbation lesson). min_wall stays
    in [3, 5]: >= 2.5 (clear of the ADR-0015 fan-interleave bound) and <= 5 (a 60 mm fan's
    bore-to-bolt land is 5.76 mm, so 6 mm would be infeasible for it -- ADR-0011). Per-family
    feasibility is still filtered in ``evaluate_family`` so no family is ever scored on a
    size it cannot be built at."""
    return [
        Perturbation(3.0, Vec3(x=400.0, y=400.0, z=40.0), "thin"),
        Perturbation(4.0, Vec3(x=400.0, y=400.0, z=40.0), "nominal"),
        Perturbation(5.0, Vec3(x=400.0, y=400.0, z=40.0), "thick"),
    ]


@dataclass(frozen=True)
class TrialResult:
    """One battery trial outcome. ``certified`` = every declared interface independently
    certified (not just 'the plate compiled') -- the oracle-trust half."""

    passed: bool
    certified: bool
    cost_usd: float
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.passed and self.certified


class TrialRunner(Protocol):
    def __call__(
        self, shape: FamilyShape, *, perturbation: Perturbation, paraphrase_idx: int
    ) -> TrialResult: ...

    def feasible(self, shape: FamilyShape, *, perturbation: Perturbation) -> bool:
        """Can this family be BUILT at this perturbation ($0, deterministic)? An infeasible
        size is DROPPED from the battery (not scored as a failure) -- the engine tests a
        family only across the sizes it can actually be built at."""
        ...


@dataclass
class Budget:
    cap_usd: float
    spent_usd: float = 0.0

    def charge(self, cost: float) -> None:
        self.spent_usd += cost

    @property
    def exhausted(self) -> bool:
        return self.spent_usd >= self.cap_usd

    def remaining(self) -> float:
        return max(0.0, self.cap_usd - self.spent_usd)


class Verdict(StrEnum):
    BANKED = "banked"
    FLAGGED = "flagged"
    REJECTED = "rejected"


@dataclass(frozen=True)
class FamilyVerdict:
    shape_key: str
    verdict: Verdict
    wilson_lb: float
    passed: int
    n: int
    cost_usd: float
    reason: str


def _near_known_bound(shape: FamilyShape, perturbations: Sequence[Perturbation]) -> bool:
    """True if the family + its perturbation range approaches a documented oracle bound:
    a fan next to another M3 interface (the ADR-0015 radial-interleave concern) at a wall
    thin enough to interleave."""
    refs = [p.ref for p in shape.placements]
    has_fan = InterfaceRef.FAN_FACE in refs
    multi_m3 = len([r for r in refs if r in (InterfaceRef.FAN_FACE, InterfaceRef.NEMA17_FACE)]) >= 2
    if not (has_fan and multi_m3):
        return False
    return any(p.min_wall_mm < _BOUND_MIN_WALL_MM for p in perturbations)


def evaluate_family(
    shape: FamilyShape,
    runner: TrialRunner,
    budget: Budget,
    *,
    perturbations: Sequence[Perturbation],
    n_paraphrases: int = 3,
    gate: float = GATE_WILSON_LB,
    bank: float = BANK_WILSON_LB,
) -> FamilyVerdict:
    """Run the battery (paraphrase x perturbation) for one family under the budget, score
    Wilson-LB, and classify. Stops mid-battery if the budget is exhausted (an incomplete
    battery can never be banked)."""
    # Test only across the sizes the family can be BUILT at -- an infeasible perturbation
    # is dropped (not scored as a failure). Otherwise a 60mm fan would be mis-rejected for
    # failing at a 6mm wall it cannot keep its bore-to-bolt land at (ADR-0011).
    perts = [p for p in perturbations if runner.feasible(shape, perturbation=p)]
    if not perts:
        return FamilyVerdict(
            shape.key, Verdict.REJECTED, 0.0, 0, 0, 0.0, "infeasible at every perturbation point"
        )
    per_pert_ok: dict[str, bool] = {p.label: True for p in perts}
    passed = 0
    n = 0
    cost = 0.0
    stopped = False
    for pert in perts:
        for k in range(n_paraphrases):
            if budget.exhausted:
                stopped = True
                break
            t = runner(shape, perturbation=pert, paraphrase_idx=k)
            budget.charge(t.cost_usd)
            cost += t.cost_usd
            n += 1
            if t.ok:
                passed += 1
            else:
                per_pert_ok[pert.label] = False
        if stopped:
            break

    wlb = wilson_lower_bound(passed, n) if n else 0.0
    every_pert_ok = all(per_pert_ok.values())
    near_bound = _near_known_bound(shape, perts)

    if stopped:
        return FamilyVerdict(
            shape.key,
            Verdict.FLAGGED,
            wlb,
            passed,
            n,
            cost,
            "budget exhausted mid-battery -- incomplete, not banked",
        )
    if wlb < gate:
        return FamilyVerdict(
            shape.key,
            Verdict.REJECTED,
            wlb,
            passed,
            n,
            cost,
            f"Wilson-LB {wlb:.2f} < gate {gate:.2f}",
        )
    if wlb >= bank and every_pert_ok and not near_bound:
        return FamilyVerdict(
            shape.key,
            Verdict.BANKED,
            wlb,
            passed,
            n,
            cost,
            f"Wilson-LB {wlb:.2f} >= {bank:.2f}, all perturbations pass",
        )
    reasons = []
    if wlb < bank:
        reasons.append(f"Wilson-LB {wlb:.2f} in [{gate:.2f},{bank:.2f}) (marginal)")
    if not every_pert_ok:
        reasons.append("fails at >=1 perturbation (size-sensitive)")
    if near_bound:
        reasons.append("near a known oracle bound (fan-interleave)")
    return FamilyVerdict(shape.key, Verdict.FLAGGED, wlb, passed, n, cost, "; ".join(reasons))


@dataclass
class EngineReport:
    banked: list[FamilyVerdict] = field(default_factory=list)
    flagged: list[FamilyVerdict] = field(default_factory=list)
    rejected: list[FamilyVerdict] = field(default_factory=list)
    total_cost_usd: float = 0.0
    stopped_on_cap: bool = False

    def add(self, v: FamilyVerdict) -> None:
        {
            Verdict.BANKED: self.banked,
            Verdict.FLAGGED: self.flagged,
            Verdict.REJECTED: self.rejected,
        }[v.verdict].append(v)
        self.total_cost_usd += v.cost_usd


def require_print_calibration() -> None:
    """Gate the $-budget scale-run on physical-print calibration of the oracle (ADR-0016
    Q3). Raises unless ``config.SCALING_PRINT_CALIBRATED`` is set. The bottleneck is oracle
    TRUST: at scale a miscalibrated validator banks thousands of confidently-wrong passes,
    so the budget run is blocked until prints confirm validator-pass => physically-correct
    part. Building + small-scale-validating the loop is NOT gated."""
    from forge_mind.config import SCALING_PRINT_CALIBRATED

    if not SCALING_PRINT_CALIBRATED:
        raise RuntimeError(
            "scale-run BLOCKED: the oracle (validator) is not print-calibrated. The budget "
            "run amplifies any oracle blind spot ~thousands-fold, so it is gated on "
            "physical-print calibration (ADR-0016 Q3). Flip config.SCALING_PRINT_CALIBRATED "
            "to True only after a few engine-banked families are printed + confirmed. Use "
            "`forge-mind scale --validate` for the (ungated) small-scale loop validation."
        )


def run_engine(
    shapes: Sequence[FamilyShape],
    runner: TrialRunner,
    *,
    cap_usd: float,
    perturbations: Sequence[Perturbation] | None = None,
    n_paraphrases: int = 3,
) -> EngineReport:
    """Test each family under the shared budget; bank/flag/reject. Stop-on-cap: when the
    cap is reached, stop proposing new families, keep what's classified, and report."""
    perts = list(perturbations) if perturbations is not None else default_perturbations()
    budget = Budget(cap_usd)
    report = EngineReport()
    for shape in shapes:
        if budget.exhausted:
            report.stopped_on_cap = True
            break
        report.add(
            evaluate_family(shape, runner, budget, perturbations=perts, n_paraphrases=n_paraphrases)
        )
    if budget.exhausted:
        report.stopped_on_cap = True
    return report


__all__ = [
    "BANK_WILSON_LB",
    "GATE_WILSON_LB",
    "Budget",
    "EngineReport",
    "FamilyVerdict",
    "Perturbation",
    "TrialResult",
    "TrialRunner",
    "Verdict",
    "default_perturbations",
    "evaluate_family",
    "require_print_calibration",
    "run_engine",
]
