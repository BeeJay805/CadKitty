"""Step 2 (ADR-0016): the scaling loop -- bank/flag/reject classification + budget +
stop-on-cap -- proven deterministically with an injectable fake runner ($0, no LLM)."""

from __future__ import annotations

from dataclasses import dataclass

from forge_mind.layout.composable_spec import BaseArchetype, Face, InterfaceRef
from forge_mind.scaling.engine import (
    Perturbation,
    TrialResult,
    Verdict,
    default_perturbations,
    evaluate_family,
    run_engine,
)
from forge_mind.scaling.enumerate import FamilyShape, Placement
from forge_mind.types import Vec3

_ENV = Vec3(x=400.0, y=400.0, z=40.0)
_ADAPTER = FamilyShape(
    BaseArchetype.FLAT_PLATE,
    (
        Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP),
        Placement(InterfaceRef.EXTRUSION_2020, Face.PLATE_BOTTOM),
    ),
)
_MOTOR_FAN = FamilyShape(
    BaseArchetype.FLAT_PLATE,
    (
        Placement(InterfaceRef.NEMA17_FACE, Face.PLATE_TOP),
        Placement(InterfaceRef.FAN_FACE, Face.PLATE_BOTTOM, "fan_60mm"),
    ),
)


@dataclass
class CountRunner:
    """Returns ok for the first ``pass_count`` calls, then not-ok. Deterministic."""

    pass_count: int
    cost: float = 0.01
    calls: int = 0

    def __call__(self, shape, *, perturbation, paraphrase_idx):  # type: ignore[no-untyped-def]
        self.calls += 1
        ok = self.calls <= self.pass_count
        return TrialResult(passed=ok, certified=ok, cost_usd=self.cost)

    def feasible(self, shape, *, perturbation) -> bool:  # type: ignore[no-untyped-def]
        return True


def _budget(cap: float = 1e9):
    from forge_mind.scaling.engine import Budget

    return Budget(cap_usd=cap)


def test_perfect_family_is_banked() -> None:
    # 45/45 -> Wilson-LB 0.921 >= 0.90, all perturbations pass, no fan (not near-bound).
    v = evaluate_family(
        _ADAPTER,
        CountRunner(45),
        _budget(),
        perturbations=default_perturbations(),
        n_paraphrases=15,
    )
    assert v.verdict is Verdict.BANKED, v
    assert v.n == 45 and v.passed == 45


def test_low_pass_rate_is_rejected() -> None:
    perts = [Perturbation(4.0, _ENV, "a"), Perturbation(4.0, _ENV, "b")]
    v = evaluate_family(_ADAPTER, CountRunner(20), _budget(), perturbations=perts, n_paraphrases=20)
    assert v.verdict is Verdict.REJECTED, v  # 20/40 -> 0.352 < gate 0.80


def test_marginal_is_flagged_not_banked() -> None:
    perts = [Perturbation(4.0, _ENV, "a"), Perturbation(4.0, _ENV, "b")]
    v = evaluate_family(_ADAPTER, CountRunner(39), _budget(), perturbations=perts, n_paraphrases=20)
    assert v.verdict is Verdict.FLAGGED, v  # 39/40 -> 0.871, in [0.80, 0.90)


def test_near_bound_family_flagged_even_when_perfect() -> None:
    # A fan-next-to-motor family with a sub-2.5mm-wall perturbation sits near the ADR-0015
    # interleave bound -> FLAGGED for audit even at 45/45 (never silently banked).
    perts = [
        Perturbation(2.0, _ENV, "tiny"),
        Perturbation(4.0, _ENV, "nominal"),
        Perturbation(6.0, _ENV, "thick"),
    ]
    v = evaluate_family(
        _MOTOR_FAN, CountRunner(45), _budget(), perturbations=perts, n_paraphrases=15
    )
    assert v.verdict is Verdict.FLAGGED, v
    assert "bound" in v.reason


def test_stop_on_cap() -> None:
    shapes = [_ADAPTER] * 5
    report = run_engine(
        shapes,
        CountRunner(10_000, cost=1.0),
        cap_usd=2.5,
        perturbations=default_perturbations(),
        n_paraphrases=5,
    )
    assert report.stopped_on_cap is True
    classified = len(report.banked) + len(report.flagged) + len(report.rejected)
    assert classified < 5  # halted before testing all families
    assert report.total_cost_usd <= 3.0  # no runaway past the cap
