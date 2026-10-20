"""Continuous-sweep certification (Sol consultation round 2, P0.4, decision-independent):
turn a DISCRETE pose sweep into a certified-collision-free interval, or refuse to certify.

Sol's rule, and the reason this module exists: *a discrete frame sweep may produce a FAIL
witness, but may not produce a PASS.* The existing ``motion_sweep`` samples 19 poses per joint;
a thin collision entirely between two samples is invisible to it, so "we sampled and saw
nothing" is not evidence of "no collision exists". That gap is a false-PASS vector on the
single claim a moving assembly most needs.

The fix is a conservative bound, not more samples. Between two consecutive poses separated by
a rotation ``dtheta`` about a joint axis, no point of the moving body can travel farther than

    delta = R_max * dtheta          (chord 2*R*sin(dtheta/2) <= R*dtheta)

where ``R_max`` is the greatest distance from the joint axis to any point of the moving body.
If the measured minimum separation at BOTH endpoint poses exceeds ``delta``, then separation
cannot reach zero anywhere in between: the interval is CERTIFIED collision-free by inequality.
Where the bound fails, the interval is refused (``not_run``) with the exact sub-interval named,
so a caller can subdivide adaptively and re-certify -- never silently pass.

Standalone by design: it consumes measurements (clearances + kinematics) rather than driving
the kernel, so it is unit-testable with hand-authored numbers and wires into ``motion_sweep``
later, under the held intent-gate reclassification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise

from forge_mind.validator.feature_geom import finite_nonneg, finite_positive
from forge_mind.validator.symmetry import GeomVerdict


@dataclass(frozen=True)
class PoseSample:
    """One sampled pose of a swept joint: the joint coordinate and the MEASURED minimum
    separation between the moving group and everything it must not touch.

    ``q`` is degrees for a revolute joint, mm for a prismatic one. ``min_separation_mm`` is the
    measured minimum distance at that pose (0.0 = touching, negative = interpenetrating). None
    means the separation could not be measured at this pose -- which makes the whole interval
    uncertifiable (fail closed), never a PASS.
    """

    q: float
    min_separation_mm: float | None


@dataclass(frozen=True)
class IntervalVerdict:
    """Per-interval certification detail, so a caller can subdivide exactly where needed."""

    q_lo: float
    q_hi: float
    max_travel_mm: float
    min_separation_mm: float | None
    certified: bool
    reason: str = ""


@dataclass(frozen=True)
class SweepCertificate:
    """The whole-sweep result: certified only when EVERY interval is certified."""

    verdict: GeomVerdict
    intervals: list[IntervalVerdict] = field(default_factory=list)

    @property
    def certified(self) -> bool:
        return self.verdict.status == "pass"


def max_travel_revolute_mm(radius_max_mm: float, dtheta_deg: float) -> float:
    """Upper bound on the distance any point of the moving body travels over a rotation of
    ``dtheta_deg`` about the joint axis: ``R_max * dtheta`` (>= the exact chord
    ``2*R*sin(dtheta/2)``, so the bound is conservative for every angle)."""
    return abs(radius_max_mm) * math.radians(abs(dtheta_deg))


def certify_revolute_sweep(
    samples: list[PoseSample],
    *,
    radius_max_mm: float,
    clearance_floor_mm: float = 0.0,
) -> SweepCertificate:
    """Certify a revolute sweep collision-free over the sampled span, or refuse.

    PASS requires, for EVERY consecutive pair: both endpoint separations are measured, both
    exceed ``clearance_floor_mm``, and the smaller of them exceeds the maximum possible travel
    ``R_max * dq`` between them (so separation cannot cross the floor in between).

    FAIL when any sampled pose is already at/inside the floor -- a measured witness.
    NOT_RUN when an endpoint separation is unmeasured, an input is malformed, or the travel
    bound does not clear the floor on some interval (uncertifiable: subdivide and retry). A
    refusal names the offending sub-interval.

    SCOPE (Sol): this bounds ONE moving body about ONE joint against STATIC surroundings. For a
    kinematic chain where several joints move at once, or where both members of a pair move,
    use :func:`certify_pair_sweep` -- a single-joint bound under-estimates a chain's travel and
    would certify intervals that can actually collide.
    """
    check_id = "motion.sweep_certified"
    cat = "assembly"
    limit = {"clearance_floor_mm": clearance_floor_mm, "radius_max_mm": radius_max_mm}

    if not finite_positive(radius_max_mm):
        return SweepCertificate(
            GeomVerdict(
                check_id, cat, "not_run", f"invalid radius_max_mm {radius_max_mm!r}", {}, limit
            )
        )
    if not finite_nonneg(clearance_floor_mm):
        return SweepCertificate(
            GeomVerdict(
                check_id,
                cat,
                "not_run",
                f"invalid clearance_floor_mm {clearance_floor_mm!r}",
                {},
                limit,
            )
        )
    if len(samples) < 2:
        return SweepCertificate(
            GeomVerdict(
                check_id,
                cat,
                "not_run",
                "need at least two sampled poses to certify an interval",
                {},
                limit,
            )
        )
    for s in samples:
        if not isinstance(s.q, (int, float)) or not math.isfinite(float(s.q)):
            return SweepCertificate(
                GeomVerdict(
                    check_id, cat, "not_run", f"malformed pose coordinate {s.q!r}", {}, limit
                )
            )
    # A measured contact/penetration at any sampled pose is a real FAIL witness -- reported
    # before ANY refusal, including the non-finite one below: a witness outranks an
    # uncertifiable measurement, so a NaN elsewhere must never mask a proven penetration.
    for s in samples:
        if (
            s.min_separation_mm is not None
            and math.isfinite(float(s.min_separation_mm))
            and s.min_separation_mm <= clearance_floor_mm
        ):
            return SweepCertificate(
                GeomVerdict(
                    check_id,
                    cat,
                    "fail",
                    f"collision witness at q={s.q:g}: min separation {s.min_separation_mm:.3f} mm "
                    f"<= floor {clearance_floor_mm} mm",
                    {"q": float(s.q), "min_separation_mm": float(s.min_separation_mm)},
                    limit,
                )
            )
    # No proven witness. A NON-FINITE separation is a FAILED MEASUREMENT, not a clearance, and
    # left unvalidated it MINTS a certification: min(30.0, nan) returns 30.0 (order-dependent!)
    # so a NaN endpoint certified, and +inf certified outright (both measured live). Refuse.
    for s in samples:
        if s.min_separation_mm is not None and not math.isfinite(float(s.min_separation_mm)):
            return SweepCertificate(
                GeomVerdict(
                    check_id,
                    cat,
                    "not_run",
                    f"non-finite separation {s.min_separation_mm!r} at q={s.q:g}: a failed "
                    "measurement can never certify an interval",
                    {"q": float(s.q)},
                    limit,
                )
            )

    ordered = sorted(samples, key=lambda s: float(s.q))
    intervals: list[IntervalVerdict] = []
    worst_margin = math.inf
    uncertified: IntervalVerdict | None = None
    for a, b in pairwise(ordered):
        dq = float(b.q) - float(a.q)
        travel = max_travel_revolute_mm(radius_max_mm, dq)
        if a.min_separation_mm is None or b.min_separation_mm is None:
            iv = IntervalVerdict(
                float(a.q), float(b.q), travel, None, False, "endpoint separation unmeasured"
            )
        else:
            sep = min(float(a.min_separation_mm), float(b.min_separation_mm))
            # Certified iff the closest either endpoint comes to the floor still exceeds the
            # farthest anything can move in between.
            margin = (sep - clearance_floor_mm) - travel
            worst_margin = min(worst_margin, margin)
            iv = IntervalVerdict(
                float(a.q),
                float(b.q),
                travel,
                sep,
                margin > 0.0,
                ""
                if margin > 0.0
                else f"travel bound {travel:.3f} mm >= clearance {sep - clearance_floor_mm:.3f} mm",
            )
        intervals.append(iv)
        if not iv.certified and uncertified is None:
            uncertified = iv

    if uncertified is not None:
        msg = (
            f"sweep NOT certified on [{uncertified.q_lo:g}, {uncertified.q_hi:g}]: "
            f"{uncertified.reason} -- subdivide this interval and re-certify (a discrete sweep "
            "may witness a FAIL but may never certify a PASS)"
        )
        return SweepCertificate(
            GeomVerdict(
                check_id,
                cat,
                "not_run",
                msg,
                {
                    "uncertified_q_lo": uncertified.q_lo,
                    "uncertified_q_hi": uncertified.q_hi,
                    "max_travel_mm": uncertified.max_travel_mm,
                },
                limit,
            ),
            intervals,
        )

    span_lo, span_hi = float(ordered[0].q), float(ordered[-1].q)
    measured = {
        "span_lo": span_lo,
        "span_hi": span_hi,
        "n_intervals": float(len(intervals)),
        "worst_certification_margin_mm": worst_margin,
    }
    msg = (
        f"sweep CERTIFIED collision-free on [{span_lo:g}, {span_hi:g}] across {len(intervals)} "
        f"interval(s): every interval's clearance exceeds its conservative travel bound "
        f"(worst margin {worst_margin:.3f} mm, floor {clearance_floor_mm} mm)"
    )
    return SweepCertificate(GeomVerdict(check_id, cat, "pass", msg, measured, limit), intervals)


@dataclass(frozen=True)
class JointMotion:
    """One joint's contribution to a body's motion over an interval (Sol's multi-joint bound).

    ``radius_max_mm`` is the greatest distance from THIS joint's axis to any point of the body
    being bounded -- so an ancestor joint high in the chain (a shoulder) uses the radius out to
    the far end of everything it carries (the wrist), not just its own link.
    ``dq`` is the joint's travel over the interval: degrees for revolute, mm for prismatic.
    """

    name: str
    radius_max_mm: float
    dq: float
    kind: str = "revolute"  # "revolute" | "prismatic"


def chain_travel_bound_mm(motions: list[JointMotion]) -> float | None:
    """Upper bound on how far ANY point of a body can travel when SEVERAL joints move at once.

    Sol's generalization: the displacement contributions of the moving ancestors superpose, so
    the sound bound is their SUM (each revolute joint contributes ``R_j * dtheta_j`` measured
    about ITS OWN axis; a prismatic joint contributes its stroke directly). Using a single
    joint's bound on a chain UNDER-estimates travel and would certify intervals that can
    actually collide -- a false PASS.

    None when any contribution is malformed (fail closed; the caller must refuse to certify)."""
    total = 0.0
    for m in motions:
        if not isinstance(m.dq, (int, float)) or not math.isfinite(float(m.dq)):
            return None
        if m.kind == "prismatic":
            total += abs(float(m.dq))
            continue
        if m.kind != "revolute":
            return None
        if not finite_positive(m.radius_max_mm):
            return None
        total += max_travel_revolute_mm(m.radius_max_mm, float(m.dq))
    return total


def certify_pair_sweep(
    samples: list[PoseSample],
    *,
    motions_a: list[list[JointMotion]],
    motions_b: list[list[JointMotion]] | None = None,
    clearance_floor_mm: float = 0.0,
    epsilon_mm: float = 0.0,
) -> SweepCertificate:
    """Certify a collision PAIR whose BOTH members may move, over a multi-joint sweep.

    ``samples`` are the pair's measured minimum separations at each sampled configuration.
    ``motions_a[i]`` / ``motions_b[i]`` describe how body A / body B move over interval ``i``
    (between sample ``i`` and ``i+1``), so there is one entry per interval: ``len(samples) - 1``.
    ``motions_b=None`` means B is static (a fixed chassis), contributing zero travel.

    The relative bound is ``U_AB = U_A + U_B`` (Sol): each body's own multi-joint travel bound,
    summed, because in the worst case they close on each other. Certified iff
    ``min(d_i, d_i+1) - clearance_floor - epsilon > U_AB`` for EVERY interval. ``epsilon_mm``
    absorbs measurement uncertainty in the separations (0 = trust them exactly).

    Same status rules as the single-joint certificate: a finite measured separation at/inside
    the floor is a FAIL witness (checked first); a non-finite separation, a malformed motion
    description, or an uncertifiable interval is ``not_run``, never a PASS."""
    check_id = "motion.sweep_certified_pair"
    cat = "assembly"
    limit = {"clearance_floor_mm": clearance_floor_mm, "epsilon_mm": epsilon_mm}

    if not finite_nonneg(clearance_floor_mm) or not finite_nonneg(epsilon_mm):
        return SweepCertificate(
            GeomVerdict(check_id, cat, "not_run", "invalid clearance floor / epsilon", {}, limit)
        )
    if len(samples) < 2:
        return SweepCertificate(
            GeomVerdict(
                check_id, cat, "not_run", "need at least two sampled configurations", {}, limit
            )
        )
    n_intervals = len(samples) - 1
    if len(motions_a) != n_intervals or (motions_b is not None and len(motions_b) != n_intervals):
        return SweepCertificate(
            GeomVerdict(
                check_id,
                cat,
                "not_run",
                f"motion description covers {len(motions_a)} interval(s) but there are "
                f"{n_intervals} -- every interval must state how the bodies move",
                {},
                limit,
            )
        )
    # Witness first (a proven contact outranks any refusal), then non-finite refusal.
    for s in samples:
        if (
            s.min_separation_mm is not None
            and math.isfinite(float(s.min_separation_mm))
            and s.min_separation_mm <= clearance_floor_mm
        ):
            return SweepCertificate(
                GeomVerdict(
                    check_id,
                    cat,
                    "fail",
                    f"collision witness at sample q={s.q:g}: separation "
                    f"{s.min_separation_mm:.3f} mm <= floor {clearance_floor_mm} mm",
                    {"q": float(s.q), "min_separation_mm": float(s.min_separation_mm)},
                    limit,
                )
            )
    for s in samples:
        if s.min_separation_mm is None or not math.isfinite(float(s.min_separation_mm)):
            return SweepCertificate(
                GeomVerdict(
                    check_id,
                    cat,
                    "not_run",
                    f"unmeasured/non-finite separation at q={s.q:g}: cannot certify",
                    {"q": float(s.q)},
                    limit,
                )
            )

    intervals: list[IntervalVerdict] = []
    worst = math.inf
    uncertified: IntervalVerdict | None = None
    for i in range(n_intervals):
        ua = chain_travel_bound_mm(motions_a[i])
        ub = chain_travel_bound_mm(motions_b[i]) if motions_b is not None else 0.0
        a, b = samples[i], samples[i + 1]
        sep = min(float(a.min_separation_mm or 0.0), float(b.min_separation_mm or 0.0))
        if ua is None or ub is None:
            iv = IntervalVerdict(
                float(a.q), float(b.q), math.inf, sep, False, "malformed joint motion description"
            )
        else:
            travel = ua + ub  # U_AB = U_A + U_B (both may close on each other)
            margin = (sep - clearance_floor_mm - epsilon_mm) - travel
            worst = min(worst, margin)
            iv = IntervalVerdict(
                float(a.q),
                float(b.q),
                travel,
                sep,
                margin > 0.0,
                ""
                if margin > 0.0
                else f"relative travel bound {travel:.3f} mm >= usable clearance "
                f"{sep - clearance_floor_mm - epsilon_mm:.3f} mm",
            )
        intervals.append(iv)
        if not iv.certified and uncertified is None:
            uncertified = iv

    if uncertified is not None:
        return SweepCertificate(
            GeomVerdict(
                check_id,
                cat,
                "not_run",
                f"pair sweep NOT certified on [{uncertified.q_lo:g}, {uncertified.q_hi:g}]: "
                f"{uncertified.reason} -- subdivide and re-certify",
                {
                    "uncertified_q_lo": uncertified.q_lo,
                    "uncertified_q_hi": uncertified.q_hi,
                    "relative_travel_mm": uncertified.max_travel_mm,
                },
                limit,
            ),
            intervals,
        )
    measured = {
        "n_intervals": float(n_intervals),
        "worst_certification_margin_mm": worst,
    }
    msg = (
        f"pair sweep CERTIFIED collision-free across {n_intervals} interval(s): every interval's "
        f"usable clearance exceeds the summed relative travel bound (worst margin {worst:.3f} mm)"
    )
    return SweepCertificate(GeomVerdict(check_id, cat, "pass", msg, measured, limit), intervals)


def required_sample_step_deg(
    radius_max_mm: float, min_expected_clearance_mm: float, clearance_floor_mm: float = 0.0
) -> float | None:
    """The coarsest sampling step (deg) that COULD certify, given the tightest clearance the
    sweep is expected to reach: ``dq < (clearance - floor) / R_max``. Planning aid for callers
    choosing a sample count; None when the inputs cannot yield a positive step."""
    if not finite_positive(radius_max_mm):
        return None
    slack = float(min_expected_clearance_mm) - float(clearance_floor_mm)
    if not math.isfinite(slack) or slack <= 0.0:
        return None
    return math.degrees(slack / radius_max_mm)


__all__ = [
    "IntervalVerdict",
    "JointMotion",
    "PoseSample",
    "SweepCertificate",
    "certify_pair_sweep",
    "certify_revolute_sweep",
    "chain_travel_bound_mm",
    "max_travel_revolute_mm",
    "required_sample_step_deg",
]
