"""Worst-case tolerance stackup (Sol consultation P0.5, decision-independent): does a declared
fit still work when every contributor sits at its worst permitted extreme?

The H3 assembly reported 0.00 mm mate gaps that "will bind as printed" -- nominal geometry that
is only correct if nothing varies. Real parts vary: an FDM hole prints undersize, a shaft
oversize, and a datum shifts. A fit verified only at nominal is not verified.

METHOD -- ONE SIGNED INTERVAL EQUATION, evaluated by worst-case interval arithmetic. Statistical
(RSS) stacking is deliberately NOT offered: it assumes independent normal contributors and a
tolerated defect rate, neither of which a one-off printed assembly has earned.

    C = sum_i a_i * X_i           X_i in [nominal_i - minus_i, nominal_i + plus_i]

    C_low  = sum_i min(a_i * X_i.low, a_i * X_i.high)
    C_high = sum_i max(a_i * X_i.low, a_i * X_i.high)

The bore enters with a = +1 and the shaft with a = -1, so the familiar
``C_min = hole_min - shaft_max`` / ``C_max = hole_max - shaft_min`` fall out of the same equation.
EVERY OTHER CONTRIBUTOR MUST DECLARE ITS OWN SIGNED COEFFICIENT -- there is no default, because a
guessed sign is a guessed result. The equation is DIAMETRAL throughout, so a radial quantity
carries the factor 2 in its coefficient (a radial coating on the shaft is ``a = -2``: it eats
diametral clearance twice).

WHY THE FIRST DRAFT WAS WRONG (both defects are regression-pinned in the tests). It sorted extras
into an "opens the joint" list applied only to C_max and a "closes the joint" list applied only to
C_min. But a two-sided quantity is not one-sided: a +/-0.20 mm datum shift widens BOTH ends. A
hole-side shift left C_min untouched (reporting 0.200 mm where the true worst case was 0.000) and
a shaft-side one left C_max untouched (0.400 where the truth was 0.600) -- minting a PASS from
cited, realistic, well-intentioned inputs. Signed interval arithmetic has no such gap.

CORRELATION IS NO LONGER DISCOUNTED. The first draft let contributors share a
``correlation_group`` and counted only the group's largest deviation, reasoning that one physical
cause cannot take two values at once. That reasoning is wrong: one cause taking ONE value can
still move several dimensions the same way, and those effects ADD (0.20 + 0.15 = 0.35, not
max(0.20, 0.15)). The discount was unconservative in exactly the direction that mints a PASS, so
it is gone. Honest correlation needs a declared shared latent variable resolved globally across
both sides of the equation; until a caller can declare that response, independent treatment is
the conservative answer -- it may overstate the range (a false FAIL, which this program allows)
but can never understate it.

ACCEPTANCE LIMITS ARE MANDATORY. There are no permissive defaults: an absent ``requirement`` is
``not_run``, never a PASS. The first draft defaulted to ``required_min_clearance_mm=0.0``, under
which the H3 condition this module exists to catch -- worst-case clearance exactly 0.000 mm --
evaluated ``0.0 >= 0.0`` and PASSED. A fit's allowed interval is frozen-spec input, not a default.
Both endpoints must be finite: ``+inf`` is refused too, because a clearance fit can be too loose,
an interference fit can crack or refuse to assemble, and a transition fit needs both extremes
bounded.

PROVENANCE. Every contributor and the requirement itself must cite a non-blank source; an uncited
tolerance is a guess, and a guess cannot certify a fit.

OPEN -- the specification-laundering gap is NOT closed here: ``source`` is a cited string, not a
reference resolved against a trusted frozen registry with a content digest, so nothing in this
module can yet detect that the acceptance limits were authored by the same planner that authored
the CAD. Until that registry exists, a PASS certifies "the geometry satisfies THESE stated
limits", not "the limits are independently authoritative". Standalone (not wired to the live
verdict).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from forge_mind.validator.symmetry import GeomVerdict

FitClass = Literal["clearance", "transition", "interference"]
_FIT_CLASSES: tuple[str, ...] = ("clearance", "transition", "interference")


@dataclass(frozen=True)
class Contributor:
    """One uncertain dimension in the stack, with its permitted range and where that came from.

    ``nominal_mm`` is the drawing value; ``minus_mm`` / ``plus_mm`` are the permitted deviation
    magnitudes (both >= 0), so the quantity spans ``[nominal - minus, nominal + plus]``.
    ``source`` cites the tolerance's origin (a supplier datasheet, a qualified print profile, a
    standard). ``coefficient`` is the SIGNED diametral sensitivity of the clearance to this
    quantity: it is left unset on the bore and the shaft (whose roles force +1 and -1) and is
    REQUIRED on every extra contributor.
    """

    name: str
    nominal_mm: float
    minus_mm: float
    plus_mm: float
    source: str
    coefficient: float | None = None

    @property
    def low_mm(self) -> float:
        return self.nominal_mm - self.minus_mm

    @property
    def high_mm(self) -> float:
        return self.nominal_mm + self.plus_mm


@dataclass(frozen=True)
class FitRequirement:
    """The frozen acceptance interval a fit must land inside, and its cited origin.

    ``[min_clearance_mm, max_clearance_mm]`` is a SIGNED diametral interval; negative values are
    interference. The interval must agree with ``fit_class``: a clearance fit may not permit
    interference, an interference fit may not permit a gap, a transition fit must straddle zero.
    """

    min_clearance_mm: float
    max_clearance_mm: float
    fit_class: FitClass
    source: str


@dataclass(frozen=True)
class StackResult:
    """The computed worst-case clearance interval plus the verdict that judged it."""

    verdict: GeomVerdict
    c_min_mm: float | None = None
    c_max_mm: float | None = None
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.verdict.status == "pass"


def _finite(v: object) -> bool:
    """A real, usable number. Bools are not numbers here, and NaN/+-inf never gate a fit."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _bad_contributor(c: Contributor, coefficient: float) -> str | None:
    """Why this contributor cannot be used, or None when it is usable (fail closed)."""
    if not (c.name or "").strip():
        return "a contributor has no name -- it could not be located in the evidence"
    for label, v in (
        ("nominal_mm", c.nominal_mm),
        ("minus_mm", c.minus_mm),
        ("plus_mm", c.plus_mm),
    ):
        if not _finite(v):
            return f"{c.name}: {label} is not a finite number ({v!r})"
    if c.minus_mm < 0.0 or c.plus_mm < 0.0:
        return f"{c.name}: tolerance magnitudes must be >= 0 (got -{c.minus_mm}/+{c.plus_mm})"
    if not (c.source or "").strip():
        return f"{c.name}: no cited tolerance source -- an uncited tolerance cannot certify a fit"
    if not _finite(coefficient) or coefficient == 0.0:
        return f"{c.name}: coefficient must be a finite non-zero number (got {coefficient!r})"
    return None


def _bad_requirement(req: FitRequirement) -> str | None:
    """Why this acceptance interval cannot gate a fit, or None when it can (fail closed)."""
    if req.fit_class not in _FIT_CLASSES:
        return f"unknown fit_class {req.fit_class!r}"
    if not (req.source or "").strip():
        return "the acceptance interval cites no source -- an uncited limit cannot gate a fit"
    lo, hi = req.min_clearance_mm, req.max_clearance_mm
    for label, v in (("min_clearance_mm", lo), ("max_clearance_mm", hi)):
        if not _finite(v):
            return (
                f"acceptance {label} must be a finite number (got {v!r}); an unbounded limit "
                f"cannot gate a fit -- a clearance fit can be too loose and an interference fit "
                f"can crack or refuse to assemble"
            )
    if lo > hi:
        return f"acceptance interval is reversed ([{lo}, {hi}])"
    if req.fit_class == "clearance" and lo < 0.0:
        return f"a clearance fit may not permit interference (min_clearance_mm={lo})"
    if req.fit_class == "interference" and hi > 0.0:
        return f"an interference fit may not permit a gap (max_clearance_mm={hi})"
    if req.fit_class == "transition" and not (lo < 0.0 < hi):
        return f"a transition fit must straddle zero (got [{lo}, {hi}])"
    return None


def cylindrical_fit_stack(
    *,
    hole: Contributor,
    shaft: Contributor,
    extras: Sequence[Contributor] = (),
    requirement: FitRequirement | None = None,
) -> StackResult:
    """Worst-case diametral clearance of a hole/shaft fit against a frozen acceptance interval.

    The bore's role forces coefficient +1 and the shaft's forces -1; every entry in ``extras``
    must declare its own signed diametral coefficient. PASS iff the ENTIRE manufactured interval
    lies inside the requirement::

        C_low >= requirement.min_clearance_mm  AND  C_high <= requirement.max_clearance_mm

    NOT_RUN -- never a PASS -- when the requirement is absent, unbounded, reversed, uncited or
    inconsistent with its fit class; when any contributor is malformed, uncited, unsigned or
    duplicated; when a diameter can reach zero; or when a derived endpoint is not usable.
    """
    cat = "assembly"

    def refuse(msg: str, check: str, lim: dict[str, float]) -> StackResult:
        return StackResult(GeomVerdict(check, cat, "not_run", msg, {}, lim))

    if requirement is None:
        return refuse(
            "no FitRequirement supplied -- a fit's allowed interval is frozen-spec input, and an "
            "absent limit can never be a PASS",
            "tolerance.stack",
            {},
        )
    check_id = f"tolerance.stack.{requirement.fit_class}"
    limit = {
        "min_clearance_mm": requirement.min_clearance_mm,
        "max_clearance_mm": requirement.max_clearance_mm,
    }
    why = _bad_requirement(requirement)
    if why is not None:
        return refuse(why, check_id, limit)

    # The bore opens the fit and the shaft closes it; those two signs come from the roles, never
    # from the caller. An extra has no role to derive a sign from, so it must declare one.
    roles: list[tuple[Contributor, float | None]] = [(hole, 1.0), (shaft, -1.0)]
    roles.extend((extra, None) for extra in extras)
    terms: list[tuple[Contributor, float]] = []
    for c, forced in roles:
        if forced is not None and c.coefficient is not None and c.coefficient != forced:
            return refuse(
                f"{c.name}: coefficient {c.coefficient} conflicts with the {forced:+.0f} forced "
                f"by its role in the fit",
                check_id,
                limit,
            )
        coefficient = forced if forced is not None else c.coefficient
        if coefficient is None:
            return refuse(
                f"{c.name}: an extra contributor must declare its signed diametral coefficient "
                f"(+1 opens the fit, -1 closes it, -2 for a radial quantity on the shaft)",
                check_id,
                limit,
            )
        bad = _bad_contributor(c, coefficient)
        if bad is not None:
            return refuse(bad, check_id, limit)
        terms.append((c, coefficient))

    names = [c.name for c, _ in terms]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        return refuse(
            f"duplicate contributor name(s) {duplicated} -- either one physical quantity counted "
            f"twice or two quantities sharing an id; the intended multiplicity is ambiguous",
            check_id,
            limit,
        )
    for diameter in (hole, shaft):
        if diameter.low_mm <= 0.0:
            return refuse(
                f"{diameter.name}: diameter can reach {diameter.low_mm:.3f} mm, which is not a "
                f"physical diameter",
                check_id,
                limit,
            )

    # Worst-case interval arithmetic: every term contributes its own signed extremes to BOTH ends.
    c_low = 0.0
    c_high = 0.0
    measured: dict[str, float] = {}
    for c, coefficient in terms:
        ends = (coefficient * c.low_mm, coefficient * c.high_mm)
        c_low += min(ends)
        c_high += max(ends)
        measured[f"{c.name}.low_mm"] = c.low_mm
        measured[f"{c.name}.high_mm"] = c.high_mm
        measured[f"{c.name}.coefficient"] = coefficient

    if not _finite(c_low) or not _finite(c_high) or c_low > c_high:
        return refuse(
            f"derived clearance interval is not usable ([{c_low}, {c_high}])", check_id, limit
        )

    margin_low = c_low - requirement.min_clearance_mm
    margin_high = requirement.max_clearance_mm - c_high
    contributions = {
        "c_min_mm": c_low,
        "c_max_mm": c_high,
        "margin_low_mm": margin_low,
        "margin_high_mm": margin_high,
    }
    measured.update(contributions)

    # A clearance floor of exactly zero supports only the weaker claim that nothing is PREDICTED
    # to interfere; it does not certify a running fit. H3's 0.00 mm hip "binds as printed".
    if requirement.fit_class != "clearance":
        claim = f"{requirement.fit_class} fit"
    elif requirement.min_clearance_mm > 0.0:
        claim = "running fit"
    else:
        claim = "no predicted interference, NOT a running fit"
    detail = (
        f"worst-case clearance [{c_low:.3f}, {c_high:.3f}] mm vs required "
        f"[{requirement.min_clearance_mm:.3f}, {requirement.max_clearance_mm:.3f}] mm "
        f"({claim}, per {requirement.source})"
    )
    if margin_low < 0.0:
        detail += f" -- BINDS at worst case (short by {-margin_low:.3f} mm)"
    if margin_high < 0.0:
        detail += f" -- TOO LOOSE at worst case (over by {-margin_high:.3f} mm)"

    status: Literal["pass", "fail"] = "pass" if margin_low >= 0.0 and margin_high >= 0.0 else "fail"
    msg = f"fit '{hole.name}'/'{shaft.name}': {detail}"
    return StackResult(
        GeomVerdict(check_id, cat, status, msg, measured, limit), c_low, c_high, contributions
    )


__all__ = ["Contributor", "FitClass", "FitRequirement", "StackResult", "cylindrical_fit_stack"]
