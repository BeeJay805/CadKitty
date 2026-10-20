"""Hardware conformance check (A4) — the no-false-PASS gate for registry-backed features.

A part (or, soon, the composition engine) declares a ``HardwareFeature``: "I built a <category>
for <component key>[/<fit>], with this <dimension> mm." The check resolves the canonical dimension
from the SAME standards-registry API the placement helpers use (so check and build cannot drift)
and verdicts CONSERVATIVELY:

  - built dimension matches the registry (within tol)  -> PASS
  - built dimension does NOT match                      -> FAIL (non-conformant)
  - category or key NOT in the registry                 -> WARN (flagged; NEVER a silent pass)

A WARN makes the overall report non-PASS (so an unrecognized feature can never false-PASS). A
match against a PROVISIONAL spec still PASSES, but the message says so -- the provenance is
surfaced, not hidden. This check is additive: it runs only on declared HardwareFeatures, so no
existing part's validation changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.standards import (
    ALL_REGISTRIES,
    Fit,
    bearing,
    bearing_housing_bore,
    captive_nut_pocket,
    clearance_hole,
    counterbore,
    dowel_hole,
    heat_set_insert,
    magnet_pocket,
    tap_drill,
    thread_nominal,
)
from forge_mind.validator.report import Check, CheckStatus, Tier

CONFORMANCE_TOL_MM = 0.05  # built dimension vs the registry canonical value


class HardwareFeature(BaseModel):
    """A declared registry-backed hardware feature on a part, with the dimension as built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: str = Field(
        ..., description="e.g. 'clearance_hole', 'bearing_seat', 'heat_set_insert'."
    )
    key: str = Field(..., description="Component key: 'M3', '608', ...")
    nominal_mm: float = Field(..., description="The dimension as built (the value to verify).")
    fit: str | None = Field(
        default=None, description="Fit class where applicable (clearance / press / slip)."
    )
    where: str = Field(default="", description="Label for the report.")


def _press_slip(fit: str | None) -> Literal["press", "slip"]:
    return cast("Literal['press', 'slip']", fit if fit in ("press", "slip") else "press")


# category -> (provenance registry name, resolver(key, fit) -> canonical dimension mm). The
# resolvers call the SAME typed API the placement helpers use, so conformance == construction.
_CATEGORIES: dict[str, tuple[str, Callable[[str, str | None], float]]] = {
    "clearance_hole": ("metric_fastener", lambda k, f: clearance_hole(k, f or Fit.NORMAL)),
    "tapped_hole": ("metric_fastener", lambda k, _f: tap_drill(k)),
    # ADR-0081: a threaded / set-screw hole is identified by its NOMINAL major diameter.
    # This category exists so thread_hole_d citations resolve to the number the helper
    # actually returned (citing tap drill for a nominal call would state the wrong value).
    "threaded_hole": ("metric_fastener", lambda k, _f: thread_nominal(k)),
    "counterbore": ("metric_fastener", lambda k, _f: counterbore(k)[0]),
    "heat_set_insert": ("heat_set_insert", lambda k, _f: heat_set_insert(k).hole_diameter_mm),
    "captive_nut_pocket": ("hex_nut", lambda k, _f: captive_nut_pocket(k)[0]),
    "bearing_seat": ("bearing", lambda k, f: bearing_housing_bore(k, _press_slip(f))),
    "bearing_od": ("bearing", lambda k, _f: bearing(k).outer_od_mm),
    "dowel_hole": ("dowel_pin", lambda k, f: dowel_hole(k, _press_slip(f))),
    "magnet_pocket": ("magnet", lambda k, _f: magnet_pocket(k)[0]),
}


def _flag(feature: HardwareFeature, message: str) -> Check:
    return Check(
        id=f"conformance.{feature.category}",
        tier=Tier.INTERFACES,
        status=CheckStatus.WARN,
        message=message,
        expected="a registry-backed hardware spec",
        actual=f"{feature.key!r} ({feature.where})",
    )


def check_conformance(feature: HardwareFeature) -> Check:
    """Verdict one declared hardware feature against the registry (conservative; never false-PASS)."""
    entry = _CATEGORIES.get(feature.category)
    if entry is None:
        return _flag(
            feature,
            f"unrecognized hardware category {feature.category!r} -- no registry spec; FLAGGED",
        )
    reg_name, resolver = entry
    try:
        expected = resolver(feature.key, feature.fit)
        spec = ALL_REGISTRIES[reg_name].get(feature.key)
    except (ValueError, KeyError):
        return _flag(
            feature, f"unknown {feature.category} {feature.key!r} -- not in the registry; FLAGGED"
        )

    ok = abs(feature.nominal_mm - expected) <= CONFORMANCE_TOL_MM
    note = " [PROVISIONAL source]" if spec.provisional else ""
    return Check(
        id=f"conformance.{feature.category}.{feature.key}",
        tier=Tier.INTERFACES,
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=(
            f"{feature.category} {feature.key} conforms to the registry{note}."
            if ok
            else f"{feature.category} {feature.key} is NON-CONFORMANT: built {feature.nominal_mm} mm, "
            f"registry says {expected} mm.{note}"
        ),
        expected=f"{expected:.3f} mm ({spec.source})",
        actual=f"{feature.nominal_mm:.3f} mm{(' @ ' + feature.where) if feature.where else ''}",
    )


def check_conformance_all(features: list[HardwareFeature]) -> list[Check]:
    return [check_conformance(f) for f in features]


# ---------------------------------------------------------------------------------------------
# Declared-vs-MEASURED join (ADR-0035 B1, advisory-first). check_conformance above verdicts the
# DECLARATION against the registry; it never looks at the geometry, so a hole cut at the wrong
# size -- or never cut at all -- with a correct declaration PASSES. This join measures the
# compiled B-rep: for each declared round feature it looks for a CONCAVE cylinder stack of the
# declared radius. Everything here emits CheckStatus.INFO (gates nothing, by construction);
# promotion to FAIL is behind the pre-registered ADR-0035 criterion.
# ---------------------------------------------------------------------------------------------

# Categories whose verified dimension is the DIAMETER of a round cut. Box-shaped pockets
# (captive_nut_pocket) are not measurable as cylinders and are deliberately absent -- absence
# means "no measured check emitted", never a fabricated verdict.
_ROUND_CATEGORIES = frozenset(
    {
        "clearance_hole",
        "tapped_hole",
        "counterbore",
        "heat_set_insert",
        "bearing_seat",
        "bearing_od",
        "dowel_hole",
        "magnet_pocket",
    }
)

MEASURED_TOL_MM = 0.05  # declared diameter vs the measured B-rep cylinder diameter

# Categories that are THROUGH features BY DEFINITION -- a bolt must pass, so a blind pocket of
# the right radius must NOT satisfy them (ADR-0082, closing the ADR-0054 through-ness escape:
# the measured-join matched on radius + concavity only, so a correct-radius blind pocket read
# MEASURED-TRUE for a declared clearance hole). Tapped holes, counterbores, seats, and pockets
# are legitimately blind and are deliberately NOT listed.
_THROUGH_CATEGORIES = frozenset({"clearance_hole"})


def check_measured_holes(part: Any, features: list[HardwareFeature]) -> list[Check]:
    """Measured-evidence join: does the compiled body actually CONTAIN each declared round
    feature? One INFO check per declared (category, key, nominal) group:

      - MEASURED-TRUE     -- >= the declared count of concave stacks at the declared radius
      - MEASURED-MISSING  -- fewer matching concave stacks than declared instances
      - UNMEASURABLE      -- no B-rep / concavity could not be determined (stated, never guessed)

    Stacks are consumed greedily across groups so one physical hole can't satisfy two declared
    features. Convex stacks (bosses/pins) never match -- a pin where a hole should be reads
    MEASURED-MISSING, the audit's flagship false-PASS."""
    from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

    rounds = [f for f in features if f.category in _ROUND_CATEGORIES]
    if not rounds:
        return []

    def _info(check_id: str, message: str, expected: str, actual: str) -> Check:
        return Check(
            id=check_id,
            tier=Tier.INTERFACES,
            status=CheckStatus.INFO,
            message=message,
            expected=expected,
            actual=actual,
        )

    # Group declared instances: composed parts emit one HardwareFeature per placement.
    grouped: dict[tuple[str, str, float], int] = {}
    for f in rounds:
        key = (f.category, f.key, f.nominal_mm)
        grouped[key] = grouped.get(key, 0) + 1

    if part is None:
        return [
            _info(
                f"conformance.measured.{cat}.{k}",
                f"{cat} {k}: UNMEASURABLE -- no B-rep available to measure against.",
                f"{n} concave bore(s) of {nom:.2f} mm",
                "unmeasurable",
            )
            for (cat, k, nom), n in grouped.items()
        ]

    try:
        stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is not False]
    except Exception:
        return [
            _info(
                f"conformance.measured.{cat}.{k}",
                f"{cat} {k}: UNMEASURABLE -- cylinder extraction failed.",
                f"{n} concave bore(s) of {nom:.2f} mm",
                "unmeasurable",
            )
            for (cat, k, nom), n in grouped.items()
        ]

    from forge_mind.validator.geometry import stack_pierces

    consumed = [False] * len(stacks)
    checks: list[Check] = []
    for (cat, k, nom), n_declared in sorted(grouped.items()):
        expected_r = nom / 2.0
        matched = 0
        unknown = 0
        for i, s in enumerate(stacks):
            if consumed[i] or abs(s.radius - expected_r) > MEASURED_TOL_MM / 2.0 + 1e-9:
                continue
            if s.concave is None:
                unknown += 1
                continue
            # Through-ness (ADR-0082): a category that must PIERCE rejects a measured-blind
            # stack -- the pocket stays unconsumed (a declared blind feature may still claim
            # it). Unmeasurable through-ness (None) matches as before: fail-open, the join
            # never manufactures a MISSING from a measurement it could not make.
            if cat in _THROUGH_CATEGORIES and stack_pierces(part, s) is False:
                continue
            consumed[i] = True
            matched += 1
            if matched >= n_declared:
                break
        check_id = f"conformance.measured.{cat}.{k}"
        expected = f"{n_declared} concave bore(s) of {nom:.2f} mm"
        if matched >= n_declared:
            checks.append(
                _info(
                    check_id,
                    f"{cat} {k}: MEASURED-TRUE -- the body contains {matched} concave "
                    f"bore(s) at {nom:.2f} mm as declared.",
                    expected,
                    f"{matched} measured",
                )
            )
        elif unknown > 0:
            checks.append(
                _info(
                    check_id,
                    f"{cat} {k}: UNMEASURABLE -- {matched} measured, {unknown} candidate "
                    "face(s) had undeterminable concavity.",
                    expected,
                    f"{matched} measured + {unknown} unmeasurable",
                )
            )
        else:
            checks.append(
                _info(
                    check_id,
                    f"{cat} {k}: MEASURED-MISSING -- declared {n_declared} but the body "
                    f"contains only {matched} concave bore(s) at {nom:.2f} mm. The declaration "
                    "conforms to the registry, but the geometry does not contain the feature.",
                    expected,
                    f"{matched} measured",
                )
            )
    return checks


__all__ = [
    "CONFORMANCE_TOL_MM",
    "MEASURED_TOL_MM",
    "HardwareFeature",
    "check_conformance",
    "check_conformance_all",
    "check_measured_holes",
]
