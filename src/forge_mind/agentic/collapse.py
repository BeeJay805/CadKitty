"""Envelope-collapse detection (the L102 construction-collapse class).

H5B measured a lineage class the existing restart triggers cannot see: the sandbox builds
FINE every iteration (no error_streak) and validation never passes (no _fidelity_stuck),
while the geometry is a sliver of the brief's stated envelope (chest_plate: best attempt
4x4x6mm vs 34x105x188; upper_arm_l: last-op bound volume pinned at 48.9mm^3 across all 5
iterations). Detection is a pure ratio test so the loop can route such lineages to the
EXISTING fresh-restart path instead of anchored revision.

The 0.30 cutoff is data-grounded (2026-07-27 sprint): across 67 archived built iterations
(H5B + F2), legitimate builds sit at ratio >= 0.78 (p10; median 1.00) while the known
collapse cases sit <= 0.114 — clean separation, zero in-sample false positives.

Dark until wired: nothing imports this module yet; the loop wiring + FORGE_COLLAPSE_RESTART
flag land with the W2D experiment-1 arms.
"""

from __future__ import annotations

import os

from forge_mind.agentic.brief import parse_envelope_mm

_COLLAPSE_RATIO_DEFAULT = 0.30
_COLLAPSE_STREAK = 2  # consecutive BUILT iterations below the ratio before we call it


def collapse_restart_enabled() -> bool:
    return os.environ.get("FORGE_COLLAPSE_RESTART", "").strip().lower() in ("1", "true", "yes")


def collapse_restart_note(ratio: float | None, envelope: str | None, attempt: int) -> str:
    """The MASSING STRATEGY note for a collapse restart: mechanical, derived from the
    measured ratio -- the earlier drafts were slivers, so the demand is scale-first."""
    measured = f"{ratio:.0%} of" if ratio is not None else "a tiny fraction of"
    return (
        f"FRESH RESTART {attempt}: earlier drafts BUILT but produced geometry spanning only "
        f"{measured} the stated envelope ({envelope}). The construction is fundamentally "
        "collapsed -- do NOT repeat or patch the previous massing. Start from the OVERALL "
        "envelope: block out the full stated extents FIRST, then carve the features from it."
    )


def collapse_ratio() -> float:
    try:
        return float(os.environ.get("FORGE_COLLAPSE_RATIO", str(_COLLAPSE_RATIO_DEFAULT)))
    except ValueError:
        return _COLLAPSE_RATIO_DEFAULT


def envelope_dims_mm(envelope: str | None) -> tuple[float, float, float] | None:
    """The brief's overall_envelope_mm as three sorted-desc dims, or None.

    Delegates to the canonical `parse_envelope_mm` (review-caught: the ad-hoc regex here
    missed the supported round-part form "76 mm dia x 14 mm tall", failing open on exactly
    the collapsed-round-part case this module exists to catch)."""
    dims = parse_envelope_mm(str(envelope)) if envelope else None
    if dims is None:
        return None
    a, b, c = sorted(dims, reverse=True)
    if c <= 0:
        return None
    return a, b, c


def build_ratio(
    extents: list[float] | tuple[float, ...] | None, envelope: str | None
) -> float | None:
    """min over axes of (built dim / stated dim), both sorted descending.

    None when either side is unmeasurable (no verdict from missing data — fail-open)."""
    env = envelope_dims_mm(envelope)
    if env is None or not extents:
        return None
    try:
        built = sorted((float(x) for x in extents), reverse=True)[:3]
    except (TypeError, ValueError):
        return None
    if len(built) < 3:
        return None
    return min(bu / en for bu, en in zip(built, env, strict=False))


def lineage_all_collapsed(ratio_history: list[float | None]) -> bool:
    """True when EVERY built iteration in the lineage collapsed (and at least one built).

    The restart rung uses this to decide whether the kept incumbent can be trusted: in an
    all-collapsed lineage the incumbent is itself a sliver, and keep-best/plateau state
    anchored on it would discard a recovered-envelope fresh draft that is still red on
    another gate (review-probed: 'recovered' geometry shipped as the collapsed incumbent)."""
    cut = collapse_ratio()
    built = [r for r in ratio_history if r is not None]
    return bool(built) and all(r < cut for r in built)


def envelope_collapsed(ratio_history: list[float | None]) -> bool:
    """True when the last _COLLAPSE_STREAK BUILT iterations all sit below the cutoff.

    ratio_history carries one entry per iteration (None for unbuilt/unmeasurable
    iterations, which break the streak — a crash is error_streak's business, not ours)."""
    cut = collapse_ratio()
    tail = ratio_history[-_COLLAPSE_STREAK:]
    if len(tail) < _COLLAPSE_STREAK:
        return False
    return all(r is not None and r < cut for r in tail)
