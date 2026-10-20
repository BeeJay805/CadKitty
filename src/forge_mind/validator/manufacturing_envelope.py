"""Orientation-bound manufacturing-capability binding (Sol round 3, the P0.5 <-> P0.9 seam):
does the tolerance the stackup was HANDED actually cover what this process can hold on THIS
feature at THIS print orientation?

P0.5 proves a fit survives its declared tolerances. It cannot prove the declared tolerances are
achievable -- it takes them on a cited string. That is the open half: our contributors cite a
"qualified print profile v1" that has no artifact behind it, and every call site in the engine
hardcodes ``print_orientation = (0, 0, 1)`` while an FDM bore's achievable diameter depends
strongly on whether its axis runs along the build direction or across the layers. A +/-0.15 mm
tolerance that is honest for a vertical bore is a fiction for the same bore printed horizontally.

CLAIM: every required fit-critical feature's stack interval CONTAINS the qualified manufacturing
error envelope for that feature, process and NUMERIC orientation::

    alpha = acos(|unit(R @ axis_part) . build_Z|)          axis-to-build-direction angle
    P     = [D + delta_low, D + delta_high]                what the process can actually produce
    S     = [nominal - minus, nominal + plus]              what P0.5 was told to assume
    m     = min(P.low - S.low, S.high - P.high)

    PASS iff m >= 0 for EVERY required feature (i.e. S contains P).

This check is INDEPENDENT of whether the fit itself is acceptable: it can PASS while P0.5 FAILS
(honest tolerances on a fit that still binds) and FAIL while P0.5 passes (a fit that only works
because it was handed a tolerance the printer cannot hold). An engineered verdict needs BOTH.

WHY IT IS BUILT AS ONE SLICE WITH THE PROFILE (Sol): a ``ManufacturingProfile`` on its own is not
a check -- it is another container the generator could fill with flattering numbers. The profile
contract and the validator that binds geometry to it ship together, and the profile's
applicability cells are CLOSED: no extrapolation, no nearest-neighbour fallback, no uncited
interpolation. A feature/orientation no cell covers is ``not_run``, never a PASS. Where cells
touch, their UNION is used -- widening P can only make containment harder, so the union is the
conservative reading.

THE FALSE-PASS THIS IS BUILT TO PREVENT, named in advance by Sol: a missing or unfavourable
orientation silently falling back to the vertical-axis (or otherwise best-qualified) cell. Every
path that could do that is refused instead -- a missing, non-finite, non-rigid or REFLECTED
transform, a prose-only orientation, an uncovered cell, a process state that differs from the one
the profile was qualified on, or a contributor citing a rule that does not apply to the feature
it is attached to. The metamorphic regression fixture prints one bore both ways and requires
opposite verdicts, so an ignored transform or a hardcoded build axis cannot survive.

EVERY REAL CALL REFUSES TODAY, by Sol's round-4 ruling. The profile's digests cannot be resolved
against a trusted store because no ``ProfileRegistry`` is implemented anywhere in this repo, and
without that resolution a generated part could simply arrive carrying a profile which asserts
that its own sloppy tolerances are exactly what the printer holds -- specification laundering
wearing a check's uniform. So the arithmetic below is a quarantined calculator: correct, tested,
and unable to emit a public PASS until the registry exists. Sol's round-4 verdict on the
PASS-authoritative version was "blocker", and this is the honest landing of it.

STILL OPEN after round 4 (filed, NOT closed here): the P0.3 receipt is trusted on its check_id
and status alone, so a receipt from another part -- or a hand-built one -- is replayable; nothing
proves the bound ``Contributor`` is the one P0.5 actually evaluated; the required-feature list
and the rotation are caller-authored and unauthenticated. All three need typed receipts binding
a geometry digest. Until then even a registry would not make this PASS-authoritative.

Standalone (not wired to the live verdict). Feature measurement is NOT re-derived here: a
``MeasuredFeature`` must carry the passing P0.3 ``feature.bore_at_station`` verdict that
established it, so a feature no one verified cannot be asserted into this check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from forge_mind.validator.symmetry import GeomVerdict
from forge_mind.validator.tolerance_stack import Contributor

Vec3 = tuple[float, float, float]
# BORES ONLY. "shaft" was removed after Sol round 4: the only establishing receipt this check
# accepts is feature.bore_at_station, so a shaft cell paired with a passing BORE receipt would
# have certified a shaft that nothing ever measured. Shafts return when a shaft-specific
# geometry receipt exists.
FeatureKind = Literal["bore"]

# The angle is derived through a normalise + acos, so its value carries floating-point error --
# measured at 14.999999999999996 deg for an axis constructed at exactly 15 deg. Comparing that
# against a closed cell boundary with == semantics silently picks ONE side of the boundary, and
# Sol demonstrated it picking the FLATTERING side (the tight cell passes at +0.05 mm where the
# hull with its neighbour fails by 0.15 mm). So cell selection uses a CERTIFIED interval
# [angle - eps, angle + eps] and hulls every cell it intersects. This is a NUMERICAL bound on
# our own arithmetic, not a manufacturing fudge factor -- it must never be widened to make a
# part pass.
_ANGLE_CERTAINTY_DEG = 1e-9

# The P0.3 check whose PASS is the only accepted proof that a feature is realized in the B-rep.
_ESTABLISHING_CHECK_ID = "feature.bore_at_station"

# A rotation is accepted as rigid only this close to orthonormal with a POSITIVE determinant; a
# reflection (det = -1) is refused outright because it silently mirrors the part.
_ORTHONORMAL_TOL = 1e-6


@dataclass(frozen=True)
class ProcessState:
    """The identity of the process that actually ran (or is planned). Every field participates in
    the comparison against the profile: a profile qualified at 0.20 mm layers on one machine says
    nothing about 0.30 mm layers on another."""

    machine: str
    material: str
    nozzle_mm: float
    layer_height_mm: float
    extrusion_width_mm: float
    slicer_revision: str
    dimensional_compensation: str


@dataclass(frozen=True)
class ProfileCell:
    """One CLOSED applicability cell: the signed diametral error envelope this process holds for
    a feature of this kind, within this diameter range, at this axis-to-build-direction angle.

    Ranges are inclusive. ``delta_low_mm`` / ``delta_high_mm`` are SIGNED (an FDM bore prints
    undersize, so a bore cell's low is typically negative)."""

    rule_id: str
    feature_kind: FeatureKind
    diameter_min_mm: float
    diameter_max_mm: float
    angle_min_deg: float
    angle_max_deg: float
    delta_low_mm: float
    delta_high_mm: float

    def covers(
        self, kind: FeatureKind, diameter_mm: float, angle_lo: float, angle_hi: float
    ) -> bool:
        """Does this cell intersect the CERTIFIED angle interval for this feature? Intersection,
        not point containment: at a cell boundary the true angle could lie either side, so every
        cell it might belong to must be hulled in."""
        return (
            self.feature_kind == kind
            and self.diameter_min_mm <= diameter_mm <= self.diameter_max_mm
            and self.angle_min_deg <= angle_hi
            and angle_lo <= self.angle_max_deg
        )


@dataclass(frozen=True)
class ManufacturingProfile:
    """A frozen, qualified statement of what a process can hold, identified by content.

    ``content_digest`` and ``qualification_digest`` are the hooks the trusted-registry work will
    resolve; here they must merely be present and non-blank, which is why the OPEN note below
    matters. ``qualified_on`` is the process state the cells were measured against."""

    profile_id: str
    revision: str
    content_digest: str
    qualification_digest: str
    qualified_on: ProcessState
    cells: tuple[ProfileCell, ...]


class ProfileRegistry(Protocol):
    """Resolves a profile's digests against a TRUSTED store.

    NOTHING IN THIS REPO IMPLEMENTS THIS YET, and that is deliberate: without it, the profile is
    whatever the caller handed over, and a generated part could arrive with a profile asserting
    that its own sloppy tolerances are exactly what the printer holds. Sol's round-4 ruling: this
    check may not emit PASS while the digests are unresolvable, so a call with no registry is
    ``not_run``. Every real call refuses today. Tests supply a fake to exercise the arithmetic.
    """

    def verify(self, profile: ManufacturingProfile) -> str | None:
        """None when both digests resolve and verify; otherwise why they do not."""


@dataclass(frozen=True)
class MeasuredFeature:
    """A cylindrical feature already VERIFIED in the built B-rep, with its axis in PART
    coordinates. ``established_by`` is the P0.3 verdict that realized it -- a feature without a
    passing ``feature.bore_at_station`` cannot be asserted into this check."""

    feature_id: str
    kind: FeatureKind
    diameter_mm: float
    axis_dir_part: Vec3
    established_by: GeomVerdict


@dataclass(frozen=True)
class FeatureBinding:
    """Links a verified feature to the P0.5 contributor claiming its tolerance, and to the exact
    profile rule that contributor cites. A free-form source string is not a citation."""

    feature: MeasuredFeature
    contributor: Contributor
    cited_rule_id: str


def _finite(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _bad_rotation(rotation: Sequence[Sequence[float]]) -> str | None:
    """Why this part-to-build transform cannot be trusted, or None when it is a proper rotation.

    Rejects a wrong shape, any non-finite entry, a non-orthonormal matrix, and -- deliberately --
    a REFLECTION (det = -1), which would mirror the part while looking like a valid orientation.
    """
    rows = list(rotation)
    if len(rows) != 3 or any(len(list(r)) != 3 for r in rows):
        return "part-to-build transform must be a 3x3 rotation matrix"
    m = [[float(v) for v in r] for r in rows]
    if any(not _finite(v) for r in m for v in r):
        return "part-to-build transform has a non-finite entry"
    for i in range(3):
        for j in range(3):
            dot = sum(m[k][i] * m[k][j] for k in range(3))
            if abs(dot - (1.0 if i == j else 0.0)) > _ORTHONORMAL_TOL:
                return "part-to-build transform is not rigid (columns are not orthonormal)"
    det = (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )
    if det < 0.0:
        return "part-to-build transform is a REFLECTION (det < 0) -- it would mirror the part"
    return None


def _process_mismatch(actual: ProcessState, qualified: ProcessState) -> list[str]:
    """Which fields of the running process differ from the one the profile was qualified on."""
    return [
        name
        for name in (
            "machine",
            "material",
            "nozzle_mm",
            "layer_height_mm",
            "extrusion_width_mm",
            "slicer_revision",
            "dimensional_compensation",
        )
        if getattr(actual, name) != getattr(qualified, name)
    ]


def _bad_profile(profile: ManufacturingProfile) -> str | None:
    """Why this profile cannot gate anything, or None when it can (fail closed)."""
    for label, v in (
        ("profile_id", profile.profile_id),
        ("revision", profile.revision),
        ("content_digest", profile.content_digest),
        ("qualification_digest", profile.qualification_digest),
    ):
        if not (v or "").strip():
            return f"profile {label} is blank -- an unidentified profile cannot be trusted"
    if not profile.cells:
        return "profile declares no applicability cells"
    rule_ids = [c.rule_id for c in profile.cells]
    duplicated = sorted({r for r in rule_ids if rule_ids.count(r) > 1})
    if duplicated:
        return f"profile has duplicate rule_id(s) {duplicated} -- a citation would be ambiguous"
    for c in profile.cells:
        if not (c.rule_id or "").strip():
            return "profile has a cell with a blank rule_id"
        for label, v_num in (
            ("diameter_min_mm", c.diameter_min_mm),
            ("diameter_max_mm", c.diameter_max_mm),
            ("angle_min_deg", c.angle_min_deg),
            ("angle_max_deg", c.angle_max_deg),
            ("delta_low_mm", c.delta_low_mm),
            ("delta_high_mm", c.delta_high_mm),
        ):
            if not _finite(v_num):
                return f"cell {c.rule_id}: {label} is not a finite number ({v_num!r})"
        if c.diameter_min_mm > c.diameter_max_mm or c.angle_min_deg > c.angle_max_deg:
            return f"cell {c.rule_id}: an applicability range is reversed"
        if c.delta_low_mm > c.delta_high_mm:
            return f"cell {c.rule_id}: error envelope is reversed"
    return None


def _axis_angle_deg(rotation: Sequence[Sequence[float]], axis_part: Vec3) -> float | None:
    """Angle between the feature axis IN BUILD COORDINATES and the build direction (+Z), folded
    to [0, 90] because a bore and its reverse print identically. None when unmeasurable."""
    if len(axis_part) != 3 or any(not _finite(v) for v in axis_part):
        return None
    m = [[float(v) for v in r] for r in rotation]
    built = tuple(sum(m[i][j] * float(axis_part[j]) for j in range(3)) for i in range(3))
    norm = math.sqrt(sum(v * v for v in built))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    cos_to_z = abs(built[2] / norm)
    return math.degrees(math.acos(max(0.0, min(1.0, cos_to_z))))


def feature_tolerance_envelope_binding(
    *,
    bindings: Sequence[FeatureBinding],
    required_feature_ids: Sequence[str],
    part_to_build_rotation: Sequence[Sequence[float]] | None = None,
    profile: ManufacturingProfile | None = None,
    actual_process: ProcessState | None = None,
    registry: ProfileRegistry | None = None,
) -> GeomVerdict:
    """Does every required feature's declared tolerance cover what the process can hold at its
    numeric print orientation?

    PASS iff ``min(P.low - S.low, S.high - P.high) >= 0`` for every required feature. NOT_RUN --
    never a PASS -- when the orientation is absent or untrustworthy, the profile or the running
    process cannot be reconciled, a required feature has no binding, a feature was not
    established by a passing P0.3 verdict, no cell covers a feature at its measured orientation,
    or a contributor cites a rule that does not apply to it.

    EVERY REAL CALL REFUSES TODAY: no ``ProfileRegistry`` exists in this repo, so the profile's
    digests cannot be resolved against a trusted store and the check cannot tell a qualified
    envelope from one the generator wrote to flatter itself.
    """
    check_id = "manufacturing.feature_tolerance_envelope_binding"
    cat = "printability"

    def refuse(msg: str, measured: dict[str, float] | None = None) -> GeomVerdict:
        return GeomVerdict(check_id, cat, "not_run", msg, measured or {}, {"min_margin_mm": 0.0})

    if registry is None:
        return refuse(
            "no trusted ProfileRegistry -- the profile's content and qualification digests cannot "
            "be resolved, so a caller-supplied profile could assert whatever envelope suits it. "
            "This check may not emit a PASS until the registry exists"
        )
    if profile is not None:
        unverified = registry.verify(profile)
        if unverified is not None:
            return refuse(f"profile digests did not verify: {unverified}")
    if profile is None:
        return refuse("no ManufacturingProfile supplied -- capability cannot be assumed")
    if actual_process is None:
        return refuse(
            "no ProcessState supplied -- a profile only applies to the process it was qualified on"
        )
    if part_to_build_rotation is None:
        return refuse(
            "no numeric part-to-build orientation supplied; a prose orientation note is not an "
            "orientation, and an FDM bore's achievable diameter depends on it"
        )
    why = _bad_profile(profile)
    if why is not None:
        return refuse(why)
    why = _bad_rotation(part_to_build_rotation)
    if why is not None:
        return refuse(why)
    mismatched = _process_mismatch(actual_process, profile.qualified_on)
    if mismatched:
        return refuse(
            f"the running process differs from the one profile {profile.profile_id} rev "
            f"{profile.revision} was qualified on ({', '.join(mismatched)}) -- its envelopes do "
            f"not transfer"
        )

    by_id = {b.feature.feature_id: b for b in bindings}
    ids = [b.feature.feature_id for b in bindings]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        return refuse(f"duplicate feature binding(s) {duplicated} -- the intended one is ambiguous")
    required = list(required_feature_ids)
    if not required:
        return refuse("no required features to check -- an empty gate certifies nothing")
    repeated = sorted({f for f in required if required.count(f) > 1})
    if repeated:
        return refuse(f"required feature id(s) {repeated} listed more than once -- ambiguous gate")
    missing = [fid for fid in required if fid not in by_id]
    if missing:
        return refuse(
            f"required fit-critical feature(s) {missing} have no binding -- an omitted feature "
            f"cannot be certified, and the frozen required list is what stops it being dropped"
        )
    # A binding nobody required would otherwise be silently dropped, which is the same
    # ignore-a-supplied-input pattern that hid the unused nominal on P0.5's extras.
    unrequired = sorted(set(ids) - set(required))
    if unrequired:
        return refuse(
            f"binding(s) supplied for {unrequired}, which the frozen required list does not name "
            f"-- reconcile the two rather than leaving a feature checked by nobody"
        )

    measured: dict[str, float] = {}
    worst_margin = math.inf
    worst_id = ""
    union_used = False
    for fid in required_feature_ids:
        b = by_id[fid]
        f = b.feature
        established = f.established_by
        if established.check_id != _ESTABLISHING_CHECK_ID or established.status != "pass":
            return refuse(
                f"{fid}: not established by a passing {_ESTABLISHING_CHECK_ID} verdict (got "
                f"{established.check_id!r} / {established.status!r}) -- an unverified feature "
                f"cannot be asserted into this check",
                measured,
            )
        if not _finite(f.diameter_mm) or f.diameter_mm <= 0.0:
            return refuse(f"{fid}: measured diameter {f.diameter_mm!r} is not usable", measured)
        angle = _axis_angle_deg(part_to_build_rotation, f.axis_dir_part)
        if angle is None:
            return refuse(f"{fid}: feature axis is degenerate in build coordinates", measured)

        # CERTIFIED angle interval, not a point: our own acos arithmetic is only good to a few
        # ULPs, and an exact-15-deg axis measures 14.999999999999996. A point comparison against a
        # closed boundary silently picks one side -- Sol demonstrated it picking the flattering
        # one. Every cell the true angle could belong to is hulled in instead.
        angle_lo, angle_hi = angle - _ANGLE_CERTAINTY_DEG, angle + _ANGLE_CERTAINTY_DEG
        covering = [c for c in profile.cells if c.covers(f.kind, f.diameter_mm, angle_lo, angle_hi)]
        if not covering:
            return refuse(
                f"{fid}: no qualified cell covers a {f.kind} of {f.diameter_mm:.3f} mm at "
                f"{angle:.1f} deg to the build direction -- extrapolating a nearby cell would be "
                f"the exact false-PASS this check exists to prevent",
                measured,
            )
        cited = [c for c in covering if c.rule_id == b.cited_rule_id]
        if not cited:
            return refuse(
                f"{fid}: contributor cites rule {b.cited_rule_id!r}, which does not apply at "
                f"{f.diameter_mm:.3f} mm / {angle:.1f} deg (applicable: "
                f"{[c.rule_id for c in covering]})",
                measured,
            )
        # Where cells touch, the UNION is the conservative reading: widening P can only make
        # containment harder, so a boundary can never be resolved into a flattering envelope.
        if len(covering) > 1:
            union_used = True
        delta_low = min(c.delta_low_mm for c in covering)
        delta_high = max(c.delta_high_mm for c in covering)

        c_ = b.contributor
        if not all(_finite(v) for v in (c_.nominal_mm, c_.minus_mm, c_.plus_mm)):
            return refuse(f"{fid}: bound contributor has a non-finite bound", measured)
        # Contributor bounds are validated inside cylindrical_fit_stack, not on the dataclass, so
        # one handed straight here has never been checked: an inverted interval must not reach
        # the containment math.
        if c_.minus_mm < 0.0 or c_.plus_mm < 0.0:
            return refuse(
                f"{fid}: bound contributor has a negative tolerance magnitude "
                f"(-{c_.minus_mm}/+{c_.plus_mm}), so its interval is inverted",
                measured,
            )
        p_low, p_high = f.diameter_mm + delta_low, f.diameter_mm + delta_high
        s_low, s_high = c_.low_mm, c_.high_mm
        # Sol's reconciliation condition. If the REALIZED diameter falls outside the band P0.5 was
        # handed, that contributor is not describing this feature -- and a band wide enough to
        # contain P would otherwise PASS a binding that was never coherent.
        if not (s_low <= f.diameter_mm <= s_high):
            return refuse(
                f"{fid}: measured diameter {f.diameter_mm:.3f} mm lies outside the stack interval "
                f"[{s_low:.3f}, {s_high:.3f}] mm bound to it -- the contributor does not describe "
                f"this feature",
                measured,
            )
        m_low, m_high = p_low - s_low, s_high - p_high
        # THE min(x, NaN) TRAP, and this program has now been bitten by it TWICE: finite inputs
        # can still overflow to +inf (D = 1e308 with a 1e308 envelope), inf - inf is NaN, and
        # Python's min(0.0, nan) returns 0.0 -- so a margin that could not be computed reads as a
        # PASS. sweep_certificate hit the identical bug earlier in this program and the written
        # lesson did not transfer, so it is a guard here rather than a note. DERIVED values must
        # be proven finite; validating the primitive inputs is not enough.
        if not all(_finite(v) for v in (p_low, p_high, s_low, s_high, m_low, m_high)):
            return refuse(
                f"{fid}: a derived endpoint or margin is not finite (P=[{p_low!r}, {p_high!r}], "
                f"S=[{s_low!r}, {s_high!r}]) -- an uncomputable margin is not a passing one",
                measured,
            )
        margin = min(m_low, m_high)
        measured[f"{fid}.angle_deg"] = angle
        measured[f"{fid}.diameter_mm"] = f.diameter_mm
        measured[f"{fid}.profile_low_mm"] = p_low
        measured[f"{fid}.profile_high_mm"] = p_high
        measured[f"{fid}.stack_low_mm"] = s_low
        measured[f"{fid}.stack_high_mm"] = s_high
        measured[f"{fid}.margin_mm"] = margin
        if margin < worst_margin:
            worst_margin, worst_id = margin, fid

    measured["worst_margin_mm"] = worst_margin
    measured["boundary_union_used"] = 1.0 if union_used else 0.0
    status: Literal["pass", "fail"] = "pass" if worst_margin >= 0.0 else "fail"
    verb = "covers" if status == "pass" else "does NOT cover"
    msg = (
        f"{len(required_feature_ids)} fit-critical feature(s) vs profile {profile.profile_id} rev "
        f"{profile.revision} ({profile.content_digest}): the declared tolerance {verb} the "
        f"qualified envelope; worst margin {worst_margin:+.3f} mm at '{worst_id}'"
    )
    if status == "fail":
        msg += " -- the stackup was handed a tolerance this process cannot hold at that orientation"
    if union_used:
        msg += "; a cell-boundary union was required"
    return GeomVerdict(check_id, cat, status, msg, measured, {"min_margin_mm": 0.0})


__all__ = [
    "FeatureBinding",
    "FeatureKind",
    "ManufacturingProfile",
    "MeasuredFeature",
    "ProcessState",
    "ProfileCell",
    "feature_tolerance_envelope_binding",
]
