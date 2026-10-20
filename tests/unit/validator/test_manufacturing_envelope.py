"""Orientation-bound manufacturing-capability binding (Sol round 3): is the tolerance P0.5 was
handed actually achievable on THIS feature at THIS print orientation?

The centrepiece is Sol's METAMORPHIC PAIR: one Ø14.30 bore whose part-space axis is X, printed
two ways. Standing the bore up along the build direction must PASS; leaving it horizontal must
FAIL, because the qualified envelope for a cross-layer bore is much wider than the +/-0.15 mm the
stackup was told to assume. A check that ignored the transform, or that hardcoded the build axis
the way every call site in the engine currently hardcodes ``print_orientation=(0,0,1)``, cannot
produce both verdicts.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from forge_mind.validator.manufacturing_envelope import (
    FeatureBinding,
    ManufacturingProfile,
    MeasuredFeature,
    ProcessState,
    ProfileCell,
    feature_tolerance_envelope_binding,
)
from forge_mind.validator.symmetry import GeomVerdict
from forge_mind.validator.tolerance_stack import Contributor

_PROCESS = ProcessState(
    machine="prusa_mk4",
    material="prusament_pla",
    nozzle_mm=0.4,
    layer_height_mm=0.2,
    extrusion_width_mm=0.45,
    slicer_revision="prusaslicer-2.8.1",
    dimensional_compensation="xy_hole_comp=0.00",
)

# Qualified on real FDM behaviour: a bore printed ALONG the build direction holds a tight, nearly
# symmetric envelope; the same bore printed ACROSS the layers sags and prints undersize.
_VERTICAL = ProfileCell("bore.vertical.0_15", "bore", 6.0, 25.0, 0.0, 15.0, -0.10, 0.10)
_HORIZONTAL = ProfileCell("bore.horizontal.75_90", "bore", 6.0, 25.0, 75.0, 90.0, -0.30, 0.08)

# Standing the part's X axis up along the build direction: R @ (1,0,0) = (0,0,1). Proper rotation
# (det = +1), so it is an orientation and not a mirror. DELIBERATELY the cyclic rotation rather
# than a symmetric one: under the wrong convention R.T @ X = (0,1,0) reads 90 deg, so a
# transposed implementation FAILS these tests. The earlier fixture could not tell R from R.T,
# because both mapped X to +/-Z and abs() hid the sign.
_X_UP = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class _TrustingRegistry:
    """Test double for the trusted profile store. NOTHING in the repo implements ProfileRegistry,
    so every REAL call refuses (pinned by test_a_real_call_has_no_registry_and_refuses); these
    fixtures supply this fake purely to exercise the arithmetic behind that refusal."""

    def verify(self, profile: ManufacturingProfile) -> str | None:
        return None


def _profile(*cells: ProfileCell, **kw) -> ManufacturingProfile:
    return ManufacturingProfile(
        profile_id=kw.pop("profile_id", "fdm_pla_mk4"),
        revision=kw.pop("revision", "v3"),
        content_digest=kw.pop("content_digest", "sha256:2f1c"),
        qualification_digest=kw.pop("qualification_digest", "sha256:9ab0"),
        qualified_on=kw.pop("qualified_on", _PROCESS),
        cells=cells or (_VERTICAL, _HORIZONTAL),
    )


def _established(status: str = "pass", check_id: str = "feature.bore_at_station") -> GeomVerdict:
    return GeomVerdict(check_id, "features", status, "bore realized at station", {}, {})


def _feature(axis=(1.0, 0.0, 0.0), diameter=14.30, **kw) -> MeasuredFeature:
    return MeasuredFeature(
        feature_id=kw.pop("feature_id", "hip_bore"),
        kind=kw.pop("kind", "bore"),
        diameter_mm=diameter,
        axis_dir_part=axis,
        established_by=kw.pop("established_by", _established()),
    )


def _binding(rule: str = "bore.vertical.0_15", minus=0.15, plus=0.15, **kw) -> FeatureBinding:
    return FeatureBinding(
        feature=kw.pop("feature", _feature()),
        contributor=Contributor("pelvis_hip_bore", 14.30, minus, plus, "hip ICD rev B"),
        cited_rule_id=rule,
    )


def _run(bindings, required=("hip_bore",), rotation=_X_UP, profile=None, process=None):
    return feature_tolerance_envelope_binding(
        bindings=bindings,
        required_feature_ids=required,
        part_to_build_rotation=rotation,
        profile=profile if profile is not None else _profile(),
        actual_process=process if process is not None else _PROCESS,
        registry=_TrustingRegistry(),
    )


def test_a_real_call_has_no_registry_and_refuses() -> None:
    # Sol's round-4 scope ruling: until the profile's digests resolve against a trusted store,
    # this check may not emit PASS -- a caller-supplied profile could assert whatever envelope
    # flatters the part. No ProfileRegistry exists in this repo, so the DEFAULT is refusal and
    # every production call gets it. The fixtures below pass a fake to reach the arithmetic.
    r = feature_tolerance_envelope_binding(
        bindings=[_binding()],
        required_feature_ids=("hip_bore",),
        part_to_build_rotation=_X_UP,
        profile=_profile(),
        actual_process=_PROCESS,
    )
    assert r.status == "not_run", r.message
    assert "no trusted ProfileRegistry" in r.message


def test_sols_metamorphic_pair_standing_the_bore_up_passes_leaving_it_flat_fails() -> None:
    # THE PAIR. Same Ø14.30 bore, same +/-0.15 declared tolerance, two orientations.
    # Standing up: axis || build Z -> 0 deg -> envelope [-0.10, +0.10] -> P = [14.20, 14.40],
    # inside S = [14.15, 14.45]. The declared tolerance is honest.
    up = _run([_binding()])
    assert up.status == "pass", up.message
    assert up.measured["hip_bore.angle_deg"] == pytest.approx(0.0, abs=1e-9)
    assert up.measured["hip_bore.profile_low_mm"] == pytest.approx(14.20, abs=1e-9)
    assert up.measured["hip_bore.profile_high_mm"] == pytest.approx(14.40, abs=1e-9)
    assert up.measured["worst_margin_mm"] == pytest.approx(0.05, abs=1e-9)

    # Flat: the same axis is now 90 deg to the build direction -> envelope [-0.30, +0.08] ->
    # P = [14.00, 14.38]. The bore can print 0.15 mm smaller than the stackup was ever told.
    flat = _run([_binding(rule="bore.horizontal.75_90")], rotation=_IDENTITY)
    assert flat.status == "fail", flat.message
    assert flat.measured["hip_bore.angle_deg"] == pytest.approx(90.0, abs=1e-9)
    assert flat.measured["hip_bore.profile_low_mm"] == pytest.approx(14.00, abs=1e-9)
    assert flat.measured["worst_margin_mm"] == pytest.approx(-0.15, abs=1e-9)
    assert "cannot hold at that orientation" in flat.message


def test_an_ignored_transform_cannot_fake_the_flat_verdict() -> None:
    # Guard on the guard: if the transform were ignored (or the build axis hardcoded), the flat
    # arm would measure 0 deg, select the vertical cell, and find the cited horizontal rule
    # inapplicable -> not_run. It can never come back PASS, and it can never come back FAIL by
    # accident either, so the pair above genuinely discriminates.
    r = _run([_binding(rule="bore.horizontal.75_90")], rotation=_X_UP)
    assert r.status == "not_run" and "does not apply" in r.message


def test_a_reflected_transform_is_refused() -> None:
    # det = -1 mirrors the part while looking like a valid orientation.
    mirror = ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    r = _run([_binding()], rotation=mirror)
    assert r.status == "not_run" and "REFLECTION" in r.message


def test_a_non_rigid_transform_is_refused() -> None:
    scaled = ((2.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 2.0))
    r = _run([_binding()], rotation=scaled)
    assert r.status == "not_run" and "not rigid" in r.message
    r = _run([_binding()], rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, float("nan"))))
    assert r.status == "not_run" and "non-finite" in r.message


def test_a_prose_only_orientation_is_not_run() -> None:
    # The engine today carries an LLM "print_orientation_note" string and nothing numeric.
    r = _run([_binding()], rotation=None)
    assert r.status == "not_run" and "prose orientation note is not an orientation" in r.message


def test_an_orientation_no_cell_covers_is_not_run_never_extrapolated() -> None:
    # 45 deg falls in the gap between the qualified 0-15 and 75-90 cells. Reaching for the
    # nearest cell is exactly the false-PASS Sol named in advance.
    diagonal = (math.sin(math.radians(45.0)), 0.0, math.cos(math.radians(45.0)))
    r = _run([_binding(feature=_feature(axis=diagonal))], rotation=_IDENTITY)
    assert r.status == "not_run", r.message
    assert "no qualified cell covers" in r.message and "45.0 deg" in r.message


def test_touching_cells_resolve_to_their_conservative_union() -> None:
    # Where cells overlap, the union must be used: widening P can only make containment harder,
    # so a boundary can never be resolved into the more flattering envelope. Alone, the tight
    # cell would PASS (P = [14.20, 14.40]); the union [-0.20, +0.10] gives P = [14.10, 14.40],
    # which S = [14.15, 14.45] does not contain.
    tight = ProfileCell("bore.tight.0_20", "bore", 6.0, 25.0, 0.0, 20.0, -0.10, 0.10)
    loose = ProfileCell("bore.mid.15_30", "bore", 6.0, 25.0, 15.0, 30.0, -0.20, 0.05)
    tilted = (math.sin(math.radians(17.0)), 0.0, math.cos(math.radians(17.0)))
    r = _run(
        [_binding(rule="bore.tight.0_20", feature=_feature(axis=tilted))],
        rotation=_IDENTITY,
        profile=_profile(tight, loose),
    )
    assert r.status == "fail", r.message
    assert r.measured["boundary_union_used"] == 1.0
    assert r.measured["hip_bore.profile_low_mm"] == pytest.approx(14.10, abs=1e-9)
    assert r.measured["hip_bore.profile_high_mm"] == pytest.approx(14.40, abs=1e-9)
    assert "boundary union was required" in r.message


def test_a_process_that_differs_from_the_qualified_one_is_not_run() -> None:
    # A profile qualified at 0.20 mm layers says nothing about 0.30 mm layers.
    other = ProcessState(
        machine="prusa_mk4",
        material="prusament_pla",
        nozzle_mm=0.4,
        layer_height_mm=0.3,
        extrusion_width_mm=0.45,
        slicer_revision="prusaslicer-2.8.1",
        dimensional_compensation="xy_hole_comp=0.00",
    )
    r = _run([_binding()], process=other)
    assert r.status == "not_run", r.message
    assert "layer_height_mm" in r.message and "do not transfer" in r.message


def test_an_omitted_required_feature_is_not_run() -> None:
    # The frozen required list is what stops the generator dropping an inconvenient feature.
    r = _run([_binding()], required=("hip_bore", "knee_bore"))
    assert r.status == "not_run", r.message
    assert "knee_bore" in r.message and "no binding" in r.message


def test_a_feature_not_established_by_p03_is_not_run() -> None:
    # Feature measurement is not re-derived here, so the P0.3 PASS is the only accepted proof
    # that the bore exists where it is claimed to.
    for verdict in (
        _established(status="fail"),
        _established(status="not_run"),
        _established(check_id="feature.something_else"),
    ):
        r = _run([_binding(feature=_feature(established_by=verdict))])
        assert r.status == "not_run", verdict
        assert "not established by a passing" in r.message


def test_an_unusable_profile_is_not_run() -> None:
    blank = _profile(content_digest="   ")
    assert _run([_binding()], profile=blank).status == "not_run"
    dupes = _profile(
        _VERTICAL, ProfileCell("bore.vertical.0_15", "bore", 6.0, 25.0, 0.0, 15.0, -0.5, 0.5)
    )
    r = _run([_binding()], profile=dupes)
    assert r.status == "not_run" and "duplicate rule_id" in r.message
    reversed_env = _profile(ProfileCell("bore.bad", "bore", 6.0, 25.0, 0.0, 15.0, 0.10, -0.10))
    r = _run([_binding(rule="bore.bad")], profile=reversed_env)
    assert r.status == "not_run" and "envelope is reversed" in r.message


def test_missing_profile_or_process_is_not_run() -> None:
    # Registry supplied so these reach the profile/process guards behind it.
    r = feature_tolerance_envelope_binding(
        bindings=[_binding()],
        required_feature_ids=("hip_bore",),
        part_to_build_rotation=_X_UP,
        profile=None,
        actual_process=_PROCESS,
        registry=_TrustingRegistry(),
    )
    assert r.status == "not_run" and "capability cannot be assumed" in r.message
    r = feature_tolerance_envelope_binding(
        bindings=[_binding()],
        required_feature_ids=("hip_bore",),
        part_to_build_rotation=_X_UP,
        profile=_profile(),
        actual_process=None,
        registry=_TrustingRegistry(),
    )
    assert r.status == "not_run" and "qualified on" in r.message


def test_an_empty_required_list_certifies_nothing() -> None:
    r = _run([_binding()], required=())
    assert r.status == "not_run" and "empty gate certifies nothing" in r.message


def test_duplicate_bindings_are_not_run() -> None:
    r = _run([_binding(), _binding()])
    assert r.status == "not_run" and "duplicate feature binding" in r.message


def test_an_inverted_contributor_interval_is_not_run() -> None:
    # Contributor bounds are validated inside cylindrical_fit_stack, not on the dataclass, so one
    # handed straight to this check has never been checked at all.
    r = _run([_binding(minus=-0.10)])
    assert r.status == "not_run", r.message
    assert "negative tolerance magnitude" in r.message


def test_a_measured_diameter_outside_its_bound_interval_is_not_run() -> None:
    # The contrived-but-real false-PASS: bind a contributor to a feature it does not describe.
    # D = 14.00 with S = [14.15, 14.45] and a cell shifting the envelope up by +0.20/+0.30 gives
    # P = [14.20, 14.30], which S CONTAINS -- so containment alone would have PASSED a binding
    # whose realized feature sits outside the band it was supposedly built to.
    shifted = ProfileCell("bore.shifted.0_15", "bore", 6.0, 25.0, 0.0, 15.0, 0.20, 0.30)
    r = _run(
        [_binding(rule="bore.shifted.0_15", feature=_feature(diameter=14.00))],
        profile=_profile(shifted),
    )
    assert r.status == "not_run", r.message
    assert "does not describe this feature" in r.message


def test_a_binding_nobody_required_is_not_run() -> None:
    # Silently dropping a supplied binding is the same ignore-the-input pattern that hid the
    # unused nominal on P0.5's extras: the feature would be checked by nobody.
    extra = _binding(feature=_feature(feature_id="knee_bore"))
    r = _run([_binding(), extra], required=("hip_bore",))
    assert r.status == "not_run", r.message
    assert "knee_bore" in r.message and "does not name" in r.message


def test_a_repeated_required_id_is_not_run() -> None:
    r = _run([_binding()], required=("hip_bore", "hip_bore"))
    assert r.status == "not_run" and "listed more than once" in r.message


def test_containment_fails_on_the_upper_side_alone() -> None:
    # Sol: every failing fixture violated only the LOWER side, so `margin = P.low - S.low` would
    # have survived the whole suite. Here P.low >= S.low but P.high > S.high: only the upper
    # containment is broken, and it must still fail.
    upper = ProfileCell("bore.upper.0_15", "bore", 6.0, 25.0, 0.0, 15.0, -0.10, 0.30)
    r = _run([_binding(rule="bore.upper.0_15")], profile=_profile(upper))
    assert r.status == "fail", r.message
    assert r.measured["hip_bore.margin_mm"] == pytest.approx(-0.15, abs=1e-9)
    assert "TOO" not in r.message  # not the P0.5 vocabulary; this is an envelope-coverage fail


def test_every_required_feature_is_checked_not_just_the_first() -> None:
    # Sol: with only one required feature per fixture, "check the first and return" would pass.
    # One good feature and one whose declared tolerance is too tight, asserted in BOTH orders.
    good = _binding()
    bad = _binding(feature=_feature(feature_id="knee_bore"), minus=0.05, plus=0.05)
    for order in (("hip_bore", "knee_bore"), ("knee_bore", "hip_bore")):
        r = _run([good, bad], required=order)
        assert r.status == "fail", (order, r.message)
        assert r.measured["hip_bore.margin_mm"] == pytest.approx(0.05, abs=1e-9)
        assert r.measured["knee_bore.margin_mm"] == pytest.approx(-0.05, abs=1e-9)
        assert r.measured["worst_margin_mm"] == pytest.approx(-0.05, abs=1e-9)
        assert "knee_bore" in r.message


def test_finite_inputs_that_overflow_to_a_nan_margin_are_not_run() -> None:
    # THE min(x, NaN) TRAP, hit twice in this program. Every input here is finite, but
    # D + delta_high overflows to +inf, inf - inf is NaN, and min(0.0, nan) returns 0.0 -- which
    # read as a zero-margin PASS. Validating primitive inputs was never enough.
    huge = ProfileCell("bore.huge", "bore", 0.0, 1e308, 0.0, 15.0, 0.0, 1e308)
    binding = FeatureBinding(
        feature=_feature(diameter=1e308),
        contributor=Contributor("pelvis_hip_bore", 1e308, 0.0, 1e308, "hip ICD rev B"),
        cited_rule_id="bore.huge",
    )
    r = _run([binding], profile=_profile(huge))
    assert r.status == "not_run", r.message
    assert "not finite" in r.message and "uncomputable margin is not a passing one" in r.message


def test_every_process_field_is_compared_not_just_the_layer_height() -> None:
    # Sol: the mismatch fixture varied only layer_height_mm, so comparing that one field would
    # have survived. Each of the seven must independently invalidate the profile.
    changes = {
        "machine": "bambu_x1c",
        "material": "generic_petg",
        "nozzle_mm": 0.6,
        "layer_height_mm": 0.3,
        "extrusion_width_mm": 0.42,
        "slicer_revision": "prusaslicer-2.9.0",
        "dimensional_compensation": "xy_hole_comp=0.15",
    }
    for field_name, new_value in changes.items():
        other = replace(_PROCESS, **{field_name: new_value})
        r = _run([_binding()], process=other)
        assert r.status == "not_run", field_name
        assert field_name in r.message, field_name


def test_an_exact_cell_boundary_hulls_both_neighbours() -> None:
    # Sol's rounding false-PASS. An axis built at EXACTLY 15 deg measures 14.999999999999996, so
    # a point comparison lands inside [0,15] and outside [15,30] -- silently choosing the tight
    # cell and PASSING at +0.05 mm. The certified interval hulls both, and the hull FAILS by
    # 0.10 mm. The lowest low and the highest high deliberately live in DIFFERENT cells, so this
    # also proves both endpoints are hulled rather than one cell being picked whole.
    a = ProfileCell("bore.a.0_15", "bore", 6.0, 25.0, 0.0, 15.0, -0.10, 0.10)
    b = ProfileCell("bore.b.15_30", "bore", 6.0, 25.0, 15.0, 30.0, -0.25, 0.02)
    on_boundary = (math.sin(math.radians(15.0)), 0.0, math.cos(math.radians(15.0)))
    r = _run(
        [_binding(rule="bore.a.0_15", feature=_feature(axis=on_boundary))],
        rotation=_IDENTITY,
        profile=_profile(a, b),
    )
    assert r.status == "fail", r.message
    assert r.measured["boundary_union_used"] == 1.0
    assert r.measured["hip_bore.profile_low_mm"] == pytest.approx(14.05, abs=1e-9)
    assert r.measured["hip_bore.profile_high_mm"] == pytest.approx(14.40, abs=1e-9)


def test_a_wider_declared_tolerance_covers_the_flat_bore() -> None:
    # The honest way to pass a cross-layer bore: declare what the process can actually hold.
    # S = [14.00, 14.38] exactly contains P, so the margin is zero and it passes on the boundary.
    r = _run([_binding(rule="bore.horizontal.75_90", minus=0.30, plus=0.08)], rotation=_IDENTITY)
    assert r.status == "pass", r.message
    assert r.measured["worst_margin_mm"] == pytest.approx(0.0, abs=1e-9)
