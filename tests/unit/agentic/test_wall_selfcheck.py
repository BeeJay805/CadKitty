"""The WALL half of the author self-check (L102's largest kill class: 5 of 8 H5B failures
were sub-floor walls the author only learned about when the validator failed the part).

The load-bearing property is SHARED MEASUREMENT: the self-check tells the author "your own
measurement", so that number must be the number the gate later reads. One function
(`measure_wall_fraction`) now serves both, and these tests pin (a) the gate's verdicts are
unchanged by the extraction, (b) the numbers agree exactly, (c) the epilogue stays
feedback-only and fails open."""

from __future__ import annotations

import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.selfcheck import build_spec, feedback_block, scoring_spec
from forge_mind.agentic.validate import _thin_wall_gate
from forge_mind.validator.feature_checks import measure_wall_fraction


def _thin_plate() -> trimesh.Trimesh:
    """A 0.8mm plate: an area-significant sub-floor WALL (parallel faces), not a knife edge."""
    return trimesh.creation.box(extents=(60.0, 60.0, 0.8))


def _thick_block() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(60.0, 60.0, 20.0))


def test_gate_still_fails_a_thin_plate_and_passes_a_thick_block() -> None:
    """Behavioral pin on the extraction: the gate's verdicts are what they were."""
    failed, msg = _thin_wall_gate(_thin_plate(), 2.8)
    assert failed is True
    assert msg is not None and "WALL GATE" in msg and "2.80 mm floor" in msg

    failed, msg = _thin_wall_gate(_thick_block(), 2.8)
    assert failed is False and msg is None


def test_gate_and_selfcheck_read_the_same_numbers() -> None:
    """The anti-second-liar property: the fraction/thinnest the author is shown are the exact
    values the gate thresholds — same function, seeded, so equality is exact."""
    mesh = _thin_plate()
    first = measure_wall_fraction(mesh, 2.8)
    second = measure_wall_fraction(mesh, 2.8)
    assert first is not None and first == second  # deterministic, not merely close
    frac, thinnest = first.thin_fraction, first.thinnest_mm
    assert frac > 0.03  # the gate's own threshold — this mesh is a genuine fail
    assert thinnest is not None and thinnest < 2.8
    # and the gate's message quotes that same thinnest value
    _failed, msg = _thin_wall_gate(mesh, 2.8)
    assert msg is not None and f"{thinnest:.2f} mm" in msg


def test_clear_wall_measures_zero_fraction_and_no_thinnest() -> None:
    measured = measure_wall_fraction(_thick_block(), 2.8)
    assert measured is not None
    frac, thinnest = measured.thin_fraction, measured.thinnest_mm
    assert frac == 0.0 and thinnest is None


def test_unmeasurable_inputs_fail_open() -> None:
    assert measure_wall_fraction(_thick_block(), 0.0) is None  # no floor -> no claim
    assert measure_wall_fraction(trimesh.Trimesh(), 2.8) is None  # empty mesh -> no claim


def _brief(wall: float = 2.8) -> DesignBrief:
    return DesignBrief(
        object_name="shell",
        summary="a tapered shell",
        visual_description="smooth tapered shell",
        overall_envelope_mm="60 x 60 x 20 mm",
        min_wall_mm=wall,
        material="PLA",
        target="cosmetic",
        print_orientation_note="upright",
        features=[
            BriefFeature(
                name="shell wall",
                purpose="the structural skin",
                approx_dims_mm="2.8 walls",
                method="lofted shell",
            )
        ],
    )


def test_spec_carries_the_deterministic_print_wall_floor() -> None:
    spec = build_spec(_brief())
    assert spec is not None and spec["wall_floor_mm"] == 2.8


def test_spec_rejects_a_brief_floor_below_the_material_floor() -> None:
    spec = build_spec(_brief(0.6))
    assert spec is not None and spec["wall_floor_mm"] == 1.0


def test_spec_uses_the_effective_floor_the_validator_will_apply() -> None:
    """Review-caught false-feedback bug: the GENERAL (machined, not-FDM) lane relaxes a
    declared wall to the structural-sanity floor, and the refs bench runs GENERAL — telling
    the author a valid 1.5mm machined wall 'fails the 2.8mm gate' would be a second liar."""
    from forge_mind.fabrication import GENERAL_MIN_WALL_MM, FabricationMode

    printed = build_spec(_brief(2.8), FabricationMode.PRINT_3D)
    general = build_spec(_brief(2.8), FabricationMode.GENERAL)
    assert printed is not None and printed["wall_floor_mm"] == 2.8
    assert general is not None and general["wall_floor_mm"] == GENERAL_MIN_WALL_MM
    # and a declared wall BELOW the general floor is never relaxed upward (never stricter)
    thin_decl = build_spec(_brief(0.6), FabricationMode.GENERAL)
    assert thin_decl is not None and thin_decl["wall_floor_mm"] == 0.6


def test_scoring_spec_drops_the_wall_target_without_mutating() -> None:
    """Review-caught cost defect: the CAD-Coder draft scorer reads envelope + bores only, so a
    wall target there would pay a 4096-ray measurement per draft that nothing reads."""
    spec = build_spec(_brief(2.8))
    assert spec is not None and spec["wall_floor_mm"] == 2.8
    scored = scoring_spec(spec)
    assert scored is not None and scored["wall_floor_mm"] is None
    assert scored["envelope"] == spec["envelope"] and scored["holes"] == spec["holes"]
    assert spec["wall_floor_mm"] == 2.8  # the caller's spec is untouched
    assert scoring_spec(None) is None


def test_wall_floor_alone_does_not_manufacture_a_spec() -> None:
    """Review-caught contract: min_wall_mm is a REQUIRED brief field, so keying spec existence
    on it would make every brief truthy and silently defeat cadcoder's early exit (its
    perfect-score check assumes a truthy spec is fully scoreable, and wall data is not scored).
    Envelope/holes stay the existence test."""
    brief = _brief().model_copy(update={"overall_envelope_mm": "no numbers here"})
    assert build_spec(brief) is None


def test_feedback_states_the_shortfall_with_the_gate_threshold() -> None:
    block = feedback_block({"wall": {"floor_mm": 2.8, "thin_fraction": 0.21, "thinnest_mm": 0.39}})
    assert "21.0% of your surface" in block
    assert "0.39 mm" in block and "2.80 mm floor" in block
    assert "THICKEN" in block and "FAILS the wall gate" in block


def test_feedback_never_claims_a_failure_the_gate_would_accept() -> None:
    """Review-caught contradiction: 0.0286 rendered as '3%' and claimed a failure, while
    `_thin_wall_gate` accepts anything under the 3% area threshold. The WORDING must follow the
    shared threshold, and the digits must not read as the threshold itself."""
    from forge_mind.validator.feature_checks import WALL_GATE_MIN_FRAC

    just_under = feedback_block(
        {
            "wall": {
                "floor_mm": 2.8,
                "thin_fraction": WALL_GATE_MIN_FRAC - 0.0014,
                "thinnest_mm": 1.2,
            }
        }
    )
    assert "ACCEPTS this" in just_under and "FAILS" not in just_under
    assert "2.86%" in just_under  # floor-rendered, so it cannot read as the 3% cutoff

    just_over = feedback_block(
        {"wall": {"floor_mm": 2.8, "thin_fraction": WALL_GATE_MIN_FRAC, "thinnest_mm": 1.2}}
    )
    assert "FAILS the wall gate" in just_over and "ACCEPTS" not in just_over


def test_a_fraction_that_only_rounds_to_the_threshold_still_reads_as_accepted() -> None:
    """Review-caught: the epilogue must carry the RAW fraction, because 105/3501 = 2.9991%
    rounds to 0.0300 at 4dp — the gate accepts it, so the feedback must not claim a failure."""
    from forge_mind.validator.feature_checks import WALL_GATE_MIN_FRAC

    raw = 105 / 3501  # 0.029991..., below the threshold; rounds UP to 0.03 at 4 decimals
    assert round(raw, 4) == WALL_GATE_MIN_FRAC and raw < WALL_GATE_MIN_FRAC
    block = feedback_block({"wall": {"floor_mm": 2.8, "thin_fraction": raw, "thinnest_mm": 1.1}})
    assert "ACCEPTS this" in block and "FAILS" not in block
    # ...and the RENDERED number must not read as the threshold either (2.9991% -> 2.99%,
    # never "3.0% ... BELOW the 3% threshold", which contradicts itself)
    assert "2.99%" in block and "3.0%" not in block


def test_a_vanishing_sub_threshold_fraction_renders_as_a_bound() -> None:
    block = feedback_block(
        {"wall": {"floor_mm": 2.8, "thin_fraction": 0.00004, "thinnest_mm": 2.7}}
    )
    assert "<0.01%" in block and "ACCEPTS this" in block


def test_feedback_says_clear_when_nothing_is_sub_floor() -> None:
    block = feedback_block({"wall": {"floor_mm": 2.8, "thin_fraction": 0.0, "thinnest_mm": None}})
    assert "no sub-floor wall measured" in block


def test_feedback_omits_wall_when_unmeasured() -> None:
    """Fail-open all the way to the prompt: an epilogue that could not measure says nothing
    (never 'clear', which would be a false reassurance)."""
    assert feedback_block({"wall": {"floor_mm": 2.8}}) == ""
    assert feedback_block({}) == ""
