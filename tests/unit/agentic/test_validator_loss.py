"""Validator-derived fit loss (ADR-0074): wall/envelope/balance deficits as loss terms, each
fail-soft to 0 -- and the headline: the optimizer fits a part to the brief's stated envelope with
NO reference at all.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.param_opt import fit_loss, polish_params
from forge_mind.agentic.params import extract_params
from forge_mind.agentic.validator_loss import ValidatorTarget, validator_loss, validator_target


def _brief(envelope: str = "40 x 10 x 10", min_wall: float = 1.0) -> DesignBrief:
    return DesignBrief(
        object_name="mounting bracket",
        summary="a bracket",
        visual_description="a rectangular bracket",
        overall_envelope_mm=envelope,
        features=[BriefFeature(name="body", purpose="mass", approx_dims_mm="40", method="box")],
        min_wall_mm=min_wall,
        print_orientation_note="flat",
    )


def _save(tmp: Path, name: str, mesh: trimesh.Trimesh) -> Path:
    p = tmp / name
    mesh.export(str(p))
    return p


def test_validator_target_from_brief() -> None:
    t = validator_target(_brief())
    assert t.min_wall_mm == 1.0
    assert t.envelope_sorted == (10.0, 10.0, 40.0)  # sorted ascending
    assert t.balance_axis is None  # a bracket is not rotating_power; no baseline_step given
    assert t.applicable


def test_envelope_term_zero_when_matched_positive_when_off(tmp_path: Path) -> None:
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=(10.0, 10.0, 40.0), balance_axis=None)
    right = _save(tmp_path, "right.stl", trimesh.creation.box(extents=[40, 10, 10]))
    stubby = _save(tmp_path, "stubby.stl", trimesh.creation.box(extents=[15, 10, 10]))
    assert validator_loss(right, t) < 0.02  # matches the brief -> ~0
    assert validator_loss(stubby, t) > 0.15  # 15 vs 40 on the long axis -> real signal


def test_wall_term_flags_a_thin_plate(tmp_path: Path) -> None:
    t = ValidatorTarget(min_wall_mm=1.0, envelope_sorted=None, balance_axis=None)
    thin = _save(tmp_path, "thin.stl", trimesh.creation.box(extents=[20, 20, 0.5]))
    thick = _save(tmp_path, "thick.stl", trimesh.creation.box(extents=[20, 20, 5]))
    assert validator_loss(thin, t) > 0.3  # 0.5mm wall vs 1.0 required -> ~0.5 deficit
    assert validator_loss(thick, t) < 0.05


def test_balance_term_flags_offset_mass(tmp_path: Path) -> None:
    axis = ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=None, balance_axis=axis)
    offset = trimesh.creation.box(extents=[10, 10, 10])
    offset.apply_translation([10, 0, 0])  # all mass 10mm off the spin axis
    centered = trimesh.creation.box(extents=[10, 10, 10])
    off_p = _save(tmp_path, "off.stl", offset)
    cen_p = _save(tmp_path, "cen.stl", centered)
    assert validator_loss(off_p, t) > 0.1  # unbalanced about the axis
    assert validator_loss(cen_p, t) < 0.05  # mass-symmetric -> balanced


def test_not_applicable_contributes_zero(tmp_path: Path) -> None:
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=None, balance_axis=None)
    assert not t.applicable
    box = _save(tmp_path, "b.stl", trimesh.creation.box(extents=[10, 10, 10]))
    assert validator_loss(box, t) == 0.0


def test_fit_loss_accepts_validator_target_only(tmp_path: Path) -> None:
    """fit_loss with NO reference target: the validity guard still rules, the vterms score."""
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=(10.0, 10.0, 40.0), balance_axis=None)
    good = _save(tmp_path, "g.stl", trimesh.creation.box(extents=[40, 10, 10]))
    assert fit_loss(good, None, t) < 0.02
    # two disconnected bodies still hard-reject even with only vterms
    a = trimesh.creation.box(extents=[5, 5, 5])
    b = trimesh.creation.box(extents=[5, 5, 5])
    b.apply_translation([50, 0, 0])
    two = _save(tmp_path, "two.stl", trimesh.util.concatenate([a, b]))
    assert fit_loss(two, None, t) == float("inf")


# authored 30 vs the brief's 40 = a 25% deviation, the realistic case (the author derives its
# dims FROM the same brief, so it lands near the envelope; the measured real failure was ~30%
# off). The +/-40% search bounds are sized for exactly this -- NOT for a 2x+ authored blunder,
# which is a STRUCTURE failure and stays the LLM's job.
CODE = 'PARAMS = {"l": 30.0, "w": 10.0, "h": 10.0}\nresult = Box(PARAMS["l"], PARAMS["w"], PARAMS["h"])\n'


def _box_builder(code: str, out_dir: Path) -> tuple[Path | None, Path | None]:
    spec = extract_params(code)
    if spec is None:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "part.stl"
    trimesh.creation.box(extents=[float(spec.values[k]) for k in ("l", "w", "h")]).export(str(stl))
    return stl, stl


def test_polish_fits_the_brief_envelope_with_no_reference(tmp_path: Path) -> None:
    """The ADR-0074 headline: NO reference mesh at all -- the optimizer stretches the authored
    30x10x10 box toward the brief's stated 40x10x10 envelope, purely from validator targets."""
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=(10.0, 10.0, 40.0), balance_axis=None)
    out = polish_params(
        CODE, None, tmp_path / "opt", vtarget=t, max_evals=40, build_fn=_box_builder
    )
    assert out is not None and out.improved
    spec = extract_params(out.best_code)
    assert spec is not None
    assert float(spec.values["l"]) > 36.0  # closed most of the 30 -> 40 gap


def test_polish_none_when_nothing_measurable(tmp_path: Path) -> None:
    empty = ValidatorTarget(min_wall_mm=None, envelope_sorted=None, balance_axis=None)
    assert polish_params(CODE, None, tmp_path / "o", vtarget=empty, build_fn=_box_builder) is None
    assert polish_params(CODE, None, tmp_path / "o2", build_fn=_box_builder) is None


# --- polish guards (ADR-0075, from the measured turbo-wheel incident) ---------------------------


def test_fit_target_rejects_degenerate_reference(tmp_path: Path) -> None:
    """The caught case: a reference rendering as 0 x 0 x 0 mm must never become a fit target."""
    from forge_mind.agentic.param_opt import fit_target

    tiny = tmp_path / "tiny.stl"
    trimesh.creation.box(extents=[0.4, 0.4, 0.4]).export(str(tiny))  # sub-mm garbage scale
    assert fit_target(tiny) is None
    normal = tmp_path / "normal.stl"
    trimesh.creation.box(extents=[40, 10, 10]).export(str(normal))
    assert fit_target(normal) is not None  # real references still fit


def test_critic_accepted_part_is_never_reference_fitted(tmp_path: Path) -> None:
    from forge_mind.agentic.loop import _reference_for_polish

    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[40, 10, 10]).export(str(ref))
    assert _reference_for_polish(ref, 4) is None  # accepted form: validator terms only
    assert _reference_for_polish(ref, 3) == ref  # not critic-accepted: reference fit allowed
    assert _reference_for_polish(None, 3) is None


# --- feature-deficit term (ADR-0078, from the hard-5 carrier) ------------------------------------


def test_feature_term_sees_missing_through_holes(tmp_path: Path) -> None:
    """A solid cylinder with 1 planned hole scores the full deficit; an annulus (genus 1) scores
    zero -- the dimples-instead-of-bores failure is now visible to the optimizer."""
    t = ValidatorTarget(
        min_wall_mm=None, envelope_sorted=None, balance_axis=None, planned_through_holes=1
    )
    assert t.applicable
    solid = _save(tmp_path, "solid.stl", trimesh.creation.cylinder(radius=10, height=8))
    tube = _save(tmp_path, "tube.stl", trimesh.creation.annulus(r_min=4.0, r_max=10.0, height=8.0))
    assert validator_loss(solid, t) > 0.9  # the planned hole is absent
    assert validator_loss(tube, t) < 0.05  # genus 1 satisfies the plan


def test_feature_term_never_rewards_extra_holes(tmp_path: Path) -> None:
    t = ValidatorTarget(
        min_wall_mm=None, envelope_sorted=None, balance_axis=None, planned_through_holes=1
    )
    tube = _save(tmp_path, "t2.stl", trimesh.creation.annulus(r_min=4.0, r_max=10.0, height=8.0))
    assert validator_loss(tube, t) >= 0.0  # deficit clamps at 0; surplus genus is not a bonus


# --- the loss must never want what the gate does not require (timing-gear incident) --------------


def test_wall_target_matches_the_gate_effective_threshold() -> None:
    """Measured on the timing-gear run: the researcher declared min_wall 2.5, the GENERAL gate
    enforces 1.0, and the loss chasing 2.5 spent four brief-stated dimensions (tip 76->92 mm)
    buying wall the gate never demanded. The loss target must be the GATE's threshold."""
    from forge_mind.fabrication import GENERAL_MIN_WALL_MM, FabricationMode

    b = _brief(min_wall=2.5)
    assert validator_target(b).min_wall_mm == 2.5  # no mode given: declared wall as-is
    t = validator_target(b, fabrication_mode=FabricationMode.GENERAL)
    assert t.min_wall_mm == GENERAL_MIN_WALL_MM  # GENERAL relaxes to the gate's floor
    t3d = validator_target(b, fabrication_mode=FabricationMode.PRINT_3D)
    assert t3d.min_wall_mm == 2.5  # PRINT_3D has no override -- declared wall stands


def test_envelope_parses_the_round_part_form() -> None:
    """Measured on the timing-gear run: '76 mm dia x 14 mm tall (...)' parsed as None, the
    envelope term vanished, and nothing anchored absolute size (the part grew to 92 mm)."""
    from forge_mind.agentic.brief import parse_envelope_mm

    assert parse_envelope_mm("76 mm dia x 14 mm tall (8 mm web + 6 mm boss proud of one face)") == (
        76.0,
        76.0,
        14.0,
    )
    assert parse_envelope_mm("Ø60 x 40 high") == (60.0, 60.0, 40.0)
    assert parse_envelope_mm("60 x 60 x 40 mm") == (60.0, 60.0, 40.0)  # the triple still wins
    assert parse_envelope_mm("a knob about 30 mm wide") is None  # no clean claim -> no term


def test_envelope_parses_the_axis_letter_form() -> None:
    """Measured on the ADR-0083 audit: the brief routinely writes '120 L x 40 W x 85 H mm'
    (~half of arbitrary_v1). The axis letters between the numbers made parse_envelope_mm
    return None, so the >20% envelope GATE (part of validation.passed) FAILED OPEN and a
    1.4x-scale part passed -- corroborated by the bench's hook_j_utility_wall. The gate now
    fires on these briefs; bare-number / round / unparseable forms are unchanged."""
    from forge_mind.agentic.brief import parse_envelope_mm

    # the escaped-audit briefs, now parsed
    assert parse_envelope_mm("120 L x 40 W x 85 H mm -- horizontal leg sets the footprint") == (
        120.0,
        40.0,
        85.0,
    )
    assert parse_envelope_mm("90 L x 60 W x 40 H mm -- dimensions stated explicitly") == (
        90.0,
        60.0,
        40.0,
    )
    assert parse_envelope_mm("110 l x 20 w x 20 h mm") == (110.0, 20.0, 20.0)  # lowercase too
    # NO regression: the bare-number triple, round-part, and unparseable forms are unchanged
    assert parse_envelope_mm("64 x 64 x 50 mm -- 60 mm flange + 4 mm") == (64.0, 64.0, 50.0)
    assert parse_envelope_mm("60x60x40mm") == (60.0, 60.0, 40.0)
    assert parse_envelope_mm("120 by 30 by 15 (handle + base)") == (120.0, 30.0, 15.0)
    assert parse_envelope_mm("76 mm dia x 14 mm tall") == (76.0, 76.0, 14.0)
    # fail-open PRESERVED: the mm-interleaved parenthetical form stays None (never a false triple)
    assert parse_envelope_mm("30 mm (L, along axis) x 8 mm (W) x 10 mm (H)") is None
