"""Mass + static torque-margin checks (Sol P0.6 / task 3-4): can the servo hold the printed
limb? Hand-authored box-limb fixtures (no kernel/model). The datasheet gives STALL (11 kg·cm
at 6V); the check must be fed the CONTINUOUS envelope, so the tests pass a derated limit and
prove that stall-vs-continuous is the caller's to decide, not the check's to invent.

Reference numbers (PLA 1.24 g/cm^3): a 20x30x100 mm box = 60 cm^3 -> 74.4 g; centered at the
origin; a horizontal joint axis 50 mm below the COM gives a 50 mm worst-case lever ->
tau = 0.0744 kg * 5 cm = 0.372 kg·cm."""

from __future__ import annotations

import pytest

from forge_mind.validator.mechanics_geom import (
    mass_properties,
    worst_case_gravity_torque_margin,
)

trimesh = pytest.importorskip("trimesh")

_AXIS_PT = (0.0, 0.0, -50.0)  # 50 mm below the box COM
_AXIS_DIR = (1.0, 0.0, 0.0)  # horizontal (X) hinge


def _box_limb():
    return trimesh.creation.box(extents=(20.0, 30.0, 100.0))  # centered at origin


def test_mass_properties_match_hand_computation() -> None:
    mp = mass_properties(_box_limb())
    assert mp.ok and mp.watertight
    assert mp.volume_mm3 == pytest.approx(60000.0, rel=1e-6)
    assert mp.mass_g == pytest.approx(74.4, rel=1e-3)  # 60 cm^3 * 1.24
    assert mp.com_mm == pytest.approx((0.0, 0.0, 0.0), abs=1e-6)


def test_torque_passes_with_adequate_continuous_envelope() -> None:
    # A derated continuous limit (~30% of the 11 kg·cm stall) comfortably exceeds 0.372.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=3.3,
    )
    assert v.status == "pass", v.message
    assert v.measured["tau_worst_case_kg_cm"] == pytest.approx(0.372, rel=1e-2)
    assert v.measured["com_lever_mm"] == pytest.approx(50.0, rel=1e-3)
    assert v.measured["axis_gravity_sin"] == pytest.approx(1.0, abs=1e-6)  # horizontal axis


def test_torque_fails_when_continuous_envelope_too_small() -> None:
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=0.2,  # weaker than the 0.372 required
    )
    assert v.status == "fail", v.message
    assert v.measured["margin_kg_cm"] < 0.0


def test_safety_factor_tightens_the_margin() -> None:
    # SF 10x turns the comfortable 3.3 kg·cm limit into a shortfall (0.372*10 = 3.72 > 3.3).
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=3.3,
        safety_factor=10.0,
    )
    assert v.status == "fail", v.message


def test_missing_mesh_is_not_run_never_pass() -> None:
    v = worst_case_gravity_torque_margin(
        None, joint_axis_point=_AXIS_PT, joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=3.3
    )
    assert v.status == "not_run"


def test_non_watertight_mesh_is_not_run() -> None:
    box = _box_limb()
    # Remove a face -> not watertight -> mass untrustworthy -> not_run (never a silent PASS).
    holed = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2])
    v = worst_case_gravity_torque_margin(
        holed, joint_axis_point=_AXIS_PT, joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=3.3
    )
    assert v.status == "not_run" and "watertight" in v.message


# ---- fail-closed on malformed inputs (cross-examiner P2: no defensible-looking false PASS) ----


def test_negative_density_is_not_run_not_a_false_pass() -> None:
    # The exact reported hole: density -1.24 makes required torque negative -> margin >= 0
    # would mint a PASS even at zero capacity. Must be not_run.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=0.0,
        density_g_cm3=-1.24,
    )
    assert v.status == "not_run", v.message


def test_nonpositive_safety_factor_is_not_run() -> None:
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=3.3,
        safety_factor=0.0,
    )
    assert v.status == "not_run", v.message


def test_negative_capacity_is_not_run() -> None:
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=-5.0,
    )
    assert v.status == "not_run", v.message


def test_degenerate_joint_axis_is_not_run() -> None:
    # Zero axis with axis-point at the COM would collapse the lever to 0 -> false PASS.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=(0.0, 0.0, 0.0),
        joint_axis_dir=(0.0, 0.0, 0.0),
        continuous_torque_kg_cm=0.0,
    )
    assert v.status == "not_run", v.message


def test_degenerate_volume_mesh_is_not_run() -> None:
    # A flattened (coplanar) mesh has ~0 signed volume -> ~0 mass -> would auto-PASS.
    flat = _box_limb().copy()
    flat.vertices[:, 2] = 0.0
    v = worst_case_gravity_torque_margin(
        flat, joint_axis_point=_AXIS_PT, joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=0.0
    )
    assert v.status == "not_run", v.message


def test_vertical_yaw_axis_has_zero_gravity_torque() -> None:
    # The physics fix (Sol + skeptic): a joint axis PARALLEL to gravity (a Z yaw joint) bears
    # ZERO gravity torque at every rotation. The old mass*rho form reported a positive load and
    # would reject a valid yaw servo. Now a near-zero-capacity servo PASSES.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=(20.0, 0.0, 0.0),  # Z axis offset in X so rho > 0
        joint_axis_dir=(0.0, 0.0, 1.0),  # vertical, parallel to gravity (0,0,-1)
        continuous_torque_kg_cm=0.001,
    )
    assert v.status == "pass", v.message
    assert v.measured["axis_gravity_sin"] == pytest.approx(0.0, abs=1e-9)
    assert v.measured["tau_worst_case_kg_cm"] == pytest.approx(0.0, abs=1e-9)


def test_exact_pose_torque_can_be_zero_while_worst_case_is_not() -> None:
    # SOL'S REQUESTED FIXTURE: the box COM sits directly BELOW the horizontal X axis (a pendulum
    # at rest), so the exact at-pose torque |a . (r x mg)| is EXACTLY zero -- yet the servo must
    # still hold 0.372 kgf·cm once the limb swings off-azimuth. Gating on the exact value would
    # pass a limb that collapses the moment it moves; the gate uses the worst case.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,  # COM hangs straight below this axis
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=3.3,
    )
    assert v.status == "pass", v.message
    assert v.measured["tau_exact_kg_cm"] == pytest.approx(0.0, abs=1e-12)
    assert v.measured["tau_worst_case_kg_cm"] == pytest.approx(0.372, rel=1e-2)
    # And the gate is driven by the worst case, not the (zero) exact value:
    short = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=0.1,  # would trivially "pass" against the exact zero
    )
    assert short.status == "fail", short.message


def test_exact_torque_is_nonzero_for_a_horizontal_com_offset() -> None:
    # COM offset horizontally from the axis (limb held out sideways): exact == worst case.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=(0.0, 50.0, 0.0),  # axis offset in Y; gravity -Z; X hinge
        joint_axis_dir=_AXIS_DIR,
        continuous_torque_kg_cm=3.3,
    )
    assert v.status == "pass", v.message
    assert v.measured["tau_exact_kg_cm"] == pytest.approx(
        v.measured["tau_worst_case_kg_cm"], rel=1e-6
    )


def test_oblique_axis_scales_between_zero_and_full() -> None:
    # A 45-degree axis carries only sin(45)=0.707 of the gravity moment: strictly between the
    # vertical (0) and horizontal (full) cases.
    v = worst_case_gravity_torque_margin(
        _box_limb(),
        joint_axis_point=_AXIS_PT,
        joint_axis_dir=(1.0, 0.0, 1.0),  # 45 deg from both horizontal and gravity
        continuous_torque_kg_cm=3.3,
    )
    assert v.status == "pass", v.message
    assert 0.0 < v.measured["axis_gravity_sin"] < 1.0
    assert v.measured["axis_gravity_sin"] == pytest.approx(0.7071, abs=1e-3)


def test_malformed_axis_point_and_gravity_are_not_run() -> None:
    box = _box_limb()
    # None point -> must not_run (not crash during iteration).
    assert (
        worst_case_gravity_torque_margin(
            box, joint_axis_point=None, joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=3.3
        ).status
        == "not_run"
    )
    # Two-component point -> not_run (would broadcast-error later).
    assert (
        worst_case_gravity_torque_margin(
            box, joint_axis_point=(0.0, 0.0), joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=3.3
        ).status
        == "not_run"
    )
    # Four-component axis -> not_run (would silently truncate).
    assert (
        worst_case_gravity_torque_margin(
            box,
            joint_axis_point=_AXIS_PT,
            joint_axis_dir=(1.0, 0.0, 0.0, 0.0),
            continuous_torque_kg_cm=3.3,
        ).status
        == "not_run"
    )
    # Zero gravity vector -> not_run.
    assert (
        worst_case_gravity_torque_margin(
            box,
            joint_axis_point=_AXIS_PT,
            joint_axis_dir=_AXIS_DIR,
            continuous_torque_kg_cm=3.3,
            gravity_dir=(0.0, 0.0, 0.0),
        ).status
        == "not_run"
    )


def test_reverse_wound_solid_measures_correctly_not_over_rejected() -> None:
    # A valid solid with inverted winding (negative signed volume) must be REPAIRED and
    # measured, not falsely rejected -- proving the hardening didn't break real parts.
    rev = _box_limb().copy()
    rev.invert()  # flip all faces -> reverse winding, still watertight
    v = worst_case_gravity_torque_margin(
        rev, joint_axis_point=_AXIS_PT, joint_axis_dir=_AXIS_DIR, continuous_torque_kg_cm=3.3
    )
    assert v.status == "pass", v.message
    assert v.measured["mass_g"] == pytest.approx(74.4, rel=1e-3)
