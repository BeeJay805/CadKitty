"""Geometry-level actuation/mass verification (Sol consultation 2026-07-24, P0.6 / task 3-4,
decision-independent): does the servo have the torque to hold the printed limb it drives?
Measured from the actual mesh mass + lever arm -- the check H3 never ran (it validated shape,
never whether an MG996R could lift the arm it was bolted to).

Two pieces:
  * ``mass_properties`` -- mass + COM from the built mesh at a cited material density.
  * ``worst_case_gravity_torque_margin`` -- worst-case static gravity torque about a joint axis
    (max over COM azimuth, the quantity that sizes a servo) vs its CONTINUOUS torque envelope;
    it also reports the exact at-pose torque as evidence, but never gates on it.

Provenance discipline (Sol: "no uncited generic limits"; cad-validator-designer): the required
torque is computed from MEASURED mass + geometry and a cited g; the CONTINUOUS torque limit is
NOT invented here -- the caller supplies it from a versioned part/profile (Sol task 4). The
MG996R datasheet gives STALL (``servo.MG996R_STALL_TORQUE_KG_CM_6V0`` = 11.0), which a servo
cannot hold continuously; passing stall as the limit would be the exact mistake Sol flagged.

Standalone (not wired to the live verdict -- the held intent-gate reclassification).

UNITS (declared explicitly, per Sol's dimensional challenge): torque is KILOGRAM-FORCE
centimetre (kgf·cm), the hobby-servo datasheet's own unit, on BOTH sides of the comparison.
A mass of m kg at lever L cm exerts m*L kgf·cm because 1 kgf is by definition the weight of
1 kg under standard gravity -- g appears on both sides and cancels. Callers must therefore
supply the servo limit in kgf·cm too (as MG996R datasheets do). No SI N·m value may be passed
without conversion.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any

from forge_mind.validator.feature_geom import (
    _point_to_line_distance,
    finite_nonneg,
    finite_point,
    finite_positive,
    valid_axis,
)
from forge_mind.validator.symmetry import GeomVerdict

Vec3 = tuple[float, float, float]

# PLA solid density (materials.py, cited). A print is <100% infill, so a solid-density mass is a
# CONSERVATIVE upper bound for a hold-torque check (heavier than reality) -- the safe direction.
_PLA_DENSITY_G_CM3 = 1.24
# Below this signed volume a "watertight" mesh is degenerate (a flattened/zero-thickness shell)
# or reverse-wound past repair -- its mass is not trustworthy, so we refuse it (no false PASS).
_MIN_VOLUME_MM3 = 1e-6


@dataclass(frozen=True)
class MassProps:
    """Measured mass properties of one body. ``ok`` False = unmeasurable (never fabricated)."""

    ok: bool
    mass_g: float
    com_mm: Vec3
    volume_mm3: float
    watertight: bool
    note: str = ""


def _usable(mesh: Any) -> bool:
    try:
        return mesh is not None and len(mesh.faces) > 0 and len(mesh.vertices) > 0
    except Exception:
        return False


def mass_properties(mesh: Any, *, density_g_cm3: float = _PLA_DENSITY_G_CM3) -> MassProps:
    """Mass (g) + COM (mm) of ``mesh`` at ``density_g_cm3`` (solid). ``ok`` False when the mesh
    is missing/empty. ``watertight`` False means the volume/COM are approximate -- a downstream
    load check should treat that as ``not_run`` (a non-watertight limb is itself a defect)."""
    if not _usable(mesh):
        return MassProps(False, 0.0, (0.0, 0.0, 0.0), 0.0, False, "mesh missing or empty")
    if not finite_positive(density_g_cm3):
        return MassProps(
            False, 0.0, (0.0, 0.0, 0.0), 0.0, False, f"invalid density {density_g_cm3!r}"
        )
    try:
        # Normalize winding on a COPY so a reverse-wound solid reports a positive signed
        # volume + a correct volumetric COM (never mutate the caller's mesh).
        m = mesh.copy()
        with contextlib.suppress(Exception):
            m.fix_normals()
        watertight = bool(m.is_watertight)
        signed_volume = float(m.volume)
        if not math.isfinite(signed_volume) or signed_volume <= _MIN_VOLUME_MM3:
            # A flattened/degenerate or unrepairably reverse-wound closed shell: mass is not
            # trustworthy (a ~0 volume would mint a torque PASS). Fail closed.
            return MassProps(
                False,
                0.0,
                (0.0, 0.0, 0.0),
                0.0,
                watertight,
                f"degenerate or non-positive volume ({signed_volume:.3g} mm^3)",
            )
        com_raw = m.center_mass if watertight else m.centroid
        com = (float(com_raw[0]), float(com_raw[1]), float(com_raw[2]))
        mass_g = signed_volume * density_g_cm3 / 1000.0  # mm^3 -> cm^3 (/1000), * g/cm^3
        note = "" if watertight else "mesh not watertight: volume/COM approximate"
        return MassProps(True, mass_g, com, signed_volume, watertight, note)
    except Exception as exc:
        return MassProps(False, 0.0, (0.0, 0.0, 0.0), 0.0, False, f"{type(exc).__name__}: {exc}")


def worst_case_gravity_torque_margin(
    mesh: Any,
    *,
    joint_axis_point: Vec3,
    joint_axis_dir: Vec3,
    continuous_torque_kg_cm: float,
    gravity_dir: Vec3 = (0.0, 0.0, -1.0),
    density_g_cm3: float = _PLA_DENSITY_G_CM3,
    safety_factor: float = 1.0,
) -> GeomVerdict:
    """WORST-CASE static gravity hold torque about the joint axis, over every rotation of the
    body about that axis, vs the servo's CONTINUOUS torque (kgf·cm on both sides).

    Two distinct quantities are measured and reported (Sol's correction -- conflating them was
    a labelling error):

    * ``tau_exact_kg_cm`` -- the torque at the body's ACTUAL pose as given:
      ``|axis . ((r_com - axis_point) x m*g)|`` (scalar triple product). A limb hanging straight
      below a horizontal axis has exact torque ZERO -- true, and useless for sizing a servo.
    * ``tau_worst_case_kg_cm`` -- the maximum over COM azimuth as the limb swings about the
      joint: ``mass * rho * |sin(angle(axis, gravity))|``, where ``rho`` is the perpendicular
      distance from the axis to the COM. A HORIZONTAL axis gives the full ``mass*rho``; a
      VERTICAL/yaw axis correctly gives ZERO (gravity cannot torque a yaw joint).

    The GATE uses the worst case, because a servo must hold the limb anywhere in its range;
    gating on the instantaneous pose would pass a limb that collapses as soon as it moves.
    PASS when ``continuous_torque_kg_cm >= tau_worst_case * safety_factor``.

    ``continuous_torque_kg_cm`` MUST be the continuous/derated envelope from a versioned
    profile, NOT the datasheet stall (Sol). Single rigid body only (summing downstream limbs,
    hardware and payload is the load-case check, P0.6-full). Fail-closed ``not_run`` on any
    malformed input or an unmeasurable/untrustworthy mass."""
    check_id = "mechanics.torque_margin"
    cat = "load"
    limit = {
        "continuous_torque_kg_cm": continuous_torque_kg_cm,
        "safety_factor": safety_factor,
    }
    # Input validation FIRST (fail closed): nonpositive density/SF or negative capacity would let
    # a real load read as zero/negative and mint a PASS; a malformed axis/point/gravity vector
    # would crash or silently broadcast. None may reach the comparison.
    if not finite_positive(density_g_cm3):
        return GeomVerdict(
            check_id, cat, "not_run", f"invalid density {density_g_cm3!r}", {}, limit
        )
    if not finite_positive(safety_factor):
        return GeomVerdict(
            check_id, cat, "not_run", f"invalid safety_factor {safety_factor!r}", {}, limit
        )
    if not finite_nonneg(continuous_torque_kg_cm):
        return GeomVerdict(
            check_id,
            cat,
            "not_run",
            f"invalid continuous_torque_kg_cm {continuous_torque_kg_cm!r}",
            {},
            limit,
        )
    if not valid_axis(joint_axis_dir):
        return GeomVerdict(
            check_id,
            cat,
            "not_run",
            f"degenerate/malformed joint axis {joint_axis_dir!r}",
            {},
            limit,
        )
    if not valid_axis(gravity_dir):
        return GeomVerdict(
            check_id, cat, "not_run", f"degenerate/malformed gravity_dir {gravity_dir!r}", {}, limit
        )
    if not finite_point(joint_axis_point):
        return GeomVerdict(
            check_id, cat, "not_run", f"malformed joint_axis_point {joint_axis_point!r}", {}, limit
        )
    mp = mass_properties(mesh, density_g_cm3=density_g_cm3)
    if not mp.ok:
        return GeomVerdict(check_id, cat, "not_run", f"mass unmeasurable: {mp.note}", {}, limit)
    if not mp.watertight:
        return GeomVerdict(check_id, cat, "not_run", f"mass untrustworthy: {mp.note}", {}, limit)
    # rho = perpendicular distance axis->COM; sin_ag = sine of the angle between the joint axis
    # and gravity (= the fraction of gravity that actually torques the joint).
    rho_mm = _point_to_line_distance(mp.com_mm, joint_axis_point, joint_axis_dir)
    au = _unit_vec(joint_axis_dir)
    gu = _unit_vec(gravity_dir)
    cos_ag = abs(au[0] * gu[0] + au[1] * gu[1] + au[2] * gu[2])
    sin_ag = math.sqrt(max(0.0, 1.0 - min(1.0, cos_ag) ** 2))
    mass_kg = mp.mass_g / 1000.0
    # Worst case over COM azimuth (the GATE): the limb swung to its most demanding pose.
    tau_worst = mass_kg * (rho_mm / 10.0) * sin_ag  # kgf·cm
    # Exact torque at the pose AS GIVEN: |axis . (r x m*g)| (scalar triple product). Reported
    # for evidence, never gated on -- a limb hanging straight down reads zero here.
    r_cm = tuple((mp.com_mm[i] - float(joint_axis_point[i])) / 10.0 for i in range(3))
    cross = (
        r_cm[1] * gu[2] - r_cm[2] * gu[1],
        r_cm[2] * gu[0] - r_cm[0] * gu[2],
        r_cm[0] * gu[1] - r_cm[1] * gu[0],
    )
    tau_exact = abs(mass_kg * (au[0] * cross[0] + au[1] * cross[1] + au[2] * cross[2]))
    required_with_sf = tau_worst * safety_factor
    margin = continuous_torque_kg_cm - required_with_sf
    ratio = continuous_torque_kg_cm / required_with_sf if required_with_sf > 1e-9 else float("inf")
    measured = {
        "mass_g": mp.mass_g,
        "com_lever_mm": rho_mm,
        "axis_gravity_sin": sin_ag,
        "tau_exact_kg_cm": tau_exact,
        "tau_worst_case_kg_cm": tau_worst,
        "tau_required_with_sf_kg_cm": required_with_sf,
        "margin_kg_cm": margin,
        "capacity_ratio": ratio,
    }
    if margin >= 0.0:
        msg = (
            f"torque OK: worst-case about-axis required {required_with_sf:.3f} kg·cm (mass {mp.mass_g:.1f} g, "
            f"lever {rho_mm:.1f} mm x sin {sin_ag:.2f}, SF {safety_factor}) <= continuous "
            f"{continuous_torque_kg_cm:.2f} kg·cm (margin {margin:.3f}, {ratio:.2f}x)"
        )
        return GeomVerdict(check_id, cat, "pass", msg, measured, limit)
    msg = (
        f"torque SHORT: worst-case about-axis required {required_with_sf:.3f} kg·cm (mass {mp.mass_g:.1f} g, "
        f"lever {rho_mm:.1f} mm x sin {sin_ag:.2f}, SF {safety_factor}) > continuous "
        f"{continuous_torque_kg_cm:.2f} kg·cm (margin {margin:.3f}) -- the servo cannot hold this limb"
    )
    return GeomVerdict(check_id, cat, "fail", msg, measured, limit)


def _unit_vec(v: Vec3) -> Vec3:
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return (v[0] / n, v[1] / n, v[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


__all__ = ["MassProps", "mass_properties", "worst_case_gravity_torque_margin"]
