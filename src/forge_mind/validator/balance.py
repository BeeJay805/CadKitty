"""Rotational balance -- a general functional check for any spinning part.

This is the first vertebra of the general functional-validation spine: it encodes a
mechanical PRINCIPLE (a rotor must spin true) as a domain-general, fail-closed check
that applies to ANY part the general engine produces -- a crankshaft, a flywheel, a
gear, a fan impeller -- with no per-family template.

A rotor spins true only if two independent conditions hold, both read from the geometry:
  - STATIC balance: the center of mass lies ON the rotation axis. Otherwise a net
    centrifugal FORCE (proportional to the CoM's eccentricity) shakes the bearings.
  - DYNAMIC balance: the rotation axis is a PRINCIPAL axis of inertia -- the products
    of inertia that couple the axis to the transverse plane vanish. Otherwise a
    centrifugal COUPLE rocks the part even when the CoM is perfectly on-axis.

We report both as dimensionless ratios so one tolerance works across any size:
  - ``static_ratio`` = CoM-to-axis offset / the part's characteristic radius
  - ``dynamic_ratio`` = |axis-coupling products of inertia| / the polar moment

Scope + honesty: this certifies GEOMETRIC balance -- was the part DESIGNED to spin
true (does the crankshaft carry counterweights; is the rotor not grossly asymmetric).
It does NOT certify fab-grade balance (surface finish, material density variation,
real machining tolerance) -- that is a physical-part claim the engine must keep
disclaiming. It FAILS CLOSED: if the mesh is not a trustworthy solid or the axis is
degenerate, the verdict is ``None`` (UNVERIFIED), never a PASS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh.geometry import align_vectors

_EPS = 1e-9
_Z_AXIS = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class BalanceReport:
    """Verdict + measured ratios for a rotor about one axis.

    ``balanced``: True = certified geometrically balanced, False = unbalanced,
    None = UNVERIFIED (the inputs could not be trusted -- fail closed, never a PASS).
    """

    balanced: bool | None
    static_offset_mm: float
    static_ratio: float
    dynamic_ratio: float
    note: str


def rotational_balance(
    mesh: trimesh.Trimesh,
    axis_point: tuple[float, float, float],
    axis_dir: tuple[float, float, float],
    *,
    static_tol: float = 0.03,
    dynamic_tol: float = 0.05,
) -> BalanceReport:
    """Certify whether ``mesh`` is balanced to spin about the axis through ``axis_point``
    along ``axis_dir``. See the module docstring for the physics and the scope. Pure and
    deterministic; never raises -- an untrustworthy input returns ``balanced=None``.
    """
    direction = np.asarray(axis_dir, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < _EPS:
        return BalanceReport(None, 0.0, 0.0, 0.0, "UNVERIFIED: degenerate rotation axis")
    axis = direction / norm

    # Mass properties are only trustworthy on a closed solid; fail closed otherwise.
    if not bool(getattr(mesh, "is_watertight", False)) or float(mesh.volume) <= _EPS:
        return BalanceReport(None, 0.0, 0.0, 0.0, "UNVERIFIED: mesh is not a watertight solid")

    com = np.asarray(mesh.center_mass, dtype=np.float64)
    delta = com - np.asarray(axis_point, dtype=np.float64)
    perpendicular = delta - float(np.dot(delta, axis)) * axis
    static_offset = float(np.linalg.norm(perpendicular))

    char_radius = float(mesh.bounding_sphere.primitive.radius)
    if char_radius <= _EPS:
        return BalanceReport(None, static_offset, 0.0, 0.0, "UNVERIFIED: degenerate extent")
    static_ratio = static_offset / char_radius

    # Inertia tensor about the CoM, rotated into a frame whose z-axis is the spin axis.
    inertia = np.asarray(mesh.moment_inertia, dtype=np.float64)
    transform = align_vectors(axis, _Z_AXIS)  # type: ignore[no-untyped-call]  # trimesh is unstubbed
    rot = np.asarray(transform[:3, :3], dtype=np.float64)
    aligned = rot @ inertia @ rot.T
    polar = float(aligned[2, 2])
    if polar <= _EPS:
        return BalanceReport(
            None, static_offset, static_ratio, 0.0, "UNVERIFIED: degenerate inertia about the axis"
        )
    dynamic_ratio = float(np.hypot(aligned[0, 2], aligned[1, 2]) / polar)

    balanced = static_ratio <= static_tol and dynamic_ratio <= dynamic_tol
    note = (
        "balanced: CoM on-axis and the axis is a principal axis (geometric balance)"
        if balanced
        else (
            f"UNBALANCED: static_ratio={static_ratio:.3f} (tol {static_tol}), "
            f"dynamic_ratio={dynamic_ratio:.3f} (tol {dynamic_tol})"
        )
    )
    return BalanceReport(balanced, static_offset, static_ratio, dynamic_ratio, note)


# NOTE on axis sourcing (why there is no autonomous "infer the axis from the mesh"):
# balance is only meaningful about an axis defined INDEPENDENTLY of the mass -- the part's
# BORE / shaft, or an axis declared by intent. Inferring the axis from the mass's own
# PRINCIPAL axes is a tautology: a principal axis through the center of mass makes BOTH
# conditions identically zero (it diagonalizes the inertia tensor, so dynamic == 0, and it
# passes through the CoM, so static == 0). Such an "autonomous" check would return
# balanced=True for every rotor -- including a disc with an off-center bore, which is
# genuinely unbalanced about the axis it actually spins on. So the caller MUST supply the
# real rotation axis (the concave bore's axis via validator.geometry.extract_cylinders on
# the B-rep, or the intent-declared axis). Fail closed: no independent axis -> do not certify.


__all__ = ["BalanceReport", "rotational_balance"]
