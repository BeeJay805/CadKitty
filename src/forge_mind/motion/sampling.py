"""Shared swept-collision machinery (ADR-0086): seeded interior sampling + depth-confirmed
containment + rotation math. One implementation for the pair and mechanism sweeps, with the
same honesty constants as the assembly gate (``validator/motion_sweep.py``): sampling is
seeded (deterministic verdicts), a backend failure counts as overlap (cannot certify -> never
silently clear), and containment must be DEEPER than tessellation noise to count.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

_SEED = 0
_POINTS_PER_PART = 800
_MIN_OVERLAP_POINTS = 2
_MIN_DEPTH_MM = 0.15
# Poses per full turn. 15-degree steps: fine enough to catch any feature that matters at
# upload scale, cheap enough to run on every report. Public: the GLB bridge reports the
# EARNED verification level as "poses:<N>" (the ck_joint contract's honesty field).
SWEEP_POSES = 24


def interior_points(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """Seeded interior samples; None = unsampleable (fail CLOSED, never silently clear)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(_SEED)
        pts = np.asarray(trimesh.sample.volume_mesh(mesh, _POINTS_PER_PART), dtype=float)
        return pts if len(pts) > 0 else None
    except Exception:
        return None
    finally:
        np.random.set_state(rng_state)


def rotation_about_line(origin: np.ndarray, direction: np.ndarray, deg: float) -> np.ndarray:
    d = direction / float(np.linalg.norm(direction))
    a = math.radians(deg)
    k = np.array([[0.0, -d[2], d[1]], [d[2], 0.0, -d[0]], [-d[1], d[0], 0.0]])
    rot = np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * (k @ k)
    m = np.eye(4)
    m[:3, :3] = rot
    m[:3, 3] = origin - rot @ origin
    return m


def deep_overlap(pts: np.ndarray, solid: trimesh.Trimesh) -> bool:
    """True when >_MIN_OVERLAP_POINTS of ``pts`` sit inside ``solid`` deeper than noise.
    A backend failure counts as overlap (cannot certify -> never silently clear)."""
    if len(pts) == 0:
        return False
    try:
        inside = solid.contains(pts)
    except Exception:
        return True
    if int(np.count_nonzero(inside)) <= _MIN_OVERLAP_POINTS:
        return False
    try:
        sd = trimesh.proximity.signed_distance(solid, pts[inside])  # type: ignore[no-untyped-call]
    except Exception:
        return True
    return int(np.count_nonzero(np.asarray(sd) > _MIN_DEPTH_MM)) > _MIN_OVERLAP_POINTS


def posed(pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return np.asarray(trimesh.transform_points(pts, transform), dtype=float)


def radial_distances(
    mesh: trimesh.Trimesh, origin: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """Each vertex's perpendicular distance from the axis line."""
    d = direction / float(np.linalg.norm(direction))
    rel = np.asarray(mesh.vertices, dtype=float) - origin
    along = rel @ d
    perp = rel - np.outer(along, d)
    return np.asarray(np.linalg.norm(perp, axis=1), dtype=float)


# AABB margin (mm) for the swept-collision pre-cull: two bodies whose axis-aligned boxes are
# apart by more than this cannot interpenetrate at that pose, so the expensive containment is
# skipped. Well above the _MIN_DEPTH_MM contact threshold, so a real near-touch is never culled.
_CULL_MARGIN_MM = 1.0


def aabb(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(pts.min(axis=0)), np.asarray(pts.max(axis=0))


def aabb_disjoint(a_lo: np.ndarray, a_hi: np.ndarray, b_lo: np.ndarray, b_hi: np.ndarray) -> bool:
    """True when the two AABBs are separated (with margin) on any axis — cannot collide."""
    return bool((a_hi + _CULL_MARGIN_MM < b_lo).any() or (b_hi + _CULL_MARGIN_MM < a_lo).any())


__all__ = [
    "SWEEP_POSES",
    "aabb",
    "aabb_disjoint",
    "deep_overlap",
    "interior_points",
    "posed",
    "radial_distances",
    "rotation_about_line",
]
