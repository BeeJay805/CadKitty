"""Discriminating shape signature: the reward-signal seed the D2 score is too coarse to be.

The measured problem (2026-07-01 breadth panel): the D2 ``shape_similarity`` saturates at ~0.85-0.95
for every part, and the vision critic mislabels (it scored a clean 5-blade impeller "weak"). Neither
can tell "right form" from "wrong form" reliably, so no generator lever can be measured or tuned.
D2's own docstring admits the gap: it captures overall proportion but "NOT fine features (a box and
a sphere score similar)".

This adds a small vector of pose- and scale-invariant descriptors that capture exactly what D2
misses, and reports the PER-DESCRIPTOR deltas between a build and a reference so the signal is
LOCALIZED (which aspect of the form is wrong), not a single opaque scalar:

  * ``axes`` -- the normalized principal-spread triple (e1>=e2>=e3, sum 1) from the covariance of
    surface points. Rotation- and scale-invariant, computable on ANY mesh (no watertight needed).
    Captures elongation (rod -> ~(1,0,0)), flatness (plate -> ~(.5,.5,0)), and AXIAL ASYMMETRY --
    a symmetric disc-stack is axisymmetric (e2~=e3) but a phased crankshaft with counterweights is
    not (e2>e3). That asymmetry is precisely the crankshaft "too-regular stack" failure D2 cannot see.
  * ``concavity`` -- the fraction of surface points that lie in a CONCAVITY (strictly interior to
    the convex hull, beyond a small tolerance). 0 for a convex solid (box/sphere/cylinder), rising
    with crescents, pockets, and the gaps between blades/webs. Computed from surface points + the
    hull of the vertices, so it needs NO watertight mesh -- it is the concavity signal that survives
    on the non-watertight artist references (where the exact ``solidity`` below is unavailable). This
    is the descriptor the eval found missing: the crankshaft's crescent counterweights vs a plain
    disc-stack are a concavity difference, and the Sketchfab reference is never watertight.
  * ``solidity`` -- mesh volume / convex-hull volume, in (0, 1]. The EXACT concavity measure, but it
    requires a watertight mesh (the sandbox build always is); ``None`` for a non-watertight artist
    reference (fail-closed). Kept for build-vs-build and watertight-reference comparisons.

ADVISORY, never a gate (no-false-PASS: an advisory signal cannot certify). It is a richer,
localized companion to the D2 score, and the metric the eval + author feedback should lean on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

_N_POINTS = 4096  # surface samples (matches the D2 metric's budget)
_CONCAVITY_TAU = (
    0.02  # a surface point deeper than 2% of the hull scale inside the hull is "concave"
)


@dataclass(frozen=True)
class ShapeSignature:
    """Pose/scale-invariant descriptors of one mesh. Each field is ``None`` when uncomputable."""

    axes: tuple[float, float, float] | None  # normalized principal spread e1>=e2>=e3, sum 1
    concavity: float | None  # fraction of surface points interior to the hull; 0=convex (any mesh)
    solidity: float | None  # volume / convex-hull volume in (0, 1]; None if not watertight

    @property
    def asymmetry(self) -> float | None:
        """|e2 - e3| -- how far the form is from axisymmetric about its long axis (0 = axisymmetric)."""
        if self.axes is None:
            return None
        return abs(self.axes[1] - self.axes[2])


def _surface_points(mesh: Any) -> np.ndarray | None:
    """Deterministic area-weighted surface sample, restoring numpy's global RNG (as the D2 metric does)."""
    rng_state = np.random.get_state()
    try:
        np.random.seed(0)
        pts = np.asarray(mesh.sample(_N_POINTS), dtype=np.float64)
    except Exception:
        return None
    finally:
        np.random.set_state(rng_state)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
        return None
    return pts


def _principal_axes(points: np.ndarray) -> tuple[float, float, float] | None:
    """Normalized eigenvalues (desc, sum 1) of the point covariance -- the pose/scale-invariant
    spread along the three principal axes. None if degenerate."""
    cov = np.cov(points.T)
    if cov.shape != (3, 3):
        return None
    vals = np.linalg.eigvalsh(cov)  # ascending, real (cov is symmetric PSD)
    vals = np.clip(vals[::-1], 0.0, None)  # descending, no tiny-negative noise
    total = float(vals.sum())
    if total <= 1e-12:
        return None
    e = vals / total
    return (float(e[0]), float(e[1]), float(e[2]))


def _solidity(mesh: Any) -> float | None:
    """Mesh volume / convex-hull volume in (0, 1]. None unless the mesh is watertight with a
    positive hull volume (a non-watertight artist mesh has an unreliable signed volume)."""
    try:
        if not bool(mesh.is_watertight):
            return None
        vol = float(mesh.volume)
        hull_vol = float(mesh.convex_hull.volume)
    except Exception:
        return None
    if not np.isfinite(vol) or not np.isfinite(hull_vol) or hull_vol <= 1e-12 or vol <= 0.0:
        return None
    return min(vol / hull_vol, 1.0)


def _concavity(mesh: Any, points: np.ndarray) -> float | None:
    """Fraction of surface points that lie in a CONCAVITY -- strictly interior to the convex hull
    by more than ``_CONCAVITY_TAU`` of the hull scale. 0 for a convex solid (box/sphere/cylinder),
    rising with crescents/pockets/blade gaps. Uses only the sampled points and the hull of the mesh
    vertices (always valid), so unlike ``_solidity`` it needs NO watertight mesh -- the concavity
    signal that survives on a non-watertight artist reference. None if the hull is degenerate."""
    try:
        hull = mesh.convex_hull
        scale = float(hull.scale)
        if not np.isfinite(scale) or scale <= 1e-9:
            return None
        from trimesh.proximity import closest_point

        _, dist, _ = closest_point(hull, points)  # type: ignore[no-untyped-call]  # dist to hull
        return float(np.mean(np.asarray(dist, dtype=np.float64) > _CONCAVITY_TAU * scale))
    except Exception:
        return None


def signature_of(mesh_path: Path) -> ShapeSignature | None:
    """Compute the shape signature of a mesh file, or None if it cannot be loaded/sampled
    (fail-closed, never raises)."""
    try:
        mesh: Any = trimesh.load(str(mesh_path), force="mesh")
    except Exception:
        return None
    points = _surface_points(mesh)
    if points is None:
        return None
    return ShapeSignature(
        axes=_principal_axes(points),
        concavity=_concavity(mesh, points),
        solidity=_solidity(mesh),
    )


def signature_deltas(build: Path, reference: Path) -> dict[str, float] | None:
    """Per-descriptor absolute deltas between the build and reference signatures. Only descriptors
    BOTH meshes could compute are included. None if either signature is unavailable."""
    a = signature_of(build)
    b = signature_of(reference)
    if a is None or b is None:
        return None
    out: dict[str, float] = {}
    if a.axes is not None and b.axes is not None:
        out["axes_l1"] = float(sum(abs(x - y) for x, y in zip(a.axes, b.axes, strict=True)))
        if a.asymmetry is not None and b.asymmetry is not None:
            out["asymmetry_delta"] = abs(a.asymmetry - b.asymmetry)
    if a.concavity is not None and b.concavity is not None:
        out["concavity_delta"] = abs(a.concavity - b.concavity)
    if a.solidity is not None and b.solidity is not None:
        out["solidity_delta"] = abs(a.solidity - b.solidity)
    return out or None


def conformance_advisory(build: Path, reference: Path) -> str | None:
    """A localized, DIRECTIONAL author hint from the build-vs-reference signature, or None when
    nothing differs enough to act on (or a signature is unavailable). Report-only: this is the
    measured, per-descriptor signal that the coarse D2 scalar cannot give -- it names WHICH aspect
    of the form is wrong and WHICH WAY to move it."""
    a = signature_of(build)
    b = signature_of(reference)
    if a is None or b is None:
        return None
    notes: list[str] = []
    # Concavity: prefer the always-available point-cloud measure (works on non-watertight artist
    # references, where the eval found the exact solidity dropping out); fall back to solidity.
    if (
        a.concavity is not None
        and b.concavity is not None
        and abs(a.concavity - b.concavity) > 0.08
    ):
        if b.concavity > a.concavity:  # the reference carves away more than the build
            notes.append(
                f"CONCAVITY: your build is too CONVEX/solid ({a.concavity:.0%} of its surface sits "
                f"in concavities vs the reference's {b.concavity:.0%}) -- the real part removes more "
                "material. Add the concave features it has (pockets, crescents, hollows, the gaps "
                "between blades/webs/spokes)."
            )
        else:
            notes.append(
                f"CONCAVITY: your build is more carved-out than the reference ({a.concavity:.0%} vs "
                f"{b.concavity:.0%}) -- add material back to match its more solid form."
            )
    if a.asymmetry is not None and b.asymmetry is not None and b.asymmetry - a.asymmetry > 0.05:
        notes.append(
            "ASYMMETRY: your build is more axially SYMMETRIC than the reference -- the reference "
            "carries off-axis mass (counterweights, lobes, an eccentric feature) your build lacks. "
            "Break the symmetry to match it."
        )
    if not notes:
        return None
    return "SHAPE SIGNATURE (measured vs the reference):\n" + "\n".join(f"  - {n}" for n in notes)


__all__ = ["ShapeSignature", "conformance_advisory", "signature_deltas", "signature_of"]
