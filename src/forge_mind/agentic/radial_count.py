"""Radial feature counting (ADR-0079): does the part carry the brief-named count of periodic
radial features (v1: gear teeth)?

Check ID ``agentic.features.radial_count``. ADVISORY / optimizer-loss only -- never a pass/fail
gate (confidence tier: estimated). The measurement is a uniform-theta polar RAY-CAST from the
detected rotation axis (tessellation-independent -- the geometry critic's finding: vertex binning
reads the TESSELLATOR, not the part), swept over candidate radial/axial bands, with an FFT peak
subject to a prominence floor AND an absolute ripple-amplitude floor (kills tessellation
micro-ripple reading as teeth).

The council's cardinal rule: the worst outcome is CONFIDENT-WRONG (it steers the optimizer), so
every doubt routes to ``not_run`` -- ambiguous axis, asymmetric part, low prominence, low
amplitude. And per the chairman's Goodhart resolution, ``not_run`` is NOT free when the brief
names a count: the loss charges 1.0 for "nothing countable" but < 1 for a detected-but-wrong
count, so erasing the features is always the most expensive move the optimizer can make.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forge_mind.agentic.profile_match import _surface_points

_MIN_COUNT = 4  # below this a "period" is indistinguishable from overall form
_MAX_COUNT = 400  # parse sanity ceiling
_MAX_BINS = 4096
_PROMINENCE_MIN = 8.0  # calibrated on spur_gear/toothed_rim ground truth (gear-tagged)
_AMPLITUDE_MODULE_FRAC = 0.1  # ripple must be >= this fraction of the implied module
_AXIS_DOMINANT = 1.25  # eigenvalue ratio that reads as a clear oblate/prolate split
_AXIS_DEGENERATE = 1.15  # the remaining pair must be at least this close to equal
_ASYM_MAX = 0.18  # median per-slice centroid offset / r_ref above this -> no axis claim
_Z_FRACTIONS = (0.15, 0.325, 0.5, 0.675, 0.85)  # candidate axial levels (interior, never faces)
_R_BANDS = ((0.85, 1.0), (0.6, 0.85), (0.35, 0.6))  # candidate radial bands, fractions of r_max

_METHOD = "polar-raycast-fft-v1"

# "120 teeth" / "120-tooth" / "teeth: 120" in a feature's name or dims text.
_TEETH_BEFORE = re.compile(r"(\d{1,3})\s*-?\s*(?:teeth|tooth)\b", re.IGNORECASE)
_TEETH_AFTER = re.compile(r"\b(?:teeth|tooth\s*count)\D{0,3}(\d{1,3})", re.IGNORECASE)


@dataclass(frozen=True)
class RadialCount:
    """One measurement: either a detected dominant angular period k_star, or a not_run reason.

    ``harmonics`` reports the spectral magnitudes at k*/2 and 2k* (the harmonic-promotion
    exploit is REPORTED, never forgiven in the loss)."""

    k_star: int | None
    not_run: str | None = None
    prominence: float | None = None
    amplitude_mm: float | None = None
    band: str | None = None
    harmonics: tuple[tuple[int, float], ...] = field(default=())
    method: str = _METHOD


def planned_radial_teeth(brief: Any) -> int | None:
    """The brief's single named tooth count, or None (no claim). Fail-closed: no count named,
    several DISTINCT counts named (one band cannot be attributed), or a count outside
    [4, 400] all mean the check never runs."""
    found: set[int] = set()
    for f in getattr(brief, "features", []):
        text = f"{f.name} {getattr(f, 'approx_dims_mm', '') or ''}"
        for rx in (_TEETH_BEFORE, _TEETH_AFTER):
            found.update(int(m) for m in rx.findall(text))
    if not found:
        text = f"{getattr(brief, 'object_name', '')} {getattr(brief, 'summary', '')}"
        for rx in (_TEETH_BEFORE, _TEETH_AFTER):
            found.update(int(m) for m in rx.findall(text))
    if len(found) != 1:
        return None
    n = found.pop()
    return n if _MIN_COUNT <= n <= _MAX_COUNT else None


def _rotation_axis(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | str:
    """(centroid, unit axis) by ASPECT RATIO -- the implementer critic's finding: a disc-like
    gear's rotation axis is the SMALLEST-spread eigenvector (profile_match's largest-spread
    convention is wrong for discs). Near-degenerate ordering -> a reason string (not_run)."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = (centered.T @ centered) / float(len(centered))
    vals, vecs = np.linalg.eigh(cov)  # ascending
    eps = 1e-12
    oblate = vals[1] / max(vals[0], eps) >= _AXIS_DOMINANT and (
        vals[2] / max(vals[1], eps) <= _AXIS_DEGENERATE
    )
    prolate = vals[2] / max(vals[1], eps) >= _AXIS_DOMINANT and (
        vals[1] / max(vals[0], eps) <= _AXIS_DEGENERATE
    )
    if oblate:
        axis = vecs[:, 0]
    elif prolate:
        axis = vecs[:, 2]
    else:
        return "no unambiguous rotation axis (eigenvalues near-degenerate)"
    t = centered @ axis
    perp = centered - np.outer(t, axis)
    r = np.linalg.norm(perp, axis=1)
    r_ref = float(np.quantile(r, 0.98))
    if r_ref <= eps:
        return "degenerate radial extent"
    edges = np.linspace(float(t.min()), float(t.max()), 9)
    offsets = []
    for i in range(8):
        mask = (t >= edges[i]) & (t <= edges[i + 1])
        if int(mask.sum()) >= 30:
            offsets.append(float(np.linalg.norm(perp[mask].mean(axis=0))))
    if offsets and float(np.median(offsets)) / r_ref > _ASYM_MAX:
        return "part is not axisymmetric about the detected axis"
    return centroid, axis / float(np.linalg.norm(axis))


def _ray_chords(
    mesh: Any,
    centroid: np.ndarray,
    axis: np.ndarray,
    t_span: tuple[float, float],
    n_bins: int,
    z_fractions: tuple[float, ...] = _Z_FRACTIONS,
    r_bands: tuple[tuple[float, float], ...] = _R_BANDS,
) -> tuple[np.ndarray, float] | None:
    """Material chord length per (level, theta) ray within each radial band: uniform-theta rays
    cast outward from on-axis origins, all crossings kept, inside intervals resolved by parity
    (odd hit count = the origin sits in material). Returns (chords[level, band, theta], r_max)."""
    tmp = np.array([1.0, 0.0, 0.0]) if abs(float(axis[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, tmp)
    u /= float(np.linalg.norm(u))
    v = np.cross(axis, u)
    theta = np.linspace(0.0, 2.0 * np.pi, n_bins, endpoint=False)
    dirs_plane = np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v)  # (n_bins, 3)
    t_lo, t_hi = t_span
    levels = [t_lo + f * (t_hi - t_lo) for f in z_fractions]
    n_lv = len(levels)
    origins = np.repeat(
        np.asarray([centroid + t * axis for t in levels], dtype=np.float64), n_bins, axis=0
    )
    dirs = np.tile(dirs_plane, (n_lv, 1))
    try:
        locs, ray_idx, _tri = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
    except Exception:
        return None
    if len(ray_idx) == 0:
        return None
    dist = np.einsum("ij,ij->i", locs - origins[ray_idx], dirs[ray_idx])
    per_ray: list[list[float]] = [[] for _ in range(n_lv * n_bins)]
    for i, d in zip(ray_idx, dist, strict=True):
        if d > 1e-9:
            per_ray[int(i)].append(float(d))
    r_max = float(dist.max())
    if r_max <= 1e-9:
        return None
    tol = 1e-6 * r_max
    chords = np.zeros((n_lv, len(r_bands), n_bins), dtype=np.float64)
    bands_abs = [(lo * r_max, hi * r_max) for lo, hi in r_bands]
    for ray in range(n_lv * n_bins):
        ts = sorted(per_ray[ray])
        deduped: list[float] = []
        for d in ts:
            if not deduped or d - deduped[-1] > tol:
                deduped.append(d)
        if len(deduped) % 2 == 1:
            deduped.insert(0, 0.0)  # origin inside material
        lv, th = divmod(ray, n_bins)
        for b, (lo, hi) in enumerate(bands_abs):
            total = 0.0
            for a, z in zip(deduped[0::2], deduped[1::2], strict=True):
                total += max(0.0, min(z, hi) - max(a, lo))
            chords[lv, b, th] = total
    return chords, r_max


def _spectral_peak(signal: np.ndarray) -> tuple[int, float, float] | None:
    """(k_star, prominence, peak-to-peak amplitude) of the dominant angular period, or None when
    the band carries no material / no periodic energy."""
    if float(signal.max()) <= 1e-9:
        return None
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    if len(spectrum) <= _MIN_COUNT + 1:
        return None
    ks = np.arange(_MIN_COUNT, len(spectrum))
    k_star = int(ks[int(np.argmax(spectrum[_MIN_COUNT:]))])
    floor = max(float(np.median(spectrum[_MIN_COUNT:])), 1e-12)
    prominence = min(float(spectrum[k_star]) / floor, 1e6)
    amplitude = 4.0 * float(spectrum[k_star]) / float(len(signal))
    return k_star, prominence, amplitude


def measure_radial_count(mesh: Any, n_expected: int) -> RadialCount:
    """Measure the dominant angular period of the part's strongest periodic radial band.

    ``n_expected`` sizes the angular resolution ONLY (>=16 bins per expected feature); detection
    never conditions on it -- the winning band is chosen by spectral prominence alone, so the
    measurement cannot cherry-pick a band that flatters the brief."""
    if n_expected < _MIN_COUNT:
        return RadialCount(k_star=None, not_run=f"expected count {n_expected} below {_MIN_COUNT}")
    if not bool(getattr(mesh, "is_watertight", False)):
        return RadialCount(k_star=None, not_run="mesh is not watertight")
    pts = _surface_points(mesh)
    if pts is None:
        return RadialCount(k_star=None, not_run="surface could not be sampled")
    axis_or_reason = _rotation_axis(pts)
    if isinstance(axis_or_reason, str):
        return RadialCount(k_star=None, not_run=axis_or_reason)
    centroid, axis = axis_or_reason
    t = (pts - centroid) @ axis
    n_bins = min(_MAX_BINS, 1 << int(np.ceil(np.log2(max(512, 16 * n_expected)))))
    cast = _ray_chords(mesh, centroid, axis, (float(t.min()), float(t.max())), n_bins)
    if cast is None:
        return RadialCount(k_star=None, not_run="ray cast produced no usable crossings")
    chords, r_max = cast
    best: tuple[float, int, float, str, np.ndarray] | None = None  # (prom, k, amp, band, signal)
    candidates = [(chords[lv], f"z{_Z_FRACTIONS[lv]:.2f}") for lv in range(chords.shape[0])]
    candidates.append((chords.mean(axis=0), "z-mean"))
    for band_chords, z_name in candidates:
        for b, (lo, hi) in enumerate(_R_BANDS):
            peak = _spectral_peak(band_chords[b])
            if peak is None:
                continue
            k_star, prominence, amplitude = peak
            implied_module = 2.0 * (hi * r_max) / float(k_star)
            if prominence < _PROMINENCE_MIN:
                continue
            if amplitude < _AMPLITUDE_MODULE_FRAC * implied_module:
                continue
            if best is None or prominence > best[0]:
                best = (
                    prominence,
                    k_star,
                    amplitude,
                    f"r[{lo:.2f},{hi:.2f}] {z_name}",
                    band_chords[b],
                )
    if best is None:
        return RadialCount(
            k_star=None, not_run="no periodic band met the prominence/amplitude floors"
        )
    prominence, k_star, amplitude, band, signal = best
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    harmonics: list[tuple[int, float]] = []
    for hk in (k_star // 2, 2 * k_star):
        if _MIN_COUNT <= hk < len(spectrum) and hk != k_star:
            harmonics.append((hk, float(spectrum[hk])))
    return RadialCount(
        k_star=k_star,
        prominence=prominence,
        amplitude_mm=amplitude,
        band=band,
        harmonics=tuple(harmonics),
    )


@dataclass(frozen=True)
class RadialLock:
    """The measurement frame of the BASELINE part's winning band, held fixed across a
    parametric fit (engine research idea 2, the balance-axis precedent: dimension changes do
    not move a gear's axis in the authored frame). Locking kills band-hopping between
    candidates (measurement noise the optimizer could exploit) and cuts the per-candidate cost
    ~6-10x (one band, its own z-levels, no axis detection). Chosen from the baseline by
    prominence alone -- never by matching the brief's count -- and the ACCEPTANCE measurement
    always re-runs the full unlocked sweep, so a fit that somehow migrated the teeth is still
    caught."""

    centroid: tuple[float, float, float]
    axis: tuple[float, float, float]
    r_band: tuple[float, float]  # fractions of the candidate's own r_max
    z_fractions: tuple[float, ...]  # the level(s) behind the winning band


def lock_from_baseline(mesh: Any, n_expected: int) -> RadialLock | None:
    """Measure the baseline once (full sweep) and freeze its frame. None when the baseline has
    no confident band -- the fit then measures each candidate the full way (no lock is better
    than a lock on noise)."""
    rc = measure_radial_count(mesh, n_expected)
    if rc.k_star is None or rc.band is None:
        return None
    pts = _surface_points(mesh)
    axis_or_reason = _rotation_axis(pts) if pts is not None else "unsampleable"
    if isinstance(axis_or_reason, str):
        return None
    centroid, axis = axis_or_reason
    # band string: "r[0.85,1.00] z0.15" or "r[0.85,1.00] z-mean"
    try:
        r_part, z_part = rc.band.split(" ")
        lo, hi = (float(x) for x in r_part[2:-1].split(","))
        z_fracs = _Z_FRACTIONS if z_part == "z-mean" else (float(z_part[1:]),)
    except Exception:
        return None
    return RadialLock(
        centroid=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        axis=(float(axis[0]), float(axis[1]), float(axis[2])),
        r_band=(lo, hi),
        z_fractions=tuple(z_fracs),
    )


def measure_radial_count_locked(mesh: Any, n_expected: int, lock: RadialLock) -> RadialCount:
    """Measure ONE candidate against the locked baseline frame: no sampling, no axis
    detection, no band sweep -- rays at the locked band's levels only, one FFT. The floors
    (prominence, amplitude) still apply: a candidate whose teeth vanished still reads not_run
    (deficit 1.0), so the suppression attractor stays dead."""
    if not bool(getattr(mesh, "is_watertight", False)):
        return RadialCount(k_star=None, not_run="mesh is not watertight")
    centroid = np.asarray(lock.centroid, dtype=np.float64)
    axis = np.asarray(lock.axis, dtype=np.float64)
    corners = np.asarray(mesh.bounds, dtype=np.float64)  # (2,3) min/max
    grid = np.array(
        [
            [corners[i, 0], corners[j, 1], corners[k, 2]]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ]
    )
    t = (grid - centroid) @ axis
    n_bins = min(_MAX_BINS, 1 << int(np.ceil(np.log2(max(512, 16 * n_expected)))))
    cast = _ray_chords(
        mesh,
        centroid,
        axis,
        (float(t.min()), float(t.max())),
        n_bins,
        z_fractions=lock.z_fractions,
        r_bands=(lock.r_band,),
    )
    if cast is None:
        return RadialCount(k_star=None, not_run="ray cast produced no usable crossings")
    chords, r_max = cast
    signal = chords.mean(axis=0)[0]
    peak = _spectral_peak(signal)
    if peak is None:
        return RadialCount(k_star=None, not_run="locked band carries no periodic energy")
    k_star, prominence, amplitude = peak
    implied_module = 2.0 * (lock.r_band[1] * r_max) / float(k_star)
    if prominence < _PROMINENCE_MIN or amplitude < _AMPLITUDE_MODULE_FRAC * implied_module:
        return RadialCount(
            k_star=None, not_run="locked band peak below the prominence/amplitude floors"
        )
    band = f"r[{lock.r_band[0]:.2f},{lock.r_band[1]:.2f}] locked"
    return RadialCount(k_star=k_star, prominence=prominence, amplitude_mm=amplitude, band=band)


def radial_deficit(planned: int, measured: RadialCount) -> float:
    """The chairman's two-level loss: not_run costs 1.0 when a count is NAMED (erasing the
    features is never the cheap move), a detected-but-wrong count costs min(1, |k*-N|/N) < 1
    (countable features always beat erased ones), an exact match costs 0."""
    if measured.k_star is None:
        return 1.0
    return min(1.0, abs(measured.k_star - planned) / float(planned))


def radial_count_feedback(mesh: Any, brief: Any) -> list[str]:
    """Advisory revision feedback (the measured_hole_feedback pattern, radial edition): when the
    brief names a tooth count, tell the author what the polar spectrum measured. Composition is
    STRUCTURE -- the author, not the optimizer, must add a dropped tooth band. [] when the brief
    names no count or the measurement matches."""
    planned = planned_radial_teeth(brief)
    if planned is None:
        return []
    try:
        rc = measure_radial_count(mesh, planned)
    except Exception:
        return []
    if rc.k_star == planned:
        return []
    if rc.k_star is None:
        return [
            f"RADIAL FEATURES: the brief names {planned} teeth, but no confident periodic tooth "
            f"pattern was measured ({rc.not_run}). If the toothed band is missing from the code, "
            f"add it -- `toothed_rim(rim_d, {planned}, width)` fuses an external ring gear onto a "
            f"disc rim; `spur_gear(module, {planned}, thickness)` builds a full gear. If the teeth "
            "are present, this may be a measurement limit (advisory)."
        ]
    return [
        f"RADIAL FEATURES: the brief names {planned} teeth; the measured dominant angular period "
        f"is {rc.k_star} (prominence {rc.prominence:.1f}, band {rc.band}). Check the tooth-count "
        "value in the code against the brief."
    ]


__all__ = [
    "RadialCount",
    "RadialLock",
    "lock_from_baseline",
    "measure_radial_count",
    "measure_radial_count_locked",
    "planned_radial_teeth",
    "radial_count_feedback",
    "radial_deficit",
]
