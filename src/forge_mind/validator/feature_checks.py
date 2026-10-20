"""Tier 2: feature checks. Measures wall thickness from the mesh by ray casting.

Wall thickness is a real "won't print / too weak" failure that cheap checks
miss. We measure it generally: sample points across the surface, step slightly
inward, cast a ray along the inward normal, and record the distance to the
first exit. The minimum across samples is the thinnest wall.

This is general (catches hole-to-edge webs, not just plate thickness) at the
cost of being sampling-based. We guard against edge/numerical noise with an
inward offset and a floor on accepted distances.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from forge_mind.dsl.program import PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.report import Check, CheckStatus, Fix, Locator, Tier

# Sampling / ray-cast tuning.
N_SAMPLES = 3000
INWARD_OFFSET_MM = 0.05  # step inside the surface before casting to dodge the origin face
NOISE_FLOOR_MM = 0.3  # ignore hits closer than this (sharp-edge / coplanar artifacts)

# A measured thickness within this of the requirement passes. Absorbs mesh
# tessellation + sampling jitter and matches real FDM dimensional tolerance.
WALL_MEASUREMENT_TOL_MM = 0.2

# sample_surface draws from the GLOBAL numpy RNG, so an unseeded validate() flickers run-to-run on
# a wall sitting exactly at min_wall. We seed it to a FIXED value around the sample (state saved +
# restored, so no RNG state leaks to the caller) -> a standalone validate() is reproducible. NOTE:
# min_wall is a SAMPLED, lower-confidence check (a sampled estimate), not an exact measurement. The
# library/composition builders OVERRIDE this seed across K values for their robust multi-seed keep
# gate; a single fixed seed here is only per-call determinism.
MIN_WALL_SAMPLE_SEED = 0

# The AREA-significance wall measurement, shared by the validator's wall GATE and the author
# self-check epilogue. Extracted (2026-07-27) so the two never drift: the self-check tells the
# author "your own measurement", and that number must be the SAME number the gate later reads
# -- the doctrine's "when the same limit exists in two places, derive one from the other",
# satisfied structurally by having exactly one measurement.
WALL_GATE_SAMPLES = 4096
WALL_GATE_SEED = 20260707
# A WALL is two near-antiparallel faces close together; an oblique-edge sliver (a 45-degree
# chamfer strip, a cone rim, a gusset junction) exits through an ANGLED face. Requiring
# dot(entry normal, exit normal) < -0.8 (~37 degrees of draft allowed) counts only true
# walls -- measured on a cone: rim slivers were 11% of surface by area, 0% by this filter.
WALL_GATE_ANTIPARALLEL = -0.8
WALL_GATE_TOL_MM = 0.2  # matches WALL_MEASUREMENT_TOL_MM: same jitter, same hysteresis
# The area fraction at which a sub-floor wall GATES (ADR-0012 E1-2: seeded sub-floor mutants
# measured 65-79% thin, a knife edge well under 1%). Lives HERE, beside the measurement, so
# the validator's verdict and the author's self-check feedback cannot disagree about where the
# line is -- review-caught: a 2.86% fraction rendered as "3% ... FAILS" while the gate passed it.
WALL_GATE_MIN_FRAC = 0.03
_WALL_GATE_MIN_MEASURED = 100  # below this, no area fraction may be claimed
WALL_GATE_MIN_THIN_SAMPLES = 3
# The hard gate needs to see membranes below the advisory instrument's 0.3 mm
# numerical-noise band. Self-hits are still rejected by the opposing-normal test.
WALL_GATE_NOISE_FLOOR_MM = 0.01


@dataclass(frozen=True)
class WallMeasurement:
    """Area-weighted evidence from the deterministic wall ray sample."""

    thin_fraction: float
    thinnest_mm: float | None
    thin_area_mm2: float
    thin_sample_count: int
    valid_sample_count: int


def wall_gate_triggered(
    thin_fraction: float,
    thin_area_mm2: float,
    thin_sample_count: int,
    min_area_mm2: float,
) -> bool:
    """Shared verdict: broad thin walls or a physically significant local membrane."""
    return thin_fraction >= WALL_GATE_MIN_FRAC or (
        thin_sample_count >= WALL_GATE_MIN_THIN_SAMPLES and thin_area_mm2 >= min_area_mm2
    )


def measure_wall_fraction(mesh: trimesh.Trimesh, floor_mm: float) -> WallMeasurement | None:
    """Area-weighted sub-floor WALL evidence for one mesh, or None when the
    measurement cannot be claimed (too few valid samples / any measurement trouble).

    Deterministic: the global RNG is seeded to ``WALL_GATE_SEED`` for the sample and restored
    afterwards, so two calls on the same mesh return identical numbers. ``thinnest`` is None
    when nothing measured sub-floor (fraction 0.0). Never raises."""
    if floor_mm <= 0:
        return None
    try:
        rng_state = np.random.get_state()
        try:
            np.random.seed(WALL_GATE_SEED)
            samples, face_idx = trimesh.sample.sample_surface(mesh, WALL_GATE_SAMPLES)
        finally:
            np.random.set_state(rng_state)
        normals = mesh.face_normals[face_idx]
        origins = samples - normals * INWARD_OFFSET_MM
        locations, ray_idx, tri_idx = mesh.ray.intersects_location(
            ray_origins=origins, ray_directions=-normals, multiple_hits=False
        )
        if len(locations) == 0:
            return None
        dists = np.linalg.norm(locations - origins[ray_idx], axis=1)
        keep = dists > WALL_GATE_NOISE_FLOOR_MM
        if int(keep.sum()) < _WALL_GATE_MIN_MEASURED:
            return None
        thickness = dists[keep] + INWARD_OFFSET_MM
        entry_n = normals[ray_idx][keep]
        exit_n = mesh.face_normals[tri_idx][keep]
        is_wall = np.einsum("ij,ij->i", entry_n, exit_n) < WALL_GATE_ANTIPARALLEL
        thin = (thickness < (floor_mm - WALL_GATE_TOL_MM)) & is_wall
        thin_count = int(thin.sum())
        frac = float(thin.mean())
        thinnest = float(thickness[thin].min()) if bool(thin.any()) else None
        return WallMeasurement(
            thin_fraction=frac,
            thinnest_mm=thinnest,
            thin_area_mm2=float(mesh.area) * frac,
            thin_sample_count=thin_count,
            valid_sample_count=int(keep.sum()),
        )
    except Exception:
        return None


def _measure_min_wall(mesh: trimesh.Trimesh) -> tuple[float, Vec3] | None:
    """Return ``(minimum wall thickness in mm, the surface point of that thinnest sample)``, or None
    if not measurable. The point is the locus the viewer marks (Z-up STL frame)."""
    # Seed the global RNG to a fixed value JUST for the sample, then restore it, so the measurement
    # is reproducible without leaking RNG state (sample_surface is the only RNG consumer here).
    rng_state = np.random.get_state()
    try:
        np.random.seed(MIN_WALL_SAMPLE_SEED)
        samples, face_idx = trimesh.sample.sample_surface(mesh, N_SAMPLES)
    finally:
        np.random.set_state(rng_state)
    normals = mesh.face_normals[face_idx]

    # Cast inward: origin just inside the surface, direction = -outward normal.
    origins = samples - normals * INWARD_OFFSET_MM
    directions = -normals

    locations, ray_idx, _tri = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )
    if len(locations) == 0:
        return None

    dists = np.linalg.norm(locations - origins[ray_idx], axis=1)
    keep = dists > NOISE_FLOOR_MM
    if not keep.any():
        return None
    kept_dists = dists[keep]
    # The thinnest accepted sample: recover its ORIGINAL surface point (samples[ray_idx][argmin]) so
    # the viewer can mark exactly where the wall is thin.
    thinnest = int(np.argmin(kept_dists))
    surface = samples[ray_idx[keep][thinnest]]
    point = Vec3(x=float(surface[0]), y=float(surface[1]), z=float(surface[2]))
    # Origins were moved INWARD_OFFSET_MM into the material, so each measured
    # distance is (true_thickness - offset). Add it back to recover thickness.
    return float(kept_dists.min()) + INWARD_OFFSET_MM, point


def run_feature_checks(stl_path: Path, program: PartProgram) -> tuple[list[Check], list[Fix]]:
    min_wall_required = program.constraints.min_wall_thickness
    if min_wall_required is None:
        return [], []

    mesh = trimesh.load(str(stl_path), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)

    result = _measure_min_wall(mesh)
    if result is None:
        return (
            [
                Check(
                    id="features.min_wall_thickness",
                    tier=Tier.FEATURES,
                    status=CheckStatus.WARN,
                    message="Could not measure wall thickness (no inward ray hits).",
                    expected=f">= {min_wall_required:.2f} mm",
                    actual="unmeasurable",
                )
            ],
            [],
        )
    measured, thinnest_point = result

    ok = measured >= min_wall_required - WALL_MEASUREMENT_TOL_MM
    check = Check(
        id="features.min_wall_thickness",
        tier=Tier.FEATURES,
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=(
            "Minimum wall thickness meets the constraint."
            if ok
            else f"Thinnest wall {measured:.2f} mm is below the required {min_wall_required:.2f} mm."
        ),
        expected=f">= {min_wall_required:.2f} mm",
        actual=f"{measured:.2f} mm",
        tolerance=f"+/-{WALL_MEASUREMENT_TOL_MM} mm, sampled N={N_SAMPLES}",
        # Mark the thinnest-wall sample so the viewer's thickness lens can fly there (Z-up STL frame).
        locator=Locator(point=thinnest_point, lens="thickness"),
    )
    fixes: list[Fix] = []
    if not ok:
        fixes.append(
            Fix(
                op_id=None,
                field="thickness",
                reason=(
                    f"Thinnest wall is {measured:.2f} mm vs required {min_wall_required:.2f} mm. "
                    "Increase the thickness of the thinnest plate/wall operation."
                ),
                suggested_change={"min_wall_thickness": min_wall_required},
            )
        )
    return [check], fixes


__all__ = ["run_feature_checks"]
