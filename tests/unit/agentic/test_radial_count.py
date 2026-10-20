"""Radial feature counting (ADR-0079): the regression that must never pass silently.

The council's cardinal rule under test everywhere: k* == truth or not_run -- NEVER
confident-wrong. Ground truth is real `spur_gear`/`toothed_rim` builds where they are cheap
(z <= 40) and a synthetic cosine-toothed disc (exact known count, built directly as a mesh)
for the high-count aliasing cases (z=97/120: tooth tips are tiny targets; dropout must land
in not_run, not a wrong count).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from build123d import Cylinder, export_stl

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.gears import spur_gear, toothed_rim
from forge_mind.agentic.radial_count import (
    measure_radial_count,
    planned_radial_teeth,
    radial_count_feedback,
    radial_deficit,
)
from forge_mind.agentic.validator_loss import ValidatorTarget, validator_loss, validator_target


def _mesh_of(part: object, tmp: Path, name: str) -> trimesh.Trimesh:
    p = tmp / name
    export_stl(part, str(p))  # type: ignore[arg-type]
    out: trimesh.Trimesh = trimesh.load(str(p), force="mesh")  # type: ignore[assignment]
    return out


_DISC_CACHE: dict[int, trimesh.Trimesh] = {}


def _toothed_disc(n_teeth: int, r_root: float = 140.0, depth: float = 6.0) -> trimesh.Trimesh:
    """Watertight cosine-toothed disc with an EXACT known tooth count -- ground truth for
    counts where a real involute build takes minutes (the 120-tooth rim measured 99 s).
    8 boundary points per tooth: the measurement ray-casts the surface, it never reads
    vertices, so mesh density only needs to carry the ripple shape. Cached per count."""
    if n_teeth in _DISC_CACHE:
        return _DISC_CACHE[n_teeth]
    h, per_tooth = 20.0, 8
    m = n_teeth * per_tooth
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    r = r_root + depth * 0.5 * (1 + np.cos(n_teeth * th))
    ring = np.column_stack([r * np.cos(th), r * np.sin(th)])
    v_bot = np.column_stack([ring, np.full(m, -h / 2)])
    v_top = np.column_stack([ring, np.full(m, h / 2)])
    verts = np.vstack([v_bot, v_top, [[0, 0, -h / 2]], [[0, 0, h / 2]]])
    cb, ct = 2 * m, 2 * m + 1
    faces = []
    for i in range(m):
        j = (i + 1) % m
        faces.extend([[i, j, m + i], [j, m + j, m + i], [cb, j, i], [ct, m + i, m + j]])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    assert mesh.is_watertight
    _DISC_CACHE[n_teeth] = mesh
    return mesh


def _pinion_shaft(n_teeth: int = 12) -> trimesh.Trimesh:
    """A PROLATE part: a long smooth shaft whose middle third carries the tooth band -- the
    pinion-shaft class. Exercises the prolate axis branch AND the z-band sweep (end levels see
    a smooth cylinder; only mid-z sees teeth)."""
    r_shaft, depth, length = 20.0, 4.0, 200.0
    m = n_teeth * 8
    th = np.linspace(0, 2 * np.pi, m, endpoint=False)
    smooth = np.full(m, r_shaft)
    # root sits 1 mm PROUD of the shaft so the step wall never degenerates to coincident
    # vertices at the roots (process=True would merge them and open the mesh)
    toothed = r_shaft + 1.0 + depth * 0.5 * (1 + np.cos(n_teeth * th))
    # (radius profile, z) vertex rings bottom-to-top; repeated z = a vertical step wall
    rings = [
        (smooth, -length / 2),
        (smooth, -length / 6),
        (toothed, -length / 6),
        (toothed, length / 6),
        (smooth, length / 6),
        (smooth, length / 2),
    ]
    layers = [np.column_stack([r * np.cos(th), r * np.sin(th), np.full(m, z)]) for r, z in rings]
    verts = np.vstack([*layers, [[0, 0, -length / 2]], [[0, 0, length / 2]]])
    n_layers = len(layers)
    cb, ct = n_layers * m, n_layers * m + 1
    faces = []
    for layer in range(n_layers - 1):
        a, b = layer * m, (layer + 1) * m
        for i in range(m):
            j = (i + 1) % m
            faces.extend([[a + i, a + j, b + i], [a + j, b + j, b + i]])
    for i in range(m):
        j = (i + 1) % m
        faces.extend([[cb, j, i], [ct, (n_layers - 1) * m + i, (n_layers - 1) * m + j]])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.asarray(faces), process=True)
    assert mesh.is_watertight
    return mesh


# --- calibration ground truth: real gears must COUNT, not just avoid being wrong ----------------


@pytest.mark.slow
def test_spur_gear_counts_exactly(tmp_path: Path) -> None:
    mesh = _mesh_of(spur_gear(2.0, 12, 6.0, bore=5.0), tmp_path, "g12.stl")
    rc = measure_radial_count(mesh, 12)
    assert rc.k_star == 12
    assert rc.prominence is not None and rc.prominence >= 8.0


@pytest.mark.slow
def test_toothed_rim_fused_on_disc_counts_exactly(tmp_path: Path) -> None:
    """The hard-5 flywheel shape: an external tooth band fused onto a plain disc."""
    mesh = _mesh_of(Cylinder(radius=50, height=8) + toothed_rim(100.0, 40, 8.0), tmp_path, "f.stl")
    rc = measure_radial_count(mesh, 40)
    assert rc.k_star == 40
    assert radial_deficit(40, rc) == 0.0


@pytest.mark.parametrize("z", [97, 120])
def test_high_count_never_confident_wrong(z: int) -> None:
    """The implementer's exposing case: at z=120 tooth tips are tiny targets; aliasing must land
    in not_run, never a wrong count. (The synthetic disc is clean, so it should COUNT.)"""
    rc = measure_radial_count(_toothed_disc(z), z)
    assert rc.k_star == z or rc.k_star is None


@pytest.mark.slow
def test_near_cubic_gear_resolves_or_not_run(tmp_path: Path) -> None:
    """Thickness ~ pitch diameter trips the oblate/prolate axis flip: either outcome is
    acceptable EXCEPT a confident wrong count."""
    mesh = _mesh_of(spur_gear(2.0, 24, 48.0, bore=6.0), tmp_path, "cube.stl")
    rc = measure_radial_count(mesh, 24)
    assert rc.k_star == 24 or rc.k_star is None


# --- the suppression attractor is dead ------------------------------------------------------------


def test_plain_cylinder_not_run_full_deficit() -> None:
    """The flattened-teeth case: a smooth cylinder against a named count must be not_run with
    deficit 1.0 -- erasing the features is never the cheap move."""
    rc = measure_radial_count(trimesh.creation.cylinder(radius=50, height=10), 24)
    assert rc.k_star is None
    assert radial_deficit(24, rc) == 1.0


def test_dome_flywheel_artifact_full_deficit() -> None:
    """The motivating hard-5 failure: a smooth dome where a 120-tooth flywheel should be."""
    dome = trimesh.creation.icosphere(subdivisions=4)
    dome.apply_scale([150, 150, 40])
    rc = measure_radial_count(dome, 120)
    assert radial_deficit(120, rc) == 1.0


def test_wrong_count_costs_less_than_erased() -> None:
    """The chairman's two-level loss: a countable-but-wrong periodic form (a box's honest 4-fold
    corner symmetry) must cost LESS than no countable features at all."""
    box = trimesh.creation.box(extents=[40, 10, 10])
    rc = measure_radial_count(box, 12)
    wrong = radial_deficit(12, rc)
    erased = radial_deficit(
        12, measure_radial_count(trimesh.creation.cylinder(radius=20, height=40), 12)
    )
    assert erased == 1.0
    assert wrong < erased


# --- band sweep: features that do NOT live at the outer rim --------------------------------------


def test_handwheel_spokes_found_at_mid_radius() -> None:
    """A 5-spoke handwheel's spokes live at mid-radius behind a smooth rim: the band sweep must
    find them (or not_run) -- never count the smooth rim as 'no features' with confidence."""
    rim = trimesh.creation.annulus(r_min=90, r_max=100, height=12)
    hub = trimesh.creation.cylinder(radius=15, height=12)
    parts = [rim, hub]
    for i in range(5):
        spoke = trimesh.creation.box(extents=[80, 8, 8])
        spoke.apply_translation([50, 0, 0])
        spoke.apply_transform(trimesh.transformations.rotation_matrix(2 * np.pi * i / 5, [0, 0, 1]))
        parts.append(spoke)
    wheel = trimesh.boolean.union(parts, engine="manifold")
    rc = measure_radial_count(wheel, 5)
    assert rc.k_star == 5 or rc.k_star is None


def test_pinion_shaft_band_found_at_mid_z() -> None:
    """The PROLATE class (a 12-tooth pinion cut into a long shaft): the axis comes from the
    largest eigenvector and only the mid-z levels see teeth -- the axial band sweep, not the
    outer-rim assumption, must find them. Never a confident wrong count."""
    rc = measure_radial_count(_pinion_shaft(12), 12)
    assert rc.k_star == 12 or rc.k_star is None


# --- the brief parser: fail-closed ----------------------------------------------------------------


def _brief(features: list[BriefFeature], summary: str = "a part") -> DesignBrief:
    return DesignBrief(
        object_name="flywheel",
        summary=summary,
        visual_description="a disc",
        overall_envelope_mm="300 x 300 x 60",
        features=features,
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _feat(name: str, dims: str = "", count: int = 1) -> BriefFeature:
    return BriefFeature(name=name, purpose="p", approx_dims_mm=dims, method="revolve", count=count)


def test_parser_reads_a_named_tooth_count() -> None:
    b = _brief([_feat("starter ring gear", "120 teeth, module 2.5"), _feat("hub", "d40")])
    assert planned_radial_teeth(b) == 120


def test_parser_reads_hyphenated_and_after_forms() -> None:
    assert planned_radial_teeth(_brief([_feat("40-tooth sprocket", "d100")])) == 40
    assert planned_radial_teeth(_brief([_feat("rim", "teeth: 96")])) == 96


def test_parser_falls_back_to_the_summary() -> None:
    b = _brief([_feat("body", "d200")], summary="a 72 tooth flywheel with starter ring")
    assert planned_radial_teeth(b) == 72


def test_parser_fail_closed() -> None:
    # several DISTINCT counts: one band cannot be attributed -> no claim
    two = _brief([_feat("ring gear", "120 teeth"), _feat("pinion", "12 teeth")])
    assert planned_radial_teeth(two) is None
    # no count named
    assert planned_radial_teeth(_brief([_feat("hub", "d40")])) is None
    # a count below 4 is indistinguishable from overall form
    assert planned_radial_teeth(_brief([_feat("lobe", "3 teeth")])) is None


# --- loss wiring (ADR-0079 x ADR-0074) ------------------------------------------------------------


def test_validator_target_carries_the_tooth_count() -> None:
    t = validator_target(_brief([_feat("starter ring gear", "120 teeth")]))
    assert t.radial_teeth == 120
    assert t.applicable


def test_loss_charges_not_run_when_count_is_named(tmp_path: Path) -> None:
    """A smooth disc against a named 24-tooth plan pays the full 1.0 -- NOT fail-soft (the
    measurement-suppression attractor)."""
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=None, balance_axis=None, radial_teeth=24)
    assert t.applicable
    disc = tmp_path / "disc.stl"
    trimesh.creation.cylinder(radius=50, height=10).export(str(disc))
    assert validator_loss(disc, t) >= 0.99


def test_loss_zero_on_the_right_tooth_count(tmp_path: Path) -> None:
    t = ValidatorTarget(min_wall_mm=None, envelope_sorted=None, balance_axis=None, radial_teeth=24)
    p = tmp_path / "d24.stl"
    _toothed_disc(24).export(str(p))
    assert validator_loss(p, t) < 0.01


# --- advisory feedback: the author hears about a dropped band ------------------------------------


def test_feedback_names_the_missing_ring_and_the_idiom() -> None:
    brief = _brief([_feat("starter ring gear", "120 teeth")])
    lines = radial_count_feedback(trimesh.creation.cylinder(radius=150, height=40), brief)
    assert len(lines) == 1
    assert "120 teeth" in lines[0]
    assert "toothed_rim" in lines[0]


def test_feedback_silent_when_count_matches_or_unclaimed() -> None:
    brief = _brief([_feat("starter ring gear", "24 teeth")])
    assert radial_count_feedback(_toothed_disc(24), brief) == []
    no_count = _brief([_feat("hub", "d40")])
    assert radial_count_feedback(trimesh.creation.cylinder(radius=50, height=10), no_count) == []


# --- the measurement-frame lock (engine research idea 2, the balance-axis precedent) -------------


def test_lock_agrees_with_the_full_sweep() -> None:
    """Locked measurement of the baseline itself must reproduce the full sweep's count."""
    from forge_mind.agentic.radial_count import lock_from_baseline, measure_radial_count_locked

    disc = _toothed_disc(24)
    lock = lock_from_baseline(disc, 24)
    assert lock is not None
    rc = measure_radial_count_locked(disc, 24, lock)
    assert rc.k_star == 24


def test_lock_is_none_when_the_baseline_has_no_band() -> None:
    """No confident baseline band -> no lock (locking onto noise would be worse than slow)."""
    from forge_mind.agentic.radial_count import lock_from_baseline

    assert lock_from_baseline(trimesh.creation.cylinder(radius=50, height=10), 24) is None


def test_locked_candidate_with_erased_teeth_still_pays() -> None:
    """The suppression attractor stays dead UNDER the lock: a candidate whose teeth vanished
    reads not_run in the locked band -> deficit 1.0."""
    from forge_mind.agentic.radial_count import lock_from_baseline, measure_radial_count_locked

    lock = lock_from_baseline(_toothed_disc(24), 24)
    assert lock is not None
    smooth = trimesh.creation.cylinder(radius=143.0, height=20.0)  # same envelope, no teeth
    rc = measure_radial_count_locked(smooth, 24, lock)
    assert rc.k_star is None
    assert radial_deficit(24, rc) == 1.0


def test_validator_target_locks_from_the_baseline_stl(tmp_path: Path) -> None:
    from forge_mind.agentic.validator_loss import validator_target

    p = tmp_path / "d24.stl"
    _toothed_disc(24).export(str(p))
    t = validator_target(_brief([_feat("ring gear", "24 teeth")]), baseline_stl=p)
    assert t.radial_teeth == 24
    assert t.radial_lock is not None


def test_feedback_names_a_wrong_count() -> None:
    """The detected-but-wrong branch: teeth exist but the count is off -- the author is told
    the MEASURED period, not just 'missing'."""
    brief = _brief([_feat("ring gear", "12 teeth")])
    lines = radial_count_feedback(_toothed_disc(8), brief)
    assert len(lines) == 1
    assert "12 teeth" in lines[0] and "8" in lines[0]
    assert "dominant angular period" in lines[0]
