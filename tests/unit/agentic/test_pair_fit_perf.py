"""The finish-grind lever (H5B kill, 2026-07-26): _pair_fit paid a full-vertex on_surface
proximity query for EVERY pair -- measured 13.7s/20k vertices, 92k-vertex pelvis_shell,
325 pairs, twice per finish pass -> hours. The fix is an EXACT AABB early-out (far pairs
skip the boolean and the proximity query entirely) plus a branch-and-bound vertex prune
above FORGE_PAIRFIT_VERT_CAP (default 8000): a stride sample upper-bounds the minimum gap,
distance-to-AABB lower-bounds each vertex's distance, and only surviving vertices get the
exact query -- the reported gap is PROVABLY the legacy exact value (clearance_fit and
_worst_case_issues consume it numerically; Codex-caught), never a sampled estimate."""

from __future__ import annotations

import trimesh

from forge_mind.agentic.functional import _pair_fit


def _snug_spheres() -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    """Two dense spheres ~0.2mm apart at the closest point (plus facet error, safely under
    the 1mm touch threshold): a touching mate."""
    a = trimesh.creation.icosphere(subdivisions=3, radius=10.0)  # 642 vertices
    b = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
    b.apply_translation((20.2, 0.0, 0.0))
    return a, b


def test_far_pair_early_outs_without_proximity(monkeypatch) -> None:
    """A pair whose boxes are 100mm apart never constructs a ProximityQuery inside
    _pair_fit. The meshes are built BEFORE the patch: this trimesh version constructs a
    ProximityQuery during Trimesh.__init__ itself (Codex-caught)."""
    a = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    b = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    b.apply_translation((120.0, 0.0, 0.0))

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("ProximityQuery must not be built for an AABB-disjoint pair")

    monkeypatch.setattr(trimesh.proximity, "ProximityQuery", _boom)
    pf = _pair_fit(a, b)
    assert pf is not None
    assert pf["touches"] is False and pf["seated"] is False
    assert pf["min_gap_mm"] >= 100.0  # the stated lower bound: box separation is 110mm
    assert pf["interference_mm3"] == 0.0


def test_pruned_gap_is_exactly_the_legacy_gap_touching(monkeypatch) -> None:
    """The clearance certification reads min_gap_mm as exact, so the pruned path must
    return the IDENTICAL dict the full query returns -- on a touching mate."""
    a, b = _snug_spheres()
    monkeypatch.setenv("FORGE_PAIRFIT_VERT_CAP", "0")
    exact = _pair_fit(a, b)
    monkeypatch.setenv("FORGE_PAIRFIT_VERT_CAP", "64")
    pruned = _pair_fit(a, b)
    assert exact is not None and pruned is not None
    assert pruned == exact
    assert pruned["touches"] is True and pruned["seated"] is True
    assert pruned["min_gap_mm"] <= 1.0


def test_pruned_gap_is_exactly_the_legacy_gap_floating(monkeypatch) -> None:
    """AABBs overlap (no early-out) and the surfaces are ~13mm apart: the pruned result
    still equals the full query's -- floating verdict, exact gap."""
    tube = trimesh.creation.annulus(r_min=20.0, r_max=25.0, height=30.0)
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))  # centered inside the bore
    monkeypatch.setenv("FORGE_PAIRFIT_VERT_CAP", "0")
    exact = _pair_fit(box, tube)  # the tube's vertex set is the pruned query side
    monkeypatch.setenv("FORGE_PAIRFIT_VERT_CAP", "64")
    pruned = _pair_fit(box, tube)
    assert exact is not None and pruned is not None
    assert pruned == exact
    assert pruned["touches"] is False and pruned["seated"] is False
    assert pruned["min_gap_mm"] > 5.0


def test_default_cap_leaves_typical_parts_untouched() -> None:
    """A typical part (fewer vertices than the 8000 default) takes the exact path with no
    sampling pass at all -- the default changes nothing for it."""
    a, b = _snug_spheres()  # 642 vertices each, far under the cap
    pf = _pair_fit(a, b)
    assert pf is not None
    assert pf["touches"] is True and pf["seated"] is True


def test_pair_fit_is_deterministic(monkeypatch) -> None:
    """Stride sampling + pruning has no randomness: two runs return identical dicts."""
    monkeypatch.setenv("FORGE_PAIRFIT_VERT_CAP", "64")
    a, b = _snug_spheres()
    assert _pair_fit(a, b) == _pair_fit(a, b)
