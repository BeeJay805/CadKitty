"""Decision-grade real-load FEA (slice 8) — the pure solver core, validated vs the analytic cantilever.

The crash-isolated subprocess path (`estimate_load_response`) wraps this with pytetwild; the PHYSICS — turning
a total force into the right traction and reading absolute deflection + stress — is tested here on a
structured MeshTet (no mesher), so it is fast + deterministic. Skipped when scikit-fem is absent."""

from __future__ import annotations

import pytest

pytest.importorskip("skfem")

from skfem import MeshTet

from forge_mind.physics.fea import _facet_area, load_response_on_tetmesh


def test_facet_area_is_exact() -> None:
    m = MeshTet().refined(2).scaled((10.0, 3.0, 4.0))
    area = _facet_area(m, lambda x: x[0] > 10.0 - 0.2)
    assert abs(area - 3.0 * 4.0) < 1e-6  # the end face is W*H


def test_internal_facets_never_receive_surface_loads() -> None:
    import numpy as np

    from forge_mind.physics.fea import _boundary_facets

    m = MeshTet().refined(2)
    facets = _boundary_facets(m, lambda x: np.ones(x.shape[1], dtype=bool))
    assert set(facets) == set(m.boundary_facets())
    assert _facet_area(m, lambda x: np.ones(x.shape[1], dtype=bool)) == pytest.approx(6)
    top = _boundary_facets(m, lambda x: x[2] > 1 - 1e-8)
    assert np.all(m.p[2, m.facets[:, top]] == 1)


def test_small_response_preserves_precision_for_limit_checks() -> None:
    m = MeshTet().refined(1).scaled((20.0, 3.0, 3.0))
    out = load_response_on_tetmesh(
        m, 0.00001, lambda x: x[0] < 1e-8, lambda x: x[0] > 20 - 1e-8, (0.0, 0.0, -1.0), 3500.0, 0.3
    )
    assert out is not None
    assert 0 < out["max_deflection_mm"] < 0.00005
    assert out["max_von_mises_mpa"] > 0


def test_native_meshed_load_uses_declared_planar_faces(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("pytetwild")
    import trimesh

    from forge_mind.physics.fea import estimate_load_response

    # This checks native meshing and physics, not the live server's 30-second
    # latency budget. Parallel suite contention can exhaust that child budget
    # before valid meshing completes; retain every geometry assertion below.
    monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", "120")
    mesh = trimesh.creation.box(extents=(5, 5, 30))
    result = estimate_load_response(
        mesh, 1, load_dir=(1, 0, 0), e_mpa=70000, nu=0.3, edge_length_fac=0.2, planar_faces=True
    )
    assert result is not None
    assert result.loaded_area_mm2 == pytest.approx(25, rel=0.005)
    assert result.fixed_area_mm2 == pytest.approx(25, rel=0.005)
    assert result.boundary_tolerance_mm < 0.01
    assert result.max_deflection_mm > 0


def test_load_response_cantilever_matches_closed_form() -> None:
    length, w, h, e_mpa, nu = 20.0, 3.0, 3.0, 3500.0, 0.3
    force_n = 50.0
    m = MeshTet().refined(2).scaled((length, w, h))
    tol = length * 0.02
    out = load_response_on_tetmesh(
        m,
        force_n,
        fixed_pred=lambda x: x[0] < 1e-6,
        load_facet_pred=lambda x: x[0] > length - tol,
        load_dir_unit=(0.0, 0.0, -1.0),
        e_mpa=e_mpa,
        nu=nu,
    )
    assert out is not None
    inertia = w * h**3 / 12.0
    analytic = force_n * length**3 / (3.0 * e_mpa * inertia)
    # Euler-Bernoulli tip deflection, reproduced by the force->traction->P2-solve path within a few percent.
    assert abs(out["max_deflection_mm"] - analytic) / analytic < 0.06
    assert out["max_von_mises_mpa"] > 0.0
    # the peak deflection is at the loaded (far) end
    assert out["deflection_at_mm"][0] > length * 0.8


def test_patch_load_deflects_more_than_face_spread() -> None:
    """The gap-c physics (T5): the SAME total force concentrated on a CORNER PATCH of the loaded
    face deflects MORE than when it is spread over the whole face -- the off-centre resultant adds
    torsion + local concentration a face-spread averages away. A part designed for a corner point
    load must be MEASURED under one, or the autotuner bisects a misaligned (over-stiff) case."""
    length, w, h, e_mpa, nu, force_n = 20.0, 4.0, 4.0, 3500.0, 0.3, 50.0
    m = MeshTet().refined(3).scaled((length, w, h))
    tol = length * 0.02
    end_face = lambda x: x[0] > length - tol  # noqa: E731  the loaded (free) end face

    def deflect(pred: object) -> float:
        out = load_response_on_tetmesh(
            m,
            force_n,
            fixed_pred=lambda x: x[0] < 1e-6,
            load_facet_pred=pred,
            load_dir_unit=(0.0, 0.0, -1.0),
            e_mpa=e_mpa,
            nu=nu,
        )
        assert out is not None
        return float(out["max_deflection_mm"])

    face = deflect(end_face)
    # a patch at the (y=0, z=0) corner of the end face, r=1.6 mm -> offset from the shear centre
    patch = deflect(lambda x: end_face(x) & ((x[1] ** 2 + x[2] ** 2) <= 1.6**2))
    assert face > 0.0 and patch > 0.0
    assert patch > face * 1.02  # the concentrated off-centre load is strictly, measurably softer


def test_patch_selecting_no_facets_is_no_result() -> None:
    """Fail-closed: a patch that lands on no facets has zero load area -> None, never a load
    silently applied to the whole face or a fabricated number."""
    length = 20.0
    m = MeshTet().refined(2).scaled((length, 3.0, 3.0))
    tol = length * 0.02
    out = load_response_on_tetmesh(
        m,
        50.0,
        fixed_pred=lambda x: x[0] < 1e-6,
        load_facet_pred=lambda x: (x[0] > length - tol) & ((x[1] - 99.0) ** 2 <= 0.01),
        load_dir_unit=(0.0, 0.0, -1.0),
        e_mpa=3500.0,
        nu=0.3,
    )
    assert out is None


def test_zero_force_is_no_result() -> None:
    m = MeshTet().refined(1)
    out = load_response_on_tetmesh(
        m,
        0.0,
        fixed_pred=lambda x: x[0] < 1e-6,
        load_facet_pred=lambda x: x[0] > 1.0 - 1e-6,
        load_dir_unit=(0.0, 0.0, -1.0),
        e_mpa=3500.0,
        nu=0.3,
    )
    assert out is None  # no force -> no fabricated number
