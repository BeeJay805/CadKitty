"""ILLUSTRATIVE FEA stress (ADR-0058). The load-bearing guarantee is CORRECTNESS of the von Mises
field (validated against an analytic cantilever, away from the clamp singularity) and NO-RESULT
(never a fake number) on degenerate input. The deps live in the [server] extra, so these run under
`--extra server` / CI `--all-extras` and skip otherwise."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

pytest.importorskip("skfem")
from skfem import MeshTet

from forge_mind.physics.fea import _solve_von_mises, estimate_stress


def test_cantilever_von_mises_matches_analytic() -> None:
    # Beam along x, section b x h. Fixed at x=0; unit downward (-z) traction on the x=L face.
    # F = traction*area = 1*(b*h); mid-span top-fibre bending stress = 3L/h (b cancels). Validate the
    # von Mises THERE — away from the clamp stress singularity, where it should be ~uniaxial bending.
    L, b, h = 100.0, 10.0, 10.0  # noqa: N806  (engineering convention: length, breadth, height)
    m = MeshTet().refined(4).scaled([L, b, h])
    vm = _solve_von_mises(
        m,
        3500.0,
        0.35,
        fixed_pred=lambda x: x[0] < 1e-6,
        load_facet_pred=lambda x: x[0] > L - 1e-6,
        load_dir=(0.0, 0.0, -1.0),
    )
    assert vm is not None
    pts = np.asarray(m.p)
    x, z = pts[0], pts[2]
    band = (x > 0.4 * L) & (x < 0.6 * L) & (z > h - 0.1 * h)
    assert band.sum() > 0
    midspan = float(vm[band].mean())
    analytic = 3.0 * L / h  # 30 MPa
    assert 0.5 * analytic < midspan < 1.6 * analytic, (
        f"midspan vm={midspan:.2f} vs analytic {analytic:.2f}"
    )
    # the peak stress sits near the clamp, not out at the loaded tip
    assert float(pts[0][int(np.argmax(vm))]) < 0.3 * L


def test_estimate_stress_on_box_is_finite_and_disclaimed() -> None:
    pytest.importorskip("pytetwild")
    box = trimesh.creation.box(extents=(60.0, 20.0, 20.0))
    box.apply_translation((0.0, 0.0, 10.0))  # rest on z=0
    sf = estimate_stress(box)
    assert sf is not None
    assert len(sf.von_mises) == len(
        box.vertices
    )  # one value per INPUT vertex (colour the GLB by index)
    assert sf.von_mises_max > 0 and np.isfinite(sf.von_mises_max)
    assert all(np.isfinite(v) for v in sf.von_mises)
    assert "NOT a safety rating" in sf.note
    assert "safe" not in sf.load_case.lower()


def test_no_result_on_degenerate_input() -> None:
    # A single triangle is not a solid -> NO RESULT (None), never a fake number.
    tri = trimesh.Trimesh(
        vertices=np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]]), faces=np.array([[0, 1, 2]])
    )
    assert estimate_stress(tri) is None
