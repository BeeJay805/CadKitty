"""structures.extrude: density-field vectorization + B-rep extrusion (Fable move #1).

The contract that matters: a SIMP density field becomes a WATERTIGHT single-plane B-rep part with
a real STEP -- the artifact pair the uniform gate requires -- with holes preserved as holes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forge_mind.structures import extrude as ex

trimesh = pytest.importorskip("trimesh")


def _ring_field(n: int = 40) -> np.ndarray:
    """A square with a square hole -- one outer loop, one hole."""
    d = np.zeros((n, n))
    d[4 : n - 4, 4 : n - 4] = 1.0
    d[14 : n - 14, 14 : n - 14] = 0.0
    return d


def test_density_to_loops_finds_outer_and_hole() -> None:
    loops = ex.density_to_loops(_ring_field(), thresh=0.5)
    assert len(loops) == 1  # one solid region
    outer, holes = loops[0]
    assert len(holes) == 1  # with one hole
    assert ex._polygon_area(outer) > ex._polygon_area(holes[0])


def test_density_to_loops_edge_touching_material_closes() -> None:
    """Material touching the domain edge still yields a closed loop (zero-padding)."""
    d = np.zeros((20, 30))
    d[0:20, 0:12] = 1.0  # block flush with three edges
    loops = ex.density_to_loops(d, thresh=0.5)
    assert len(loops) == 1
    assert not loops[0][1]  # no holes


def test_density_to_loops_empty_field_is_empty() -> None:
    assert ex.density_to_loops(np.zeros((10, 10))) == []


def test_extrude_density_emits_watertight_brep_with_hole(tmp_path: Path) -> None:
    res = ex.extrude_density(
        _ring_field(),
        out_dir=tmp_path,
        scale_xy=(1.0, 1.0),
        height_mm=8.0,
        thresh=0.5,
    )
    assert res is not None
    assert res.n_outer == 1 and res.n_holes == 1
    assert res.step.exists() and res.step.stat().st_size > 0  # a REAL STEP (the C2 contract)
    mesh = trimesh.load(str(res.stl), force="mesh")
    assert mesh.is_watertight
    # the hole survived: volume well below the solid prism's
    solid_volume = 32 * 32 * 8.0
    assert float(mesh.volume) < solid_volume * 0.95


def test_opening_deletes_sub_min_slivers_but_keeps_members() -> None:
    """The min-member guarantee: a 1-cell-wide sliver bridging two blocks is DELETED by
    open_radius_cells=1 while the thick blocks survive (rmin alone cannot promise this --
    measured 0.36-0.51 mm necks failed the min-wall gate in proof runs 1-2)."""
    d = np.zeros((20, 40))
    d[2:18, 2:12] = 1.0  # thick block A
    d[2:18, 28:38] = 1.0  # thick block B
    d[9, 12:28] = 1.0  # a 1-cell sliver bridging them
    with_sliver = ex.density_to_loops(d, thresh=0.5)
    opened = ex.density_to_loops(d, thresh=0.5, open_radius_cells=1)
    assert len(with_sliver) == 1  # sliver connects A+B into one loop
    assert len(opened) == 2  # opening deleted the sliver; two clean blocks remain
    for outer, holes in opened:
        assert ex._polygon_area(outer) > 100.0  # both survivors are the thick blocks
        assert not holes


def test_extrude_density_fail_closed_on_empty(tmp_path: Path) -> None:
    assert (
        ex.extrude_density(np.zeros((10, 10)), out_dir=tmp_path, scale_xy=(1.0, 1.0), height_mm=5.0)
        is None
    )


@pytest.mark.slow
def test_simp_to_brep_roundtrip_passes_real_fea(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full Fable-#1 loop: SIMP density -> vectorize -> B-rep -> the real tet FEA runs on it.
    This is the gate the voxel/marching-cubes path could not pass (stairstep defeats pytetwild)."""
    pytest.importorskip("skfem")
    pytest.importorskip("pytetwild")
    monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", "180")
    from forge_mind.structures.loadcase import canonical_link_case, run_load_case
    from forge_mind.structures.topopt import cantilever, optimize

    res = optimize(cantilever(nelx=60, nely=14, volfrac=0.45, rmin=1.4), max_iter=50)
    out = ex.extrude_density(
        res["density"],
        out_dir=tmp_path,
        scale_xy=(80.0 / 60, 20.0 / 14),  # an 80 x 20 mm plate
        height_mm=16.0,
        thresh=0.4,
    )
    assert out is not None
    mesh = trimesh.load(str(out.stl), force="mesh")
    assert mesh.is_watertight
    lr = run_load_case(mesh, canonical_link_case(mesh, payload_kg=0.2))
    assert lr is not None  # the B-rep mesh tetrahedralizes -- the loop is CLOSED
    assert lr.max_deflection_mm > 0.0
