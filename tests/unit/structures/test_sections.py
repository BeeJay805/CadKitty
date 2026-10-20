"""structures.sections: the closed-form pieces + the builder + the wired print gates.

Ported from tests/unit/arm_search/test_analytic_link.py (ADR-0097 consolidation) and extended:
print_gates is now part of the author (Fable's reproducibility fix), so its min-wall half is
tested here (the slice half is tri-state and depends on an installed slicer).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.structures import sections as sec

trimesh = pytest.importorskip("trimesh")


def test_i_required_matches_euler_bernoulli() -> None:
    f, length, e, d = 4.9, 142.0, 3500.0, 0.0457
    assert abs(sec.i_required(f, length, e, d) - f * length**3 / (3 * e * d)) < 1e-6


def test_section_i_increases_with_flange_thickness() -> None:
    assert sec.i_isection(2.0, 24.5, 67.3, 1.2) > sec.i_isection(1.0, 24.5, 67.3, 1.2)


def test_buckling_margins_shape_and_safe_at_light_load() -> None:
    m = sec.buckling_margins(
        3.0, hy=24.5, wx=67.3, tw=1.2, force_n=4.9, length=142.0, e_mpa=3500.0, nu=0.35
    )
    assert set(m) == {"peak_stress_mpa", "yield_margin", "buckling_sig_cr_mpa", "buckling_margin"}
    assert m["buckling_margin"] > 2.0 and m["yield_margin"] > 2.0


def test_build_link_is_single_watertight_solid(tmp_path: Path) -> None:
    from build123d import export_stl

    part = sec.build_link(
        3.0, length=142.0, hy=24.5, wx=67.3, tw=1.2, pad=6.0, hole_d=3.4, hole_sp=20.0
    )
    stl = tmp_path / "link.stl"
    export_stl(part, str(stl))
    mesh = trimesh.load(str(stl), force="mesh")
    assert mesh.is_watertight
    assert int(mesh.body_count) == 1


def test_print_gates_min_wall_measured(tmp_path: Path) -> None:
    """The gates run on the artifact: a 10 mm cube has a measured min-wall >= PLA's 1.0 mm; the
    slice half is tri-state (None when no slicer is installed -- unproven, never claimed)."""
    from build123d import Box, export_stl

    stl = tmp_path / "cube.stl"
    export_stl(Box(10, 10, 10), str(stl))
    gates = sec.print_gates(stl, min_wall_required_mm=1.0)
    assert gates["min_wall_pass"] is True
    assert gates["min_wall_mm"] is not None and gates["min_wall_mm"] >= 1.0
    assert gates["slice_ok"] in (True, False, None)  # tri-state by design
