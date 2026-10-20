"""structures.topopt: the SIMP optimizer + the component keep-solid/keep-void rule.

Ported from tests/unit/arm_search/test_topopt2d.py (ADR-0097 consolidation).
"""

from __future__ import annotations

from forge_mind.structures import topopt as t2d


def test_simp_reduces_compliance_on_cantilever() -> None:
    res = t2d.optimize(t2d.cantilever(nelx=40, nely=20, volfrac=0.4, rmin=1.5), max_iter=40)
    assert res["density"].shape == (20, 40)
    assert res["compliance"][-1] < res["compliance"][0] * 0.5


def test_simp_respects_volume_budget() -> None:
    res = t2d.optimize(t2d.cantilever(nelx=40, nely=20, volfrac=0.3, rmin=1.5), max_iter=30)
    assert abs(float(res["density"].mean()) - 0.3) < 0.06


def test_simp_carves_real_structure_not_a_grey_blob() -> None:
    res = t2d.optimize(t2d.cantilever(nelx=40, nely=20, volfrac=0.4, rmin=1.5), max_iter=40)
    d = res["density"]
    assert (d > 0.7).mean() > 0.2  # genuine solid members
    assert (d < 0.3).mean() > 0.3  # genuine void
    assert ((d > 0.3) & (d < 0.7)).mean() < 0.3  # discrete, not grey mush


def test_component_pocket_stays_void_and_mount_stays_solid() -> None:
    """The founder's rule: a keep-void motor pocket stays a clean cutout; a keep-solid mount
    stays material; SIMP routes structure around them."""
    dom = t2d.cantilever(nelx=40, nely=20, volfrac=0.5, rmin=1.5)
    pocket = dom.elements_in_box(22, 6, 30, 14)
    mount = dom.elements_in_box(37, 8, 40, 12)
    dom.keep_void = pocket
    dom.keep_solid = mount
    d = t2d.optimize(dom, max_iter=30)["density"]  # (nely, nelx); e = ix*nely + iy
    for e in pocket:
        assert d[e % dom.nely, e // dom.nely] < 0.05
    for e in mount:
        assert d[e % dom.nely, e // dom.nely] > 0.95
