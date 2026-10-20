"""Swept-form archetypes (flow_forms): the curved vocabulary for the review's class-D gap.

Each helper must produce ONE watertight solid at the documented envelope -- the same contract
`gears.spur_gear` keeps. The sandbox execution of the matching cheatsheet idioms is covered by
`test_cheatsheet_idioms.py` (slow lane); these direct calls are the fast-lane gate.
"""

from __future__ import annotations

import math

import pytest
import trimesh
from build123d import export_stl

from forge_mind.agentic.flow_forms import backward_curved_impeller, crank_web, tapered_fin_stack


def _mesh(solid, tmp_path, name):
    p = tmp_path / f"{name}.stl"
    export_stl(solid, str(p))
    m = trimesh.load(str(p), force="mesh")
    assert isinstance(m, trimesh.Trimesh)
    return m


def test_impeller_is_one_watertight_solid_at_envelope(tmp_path) -> None:
    m = _mesh(backward_curved_impeller(35, 4, 12, 7, hub_r=8, hub_h=6, bore=6), tmp_path, "imp")
    assert m.is_watertight and m.volume > 0
    assert m.extents[0] == pytest.approx(70, abs=0.5)
    assert m.extents[2] == pytest.approx(16, abs=0.2)  # disc 4 + blades 12
    assert m.body_count == 1, "blades and hub must FUSE into the disc"


def test_impeller_blades_actually_curve_backward(tmp_path) -> None:
    """The whole point vs the toy fan: material must exist OFF the radial spoke lines.

    A flat radial-plate fan concentrates all blade material along N radial rays. A
    backward-curved blade's outer half sits rotated AWAY from the ray its root starts on.
    Measured, not eyeballed: sample the mesh at the rim band and require mass at angles
    between the 7 root rays.
    """
    m = _mesh(
        backward_curved_impeller(35, 4, 12, 7, hub_r=8, hub_h=6, bore=0, sweep_deg=35),
        tmp_path,
        "imp2",
    )
    pts, _ = trimesh.sample.sample_surface(m, 6000, seed=0)
    band = pts[(pts[:, 2] > 6.0)]  # above the disc: blade material only
    ang = (180 / math.pi) * (__import__("numpy").arctan2(band[:, 1], band[:, 0]))  # -180..180
    root_rays = [(360.0 / 7) * i for i in range(7)]
    off_ray = [
        a
        for a in ((ang + 360) % 360)
        if min(abs(((a - r) + 180) % 360 - 180) for r in root_rays) > 12.0
    ]
    assert len(off_ray) > len(band) * 0.25, (
        "blade material is concentrated on the radial rays -- the blades did not curve"
    )


def test_crank_web_has_the_counterweight(tmp_path) -> None:
    m = _mesh(
        crank_web(6, 12, main_boss_r=9, pin_boss_r=6, cw_r=16, main_bore=6, pin_bore=5),
        tmp_path,
        "web",
    )
    assert m.is_watertight and m.body_count == 1
    lo, hi = m.bounds
    assert lo[0] == pytest.approx(-16, abs=0.2), "counterweight lobe must reach -cw_r"
    assert hi[0] == pytest.approx(12 + 6, abs=0.2), "pin boss at throw + pin_boss_r"
    # the crescent puts MORE material on the counterweight side than the pin side carries
    com_x = m.center_mass[0]
    assert com_x < 2.0, f"center of mass {com_x:.1f} should sit pulled toward the counterweight"


def test_fin_stack_tapers_and_stays_open(tmp_path) -> None:
    m = _mesh(tapered_fin_stack(14, 10, 60, 9, fin_r=20, t_root=2.4), tmp_path, "fins")
    assert m.is_watertight and m.body_count == 1
    assert m.extents[2] == pytest.approx(60, abs=0.2)
    assert m.extents[0] == pytest.approx(40, abs=0.2)
    # the bore is OPEN end to end (the review's barrel closed its own tube): a ray down the
    # axis must pass through without hitting material.
    hits = m.ray.intersects_location([[0, 0, 100]], [[0, 0, -1]])[0]
    on_axis = [h for h in hits if abs(h[0]) < 1e-6 and abs(h[1]) < 1e-6]
    assert not on_axis, "the fin-stack bore must stay open end to end"


def test_guards_reject_non_mechanisms() -> None:
    with pytest.raises(ValueError, match="blades"):
        backward_curved_impeller(35, 4, 12, 1, hub_r=8, hub_h=6)
    with pytest.raises(ValueError, match="hub_r"):
        backward_curved_impeller(35, 4, 12, 7, hub_r=40, hub_h=6)
    with pytest.raises(ValueError, match="cw_r"):
        crank_web(6, 12, main_boss_r=9, pin_boss_r=6, cw_r=8)
    with pytest.raises(ValueError, match="fin_r"):
        tapered_fin_stack(14, 10, 60, 9, fin_r=10, t_root=2.4)
    with pytest.raises(ValueError, match="bore_r"):
        tapered_fin_stack(14, 15, 60, 9, fin_r=20, t_root=2.4)
