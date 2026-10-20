"""Fable moves #2/#3 + the router + the uniform gate: the deterministic logic, fast.

autotune's bisection is tested with a monkeypatched measure (monotone synthetic physics), so the
logic is proven without minutes of FEA; the real end-to-end runs as the proof script (TEST_LOG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from forge_mind.structures import autotune as at
from forge_mind.structures import hardware as hw
from forge_mind.structures import router
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.loadcase import LoadCase
from forge_mind.structures.spec import ComponentPlacement, RequirementSpec
from forge_mind.structures.topopt import Domain, cantilever
from forge_mind.structures.verify import verify_artifact

trimesh = pytest.importorskip("trimesh")


# ---- autotune ----------------------------------------------------------------------------------


def test_tune_bisects_to_lightest_passing_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic monotone physics: deflection = 0.1/vf, bound 0.25 -> the crossover is vf=0.4;
    bisection from [0.15, 0.85] must land AT or just above it, never below (fail-closed)."""
    monkeypatch.setattr(at, "optimize", lambda dom, max_iter: {"density": np.ones((4, 4))})

    def fake_extrude(density: Any, *, out_dir: Path, name: str, **kw: Any) -> Any:
        stl = out_dir / f"{name}.stl"
        stl.write_text("x")
        return at.ExtrudeResult(stl=stl, step=stl, n_outer=1, n_holes=0)

    monkeypatch.setattr(at, "extrude_density", fake_extrude)
    state: dict[str, float] = {}

    def fake_measure(res: Any, load_case: Any, material_id: str) -> tuple[float, float]:
        vf = float(res.stl.stem.split("vf")[1])
        state["last_vf"] = vf
        return vf * 100.0, 0.1 / vf  # mass grows, deflection shrinks with volume

    monkeypatch.setattr(at, "_measure", fake_measure)
    out = at.tune_volfrac_to_bound(
        lambda vf: cantilever(nelx=4, nely=4, volfrac=vf),
        load_case=LoadCase(name="t", force_n=1.0),
        bound_mm=0.25,
        out_dir=tmp_path,
        scale_xy=(1.0, 1.0),
        height_mm=5.0,
        max_fea_evals=6,
    )
    assert out.meets_bound is True
    assert out.volfrac is not None and out.volfrac >= 0.1 / 0.25 - 1e-9  # never below crossover
    assert out.volfrac < 0.47  # and bisection actually tightened toward it
    assert out.deflection_mm is not None and out.deflection_mm <= 0.25 - 3 * at._FEA_SCATTER_MM


def test_tune_fail_closed_when_nothing_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(at, "optimize", lambda dom, max_iter: {"density": np.ones((4, 4))})

    def fake_extrude(density: Any, *, out_dir: Path, name: str, **kw: Any) -> Any:
        stl = out_dir / f"{name}.stl"
        stl.write_text("x")
        return at.ExtrudeResult(stl=stl, step=stl, n_outer=1, n_holes=0)

    monkeypatch.setattr(at, "extrude_density", fake_extrude)
    monkeypatch.setattr(at, "_measure", lambda r, lc, m: (50.0, 9.9))  # always too floppy
    out = at.tune_volfrac_to_bound(
        lambda vf: cantilever(nelx=4, nely=4, volfrac=vf),
        load_case=LoadCase(name="t", force_n=1.0),
        bound_mm=0.01,
        out_dir=tmp_path,
        scale_xy=(1.0, 1.0),
        height_mm=5.0,
        max_fea_evals=4,
    )
    assert out.meets_bound is False
    assert out.stl is None  # no claim, no winner artifact


# ---- hardware-as-data --------------------------------------------------------------------------


def test_place_component_carves_pocket_and_mount_tabs() -> None:
    dom = Domain(nelx=142, nely=25)  # 1 mm cells: mm == elements
    comp = ComponentPlacement(kind="mg996r_servo", at_mm=(100.0, 12.0), clearance_mm=0.5)
    sizes = hw.place_component(dom, comp, cell_mm=(1.0, 1.0))
    assert sizes["pocket_elements"] > 0 and sizes["mount_elements"] > 0
    assert set(dom.keep_void).isdisjoint(dom.keep_solid)  # pocket wins where they touch
    # the pocket covers the body center; the mounts sit beyond the body on each x side
    center_e = 100 * dom.nely + 12
    assert center_e in dom.keep_void
    left_tab_e = (100 - 24) * dom.nely + 12  # ~body/2 + tab midpoint left of center
    assert left_tab_e in dom.keep_solid


def test_unknown_hardware_fails_closed() -> None:
    dom = Domain(nelx=10, nely=10)
    with pytest.raises(KeyError):
        hw.place_component(
            dom, ComponentPlacement(kind="warp_drive", at_mm=(5, 5)), cell_mm=(1.0, 1.0)
        )


# ---- the router table --------------------------------------------------------------------------


def test_router_table() -> None:
    no_load = RequirementSpec(envelope_mm=(100, 20, 20))
    assert router.route(no_load) == router.AGENTIC  # no load -> no physics, ever
    loaded = RequirementSpec(
        envelope_mm=(100, 20, 20), load=LoadCase(name="c", force_n=5.0), deflection_bound_mm=0.1
    )
    assert router.route(loaded) == router.ANALYTIC  # bare classical span
    with_comp = RequirementSpec(
        envelope_mm=(142, 25, 67),
        load=LoadCase(name="c", force_n=5.0),
        deflection_bound_mm=0.1,
        components=[ComponentPlacement(kind="mg996r_servo", at_mm=(100, 12))],
    )
    assert router.route(with_comp) == router.TOPOPT  # components -> grow around them


# ---- the uniform gate --------------------------------------------------------------------------


def test_verify_artifact_na_and_pass_paths(tmp_path: Path) -> None:
    from build123d import Box, export_step, export_stl

    stl, step = tmp_path / "p.stl", tmp_path / "p.step"
    part = Box(20, 10, 10)
    export_stl(part, str(stl))
    export_step(part, str(step))
    spec = RequirementSpec(envelope_mm=(20, 10, 10))  # no load, no reference
    out = verify_artifact(AuthorResult(provenance="test", stl=stl, step=step), spec)
    assert out["checks"]["watertight"] == "pass"
    assert out["checks"]["step_present"] == "pass"
    assert out["checks"]["deflection_bound"] == "n/a"  # explicit, never silent
    assert out["checks"]["lighter_than_reference"] == "n/a"
    assert out["checks"]["min_wall"] == "pass"
    assert out["measured"]["mass_g"] is not None


def test_verify_artifact_missing_artifact_fails_closed() -> None:
    spec = RequirementSpec(envelope_mm=(10, 10, 10))
    out = verify_artifact(AuthorResult(provenance="test", stl=None, step=None), spec)
    assert out["claimable"] is False
    assert out["checks"]["artifact"] == "fail"
