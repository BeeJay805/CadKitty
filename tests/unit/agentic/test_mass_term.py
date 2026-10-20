"""The bounded mass_term (superhuman-arm M1 mass lever, param_opt).

One-sided hinge (penalize mass ABOVE target only -> no Goodhart erode-below), off unless a caller
passes a MassTarget (the live pipeline never does), and it composes with the watertight +inf guard.
The integration test proves the ES actually spends the knob it was given.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.agentic.param_opt import MassTarget, _mass_term, fit_loss, polish_params
from forge_mind.agentic.params import extract_params

trimesh = pytest.importorskip("trimesh")

_PLA = 1.24  # g/cm^3


def test_mass_term_penalizes_above_target_only() -> None:
    mesh = SimpleNamespace(volume=1000.0)  # 1000 mm^3 = 1 cm^3 -> 1.24 g at PLA
    assert abs(_mass_term(mesh, MassTarget(1.0, _PLA)) - 0.24) < 1e-9  # (1.24-1.0)/1.0
    assert _mass_term(mesh, MassTarget(1.24, _PLA)) == 0.0  # exactly at target
    assert _mass_term(mesh, MassTarget(2.0, _PLA)) == 0.0  # below target: no penalty, no reward


def test_mass_term_weight_scales_and_guards_bad_target() -> None:
    mesh = SimpleNamespace(volume=2000.0)  # 2.48 g at PLA
    assert abs(_mass_term(mesh, MassTarget(1.0, _PLA, weight=3.0)) - 4.44) < 1e-9  # 1.48 * 3
    assert _mass_term(mesh, MassTarget(0.0, _PLA)) == 0.0  # non-positive target -> no-op


def test_mass_term_rejects_nonpositive_volume() -> None:
    """A degenerate (inverted/zero) volume is +inf, never scored as a free 'zero mass' win."""
    assert _mass_term(SimpleNamespace(volume=-1000.0), MassTarget(1.0, _PLA)) == float("inf")
    assert _mass_term(SimpleNamespace(volume=0.0), MassTarget(1.0, _PLA)) == float("inf")


def test_fit_loss_adds_the_mass_hinge(tmp_path: Path) -> None:
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(str(stl))  # 1000 mm^3 -> 1.24 g
    assert abs(fit_loss(stl, None, None, mass_target=MassTarget(1.0, _PLA)) - 0.24) < 1e-6
    assert fit_loss(stl, None, None, mass_target=MassTarget(1.24, _PLA)) == 0.0


def test_fit_loss_unchanged_without_mass_target(tmp_path: Path) -> None:
    stl = tmp_path / "cube.stl"
    trimesh.creation.box(extents=(10.0, 10.0, 10.0)).export(str(stl))
    assert fit_loss(stl, None, None) == 0.0  # the live path: no targets -> 0.0, byte-identical


def test_watertight_guard_beats_the_mass_hinge(tmp_path: Path) -> None:
    """A non-watertight mesh is +inf regardless of how light it is (no-Goodhart guard wins)."""
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    stl = tmp_path / "open.stl"
    open_mesh.export(str(stl))
    assert fit_loss(stl, None, None, mass_target=MassTarget(0.1, _PLA)) == float("inf")


def test_fit_loss_idiom_constraint_open_channel_vs_hollow_void(tmp_path: Path) -> None:
    """M1 lightweighting must be SINGLE-SHELL (I-beam / ribs / open channels): an internal void is
    two surface shells (body_count 2) and fit_loss's no-Goodhart guard rejects it (+inf), while an
    open channel stays body_count 1 and passes. This is why the M1 brief specifies OPEN sections
    (which are also the FDM-printable ones -- no trapped internal support). Verified 2026-07-13."""
    from build123d import Box, export_stl

    hollow = tmp_path / "hollow.stl"
    export_stl(Box(60, 20, 20) - Box(40, 10, 10), str(hollow))  # enclosed void -> 2 shells
    assert fit_loss(hollow, None, None) == float("inf")

    open_ch = tmp_path / "open.stl"  # cross-slots through the mid-span, pads keep it one body
    export_stl(Box(60, 20, 20) - Box(40, 8, 30) - Box(40, 30, 8), str(open_ch))
    assert fit_loss(open_ch, None, None) == 0.0  # single watertight solid, no other targets -> 0


_CODE = 'PARAMS = {"l": 15.0, "w": 10.0, "h": 10.0}\nresult = Box(PARAMS["l"], PARAMS["w"], PARAMS["h"])\n'


def _box_builder(code: str, out_dir: Path) -> tuple[Path | None, Path | None]:
    spec = extract_params(code)
    if spec is None:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "part.stl"
    trimesh.creation.box(extents=[float(spec.values[k]) for k in ("l", "w", "h")]).export(str(stl))
    return stl, stl


def test_polish_shrinks_toward_the_mass_target(tmp_path: Path) -> None:
    """Mass-only optimization (no reference, no vtarget): the guard must ALLOW it, and the ES must
    shrink the 15x10x10 (1500 mm^3 = 1.86 g) box toward a 1.0 g target."""
    out = polish_params(
        _CODE,
        None,
        tmp_path / "opt",
        max_evals=48,
        build_fn=_box_builder,
        mass_target=MassTarget(target_mass_g=1.0, density_g_cm3=_PLA),
    )
    assert out is not None and out.improved
    assert out.best_loss < 0.05  # reached (or nearly) the below-target zero-loss region
    spec = extract_params(out.best_code)
    assert spec is not None
    vol = float(spec.values["l"]) * float(spec.values["w"]) * float(spec.values["h"])
    mass_g = vol / 1000.0 * _PLA
    assert mass_g < 1.86 * 0.8  # a real shrink toward target, not epsilon noise
