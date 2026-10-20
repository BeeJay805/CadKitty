"""End-to-end M1 recipe regression guard (deterministic, no LLM), on the promoted package:
hand-authored open-channel link -> bounded mass lever -> the SAME canonical-FEA gate. Marked slow
(real FEA); skips cleanly without scikit-fem / pytetwild. Ported from arm_search (ADR-0097)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic.param_opt import MassTarget, polish_params
from forge_mind.agentic.params import extract_params
from forge_mind.materials import mass_grams, resolve_material
from forge_mind.structures.loadcase import canonical_link_case
from forge_mind.structures.objectives import fea_deflection_within

trimesh = pytest.importorskip("trimesh")

_PLA = resolve_material("pla")
_CODE = 'PARAMS = {"l": 50.0, "w": 16.0, "h": 16.0, "cut": 5.0}\nresult = None\n'
_PAD = 8.0


def _beam_builder(code: str, out_dir: Path) -> tuple[Path | None, Path | None]:
    from build123d import Box, export_stl

    spec = extract_params(code)
    assert spec is not None
    length, width, height, cut = (float(spec.values[k]) for k in ("l", "w", "h", "cut"))
    cut = max(1.0, min(cut, width - 4.0, height - 4.0))
    span = max(length - 2 * _PAD, 1.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "part.stl"
    export_stl(
        Box(length, width, height) - Box(span, cut, height + 2) - Box(span, width + 2, cut),
        str(stl),
    )
    return stl, stl


def _mass(stl: Path) -> float:
    return mass_grams(float(trimesh.load(str(stl), force="mesh").volume), _PLA)


@pytest.mark.slow
def test_m1_recipe_lever_then_fea_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("skfem")
    pytest.importorskip("pytetwild")
    monkeypatch.setenv("FORGE_FEA_TIMEOUT_S", "120")

    base_stl, _ = _beam_builder(_CODE, tmp_path / "base")
    assert base_stl is not None
    base_mass = _mass(base_stl)

    out = polish_params(
        _CODE,
        None,
        tmp_path / "opt",
        vtarget=None,
        stated=frozenset({50.0, 16.0}),
        mass_target=MassTarget(target_mass_g=base_mass * 0.7, density_g_cm3=_PLA.density_g_cm3),
        build_fn=_beam_builder,
        max_evals=40,
    )
    assert out is not None and out.improved and out.best_stl is not None
    assert _mass(out.best_stl) < base_mass * 0.95

    mesh = trimesh.load(str(out.best_stl), force="mesh")
    verdict = fea_deflection_within(mesh, canonical_link_case, 5.0)
    assert verdict is True
