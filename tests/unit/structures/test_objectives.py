"""structures.objectives: per-objective fixtures + gaming guards + the §4 hard filter.

Ported from tests/unit/arm_search/test_score_arm.py (ADR-0097 consolidation); same assertions,
real imports. PLA is 1.24 g/cm^3, so a 10 mm cube = exactly 1.24 g.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.structures import objectives as ob
from forge_mind.structures.loadcase import SMOKE_CANTILEVER_5N, canonical_link_case

trimesh = pytest.importorskip("trimesh")

_PLA_CUBE_MASS_G = 1.24


def _cube(mm: float = 10.0):  # type: ignore[no-untyped-def]
    return trimesh.creation.box(extents=(mm, mm, mm))


def test_mass_of_unit_cube_is_exact() -> None:
    p = ob.score_mesh(_cube(10.0), load_case=None)
    assert p.watertight is True
    assert p.volume_mm3 is not None and abs(p.volume_mm3 - 1000.0) < 1e-6
    assert p.mass_g is not None and abs(p.mass_g - _PLA_CUBE_MASS_G) < 1e-6
    assert p.deflection_mm is None
    assert p.material == "pla"


def test_open_mesh_gets_no_mass() -> None:
    """Watertight guard: a mesh with a hole has no trustworthy volume -> no mass number."""
    box = _cube(10.0)
    open_mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[:-2], process=False)
    assert open_mesh.is_watertight is False
    p = ob.score_mesh(open_mesh, load_case=None)
    assert p.watertight is False
    assert p.volume_mm3 is None
    assert p.mass_g is None


def test_aggregate_total_mass_is_fail_closed() -> None:
    good = ob.PartObjectives("a", True, 1000.0, 1.24, "pla", 0.2, 1.0, "c")
    bad = ob.PartObjectives("b", False, None, None, "pla", None, None, None)
    arm = ob._aggregate([good, bad], material="pla")
    assert arm.part_count == 2
    assert arm.total_mass_g is None
    assert arm.max_deflection_mm == 0.2


def test_aggregate_sums_and_takes_worst_deflection() -> None:
    a = ob.PartObjectives("a", True, 1000.0, 1.24, "pla", 0.10, 1.0, "c")
    b = ob.PartObjectives("b", True, 2000.0, 2.48, "pla", 0.30, 2.0, "c")
    arm = ob._aggregate([a, b], material="pla")
    assert arm.total_mass_g is not None and abs(arm.total_mass_g - 3.72) < 1e-6
    assert arm.max_deflection_mm == 0.30


def test_deflection_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ob, "run_load_case", lambda *a, **k: None)
    p = ob.score_mesh(_cube(10.0), load_case=SMOKE_CANTILEVER_5N)
    assert p.mass_g is not None
    assert p.deflection_mm is None
    assert p.von_mises_mpa is None


def test_score_stl_dir_counts_and_sums(tmp_path: Path) -> None:
    for i in range(2):
        _cube(10.0).export(str(tmp_path / f"link_{i}.stl"))
    arm = ob.score_stl_dir(tmp_path, load_case=None)
    assert arm.part_count == 2
    assert arm.total_mass_g is not None and abs(arm.total_mass_g - 2 * _PLA_CUBE_MASS_G) < 1e-4


def _write_report(run: Path, disposition: str, n_parts: int) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "functional_report.json").write_text(
        json.dumps(
            {"disposition": disposition, "parts": [{"name": f"p{i}"} for i in range(n_parts)]}
        ),
        encoding="utf-8",
    )


def test_run_hard_filter_refuses_non_shipped(tmp_path: Path) -> None:
    rejected = tmp_path / "rejected_run"
    _write_report(rejected, "rejected", 3)
    assert ob.run_is_scorable(rejected) is False
    arm = ob.score_run_dir(rejected)
    assert arm.scorable is False
    assert arm.disposition == "rejected"
    assert arm.part_count == 3
    assert arm.total_mass_g is None

    shipped = tmp_path / "shipped_run"
    _write_report(shipped, "shipped", 2)
    assert ob.run_is_scorable(shipped) is True
    assert ob.score_run_dir(shipped).scorable is True

    caveats = tmp_path / "caveats_run"
    _write_report(caveats, "shipped_with_caveats", 1)
    assert ob.run_is_scorable(caveats) is True


def test_missing_report_is_not_scorable(tmp_path: Path) -> None:
    assert ob.run_disposition(tmp_path) is None
    assert ob.run_is_scorable(tmp_path) is False


def test_fea_deflection_within_tri_state(monkeypatch: pytest.MonkeyPatch) -> None:
    mesh = _cube(10.0)
    lc = SMOKE_CANTILEVER_5N
    monkeypatch.setattr(ob, "run_load_case", lambda m, c: SimpleNamespace(max_deflection_mm=0.03))
    assert ob.fea_deflection_within(mesh, lc, 0.05) is True
    monkeypatch.setattr(ob, "run_load_case", lambda m, c: SimpleNamespace(max_deflection_mm=0.08))
    assert ob.fea_deflection_within(mesh, lc, 0.05) is False
    monkeypatch.setattr(ob, "run_load_case", lambda m, c: None)
    assert ob.fea_deflection_within(mesh, lc, 0.05) is None  # fail-open on unmeasurable


def test_fea_deflection_within_accepts_a_case_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(m: object, c: object) -> object:
        seen["case_name"] = c.name  # type: ignore[attr-defined]
        return SimpleNamespace(max_deflection_mm=0.01)

    monkeypatch.setattr(ob, "run_load_case", fake_run)
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 60.0))
    assert ob.fea_deflection_within(mesh, canonical_link_case, 0.05) is True
    assert seen["case_name"] == "arm_link_v0"


@pytest.mark.slow
def test_real_fea_deflection_positive() -> None:
    pytest.importorskip("skfem")
    pytest.importorskip("pytetwild")
    bar = trimesh.creation.box(extents=(10.0, 10.0, 60.0))
    p = ob.score_mesh(bar, load_case=SMOKE_CANTILEVER_5N)
    assert p.mass_g is not None
    assert p.deflection_mm is not None and p.deflection_mm > 0.0
    assert p.von_mises_mpa is not None and p.von_mises_mpa > 0.0
