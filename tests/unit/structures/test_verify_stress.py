"""ADR-0107 stress_safety: the uniform gate now screens STRENGTH (won't yield), not only STIFFNESS
(won't sag). A part that passes its deflection bound but over-stresses its material is caught --
the claim-level false-PASS the certificate had ("holds X kg" proved only sag). Fail-closed tri-state.

run_load_case + print_gates are mocked so these tests isolate the STRESS verdict deterministically
(no FEA solve, no slicer); the allowable resolution is unit-tested against the real registry.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

import forge_mind.structures.verify as verify_mod
from forge_mind.materials import get_material, strength_allowable
from forge_mind.physics.fea import LoadResponse
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.loadcase import LoadCase
from forge_mind.structures.spec import RequirementSpec
from forge_mind.structures.verify import verify_artifact


def _lr(von_mises_mpa: float, deflection_mm: float = 0.1) -> LoadResponse:
    return LoadResponse(
        max_deflection_mm=deflection_mm,
        deflection_at_mm=(0.0, 0.0, 0.0),
        max_von_mises_mpa=von_mises_mpa,
        stress_at_mm=(0.0, 0.0, 0.0),
        load_n=100.0,
        load_case="test",
        material_e_mpa=3500.0,
        note="test",
    )


@pytest.fixture
def part(tmp_path: Path) -> AuthorResult:
    """A real watertight box + a present (dummy) STEP, so every gate check but the one under test
    passes -- claimable then flips on stress ALONE."""
    stl = tmp_path / "box.stl"
    trimesh.creation.box(extents=(40.0, 20.0, 10.0)).export(str(stl))
    step = tmp_path / "box.step"
    step.write_text("dummy", encoding="utf-8")  # step_present checks existence only
    return AuthorResult(provenance="analytic_isection", stl=stl, step=step)


def _spec(material_id: str = "pla", load: bool = True) -> RequirementSpec:
    return RequirementSpec(
        envelope_mm=(40.0, 20.0, 10.0),
        material_id=material_id,
        load=LoadCase(name="t", force_n=100.0) if load else None,
        deflection_bound_mm=2.0 if load else None,
    )


@pytest.fixture(autouse=True)
def _fast_print_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    # isolate the stress verdict: the slice/min-wall gate is covered by its own tests
    monkeypatch.setattr(
        verify_mod,
        "print_gates",
        lambda *a, **k: {"min_wall_mm": 2.0, "min_wall_pass": True, "slice_ok": True},
    )


def test_overstress_is_caught_even_when_deflection_passes(
    monkeypatch: pytest.MonkeyPatch, part: AuthorResult
) -> None:
    # PLA allowable 20 MPa. von Mises 15 -> FoS 1.33 < 2 -> FAIL, though sag (0.1) clears the 2.0 bound.
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: _lr(15.0))
    out = verify_artifact(part, _spec("pla"))
    assert out["checks"]["deflection_bound"] == "pass"  # stiffness OK ...
    assert out["checks"]["stress_safety"] == "fail"  # ... but strength is NOT -> the closed gap
    assert out["claimable"] is False  # flips on stress alone (everything else passes)
    assert out["measured"]["stress_fos"] == 1.33
    assert out["measured"]["stress_allowable_mpa"] == 20.0
    assert "PLA" in out["measured"]["stress_basis"]


def test_adequate_strength_passes_and_is_claimable(
    monkeypatch: pytest.MonkeyPatch, part: AuthorResult
) -> None:
    # von Mises 4 -> FoS 5.0 >= 2 -> PASS; the whole gate is claimable.
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: _lr(4.0))
    out = verify_artifact(part, _spec("pla"))
    assert out["checks"]["stress_safety"] == "pass"
    assert out["measured"]["stress_fos"] == 5.0
    assert out["claimable"] is True


def test_metal_uses_the_certifiable_yield_basis(
    monkeypatch: pytest.MonkeyPatch, part: AuthorResult
) -> None:
    # aluminum yield 276 MPa; von Mises 100 -> FoS 2.76 >= 2 -> pass, basis names the alloy/temper.
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: _lr(100.0))
    out = verify_artifact(part, _spec("aluminum"))
    assert out["checks"]["stress_safety"] == "pass"
    assert "6061-T6" in out["measured"]["stress_basis"]
    # push it over: von Mises 200 -> FoS 1.38 -> fail
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: _lr(200.0))
    assert verify_artifact(part, _spec("aluminum"))["checks"]["stress_safety"] == "fail"


def test_fail_closed_degradation(monkeypatch: pytest.MonkeyPatch, part: AuthorResult) -> None:
    # no load -> n/a
    assert verify_artifact(part, _spec("pla", load=False))["checks"]["stress_safety"] == "n/a"
    # failed FEA solve -> unproven (NEVER pass)
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: None)
    assert verify_artifact(part, _spec("pla"))["checks"]["stress_safety"] == "unproven"
    # a material with no defensible basis (TPU) -> n/a + a VISIBLE note, not a guessed pass
    monkeypatch.setattr(verify_mod, "run_load_case", lambda m, c: _lr(15.0))
    out = verify_artifact(part, _spec("tpu"))
    assert out["checks"]["stress_safety"] == "n/a"
    assert "not screened" in out["measured"]["stress_basis"].lower()


def test_strength_allowable_resolution_against_the_registry() -> None:
    pla_a, pla_b = strength_allowable(get_material("pla"))
    assert pla_a == 20.0 and pla_b is not None and "PLA" in pla_b
    assert strength_allowable(get_material("abs"))[0] == 15.0
    al_a, al_b = strength_allowable(get_material("aluminum"))
    assert al_a == 276.0 and al_b is not None and "6061-T6" in al_b  # derived from mechanical yield
    for no_basis in ("tpu", "wood", "resin"):
        assert strength_allowable(get_material(no_basis)) == (None, None)
