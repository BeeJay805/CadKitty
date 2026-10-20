"""T2 acceptance: the dispatch loop routes all three spec shapes through ``author_part`` and
returns AuthorResult + per-check gate verdicts; fallback is monotone and gate-driven; the
agentic floor never falls through; adapters map the spec onto the real engines' arguments.

Engine internals (FEA, SIMP, the LLM loop) are mocked at the module boundary -- their own
suites cover them; THESE tests cover the circulatory system (ADR-0098 SS4 made executable)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh

from forge_mind.structures import dispatch as dispatch_mod
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.autotune import TuneOutcome
from forge_mind.structures.dispatch import (
    analytic_author,
    author_part,
    render_prompt,
    topopt_author,
)
from forge_mind.structures.loadcase import G_M_S2, LoadCase
from forge_mind.structures.spec import ComponentPlacement, RequirementSpec

# ---------------------------------------------------------------- fixtures


def _stl(tmp: Path, name: str = "part.stl") -> Path:
    out = tmp / name
    out.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(20.0, 10.0, 5.0)).export(str(out))
    return out


def _spec_no_load() -> RequirementSpec:
    return RequirementSpec(envelope_mm=(120.0, 40.0, 40.0))


def _spec_load_only() -> RequirementSpec:
    return RequirementSpec(
        envelope_mm=(130.7, 24.0, 64.4),
        load=LoadCase(name="arm_link_v0", force_n=0.5 * G_M_S2),
        deflection_bound_mm=0.0459,
        ref_mass_g=115.19,
    )


def _spec_load_and_component() -> RequirementSpec:
    return RequirementSpec(
        envelope_mm=(142.0, 24.5, 67.0),
        load=LoadCase(name="drone_arm_inplane_0p5kg", force_n=0.5 * G_M_S2),
        deflection_bound_mm=0.15,
        components=[ComponentPlacement(kind="mg996r_servo", at_mm=(112.0, 33.5))],
    )


def _fake_author(provenance: str, tmp: Path, *, fail: bool = False):
    def author(spec: RequirementSpec, work_dir: Path) -> AuthorResult:
        if fail:
            return AuthorResult(provenance=provenance, stl=None, step=None, self_reported={})
        return AuthorResult(
            provenance=provenance,
            stl=_stl(work_dir),
            step=None,
            self_reported={"from": provenance},
        )

    return author


def _fake_gate(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Gate stub: claimable iff the author produced geometry; realistic per-check shape."""
    calls: list[dict[str, Any]] = []

    def gate(result: AuthorResult, spec: RequirementSpec, *, part_name: str) -> dict[str, Any]:
        ok = result.stl is not None
        out = {
            "checks": {"artifact": "pass" if ok else "fail", "watertight": "pass" if ok else "n/a"},
            "measured": {},
            "claimable": ok,
        }
        calls.append({"provenance": result.provenance, "gate": out})
        return out

    monkeypatch.setattr(dispatch_mod, "verify_full", gate)
    return calls


# ---------------------------------------------------------------- routing: the three shapes


def test_routes_all_three_spec_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gate(monkeypatch)
    cases = [
        (_spec_no_load(), "agentic"),
        (_spec_load_only(), "analytic"),
        (_spec_load_and_component(), "topopt"),
    ]
    for spec, expect in cases:
        authors = {
            r: _fake_author(f"fake_{r}", tmp_path) for r in ("analytic", "topopt", "agentic")
        }
        report = author_part(spec, tmp_path / expect, authors=authors)
        assert report["chosen"] == expect
        assert [a["rung"] for a in report["attempts"]] == [expect]  # claimable first try
        assert isinstance(report["final"], AuthorResult)
        assert report["final"].provenance == f"fake_{expect}"
        gate = report["attempts"][0]["gate"]
        assert gate["claimable"] is True and gate["checks"]["artifact"] == "pass"
        # per-attempt artifacts live in their own rung dir
        assert (tmp_path / expect / f"attempt_{expect}").is_dir()


# ---------------------------------------------------------------- fallback ladder


def test_monotone_fallback_collects_every_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gate(monkeypatch)

    def never(spec: RequirementSpec, work_dir: Path) -> AuthorResult:
        raise AssertionError("the agentic floor must not run when topopt is claimable")

    report = author_part(
        _spec_load_only(),
        tmp_path,
        authors={
            "analytic": _fake_author("fake_analytic", tmp_path, fail=True),
            "topopt": _fake_author("fake_topopt", tmp_path),
            "agentic": never,
        },
    )
    assert report["chosen"] == "analytic"
    assert [a["rung"] for a in report["attempts"]] == ["analytic", "topopt"]
    assert report["attempts"][0]["gate"]["claimable"] is False  # the refusal's gate is kept
    assert report["attempts"][1]["gate"]["claimable"] is True
    assert report["final"].provenance == "fake_topopt"


def test_agentic_floor_never_falls_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_gate(monkeypatch)
    report = author_part(
        _spec_no_load(),
        tmp_path,
        authors={"agentic": _fake_author("agentic", tmp_path, fail=True)},
    )
    assert [a["rung"] for a in report["attempts"]] == ["agentic"]
    assert report["attempts"][0]["gate"]["claimable"] is False  # honest floor, still returned
    assert report["final"].stl is None


def test_crashing_author_degrades_to_fail_soft_and_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_gate(monkeypatch)

    def boom(spec: RequirementSpec, work_dir: Path) -> AuthorResult:
        raise RuntimeError("engine exploded")

    report = author_part(
        _spec_load_only(),
        tmp_path,
        authors={
            "analytic": boom,
            "topopt": _fake_author("fake_topopt", tmp_path),
            "agentic": _fake_author("agentic", tmp_path),
        },
    )
    assert [a["rung"] for a in report["attempts"]] == ["analytic", "topopt"]
    assert report["final"].provenance == "fake_topopt"
    # the crash is DIAGNOSABLE from the report (row-78 lesson: structured feedback or bust)
    assert "engine exploded" in report["attempts"][0]["self_reported"]["error"]


# ---------------------------------------------------------------- adapters map the spec


def test_analytic_adapter_maps_spec_onto_the_section_author(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def record(**kwargs: Any) -> dict[str, Any]:
        seen.update(kwargs)
        stl = _stl(tmp_path)
        return {"stl": str(stl), "step": str(tmp_path / "part.step"), "mass_g": 75.6}

    monkeypatch.setattr(dispatch_mod, "author_isection_link", record)
    spec = _spec_load_only()
    result = analytic_author(spec, tmp_path)
    assert result.provenance == "analytic_isection"
    assert result.stl is not None and result.stl.exists()
    # envelope -> length/hy/wx verbatim; bound sizes; payload from the spec's own case
    assert (seen["length"], seen["hy"], seen["wx"]) == (130.7, 24.0, 64.4)
    assert seen["ref_deflection_mm"] == 0.0459
    assert seen["ref_mass_g"] == 115.19
    assert seen["payload_kg"] == pytest.approx(0.5)
    assert seen["out_dir"] == tmp_path


def test_analytic_adapter_without_reference_strips_the_inf_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def record(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["ref_mass_g"] == float("inf")
        return {
            "stl": str(_stl(tmp_path)),
            "step": None,
            "lighter_than_reference": True,
            "mass_vs_reference_pct": -100.0,
            "reference": {"mass_g": float("inf")},
        }

    monkeypatch.setattr(dispatch_mod, "author_isection_link", record)
    spec = RequirementSpec(
        envelope_mm=(100.0, 20.0, 50.0),
        load=LoadCase(name="c", force_n=5.0),
        deflection_bound_mm=0.1,
    )
    result = analytic_author(spec, tmp_path)
    assert result.self_reported["lighter_than_reference"] is None
    assert result.self_reported["mass_vs_reference_pct"] is None
    assert result.self_reported["reference"] is None


def test_analytic_adapter_refuses_without_a_bound(tmp_path: Path) -> None:
    spec = RequirementSpec(envelope_mm=(100.0, 20.0, 50.0), load=LoadCase(name="c", force_n=5.0))
    result = analytic_author(spec, tmp_path)
    assert result.stl is None and "error" in result.self_reported


def test_topopt_adapter_builds_the_domain_and_maps_the_tuner_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def record(builder: Any, **kwargs: Any) -> TuneOutcome:
        seen.update(kwargs)
        seen["builder"] = builder
        stl = _stl(tmp_path)
        return TuneOutcome(
            meets_bound=True,
            volfrac=0.4,
            mass_g=200.0,
            deflection_mm=0.1,
            stl=stl,
            step=None,
            history=[],
        )

    monkeypatch.setattr(dispatch_mod, "tune_volfrac_to_bound", record)
    spec = _spec_load_and_component()
    result = topopt_author(spec, tmp_path, nelx=120, nely=57)
    assert result.provenance == "topopt2d_extrude"
    assert seen["bound_mm"] == 0.15
    assert seen["height_mm"] == 24.5  # extrusion depth = envelope W
    assert seen["scale_xy"] == (142.0 / 120, 67.0 / 57)
    # the captured domain builder carves the declared component and loads at its mount
    dom = seen["builder"](0.5)
    assert dom.nelx == 120 and dom.nely == 57
    assert dom.fixed_dofs and dom.load_dofs and dom.load_vals == [-0.5 * G_M_S2]
    assert "mg996r_servo_0" in result.self_reported["regions"]
    assert result.self_reported["load_at_convention"] == "component mount"


def test_render_prompt_carries_the_spec_facts() -> None:
    p = render_prompt(_spec_load_and_component())
    assert "142 x 25 x 67" in p or "142 x 24 x 67" in p  # envelope, rounded
    assert "mg996r_servo" in p and "pocket" in p
    assert "deflection" in p and "0.15" in p
    q = render_prompt(_spec_no_load())
    assert "carry this load" not in q  # no invented load: the honesty bit
