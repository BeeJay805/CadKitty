"""The shipped critique must describe the SHIPPED geometry (2026-07-27, chip 2/3).

The hole-repair backstop CUTS HOLES into the finished part, then the loop rebuilt its kept
candidate carrying the critique of the PRE-repair render. `critic_approved` reads `looks_right`
off that, so a stale approval could ride changed geometry -- a potential false-PASS (measured on
H5B's foot_r, whose recorded rejection described geometry the repair had already replaced).

These exercise `_recritique_repaired` itself (the production function the loop calls), with the
critic stubbed -- no vision spend, real behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="mounting plate",
        summary="a plate",
        visual_description="a flat square plate",
        overall_envelope_mm="60 x 60 x 6",
        min_wall_mm=2.0,
        material="PLA",
        target="functional",
        print_orientation_note="flat",
        features=[
            BriefFeature(
                name="bolt holes",
                purpose="mounting",
                approx_dims_mm="four M4 on a 40 mm square pattern",
                method="bolt_holes",
            )
        ],
    )


class _Outcome:
    """Shaped like RepairOutcome for the fields the function reads."""

    def __init__(self, *, hard_ok: bool) -> None:
        self.stl = Path("repaired.stl")
        self.step = Path("repaired.step")
        self.validation = type("V", (), {"hard_ok": hard_ok, "issues": []})()


def _stub_critique(name: str) -> Any:
    return type("C", (), {"looks_right": True, "notes": name})()


def _call(
    outcome: _Outcome, render: Path | None, tmp_path: Path, *, budget: float = 2.0
) -> tuple[Any, float]:
    return loop_mod._recritique_repaired(
        outcome,
        _brief(),
        render,
        client=None,
        model=None,
        work_dir=tmp_path,
        reference_stl=None,
        budget_left_usd=budget,
    )


def test_returns_a_fresh_critique_of_the_repaired_part(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, Any] = {}

    def fake_critique(stl: Any, brief: Any, **kw: Any) -> tuple[Any, Any]:
        seen["stl"] = stl
        seen["render"] = kw.get("render_path")
        return _stub_critique("fresh"), type("U", (), {"cost_usd": 0.07})()

    monkeypatch.setattr(loop_mod, "critique_part", fake_critique)
    render = tmp_path / "render.png"
    critique, cost = _call(_Outcome(hard_ok=True), render, tmp_path)
    assert critique is not None and critique.notes == "fresh"
    assert cost == pytest.approx(0.07)  # the caller must be able to bill it
    # the critic looked at the REPAIRED artifacts, not the pre-repair ones
    assert seen["stl"] == Path("repaired.stl") and seen["render"] == render


def test_a_critic_failure_ships_no_critique_rather_than_the_stale_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def boom(*a: Any, **kw: Any) -> tuple[Any, Any]:
        raise RuntimeError("vision provider down")

    monkeypatch.setattr(loop_mod, "critique_part", boom)
    critique, cost = _call(_Outcome(hard_ok=True), tmp_path / "render.png", tmp_path)
    assert critique is None and cost == 0.0  # fail-CLOSED: no verdict beats a wrong verdict


def test_no_render_means_no_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def never(*a: Any, **kw: Any) -> tuple[Any, Any]:
        raise AssertionError("must not critique without a render")

    monkeypatch.setattr(loop_mod, "critique_part", never)
    assert _call(_Outcome(hard_ok=True), None, tmp_path) == (None, 0.0)


def test_an_incoherent_repaired_part_is_never_critiqued(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repaired part that is not one watertight solid gets no vision call and no verdict."""

    def never(*a: Any, **kw: Any) -> tuple[Any, Any]:
        raise AssertionError("must not critique a non-hard_ok part")

    monkeypatch.setattr(loop_mod, "critique_part", never)
    assert _call(_Outcome(hard_ok=False), tmp_path / "render.png", tmp_path) == (None, 0.0)


def test_an_exhausted_budget_buys_no_verdict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review-caught: a run that stopped ON the budget ceiling must not spend past it for a
    critique. No call, no claim (so the part reads unapproved, not falsely approved)."""

    def never(*a: Any, **kw: Any) -> tuple[Any, Any]:
        raise AssertionError("must not critique with no budget left")

    monkeypatch.setattr(loop_mod, "critique_part", never)
    for spent in (0.0, -0.5):
        assert _call(_Outcome(hard_ok=True), tmp_path / "r.png", tmp_path, budget=spent) == (
            None,
            0.0,
        )


def test_the_measured_refutation_override_still_applies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Review-caught consistency gap: the iteration path post-processes critiques through
    refutation_override (ADR-0121, a promoted lane default). Without it here, a validator-green
    repaired part could be rejected purely for having taken the repair path."""
    from forge_mind.agentic import critic_refutation

    rejected = type("C", (), {"looks_right": False, "summary": "looks wrong", "model_copy": None})()

    def _model_copy(update: dict[str, Any]) -> Any:
        return type(
            "C2", (), {"looks_right": update["looks_right"], "summary": update["summary"]}
        )()

    object.__setattr__(rejected, "model_copy", lambda update: _model_copy(update))
    monkeypatch.setattr(
        loop_mod, "critique_part", lambda *a, **k: (rejected, type("U", (), {"cost_usd": 0.0})())
    )
    monkeypatch.setattr(critic_refutation, "refutation_override", lambda *a, **k: True)
    critique, _cost = _call(_Outcome(hard_ok=True), tmp_path / "r.png", tmp_path)
    assert critique is not None and critique.looks_right is True
    assert critic_refutation.OVERRIDE_MARKER in critique.summary
