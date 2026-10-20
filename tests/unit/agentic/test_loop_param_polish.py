"""The polish accept path (ADR-0080 wiring): the stated-dim audit is a CONJUNCT of acceptance --
a drift rejection keeps the pre-polish candidate and says so in the iteration note; a clean audit
accepts as before. All heavy machinery is monkeypatched at the loop seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from forge_mind.agentic import loop as loop_mod
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.loop import _Candidate, _param_polish
from forge_mind.agentic.param_opt import PolishResult
from forge_mind.agentic.stated_dims import StatedDimAudit
from forge_mind.agentic.validate import AgenticValidation
from forge_mind.fabrication import FabricationMode


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="timing gear",
        summary="a gear",
        visual_description="a disc with teeth",
        overall_envelope_mm="76 x 76 x 14",
        features=[BriefFeature(name="web", purpose="bulk", approx_dims_mm="70 dia", method="box")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _val(passed: bool = True) -> AgenticValidation:
    return AgenticValidation(
        passed=passed, hard_ok=True, report=None, body_count=1, watertight=True, volume_mm3=100.0
    )


def _run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, audit: StatedDimAudit
) -> tuple[Any, list]:
    stl = tmp_path / "a.stl"
    step = tmp_path / "a.step"
    stl.write_text("x")
    step.write_text("x")
    best = _Candidate(
        score=3, code="orig", stl=stl, step=step, render=None, validation=_val(), critique=None
    )
    outcome = PolishResult(
        improved=True,
        baseline_loss=1.0,
        best_loss=0.4,
        evals=8,
        best_code="polished",
        best_stl=stl,
        best_step=step,
        pinned={"bore_d": (12.0, 12.1)},
    )
    monkeypatch.setattr(loop_mod, "validator_target", lambda *a, **k: None)
    monkeypatch.setattr(loop_mod, "stated_numbers", lambda *a, **k: frozenset({12.0}))
    monkeypatch.setattr(loop_mod, "polish_params", lambda *a, **k: outcome)
    monkeypatch.setattr(loop_mod, "validate_authored_part", lambda *a, **k: _val())
    monkeypatch.setattr(loop_mod, "stated_dim_audit", lambda *a, **k: audit)
    iterations: list = []
    result = _param_polish(
        best,
        None,
        _brief(),
        tmp_path,
        iterations,
        fabrication_mode=FabricationMode.PRINT_3D,
        material=None,
        sandbox_limits=None,
        part_name="part",
        prompt="a 76 mm gear with a 12 mm bore",
    )
    return result, iterations


def test_drift_rejection_keeps_the_original(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audit = StatedDimAudit(ok=False, notes=["'bore' Ø12: winner deviates 0.50mm (rejected)"])
    result, iterations = _run(monkeypatch, tmp_path, audit)
    assert result.code == "orig"  # the polish was discarded
    note = iterations[-1].approach
    assert "STATED-DRIFT-REJECT" in note and "(kept original)" in note
    assert "pinned[bore_d 12->12.1]" in note  # the itemized pin evidence rides the record


def test_audit_error_rejects_distinctly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audit = StatedDimAudit(ok=False, error=True, notes=["stated-dim audit failed to run"])
    result, iterations = _run(monkeypatch, tmp_path, audit)
    assert result.code == "orig"
    assert "STATED-AUDIT-ERROR-REJECT" in iterations[-1].approach


def test_clean_audit_accepts_the_polish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audit = StatedDimAudit(ok=True, notes=["'bore' Ø12: held (0.05mm <= 0.24mm)"])
    result, iterations = _run(monkeypatch, tmp_path, audit)
    assert result.code == "polished"
    assert "ACCEPTED" in iterations[-1].approach
