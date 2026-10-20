"""The author self-check epilogue (FORGE_AUTHOR_SELFCHECK=1): the sandbox measures the
built part against the brief's own targets and the structured deltas ride result.json.
Pins: spec derivation, a REAL sandbox run measuring a plate with two bores, the feedback
rendering, and the deployed default (no spec -> no epilogue -> result byte-compatible)."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import selfcheck
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="drilled plate",
        summary="a plate with bolt holes",
        visual_description="a flat plate with two holes",
        overall_envelope_mm="60 x 40 x 6",
        features=[
            BriefFeature(
                name="bolt holes",
                purpose="mounting",
                approx_dims_mm="M5 clearance",
                method="bolt_holes",
                count=2,
                standard="M5",
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def test_build_spec_derives_envelope_and_holes() -> None:
    spec = selfcheck.build_spec(_brief())
    assert spec is not None
    assert spec["envelope"] == [60.0, 40.0, 6.0]
    assert len(spec["holes"]) == 1
    h = spec["holes"][0]
    assert h["label"] == "bolt holes" and h["count"] == 2
    assert h["d_mm"] == pytest.approx(5.5, abs=0.2)  # ISO 273 M5 clearance


def test_build_spec_none_when_nothing_measurable() -> None:
    brief = _brief().model_copy(
        update={
            "overall_envelope_mm": "roughly palm-sized",
            "features": [
                BriefFeature(name="body", purpose="bulk", approx_dims_mm="-", method="box")
            ],
        }
    )
    assert selfcheck.build_spec(brief) is None


def test_real_sandbox_measures_bores_and_envelope(tmp_path: Path) -> None:
    """End-to-end through the REAL kernel: a 60x40x6 plate with two M5 bores measures
    2 matched bores and ~zero envelope delta; a deficit run (1 bore) reports 1 of 2."""
    spec = selfcheck.build_spec(_brief())
    code_two = (
        "with BuildPart() as bp:\n"
        "    Box(60, 40, 6)\n"
        "    with BuildSketch(bp.faces().sort_by(Axis.Z)[-1]):\n"
        "        with GridLocations(40, 0, 2, 1):\n"
        "            Circle(2.75)\n"
        "    extrude(amount=-12, mode=Mode.SUBTRACT)\n"
        "result = bp.part"
    )
    res = run_build123d_code(
        code_two, tmp_path / "two", limits=SandboxLimits(timeout_s=120), selfcheck=spec
    )
    assert res.ok, f"{res.error_type}: {res.error_message}"
    assert res.selfcheck is not None
    env = res.selfcheck["envelope"]
    assert env["target_sorted"] == [6.0, 40.0, 60.0]
    assert all(abs(d) < 0.5 for d in env["delta_mm"])
    holes = res.selfcheck["holes"]
    assert holes[0]["want"] == 2 and holes[0]["matched"] == 2

    code_one = code_two.replace("GridLocations(40, 0, 2, 1)", "GridLocations(40, 0, 1, 1)")
    res1 = run_build123d_code(
        code_one, tmp_path / "one", limits=SandboxLimits(timeout_s=120), selfcheck=spec
    )
    assert res1.ok and res1.selfcheck is not None
    assert res1.selfcheck["holes"][0]["matched"] == 1  # the deficit is named, pre-validator


def test_no_spec_means_no_epilogue(tmp_path: Path) -> None:
    res = run_build123d_code(
        "result = Box(10, 10, 10)", tmp_path, limits=SandboxLimits(timeout_s=120)
    )
    assert res.ok
    assert res.selfcheck is None


def test_feedback_block_renders_the_deltas() -> None:
    sc = {
        "envelope": {
            "built_sorted": [6.0, 38.0, 60.0],
            "target_sorted": [6.0, 40.0, 60.0],
            "delta_mm": [0.0, -2.0, 0.0],
            "delta_pct": [0.0, -5.0, 0.0],
        },
        "bores": [{"d_mm": 5.5, "at": [20.0, 0.0, 0.0]}],
        "holes": [{"label": "bolt holes", "want": 2, "d_mm": 5.5, "matched": 1}],
    }
    block = selfcheck.feedback_block(sc)
    assert "YOUR OWN MEASUREMENTS" in block
    assert "only 1 of 2 present" in block
    assert "Ø5.50 at (20, 0, 0)" in block


def test_feedback_block_empty_for_none() -> None:
    assert selfcheck.feedback_block(None) == ""
    assert selfcheck.feedback_block({}) == ""


def test_flag_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_AUTHOR_SELFCHECK", raising=False)
    assert selfcheck.enabled() is False
    monkeypatch.setenv("FORGE_AUTHOR_SELFCHECK", "1")
    assert selfcheck.enabled() is True
