"""L71 critic spec-grounding: the measured block carries the stated dims as the rubric.

Pinned to the generation-4 measurement: the critic demanded thicker walls on tubes
MEASURED at their stated 2.5 mm, and obeying it broke validation twice. With the spec in
the block, slender-but-stated reads as CORRECT.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic.critic_context import critic_measured_block


def _validation(passed: bool, issues: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        passed=passed, issues=issues, volume_mm3=174346.0, body_count=1, watertight=True
    )


_SANDBOX = SimpleNamespace(extents=(120.0, 150.0, 182.0))

_BRIEF = SimpleNamespace(
    features=[
        SimpleNamespace(
            name="runner tubes",
            approx_dims_mm="24 OD x 2.5 wall (19 ID matching the ports), sweeping 90-120 deg",
        ),
        SimpleNamespace(name="head flange plate", approx_dims_mm="150 x 45 x 8 plate"),
        SimpleNamespace(name="unstated feature", approx_dims_mm=""),
    ]
)


@pytest.fixture(autouse=True)
def _flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_CRITIC_MEASURED", "1")


def test_flag_off_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_CRITIC_MEASURED", raising=False)
    assert critic_measured_block(_validation(True, []), _SANDBOX, _BRIEF) is None


def test_spec_section_carries_stated_dims_as_the_rubric() -> None:
    block = critic_measured_block(_validation(True, []), _SANDBOX, _BRIEF) or ""
    assert "THE SPEC" in block
    assert "24 OD x 2.5 wall" in block
    assert "is CORRECT even where it looks slender" in block
    assert "unstated feature" not in block  # dim-less features contribute nothing


def test_passed_residuals_stay_labeled_advisory() -> None:
    block = (
        critic_measured_block(
            _validation(True, ["features.min_wall_thickness: Thinnest wall 0.35 mm ..."]),
            _SANDBOX,
            _BRIEF,
        )
        or ""
    )
    assert "SUB-THRESHOLD residuals" in block and "OPEN validator complaints" not in block


def test_failed_validation_keeps_open_complaint_framing() -> None:
    block = (
        critic_measured_block(
            _validation(False, ["WALL GATE: 19% of the measured surface ..."]), _SANDBOX, _BRIEF
        )
        or ""
    )
    assert "OPEN validator complaints" in block and "THE SPEC" in block


def test_briefless_call_still_produces_the_measured_core() -> None:
    block = critic_measured_block(_validation(True, []), _SANDBOX, None) or ""
    assert "MEASURED BY THE VALIDATOR" in block and "THE SPEC" not in block
