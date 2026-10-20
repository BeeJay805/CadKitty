"""Lever d (ADR-0045): the six-panel critic sheet -- section cuts reveal internals,
and the flag gates it to the CRITIC only (money-free)."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.critic import DesignCritique, critique_part
from forge_mind.agentic.critic_views import render_critic_sheet
from forge_mind.llm.cost import LLMUsage


def _bored_box_stl(tmp_path: Path) -> Path:
    """A 40x30x20 box with a 10 mm through-bore -- the internal feature a 3-view
    shaded render cannot show but a section cut must."""
    box = trimesh.creation.box(extents=[40, 30, 20])
    bore = trimesh.creation.cylinder(radius=5, height=30)
    solid = trimesh.boolean.difference([box, bore])
    p = tmp_path / "bored_box.stl"
    solid.export(str(p))
    return p


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="bored block",
        summary="a block with a vertical bore",
        visual_description="a rectangular block with a round hole through the top",
        overall_envelope_mm="40 x 30 x 20",
        features=[
            BriefFeature(
                name="bore",
                purpose="shaft passage",
                approx_dims_mm="10 dia through",
                method="boolean_cut",
            )
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def test_sheet_renders_with_sections(tmp_path: Path) -> None:
    stl = _bored_box_stl(tmp_path)
    out = render_critic_sheet(stl, tmp_path / "sheet.png", "bored block")
    assert out.exists() and out.stat().st_size > 50_000  # a real six-panel figure


class _Sentinel:
    def generate_structured_with_image(
        self, *, system, user, image_png, output_model, model, max_tokens
    ):
        return (
            DesignCritique(
                looks_right=True,
                resemblance="strong",
                feature_verdicts=[],
                prioritized_fixes=[],
                summary="ok",
            ),
            LLMUsage(model=model, input_tokens=1, output_tokens=1, cost_usd=0.0),
        )


def test_flag_off_keeps_the_standard_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORGE_CRITIC_VIEWS", raising=False)
    stl = _bored_box_stl(tmp_path)
    critique_part(stl, _brief(), client=_Sentinel(), work_dir=tmp_path)
    assert not list(tmp_path.glob("*__critic_views.png"))  # byte-identical default


def test_flag_on_gives_the_critic_the_six_panel_sheet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_CRITIC_VIEWS", "1")
    stl = _bored_box_stl(tmp_path)
    critique_part(stl, _brief(), client=_Sentinel(), work_dir=tmp_path)
    assert list(tmp_path.glob("*__critic_views.png"))
