"""Step 1 of the reference-model spine: when a reference mesh is supplied AND
FORGE_REFERENCE_MODEL=1, the critic judges the build against that real ground truth (a
build+reference composite image, reference-aware prompt). Flag off / no reference =
byte-identical text-only critique. Fail-closed."""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.critic import CRITIC_SYSTEM, DesignCritique, critique_part
from forge_mind.llm.cost import LLMUsage


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="flywheel",
        summary="a flywheel",
        visual_description="a round disc",
        overall_envelope_mm="80 x 80 x 15",
        features=[BriefFeature(name="rim", purpose="mass", approx_dims_mm="80", method="revolve")],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


def _stl(tmp_path: Path, name: str, mesh: trimesh.Trimesh) -> Path:
    p = tmp_path / name
    mesh.export(str(p))
    return p


class _Capture:
    """Records the system prompt + image size the critic sent (no real LLM call)."""

    def __init__(self) -> None:
        self.system: str | None = None
        self.image_len = 0

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: object,
        model: str,
        max_tokens: int,
    ) -> tuple[DesignCritique, LLMUsage]:
        self.system = system
        self.image_len = len(image_png)
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


def test_flag_off_ignores_the_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_REFERENCE_MODEL", raising=False)
    build = _stl(tmp_path, "build.stl", trimesh.creation.box([20, 20, 20]))
    ref = _stl(tmp_path, "ref.stl", trimesh.creation.cylinder(radius=10, height=30))
    cap = _Capture()
    critique_part(build, _brief(), client=cap, work_dir=tmp_path, reference_stl=ref)
    assert cap.system == CRITIC_SYSTEM  # text-only, byte-identical
    assert not list(tmp_path.glob("*__vs_reference.png"))


def test_flag_on_judges_against_the_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_REFERENCE_MODEL", "1")
    build = _stl(tmp_path, "build.stl", trimesh.creation.box([20, 20, 20]))
    ref = _stl(tmp_path, "ref.stl", trimesh.creation.cylinder(radius=10, height=30))
    cap = _Capture()
    critique_part(build, _brief(), client=cap, work_dir=tmp_path, reference_stl=ref)
    assert cap.system != CRITIC_SYSTEM and "REFERENCE" in (cap.system or "")
    assert list(tmp_path.glob("*__vs_reference.png"))  # a build+reference composite was built


def test_flag_on_without_a_reference_stays_text_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_REFERENCE_MODEL", "1")
    build = _stl(tmp_path, "build.stl", trimesh.creation.box([20, 20, 20]))
    cap = _Capture()
    critique_part(build, _brief(), client=cap, work_dir=tmp_path)  # no reference
    assert cap.system == CRITIC_SYSTEM
