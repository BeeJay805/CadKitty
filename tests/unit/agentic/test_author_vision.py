"""Multimodal author (ADR-0066, OPT-IN FORGE_AUTHOR_VISION): when the loop hands
``author_revision`` a render of the build being revised, the author revises WITH vision -- the
revise prompt gains a "you can see your build" note and the call goes through the image path.
``image_png=None`` (the deployed default) is byte-identical to the text-only revise call. A
BUILD-over-REFERENCE composite selects the reference-aware note. The loop's ``_author_vision_image``
is fail-closed: flag off / no render / any failure -> ``(None, False)``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.agentic.author import (
    author_decompose_enabled,
    author_initial,
    author_revision,
    author_vision_enabled,
)
from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.loop import _author_vision_image, _reference_render_bytes
from forge_mind.llm.cost import LLMUsage


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="crankshaft",
        summary="a crankshaft",
        visual_description="offset journals with webs and counterweights",
        overall_envelope_mm="400 x 90 x 90",
        features=[
            BriefFeature(
                name="crank_pins", purpose="rod journals", approx_dims_mm="48", method="revolve"
            )
        ],
        min_wall_mm=3.0,
        print_orientation_note="along the axis",
    )


class _Capture:
    """Records which client method the author used + the system prompt it sent (no real LLM)."""

    def __init__(self) -> None:
        self.text_calls = 0
        self.image_calls = 0
        self.system: str | None = None
        self.image_len = 0

    def _ok(self, output_model: object, model: str) -> tuple[object, LLMUsage]:
        parsed = output_model(approach="revise", code="result = Box(1, 1, 1)")  # type: ignore[operator]
        return parsed, LLMUsage(model=model, input_tokens=1, output_tokens=1, cost_usd=0.0)

    def generate_structured(
        self,
        *,
        system: str,
        user: str,
        output_model: object,
        model: str,
        max_tokens: int,
        cache_system: bool = True,
    ) -> tuple[object, LLMUsage]:
        self.text_calls += 1
        self.system = system
        return self._ok(output_model, model)

    def generate_structured_with_image(
        self,
        *,
        system: str,
        user: str,
        image_png: bytes,
        output_model: object,
        model: str,
        max_tokens: int,
    ) -> tuple[object, LLMUsage]:
        self.image_calls += 1
        self.system = system
        self.image_len = len(image_png)
        return self._ok(output_model, model)


def test_no_image_is_text_only() -> None:
    cap = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix it", client=cap)
    assert cap.text_calls == 1 and cap.image_calls == 0
    assert "YOU CAN SEE" not in (cap.system or "")  # the text-only revise prompt, unchanged


def test_build_image_uses_vision_build_note() -> None:
    cap = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix it", client=cap, image_png=b"PNGBYTES")
    assert cap.image_calls == 1 and cap.text_calls == 0
    assert cap.image_len == len(b"PNGBYTES")
    assert "YOU CAN SEE YOUR BUILD" in (cap.system or "")
    assert "AGAINST THE REAL TARGET" not in (cap.system or "")  # build-only, no reference note


def test_reference_image_uses_reference_note() -> None:
    cap = _Capture()
    author_revision(
        _brief(),
        "result = Box(1, 1, 1)",
        "fix it",
        client=cap,
        image_png=b"COMPOSITE",
        image_has_reference=True,
    )
    assert cap.image_calls == 1 and cap.text_calls == 0
    assert "AGAINST THE REAL TARGET" in (cap.system or "")  # the reference-aware note


def test_author_vision_enabled_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_AUTHOR_VISION", raising=False)
    assert author_vision_enabled() is False
    monkeypatch.setenv("FORGE_AUTHOR_VISION", "1")
    assert author_vision_enabled() is True
    monkeypatch.setenv("FORGE_AUTHOR_VISION", "0")
    assert author_vision_enabled() is False


def test_vision_image_flag_off_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FORGE_AUTHOR_VISION", raising=False)
    png = tmp_path / "render.png"
    png.write_bytes(b"PNGBYTES")
    anchor = SimpleNamespace(render=png)
    assert _author_vision_image(anchor, None, tmp_path, "crankshaft") == (None, False)


def test_vision_image_build_only_when_flag_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_AUTHOR_VISION", "1")
    png = tmp_path / "render.png"
    png.write_bytes(b"PNGBYTES")
    anchor = SimpleNamespace(render=png)
    image, has_reference = _author_vision_image(anchor, None, tmp_path, "crankshaft")
    assert image == b"PNGBYTES" and has_reference is False


def test_vision_image_fail_closed_when_render_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_AUTHOR_VISION", "1")
    assert _author_vision_image(SimpleNamespace(render=None), None, tmp_path, "x") == (None, False)
    gone = tmp_path / "missing.png"  # a render path that does not exist -> fail-closed
    assert _author_vision_image(SimpleNamespace(render=gone), None, tmp_path, "x") == (None, False)


# --- decomposition lever (FORGE_DECOMPOSE, ADR-0067) -------------------------------------------


def test_decompose_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DECOMPOSE", raising=False)
    assert author_decompose_enabled() is False
    monkeypatch.setenv("FORGE_DECOMPOSE", "1")
    assert author_decompose_enabled() is True


def test_decompose_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DECOMPOSE", raising=False)
    cap = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix it", client=cap)
    assert "REPEATED-UNIT" not in (cap.system or "")


def test_decompose_on_appends_to_revision_text_and_vision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_DECOMPOSE", "1")
    cap = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix it", client=cap)
    assert "REPEATED-UNIT" in (cap.system or "")
    cap2 = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix it", client=cap2, image_png=b"PNG")
    # both the vision note AND the decompose guidance ride the vision-path system prompt
    assert "REPEATED-UNIT" in (cap2.system or "") and "YOU CAN SEE" in (cap2.system or "")


def test_decompose_on_appends_to_initial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_DECOMPOSE", "1")
    cap = _Capture()
    author_initial(_brief(), client=cap)
    assert "REPEATED-UNIT" in (cap.system or "")


def test_decompose_off_initial_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DECOMPOSE", raising=False)
    cap = _Capture()
    author_initial(_brief(), client=cap)
    assert "REPEATED-UNIT" not in (cap.system or "")


# --- parametric contract note (ADR-0073, FORGE_PARAM_OPT) --------------------------------------


def test_params_note_off_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_PARAM_OPT", raising=False)
    cap = _Capture()
    author_initial(_brief(), client=cap)
    assert "PARAMETRIC CONTRACT" not in (cap.system or "")
    cap2 = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix", client=cap2)
    assert "PARAMETRIC CONTRACT" not in (cap2.system or "")


def test_params_note_on_reaches_initial_and_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_PARAM_OPT", "1")
    cap = _Capture()
    author_initial(_brief(), client=cap)
    assert "PARAMETRIC CONTRACT" in (cap.system or "")
    cap2 = _Capture()
    author_revision(_brief(), "result = Box(1, 1, 1)", "fix", client=cap2)
    assert "PARAMETRIC CONTRACT" in (cap2.system or "")


# --- reference at the initial massing (ADR-0069) ----------------------------------------------


def test_author_initial_no_image_is_text_only() -> None:
    cap = _Capture()
    author_initial(_brief(), client=cap)
    assert cap.text_calls == 1 and cap.image_calls == 0
    assert "YOU CAN SEE THE REAL TARGET" not in (
        cap.system or ""
    )  # text-only first pass, unchanged


def test_author_initial_with_reference_image_uses_vision() -> None:
    cap = _Capture()
    author_initial(_brief(), client=cap, image_png=b"REFPNG")
    assert cap.image_calls == 1 and cap.text_calls == 0
    assert cap.image_len == len(b"REFPNG")
    assert "YOU CAN SEE THE REAL TARGET" in (cap.system or "")


def test_reference_render_bytes_flag_off_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trimesh

    monkeypatch.delenv("FORGE_AUTHOR_VISION", raising=False)
    monkeypatch.setenv("FORGE_REFERENCE_MODEL", "1")
    ref = tmp_path / "ref.stl"
    trimesh.creation.box([10, 10, 10]).export(str(ref))
    assert _reference_render_bytes(ref, "box") is None  # vision off -> byte-identical initial pass
    assert _reference_render_bytes(None, "box") is None  # no reference -> None


def test_reference_render_bytes_renders_when_flags_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import trimesh

    monkeypatch.setenv("FORGE_AUTHOR_VISION", "1")
    monkeypatch.setenv("FORGE_REFERENCE_MODEL", "1")
    ref = tmp_path / "ref.stl"
    trimesh.creation.box([10, 10, 10]).export(str(ref))
    out = _reference_render_bytes(ref, "box")
    assert out is not None and len(out) > 0  # a real reference render was produced and read
