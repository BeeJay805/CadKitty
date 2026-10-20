"""Metric-steered best-of-N revision (ADR-0070, FORGE_REVISION_BEST_OF_N). Multi-objective rank:
validity tier FIRST (a broken candidate can never win -> no Goodhart), shape-signature proximity
only breaks ties among equally-valid candidates. k=1 = the current single revision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.revision import (
    _signature_distance,
    _validity_tier,
    best_revision,
    revision_best_of_n,
)
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.cost import LLMUsage


def _brief() -> DesignBrief:
    return DesignBrief(
        object_name="block",
        summary="a rectangular block",
        visual_description="a solid block",
        overall_envelope_mm="10 x 10 x 10",
        features=[BriefFeature(name="body", purpose="mass", approx_dims_mm="10", method="box")],
        min_wall_mm=1.0,
        print_orientation_note="flat",
    )


class _KClient:
    """Returns a preset list of scripts on successive generate_structured calls (fakes k drafts)."""

    def __init__(self, codes: list[str]) -> None:
        self.codes = codes
        self.i = 0

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
        code = self.codes[min(self.i, len(self.codes) - 1)]
        self.i += 1
        return (
            output_model(approach="c", code=code),  # type: ignore[operator]
            LLMUsage(model=model, input_tokens=1, output_tokens=1, cost_usd=0.0),
        )

    def generate_structured_with_image(self, **_: object) -> tuple[object, LLMUsage]:
        raise AssertionError("image path not used in these tests (image_png=None)")


def test_revision_best_of_n_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_REVISION_BEST_OF_N", raising=False)
    assert revision_best_of_n() == 1
    monkeypatch.setenv("FORGE_REVISION_BEST_OF_N", "3")
    assert revision_best_of_n() == 3
    monkeypatch.setenv("FORGE_REVISION_BEST_OF_N", "0")
    assert revision_best_of_n() == 1  # clamped to >=1
    monkeypatch.setenv("FORGE_REVISION_BEST_OF_N", "junk")
    assert revision_best_of_n() == 1  # invalid -> default


def test_validity_tier_ordering() -> None:
    assert _validity_tier(True, True, False) == 3  # coherent + all gates green
    assert _validity_tier(True, False, False) == 2  # coherent + features present, some gate red
    assert _validity_tier(True, False, True) == 1  # coherent but dropped a feature
    assert _validity_tier(False, False, True) == 0  # not a coherent solid


def test_signature_distance(tmp_path: Path) -> None:
    box = tmp_path / "box.stl"
    trimesh.creation.box([10, 10, 10]).export(str(box))
    tube = tmp_path / "tube.stl"
    trimesh.creation.annulus(r_min=5.0, r_max=10.0, height=8.0).export(str(tube))
    assert _signature_distance(box, None) == 0.0  # no reference -> no discrimination
    assert _signature_distance(box, box) < 1e-4  # identical -> ~0
    assert _signature_distance(box, tube) > 0.0  # different shapes


def test_best_revision_picks_the_valid_candidate(tmp_path: Path) -> None:
    # one candidate builds, one is broken -> the builder wins on validity tier (no reference)
    client = _KClient(["result = Box(10, 10, 10)", "this is not ( valid python"])
    best, _cost, n_built = best_revision(
        _brief(),
        "result = Box(5, 5, 5)",
        "fix",
        k=2,
        client=client,
        author_model=None,
        fabrication_mode=FabricationMode.GENERAL,
        image_png=None,
        image_has_reference=False,
        reference_stl=None,
        material=None,
        sandbox_limits=None,
        work_dir=tmp_path,
        part_name="t",
    )
    assert best == "result = Box(10, 10, 10)" and n_built == 1


def test_best_revision_salts_candidates_only_when_k_gt_1(tmp_path: Path) -> None:
    # Measured 2026-07-18: greedy local decoding returns near-identical completions on identical
    # prompts, so k>1 must diversify the PROMPT. k=1 must stay byte-identical (the ADR-0070
    # default-path contract).
    seen: list[str] = []

    class _SpyClient(_KClient):
        def generate_structured(self, *, user: str, **kw: object) -> tuple[object, LLMUsage]:
            seen.append(user)
            return super().generate_structured(user=user, **kw)  # type: ignore[misc]

    client = _SpyClient(["result = Box(10, 10, 10)"])
    best_revision(
        _brief(),
        "result = Box(5, 5, 5)",
        "fix the wall",
        k=1,
        client=client,
        author_model=None,
        fabrication_mode=FabricationMode.GENERAL,
        image_png=None,
        image_has_reference=False,
        reference_stl=None,
        material=None,
        sandbox_limits=None,
        work_dir=tmp_path / "k1",
        part_name="t",
    )
    assert len(seen) == 1 and "CANDIDATE" not in seen[0]  # k=1: no salt, byte-identical prompt

    seen.clear()
    client = _SpyClient(["result = Box(10, 10, 10)"])
    best_revision(
        _brief(),
        "result = Box(5, 5, 5)",
        "fix the wall",
        k=3,
        client=client,
        author_model=None,
        fabrication_mode=FabricationMode.GENERAL,
        image_png=None,
        image_has_reference=False,
        reference_stl=None,
        material=None,
        sandbox_limits=None,
        work_dir=tmp_path / "k3",
        part_name="t",
    )
    assert len(seen) == 3
    assert all("fix the wall" in u for u in seen)  # the real feedback always survives
    assert len(set(seen)) == 3  # every candidate prompt is DISTINCT
    assert all(f"CANDIDATE {i + 1} of 3" in seen[i] for i in range(3))


def test_best_revision_falls_back_to_base_when_none_build(tmp_path: Path) -> None:
    client = _KClient(["bad ( syntax", "also bad ("])
    best, _cost, n_built = best_revision(
        _brief(),
        "result = Box(5, 5, 5)",
        "fix",
        k=2,
        client=client,
        author_model=None,
        fabrication_mode=FabricationMode.GENERAL,
        image_png=None,
        image_has_reference=False,
        reference_stl=None,
        material=None,
        sandbox_limits=None,
        work_dir=tmp_path,
        part_name="t",
    )
    assert best == "result = Box(5, 5, 5)" and n_built == 0  # base returned, no regression


def test_advisory_overhang_parses_the_validator_emission() -> None:
    """The parser must round-trip the EXACT string printability.py emits (pinned together)."""
    from forge_mind.agentic.revision import advisory_overhang_mm2

    issue = (
        "printability.overhang_max_angle: 4 unsupported overhang region(s), total 313 mm^2. "
        "Largest: 90 mm^2 at 90 deg."
    )
    assert advisory_overhang_mm2([issue]) == 313.0
    assert advisory_overhang_mm2(["ENVELOPE: built 9 x 40 x 60 mm"]) == 0.0
    assert advisory_overhang_mm2([]) == 0.0
