"""Feature-completeness localization: the genus-based through-hole counter that catches the
measured "author dropped the bracket's bolt holes" gap. A holeless part short of its planned
through-holes must be flagged ``missing`` so the loop keeps drilling (ADR-0030)."""

from __future__ import annotations

import trimesh

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.spatial import feature_count_feedback


def _brief(*features: BriefFeature) -> DesignBrief:
    return DesignBrief(
        object_name="test part",
        summary="a test part",
        visual_description="a flat plate",
        overall_envelope_mm="40 x 40 x 10",
        features=list(features),
        min_wall_mm=2.0,
        print_orientation_note="flat side down",
    )


def test_holeless_part_is_flagged_missing() -> None:
    # the brief plans 2 bolt holes; a plain box (genus 0) has none -> missing.
    brief = _brief(
        BriefFeature(
            name="bolt holes", purpose="mounting", approx_dims_mm="M4", method="bolt_holes", count=2
        )
    )
    mesh = trimesh.creation.box(extents=(40, 40, 10))
    lines, missing = feature_count_feedback(mesh, brief)
    assert missing is True
    assert lines and "only 0" in lines[0]
    assert "MISSING FEATURES" in lines[0]


def test_present_hole_is_not_flagged() -> None:
    # the brief plans 1 through-hole; a torus (genus 1) has exactly one -> satisfied.
    brief = _brief(
        BriefFeature(
            name="clearance hole", purpose="bolt", approx_dims_mm="M5", method="bolt_holes", count=1
        )
    )
    mesh = trimesh.creation.torus(major_radius=12, minor_radius=4)
    lines, missing = feature_count_feedback(mesh, brief)
    assert missing is False
    assert lines == []


def test_brief_without_holes_never_flags() -> None:
    # no hole-bearing feature -> the counter never fires, whatever the geometry.
    brief = _brief(
        BriefFeature(
            name="handle", purpose="grip", approx_dims_mm="80 long", method="loft", count=1
        )
    )
    mesh = trimesh.creation.box(extents=(40, 40, 10))
    lines, missing = feature_count_feedback(mesh, brief)
    assert missing is False
    assert lines == []


def test_degenerate_body_feedback_names_the_state_not_the_garbage() -> None:
    """Measured 2026-07-18 (cable clip 8/8): an invalid sweep produced a body at x ~ -5.9e15 and
    a NEGATIVE-volume body; the feedback printed 16-digit coordinates verbatim -- unactionable.
    A degenerate body must be NAMED degenerate with rebuild advice, and the gap line (a distance
    to garbage coordinates) must be omitted."""
    from types import SimpleNamespace

    from forge_mind.agentic.spatial import _is_degenerate, disconnection_feedback

    def _bb(lo: float, hi: float) -> SimpleNamespace:
        v = lambda x: SimpleNamespace(X=x, Y=x, Z=x)  # noqa: E731
        return SimpleNamespace(min=v(lo), max=v(hi))

    assert _is_degenerate(_bb(-5.9e15, -5.9e15), 100.0) is True  # absurd coordinates
    assert _is_degenerate(_bb(0, 10), -264.0) is True  # negative volume
    assert _is_degenerate(_bb(0, 10), 100.0) is False  # a real body

    class _Solid:
        def __init__(self, bb: SimpleNamespace, vol: float) -> None:
            self._bb, self.volume = bb, vol

        def bounding_box(self) -> SimpleNamespace:
            return self._bb

    class _Part:
        def __init__(self, solids: list[_Solid]) -> None:
            self._solids = solids

        def solids(self) -> list[_Solid]:
            return self._solids

    lines = disconnection_feedback(
        _Part([_Solid(_bb(-5.9e15, -5.9e15), 1e9), _Solid(_bb(0, 10), -264.0)])
    )
    joined = "\n".join(lines)
    assert "DEGENERATE" in joined and "REBUILD" in joined
    assert "e+15" not in joined and "5864062" not in joined  # no garbage numbers
    assert "apart" not in joined and "OVERLAP in bounds" not in joined  # gap line omitted
