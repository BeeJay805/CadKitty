"""The infeasible-spec refusal (founder ruling, engine deep review 2026-07-04).

"We should have the infeasible one be a test and it should know to refuse it and say why."
The measured case: cam_indexing_template_plate — 8 detent holes on a 90 mm bolt circle
through a 120 x 80 mm plate. The 90°/270° stations sit at ±45 mm on a ±40 mm plate; at the
best rotation 4 of 8 stations are still outside. Both author arms burned 9 iterations
against it (rows 22/24). The engine must now refuse BEFORE authoring and state the numbers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.feasibility import _pattern_fits, spec_conflicts
from forge_mind.llm.cost import LLMUsage


def _feat(name: str, dims: str, count: int = 1, method: str = "bore") -> BriefFeature:
    return BriefFeature(name=name, purpose="p", approx_dims_mm=dims, method=method, count=count)


def _brief(env: str, feats: list[BriefFeature]) -> DesignBrief:
    return DesignBrief(
        object_name="plate",
        summary="s",
        visual_description="v",
        overall_envelope_mm=env,
        features=feats,
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


_CAM_BRIEF = _brief(
    "120 x 80 x 6 mm",
    [
        _feat("detent holes", "5 mm dia, on a 90 mm diameter bolt circle", count=8),
        _feat("central bore", "20 mm dia through"),
        _feat("corner mounting holes", "4.4 mm dia", count=4),
    ],
)


def test_cam_indexing_spec_is_refused_with_the_arithmetic() -> None:
    conflicts = spec_conflicts(_CAM_BRIEF)
    assert len(conflicts) == 1
    why = conflicts[0]
    assert "INFEASIBLE" in why
    assert "90" in why and "120" in why and "80" in why  # the conflicting numbers, named
    assert "rotation" in why.lower()  # it explains the contradiction, not just asserts it
    assert "detent holes" in why  # and names the feature


def test_same_pattern_on_a_wide_enough_plate_is_feasible() -> None:
    b = _brief(
        "120 x 100 x 6 mm",
        [_feat("detent holes", "5 mm dia, on a 90 mm diameter bolt circle", count=8)],
    )
    assert spec_conflicts(b) == []


def test_four_stations_can_rotate_into_a_rectangle() -> None:
    # N=4 on the SAME 90mm circle fits the 120x80 plate at 45 degrees -- a pattern the author
    # can save by rotating must NEVER be refused (the precision-first law)
    b = _brief(
        "120 x 80 x 6 mm",
        [_feat("index holes", "5 mm dia, on a 90 mm diameter bolt circle", count=4)],
    )
    assert spec_conflicts(b) == []
    assert _pattern_fits(47.5, 4, 59.0, 39.0)  # the underlying sweep, directly


def test_bolt_circle_flange_fixture_is_feasible() -> None:
    # the audit-fixture / holdout family the engine passes 54/54 -- must never refuse
    b = _brief(
        "70 x 70 x 8 mm",
        [_feat("bolt holes", "5.4 mm dia on a 40 mm bolt circle", count=6)],
    )
    assert spec_conflicts(b) == []


def test_unparseable_envelope_fails_soft() -> None:
    b = _brief(
        "roughly palm-sized",
        [_feat("detent holes", "5 mm dia on a 90 mm circle", count=8)],
    )
    assert spec_conflicts(b) == []


def test_produce_part_refuses_before_authoring() -> None:
    """End-to-end seam: an infeasible brief short-circuits produce_part BEFORE any author
    call -- disposition refused_infeasible, reasons on the loop, report carries them."""
    import json

    from forge_mind.agentic import result as result_mod

    calls = {"author": 0}

    class _NoAuthor:
        def structured(self, *a: object, **k: object) -> object:
            calls["author"] += 1
            raise AssertionError("the author must never run on a refused spec")

        complete = structured

    with patch.object(result_mod, "research_brief", return_value=(_CAM_BRIEF, LLMUsage(0, 0, 0.0))):
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            res = result_mod.produce_part("an indexing plate", Path(td), client=_NoAuthor())
            assert res.disposition == "refused_infeasible"
            assert res.loop.stopped_reason == "infeasible_spec"
            assert res.loop.refusal_reasons and "90" in res.loop.refusal_reasons[0]
            assert calls["author"] == 0
            report = json.loads((Path(td) / "report.json").read_text(encoding="utf-8"))
            assert report["disposition"] == "refused_infeasible"
            assert report["refusal_reasons"]
