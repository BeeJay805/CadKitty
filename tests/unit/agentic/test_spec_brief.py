"""SpecGraph slice 2 (ADR-0088): SpecBrief schema + the emission-validity scorer.

Byte-identical-when-off is the load-bearing property: the flag OFF path emits a plain
DesignBrief exactly as today; the SpecBrief fields are additive and a SpecBrief IS-A
DesignBrief, so no existing consumer changes.
"""

from __future__ import annotations

from forge_mind.agentic.brief import (
    BriefFeature,
    DesignBrief,
    SpecBrief,
    SpecDim,
    SpecFeature,
    feature_spec_agreement,
    spec_brief_enabled,
    spec_dim_agrees,
)


def _feat(**kw) -> dict:
    base = {
        "name": "mounting holes",
        "purpose": "bolts",
        "approx_dims_mm": "3.4 mm dia, 76.2 mm center-to-center pitch",
        "method": "hole",
        "count": 2,
    }
    base.update(kw)
    return base


def test_spec_feature_is_a_brief_feature() -> None:
    f = SpecFeature.model_validate(_feat(feature_id="case_mount"))
    assert isinstance(f, BriefFeature)
    assert f.feature_id == "case_mount"
    assert f.spec_dims is None  # optional


def test_spec_brief_is_a_design_brief() -> None:
    sb = SpecBrief.model_validate(
        {
            "object_name": "bracket",
            "summary": "s",
            "visual_description": "v",
            "overall_envelope_mm": "60 x 20 x 46 mm",
            "features": [_feat(feature_id="case_mount")],
            "min_wall_mm": 2.0,
            "print_orientation_note": "flat",
        }
    )
    assert isinstance(sb, DesignBrief)
    assert isinstance(sb.features[0], BriefFeature)
    assert sb.features[0].feature_id == "case_mount"


def test_flag_defaults_off() -> None:
    assert spec_brief_enabled() is False


def test_agreement_counts_only_prose_traceable_dims() -> None:
    prose = "3.4 mm dia, 76.2 mm pitch"
    assert spec_dim_agrees(SpecDim(role="diameter", value_mm=3.4), prose) is True
    assert spec_dim_agrees(SpecDim(role="pitch", value_mm=76.2), prose) is True
    assert spec_dim_agrees(SpecDim(role="length", value_mm=99.0), prose) is False


def test_feature_agreement_flags_an_invented_number() -> None:
    faithful = SpecFeature.model_validate(
        _feat(
            spec_dims=[
                {"role": "diameter", "value_mm": 3.4},
                {"role": "pitch", "value_mm": 76.2},
            ]
        )
    )
    assert feature_spec_agreement(faithful) == (2, 2)
    liar = SpecFeature.model_validate(
        _feat(
            spec_dims=[{"role": "diameter", "value_mm": 3.4}, {"role": "length", "value_mm": 50.0}]
        )
    )
    assert feature_spec_agreement(liar) == (1, 2)  # 50 is not in the prose -> the second liar
    none = SpecFeature.model_validate(_feat())
    assert feature_spec_agreement(none) == (0, 0)  # emitted nothing -> vacuous, not a failure
