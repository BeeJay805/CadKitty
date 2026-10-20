"""DimTable (ADR-0088 slice 1): the single typed-quantity extraction surface.

The load-bearing test is PARITY: DimTable.stated_values / envelope / positions must reproduce
stated_numbers / parse_envelope_mm / stated_bore_positions exactly, so the flag can default on
only after the audit confirms no verdict moved. The rest pin the new typed/provenance facts.
"""

from __future__ import annotations

from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.dim_table import (
    DERIVED,
    ENVELOPE,
    STATED,
    VERIFIED,
    build_dim_table,
)
from forge_mind.agentic.mapped_dims import stated_bore_positions
from forge_mind.agentic.stated_dims import stated_numbers


def _brief(envelope: str, features: list[dict]) -> DesignBrief:
    return DesignBrief.model_validate(
        {
            "object_name": "part",
            "summary": "s",
            "visual_description": "v",
            "overall_envelope_mm": envelope,
            "features": features,
            "min_wall_mm": 2.0,
            "print_orientation_note": "flat",
        }
    )


_HOLE = {
    "name": "mounting holes",
    "purpose": "bolts",
    "approx_dims_mm": "3.4 mm dia, 76.2 mm center-to-center",
    "method": "hole",
    "count": 2,
    "standard": "M3",
}
_BEARING = {
    "name": "bearing bore",
    "purpose": "bearing",
    "approx_dims_mm": "22 mm diameter, centered 35 mm above the base",
    "method": "bore",
    "count": 1,
}

_PROMPTS = [
    (
        "Design a pillow block: base 80 x 40 x 10 mm, bearing bore 22 mm diameter centered "
        "35 mm above the base, four M5 clearance holes.",
        _brief("80 x 40 x 45 mm", [_BEARING]),
    ),
    (
        "Model a bracket 60 x 20 x 4 mm with two M3 mounting holes 76.2 mm center-to-center, "
        "Ø3.4 clearance, 15 degrees back-angle.",
        _brief("60 x 20 x 46 mm", [_HOLE]),
    ),
    ("A plain box, roughly palm sized.", _brief("no clean triple here", [_BEARING])),
]


def test_stated_values_parity_with_stated_numbers() -> None:
    for prompt, brief in _PROMPTS:
        table = build_dim_table(prompt, brief)
        assert table.stated_values() == stated_numbers(prompt, brief), prompt[:40]


def test_positions_parity_with_stated_bore_positions() -> None:
    for prompt, brief in _PROMPTS:
        table = build_dim_table(prompt, brief)
        legacy = stated_bore_positions(brief, prompt, stated_numbers(prompt, brief))
        assert table.positions() == legacy, prompt[:40]


def test_envelope_parity() -> None:
    for prompt, brief in _PROMPTS:
        table = build_dim_table(prompt, brief)
        assert table.envelope() == parse_envelope_mm(brief.overall_envelope_mm)


def test_provenance_stated_vs_derived_on_the_envelope() -> None:
    # bracket: 60 and 20 are in the prompt (STATED); 46 is the model's own stack (DERIVED)
    prompt, brief = _PROMPTS[1]
    table = build_dim_table(prompt, brief)
    env_q = {round(q.value_mm): q.provenance for q in table.quantities if q.role == ENVELOPE}
    assert env_q[60] == STATED
    assert env_q[20] == STATED
    assert env_q[46] == DERIVED  # the row-41 lesson, now a parse-time fact


def test_registry_diameter_is_verified() -> None:
    # a hole with NO stated diameter resolves from the ISO registry -> a VERIFIED quantity
    # (a STATED diameter would correctly outrank the registry, row-17 -- so it must be absent here)
    brief = _brief(
        "60 x 20 x 8 mm",
        [
            {
                "name": "mounting holes",
                "purpose": "bolts",
                "approx_dims_mm": "M5 clearance for mounting",
                "method": "hole",
                "count": 4,
                "standard": "M5",
            }
        ],
    )
    table = build_dim_table("a plate with four M5 mounting holes", brief)
    assert any(q.provenance == VERIFIED for q in table.quantities)


def test_roles_are_typed_from_parse_evidence() -> None:
    prompt, brief = _PROMPTS[1]
    table = build_dim_table(prompt, brief)
    roles = {q.role for q in table.quantities}
    assert "diameter" in roles  # Ø3.4
    assert "angle" in roles  # 15 degrees
    assert "length" in roles  # the x-chain
    # a stated scalar carries a prompt span; registry/brief values do not
    spanned = [q for q in table.quantities if q.provenance == STATED and q.span is not None]
    assert spanned and all(0 <= q.span[0] < q.span[1] <= len(prompt) for q in spanned)


def test_empty_prompt_is_safe() -> None:
    brief = _brief("10 x 10 x 10 mm", [_BEARING])
    table = build_dim_table("", brief)
    assert table.stated_values() == stated_numbers("", brief)
    assert isinstance(table.positions(), list)


def test_spec_dims_ingested_typed_and_verdict_neutral() -> None:
    """SpecGraph slice 2: a SpecFeature's spec_dims become typed feature-scoped quantities, and
    stated_values() is UNCHANGED vs the same feature without spec_dims (the value is STATED only
    when already in the prompt set, so the protected currency cannot move)."""
    from forge_mind.agentic.brief import SpecBrief

    prompt = "a bracket 60 x 20 x 4 mm with two M3 holes 76.2 mm center-to-center, Ø3.4"
    plain = _brief(
        "60 x 20 x 46 mm",
        [
            {
                "name": "mounting holes",
                "purpose": "bolts",
                "approx_dims_mm": "3.4 mm dia, 76.2 mm center-to-center",
                "method": "hole",
                "count": 2,
            }
        ],
    )
    spec = SpecBrief.model_validate(
        {
            **plain.model_dump(),
            "features": [
                {
                    **plain.features[0].model_dump(),
                    "feature_id": "case_mount_holes",
                    "spec_dims": [
                        {"role": "diameter", "value_mm": 3.4},
                        {"role": "pitch", "value_mm": 76.2},
                    ],
                }
            ],
        }
    )
    t_plain = build_dim_table(prompt, plain)
    t_spec = build_dim_table(prompt, spec)
    # verdict-neutral: the protected-currency set does not move
    assert t_spec.stated_values() == t_plain.stated_values()
    # but the table now carries typed, feature-identified quantities the prose path never had
    spec_qs = [q for q in t_spec.quantities if q.feature_label == "case_mount_holes"]
    assert {q.role for q in spec_qs} == {"diameter", "pitch"}
    assert all(q.provenance in ("stated", "derived") for q in spec_qs)
