"""Dimension provenance tiers (ADR-0040): earned in code, never self-claimed.

The anchor invariant under test: promotion above ``estimated`` requires exact code
resolution against an artifact the model did not author (a registry row or the
user-confirmed intake block). All money-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.provenance import (
    Tier,
    derive_provenance,
    parse_stated_dimensions,
    resolve_standard,
    tier_counts,
)


def _brief(features: list[BriefFeature]) -> DesignBrief:
    return DesignBrief(
        object_name="test object",
        summary="s",
        visual_description="v",
        overall_envelope_mm="100 x 50 x 20",
        features=features,
        min_wall_mm=2.5,
        material="PLA",
        print_orientation_note="flat",
    )


def _feature(name: str, dims: str, **kw: object) -> BriefFeature:
    return BriefFeature(name=name, purpose="p", approx_dims_mm=dims, method="box", **kw)  # type: ignore[arg-type]


# Realistic STATED-FACTS block (the exact shape intake.brief.brief_to_context emits).
_INTAKE = """STATED FACTS from the guided intake (typed, user-confirmed or standards-resolved).
Copy these exactly; do not contradict or re-derive them:
- object: a wall bracket for my monitor
- fabrication: fdm 3d printing; material: pla
- dimension: mount width = 120 mm (source: user)
- dimension: vesa pattern = 100 mm (source: standard)
- dimension: arm reach = 250 mm (source: default)
"""


# ---------------------------------------------------------------------------
# Tier earning: registry hits.
# ---------------------------------------------------------------------------


def test_standard_fastener_grounds_verified_with_citation() -> None:
    """The headline WS1 test: a part needing a standard fastener grounds at VERIFIED
    with the registry's own citation."""
    brief = _brief([_feature("mounting holes", "4x M3 on 31mm square", standard="M3")])
    (rec,) = derive_provenance(brief)
    assert rec.tier is Tier.VERIFIED
    assert rec.mechanism == "registry_resolution"
    assert rec.citation is not None and "ISO 273" in rec.citation


def test_provisional_registry_row_caps_at_sourced() -> None:
    """provisional=True is what the flag is FOR: a manufacturer/rule-of-thumb row can
    never read as a published standard."""
    brief = _brief([_feature("magnet pocket", "6mm disc magnet", standard="6x3")])
    (rec,) = derive_provenance(brief)
    assert rec.tier is Tier.SOURCED
    assert rec.mechanism == "provisional_registry"
    assert rec.citation is not None and "PROVISIONAL" in rec.citation


def test_model_invented_designation_earns_nothing_and_records_demand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tier-laundering guard: a model-authored citation string the registry cannot
    resolve stays ESTIMATED -- and lands a DEMAND record in the ledger (curation queue)."""
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("FORGE_PROVENANCE_LEDGER", str(ledger))
    brief = _brief([_feature("crank journal", "48mm dia", standard="ISO 99999 J7")])
    (rec,) = derive_provenance(brief)
    assert rec.tier is Tier.ESTIMATED
    assert rec.citation is None
    lines = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert lines and lines[0]["kind"] == "demand"
    assert lines[0]["designation"] == "ISO 99999 J7"


# ---------------------------------------------------------------------------
# Tier earning: intake facts (sourced / stated); defaults never promote.
# ---------------------------------------------------------------------------


def test_intake_parser_maps_sources_and_skips_defaults() -> None:
    dims = parse_stated_dimensions(_INTAKE)
    by_value = {d.value: d for d in dims}
    assert by_value[120.0].tier is Tier.STATED  # source: user -> "you told us"
    assert by_value[100.0].tier is Tier.SOURCED  # source: standard -> a real anchor
    assert 250.0 not in by_value  # source: default NEVER promotes (not a fact)


def test_user_stated_dimension_earns_stated_not_sourced() -> None:
    """The registry-keeper's fourth tier: the user's number is authoritative for INTENT,
    validating nothing about the world -- 'you told us', never 'we checked'."""
    brief = _brief([_feature("mount plate", "120 wide x 60 tall")])
    (rec,) = derive_provenance(brief, extra_context=_INTAKE)
    assert rec.tier is Tier.STATED
    assert rec.mechanism == "intake_stated"
    assert rec.citation is not None and "mount width = 120 mm" in rec.citation


def test_intake_standard_fact_earns_sourced_and_records_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("FORGE_PROVENANCE_LEDGER", str(ledger))
    brief = _brief([_feature("vesa boss pattern", "100 x 100 square")])
    (rec,) = derive_provenance(brief, extra_context=_INTAKE)
    assert rec.tier is Tier.SOURCED
    assert rec.mechanism == "intake_sourced"
    kinds = [json.loads(line)["kind"] for line in ledger.read_text().splitlines()]
    assert "candidate" in kinds


# ---------------------------------------------------------------------------
# The anchor invariant: nothing model-authored raises a tier.
# ---------------------------------------------------------------------------


def test_novel_part_builds_estimated_with_basis_flagged() -> None:
    """The crankshaft-class case at the WS1 level: a novel part BUILDS (no refusal) and
    every unanchored dimension is flagged estimated with its self-reported basis."""
    brief = _brief(
        [
            _feature(
                "crank journal",
                "48mm dia x 25 wide",
                estimate_basis="anchor_scaled",
                estimate_anchor="typical 4-cyl journal ~0.6x bore (~80mm bore)",
            ),
            _feature("counterweight", "70mm radius lobe"),
        ]
    )
    recs = derive_provenance(brief)
    assert all(r.tier is Tier.ESTIMATED for r in recs)
    assert recs[0].estimate_basis is not None
    assert recs[0].estimate_basis.method == "anchor_scaled"
    assert "4-cyl" in (recs[0].estimate_basis.anchor or "")
    # No self-report -> default_guess, honestly.
    assert recs[1].estimate_basis is not None
    assert recs[1].estimate_basis.method == "default_guess"


def test_self_reported_basis_never_raises_the_tier() -> None:
    """A model may CLAIM its estimate is anchored to a standard in prose -- the tier
    stays estimated unless code resolved a real anchor (the laundering guard)."""
    brief = _brief(
        [
            _feature(
                "flange",
                "52mm bolt circle",
                estimate_basis="anchor_scaled",
                estimate_anchor="per ISO 4762 and SAE J429 (model's own claim)",
            )
        ]
    )
    (rec,) = derive_provenance(brief)
    assert rec.tier is Tier.ESTIMATED
    assert rec.citation is None  # an anchor STRING is not a citation


def test_tier_counts_summarize() -> None:
    brief = _brief(
        [
            _feature("holes", "4x M3", standard="M3"),
            _feature("magnet seat", "6mm pocket", standard="6x3"),
            _feature("plate", "120 x 60"),
            _feature("rib", "3mm web"),
        ]
    )
    counts = tier_counts(derive_provenance(brief, extra_context=_INTAKE))
    assert (counts.verified, counts.sourced, counts.stated, counts.estimated) == (1, 1, 1, 1)


def test_resolver_is_exact_match_only() -> None:
    """No fuzzy matching -- fuzz is how laundering starts."""
    assert resolve_standard("M3") is not None
    assert resolve_standard("m3") is not None  # case-normalized, still exact
    assert resolve_standard("M3-ish") is None
    assert resolve_standard("about M3") is None  # first token judged, 'about' is not a key
    assert resolve_standard("") is None
