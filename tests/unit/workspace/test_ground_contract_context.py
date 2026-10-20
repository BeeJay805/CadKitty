"""Full source context survives grounding; no CAD constructors or model calls."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from forge_mind.workspace.document import digest
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.photo_nominal import validate_photo_source_replay
from forge_mind.workspace.source_delivery_contract_draft import (
    PhotoSourceContractDraftV2,
    preflight_source_delivery_draft,
)
from forge_mind.workspace.verification_admission import verification_diagnostics


def compound_photo_source():
    """The actual003 five-facet/two-facet shape, using portable retained source."""
    retained = json.loads(
        (Path(__file__).parents[2] / "fixtures" / "photo_source_actual002.json").read_text(
            encoding="utf-8"
        )
    )
    source = retained["source"]
    raw = retained["raw_attempts"][0]
    raw.pop("requirements")
    raw.pop("photo_nominal_claims")
    common = {
        "source": "request",
        "critical": True,
        "subject_ids": ["s1"],
        "size_fact_ids": ["f1", "f2", "f3"],
        "hole_count_fact_id": "f4",
        "frame_policy": "image_plane_xy",
    }
    raw["photo_requirements"] = [
        {
            **common,
            "id": "r1",
            "quote_ref": "q1",
            "facets": [
                "visible_profile",
                "rectangular_outline",
                "extent_u",
                "extent_v",
                "thickness",
            ],
        },
        {**common, "id": "r2", "quote_ref": "q2", "facets": ["opening_count", "straight_through"]},
    ]
    raw["requirement_order"] = ["r1", "r2"]
    for group in raw["fact_groups"]:
        group["requirement_ids"] = ["r2" if group["unit"] == "holes" else "r1"]
    preflight = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(raw), source
    )
    assert preflight.compilation["status"] == "ready", preflight.compilation
    return source, preflight


def test_actual003_compound_cross_row_facts_survive_real_grounder_and_exact_replay():
    source, preflight = compound_photo_source()
    original = preflight.draft
    before = original.model_dump(mode="json")
    result = ground_contract(original.contract, source, attribution=original.attribution)
    assert result == original.contract
    assert original.model_dump(mode="json") == before
    document = {
        "intent_compilation": {
            **preflight.compilation,
            "job_id": "job",
            "source_hash": digest(source),
            "source_proposal": preflight.source_proposal,
        }
    }
    validate_photo_source_replay(
        document, "job", source, result, original.attribution, require_photo=True
    )


def test_invalid_peer_downgrades_only_its_own_typed_row():
    source, preflight = compound_photo_source()
    changed = preflight.draft.model_copy(deep=True)
    changed.contract.requirements[1].verification.source_quote = "Not an original quote"
    before = deepcopy(changed.contract.requirements[0].model_dump(mode="json"))
    result = ground_contract(changed.contract, source, attribution=changed.attribution)
    assert result.requirements[0].model_dump(mode="json") == before
    failed = result.requirements[1]
    assert failed.verification is None and failed.critical and failed.metric == "unmeasured"
    assert failed.text.startswith("Typed verification unavailable:")


@pytest.mark.parametrize("change", ["missing", "wrong_requirement", "wrong_value", "wrong_subject"])
def test_shared_source_fact_corruption_still_blocks_dependent_rows(change):
    source, preflight = compound_photo_source()
    changed = preflight.draft.model_copy(deep=True)
    if change == "missing":
        changed.attribution.facts.pop()
    elif change == "wrong_requirement":
        changed.attribution.facts[-1].requirement_ids = ["absent"]
    elif change == "wrong_value":
        changed.attribution.facts[-1].value = 3.0
    else:
        changed.attribution.facts[-1].subject_id = "absent"
    result = ground_contract(changed.contract, source, attribution=changed.attribution)
    assert all(row.verification is None and row.critical for row in result.requirements)


def conflicting_inventories():
    from forge_mind.workspace.verification_models import NominalSubjectInventory

    rows = []
    quotes = []
    for index, name in enumerate(("plate", "lid"), 1):
        quote = f"Make one nominal CAD part named {name}."
        quotes.append(quote)
        rows.append(
            Requirement(
                id=f"r{index}",
                text=quote,
                source_quote=quote,
                verification=NominalSubjectInventory.model_validate(
                    {
                        "kind": "nominal_subject_inventory",
                        "scope": "nominal_cad",
                        "coverage": "all_made",
                        "source_quote": quote,
                        "items": [
                            {
                                "subject_id": name,
                                "subject_role": "designed",
                                "required_name": name,
                                "quantity": 1,
                            }
                        ],
                    }
                ),
            )
        )
    return Contract(requirements=rows), " ".join(quotes)


def test_global_conflicting_targets_fail_closed_before_grounding_changes_rows():
    contract, source = conflicting_inventories()
    before = contract.model_dump(mode="json")
    assert any(
        row["requirement_id"] is None and row["code"] == "conflicting_targets"
        for row in verification_diagnostics(contract, source, None)
    )
    with pytest.raises(ValueError, match="Conflicting frozen all-made"):
        ground_contract(contract, source)
    assert contract.model_dump(mode="json") == before


def test_bound_customer_nominal_interpretation_is_preserved_with_full_context():
    from tests.unit.workspace.test_customer_interpretation import SOURCE, wire

    from forge_mind.workspace.source_delivery_contract_draft import lower_source_delivery_draft

    interpreted = lower_source_delivery_draft(wire(), SOURCE)
    before = interpreted.model_dump(mode="json")
    assert (
        ground_contract(interpreted.contract, SOURCE, attribution=interpreted.attribution)
        == interpreted.contract
    )
    assert interpreted.model_dump(mode="json") == before
