"""Explicit test encoding for the prospective source-delivery profile."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy

from tests.unit.workspace.compact_source_fixtures import compact_wire_fixture


def source_delivery_wire_fixture(value, *, digital_requirement_ids=()):
    """Select digital rows explicitly; never normalize an unselected legacy error."""
    canonical = value.model_dump(mode="json") if hasattr(value, "model_dump") else deepcopy(value)
    selected = set(digital_requirement_ids)
    requirements = canonical["contract"]["requirements"]
    if selected - {row["id"] for row in requirements}:
        raise ValueError("Explicit digital fixture IDs must identify existing requirements")
    for row in requirements:
        criterion = row.get("verification") or {}
        digital = criterion.get("kind") == "digital_delivery"
        if digital and row["id"] not in selected:
            raise ValueError("Select each legacy digital fixture ID explicitly")
        if row["id"] in selected and (
            not digital or criterion["source_quote"] != row["source_quote"]
        ):
            raise ValueError("Selected digital fixture needs its exact matching criterion quote")
    wire = compact_wire_fixture(canonical, source_only_requirement_ids=selected)
    criteria = {row["id"]: row for row in wire.pop("digital_delivery")}
    direct = []
    normal = []
    for row in wire["requirements"]:
        if row["id"] not in selected:
            normal.append(row)
            continue
        criterion = criteria[row["verification_id"]]
        assert criterion["quote_ref"] == row["quote_ref"]
        direct.append(
            {
                "id": row["id"],
                "source": row["source"],
                "critical": row["critical"],
                "quote_ref": row["quote_ref"],
                "subject_ids": deepcopy(row["subject_ids"]),
                "deliverable": criterion["deliverable"],
            }
        )
    wire["requirements"] = normal
    wire["digital_requirements"] = direct
    return wire


@contextmanager
def legacy_compact_profile(monkeypatch):
    """Explicit test-only legacy seam; never called by fresh-profile test cases."""
    from forge_mind.workspace import source_delivery_contract_draft as initial
    from forge_mind.workspace.compact_source_contract_draft import (
        COMPACT_SOURCE_DRAFT_INSTRUCTIONS,
        CompactSourceContractDraft,
        compact_source_proposal_record,
        lower_compact_source_draft,
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(initial, "SourceDeliveryContractDraft", CompactSourceContractDraft)
        scoped.setattr(
            initial, "SOURCE_DELIVERY_DRAFT_INSTRUCTIONS", COMPACT_SOURCE_DRAFT_INSTRUCTIONS
        )
        scoped.setattr(initial, "lower_source_delivery_draft", lower_compact_source_draft)
        scoped.setattr(initial, "source_delivery_proposal_record", compact_source_proposal_record)
        yield
