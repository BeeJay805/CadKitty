"""Explicit test-only compact encoding; production has no legacy-wire fallback."""

from __future__ import annotations

from copy import deepcopy


def compact_wire_fixture(draft, source_only_requirement_ids=()):
    """Keep canonical escape rows unchanged, including deliberate admission errors."""
    canonical = draft.model_dump(mode="json") if hasattr(draft, "model_dump") else deepcopy(draft)
    wire = {
        "quotes": [],
        "requirements": [],
        "canonical_requirements": [],
        "requirement_order": [],
        "nominal_through_hole_pattern": [],
        "nominal_perforated_box": [],
        "digital_delivery": [],
        "nominal_subject_inventory": [],
        "nominal_delivery_scope": [],
        "subjects": [],
        "fact_groups": [],
        "fact_order": [],
        "issues": deepcopy(canonical["attribution"]["issues"]),
        "coverage_notes": deepcopy(canonical["contract"]["coverage_notes"]),
    }
    quote_ids = {}

    def quote_ref(text):
        if text not in quote_ids:
            identifier = f"q_{len(quote_ids)}"
            quote_ids[text] = identifier
            wire["quotes"].append({"id": identifier, "text": text})
        return quote_ids[text]

    selected = set(source_only_requirement_ids)
    requirement_subjects = canonical["attribution"]["requirement_subjects"]
    requirement_ids = {row["id"] for row in canonical["contract"]["requirements"]}
    # The wire has per-row links. Do not silently erase an unrepresentable fixture error.
    assert requirement_subjects.keys() <= requirement_ids
    assert all(requirement_subjects.values())
    for row in canonical["contract"]["requirements"]:
        wire["requirement_order"].append(row["id"])
        subjects = deepcopy(requirement_subjects.get(row["id"], []))
        if row["id"] not in selected:
            wire["canonical_requirements"].append(
                {"requirement": deepcopy(row), "subject_ids": subjects}
            )
            continue
        criterion = row["verification"]
        assert criterion["kind"] == "digital_delivery"
        assert criterion["source_quote"] == row["source_quote"]
        criterion_id = f"v_{row['id']}"
        reference = quote_ref(row["source_quote"])
        wire["digital_delivery"].append(
            {
                "id": criterion_id,
                "quote_ref": reference,
                "deliverable": criterion["deliverable"],
            }
        )
        wire["requirements"].append(
            {
                "id": row["id"],
                "text": None,
                "source": row["source"],
                "critical": row["critical"],
                "quote_ref": reference,
                "subject_ids": subjects,
                "verification_id": criterion_id,
                "scalar": None,
            }
        )
    assert selected <= set(wire["requirement_order"])
    for subject in canonical["attribution"]["subjects"]:
        wire["subjects"].append(
            {
                "id": subject["id"],
                "label": subject["label"],
                "role": subject["role"],
                "quote_ref": quote_ref(subject["source_quote"]),
            }
        )
    for fact in canonical["attribution"]["facts"]:
        # Single-entry groups preserve original interleaving and malformed links.
        wire["fact_order"].append(fact["id"])
        wire["fact_groups"].append(
            {
                "subject_id": fact["subject_id"],
                "candidate_subject_ids": deepcopy(fact["candidate_subject_ids"]),
                "unit": fact["unit"],
                "quote_ref": quote_ref(fact["source_quote"]),
                "requirement_ids": deepcopy(fact["requirement_ids"]),
                "values": [
                    {"id": fact["id"], "property": fact["property"], "value": fact["value"]}
                ],
            }
        )
    return wire
