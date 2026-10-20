"""Closed owner-envelope grounding; no model, CAD or worker runs."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from forge_mind.workspace.intent_attribution import IntentAttribution
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Contract

# Portable baseline004 call0000 scalar proposals and call0002 original source/attribution.
# Historical already-grounded stored rows are never upgraded.
BASELINE = json.loads(r"""
{
  "source": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers. Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z. Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype. Use conventional milling, turning and drilling. No combustion qualification is requested.",
  "contract": {
    "requirements": [
      {
        "id": "r1",
        "text": "The assembly must be machinable.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r2",
        "text": "Include a hand-turnable crankshaft.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r3",
        "text": "Include a camshaft.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r4",
        "text": "Include pistons.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r5",
        "text": "Include rods.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r6",
        "text": "Include bearings.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r7",
        "text": "Provide lubrication paths.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r8",
        "text": "Include removable service covers.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r9",
        "text": "Provide four cylinders.",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r10",
        "text": "Keep the assembled X extent at or below 200 mm.",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "source": "request",
        "metric": "extent_x",
        "comparator": "le",
        "value": 200.0,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "x",
        "center": null,
        "position_tolerance": 1e-06,
        "critical": true
      },
      {
        "id": "r11",
        "text": "Keep the assembled Y extent at or below 120 mm.",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "source": "request",
        "metric": "extent_y",
        "comparator": "le",
        "value": 120.0,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "y",
        "center": null,
        "position_tolerance": 1e-06,
        "critical": true
      },
      {
        "id": "r12",
        "text": "Keep the assembled Z extent at or below 160 mm.",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "source": "request",
        "metric": "extent_z",
        "comparator": "le",
        "value": 160.0,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 1e-06,
        "critical": true
      },
      {
        "id": "r13",
        "text": "Arrange the cylinders in a row along X.",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r14",
        "text": "Orient the cylinder axes along Z.",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r15",
        "text": "Use a machinable aluminum block.",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r16",
        "text": "Use steel shafts.",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r17",
        "text": "Provide bronze bushings for the shafts.",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r18",
        "text": "The bronze bushings must be replaceable.",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r19",
        "text": "Choose the remaining dimensions for a desktop mechanical prototype.",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      },
      {
        "id": "r20",
        "text": "Use conventional milling, turning and drilling.",
        "source_quote": "Use conventional milling, turning and drilling.",
        "source": "request",
        "metric": "unmeasured",
        "comparator": "eq",
        "value": null,
        "tolerance": 0.0,
        "part_id": null,
        "axis": "z",
        "center": null,
        "position_tolerance": 0.1,
        "critical": true
      }
    ],
    "coverage_notes": [
      "The source designates a small inline-four engine assembly and a desktop mechanical prototype. These designations do not establish unspecified engine operation.",
      "The source states: No combustion qualification is requested.",
      "The named crankshaft and camshaft are interpreted as the steel shafts. Bearings and bronze bushings may overlap; the source does not require separate bearing components.",
      "Cylinders are designed features; their subject record does not require separate cylinder parts. The source provides no exact saved part names or complete occurrence inventory.",
      "Remaining dimensions are delegated to the designer. The source gives no bore, stroke, shaft diameter, fits, clearances, material grades, lubrication details or manufacturing tolerances.",
      "No reference or environment object, customer photo, or explicit owner file deliverable is provided. Server delivery choices remain separate metadata. No nominal-only acceptance restriction is stated.",
      "Correction: f1 retains its cylinder count, subject and unit. Its source quote now includes both adjacent source sentences, so the literal unit cylinders occurs in the quote.",
      "The retained position_tolerance fields do not add dimensional allowances. No hole selector is active, and all envelope tolerances remain zero."
    ]
  },
  "attribution": {
    "subjects": [
      {
        "id": "s1",
        "label": "Small inline-four engine assembly; desktop mechanical prototype",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s2",
        "label": "Block",
        "role": "designed",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype."
      },
      {
        "id": "s3",
        "label": "Crankshaft",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s4",
        "label": "Camshaft",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s5",
        "label": "Pistons",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s6",
        "label": "Rods",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s7",
        "label": "Bearings",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s8",
        "label": "Service covers",
        "role": "designed",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers."
      },
      {
        "id": "s9",
        "label": "Bushings",
        "role": "designed",
        "source_quote": "Use a machinable aluminum block and steel shafts with replaceable bronze bushings; choose the remaining dimensions for a desktop mechanical prototype."
      },
      {
        "id": "s10",
        "label": "Cylinders",
        "role": "designed",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z."
      }
    ],
    "facts": [
      {
        "id": "f1",
        "subject_id": "s10",
        "candidate_subject_ids": [],
        "property": "cylinder_count_from_inline_four",
        "value": 4.0,
        "unit": "cylinders",
        "source_quote": "Design a small machinable inline-four engine assembly with a hand-turnable crankshaft, camshaft, pistons, rods, bearings, lubrication paths and removable service covers. Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "requirement_ids": [
          "r9"
        ]
      },
      {
        "id": "f2",
        "subject_id": "s1",
        "candidate_subject_ids": [],
        "property": "maximum_assembled_extent_x",
        "value": 200.0,
        "unit": "mm",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "requirement_ids": [
          "r10"
        ]
      },
      {
        "id": "f3",
        "subject_id": "s1",
        "candidate_subject_ids": [],
        "property": "maximum_assembled_extent_y",
        "value": 120.0,
        "unit": "mm",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "requirement_ids": [
          "r11"
        ]
      },
      {
        "id": "f4",
        "subject_id": "s1",
        "candidate_subject_ids": [],
        "property": "maximum_assembled_extent_z",
        "value": 160.0,
        "unit": "mm",
        "source_quote": "Keep the assembled envelope within X 200, Y 120, Z 160 mm, with cylinders in a row along X and their axes along Z.",
        "requirement_ids": [
          "r12"
        ]
      }
    ],
    "requirement_subjects": {
      "r1": [
        "s1"
      ],
      "r2": [
        "s1",
        "s3"
      ],
      "r3": [
        "s1",
        "s4"
      ],
      "r4": [
        "s1",
        "s5"
      ],
      "r5": [
        "s1",
        "s6"
      ],
      "r6": [
        "s1",
        "s7"
      ],
      "r7": [
        "s1"
      ],
      "r8": [
        "s1",
        "s8"
      ],
      "r9": [
        "s1",
        "s10"
      ],
      "r10": [
        "s1"
      ],
      "r11": [
        "s1"
      ],
      "r12": [
        "s1"
      ],
      "r13": [
        "s1",
        "s10"
      ],
      "r14": [
        "s1",
        "s10"
      ],
      "r15": [
        "s1",
        "s2"
      ],
      "r16": [
        "s3",
        "s4"
      ],
      "r17": [
        "s3",
        "s4",
        "s9"
      ],
      "r18": [
        "s9"
      ],
      "r19": [
        "s1"
      ],
      "r20": [
        "s1"
      ]
    },
    "issues": []
  }
}
""")

QUOTE = BASELINE["contract"]["requirements"][9]["source_quote"]
IDS = ("r10", "r11", "r12")


def proposed(clause=QUOTE, *, values=(200.0, 120.0, 160.0), unit="mm"):
    raw = deepcopy(BASELINE)
    raw["source"] = raw["source"].replace(QUOTE, clause)
    for row in raw["contract"]["requirements"]:
        if row["id"] in IDS:
            index = IDS.index(row["id"])
            row.update(value=values[index], source_quote=clause)
    for subject in raw["attribution"]["subjects"]:
        subject["source_quote"] = subject["source_quote"].replace(QUOTE, clause)
    for index, fact in enumerate(raw["attribution"]["facts"][1:]):
        fact.update(value=values[index], unit=unit, source_quote=clause)
    return (
        raw["source"],
        Contract.model_validate(raw["contract"]),
        IntentAttribution.model_validate(raw["attribution"]),
    )


def rows(contract):
    return [row for row in contract.requirements if row.id in IDS]


def decision(requirement, source, attribution):
    from forge_mind.workspace.source_envelope import ground_source_envelope

    return ground_source_envelope(requirement, source, attribution)


def test_real_baseline_source_preserves_exact_xyz_bounds_and_all_twenty_duties():
    source, contract, attribution = proposed()
    original = contract.model_dump()
    result = ground_contract(contract, source, attribution=attribution)
    assert len(result.requirements) == 20
    assert [(row.metric, row.comparator, row.value) for row in rows(result)] == [
        ("extent_x", "le", 200.0),
        ("extent_y", "le", 120.0),
        ("extent_z", "le", 160.0),
    ]
    for before, after in zip(contract.requirements, result.requirements, strict=True):
        assert after.id == before.id and after.critical
        assert after.source_quote == before.source_quote
        if before.id not in IDS:
            assert after == before
    assert contract.model_dump() == original


@pytest.mark.parametrize(
    "unit,scale", [("mm", 1), ("cm", 10), ("in", 25.4), ("inch", 25.4), ("inches", 25.4)]
)
def test_common_unit_is_bound_to_tuple_not_preposition_in_orientation(unit, scale):
    clause = f"Keep the assembled envelope within X 20, Y 12, Z 16 {unit}, with cylinders in a row along X and their axes along Z."
    source, contract, attribution = proposed(clause, values=(20.0, 12.0, 16.0), unit=unit)
    for row in rows(contract):
        row.value *= scale
    result = ground_contract(contract, source, attribution=attribution)
    assert [row.metric for row in rows(result)] == ["extent_x", "extent_y", "extent_z"]
    assert [row.value for row in rows(result)] == [20 * scale, 12 * scale, 16 * scale]


def test_reordered_axis_entries_use_labels_and_do_not_change_source_targets():
    source, contract, attribution = proposed(
        "Keep the assembled envelope within Z 160, X 200, Y 120 mm."
    )
    result = ground_contract(contract, source, attribution=attribution)
    assert [row.metric for row in rows(result)] == ["extent_x", "extent_y", "extent_z"]


@pytest.mark.parametrize(
    "mutation",
    [
        "swapped",
        "inflated_inch",
        "ge",
        "eq",
        "tolerance",
        "selector",
        "center",
        "no_attribution",
        "reference_role",
        "wrong_subject",
        "forged_role",
        "fact_subject",
        "fact_value",
        "converted_fact_unit",
    ],
)
def test_recognized_family_refuses_ineligible_target_or_source_authority(mutation):
    source, contract, attribution = proposed()
    target = rows(contract)[0]
    if mutation == "swapped":
        target.value = 120
    elif mutation == "inflated_inch":
        target.value = 200 * 25.4
    elif mutation in {"ge", "eq"}:
        target.comparator = mutation
    elif mutation == "tolerance":
        target.tolerance = 1
    elif mutation == "selector":
        target.part_id = "block"
    elif mutation == "center":
        target.center = (0, 0)
    elif mutation == "no_attribution":
        attribution = None
    elif mutation == "reference_role":
        attribution.subjects[0].role = "reference"
    elif mutation == "wrong_subject":
        attribution.requirement_subjects[target.id] = ["s2"]
    elif mutation == "forged_role":
        old = attribution.subjects[0].source_quote
        source = source.replace(old, "Use the reference assembly only.")
        attribution.subjects[0].source_quote = "Use the reference assembly only."
    elif mutation == "fact_subject":
        attribution.facts[1].subject_id = "s2"
    elif mutation == "converted_fact_unit":
        attribution.facts[1].value = 200 / 25.4
        attribution.facts[1].unit = "in"
    else:
        attribution.facts[1].value = 120
    assert decision(target, source, attribution).state == "unsupported"
    grounded = ground_contract(contract, source, attribution=attribution)
    assert rows(grounded)[0].metric == "unmeasured"
    assert len(grounded.requirements) == 20


@pytest.mark.parametrize(
    "prefix",
    [
        "Do not ",
        "Never ",
        "For reference only: ",
        'Supplier says: "',
        "If possible, ",
        "Unless approved, ",
    ],
)
def test_full_original_clause_prevents_stripped_negation_reference_or_condition(prefix):
    source, contract, attribution = proposed()
    source = source.replace(QUOTE, prefix + QUOTE)
    assert all(row.source_quote in source for row in rows(contract))
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


@pytest.mark.parametrize(
    "clause",
    [
        "Do not keep the assembled envelope at maximum X 200, Y 120, Z 160 mm.",
        "Keep the assembled envelope within 1 mm of nominal X 200, Y 120, Z 160 mm maximum.",
        "Keep the assembled envelope within X 200, X 120, Z 160 mm.",
        "Keep the assembled envelope within X 200, Y 120 mm.",
        "Keep the assembled envelope within X 200, Y 120, Z 160.",
        "Keep the assembled envelope within X 200 cm, Y 120, Z 160 mm.",
        "Keep the assembled envelope within X -200, Y 120, Z 160 mm maximum.",
        "Keep the assembled envelope within X 200e0, Y 120, Z 160 mm maximum.",
        "Keep the assembled envelope within X 200 ± 1, Y 120, Z 160 mm maximum.",
        "Keep the assembled envelope within approximately X 200, Y 120, Z 160 mm maximum.",
        "Keep the assembled envelope within X 200, Y 120, Z 160 mm except larger parts are allowed.",
        "Keep the reference envelope within X 200, Y 120, Z 160 mm maximum.",
    ],
)
def test_malformed_envelope_cannot_escape_through_legacy_maximum_or_number_bag(clause):
    source, contract, attribution = proposed(clause)
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))
    assert all(decision(row, source, attribution).state == "unsupported" for row in rows(contract))


def test_conflicting_original_envelope_clauses_refuse_even_when_quoted_one_matches():
    source, contract, attribution = proposed()
    source += " Keep the assembled envelope within X 210, Y 120, Z 160 mm."
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


def test_structured_result_reports_exact_bound_span_and_safe_reason_code():
    source, contract, attribution = proposed()
    result = decision(rows(contract)[0], source, attribution)
    assert result.state == "supported" and result.code == "envelope_bound"
    assert source[slice(*result.source_span)].startswith("Keep the assembled envelope within X 200")
    assert result.bounds_mm == (("x", 200.0), ("y", 120.0), ("z", 160.0))


def test_unrelated_legacy_dimensions_keep_their_existing_authority():
    from forge_mind.workspace.models import Requirement

    source = "Width maximum 20 mm."
    target = Requirement(
        id="width", text="Width", source_quote=source, metric="extent_x", comparator="le", value=20
    )
    assert decision(target, source, None).state == "not_applicable"
    assert (
        ground_contract(Contract(requirements=[target]), source).requirements[0].metric
        == "extent_x"
    )


def test_previously_unmeasured_stored_rows_are_not_upgraded():
    source, contract, attribution = proposed()
    for row in rows(contract):
        row.metric = "unmeasured"
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


@pytest.mark.parametrize(
    "context", ['Supplier instruction:\n"\n', "For reference only:\n", "Do not:\n"]
)
def test_line_breaks_cannot_strip_quoted_or_qualified_original_context(context):
    source, contract, attribution = proposed()
    source = source.replace(QUOTE, context + QUOTE)
    assert all(
        row.metric == "unmeasured"
        for row in rows(ground_contract(contract, source, attribution=attribution))
    )


def test_duplicate_identical_owner_clauses_remain_ambiguous():
    source, contract, attribution = proposed()
    source += " " + QUOTE
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


def test_forged_designed_role_cannot_override_reference_assembly_label():
    source, contract, attribution = proposed()
    attribution.subjects[0].label = "Reference assembly"
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


def test_real_scripted_source_intake_retains_envelope_without_an_extra_model_call(
    tmp_path, monkeypatch
):
    from tests.unit.workspace.source_delivery_fixtures import source_delivery_wire_fixture
    from tests.unit.workspace.test_source_contract_draft_lifecycle import run_source_intake

    from forge_mind.workspace.intent_attribution import ContractDraft

    source, contract, attribution = proposed()
    wire = source_delivery_wire_fixture(ContractDraft(contract=contract, attribution=attribution))
    result = run_source_intake(tmp_path, monkeypatch, source, [wire])
    assert len(result.calls) == 1
    assert result.saved["contract"] is None
    assert result.saved["intent_compilation"]["status"] == "ready"
    saved = Contract.model_validate(result.saved["intent_attribution"]["proposed_contract"])
    assert len(saved.requirements) == 20
    assert [row.metric for row in rows(saved)] == ["extent_x", "extent_y", "extent_z"]
    assert [row.value for row in rows(saved)] == [200, 120, 160]
    assert result.saved.get("accepted_revision") is None


@pytest.mark.parametrize(
    "suffix",
    [
        ", with mounting slots parallel to X and their axes along Z.",
        ", with openings facing negative Z.",
        ", with the long axis along Y and supports aligned with X.",
    ],
)
def test_generic_orientation_suffixes_do_not_require_engine_specific_words(suffix):
    source, contract, attribution = proposed(
        "Keep the assembled envelope within X 200, Y 120, Z 160 mm" + suffix
    )
    result = ground_contract(contract, source, attribution=attribution)
    assert [row.metric for row in rows(result)] == ["extent_x", "extent_y", "extent_z"]


@pytest.mark.parametrize(
    "prefix",
    [
        "Do not follow the next instruction. ",
        "These are reference instructions. ",
        "The supplier instructions follow.\n",
        "Ignore the next instruction. ",
        "Disregard the following sentence. ",
        "The next clause is illustrative. ",
        "The paragraph below is a discarded specification. ",
        "The text below is illustrative only. ",
        "Obsolete specification follows. ",
    ],
)
def test_complete_prior_qualifying_sentences_cannot_launder_owner_authority(prefix):
    source, contract, attribution = proposed()
    source = source.replace(QUOTE, prefix + QUOTE)
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


@pytest.mark.parametrize(
    "suffix",
    [
        ", with maximum X 250 mm instead.",
        ", with the larger envelope parallel to X.",
        ", with the tolerance aligned with X.",
        ", with missing stock allowed.",
    ],
)
def test_unsupported_or_contradictory_suffix_remains_unmeasured(suffix):
    source, contract, attribution = proposed(
        "Keep the assembled envelope within X 200, Y 120, Z 160 mm" + suffix
    )
    result = ground_contract(contract, source, attribution=attribution)
    assert all(row.metric == "unmeasured" for row in rows(result))


@pytest.mark.parametrize(
    "introduction,supported",
    [
        (
            "Build a modular fixture assembly with a frame, mounting slots and removable supports.",
            True,
        ),
        ("Create an assembly.", True),
        ("Design an assembly as an obsolete specification.", False),
        ("Design an assembly with parts which are illustrative only.", False),
    ],
)
def test_closed_design_prelude_requires_positive_source_even_with_matching_role_labels(
    introduction, supported
):
    source, contract, attribution = proposed()
    previous = attribution.subjects[0].source_quote
    source = source.replace(previous, introduction)
    for requirement in contract.requirements:
        requirement.source_quote = requirement.source_quote.replace(previous, introduction)
    for subject in attribution.subjects:
        subject.source_quote = subject.source_quote.replace(previous, introduction)
    for fact in attribution.facts:
        fact.source_quote = fact.source_quote.replace(previous, introduction)
    attribution.subjects[0].label = "Designed assembly"
    result = ground_contract(contract, source, attribution=attribution)
    assert all((row.metric != "unmeasured") is supported for row in rows(result))
