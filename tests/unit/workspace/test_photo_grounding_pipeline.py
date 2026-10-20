"""Retained actual003 source admission through real grounding, without CAD/provider work."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_customer_references import reference

from forge_mind.workspace.customer_interpretation import validate_server_interpretation_proposal
from forge_mind.workspace.document import digest
from forge_mind.workspace.intent_attribution_service import record_proposal
from forge_mind.workspace.measure import ground_contract
from forge_mind.workspace.models import Contract
from forge_mind.workspace.photo_nominal import validate_photo_source_replay
from forge_mind.workspace.source_delivery_contract_draft import (
    PhotoSourceContractDraftV2,
    preflight_source_delivery_draft,
)
from forge_mind.workspace.store import Store

# Source-only extraction retained from actual003. This is not a measured image result.
SOURCE = (
    "Copy this 50 by 30 mm rectangular plate, 5 mm thick. Both visible holes go straight through."
)
RETAINED_RAW = json.loads(r"""
{
  "canonical_requirements": [],
  "coverage_notes": [
    "Both owner clauses remain separate critical photo duties. The subject designation is rectangular plate.",
    "The declared image-plane frame assigns 50 mm to plane_u, 30 mm to plane_v, and 5 mm to thickness.",
    "The word Both supplies the count of two holes. Their straight-through character remains an explicit duty.",
    "The owner supplies no hole diameters, hole centers, or tolerances. Visual estimates may guide the copy but are not owner dimensional facts.",
    "Server delivery policy is separate metadata. It supplies no owner quotes, nominal acceptance authority, or additional owner requirements."
  ],
  "digital_requirements": [],
  "fact_groups": [
    {
      "candidate_subject_ids": [],
      "quote_ref": "q1",
      "requirement_ids": [
        "r1"
      ],
      "subject_id": "plate",
      "unit": "mm",
      "values": [
        {
          "id": "f_plane_u",
          "property": "plane_u",
          "value": 50.0
        },
        {
          "id": "f_plane_v",
          "property": "plane_v",
          "value": 30.0
        },
        {
          "id": "f_thickness",
          "property": "thickness",
          "value": 5.0
        }
      ]
    },
    {
      "candidate_subject_ids": [],
      "quote_ref": "q2",
      "requirement_ids": [
        "r2"
      ],
      "subject_id": "plate",
      "unit": "holes",
      "values": [
        {
          "id": "f_hole_count",
          "property": "hole_count",
          "value": 2.0
        }
      ]
    }
  ],
  "fact_order": [
    "f_plane_u",
    "f_plane_v",
    "f_thickness",
    "f_hole_count"
  ],
  "issues": [],
  "nominal_delivery_scope": [],
  "nominal_perforated_box": [],
  "nominal_subject_inventory": [],
  "nominal_targets": [],
  "nominal_through_hole_pattern": [],
  "photo_reference": {
    "role": "copy",
    "source_quote": "Copy this 50 by 30 mm rectangular plate, 5 mm thick."
  },
  "photo_requirements": [
    {
      "critical": true,
      "facets": [
        "visible_profile",
        "rectangular_outline",
        "extent_u",
        "extent_v",
        "thickness"
      ],
      "frame_policy": "image_plane_xy",
      "hole_count_fact_id": "f_hole_count",
      "id": "r1",
      "quote_ref": "q1",
      "size_fact_ids": [
        "f_plane_u",
        "f_plane_v",
        "f_thickness"
      ],
      "source": "request",
      "subject_ids": [
        "plate"
      ]
    },
    {
      "critical": true,
      "facets": [
        "opening_count",
        "straight_through"
      ],
      "frame_policy": "image_plane_xy",
      "hole_count_fact_id": "f_hole_count",
      "id": "r2",
      "quote_ref": "q2",
      "size_fact_ids": [
        "f_plane_u",
        "f_plane_v",
        "f_thickness"
      ],
      "source": "request",
      "subject_ids": [
        "plate"
      ]
    }
  ],
  "quotes": [
    {
      "id": "q1",
      "text": "Copy this 50 by 30 mm rectangular plate, 5 mm thick."
    },
    {
      "id": "q2",
      "text": "Both visible holes go straight through."
    }
  ],
  "requirement_order": [
    "r1",
    "r2"
  ],
  "requirements": [],
  "subjects": [
    {
      "id": "plate",
      "label": "rectangular plate",
      "quote_ref": "q1",
      "role": "designed"
    }
  ]
}
""")


def test_retained_photo_compound_duties_survive_real_grounding_and_atomic_admission(tmp_path):
    # These identities tie the portable fixture to the immutable failed actual003 record.
    assert digest(SOURCE) == "058948ac1a31cc0e77a1794fc1750c8c1eb2d9064061a9df2ac3233f86604649"
    assert (
        digest(RETAINED_RAW) == "f480539c43a576754acc1bae1ebb138b81a05b4eec5e12230ca0b91f5972deec"
    )
    before = deepcopy(RETAINED_RAW)
    preflight = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(before), SOURCE
    )
    assert preflight.compilation["status"] == "ready", preflight.compilation
    draft = preflight.draft
    assert draft is not None
    assert len(draft.contract.requirements) == 2
    assert [row.id for row in draft.contract.requirements] == ["r1", "r2"]
    assert [row.verification.claims for row in draft.contract.requirements] == [
        ["visible_profile", "rectangular_outline", "extent_u", "extent_v", "thickness"],
        ["opening_count", "straight_through"],
    ]
    assert all(row.critical and row.verification is not None for row in draft.contract.requirements)
    # The sibling count belongs to r2; sizes belong to r1. Neither fact link is broadened.
    assert [fact.requirement_ids for fact in draft.attribution.facts] == [["r1"]] * 3 + [["r2"]]
    grounded = ground_contract(draft.contract, SOURCE, attribution=draft.attribution)
    assert grounded.model_dump(mode="json") == draft.contract.model_dump(mode="json")
    assert all(row.verification is not None for row in grounded.requirements)
    assert not any("Typed verification unavailable" in row.text for row in grounded.requirements)

    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "b" * 32,
            "prompt": SOURCE,
            "reference": reference(known_dimension=None),
            "approval_required": False,
        },
    )
    sid = session["id"]
    job = store.claim()
    assert job is not None
    compilation = {
        **preflight.compilation,
        "job_id": job["id"],
        "source_hash": digest(SOURCE),
        "source_proposal": preflight.source_proposal,
    }
    store.update(sid, intent_compilation=compilation)
    current = store.get(sid, "owner")
    validate_server_interpretation_proposal(current, job["id"], SOURCE, grounded, draft.attribution)
    validate_photo_source_replay(
        current, job["id"], SOURCE, grounded, draft.attribution, require_photo=True
    )
    admitted = record_proposal(
        store,
        sid,
        "owner",
        job["id"],
        draft.attribution,
        grounded,
        SOURCE,
        store.edit_basis(sid, "owner"),
    )
    assert admitted["status"] == "interpreted"
    assert Contract.model_validate(admitted["proposed_contract"]) == grounded
    assert admitted["attribution"] == draft.attribution.model_dump(mode="json")
    saved = store.get(sid, "owner")
    assert saved["intent_compilation"]["source_proposal"]["raw_response"] == before
    assert saved["customer_reference"]["known_dimension"] is None
    assert saved["approval_required"] is False
    assert not saved["revisions"] and saved.get("accepted_revision") is None
    assert before == RETAINED_RAW


def test_malformed_photo_source_still_fails_before_grounding_or_admission():
    raw = deepcopy(RETAINED_RAW)
    raw["fact_groups"][0]["values"][0]["value"] = 51.0
    preflight = preflight_source_delivery_draft(
        PhotoSourceContractDraftV2.model_validate(raw), SOURCE
    )
    assert preflight.compilation["status"] == "invalid_interpretation"
    assert any(
        item["code"] == "invalid_source_lowering" for item in preflight.compilation["diagnostics"]
    )
    with pytest.raises(ValueError):
        from forge_mind.workspace.source_delivery_contract_draft import lower_source_delivery_draft

        lower_source_delivery_draft(PhotoSourceContractDraftV2.model_validate(raw), SOURCE)
    assert RETAINED_RAW["fact_groups"][0]["values"][0]["value"] == 50.0


def test_actual_photo_intake_reaches_atomic_admission_with_real_grounder(tmp_path, monkeypatch):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.customer_references import reference_image
    from forge_mind.workspace.executor import Executor

    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "c" * 32,
            "prompt": SOURCE,
            "reference": reference(known_dimension=None),
            "approval_required": False,
        },
    )
    calls = []

    class ScriptedSource:
        def generate_structured_with_image(self, **kwargs):
            calls.append(kwargs)
            assert len(calls) == 1, "A valid source extraction needs no corrective model call"
            assert kwargs["output_model"] is PhotoSourceContractDraftV2
            return PhotoSourceContractDraftV2.model_validate(RETAINED_RAW), LLMUsage(
                "retained-actual003-source", 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: ScriptedSource())

    def forbidden(*args, **kwargs):
        pytest.fail("Source admission cannot execute CAD, native workers or provider services")

    monkeypatch.setattr(service, "build", forbidden)
    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbidden)
    job = store.claim()
    assert job is not None
    # Zero construction actions retain the actual extraction -> grounding -> admission path.
    service._run_agent_job({**job, "max_actions": 0, "max_seconds": 60})
    saved = store.get(session["id"], "owner")
    assert len(calls) == len(saved["usage"]) == 1
    assert calls[0]["image_png"] == reference_image(store, session["id"], "owner")[0]
    assert saved["intent_compilation"]["status"] == "ready"
    assert saved["intent_compilation"]["source_proposal"]["raw_response"] == RETAINED_RAW
    assert saved["intent_attribution"]["status"] == "interpreted", saved["events"]
    admitted = saved["intent_attribution"]["proposed_contract"]["requirements"]
    assert [row["id"] for row in admitted] == ["r1", "r2"]
    assert all(row["critical"] and row["verification"] is not None for row in admitted)
    assert [row["source_quote"] for row in admitted] == [
        row["text"] for row in RETAINED_RAW["quotes"]
    ]
    assert saved["customer_reference"]["known_dimension"] is None
    assert saved["approval_required"] is False
    assert not saved["revisions"] and saved.get("accepted_revision") is None
