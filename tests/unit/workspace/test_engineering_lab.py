"""Owner boundaries, executable challenges and cross-tool scientific workflow controls."""

from __future__ import annotations

import asyncio
import copy
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.design_reasoning import affected_paths
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.experiment_service import synthetic_tutorial
from forge_mind.workspace.lab_records import capture_binding, read_record, save_record
from forge_mind.workspace.lab_service import check_lab_job
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.synthesis_models import example_synthesis_request


@pytest.fixture
def lab(tmp_path, monkeypatch):
    service = WorkspaceService(
        Store(tmp_path),
        Executor("trusted_local"),
        client_factory=lambda **_: pytest.fail("Unexpected model call"),
    )
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    sid = service.store.create(
        "local-owner", "Keep the original requirement", {"requirements": []}
    )["id"]
    rid = "a" * 32
    doc = DesignDocument.model_validate(
        {
            "name": "Rectangular experimental specimen",
            "variables": {"thickness": 4, "unused": 9},
            "features": [
                {
                    "id": "beam",
                    "name": "Test beam",
                    "kind": "box",
                    "params": {"x": 180, "y": 20, "z": 4},
                    "bindings": {"z": "thickness"},
                }
            ],
            "outputs": ["beam"],
        }
    ).model_dump()
    contract = {
        "requirements": [
            {
                "id": "keep",
                "text": "Keep the original requirement",
                "source_quote": "Keep the original requirement",
                "metric": "unmeasured",
            }
        ]
    }
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "document": doc,
            "contract": contract,
            "measurement": {
                "geometry_valid": True,
                "artifact_hashes": {"part.step": "b" * 64},
                "facts": {"solid_count": 1},
                "findings": [{"requirement_id": "keep", "status": "unverified"}],
            },
        },
    )
    service.store.update(sid, selected_revision=rid, contract=contract)
    return service, TestClient(create_app(local=True, service=service)), sid, rid, doc


def post(client, sid, operation, payload):
    response = client.post(f"/workspace/sessions/{sid}/lab/{operation}", json=payload)
    assert response.status_code in {200, 202}, response.text
    return response.json()


def test_document_explanation_retains_all_variables_including_unbound_controls(lab):
    _service, client, sid, rid, _document = lab
    response = client.get(f"/workspace/sessions/{sid}/lab/explain", params={"revision_id": rid})
    assert response.status_code == 200, response.text
    explanation = response.json()
    assert explanation["part"]["selection_scope"] == "document"
    variables = {row["name"]: row for row in explanation["variables"]}
    assert set(variables) == {"thickness", "unused"}
    assert variables["thickness"]["impact"]["features"] == ["beam"]
    assert all(not items for items in variables["unused"]["impact"].values())


def test_challenge_preserves_contract_and_only_queues_exact_proposal(lab):
    service, client, sid, rid, document = lab
    before = copy.deepcopy(service.store.session_record(sid, "local-owner"))
    record = post(
        client,
        sid,
        "challenges",
        {
            "revision_id": rid,
            "input_version": 0,
            "question": "Make the specimen thicker",
            "variables": {"thickness": 5},
        },
    )
    assert record["impact"]["features"] == ["beam"]
    assert record["impact"]["body_indices"] == [0]
    assert record["candidate_document"]["variables"]["thickness"] == 5
    assert service.store.revision(sid, rid)["document"] == document
    assert service.store.session_record(sid, "local-owner")["contract"] == before["contract"]
    result = post(client, sid, f"challenges/{record['id']}/apply", {})
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (result["job_id"],)).fetchone()[0]
        )
    assert payload["contract"] == before["contract"]
    assert payload["expected_current_revision"] == rid
    assert digest(payload["document"]) == record["candidate_document_hash"]
    assert payload["preview_only"] is True
    check_lab_job(service, {**payload, "session": sid, "owner": "local-owner"})
    payload["document"]["variables"]["thickness"] = 6
    with pytest.raises(WorkspaceError, match="diverges"):
        check_lab_job(service, {**payload, "session": sid, "owner": "local-owner"})


def test_changed_head_inputs_and_unknown_references_fail_closed(lab):
    service, client, sid, rid, _ = lab
    invalid = client.post(
        f"/workspace/sessions/{sid}/lab/rationales",
        json={
            "revision_id": rid,
            "input_version": 0,
            "target": {"feature_id": "beam"},
            "purpose": "Support load",
            "requirement_ids": ["invented"],
        },
    )
    assert invalid.status_code == 409
    record = post(
        client,
        sid,
        "challenges",
        {
            "revision_id": rid,
            "input_version": 0,
            "question": "Thicker",
            "variables": {"thickness": 5},
        },
    )
    service.store.update(sid, selected_revision=None)
    assert read_record(service, sid, "local-owner", record["id"])["stale"]
    assert (
        client.post(
            f"/workspace/sessions/{sid}/lab/challenges/{record['id']}/apply", json={}
        ).status_code
        == 409
    )
    with pytest.raises(WorkspaceError, match="not found"):
        read_record(service, sid, "another-owner", record["id"])


def test_selected_nested_part_excludes_unrelated_placement_variables(lab):
    service, client, sid, rid, _ = lab
    parts = [
        {
            "id": key,
            "name": key,
            "features": [
                {"id": "body", "name": key, "kind": "box", "params": {"x": 4, "y": 4, "z": 4}}
            ],
            "output": "body",
        }
        for key in ("left", "right")
    ]
    doc = DesignDocument.model_validate(
        {
            "variables": {"left_offset": 10, "right_offset": 30},
            "assembly": {
                "parts": parts,
                "subassemblies": [
                    {
                        "id": "carrier",
                        "name": "Carrier",
                        "children": [{"id": "pad", "definition": "left"}],
                    }
                ],
                "roots": [
                    {"id": "mounted", "definition": "carrier", "position": ["left_offset", 0, 0]},
                    {"id": "other", "definition": "right", "position": ["right_offset", 0, 0]},
                ],
            },
        }
    )
    assert affected_paths(doc, ["root/mounted"]) == {"mounted/pad"}
    revision = service.store.revision(sid, rid)
    revision["document"] = doc.model_dump()
    revision["measurement"]["facts"] = {
        "solid_count": 2,
        "assembly": {
            "occurrences": [
                {"body_index": 0, "path": "mounted/pad", "definition": "left"},
                {"body_index": 1, "path": "other", "definition": "right"},
            ]
        },
    }
    with service.store.connect() as con:
        con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))
    explanation = client.get(
        f"/workspace/sessions/{sid}/lab/explain", params={"revision_id": rid, "body_index": 0}
    ).json()
    assert [v["name"] for v in explanation["variables"]] == ["left_offset"]
    assert explanation["purpose"] == []
    record = post(
        client,
        sid,
        "challenges",
        {
            "revision_id": rid,
            "input_version": 0,
            "question": "Move parent",
            "variables": {"left_offset": 12},
        },
    )
    assert record["impact"]["body_indices"] == [0]


def test_standalone_body_never_inherits_unrelated_feature_purpose(lab):
    service, client, sid, rid, _ = lab
    post(
        client,
        sid,
        "rationales",
        {
            "revision_id": rid,
            "input_version": 0,
            "target": {"feature_id": "beam"},
            "purpose": "Only the beam supports this load",
        },
    )
    single = client.get(
        f"/workspace/sessions/{sid}/lab/explain", params={"revision_id": rid, "body_index": 0}
    ).json()
    assert single["part"]["feature_id"] == "beam"
    assert [v["name"] for v in single["variables"]] == ["thickness"]
    assert len(single["purpose"]) == 1
    revision = service.store.revision(sid, rid)
    revision["document"]["features"].append(
        {"id": "unrelated", "name": "Unrelated", "kind": "box", "params": {"x": 2, "y": 2, "z": 2}}
    )
    revision["document"]["outputs"].append("unrelated")
    revision["measurement"]["facts"]["solid_count"] = 2
    # Use a distinct revision, preserving the original receipt identity.
    revision["id"] = next_rid = "d" * 32
    service.store.add_revision(sid, revision)
    service.store.update(sid, selected_revision=next_rid)
    post(
        client,
        sid,
        "rationales",
        {
            "revision_id": next_rid,
            "input_version": 0,
            "target": {"feature_id": "beam"},
            "purpose": "Still only the beam supports this load",
        },
    )
    other = client.get(
        f"/workspace/sessions/{sid}/lab/explain", params={"revision_id": next_rid, "body_index": 1}
    ).json()
    assert other["part"]["selection_scope"] == "unmapped_body"
    assert other["purpose"] == []
    assert other["variables"] == []
    assert other["part"]["features"] == []


def test_poll_metadata_excludes_dataset_and_immutable_ancestry_detects_tampering(lab):
    service, client, sid, rid, _ = lab
    binding = capture_binding(service, sid, "local-owner", rid)
    parent = save_record(
        service,
        sid,
        "local-owner",
        "experiment",
        binding,
        {"observations": [1, 2], "csv_text": "large"},
    )
    child = save_record(
        service,
        sid,
        "local-owner",
        "calibration",
        binding,
        {"status": "ambiguous"},
        parents=(parent["id"],),
    )
    listed = client.get(f"/workspace/sessions/{sid}/lab/records").json()["records"]
    assert all("observations" not in r and "csv_text" not in r for r in listed)
    assert all(r["project_id"] == sid for r in listed)
    with service.store.connect() as con:
        con.execute("UPDATE lab_records SET document='{}' WHERE id=?", (parent["id"],))
    with pytest.raises(WorkspaceError, match="integrity"):
        read_record(service, sid, "local-owner", child["id"])


def test_shared_evidence_ancestry_is_verified_once_per_distinct_record(lab, monkeypatch):
    from forge_mind.workspace import lab_records

    service, _, sid, rid, _ = lab
    binding = capture_binding(service, sid, "local-owner", rid)
    records = []
    for index in range(18):
        records.append(
            save_record(
                service,
                sid,
                "local-owner",
                "rationale",
                binding,
                {"purpose": f"Shared evidence derivation {index}"},
                parents=tuple(record["id"] for record in records[-2:]),
            )
        )
    verifications = []
    original = lab_records.digest

    def count_receipts(value):
        if isinstance(value, dict) and value.get("kind") == "rationale":
            verifications.append(value["id"])
        return original(value)

    monkeypatch.setattr(lab_records, "digest", count_receipts)
    assert read_record(service, sid, "local-owner", records[-1]["id"])["receipt_sha256"]
    assert len(verifications) == len(set(verifications)) == len(records)


def test_experiment_to_conditional_design_proposal_and_mapping_controls(lab):
    service, client, sid, rid, _ = lab
    example = synthetic_tutorial()
    experiment = post(
        client,
        sid,
        "experiments",
        {
            "revision_id": rid,
            "expected_input_version": 0,
            **{k: example[k] for k in ("title", "provenance", "protocol", "observations")},
        },
    )
    calibration = post(
        client,
        sid,
        "calibrations",
        {
            "experiment_id": experiment["id"],
            "observation_ids": [o["observation_id"] for o in example["observations"]],
        },
    )
    payload = {
        "variable_name": "thickness",
        "hypothesis": "fixture_translation",
        "condition": {"span_mm": 160, "force_n": 10, "force_uncertainty_n": 0.01},
        "target_displacement_mm": 0.8,
        "min_thickness_mm": 4,
        "max_thickness_mm": 8,
        "mapping_basis": "Direct unmodified box section z dimension",
        "feature_id": "beam",
        "thickness_parameter": "z",
    }
    proposal = post(client, sid, f"calibrations/{calibration['id']}/design-proposal", payload)
    assert proposal["prediction"]["recommended_thickness_mm"] > 4
    assert proposal["physical_qualification"] == "unverified"
    challenge = read_record(service, sid, "local-owner", proposal["challenge_id"], "challenge")
    assert challenge["original_contract_hash"] == digest(
        service.store.revision(sid, rid)["contract"]
    )
    payload["thickness_parameter"] = "x"
    assert (
        client.post(
            f"/workspace/sessions/{sid}/lab/calibrations/{calibration['id']}/design-proposal",
            json=payload,
        ).status_code
        == 409
    )
    payload["thickness_parameter"] = "z"
    payload["target_displacement_mm"] = 0.001
    impossible = client.post(
        f"/workspace/sessions/{sid}/lab/calibrations/{calibration['id']}/design-proposal",
        json=payload,
    )
    assert impossible.status_code == 409, impossible.text
    assert "fixture" in impossible.text.lower()


def test_synthesis_realizes_separate_project_without_moving_original(lab):
    service, client, sid, rid, _ = lab
    request = example_synthesis_request().model_dump()
    comparison = post(
        client, sid, "comparisons", {"revision_id": rid, "input_version": 0, "request": request}
    )
    result = post(
        client,
        sid,
        f"comparisons/{comparison['id']}/realize",
        {"candidate_id": "opposed_screw", "name": "Screw alternative"},
    )
    assert result["project_id"] != sid
    assert service.store.session_record(sid, "local-owner")["selected_revision"] == rid
    project = service.store.session_record(result["project_id"], "local-owner")
    assert project["state"] == "queued"
    assert len(project["contract"]["requirements"]) >= 5
    assert all(r["metric"] == "unmeasured" for r in project["contract"]["requirements"])


def test_mcp_experiment_and_challenge_reach_identical_http_service(lab, tmp_path):
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    service, _, sid, rid, _ = lab
    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=httpx.ASGITransport(app=create_app(local=True, service=service)),
    )

    async def scenario():
        async with Client(create_server(adapter)) as client:
            response = await client.call_tool(
                "preview_design_challenge",
                {
                    "project_id": sid,
                    "challenge": {
                        "revision_id": rid,
                        "input_version": 0,
                        "question": "Increase stiffness",
                        "variables": {"thickness": 5},
                    },
                },
            )
            assert not response.is_error, response.content
            value = response.structured_content
            assert value["candidate_document_hash"]
            assert "candidate_document" not in value
            stored = read_record(service, sid, "local-owner", value["id"])
            assert value["receipt_sha256"] == stored["receipt_sha256"]
            assert stored["candidate_document"]["variables"]["thickness"] == 5

    asyncio.run(scenario())


def test_challenge_of_unaccepted_candidate_survives_transactional_admission(lab):
    service, client, sid, rid, _ = lab
    candidate = copy.deepcopy(service.store.revision(sid, rid))
    candidate["id"] = "c" * 32
    service.store.add_revision(sid, candidate)
    service.store.update(
        sid,
        accepted_revision=rid,
        proposal={
            "revision_id": candidate["id"],
            "based_on": rid,
            "eligible": True,
            "input_version": 0,
            "contract": candidate["contract"],
        },
    )
    challenge = post(
        client,
        sid,
        "challenges",
        {
            "revision_id": candidate["id"],
            "input_version": 0,
            "question": "Improve this candidate",
            "variables": {"thickness": 5},
        },
    )
    queued = post(client, sid, f"challenges/{challenge['id']}/apply", {})
    with service.store.connect() as con:
        payload = json.loads(
            con.execute("SELECT payload FROM jobs WHERE id=?", (queued["job_id"],)).fetchone()[0]
        )
    assert service.store.session_record(sid, "local-owner").get("proposal") is None
    check_lab_job(service, {**payload, "session": sid, "owner": "local-owner"})


@pytest.mark.parametrize("mismatch", ["reused_width_as_span", "coupled_width"])
def test_calibrated_section_mapping_cannot_reuse_or_change_longitudinal_dimension(lab, mismatch):
    from forge_mind.workspace.experiment_models import ExperimentProtocol, Observation
    from forge_mind.workspace.experiments import compare_hypotheses

    service, client, sid, rid, _ = lab
    example = synthetic_tutorial()
    protocol = ExperimentProtocol.model_validate(example["protocol"])
    revision = service.store.revision(sid, rid)
    if mismatch == "reused_width_as_span":
        protocol.width_mm = 180
    else:
        revision["document"]["relations"] = [
            {"name": "coupled_width", "operation": "multiply", "a": "thickness", "b": 5}
        ]
        revision["document"]["features"][0]["bindings"]["y"] = "coupled_width"
        with service.store.connect() as con:
            con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(revision), rid))
    rows = [
        Observation.model_validate(
            {
                **row,
                "displacement_mm": row["force_n"]
                * (row["span_mm"] ** 3 / (3 * 70000 * protocol.width_mm * 4**3 / 12) + 0.005),
            }
        )
        for row in example["observations"]
    ]
    calibration = save_record(
        service,
        sid,
        "local-owner",
        "calibration",
        capture_binding(service, sid, "local-owner", rid),
        {
            "protocol": protocol.model_dump(),
            "provenance": "synthetic",
            **compare_hypotheses(protocol, rows),
        },
    )
    response = client.post(
        f"/workspace/sessions/{sid}/lab/calibrations/{calibration['id']}/design-proposal",
        json={
            "variable_name": "thickness",
            "hypothesis": "fixture_translation",
            "condition": {"span_mm": 160, "force_n": 10, "force_uncertainty_n": 0.01},
            "target_displacement_mm": 0.8 if mismatch == "coupled_width" else 0.1,
            "min_thickness_mm": 4,
            "max_thickness_mm": 10,
            "mapping_basis": "Adversarial mapping control",
            "feature_id": "beam",
            "thickness_parameter": "z",
        },
    )
    assert response.status_code == 409, response.text
    assert "specimen" in response.text or "coupled" in response.text
