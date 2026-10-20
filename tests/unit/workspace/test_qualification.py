"""Independent qualification checks use real receipts and real field-file integrity.

The worker and topology inventory are fixture data, not a physics validation.
"""

import copy
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.qualification import qualification_review
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.structural_service import study_path


def test_structural_volume_and_quadrature_share_declared_world_frame():
    from forge_mind.workspace.structural import analysis_frame, world_field

    local = np.array([[0, 0, 0], [1, 2, 3], [-2, 4, 1], [0, 0, 5]], dtype=float)
    frame = analysis_frame([1, -2, 3])
    origin = np.array([120, -70, 50])
    field = {
        "positions_mm": local.tolist(),
        "volume_positions_mm": local.tolist(),
        "quadrature_positions_mm": [local.mean(axis=0).tolist()],
        "displacements_mm": [[0, 0, 1]] * 4,
        "tetrahedra": [[0, 1, 2, 3]],
        "quadrature_element_ids": [0],
    }
    world_field(field, frame, origin)
    expected = local @ frame.T + origin
    assert field["frame"] == "cad_z_up_mm"
    np.testing.assert_allclose(field["positions_mm"], expected)
    np.testing.assert_allclose(field["volume_positions_mm"], expected)
    np.testing.assert_allclose(field["quadrature_positions_mm"], [expected.mean(axis=0)])
    np.testing.assert_allclose(field["displacements_mm"], np.tile(frame[:, 2], (4, 1)))


@pytest.fixture
def qualification_workspace(tmp_path, monkeypatch):
    from forge_mind.workspace import (
        document_service,
        product_service,
        qualification,
        structural_service,
    )

    contract = Contract(
        requirements=[
            Requirement(
                id=f"frozen_{i}",
                text=f"Independent physical obligation {i}",
                source_quote=f"Independent physical obligation {i}",
                metric="unmeasured",
            )
            for i in range(25)
        ]
    ).model_dump()
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: None)
    monkeypatch.setattr(service, "_load_revision", service.store.revision)
    for module in (qualification, structural_service, product_service):
        monkeypatch.setattr(module, "engine_fingerprint", lambda: "engine")
    sid = service.store.create("local-owner", "Independent qualification fixture", contract)["id"]
    rid = "a" * 32
    parts = []
    for identifier, material in [
        ("unknown", None),
        ("steel", {"name": "Steel", "density_kg_m3": 7800}),
    ]:
        parts.append(
            {
                "id": identifier,
                "name": identifier,
                "features": [{"id": "solid", "name": "Solid", "kind": "box"}],
                "output": "solid",
                "production": {"role": "make", "process": "cnc", "material": material},
            }
        )
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": parts,
                "roots": [
                    {"id": "unassigned", "definition": "unknown"},
                    {"id": "left", "definition": "steel"},
                    {"id": "right", "definition": "steel"},
                ],
            }
        }
    ).model_dump()
    service.store.add_revision(
        sid,
        {
            "id": rid,
            "contract": contract,
            "document": document,
            "measurement": {
                "geometry_valid": True,
                "engine_fingerprint": "engine",
                "artifact_hashes": {"part.step": "step", "topology.json": "topology"},
                "findings": [
                    {"requirement_id": r["id"], "status": "unverified", "reason": "Unmeasured"}
                    for r in contract["requirements"]
                ],
                "facts": {
                    "assembly": {
                        "occurrences": [
                            {
                                "path": path,
                                "body_index": index,
                                "definition": "unknown" if index == 0 else "steel",
                                "volume_mm3": 1000,
                            }
                            for index, path in [(2, "right"), (0, "unassigned"), (1, "left")]
                        ]
                    }
                },
            },
        },
    )
    service.store.update(sid, proposal={"revision_id": rid, "contract": contract})
    monkeypatch.setattr(
        document_service,
        "read_topology",
        lambda *args: {
            "entities": [
                {"kind": "body", "body_index": i, "id": f"body_{i}", "signature": str(i) * 64}
                for i in [1, 2, 0]
            ]
        },
    )

    def computed(directory, *_args):
        (directory / "field.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "frame": "cad_z_up_mm",
                    "positions_mm": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "triangles": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
                    "displacements_mm": [[0, 0, 0]] * 4,
                    "von_mises_mpa": [0] * 4,
                    "strain_equivalent": [0] * 4,
                    "fixed_triangles": [0],
                    "loaded_triangles": [3],
                }
            ),
            encoding="utf-8",
        )
        return {
            "status": "computed",
            "comparisons": [{"id": "declared_limit", "status": "pass"}],
            "numerical_quality": {"status": "pass"},
        }

    monkeypatch.setattr(structural_service, "process_revision", computed)
    client = TestClient(create_app(service=service, local=True))
    response = client.post(
        f"/workspace/sessions/{sid}/structural-studies",
        json={
            "revision_id": rid,
            "input_version": 0,
            "study": {
                "name": "Independent receipt fixture",
                "target": {"body_index": 0},
                "material": {
                    "name": "Declared numerical material",
                    "source": "Synthetic test fixture, no physical evidence",
                    "density_kg_m3": 2700,
                    "elastic_modulus_mpa": 70000,
                    "poisson_ratio": 0.3,
                },
                "force_n": 100,
                "direction": [0, 0, 1],
                "basis": "Isolate qualification derivation from the solver",
            },
        },
    )
    assert response.status_code == 202, response.text
    service.run_job(service.store.claim())
    return service, sid, rid, response.json()["study_id"]


def category(review, identifier):
    return next(c for c in review["categories"] if c["id"] == identifier)


@pytest.mark.parametrize("mutation", ["changed", "missing"])
def test_qualification_rejects_unavailable_current_field_bytes(qualification_workspace, mutation):
    service, sid, rid, study_id = qualification_workspace
    assert (
        category(qualification_review(service, sid, "local-owner", rid), "structure")["status"]
        == "computed"
    )
    path = study_path(service, sid, study_id) / "field.json"
    if mutation == "changed":
        path.write_text("{}", encoding="utf-8")
    else:
        path.unlink()
    with pytest.raises(WorkspaceError):
        qualification_review(service, sid, "local-owner", rid)


def test_passing_study_does_not_rewrite_any_of_25_frozen_obligations(qualification_workspace):
    service, sid, rid, _ = qualification_workspace
    before = service.store.revision(sid, rid)
    review = qualification_review(service, sid, "local-owner", rid)
    assert category(review, "structure")["status"] == "computed"
    assert category(review, "requirements")["status"] == "unverified"
    assert review["physical_readiness"] == "unverified"
    assert len(review["requirements"]) == 25
    for obligation, original in zip(
        review["requirements"], before["contract"]["requirements"], strict=True
    ):
        assert {key: obligation[key] for key in original} == original
        assert obligation["status"] == "unverified"
        assert obligation["finding"]["reason"] == "Unmeasured"
        assert obligation["interface_evidence"] == []
    assert service.store.revision(sid, rid) == before
    assert service.store.get(sid, "local-owner")["contract"] == before["contract"]


def test_historical_study_cannot_supply_new_revision_coverage(qualification_workspace):
    service, sid, rid, _ = qualification_workspace
    newer = "b" * 32
    revision = copy.deepcopy(service.store.revision(sid, rid))
    revision["id"] = newer
    service.store.add_revision(sid, revision)
    service.store.update(sid, proposal={"revision_id": newer})
    review = qualification_review(service, sid, "local-owner", newer)
    assert review["structural_studies"] == []
    assert category(review, "structure")["status"] == "unverified"
    historical = qualification_review(service, sid, "local-owner", rid)
    assert category(historical, "structure")["status"] == "computed"
    assert historical["structural_studies"][0]["matches_current_design"] is False
    assert historical["revision_id"] == rid


def test_changed_inputs_exclude_study_from_current_coverage(qualification_workspace):
    service, sid, rid, _ = qualification_workspace
    inputs = service.store.engineering(sid)["inputs"]
    service.store.save_engineering(sid, "local-owner", 0, inputs)
    review = qualification_review(service, sid, "local-owner", rid)
    assert review["structural_studies"][0]["stale"] is True
    assert category(review, "structure")["status"] == "unverified"
    assert category(review, "structure")["evidence_count"] == 0


def test_inputs_changed_between_listing_and_verified_read_cannot_supply_coverage(
    qualification_workspace, monkeypatch
):
    from forge_mind.workspace import qualification

    service, sid, rid, _ = qualification_workspace
    original = qualification.list_studies

    def changed_after_listing(*args):
        listed = original(*args)
        service.store.save_engineering(
            sid, "local-owner", 0, service.store.engineering(sid)["inputs"]
        )
        return listed

    monkeypatch.setattr(qualification, "list_studies", changed_after_listing)
    review = qualification_review(service, sid, "local-owner", rid)
    assert category(review, "structure")["status"] == "unverified"
    assert category(review, "structure")["evidence_count"] == 0


def test_body_declarations_follow_identity_and_missing_material_stays_unknown(
    qualification_workspace,
):
    service, sid, rid, _ = qualification_workspace
    review = qualification_review(service, sid, "local-owner", rid)
    bodies = {body["body_index"]: body for body in review["spatial"]["bodies"]}
    assert bodies[0]["occurrence_path"] == "unassigned"
    assert bodies[0]["production"]["material"] is None
    assert bodies[1]["occurrence_path"] == "left"
    assert bodies[2]["occurrence_path"] == "right"
    for index in [1, 2]:
        assert bodies[index]["production"]["material"]["name"] == "Steel"
        assert bodies[index]["production"]["material"]["density_kg_m3"] == 7800
    assert review["production"]["mass_kg"] is None
    assert "unknown: material unspecified" in category(review, "materials")["missing"]
    assert category(review, "materials")["status"] == "unverified"
