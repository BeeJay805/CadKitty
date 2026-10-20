from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
from build123d import Box, Compound, Cylinder, Pos, Rectangle, export_step, import_step

from forge_mind.server.workspace_app import create_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument, Sketch
from forge_mind.workspace.document_service import build_document, protection_results
from forge_mind.workspace.engineering import (
    assembly_review,
    assembly_transforms,
    simulation_review,
    tolerance_review,
)
from forge_mind.workspace.engineering_models import EngineeringInputs, JointInput
from forge_mind.workspace.engineering_service import release_package, run_report_job
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract
from forge_mind.workspace.sketch import solve_sketch
from forge_mind.workspace.store import Store, WorkspaceError
from forge_mind.workspace.topology import resolve, signature, topology_index


def feature(
    fid: str,
    kind: str,
    params: dict[str, float] | None = None,
    inputs: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": fid,
        "name": fid,
        "kind": kind,
        "params": params or {},
        "inputs": inputs or [],
        **extra,
    }


def document(*features: dict[str, Any], **extra: Any) -> DesignDocument:
    return DesignDocument.model_validate({"features": list(features), **extra})


def box_document(x: float = 20) -> DesignDocument:
    return document(feature("base", "box", {"x": x, "y": 30, "z": 10}), outputs=["base"])


def test_local_api_exposes_exact_document_authoring_contract(
    service: WorkspaceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(service, "start", lambda: None)
    client = TestClient(create_app(local=True, service=service))
    result = client.get("/workspace/document-schema")
    assert result.status_code == 200
    body = result.json()
    assert "features" in body["document"]["properties"]
    assert "requirements" in body["contract"]["properties"]
    assert body["empty_document"] == DesignDocument().model_dump(mode="json")


def compile_volume(doc: DesignDocument) -> float:
    from forge_mind.workspace.kernel import compile_document

    result, _ = compile_document(doc, {})
    return float(result.volume)


def test_document_rejects_forward_dependencies_and_nan() -> None:
    with pytest.raises(ValueError, match="precede"):
        document(feature("cut", "cut", inputs=["later"]))
    with pytest.raises(ValueError):
        box_document(float("nan"))


def test_worker_byte_watchdog_tolerates_removed_intermediates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from forge_mind.workspace.processing import directory_size

    (tmp_path / "solid_0.stl").write_bytes(b"temporary")
    (tmp_path / "part.step").write_bytes(b"retained")
    original = Path.lstat

    def disappearing(path: Path) -> Any:
        if path.name == "solid_0.stl":
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "lstat", disappearing)
    assert directory_size(tmp_path) == len(b"retained")


def test_kernel_boolean_transform_mirror_pattern() -> None:
    doc = document(
        feature("box", "box", {"x": 20, "y": 20, "z": 10}),
        feature("hole", "cylinder", {"radius": 2, "height": 30}),
        feature("cut", "cut", inputs=["box", "hole"]),
        feature("shift", "transform", {"x": 50}, ["cut"]),
        feature("mirror", "mirror", {"plane": 0}, ["shift"]),
        feature("pattern", "pattern", {"count": 2, "x": 30}, ["mirror"]),
    )
    assert compile_volume(doc) == pytest.approx(2 * (4000 - 40 * np.pi), rel=1e-7)


def test_kernel_edge_edits_shell_and_failures() -> None:
    box = Box(20, 20, 10)
    edge = signature(box.edges()[0])
    for kind, params in (("fillet", {"radius": 1}), ("chamfer", {"length": 1})):
        doc = document(
            feature("base", "box", {"x": 20, "y": 20, "z": 10}),
            feature("finish", kind, params, ["base"], selection=[edge]),
        )
        assert 3000 < compile_volume(doc) < 4000
    face = signature(max(box.faces(), key=lambda f: f.center().Z))
    assert compile_volume(
        document(
            feature("base", "box", {"x": 20, "y": 20, "z": 10}),
            feature("shell", "shell", {"thickness": 2}, ["base"], selection=[face]),
        )
    ) == pytest.approx(4000 - 16 * 16 * 8)
    with pytest.raises(ValueError, match="input stage"):
        compile_volume(
            document(
                feature("base", "box", {"x": 20, "y": 20, "z": 10}),
                feature("f", "fillet", {"radius": 1}, ["base"], selection=["not_a_reference"]),
            )
        )
    with pytest.raises(ValueError, match="Feature"):
        compile_volume(
            document(
                feature("base", "box", {"x": 20, "y": 20, "z": 10}, suppressed=True),
                feature("f", "transform", {"x": 20}, ["base"]),
            )
        )


def rectangular_sketch() -> dict[str, Any]:
    return {
        "points": [
            {"id": "a", "x": 0, "y": 0},
            {"id": "b", "x": 20, "y": 0},
            {"id": "c", "x": 20, "y": 10},
            {"id": "d", "x": 0, "y": 10},
        ],
        "entities": [
            {"id": str(i), "kind": "line", "points": list(pair)}
            for i, pair in zip(("ab", "bc", "cd", "da"), ("ab", "bc", "cd", "da"), strict=True)
        ],
        "constraints": [
            {"id": "anchor", "kind": "fix", "points": ["a"]},
            {"id": "width", "kind": "distance_x", "points": ["a", "b"], "value": 20},
            {"id": "flat", "kind": "horizontal", "entities": ["ab"]},
        ],
    }


def test_sketch_solver_conflicts_and_real_extrusion() -> None:
    sketch = Sketch.model_validate(rectangular_sketch())
    solved = solve_sketch(sketch)
    assert solved["status"] == "underconstrained"
    assert solved["degrees_of_freedom"] > 0
    assert compile_volume(
        document(
            feature("sketch", "sketch", sketch=sketch.model_dump()),
            feature("extrude", "extrude", {"amount": 5}, ["sketch"]),
        )
    ) == pytest.approx(1000)
    bad = sketch.model_dump()
    bad["constraints"].append(
        {"id": "conflict", "kind": "distance_x", "points": ["a", "b"], "value": 30}
    )
    result = solve_sketch(Sketch.model_validate(bad))
    assert result["status"] == "conflicting"
    with pytest.raises(ValueError, match="Conflicting"):
        compile_volume(
            document(
                feature("sketch", "sketch", sketch=bad),
                feature("extrude", "extrude", {"amount": 5}, ["sketch"]),
            )
        )


def test_full_analytic_reference_survives_step_roundtrip(tmp_path: Path) -> None:
    for part in (Box(20, 30, 10), Cylinder(5, 20)):
        export_step(part, str(tmp_path / "part.step"))
        rebuilt = import_step(str(tmp_path / "part.step"))
        assert set(map(signature, part.faces())) == set(map(signature, rebuilt.faces()))
        assert set(map(signature, part.edges())) == set(map(signature, rebuilt.edges()))


def test_references_reject_symmetric_duplicates_and_changed_boundaries() -> None:
    index = topology_index(Compound(children=[Box(10, 10, 10), Box(10, 10, 10)]))
    face = next(e for e in index["entities"] if e["kind"] == "face")
    assert resolve(face, index)["status"] == "ambiguous"
    # Identical area, center and bounds, different complete hole boundaries.
    first = Rectangle(40, 40) - Pos(-8, 0) * Rectangle(4, 4) - Pos(8, 0) * Rectangle(4, 4)
    second = Rectangle(40, 40) - Pos(0, -8) * Rectangle(4, 4) - Pos(0, 8) * Rectangle(4, 4)
    assert first.area == pytest.approx(second.area)
    assert signature(first.faces()[0]) != signature(second.faces()[0])


def test_assembly_pose_collision_and_loop_refusal() -> None:
    joint = JointInput(id="joint", kind="slider", parent=0, child=1, value=20, lower=0, upper=30)
    transforms = assembly_transforms(2, [joint])
    assert transforms[1][2, 3] == pytest.approx(20)
    inputs = EngineeringInputs.model_validate(
        {"assembly_revision": "a" * 32, "joints": [joint.model_dump()]}
    )
    report, posed = assembly_review([Box(10, 10, 10), Box(10, 10, 10)], inputs, "a" * 32)
    assert report["status"] == "pass"
    assert pytest.approx(20) == posed[1].center().Z
    collision, _ = assembly_review(
        [Box(10, 10, 10), Box(10, 10, 10)], EngineeringInputs(), "a" * 32
    )
    assert collision["status"] == "fail"
    reverse = JointInput(id="reverse", kind="fixed", parent=1, child=0)
    with pytest.raises(ValueError, match="Closed-loop"):
        assembly_transforms(2, [joint, reverse])
    stale, _ = assembly_review([Box(10, 10, 10)], inputs, "b" * 32)
    assert stale["status"] == "unverified"


def test_tolerances_and_actual_analytic_response() -> None:
    inputs = EngineeringInputs.model_validate(
        {
            "material": {
                "name": "Steel",
                "source": "Declared test material",
                "density_kg_m3": 7800,
                "elastic_modulus_mpa": 200000,
                "poisson_ratio": 0.3,
                "yield_strength_mpa": 250,
            },
            "loads": [
                {
                    "id": "load",
                    "method": "rectangular_cantilever",
                    "force_n": 10,
                    "max_deflection_mm": 1,
                    "min_safety_factor": 2,
                    "source": "Test acceptance",
                }
            ],
            "fits": [
                {
                    "id": "fit",
                    "hole": {
                        "name": "hole",
                        "nominal_mm": 10.2,
                        "minus_mm": 0.05,
                        "plus_mm": 0.05,
                        "source": "Drawing",
                    },
                    "shaft": {
                        "name": "shaft",
                        "nominal_mm": 10,
                        "minus_mm": 0.02,
                        "plus_mm": 0.02,
                        "source": "Drawing",
                    },
                    "min_clearance_mm": 0.1,
                    "max_clearance_mm": 0.3,
                    "fit_class": "clearance",
                    "source": "Acceptance",
                }
            ],
        }
    )
    result = simulation_review(None, Box(5, 10, 100), inputs)[0]
    assert result["max_deflection_mm"] == pytest.approx(
        10 * 100**3 / (3 * 200000 * (10 * 5**3 / 12))
    )
    assert result["status"] == "unverified"  # computed, not a physical safety certificate
    fit = tolerance_review(inputs)[0]
    assert fit["c_min_mm"] == pytest.approx(0.13)
    assert fit["c_max_mm"] == pytest.approx(0.27)
    assert fit["verdict"]["status"] == "pass"


@pytest.fixture
def service(tmp_path: Path) -> WorkspaceService:
    return WorkspaceService(Store(tmp_path), Executor("trusted_local"))


def create_revision(
    service: WorkspaceService, sid: str, x: float = 20, parent: str | None = None
) -> dict[str, Any]:
    return build_document(
        service,
        sid,
        "owner",
        box_document(x),
        Contract(),
        parent,
        "Integration fixture",
        90,
        lambda: False,
    )


def test_candidate_acceptance_stale_inputs_and_immutable_base(service: WorkspaceService) -> None:
    store = service.store
    session = store.create("owner", "Manual design", Contract().model_dump())
    sid = session["id"]
    store.enqueue(sid, "owner", {"document": box_document().model_dump(), "preview_only": True})
    job = store.claim()
    assert job
    service.run_job(job)
    current = store.get(sid, "owner")
    assert current["selected_revision"] is None
    proposal = current["proposal"]
    assert proposal["eligible"], current["revisions"][-1]["measurement"]["errors"]
    store.accept_proposal(sid, "owner", proposal["revision_id"], None, 0)
    first = proposal["revision_id"]
    store.enqueue(
        sid,
        "owner",
        {"document": box_document(22).model_dump(), "preview_only": True, "revision_id": first},
    )
    job = store.claim()
    assert job
    service.run_job(job)
    second = store.get(sid, "owner")["proposal"]["revision_id"]
    store.save_engineering(sid, "owner", 0, EngineeringInputs().model_dump())
    with pytest.raises(WorkspaceError, match="stale"):
        store.accept_proposal(sid, "owner", second, first, 0)
    assert store.get(sid, "owner")["accepted_revision"] == first
    assert store.revision(sid, first)["measurement"]["facts"]["extents_mm"][0] == pytest.approx(20)


def test_protected_dependencies_and_release_snapshot(service: WorkspaceService) -> None:
    sid = service.store.create("owner", "Manual design", Contract().model_dump())["id"]
    first = create_revision(service, sid)
    assert first["measurement"]["geometry_valid"], first["measurement"]["errors"]
    second = create_revision(service, sid, 22, first["id"])
    inputs = EngineeringInputs.model_validate(
        {
            "protections": [
                {
                    "id": "protect",
                    "revision_id": first["id"],
                    "feature_id": "base",
                    "reason": "Keep base",
                }
            ]
        }
    )
    assert protection_results(service, sid, first, inputs)[0]["status"] == "pass"
    assert protection_results(service, sid, second, inputs)[0]["status"] == "fail"
    service.store.update(sid, selected_revision=first["id"])
    run_report_job(
        service,
        {"session": sid, "owner": "owner", "revision_id": first["id"], "max_seconds": 120},
        lambda: False,
    )
    package = release_package(service, sid, "owner", first["id"])
    with ZipFile(package) as archive:
        assert {
            "part.step",
            "document.json",
            "engineering/drawing.svg",
            "engineering/inspection.csv",
            "engineering/report.json",
            "manifest.json",
        } <= set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["revision_id"] == first["id"]
        import hashlib

        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    service.store.save_engineering(sid, "owner", 0, inputs.model_dump())
    with pytest.raises(WorkspaceError, match="current inputs"):
        release_package(service, sid, "owner", first["id"])


def test_owned_import_and_forged_annotation_rejected(
    service: WorkspaceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(service, "start", lambda: None)
    app = create_app(local=True, service=service)
    client = TestClient(app)
    sid = client.post("/workspace/manual-sessions").json()["id"]
    other = service.store.create("someone-else", "Private", None)["id"]
    assert client.get(f"/workspace/sessions/{other}").status_code == 404
    assert (
        client.post(f"/workspace/sessions/{other}/assets", content=b"ISO-10303-21;").status_code
        == 404
    )
    assert client.post(f"/workspace/sessions/{sid}/assets", content=b"not STEP").status_code == 422
    response = client.post(
        f"/workspace/sessions/{sid}/engineering-inputs",
        json={
            "version": 0,
            "inputs": {
                "annotations": [
                    {
                        "id": "note",
                        "text": "Fake",
                        "reference": {
                            "revision_id": "f" * 32,
                            "entity_id": "face_" + "f" * 20 + "_0",
                            "signature": "f" * 64,
                            "body_signature": "f" * 64,
                            "kind": "face",
                            "point": [0, 0, 0],
                        },
                    }
                ]
            },
        },
    )
    assert response.status_code == 409


def test_release_contains_verified_structural_bytes_and_skips_newly_stale_study(
    service: WorkspaceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from forge_mind.workspace import structural_service
    from forge_mind.workspace.product_sources import _append

    sid = service.store.create("owner", "Structural package fixture", Contract().model_dump())["id"]
    revision = create_revision(service, sid)
    rid = revision["id"]
    assert revision["measurement"]["geometry_valid"]
    service.store.update(sid, selected_revision=rid)
    run_report_job(service, {"session": sid, "owner": "owner", "revision_id": rid}, lambda: False)
    # Isolate packaging and publication from numerical solving. Geometry/report and
    # persisted receipt/field integrity still use their real implementations.
    field = {
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

    def computed(directory: Path, *_args: Any) -> dict[str, Any]:
        (directory / "field.json").write_text(json.dumps(field), encoding="utf-8")
        for stage in ("coarse", "fine"):
            target = directory / stage
            target.mkdir()
            # Preserve the real single-solid exported source in this package test.
            (target / "selected.step").write_bytes(
                (service.store.directory(sid, rid) / "part.step").read_bytes()
            )
            np.savez_compressed(
                target / "mesh.npz",
                positions_mm=np.asarray(field["positions_mm"]),
                tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int64),
            )
        return {"status": "computed", "comparisons": []}

    monkeypatch.setattr(structural_service, "process_revision", computed)
    study_id = "c" * 32
    service.store.enqueue(
        sid,
        "owner",
        {
            "revision_id": rid,
            "input_version": 0,
            "study_id": study_id,
            "engineering_operation": "structural_study",
            "study": {
                "name": "Packaged field fixture",
                "mesher": "cad",
                "target": {"body_index": 0},
                "material": {
                    "name": "Synthetic material",
                    "source": "Packaging test fixture, no physical claim",
                    "density_kg_m3": 2700,
                    "elastic_modulus_mpa": 70000,
                    "poisson_ratio": 0.3,
                },
                "force_n": 100,
                "direction": [0, 0, 1],
                "basis": "Test immutable packaging only",
            },
        },
    )
    job = service.store.claim()
    assert job
    service.run_job(job)
    receipt = structural_service.read_study(service, sid, "owner", study_id)
    field_path = structural_service.field_path(service, sid, "owner", study_id)
    original_bytes = field_path.read_bytes()
    prefix = f"structural-studies/{study_id}/"
    with ZipFile(release_package(service, sid, "owner", rid)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        archived_receipt = json.loads(archive.read(prefix + "receipt.json"))
        assert archived_receipt == receipt
        assert archived_receipt["qualification_status"] == "unverified"
        assert set(receipt["hashes"]) == structural_service.STUDY_ARTIFACTS
        for name, expected in receipt["hashes"].items():
            assert hashlib.sha256(archive.read(prefix + name)).hexdigest() == expected
            assert (
                archive.read(prefix + name)
                == (structural_service.study_path(service, sid, study_id) / name).read_bytes()
            )
        assert archive.read(prefix + "field.json") == original_bytes
        assert hashlib.sha256(original_bytes).hexdigest() == receipt["hashes"]["field.json"]
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected

    field_path.write_text("{}", encoding="utf-8")
    with pytest.raises(WorkspaceError, match="artifact changed"):
        release_package(service, sid, "owner", rid)
    field_path.write_bytes(original_bytes)

    for name in structural_service.STUDY_ARTIFACTS - {"field.json"}:
        artifact = structural_service.study_path(service, sid, study_id) / name
        original = artifact.read_bytes()
        for mutation in ("tamper", "remove"):
            if mutation == "tamper":
                artifact.write_bytes(b"changed mesh provenance")
            else:
                artifact.unlink()
            with pytest.raises(WorkspaceError, match="artifact changed"):
                release_package(service, sid, "owner", rid)
            artifact.write_bytes(original)

    original_list = structural_service.list_studies

    def changed_after_listing(*args: Any) -> list[dict[str, Any]]:
        summaries = original_list(*args)
        assert len(summaries) == 1 and not summaries[0]["stale"]
        _append(service.store, sid, "owner", {"kind": "artifact", "data": {"id": "new_source"}})
        return summaries

    monkeypatch.setattr(structural_service, "list_studies", changed_after_listing)
    with ZipFile(release_package(service, sid, "owner", rid)) as archive:
        assert not any(name.startswith("structural-studies/") for name in archive.namelist())
        manifest = json.loads(archive.read("manifest.json"))
        assert not any(name.startswith("structural-studies/") for name in manifest["files"])
    assert structural_service.read_study(service, sid, "owner", study_id)["stale"]


def test_api_pose_uses_captured_bodies_and_preserves_accepted_revision(
    service: WorkspaceService, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setattr(service, "start", lambda: None)
    client = TestClient(create_app(local=True, service=service))
    sid = client.post("/workspace/manual-sessions").json()["id"]
    doc = document(
        feature("base", "box", {"x": 10, "y": 10, "z": 10}),
        feature("second", "transform", {"z": 20}, ["base"]),
        outputs=["base", "second"],
    )
    assert (
        client.post(
            f"/workspace/sessions/{sid}/document", json={"document": doc.model_dump()}
        ).status_code
        == 202
    )
    job = service.store.claim()
    assert job
    service.run_job(job)
    session = service.store.get(sid, "local-owner")
    proposal = session["proposal"]
    rid = proposal["revision_id"]
    assert (
        client.post(
            f"/workspace/sessions/{sid}/accept",
            json={"revision_id": rid, "expected_revision": None, "input_version": 0},
        ).status_code
        == 200
    )
    settings = EngineeringInputs(
        assembly_revision=rid,
        joints=[
            JointInput(id="slide", kind="slider", parent=0, child=1, value=25, lower=0, upper=50)
        ],
    )
    service.store.save_engineering(sid, "local-owner", 0, settings.model_dump())
    run_report_job(
        service, {"session": sid, "owner": "local-owner", "revision_id": rid}, lambda: False
    )
    report = service.store.get(sid, "local-owner")["reports"][-1]
    response = client.post(
        f"/workspace/sessions/{sid}/pose", json={"revision_id": rid, "report_id": report["id"]}
    )
    assert response.status_code == 202, response.text
    job = service.store.claim()
    assert job
    service.run_job(job)
    session = service.store.get(sid, "local-owner")
    assert session["accepted_revision"] == rid
    candidate = service.store.revision(sid, session["proposal"]["revision_id"])
    assert session["proposal"]["eligible"], candidate["measurement"]["errors"]
    assert candidate["measurement"]["facts"]["extents_mm"][2] == pytest.approx(55)
    assert candidate["assets"][0]["source_revision"] == rid


def test_preview_expiry_cancel_and_stale_pose_admission(tmp_path: Path) -> None:
    store = Store(tmp_path)
    # This lease/pose test starts from known accepted intent, not pending extraction.
    sid = store.create("owner", "Design", Contract().model_dump())["id"]
    baseline, candidate = "a" * 32, "b" * 32
    store.update(sid, accepted_revision=baseline, selected_revision=baseline)
    first = store.enqueue(sid, "owner", {"preview_only": True})
    assert store.claim()
    store.update(sid, selected_revision=candidate)
    store.finish(first, sid, "needs_review")
    with store.connect() as con:
        con.execute("UPDATE jobs SET deadline=0 WHERE id=?", (first,))
    assert store.claim() is None
    # Elapsed time cannot interrupt a live owner or discard its preview.
    assert store.get(sid, "owner")["selected_revision"] == candidate
    with store.connect() as con:
        # A reused PID, unlike a deadline, confirms that the recorded owner died.
        con.execute("UPDATE jobs SET worker_created=-1 WHERE id=?", (first,))
    assert store.claim() is None
    recovered = store.get(sid, "owner")
    assert recovered["state"] == "interrupted"
    assert recovered["selected_revision"] == recovered["accepted_revision"] == baseline
    assert "pending_finish" not in recovered
    next_job = store.enqueue(sid, "owner", {"preview_only": True})
    assert store.claim()
    store.update(sid, selected_revision=candidate)
    store.finish(next_job, sid, "needs_review")
    store.cancel(sid, "owner")
    store.complete_preview(sid, next_job, baseline, {"eligible": True})
    cancelled = store.get(sid, "owner")
    assert cancelled["state"] == "cancelled" and cancelled["proposal"] is None
    assert cancelled["selected_revision"] == baseline
    store.complete_preview(sid, first, None, {"eligible": True})
    assert store.get(sid, "owner")["selected_revision"] == baseline
    store.save_engineering(sid, "owner", 0, EngineeringInputs().model_dump())
    with pytest.raises(WorkspaceError, match="changed before admission"):
        store.enqueue(sid, "owner", {"preview_only": True, "expected_input_version": 0})


def test_preview_intent_commits_only_with_candidate_acceptance(tmp_path: Path) -> None:
    store = Store(tmp_path)
    original = {"requirements": [], "coverage_notes": ["Accepted target set"]}
    candidate_contract = {"requirements": [], "coverage_notes": ["Candidate target set"]}
    sid = store.create("owner", "Design", original)["id"]
    for terminal in ("error", "needs_review"):
        jid = store.enqueue(sid, "owner", {"preview_only": True})
        assert store.claim()
        store.update(sid, contract=candidate_contract, instructions=["Candidate instruction"])
        store.finish(jid, sid, terminal)
        store.complete_preview(
            sid,
            jid,
            None,
            {
                "revision_id": "a" * 32,
                "based_on": None,
                "input_version": 0,
                "eligible": True,
                "contract": candidate_contract,
                "instructions": ["Candidate instruction"],
            },
        )
        session = store.get(sid, "owner")
        assert session["contract"] == original
        assert session["instructions"] == []
        if terminal == "error":
            assert session["proposal"] is None
        else:
            store.accept_proposal(sid, "owner", "a" * 32, None, 0)
            assert store.get(sid, "owner")["contract"] == candidate_contract
            assert store.get(sid, "owner")["instructions"] == ["Candidate instruction"]


def test_spherical_export_is_watertight_after_zero_area_cleanup(service: WorkspaceService) -> None:
    sid = service.store.create("owner", "Sphere", Contract().model_dump())["id"]
    revision = build_document(
        service,
        sid,
        "owner",
        document(feature("sphere", "sphere", {"radius": 2})),
        Contract(),
        None,
        "Small curved export",
        90,
        lambda: False,
    )
    assert revision["measurement"]["geometry_valid"], revision["measurement"]
    assert revision["measurement"]["facts"]["watertight"]


def test_concurrent_imports_are_not_lost(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    store = Store(tmp_path)
    sid = store.create("owner", "Design", None)["id"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: store.append_asset(sid, "owner", {"id": str(i)}), range(20)))
    assert len(store.get(sid, "owner")["assets"]) == 20
    with pytest.raises(WorkspaceError, match="not found"):
        store.append_asset(sid, "other", {"id": "private"})


def test_geometry_reference_never_migrates_to_another_body() -> None:
    original = topology_index(Box(10, 10, 10))
    ref = next(e for e in original["entities"] if e["kind"] == "face" and e["center"][2] == 5)
    # Same top face, but a different body below it.
    changed = topology_index(Pos(0, 0, -5) * Box(10, 10, 20))
    assert any(e["signature"] == ref["signature"] for e in changed["entities"])
    assert resolve(ref, changed)["status"] == "unresolved"


@pytest.mark.parametrize("operation", ["extrude", "revolve"])
def test_profile_operations_reject_solid_sources(operation: str) -> None:
    with pytest.raises(ValueError, match="sketch"):
        document(
            feature("solid", "box", {"x": 10, "y": 10, "z": 10}),
            feature("op", operation, inputs=["solid"]),
        )


def test_service_path_collision_propagates_to_assembly_verdict() -> None:
    inputs = EngineeringInputs.model_validate(
        {
            "assembly_revision": "a" * 32,
            "service_paths": [
                {
                    "id": "path",
                    "body": 1,
                    "travel": [-30, 0, 0],
                    "samples": 4,
                    "task": "Remove slider",
                }
            ],
        }
    )
    report, _ = assembly_review(
        [Box(10, 10, 10), Pos(30, 0, 0) * Box(10, 10, 10)], inputs, "a" * 32
    )
    assert report["pose_status"] == "pass"
    assert report["status"] == report["service_paths"][0]["status"] == "fail"
    assert report["service_paths"][0]["witness"]["overlap_mm3"] > 0


def test_public_typed_worker_fails_closed(tmp_path: Path) -> None:
    service = WorkspaceService(Store(tmp_path), Executor("docker"))
    app = create_app(local=False, service=service)
    assert not any(getattr(route, "path", "").endswith("/document") for route in app.routes)
    sid = service.store.create("owner", "Design", None)["id"]
    with pytest.raises(WorkspaceError, match="private-local"):
        create_revision(service, sid)


def test_captured_hardware_inventory_survives_pose() -> None:
    from forge_mind.workspace.hardware import bill_of_materials, search

    part = search("bearing")[0]
    source = document(
        feature("bearing", "hardware", catalog_id=part["id"]),
        feature("array", "pattern", {"count": 2, "x": 40}, ["bearing"]),
    )
    inventory = bill_of_materials(source)
    assert inventory["lines"][0]["quantity"] == 2
    asset = {"id": "a" * 32, "name": "Captured assembly", "solid_count": 2, "inventory": inventory}
    posed = document(
        feature("first", "import", {"solid_index": 0}, asset_id=asset["id"]),
        feature("second", "import", {"solid_index": 1}, asset_id=asset["id"]),
        outputs=["first", "second"],
    )
    assert bill_of_materials(posed, [asset])["lines"] == inventory["lines"]
    partial = posed.model_copy(update={"outputs": ["first"]})
    assert not bill_of_materials(partial, [asset])["lines"]
    assert "partial body selection" in bill_of_materials(partial, [asset])["unmapped"][0]
    asset["inventory"]["lines"][0]["source"] = "Frozen historical manufacturer drawing"
    assert (
        bill_of_materials(posed, [asset])["lines"][0]["source"]
        == "Frozen historical manufacturer drawing"
    )


def test_boolean_hardware_does_not_claim_unmodified_inventory() -> None:
    from forge_mind.workspace.hardware import bill_of_materials, search

    part = search("washer")[0]
    hardware = feature("hardware", "hardware", catalog_id=part["id"])
    with pytest.raises(ValueError, match="unique"):
        document(hardware, feature("union", "union", inputs=["hardware", "hardware"]))
    modified = document(
        hardware,
        feature("tool", "box", {"x": 10, "y": 10, "z": 10}),
        feature("cut", "cut", inputs=["hardware", "tool"]),
    )
    bom = bill_of_materials(modified)
    assert not bom["lines"]
    assert "boolean-modified" in bom["unmapped"][0]


def test_revolve_and_hardware_generate_native_solids() -> None:
    from forge_mind.workspace.hardware import search

    profile = rectangular_sketch()
    profile["constraints"] = []
    for point in profile["points"]:
        point["x"] += 5
    revolved = document(
        feature("profile", "sketch", sketch=profile),
        feature("turn", "revolve", {"angle": 360}, ["profile"]),
    )
    assert compile_volume(revolved) == pytest.approx(np.pi * (25**2 - 5**2) * 10)
    bearing = search("bearing")[0]
    assert compile_volume(document(feature("hardware", "hardware", catalog_id=bearing["id"]))) > 0
