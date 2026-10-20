from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace import executor, measure, processing
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.models import Contract, Finding, Measurement, Requirement

RID = "a" * 32
SCOPE = (
    "Numerical export consistency only. Exact dimensions and requirement checks use STEP; "
    "sampled tessellation bounds do not define manufacturing precision."
)


def requirement(part_id: str | None = "cap") -> Requirement:
    return Requirement(
        id="width", text="Width", source_quote="2 mm", metric="extent_x", value=2, part_id=part_id
    )


def part_measurement(requirements: list[Requirement] | None = None) -> Measurement:
    return Measurement(
        revision_id=RID,
        contract_hash=Contract().fingerprint(),
        engine_fingerprint="fixture",
        artifact_hashes={},
        geometry_valid=True,
        facts={
            "solid_count": 1,
            "bbox_min": [-1.0, -1.5, -2.0],
            "bbox_max": [1.0, 1.5, 2.0],
            "tessellation": {
                "linear_deflection": 0.01,
                "linear_deflection_is_relative": True,
                "angular_deflection": 0.1,
            },
            "export_consistency": {
                "volume_within_relative_0_001": True,
                "bounds_within_0_01_mm": True,
                "maximum_sampled_bounds_difference_mm": 0.0,
                "reference_tessellation": None,
                "scope": SCOPE,
            },
        },
        findings=[
            Finding(
                requirement_id=r.id,
                status="pass",
                measured=2,
                expected=2,
                method="Native dimension",
                reason="Fixture width",
            )
            for r in requirements or []
        ],
    )


def stub_definition(monkeypatch, directory, *, measurement=None, export_error=None):
    (directory / "definition_cap.step").write_bytes(b"native definition fixture")
    exports, measurements = [], []

    def export(work):
        exports.append(work)
        if export_error is not None:
            raise export_error
        (work / "part.stl").write_bytes(b"closed mesh fixture")
        return 0.01, 0.1, 2

    def measured(work, rid, contract):
        measurements.append((work, rid, contract))
        return measurement or part_measurement(contract.requirements)

    monkeypatch.setattr(measure, "export_verified_tessellation", export)
    monkeypatch.setattr(measure, "measure_revision", measured)
    return exports, measurements


def test_retain_same_verified_mesh_without_changing_requirement_findings(tmp_path, monkeypatch):
    exports, measurements = stub_definition(monkeypatch, tmp_path)
    evidence, findings = measure.definition_measurements(
        tmp_path, RID, "cap", [requirement()], retain_stl=True
    )
    delivery = evidence["mesh_delivery"]
    assert len(exports) == len(measurements) == 1
    assert measurements[0][2].requirements[0].part_id is None
    assert delivery["status"] == "available"
    assert delivery["tessellation"]["collapsed_triangles_removed"] == 2
    assert delivery["export_consistency"]["reference_tessellation"] is None
    assert delivery["stl_sha256"] == file_sha256(tmp_path / "definition_cap.stl")
    assert (tmp_path / "definition_cap.stl").read_bytes() == b"closed mesh fixture"
    assert findings[0].status == "pass"
    assert findings[0].measured == 2


def test_default_measurement_retains_no_new_artifact_or_delivery_fields(tmp_path, monkeypatch):
    exports, _ = stub_definition(monkeypatch, tmp_path)
    evidence, findings = measure.definition_measurements(tmp_path, RID, "cap", [requirement()])
    assert len(exports) == 1
    assert "mesh_delivery" not in evidence
    assert not (tmp_path / "definition_cap.stl").exists()
    assert findings[0].status == "pass"


def test_exact_reference_fallback_preserves_existing_valid_export_policy(tmp_path, monkeypatch):
    measured = part_measurement()
    checks = measured.facts["export_consistency"]
    checks["bounds_within_0_01_mm"] = False
    checks["maximum_sampled_bounds_difference_mm"] = 0.02
    checks["reference_tessellation"] = {"triangle_inventory_matches": True, "scope": "fixture"}
    stub_definition(monkeypatch, tmp_path, measurement=measured)
    evidence, _ = measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    assert evidence["mesh_delivery"]["status"] == "available"
    assert evidence["mesh_delivery"]["export_consistency"] == checks


@pytest.mark.parametrize("kind", ["geometry", "volume", "bounds"])
def test_failed_mesh_checks_have_no_deliverable_stl(tmp_path, monkeypatch, kind):
    measured = part_measurement()
    if kind == "geometry":
        measured.geometry_valid = False
    elif kind == "volume":
        measured.facts["export_consistency"]["volume_within_relative_0_001"] = False
    else:
        measured.facts["export_consistency"]["bounds_within_0_01_mm"] = False
    stub_definition(monkeypatch, tmp_path, measurement=measured)
    evidence, _ = measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    assert evidence["mesh_delivery"]["status"] == "failed"
    assert "stl_file" not in evidence["mesh_delivery"]
    assert not (tmp_path / "definition_cap.stl").exists()
    assert (tmp_path / "definition_cap.step").read_bytes() == b"native definition fixture"


@pytest.mark.parametrize("error_type", [ValueError, OSError, RuntimeError])
def test_optional_tessellation_failure_retains_step_and_full_error(
    tmp_path, monkeypatch, error_type
):
    original = error_type("native tessellation failed: " + "x" * 3000)
    stub_definition(monkeypatch, tmp_path, export_error=original)
    evidence, findings = measure.definition_measurements(
        tmp_path, RID, "cap", [requirement()], retain_stl=True
    )
    assert evidence["mesh_delivery"]["status"] == "failed"
    assert len(evidence["mesh_delivery"]["error"]) <= 2000
    assert "truncated" in evidence["mesh_delivery"]["error"]
    assert str(original) in evidence["mesh_error_detail"]
    assert findings[0].status == "unverified"
    assert str(original) in findings[0].reason
    assert (tmp_path / "definition_cap.step").read_bytes() == b"native definition fixture"


def test_optional_file_failure_preserves_measured_findings_and_existing_file(tmp_path, monkeypatch):
    stub_definition(monkeypatch, tmp_path)
    target = tmp_path / "definition_cap.stl"
    target.write_bytes(b"preexisting unrelated output")
    evidence, findings = measure.definition_measurements(
        tmp_path, RID, "cap", [requirement()], retain_stl=True
    )
    assert evidence["mesh_delivery"]["status"] == "failed"
    assert findings[0].status == "pass"
    assert target.read_bytes() == b"preexisting unrelated output"


def test_long_retention_error_is_bounded_before_manifest_validation(tmp_path, monkeypatch):
    stub_definition(monkeypatch, tmp_path)
    from forge_mind.workspace import processing as transfer

    original_copy = transfer.copy_artifact

    def copy(source, target):
        if target.name == "definition_cap.stl":
            raise OSError("retention unavailable: " + "x" * 3000)
        return original_copy(source, target)

    monkeypatch.setattr(transfer, "copy_artifact", copy)
    evidence, findings = measure.definition_measurements(
        tmp_path, RID, "cap", [requirement()], retain_stl=True
    )
    (tmp_path / "document.json").write_bytes(b"saved document")
    result = {"artifact_hashes": {}, "facts": {"definitions": {"cap": evidence}}}
    processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": 1}, result)
    manifest = json.loads((tmp_path / "definition-meshes.json").read_bytes())
    assert manifest["parts"]["cap"]["status"] == "failed"
    assert len(manifest["parts"]["cap"]["error"]) <= 2000
    assert len(evidence["mesh_error_detail"]) > 3000
    assert findings[0].status == "pass"
    assert "definition_cap.stl" not in result["artifact_hashes"]


@pytest.mark.parametrize(
    "error", [TimeoutError("deadline"), asyncio.CancelledError(), KeyboardInterrupt()]
)
def test_fatal_termination_is_not_an_optional_mesh_failure(tmp_path, monkeypatch, error):
    stub_definition(monkeypatch, tmp_path, export_error=error)
    with pytest.raises(type(error)):
        measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    assert not (tmp_path / "definition_cap.stl").exists()


def fake_whole_measurement(monkeypatch, directory):
    import build123d

    bounds = np.array([[-1.0, -1.5, -2.0], [1.0, 1.5, 2.0]])

    def xyz(a):
        return SimpleNamespace(X=a[0], Y=a[1], Z=a[2])

    shape = SimpleNamespace(
        is_valid=True,
        solids=lambda: [SimpleNamespace(volume=24.0)],
        faces=lambda: [],
        bounding_box=lambda: SimpleNamespace(
            min=xyz(bounds[0]), max=xyz(bounds[1]), size=xyz([2, 3, 4])
        ),
    )
    mesh = SimpleNamespace(
        volume=24.0, bounds=bounds, is_watertight=True, is_winding_consistent=True
    )
    for name in ("program.py", "part.step", "part.stl"):
        (directory / name).write_bytes(b"fixture")
    monkeypatch.setattr(build123d, "import_step", lambda _: shape)
    monkeypatch.setattr(measure.trimesh, "load", lambda *args, **kwargs: mesh)
    monkeypatch.setattr(measure, "engine_fingerprint", lambda: "fixture")


def test_union_measurement_once_preserves_valid_whole_step_and_global_findings(
    tmp_path, monkeypatch
):
    fake_whole_measurement(monkeypatch, tmp_path)
    seen = []

    def definition(directory, rid, part_id, requirements, *, retain_stl=False):
        seen.append((part_id, retain_stl, [r.id for r in requirements]))
        return {"frame": "definition_local", "mesh_delivery": {"status": "failed"}}, [
            Finding(
                requirement_id=r.id,
                status="pass",
                measured=2,
                expected=2,
                method="Fixture",
                reason="Measured",
            )
            for r in requirements
        ]

    monkeypatch.setattr(measure, "definition_measurements", definition)
    contract = Contract(
        requirements=[requirement(), requirement(None).model_copy(update={"id": "global"})]
    )
    result = measure.measure_revision(
        tmp_path, RID, contract, definition_mesh_ids=("cap", "unused", "cap")
    )
    assert seen == [("cap", True, ["width"]), ("unused", True, [])]
    assert result.geometry_valid
    assert [f.status for f in result.findings] == ["pass", "pass"]


def test_outer_measurement_does_not_swallow_timeout(tmp_path, monkeypatch):
    fake_whole_measurement(monkeypatch, tmp_path)

    def definition(*args, **kwargs):
        raise TimeoutError("definition deadline")

    monkeypatch.setattr(measure, "definition_measurements", definition)
    with pytest.raises(TimeoutError):
        measure.measure_revision(tmp_path, RID, Contract(), definition_mesh_ids=("cap",))


def test_optional_native_error_keeps_whole_geometry_and_global_findings(tmp_path, monkeypatch):
    fake_whole_measurement(monkeypatch, tmp_path)
    (tmp_path / "definition_cap.step").write_bytes(b"native definition fixture")

    def exporter(work):
        raise RuntimeError("native tessellation fixture error")

    monkeypatch.setattr(measure, "export_verified_tessellation", exporter)
    contract = Contract(
        requirements=[requirement(), requirement(None).model_copy(update={"id": "global"})]
    )
    result = measure.measure_revision(tmp_path, RID, contract, definition_mesh_ids=("cap",))
    assert result.geometry_valid
    assert [finding.status for finding in result.findings] == ["unverified", "pass"]
    assert result.facts["definitions"]["cap"]["mesh_delivery"]["status"] == "failed"
    assert result.errors == []
    assert not (tmp_path / "definition_cap.stl").exists()


def test_manifest_binds_success_and_failure_including_unused_quantity(tmp_path, monkeypatch):
    (tmp_path / "document.json").write_bytes(b"exact saved document")
    stub_definition(monkeypatch, tmp_path)
    evidence, _ = measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    result = {"artifact_hashes": {}, "facts": {"definitions": {"cap": evidence}}}
    processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": 2, "unused": 0}, result)
    path = tmp_path / "definition-meshes.json"
    manifest = json.loads(path.read_bytes())
    assert manifest["document_sha256"] == file_sha256(tmp_path / "document.json")
    assert manifest["parts"]["cap"]["quantity"] == 2
    assert manifest["parts"]["unused"]["quantity"] == 0
    assert manifest["parts"]["unused"]["status"] == "failed"
    assert manifest["parts"]["unused"]["step_sha256"] is None
    assert result["artifact_hashes"] == {
        "definition_cap.stl": file_sha256(tmp_path / "definition_cap.stl"),
        "definition-meshes.json": file_sha256(path),
    }
    with pytest.raises(FileExistsError):
        processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": 2}, result)


def test_opt_out_writer_is_noop_and_failed_manifest_is_explicit(tmp_path):
    result = {"artifact_hashes": {}, "facts": {}}
    before = deepcopy(result)
    processing._write_definition_mesh_manifest(tmp_path, RID, {}, result)
    assert result == before
    assert not list(tmp_path.iterdir())
    (tmp_path / "document.json").write_bytes(b"saved document")
    processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": 1}, result)
    manifest = json.loads((tmp_path / "definition-meshes.json").read_bytes())
    assert manifest["parts"]["cap"]["status"] == "failed"
    assert "stl_file" not in manifest["parts"]["cap"]


def test_manifest_refuses_changed_mesh_as_deliverable_and_keeps_measurement(tmp_path, monkeypatch):
    (tmp_path / "document.json").write_bytes(b"saved document")
    stub_definition(monkeypatch, tmp_path)
    evidence, _ = measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    result = {"artifact_hashes": {}, "facts": {"definitions": {"cap": evidence}}}
    before = deepcopy(evidence)
    (tmp_path / "definition_cap.stl").write_bytes(b"changed mesh")
    processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": 1}, result)
    manifest = json.loads((tmp_path / "definition-meshes.json").read_bytes())
    assert manifest["parts"]["cap"]["status"] == "failed"
    assert "stl_file" not in manifest["parts"]["cap"]
    assert "definition_cap.stl" not in result["artifact_hashes"]
    assert evidence == before


def test_invalid_manifest_does_not_partially_bind_mesh_artifacts(tmp_path, monkeypatch):
    (tmp_path / "document.json").write_bytes(b"saved document")
    stub_definition(monkeypatch, tmp_path)
    evidence, _ = measure.definition_measurements(tmp_path, RID, "cap", [], retain_stl=True)
    result = {"artifact_hashes": {}, "facts": {"definitions": {"cap": evidence}}}
    before = deepcopy(result)
    with pytest.raises(ValueError):
        processing._write_definition_mesh_manifest(tmp_path, RID, {"cap": -1}, result)
    assert result == before
    assert not (tmp_path / "definition-meshes.json").exists()


@pytest.mark.parametrize(
    "name", ["definition_cap.stl", "definition_A1.stl", "definition-meshes.json"]
)
def test_typed_worker_accepts_only_exact_new_artifact_names(name):
    assert executor.typed_artifact_name(name)


@pytest.mark.parametrize(
    "name",
    [
        "../definition_cap.stl",
        "definition_cap.stl.json",
        "definition_.stl",
        "definition_1cap.stl",
        "definition-cap.stl",
        "definition_cap.STL",
        "definition_A" + "b" * 64 + ".stl",
    ],
)
def test_typed_worker_refuses_path_and_format_variants(name):
    assert not executor.typed_artifact_name(name)


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("aggregate_failed", [False, True])
def test_processing_propagates_only_explicit_validated_selection(
    tmp_path, monkeypatch, selected, aggregate_failed
):
    import build123d

    from forge_mind.workspace import kernel, preview_inspection, topology

    monkeypatch.chdir(tmp_path)
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "cap",
                        "name": "Cap",
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 2, "y": 3, "z": 4},
                            }
                        ],
                        "output": "body",
                        "production": {"role": "make", "deliver_stl": selected},
                    }
                ],
                "roots": [
                    {"id": "left", "definition": "cap"},
                    {"id": "right", "definition": "cap"},
                ],
            }
        }
    )
    (tmp_path / "document.json").write_bytes(document.model_dump_json().encode())
    (tmp_path / "program.py").write_bytes(b"inert fixture")
    request = tmp_path / "processing.json"
    request.write_text(
        json.dumps({"revision_id": RID, "document": document.model_dump(), "contract": {}})
    )
    shape = SimpleNamespace(solids=lambda: [])

    def compile_document(doc, assets, directory, cache):
        (directory / "definition_cap.step").write_bytes(b"definition fixture")
        return shape, None

    monkeypatch.setattr(kernel, "compile_document", compile_document)
    monkeypatch.setattr(build123d, "import_step", lambda _: shape)
    monkeypatch.setattr(
        build123d, "export_step", lambda part, target: Path(target).write_bytes(b"assembly fixture")
    )
    monkeypatch.setattr(
        build123d,
        "export_stl",
        lambda part, target, **kwargs: Path(target).write_bytes(b"preview fixture"),
    )
    monkeypatch.setattr(topology, "topology_index", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        processing, "export_verified_tessellation", lambda directory: (0.01, 0.1, 0)
    )
    monkeypatch.setattr(
        processing.trimesh if hasattr(processing, "trimesh") else measure.trimesh,
        "load",
        lambda *args, **kwargs: SimpleNamespace(bounds=np.zeros((2, 3))),
    )
    monkeypatch.setattr(
        measure.trimesh,
        "Scene",
        lambda: SimpleNamespace(export=lambda name: Path(name).write_bytes(b"scene fixture")),
    )
    monkeypatch.setattr(preview_inspection, "add_inspection_preview", lambda *args: None)
    import forge_mind.viz.render

    monkeypatch.setattr(
        forge_mind.viz.render,
        "render_workspace_preview",
        lambda source, target, **kwargs: target.write_bytes(b"preview image fixture"),
    )
    observed = []

    def measured(directory, rid, contract, **kwargs):
        observed.append(kwargs)
        result = part_measurement()
        if aggregate_failed:
            result.geometry_valid = False
            result.errors = ["Aggregate measurement failed in fixture"]
        elif selected:
            target = directory / "definition_cap.stl"
            target.write_bytes(b"selected mesh fixture")
            result.facts["definitions"] = {
                "cap": {
                    "mesh_delivery": {
                        "status": "available",
                        "step_file": "definition_cap.step",
                        "step_sha256": file_sha256(directory / "definition_cap.step"),
                        "stl_file": target.name,
                        "stl_sha256": file_sha256(target),
                        "stl_bytes": target.stat().st_size,
                        "body_count": 1,
                        "bounds_mm": [result.facts["bbox_min"], result.facts["bbox_max"]],
                        "tessellation": {
                            **result.facts["tessellation"],
                            "collapsed_triangles_removed": 0,
                        },
                        "export_consistency": result.facts["export_consistency"],
                    }
                }
            }
        return result

    monkeypatch.setattr(measure, "measure_revision", measured)
    processing.main(request)
    assert observed == [
        {"structured_assembly": True, "definition_mesh_ids": ("cap",) if selected else ()}
    ]
    result = json.loads((tmp_path / "processed.json").read_bytes())
    assert result["geometry_valid"] is not aggregate_failed
    if selected:
        manifest = json.loads((tmp_path / "definition-meshes.json").read_bytes())
        assert manifest["parts"]["cap"]["quantity"] == 2
        if aggregate_failed:
            assert manifest["parts"]["cap"]["status"] == "failed"
            assert result["errors"] == ["Aggregate measurement failed in fixture"]
            assert "definition_cap.stl" not in result["artifact_hashes"]
            assert not (tmp_path / "definition_cap.stl").exists()
        else:
            assert result["artifact_hashes"]["definition_cap.stl"] == file_sha256(
                tmp_path / "definition_cap.stl"
            )
    else:
        assert not (tmp_path / "definition-meshes.json").exists()
        assert not (tmp_path / "definition_cap.stl").exists()


@pytest.mark.parametrize("fault", [None, "hash", "size", "input"])
def test_typed_worker_transfer_checks_bound_mesh_outputs(tmp_path, monkeypatch, fault):
    directory = tmp_path / "revision"
    directory.mkdir()
    for name in ("program.py", "document.json"):
        (directory / name).write_bytes(b"immutable input fixture")
    client = executor.Executor("docker")
    client._typed_container_verified = True  # Mocked transfer admission, not runtime qualification.
    monkeypatch.setattr(executor.shutil, "which", lambda _: "fixture-docker")
    monkeypatch.setitem(executor.HOSTED_LIMITS, "file_bytes", 64)

    def container(_docker, scratch, _module, _request, _remaining, _cancelled, **kwargs):
        (scratch / "definition_cap.stl").write_bytes(
            b"mesh fixture" if fault != "size" else b"x" * 65
        )
        (scratch / "definition-meshes.json").write_bytes(b"manifest fixture")
        if fault == "input":
            (scratch / "document.json").write_bytes(b"changed source")
        hashes = {
            name: file_sha256(scratch / name)
            for name in (
                "program.py",
                "document.json",
                "definition_cap.stl",
                "definition-meshes.json",
            )
        }
        if fault == "hash":
            hashes["definition_cap.stl"] = "0" * 64
        result = Measurement(
            revision_id=RID,
            contract_hash=Contract().fingerprint(),
            engine_fingerprint="fixture",
            artifact_hashes=hashes,
            geometry_valid=False,
        )
        (scratch / "processed.json").write_text(result.model_dump_json())

    monkeypatch.setattr(client, "_run_container", container)

    def invoke():
        return client.process_document(
            directory, {"revision_id": RID, "contract": {}, "assets": {}}, 20, lambda: False
        )

    if fault is None:
        result = invoke()
        assert result["artifact_hashes"]["definition_cap.stl"] == file_sha256(
            directory / "definition_cap.stl"
        )
        assert (directory / "definition-meshes.json").read_bytes() == b"manifest fixture"
    else:
        with pytest.raises(ValueError):
            invoke()
    assert (directory / "document.json").read_bytes() == b"immutable input fixture"
