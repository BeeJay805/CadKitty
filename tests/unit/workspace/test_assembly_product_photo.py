"""Assembly photo controls use actual captured packages and mocked graphics, never CAD."""

import hashlib
import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_product_photo import PNG, write_result
from tests.unit.workspace.test_project_delivery import RID, saved_project
from tests.unit.workspace.test_topology_meshes import topology_fixture

from forge_mind.workspace import product_photo as worker
from forge_mind.workspace import product_photo_contract as contract
from forge_mind.workspace import product_photo_render as renderer
from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace import project_photo as photo
from forge_mind.workspace.document import DesignDocument


def replace_record(service, sid, record):
    with service.store.connect() as connection:
        connection.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(record), sid, RID),
        )


def assembly_capture(tmp_path, monkeypatch, *, include_stl=False, recorded=True):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    service.executor.mode = "trusted_local"
    if recorded:
        record = service.store.revision(sid, RID)
        hashes = record["measurement"]["artifact_hashes"]
        rows = [
            ("base", "base"),
            ("cap_a", "cap"),
            ("cap_b", "cap"),
            ("bolt", "bolt"),
            ("guide", "guide"),
        ]
        assembly = {
            "revision_id": RID,
            "step_sha256": hashes["part.step"],
            "engine_fingerprint": record["measurement"]["engine_fingerprint"],
            "occurrences": [
                {"path": path, "definition": part, "body_index": index, "volume_mm3": 1}
                for index, (path, part) in enumerate(rows)
            ],
        }
        record["measurement"]["facts"]["assembly"] = assembly
        for name, data in (
            ("assembly.json", json.dumps(assembly).encode()),
            ("topology.json", json.dumps(topology_fixture(5)).encode()),
        ):
            (source / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        replace_record(service, sid, record)
    package = delivery.create_project_export(service, sid, "owner", RID, include_stl=include_stl)
    calls = []

    def render(command, scratch, timeout, cancelled, **kwargs):
        assert not cancelled() and 0 < timeout <= 15
        request = json.loads((scratch / "request.json").read_bytes())
        if "source_kind" in request:
            assert {p.name for p in scratch.iterdir()} == {"request.json", "topology.json"}
            assert (scratch / "topology.json").read_bytes() == (
                source / "topology.json"
            ).read_bytes()

            def amend(value, _):
                value["scope"] = contract.ASSEMBLY_SCOPE
                value["rendering"]["source_tessellation"] = {
                    "purpose": "Selection preview only; not manufacturing geometry",
                    "relative_linear_deflection": 0.01,
                    "angular_deflection_radians": 0.1,
                    "selected_body_count": 3,
                    "discarded_zero_area_triangles": 0,
                    "vertex_precision": contract.DISPLAY_VERTEX_PRECISION,
                }
        else:
            assert (scratch / "part.stl").read_bytes() == (
                source / f"definition_{request['part_id']}.stl"
            ).read_bytes()

            def amend(value, _):
                value["scope"] = contract.CAPTURED_SCOPE

        calls.append(deepcopy(request))
        write_result(scratch, request, mutate=amend)

    monkeypatch.setattr(worker, "bounded_process", render)
    return service, sid, source, package, calls


def create(service, sid, package, **kwargs):
    return photo.create_project_photo(
        service, sid, "owner", package["package_id"], package["sha256"], **kwargs
    )


@pytest.mark.parametrize("include_stl", [False, True])
def test_default_renders_every_repeated_make_pose_without_stl_dependency(
    tmp_path, monkeypatch, include_stl
):
    service, sid, source, package, calls = assembly_capture(
        tmp_path, monkeypatch, include_stl=include_stl
    )
    before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    result = create(service, sid, package)
    receipt = result["receipt"]
    assert result["status"] == "available" and receipt["scope"] == contract.ASSEMBLY_SCOPE
    assert [(r["occurrence_path"], r["body_index"]) for r in receipt["selection"]["selected"]] == [
        ("base", 0),
        ("cap_a", 1),
        ("cap_b", 2),
    ]
    assert [(r["part_id"], r["role"]) for r in receipt["selection"]["omitted"]] == [
        ("bolt", "buy"),
        ("guide", "reference"),
    ]
    assert receipt["selection"]["total_bodies"] == 5
    assert calls == [{key: receipt[key] for key in contract.ASSEMBLY_REQUEST_KEYS}]
    assert "mesh_sha256" not in receipt and "part_id" not in receipt
    service.store.update(sid, selected_revision="c" * 32)
    image, readback = photo.read_project_photo(
        service,
        sid,
        "owner",
        package["package_id"],
        package["sha256"],
        result["photo_id"],
        receipt["image"]["sha256"],
    )
    assert image == PNG and readback == receipt
    assert before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    photo.discard_project_photo(service, sid, "owner", package["package_id"], result["photo_id"])
    assert not photo._directory(service, sid, package["package_id"], result["photo_id"]).exists()


def test_explicit_definition_keeps_legacy_member_variant_and_quantity(tmp_path, monkeypatch):
    service, sid, _, package, calls = assembly_capture(tmp_path, monkeypatch, include_stl=True)
    value = create(service, sid, package, part_id="cap")
    assert (
        value["receipt"]["quantity"] == 2 and value["receipt"]["scope"] == contract.CAPTURED_SCOPE
    )
    assert set(calls[0]) == contract.CAPTURED_REQUEST_KEYS


def test_explicit_stl_absence_and_legacy_mapping_absence_are_distinct(tmp_path, monkeypatch):
    service, sid, _, package, calls = assembly_capture(tmp_path, monkeypatch, recorded=False)
    assert create(service, sid, package)["code"] == "source_unavailable"
    assert create(service, sid, package, part_id="cap")["code"] == "missing_stl"
    assert calls == []


@pytest.mark.parametrize("name", ["document.json", "assembly.json", "topology.json", "part.step"])
@pytest.mark.parametrize("action", ["change", "delete"])
def test_recorded_source_tamper_or_disappearance_refuses(tmp_path, monkeypatch, name, action):
    service, sid, source, package, calls = assembly_capture(tmp_path, monkeypatch)
    if action == "change":
        (source / name).write_bytes(b"changed")
    else:
        (source / name).unlink()
    with pytest.raises((ValueError, OSError)):
        create(service, sid, package)
    assert calls == []


@pytest.mark.parametrize(
    "fault",
    ["body_bool", "body_duplicate", "definition", "missing_path", "provenance", "saved_json"],
)
def test_saved_native_correspondence_never_guessed(tmp_path, monkeypatch, fault):
    service, sid, source, package, calls = assembly_capture(tmp_path, monkeypatch)
    record = service.store.revision(sid, RID)
    assembly = record["measurement"]["facts"]["assembly"]
    if fault == "body_bool":
        assembly["occurrences"][0]["body_index"] = False
    elif fault == "body_duplicate":
        assembly["occurrences"][1]["body_index"] = 0
    elif fault == "definition":
        assembly["occurrences"][0]["definition"] = "cap"
    elif fault == "missing_path":
        assembly["occurrences"].pop()
    elif fault == "provenance":
        assembly["revision_id"] = "0" * 32
    data = json.dumps({} if fault == "saved_json" else assembly).encode()
    (source / "assembly.json").write_bytes(data)
    record["measurement"]["artifact_hashes"]["assembly.json"] = hashlib.sha256(data).hexdigest()
    replace_record(service, sid, record)
    with pytest.raises(ValueError):
        create(service, sid, package)
    assert calls == []


def test_optional_graphics_failure_rechecks_sources_and_preserves_complete_candidates(
    tmp_path, monkeypatch
):
    service, sid, source, package, _ = assembly_capture(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        raise contract.PhotoUnavailableError("renderer_unavailable", "Optional graphics absent")

    monkeypatch.setattr(photo, "render_mesh", unavailable)
    assert create(service, sid, package) == {
        "status": "unavailable",
        "code": "renderer_unavailable",
        "reason": "Optional graphics absent",
        "candidates": ["base", "cap"],
    }

    def tamper(*args, **kwargs):
        (source / "topology.json").write_bytes(b"changed")
        raise contract.PhotoUnavailableError("renderer_unavailable", "Optional graphics absent")

    monkeypatch.setattr(photo, "render_mesh", tamper)
    with pytest.raises(ValueError):
        create(service, sid, package)


def test_cancel_does_not_become_optional_absence(tmp_path, monkeypatch):
    service, sid, _, package, _ = assembly_capture(tmp_path, monkeypatch)

    def cancelled(*args, **kwargs):
        service.stop.set()
        raise contract.PhotoUnavailableError("render_failed", "deadline")

    monkeypatch.setattr(photo, "render_mesh", cancelled)
    with pytest.raises(ValueError, match="cancelled"):
        create(service, sid, package)


def test_durable_assembly_read_refuses_changed_source_and_foreign_owner(tmp_path, monkeypatch):
    service, sid, source, package, _ = assembly_capture(tmp_path, monkeypatch)
    value = create(service, sid, package)
    args = (
        package["package_id"],
        package["sha256"],
        value["photo_id"],
        value["receipt"]["image"]["sha256"],
    )
    with pytest.raises(ValueError):
        photo.read_project_photo(service, sid, "other", *args)
    (source / "topology.json").write_bytes(b"changed after publication")
    with pytest.raises(ValueError):
        photo.read_project_photo(service, sid, "owner", *args)


def test_renderer_limit_remains_optional_but_unrelated_failure_refuses(tmp_path, monkeypatch):
    service, sid, _, package, _ = assembly_capture(tmp_path, monkeypatch)

    def limited(*args, **kwargs):
        raise contract.PhotoUnavailableError("render_limit", "Saved display triangles exceed limit")

    monkeypatch.setattr(photo, "render_mesh", limited)
    value = create(service, sid, package)
    assert value["code"] == "render_limit" and value["candidates"] == ["base", "cap"]

    def unrelated(*args, **kwargs):
        raise ValueError("Container cleanup failed")

    monkeypatch.setattr(photo, "render_mesh", unrelated)
    with pytest.raises(ValueError, match="cleanup"):
        create(service, sid, package)


def test_flat_step_only_capture_keeps_missing_stl_without_rebuild(tmp_path, monkeypatch):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    record = service.store.revision(sid, RID)
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "base", "name": "Body", "kind": "box"}],
            "outputs": ["base"],
            "production": {"role": "make"},
        }
    ).model_dump()
    record["document"] = document
    data = json.dumps(document).encode()
    (source / "document.json").write_bytes(data)
    record["measurement"]["artifact_hashes"]["document.json"] = hashlib.sha256(data).hexdigest()
    replace_record(service, sid, record)
    package = delivery.create_project_export(service, sid, "owner", RID, include_stl=False)
    monkeypatch.setattr(
        photo,
        "render_mesh",
        lambda *args, **kwargs: pytest.fail("Flat STEP-only must not render/rebuild"),
    )
    assert create(service, sid, package)["code"] == "missing_stl"
    with_stl = delivery.create_project_export(service, sid, "owner", RID, include_stl=True)
    assert with_stl["made_parts"] == [
        {
            "part_id": "part",
            "quantity": 1,
            "files": ["make/part.step", "make/part.stl"],
            "frame": "document_local",
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "bool_index",
        "missing_omitted",
        "wrong_role",
        "missing_tessellation",
        "bool_discard",
        "precision",
    ],
)
def test_assembly_receipt_strict_variants_refuse(tmp_path, monkeypatch, mutation):
    service, sid, _, package, _ = assembly_capture(tmp_path, monkeypatch)
    value = create(service, sid, package)
    receipt = deepcopy(value["receipt"])
    request = deepcopy({key: receipt[key] for key in contract.ASSEMBLY_REQUEST_KEYS})
    if mutation == "bool_index":
        receipt["selection"]["selected"][0]["body_index"] = False
    elif mutation == "missing_omitted":
        receipt["selection"]["omitted"].pop()
    elif mutation == "wrong_role":
        receipt["selection"]["selected"][0]["role"] = "reference"
    elif mutation == "missing_tessellation":
        receipt["rendering"].pop("source_tessellation")
    elif mutation == "bool_discard":
        receipt["rendering"]["source_tessellation"]["discarded_zero_area_triangles"] = False
    else:
        receipt["rendering"]["source_tessellation"]["vertex_precision"] = "Exact print STL"
    with pytest.raises(ValueError):
        contract.validate_receipt(receipt, request)


def test_worker_assembly_selection_is_once_and_rendered_without_transform(tmp_path, monkeypatch):
    import numpy as np

    from forge_mind.workspace import occurrence_render

    topology = topology_fixture(5)
    content = json.dumps(topology).encode()
    request = {
        "version": 1,
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "package_id": "c" * 32,
        "package_sha256": "d" * 64,
        "preset": "plastic",
        "frame": contract.FRAME,
        "source_kind": "saved_topology_made_assembly",
        "source": {
            key: hashlib.sha256(content).hexdigest() if key == "topology_sha256" else "e" * 64
            for key in contract.ASSEMBLY_SOURCE_KEYS
        },
        "selection": {
            "total_bodies": 5,
            "selected": [
                {"occurrence_path": f"root_{i}", "part_id": "base", "role": "make", "body_index": i}
                for i in [0, 2]
            ],
            "omitted": [
                {
                    "occurrence_path": f"root_{i}",
                    "part_id": "ref",
                    "role": "reference",
                    "body_index": i,
                }
                for i in [1, 3, 4]
            ],
        },
    }
    (tmp_path / "topology.json").write_bytes(content)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    selected = []
    original = occurrence_render.selected_topology_meshes

    def select(value, indices):
        selected.append(indices)
        return original(value, indices)

    monkeypatch.setattr(occurrence_render, "selected_topology_meshes", select)

    def render(triangles, preset):
        assert np.array_equal(triangles, original(topology, [0, 2])[0].triangles)
        scratch = tmp_path / "facts"
        scratch.mkdir()
        write_result(
            scratch,
            {
                "project_id": "a" * 32,
                "revision_id": "b" * 32,
                "mesh_sha256": "f" * 64,
                "preset": preset,
                "frame": contract.FRAME,
                "version": 1,
            },
        )
        return PNG, json.loads((scratch / "receipt.json").read_bytes())["rendering"]

    monkeypatch.setattr(renderer, "_render_triangles", render)
    result = renderer.render_product_photo(request_path)
    assert selected == [[0, 2]] and result["scope"] == contract.ASSEMBLY_SCOPE
    assert result["rendering"]["source_tessellation"]["selected_body_count"] == 2
    assert content == (tmp_path / "topology.json").read_bytes()
    assert not (tmp_path / "part.stl").exists()
