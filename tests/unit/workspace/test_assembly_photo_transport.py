"""Actual SDK photo delivery with verified ZIPs and replayed images, never a renderer."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from tests.unit.workspace.test_mesh_handoff import fixture_bytes
from tests.unit.workspace.test_product_photo_transport import PNG, envelope
from tests.unit.workspace.test_project_delivery import RID as SAVED_RID
from tests.unit.workspace.test_project_delivery import saved_project
from tests.unit.workspace.test_project_export_local import saved_package
from tests.unit.workspace.test_project_export_remote import RESOURCE, result, rpc
from tests.unit.workspace.test_project_members_remote import app
from tests.unit.workspace.test_project_photo_delivery_transport import (
    ALTERNATE_ID,
    PHOTO_ID,
    captured_photo,
    jobs,
    local_boundary,
    local_call,
    photo_pair,
    remote_export,
)

from forge_mind.workspace import mcp_server, project_delivery, project_photo
from forge_mind.workspace.handoff import handoff_context
from forge_mind.workspace.product_photo_contract import ASSEMBLY_SCOPE, DISPLAY_VERTEX_PRECISION
from forge_mind.workspace.production import manufacture_bundle
from forge_mind.workspace.project_members import captured_cad_files, captured_photo_context
from forge_mind.workspace.project_photo_transport import validate_project_photo


def nested_package(tmp_path):
    """Create a valid metadata/byte capture with nested repeated instances, no CAD build."""
    source = tmp_path / "source"
    document, hashes = fixture_bytes(source)
    raw = document.model_dump()
    roots = raw["assembly"]["roots"]
    raw["assembly"]["subassemblies"] = [
        {
            "id": "caps",
            "name": "Nested caps",
            "children": [row for row in roots if row["definition"] == "cap"],
        }
    ]
    raw["assembly"]["roots"] = [row for row in roots if row["definition"] != "cap"] + [
        {"id": "nested", "definition": "caps"}
    ]
    document = type(document).model_validate(raw)
    content = document.model_dump_json(indent=2).encode()
    (source / "document.json").write_bytes(content)
    hashes["document.json"] = hashlib.sha256(content).hexdigest()
    step = b"Saved overall STEP transport fixture; not native geometry evidence"
    (source / "part.step").write_bytes(step)
    hashes["part.step"] = hashlib.sha256(step).hexdigest()
    from tests.unit.workspace.test_project_export_local import RID, SID

    context = handoff_context(
        {
            "id": SID,
            "proposal": {"revision_id": RID},
            "engineering": {"version": 0, "inputs": {}},
            "source_fingerprint": "fixture-source",
        },
        {
            "id": RID,
            "document": document.model_dump(),
            "measurement": {"engine_fingerprint": "fixture-engine"},
        },
        "fixture-export-engine",
    )
    path = manufacture_bundle(
        document,
        source,
        tmp_path / "saved.zip",
        hashes,
        sources={"fingerprint": "fixture-source"},
        engineering_evidence={"revision_id": RID, "physical_readiness": "unverified"},
        engineering_inputs={"version": 0, "inputs": {}},
        revision_context=context,
        include_assembly_step=True,
    )
    data = path.read_bytes()
    return {
        **project_delivery._package_summary(path),
        "package_id": "c" * 32,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }, data


def assembly_photo(package, context, preset="plastic"):
    """Replay the strict assembly result from independently verified capture identities."""
    value = envelope(package["project_id"], preset)
    receipt = value["receipt"]
    receipt.pop("mesh_sha256")
    receipt.update(
        revision_id=package["revision_id"],
        package_id=package["package_id"],
        package_sha256=package["sha256"],
        source_kind="saved_topology_made_assembly",
        scope=ASSEMBLY_SCOPE,
        source={
            "saved_document_sha256": context["captured_document_sha256"],
            "captured_document_sha256": context["captured_document_sha256"],
            "assembly_step_sha256": context["assembly_step_sha256"],
            "assembly_facts_sha256": "1" * 64,
            "topology_sha256": "2" * 64,
        },
    )
    all_paths = sorted(
        row["occurrence_path"] for key in ("selected", "omitted") for row in context[key]
    )
    indices = {path: index for index, path in enumerate(all_paths)}
    receipt["selection"] = {
        key: [{**row, "body_index": indices[row["occurrence_path"]]} for row in context[key]]
        for key in ("selected", "omitted")
    }
    receipt["selection"]["total_bodies"] = len(all_paths)
    receipt["rendering"]["source_tessellation"] = {
        "purpose": "selection/display tessellation",
        "relative_linear_deflection": 0.05,
        "angular_deflection_radians": 0.15,
        "selected_body_count": len(context["selected"]),
        "discarded_zero_area_triangles": 0,
        "vertex_precision": DISPLAY_VERTEX_PRECISION,
    }
    value.update(status="available", photo_id=PHOTO_ID)
    return value


@pytest.mark.parametrize("include_stl", [False, True])
def test_local_default_assembly_delivers_all_repeated_makes_without_requiring_stl(
    tmp_path, include_stl
):
    package, data = saved_package(tmp_path, include_stl)
    context = captured_photo_context(data, package)
    assert context["structured_assembly"] is True
    assert context["make_quantities"] == [
        {"part_id": "base", "quantity": 1},
        {"part_id": "cap", "quantity": 2},
    ]
    value = assembly_photo(package, context)
    transport, calls = local_boundary(package, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=transport,
    )
    response = local_call(adapter, include_stl=include_stl, include_individual_files=True)
    assert not response.is_error
    metadata = response.structured_content
    preview = metadata["preview"]
    assert preview["receipt"] == value["receipt"]
    assert len(preview["receipt"]["selection"]["selected"]) == 3
    assert {row["role"] for row in preview["receipt"]["selection"]["omitted"]} == {
        "buy",
        "reference",
    }
    assert calls[-1][-1] == {"preset": "plastic", "part_id": None, "include_alternate": True}
    assert Path(metadata["path"]).read_bytes() == data
    assert Path(preview["path"]).read_bytes() == PNG
    assert json.loads(Path(preview["receipt_path"]).read_bytes()) == value["receipt"]
    alternate = metadata["preview_alternate"]
    assert alternate["receipt"] == photo_pair(value)["alternate"]["receipt"]
    assert Path(alternate["path"]).read_bytes() == PNG
    assert json.loads(Path(alternate["receipt_path"]).read_bytes()) == alternate["receipt"]
    assert len(response.content) == 3
    assert base64.b64decode(response.content[2].data, validate=True) == PNG
    assert base64.b64decode(response.content[1].data, validate=True) == PNG
    assert json.loads(response.content[0].text) == metadata
    assert "capture_context" not in metadata and "image_base64" not in json.dumps(metadata)
    if not include_stl:
        assert all(row["format"] == "step" for row in metadata["individual_files"])
    for row in metadata["individual_files"]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]


def test_nested_repeated_paths_survive_default_sdk_delivery(tmp_path):
    package, data = nested_package(tmp_path)
    context = captured_photo_context(data, package)
    assert [row["occurrence_path"] for row in context["selected"]] == [
        "base",
        "nested/cap_a",
        "nested/cap_b",
    ]
    value = assembly_photo(package, context)
    transport, _ = local_boundary(package, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=transport,
    )
    response = local_call(adapter)
    assert not response.is_error
    selected = response.structured_content["preview"]["receipt"]["selection"]["selected"]
    assert [row["occurrence_path"] for row in selected] == ["base", "nested/cap_a", "nested/cap_b"]
    assert len({row["body_index"] for row in selected}) == 3
    assert base64.b64decode(response.content[1].data, validate=True) == PNG
    assert Path(response.structured_content["path"]).read_bytes() == data


def test_unique_structured_default_requires_assembly_result_not_definition_fallback(tmp_path):
    package, data = saved_package(tmp_path, True, unique_made=True)
    members = captured_cad_files(data, package)
    context = captured_photo_context(data, package)
    value = assembly_photo(package, context)
    metadata, image = validate_project_photo(
        value,
        package,
        members,
        "plastic",
        None,
        capture_context=context,
    )
    assert image == PNG and metadata["receipt"]["source_kind"] == "saved_topology_made_assembly"
    with pytest.raises(ValueError, match="assembly photo receipt"):
        validate_project_photo(
            captured_photo(package, members, "base", "plastic"),
            package,
            members,
            "plastic",
            None,
            capture_context=context,
        )


def test_explicit_part_keeps_definition_receipt_and_step_only_absence(tmp_path):
    package, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, package)
    context = captured_photo_context(data, package)
    value = captured_photo(package, members, "cap", "metal")
    metadata, image = validate_project_photo(
        value,
        package,
        members,
        "metal",
        "cap",
        capture_context=context,
    )
    assert image == PNG and metadata["receipt"] == value["receipt"]
    assert metadata["receipt"]["quantity"] == 2
    step_root = tmp_path / "step-only"
    step_root.mkdir()
    step_package, step_data = saved_package(step_root, False)
    absent = {
        "status": "unavailable",
        "code": "missing_stl",
        "reason": "No captured definition STL",
        "candidates": ["base", "cap"],
    }
    transport, _ = local_boundary(step_package, step_data, absent)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=transport,
    )
    response = local_call(adapter, photo_part_id="cap")
    assert not response.is_error and response.structured_content["preview"] == absent
    assert len(response.content) == 1
    assert response.structured_content["preview_alternate"] == photo_pair(absent)["alternate"]


def test_none_is_exact_legacy_and_never_reads_photo_context(tmp_path, monkeypatch):
    package, data = saved_package(tmp_path)
    transport, calls = local_boundary(package, data, {})
    monkeypatch.setattr(
        "forge_mind.workspace.project_members.captured_photo_context",
        lambda *_: pytest.fail("Disabled photography must not read extra context"),
    )
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=transport,
    )
    response = local_call(adapter, preview="none")
    assert not response.is_error and len(response.content) == 1
    assert {
        key: value for key, value in response.structured_content.items() if key != "path"
    } == package
    assert len(calls) == 2 and "preview" not in response.structured_content
    assert "preview_alternate" not in response.structured_content


@pytest.mark.parametrize(
    "code",
    [
        "source_unavailable",
        "render_limit",
        "execution_unqualified",
        "renderer_unavailable",
        "render_failed",
    ],
)
def test_default_optional_absence_preserves_step_only_cad(tmp_path, code):
    package, data = saved_package(tmp_path)
    context = captured_photo_context(data, package)
    value = {
        "status": "unavailable",
        "code": code,
        "reason": "Declared optional absence",
        "candidates": ["base", "cap"],
    }
    metadata, image = validate_project_photo(
        value,
        package,
        captured_cad_files(data, package),
        "plastic",
        None,
        capture_context=context,
    )
    assert metadata == value and image is None
    if code in {"source_unavailable", "render_limit"}:
        with pytest.raises(ValueError, match="capture context"):
            validate_project_photo(
                value, package, captured_cad_files(data, package), "plastic", None
            )


@pytest.mark.parametrize(
    "field", ["package_sha256", "captured_document_sha256", "quantity", "path"]
)
def test_capture_context_cannot_substitute_another_package_or_drop_repeated_occurrence(
    tmp_path, field
):
    package, data = saved_package(tmp_path)
    context = captured_photo_context(data, package)
    value = assembly_photo(package, context)
    changed = copy.deepcopy(context)
    if field in {"package_sha256", "captured_document_sha256"}:
        changed[field] = "9" * 64
    elif field == "quantity":
        changed["make_quantities"][0]["quantity"] = True
    else:
        changed["selected"].pop()
    with pytest.raises(ValueError):
        validate_project_photo(
            value,
            package,
            captured_cad_files(data, package),
            "plastic",
            None,
            capture_context=changed,
        )


@pytest.mark.parametrize(
    "field",
    [
        "missing_selected",
        "missing_omitted",
        "role",
        "path",
        "body_index",
        "total_bodies",
        "captured_document_sha256",
        "assembly_step_sha256",
        "package_sha256",
        "preset",
        "source_kind",
        "png",
    ],
)
def test_assembly_receipt_mismatch_refuses_local_publication(tmp_path, field):
    package, data = saved_package(tmp_path)
    value = assembly_photo(package, captured_photo_context(data, package))
    receipt = value["receipt"]
    if field == "missing_selected":
        receipt["selection"]["selected"].pop()
    elif field == "missing_omitted":
        receipt["selection"]["omitted"].pop()
    elif field == "role":
        receipt["selection"]["omitted"][0]["role"] = "reference"
    elif field == "path":
        receipt["selection"]["selected"][0]["occurrence_path"] = "invented/path"
    elif field == "body_index":
        receipt["selection"]["selected"][0]["body_index"] = True
    elif field == "total_bodies":
        receipt["selection"]["total_bodies"] += 1
    elif field in {"captured_document_sha256", "assembly_step_sha256"}:
        receipt["source"][field] = "9" * 64
    elif field == "package_sha256":
        receipt[field] = "9" * 64
    elif field == "preset":
        receipt[field] = "metal"
    elif field == "source_kind":
        receipt[field] = "whole_context_fallback"
    else:
        value["image_base64"] = base64.b64encode(PNG + b"changed").decode()
    transport, calls = local_boundary(package, data, value)
    adapter = mcp_server.WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path / "exports",
        transport=transport,
    )
    response = local_call(adapter)
    assert response.is_error
    assert list(adapter.export_dir.iterdir()) == []
    assert [row[-1] for row in calls if row[0] == "DISCARD"] == [ALTERNATE_ID, PHOTO_ID]


def replay_assembly(monkeypatch):
    """Replay only renderer results; package capture, ownership and SDK dispatch stay real."""
    photos = {}
    discarded = []

    def create(
        service, sid, principal, package_id, digest, *, preset="plastic", part_id=None, timeout=15
    ):
        assert part_id is None
        data, package = project_delivery.read_project_export(
            service, sid, principal, package_id, digest
        )
        value = assembly_photo(package, captured_photo_context(data, package), preset)
        photos[(sid, package_id, PHOTO_ID)] = copy.deepcopy(value)
        return value

    def create_pair(*args, **kwargs):
        value = photo_pair(create(*args, **kwargs))
        sid, package_id = args[1], args[3]
        for item in value.values():
            photos[(sid, package_id, item["photo_id"])] = copy.deepcopy(item)
        return value

    def read(service, sid, principal, package_id, digest, photo_id, image_sha256):
        project_delivery.read_project_export(service, sid, principal, package_id, digest)
        value = photos[(sid, package_id, photo_id)]
        assert value["receipt"]["image"]["sha256"] == image_sha256
        return base64.b64decode(value["image_base64"]), value["receipt"]

    def discard(service, sid, principal, package_id, photo_id):
        service.store.get(sid, principal)
        discarded.append((package_id, photo_id))
        photos.pop((sid, package_id, photo_id), None)

    monkeypatch.setattr(project_photo, "create_project_photo", create)
    monkeypatch.setattr(project_photo, "create_project_photo_pair", create_pair)
    monkeypatch.setattr(project_photo, "read_project_photo", read)
    monkeypatch.setattr(project_photo, "discard_project_photo", discard)
    return photos, discarded


def test_remote_default_step_only_assembly_image_and_durable_owned_readback(tmp_path, monkeypatch):
    service, sid, directory = saved_project(tmp_path, monkeypatch)
    original = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    photos, _ = replay_assembly(monkeypatch)
    before = jobs(service)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(client, sid)
        assert not delivered.get("isError")
        metadata = delivered["structuredContent"]
        assert metadata["include_stl"] is False
        assert "capture_context" not in metadata
        serialized = json.dumps(metadata)
        # Occurrence paths identify assembly parts; local filesystem fields are forbidden.
        assert '"path":' not in serialized and '"receipt_path":' not in serialized
        preview = metadata["preview"]
        assert preview["receipt"]["source_kind"] == "saved_topology_made_assembly"
        assert len(preview["receipt"]["selection"]["selected"]) == 3
        assert base64.b64decode(delivered["content"][1]["data"], validate=True) == PNG
        alternate = metadata["preview_alternate"]
        assert alternate["receipt"]["version"] == 2
        assert (
            alternate["receipt"] == photos[(sid, metadata["package_id"], ALTERNATE_ID)]["receipt"]
        )
        assert len(delivered["content"]) == 3
        assert base64.b64decode(delivered["content"][2]["data"], validate=True) == PNG
        fetched_alternate = result(
            rpc(client, "owner", "resources/read", {"uri": alternate["uri"]})
        )["contents"][0]
        assert base64.b64decode(fetched_alternate["blob"], validate=True) == PNG
        uri = preview["uri"]
        assert metadata["revision_id"] == SAVED_RID
        fetched = result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]
        assert fetched["mimeType"] == "image/png" and base64.b64decode(fetched["blob"]) == PNG
        assert (
            rpc(client, "foreign", "resources/read", {"uri": uri}).json()["error"]["code"] == -32603
        )
        assert (
            base64.b64decode(
                result(rpc(client, "owner", "resources/read", {"uri": uri}))["contents"][0]["blob"]
            )
            == PNG
        )
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as restarted:
        assert (
            base64.b64decode(
                result(rpc(restarted, "owner", "resources/read", {"uri": uri}))["contents"][0][
                    "blob"
                ]
            )
            == PNG
        )
        value = photos[(sid, metadata["package_id"], PHOTO_ID)]
        value["receipt"]["selection"]["selected"][0]["occurrence_path"] = "wrong/path"
        assert "error" in rpc(restarted, "owner", "resources/read", {"uri": uri}).json()
    assert {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    } == original
    assert jobs(service) == before == 0


def test_remote_receipt_mismatch_cleans_unpublished_photo_and_package(tmp_path, monkeypatch):
    service, sid, _ = saved_project(tmp_path, monkeypatch)
    _, discarded = replay_assembly(monkeypatch)
    original_create = project_photo.create_project_photo_pair

    def mismatched(*args, **kwargs):
        value = original_create(*args, **kwargs)
        value["primary"]["receipt"]["source"]["captured_document_sha256"] = "9" * 64
        return value

    monkeypatch.setattr(project_photo, "create_project_photo_pair", mismatched)
    with TestClient(app(service), base_url=RESOURCE.removesuffix("/mcp")) as client:
        delivered = remote_export(client, sid)
        assert delivered.get("isError")
    assert len(discarded) == 2 and [row[1] for row in discarded] == [ALTERNATE_ID, PHOTO_ID]
    package_path = service.store.root / "project-exports" / sid / discarded[0][0]
    assert not package_path.exists()
    assert jobs(service) == 0
