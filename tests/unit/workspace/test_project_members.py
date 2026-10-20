"""Captured member tests use saved byte fixtures, never native CAD generation."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from zipfile import ZipFile

import pytest
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace import project_members as members
from forge_mind.workspace.store import WorkspaceError


def saved_capture(tmp_path, monkeypatch, *, include_stl=True):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    receipt = delivery.create_project_export(service, sid, "owner", RID, include_stl=include_stl)
    content, _ = delivery.read_project_export(
        service, sid, "owner", receipt["package_id"], receipt["sha256"]
    )
    return content, receipt, source


def rewrite(content, receipt, change, *, rebind_manifest=False):
    with ZipFile(io.BytesIO(content)) as archive:
        files = {item.filename: archive.read(item) for item in archive.infolist()}
    change(files)
    if rebind_manifest:
        manifest = json.loads(files["manifest.json"])
        manifest["files"] = {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
            for name, data in files.items()
            if name != "manifest.json"
        }
        files["manifest.json"] = json.dumps(manifest).encode()
    target = io.BytesIO()
    with ZipFile(target, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    changed = target.getvalue()
    result = copy.deepcopy(receipt)
    result.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
    return changed, result


@pytest.mark.parametrize("include_stl", [False, True])
def test_verified_saved_capture_delivers_exact_members_and_preserves_inputs(
    tmp_path, monkeypatch, include_stl
):
    content, receipt, source = saved_capture(tmp_path, monkeypatch, include_stl=include_stl)
    original_receipt = copy.deepcopy(receipt)
    original_source = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    rows = members.captured_cad_files(content, receipt)
    assert [row["file_id"] for row in rows] == ["assembly-step"] + [
        f"part-{part}-{format_name}"
        for part in ("base", "cap")
        for format_name in (("step", "stl") if include_stl else ("step",))
    ]
    assert rows[0]["role"] == "assembly" and rows[0]["part_id"] is None
    assert rows[0]["frame"] == "cad_z_up_mm_world" and rows[0]["quantity"] == 1
    with ZipFile(io.BytesIO(content)) as archive:
        for row in rows:
            expected = archive.read(row["archive_path"])
            assert members.read_captured_cad_file(content, receipt, row["file_id"]) == expected
            assert row["bytes"] == len(expected)
            assert row["sha256"] == hashlib.sha256(expected).hexdigest()
            assert set(row) == {
                "file_id",
                "archive_path",
                "format",
                "part_id",
                "quantity",
                "frame",
                "role",
                "sha256",
                "bytes",
            }
    made = [row for row in rows if row["role"] == "make"]
    assert {row["part_id"]: row["quantity"] for row in made} == {"base": 1, "cap": 2}
    assert all(row["frame"] == "definition_local" for row in made)
    assert not any(row["part_id"] in ("bolt", "guide") for row in rows)
    assert receipt == original_receipt
    assert hashlib.sha256(content).hexdigest() == receipt["sha256"]
    assert original_source == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}


@pytest.mark.parametrize(
    "file_id", ["unknown", "part-guide-step", "part-base-stl", "../base", None]
)
def test_unknown_or_absent_stl_is_not_read(tmp_path, monkeypatch, file_id):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch, include_stl=False)
    with pytest.raises(WorkspaceError, match="unknown file identifier"):
        members.read_captured_cad_file(content, receipt, file_id)


@pytest.mark.parametrize("mutation", ["content", "hash", "length", "boolean_length"])
def test_package_binding_refuses_before_zip_open(tmp_path, monkeypatch, mutation):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch)
    if mutation == "content":
        content += b"changed"
    elif mutation == "hash":
        receipt["sha256"] = "0" * 64
    elif mutation == "length":
        receipt["bytes"] += 1
    else:
        receipt["bytes"] = True
    monkeypatch.setattr(members, "ZipFile", lambda *args: pytest.fail("Cannot open unbound ZIP"))
    for operation in (
        members.captured_cad_files,
        lambda c, r: members.read_captured_cad_file(c, r, "assembly-step"),
    ):
        with pytest.raises(WorkspaceError, match="length or checksum"):
            operation(content, receipt)


@pytest.mark.parametrize(
    "mutation", ["missing", "tamper", "extra", "duplicate", "traversal", "symlink"]
)
def test_member_inventory_and_hash_refusal(tmp_path, monkeypatch, mutation):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch)
    if mutation in ("duplicate", "symlink"):
        from zipfile import ZipInfo

        target = io.BytesIO()
        with ZipFile(io.BytesIO(content)) as original, ZipFile(target, "w") as archive:
            for item in original.infolist():
                if mutation == "symlink" and item.filename == "make/definition_base.step":
                    item = ZipInfo(item.filename)
                    item.create_system = 3
                    item.external_attr = 0o120777 << 16
                archive.writestr(item, original.read(item.filename))
            if mutation == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("make/definition_base.step", b"second")
        content = target.getvalue()
        receipt.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    else:

        def change(files):
            if mutation == "missing":
                del files["make/definition_base.step"]
            elif mutation == "tamper":
                files["make/definition_base.step"] = b"changed base bytes"
            elif mutation == "extra":
                files["make/definition_guide.step"] = b"reference cannot become made"
            else:
                files["../escaped.step"] = b"unsafe"

        content, receipt = rewrite(content, receipt, change, rebind_manifest=mutation == "extra")
    with pytest.raises(WorkspaceError):
        members.captured_cad_files(content, receipt)
    with pytest.raises(WorkspaceError):
        members.read_captured_cad_file(content, receipt, "part-base-step")


@pytest.mark.parametrize(
    "mutation",
    ["quantity", "boolean_quantity", "duplicate_part", "frame", "stl_option", "assembly_frame"],
)
def test_malformed_receipt_does_not_change_allowed_members(tmp_path, monkeypatch, mutation):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch)
    if mutation == "quantity":
        receipt["made_parts"][0]["quantity"] += 1
    elif mutation == "boolean_quantity":
        # True compares equal to1; reject it through strict metadata validation.
        receipt["made_parts"][0]["quantity"] = True
    elif mutation == "duplicate_part":
        receipt["made_parts"].append(copy.deepcopy(receipt["made_parts"][0]))
    elif mutation == "frame":
        receipt["made_parts"][0]["frame"] = "cad_z_up_mm_world"
    elif mutation == "stl_option":
        receipt["include_stl"] = 1
    else:
        receipt["assembly"]["frame"] = "definition_local"
    with pytest.raises(WorkspaceError):
        members.captured_cad_files(content, receipt)


@pytest.mark.parametrize("mutation", ["reference_role", "nonfinite", "duplicate_json_key"])
def test_bound_source_metadata_remains_literal_and_strict(tmp_path, monkeypatch, mutation):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch)

    def change(files):
        document = json.loads(files["document.json"])
        if mutation == "reference_role":
            document["assembly"]["parts"][0]["production"]["role"] = "reference"
            files["document.json"] = json.dumps(document).encode()
        elif mutation == "nonfinite":
            document["variables"] = {"unexpected": float("inf")}
            files["document.json"] = json.dumps(document).encode()
        else:
            files["document.json"] = b'{"assembly":null,"assembly":{}}'

    content, receipt = rewrite(content, receipt, change, rebind_manifest=True)
    with pytest.raises(WorkspaceError):
        members.captured_cad_files(content, receipt)


@pytest.mark.parametrize("limit", ["MAX_PACKAGE_BYTES", "MAX_JOB_BYTES", "MAX_FILE_BYTES"])
def test_limits_refuse_before_member_decompression(tmp_path, monkeypatch, limit):
    content, receipt, _ = saved_capture(tmp_path, monkeypatch)
    monkeypatch.setattr(members, limit, 1)
    monkeypatch.setattr(
        ZipFile, "open", lambda *args, **kwargs: pytest.fail("Oversized members cannot be opened")
    )
    with pytest.raises(WorkspaceError):
        members.captured_cad_files(content, receipt)


def test_flat_document_local_frame_and_part_id():
    files = {
        "document.json": json.dumps({"production": {"role": "make"}}).encode(),
        "production.json": json.dumps(
            {"make": [{"part_id": "part", "role": "make", "quantity": 1}]}
        ).encode(),
        "assembly/part.step": b"flat overall saved STEP",
        "make/part.step": b"flat made saved STEP",
    }
    assembly = {
        "file": "assembly/part.step",
        "sha256": hashlib.sha256(files["assembly/part.step"]).hexdigest(),
        "frame": "cad_z_up_mm_world",
        "units": "mm",
    }
    files["manifest.json"] = json.dumps(
        {
            "include_assembly_step": True,
            "assembly": assembly,
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                for name, data in files.items()
            },
        }
    ).encode()
    target = io.BytesIO()
    with ZipFile(target, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    content = target.getvalue()
    receipt = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "include_stl": False,
        "assembly": assembly,
        "made_parts": [
            {
                "part_id": "part",
                "quantity": 1,
                "frame": "document_local",
                "files": ["make/part.step"],
            }
        ],
    }
    rows = members.captured_cad_files(content, receipt)
    assert rows[1]["file_id"] == "part-part-step" and rows[1]["frame"] == "document_local"
    assert (
        members.read_captured_cad_file(content, receipt, "part-part-step")
        == files["make/part.step"]
    )
