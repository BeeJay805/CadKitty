"""Public bundle controls use captured byte fixtures, never CAD or providers."""

from __future__ import annotations

import json

import pytest
from scripts.customer_outcome_bench import sha

from tests.unit.workspace.test_project_members import rewrite, saved_capture


def inputs(tmp_path, monkeypatch):
    raw, receipt, _ = saved_capture(tmp_path, monkeypatch)
    package = tmp_path / "captured.zip"
    package.write_bytes(raw)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    metadata = {
        "case_id": "owned-example",
        "title": "Owned CAD example",
        "prompt": "Make the requested editable parts.",
        "model": "scripted-fixture",
        "limits": {"seconds": 600, "actions": 16},
        "origin": "saved_partial",
        "rights_owned_or_authorized": True,
    }
    metadata_path = tmp_path / "case.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return raw, receipt, package, receipt_path, metadata, metadata_path


def export(package, receipt_path, metadata_path, output):
    from scripts.export_showcase import export_showcase

    return export_showcase(package, receipt_path, metadata_path, output)


def test_exact_portable_curated_bundle_keeps_uncertainty_and_original_bytes(tmp_path, monkeypatch):
    from scripts.customer_outcome_bench import verified_archive

    raw, receipt, package, receipt_path, metadata, metadata_path = inputs(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in (package, receipt_path, metadata_path)}
    output = tmp_path / "public"
    manifest = export(package, receipt_path, metadata_path, output)
    original = verified_archive(raw, receipt)
    assert manifest["case"] == metadata
    assert manifest["physical_readiness"] == "unverified"
    assert manifest["full_task_success"] is None
    assert manifest["package"] == {"sha256": receipt["sha256"], "bytes": receipt["bytes"]}
    assert (output / "assembly/part.step").read_bytes() == original["assembly/part.step"]
    assert (output / "document.json").read_bytes() == original["document.json"]
    assert "customer-guidance.json" not in manifest["files"]
    assert "source_archive_path" not in manifest["files"]["README.md"]
    assert manifest["files"]["GUIDE.md"]["source_member_sha256"] == sha(
        original["customer-guidance.json"]
    )
    assert not (output / "receipt.json").exists()
    for name in (
        "operation-record.json",
        "engineering-inputs.json",
        "sources.json",
        "history.json",
    ):
        assert not (output / name).exists()
    for name, record in manifest["files"].items():
        content = (output / name).read_bytes()
        assert record["sha256"] == sha(content) and record["bytes"] == len(content)
    public_text = json.dumps(manifest) + (output / "README.md").read_text(encoding="utf-8")
    assert receipt["project_id"] not in public_text
    assert receipt["revision_id"] not in public_text
    assert str(tmp_path) not in public_text
    assert "saved_partial" in public_text
    assert {path: path.read_bytes() for path in before} == before


def test_same_inputs_produce_reproducible_content_in_two_new_directories(tmp_path, monkeypatch):
    _, _, package, receipt_path, _, metadata_path = inputs(tmp_path, monkeypatch)
    export(package, receipt_path, metadata_path, tmp_path / "one")
    export(package, receipt_path, metadata_path, tmp_path / "two")

    def files(root):
        return {
            path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()
        }

    assert files(tmp_path / "one") == files(tmp_path / "two")


@pytest.mark.parametrize(
    "origin", ["actual_model_generated", "saved_partial", "maintainer_authored"]
)
def test_origin_is_explicit_and_never_promotes_full_task(tmp_path, monkeypatch, origin):
    _, _, package, receipt_path, metadata, metadata_path = inputs(tmp_path, monkeypatch)
    metadata["origin"] = origin
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    result = export(package, receipt_path, metadata_path, tmp_path / "new")
    assert result["case"]["origin"] == origin
    assert result["full_task_success"] is None


@pytest.mark.parametrize(
    "change", ["rights", "unknown", "success", "physical", "missing_origin", "limits"]
)
def test_unsafe_or_ambiguous_public_metadata_refuses_before_creating_output(
    tmp_path, monkeypatch, change
):
    _, _, package, receipt_path, metadata, metadata_path = inputs(tmp_path, monkeypatch)
    if change == "rights":
        metadata["rights_owned_or_authorized"] = False
    elif change == "unknown":
        metadata["store_path"] = "private-store-canary"
    elif change == "success":
        metadata["full_task_success"] = True
    elif change == "physical":
        metadata["physical_readiness"] = "verified"
    elif change == "missing_origin":
        del metadata["origin"]
    else:
        metadata["limits"]["seconds"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        export(package, receipt_path, metadata_path, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["traversal", "duplicate", "symlink", "false_hash", "member_hash", "oversize"]
)
def test_existing_archive_reader_refuses_hostile_capture(tmp_path, monkeypatch, mutation):
    import io
    from zipfile import ZipFile, ZipInfo

    from forge_mind.workspace import project_members

    raw, receipt, package, receipt_path, _, metadata_path = inputs(tmp_path, monkeypatch)
    if mutation in {"duplicate", "symlink"}:
        target = io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as original, ZipFile(target, "w") as archive:
            for info in original.infolist():
                if mutation == "symlink" and info.filename == "make/definition_base.step":
                    info = ZipInfo(info.filename)
                    info.create_system = 3
                    info.external_attr = 0o120777 << 16
                archive.writestr(info, original.read(info.filename))
            if mutation == "duplicate":
                with pytest.warns(UserWarning, match="Duplicate name"):
                    archive.writestr("make/definition_base.step", b"changed")
        raw = target.getvalue()
        receipt.update(bytes=len(raw), sha256=sha(raw))
    elif mutation == "traversal":
        raw, receipt = rewrite(
            raw,
            receipt,
            lambda members: members.update({"../outside.step": b"bad"}),
            rebind_manifest=True,
        )
    elif mutation == "false_hash":
        receipt["sha256"] = "0" * 64
    elif mutation == "member_hash":
        raw, receipt = rewrite(
            raw, receipt, lambda members: members.update({"make/definition_base.step": b"changed"})
        )
    else:
        monkeypatch.setattr(project_members, "MAX_FILE_BYTES", 1)
    package.write_bytes(raw)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    output = tmp_path / "new"
    with pytest.raises(ValueError):
        export(package, receipt_path, metadata_path, output)
    assert not output.exists()


def test_existing_output_and_input_links_are_never_modified(tmp_path, monkeypatch):
    _, _, package, receipt_path, _, metadata_path = inputs(tmp_path, monkeypatch)
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "untouched"
    sentinel.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        export(package, receipt_path, metadata_path, output)
    assert sentinel.read_bytes() == b"keep"
    from scripts import export_showcase

    original = export_showcase.regular_file
    monkeypatch.setattr(
        export_showcase,
        "regular_file",
        lambda path, *args: False if path == package else original(path, *args),
    )
    with pytest.raises(ValueError):
        export(package, receipt_path, metadata_path, tmp_path / "new")


def test_guide_projects_only_recorded_instruction_fields_not_history_or_basis(
    tmp_path, monkeypatch
):
    from scripts.export_showcase import _guide

    data = {
        "status": "current",
        "history": ["PRIVATE_HISTORY_CANARY"],
        "basis": {"private": "PRIVATE_BASIS_CANARY"},
        "record": {
            "guidance": {
                "print": "Use the supplied CAD with care.",
                "installation": "Check fit.",
                "use_and_removal": "Keep access clear.",
                "unresolved": "Physical use remains unverified.",
            }
        },
    }
    guide_text = _guide(json.dumps(data).encode()).decode()
    assert "Check fit." in guide_text
    assert "PRIVATE_HISTORY_CANARY" not in guide_text
    assert "PRIVATE_BASIS_CANARY" not in guide_text
