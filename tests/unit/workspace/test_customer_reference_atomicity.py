"""Failed reference capture must not retain an unowned original image."""

import base64
import io
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from forge_mind.workspace import customer_references
from forge_mind.workspace.customer_references import CustomerReference, reference_image
from forge_mind.workspace.product_sources import _table
from forge_mind.workspace.store import Store


def request() -> dict:
    image = io.BytesIO()
    Image.new("RGB", (20, 10), "blue").save(image, format="PNG")
    return {
        "request_id": "a" * 32,
        "prompt": "Match this reference",
        "reference": {
            "filename": "reference.png",
            "media_type": "image/png",
            "content_base64": base64.b64encode(image.getvalue()).decode(),
        },
    }


@pytest.mark.parametrize("filename", ["source.original", "vision.png"])
def test_partial_reference_write_rolls_back_files_and_request(tmp_path, monkeypatch, filename):
    store = Store(tmp_path)
    unrelated = tmp_path / "sessions" / "preserved" / "source-artifacts" / "existing"
    unrelated.mkdir(parents=True)
    sentinel = unrelated / "source.original"
    sentinel.write_bytes(b"existing owner image")
    write = Path.write_bytes

    def fail(path, data):
        if path.name == filename:
            write(path, data[:3])
            raise OSError("synthetic full disk")
        return write(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", fail)
        with pytest.raises(OSError, match="synthetic full disk"):
            store.create_customer_project("alice", request())

    assert store.list_sessions("alice") == []
    assert list(tmp_path.rglob("source.original")) == [sentinel]
    assert not list(tmp_path.rglob("vision.png"))
    assert sentinel.read_bytes() == b"existing owner image"
    session = store.create_customer_project("alice", request())
    assert reference_image(store, session["id"], "alice")[0]


def test_source_journal_failure_removes_both_reference_files(tmp_path):
    store = Store(tmp_path)
    _table(store)
    with store.connect() as con:
        con.execute(
            "CREATE TRIGGER reject_reference BEFORE INSERT ON product_source_events "
            "BEGIN SELECT RAISE(ABORT, 'synthetic source journal failure'); END"
        )

    with pytest.raises(sqlite3.IntegrityError, match="synthetic source journal failure"):
        store.create_customer_project("alice", request())

    assert store.list_sessions("alice") == []
    assert not list(tmp_path.rglob("source.original"))
    assert not list(tmp_path.rglob("vision.png"))
    with store.connect() as con:
        for table in ("sessions", "jobs", "customer_requests", "product_source_events"):
            assert con.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_reference_cleanup_failure_preserves_original_failure(tmp_path, monkeypatch):
    store = Store(tmp_path)
    write = Path.write_bytes

    def fail_write(path, data):
        if path.name == "vision.png":
            raise OSError("synthetic original write failure")
        return write(path, data)

    def fail_unlink(path, *args, **kwargs):
        raise PermissionError("synthetic cleanup denial")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(OSError, match="synthetic original write failure") as failure:
        store.create_customer_project("alice", request())

    assert any("cleanup" in note.lower() for note in failure.value.__notes__)
    assert store.list_sessions("alice") == []


def test_admission_cleanup_failure_preserves_original_failure(tmp_path, monkeypatch):
    store = Store(tmp_path)

    def fail_admission(*args, **kwargs):
        raise ValueError("synthetic admission rejection")

    def fail_unlink(path, *args, **kwargs):
        raise PermissionError("synthetic cleanup denial")

    monkeypatch.setattr(store, "_enqueue_with_connection", fail_admission)
    monkeypatch.setattr(Path, "unlink", fail_unlink)
    with pytest.raises(ValueError, match="synthetic admission rejection") as failure:
        store.create_customer_project("alice", request())

    assert any("cleanup" in note.lower() for note in failure.value.__notes__)
    assert store.list_sessions("alice") == []


@pytest.mark.parametrize("symlink", [False, True])
def test_existing_artifact_directory_is_never_cleaned(tmp_path, monkeypatch, symlink):
    store = Store(tmp_path)
    sid = store.create("alice", "plate", None)["id"]
    aid = "b" * 32
    artifact = tmp_path / "sessions" / sid / "source-artifacts" / aid
    artifact.parent.mkdir(parents=True)
    target = tmp_path / "existing"
    target.mkdir()
    if symlink:
        try:
            artifact.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"Directory symlinks unavailable: {exc}")
    else:
        artifact.mkdir()
        target = artifact
    sentinel = target / "source.original"
    sentinel.write_bytes(b"preserved original")
    monkeypatch.setattr(
        customer_references, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=aid))
    )
    reference = CustomerReference.model_validate(request()["reference"])

    with store.connect() as con, pytest.raises(FileExistsError):
        customer_references.retain_reference(store, con, sid, reference, (b"raw", b"vision"))

    assert sentinel.read_bytes() == b"preserved original"
    assert artifact.is_symlink() == symlink


def test_cleanup_skips_replaced_directory(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    sentinel = artifact / "source.original"
    sentinel.write_bytes(b"preserved replacement")
    resolve = Path.resolve

    def replaced(path, *args, **kwargs):
        return tmp_path / "replacement" if path == artifact else resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", replaced)
    failure = OSError("synthetic write failure")
    customer_references._rollback_reference(artifact, failure)

    assert sentinel.read_bytes() == b"preserved replacement"
    assert "redirected" in failure.__notes__[0]


def test_cleanup_resolution_error_preserves_original_failure(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    sentinel = artifact / "source.original"
    sentinel.write_bytes(b"preserved unresolved directory")

    def fail_resolve(path, *args, **kwargs):
        raise RuntimeError("synthetic symlink loop")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    failure = OSError("synthetic write failure")
    customer_references._rollback_reference(artifact, failure)

    assert sentinel.read_bytes() == b"preserved unresolved directory"
    assert "synthetic symlink loop" in failure.__notes__[0]
