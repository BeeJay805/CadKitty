"""Capacity integration controls: controlled transport, real owned API and PDF worker.

Scratch runs use pytest -o pythonpath=<explicit source overlay> and set
FORGE_CAPACITY_TEST_SOURCE_ROOT to that directory. Native cases require
FORGE_PDF_NATIVE_TEST_PYTHON. No network/model calls.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Header
from fastapi.testclient import TestClient

from forge_mind.workspace import pdf_render, product_source_api, source_page_models
from forge_mind.workspace import product_sources as sources
from forge_mind.workspace import source_observations as observations
from forge_mind.workspace.store import Store, WorkspaceError

DEFAULT_LIMIT = 2 * 1024 * 1024
PRIVATE_LIMIT = 8 * 1024 * 1024
SOURCE_URL = "https://catalog.example.test/fixture.pdf"


@pytest.fixture(autouse=True)
def explicit_source_origin():
    configured = os.environ.get("FORGE_CAPACITY_TEST_SOURCE_ROOT")
    if not configured:
        return
    root = Path(configured).resolve()
    for module in (sources, observations, pdf_render, product_source_api, source_page_models):
        assert (
            Path(module.__file__).resolve()
            == (
                root / "forge_mind" / "workspace" / f"{module.__name__.rsplit('.', 1)[1]}.py"
            ).resolve()
        ), f"Wrong source tree imported: {module.__file__}"


def padded_pdf(padding_size: int = DEFAULT_LIMIT + 128) -> bytes:
    """Valid single-page PDF with an unused padding stream, not trailing fake bytes."""
    content = b"BT /F1 14 Tf 20 100 Td (Variant A thickness 27 mm) Tj ET"
    padding = b"x" * padding_size
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [4 0 R] >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 200] /Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream",
        f"<< /Length {len(padding)} >>\nstream\n".encode() + padding + b"\nendstream",
    ]
    raw = bytearray(b"%PDF-1.7\n")
    offsets = []
    for index, obj in enumerate(objects, 1):
        offsets.append(len(raw))
        raw.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(raw)
    raw.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        raw.extend(f"{offset:010} 00000 n \n".encode())
    raw.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(raw)


@pytest.fixture
def captured(tmp_path, monkeypatch, request):
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", str(PRIVATE_LIMIT))
    configured = os.environ.get("FORGE_PDF_NATIVE_TEST_PYTHON")
    if not configured:
        pytest.skip("Set FORGE_PDF_NATIVE_TEST_PYTHON for native source-capacity controls")
    monkeypatch.setenv("FORGE_PDF_PYTHON", configured)
    store = Store(tmp_path)
    sid = store.create("alice", "Large catalog capacity control", None)["id"]
    service = SimpleNamespace(store=store)
    raw = padded_pdf(getattr(request, "param", DEFAULT_LIMIT + 128))
    calls = []

    def retrieve(url, timeout, cancelled, max_bytes):
        calls.append((url, max_bytes))
        assert url == SOURCE_URL and not cancelled()
        assert len(raw) <= max_bytes
        return raw, "application/pdf", url

    monkeypatch.setattr(sources, "retrieve", retrieve)

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(product_source_api.source_router(store, service, owner))
    with TestClient(app) as client:
        base = f"/workspace/sessions/{sid}"
        response = client.post(
            f"{base}/sources/fetch", headers={"Authorization": "alice"}, json={"url": SOURCE_URL}
        )
        assert response.status_code == 200, response.text
        artifact = response.json()["artifacts"][-1]
        path = store.root / "sessions" / sid / "source-artifacts" / artifact["id"] / "source.pdf"
        yield SimpleNamespace(
            client=client,
            store=store,
            sid=sid,
            service=service,
            base=base,
            artifact=artifact,
            raw=raw,
            calls=calls,
            path=path,
            headers={"Authorization": "alice"},
        )


def observe(captured):
    response = captured.client.post(
        f"{captured.base}/sources/{captured.artifact['id']}/pages",
        headers=captured.headers,
        json={"page_number": 1},
    )
    assert response.status_code == 200, response.text
    return response.json()


def propose(captured, receipt):
    response = captured.client.post(
        f"{captured.base}/source-claims",
        headers=captured.headers,
        json={
            "source_artifact": captured.artifact["id"],
            "source_observation_id": receipt["id"],
            "quantity": 27,
            "unit": "mm",
            "exact_quote": "thickness 27 mm",
            "configuration": "Variant A",
            "datum": "front to rear",
            "locator": "PDF page 1",
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_large_native_observation_and_claim_preserve_exact_capture(captured):
    item = captured
    assert DEFAULT_LIMIT < len(item.raw) < PRIVATE_LIMIT
    assert item.calls == [(SOURCE_URL, PRIVATE_LIMIT)]
    assert item.artifact["capture_policy"]["source_limit_bytes"] == PRIVATE_LIMIT
    assert item.artifact["raw_pdf"]["bytes"] == len(item.raw)
    assert item.artifact["content_hash"] == hashlib.sha256(item.raw).hexdigest()
    assert item.path.read_bytes() == item.raw
    before = sources.read_sources(item.store, item.sid, "alice")
    receipt = observe(item)
    assert receipt["source_sha256"] == item.artifact["content_hash"]
    assert receipt["source_bytes"] == len(item.raw)
    assert receipt["metadata"]["source_limit_bytes"] == PRIVATE_LIMIT
    assert receipt["metadata"]["backend"]["pypdfium2"] == "5.13.0"
    assert "thickness 27 mm" in receipt["text"]
    assert sources.read_sources(item.store, item.sid, "alice") == before
    resource = f"{item.base}/source-observations/{receipt['id']}"
    assert item.client.get(resource, headers=item.headers).json() == receipt
    image = item.client.get(resource + "/image.png", headers=item.headers)
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG\r\n\x1a\n")
    assert hashlib.sha256(image.content).hexdigest() == receipt["artifacts"]["image"]["sha256"]
    claim = propose(item, receipt)["claims"][-1]
    assert claim["confirmation"] is None
    assert claim["source_observation_digest"] == receipt["observation_digest"]


def test_lower_policy_preserves_confirmed_history_but_blocks_new_worker(captured, monkeypatch):
    item = captured
    receipt = observe(item)
    state = propose(item, receipt)
    response = item.client.post(
        f"{item.base}/source-claims/{state['claims'][-1]['id']}/confirm",
        headers=item.headers,
        json={"version": state["version"]},
    )
    assert response.status_code == 200, response.text
    confirmed = response.json()
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", str(DEFAULT_LIMIT))

    def no_worker(*args, **kwargs):
        pytest.fail("Oversized retained PDF reached the renderer after policy lowering")

    monkeypatch.setattr(pdf_render, "render_pdf_page", no_worker)
    actual_open = Path.open

    class StreamingOnly:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def read(self, size=-1):
            assert 0 < size <= DEFAULT_LIMIT, "Historical verification allocated the entire source"
            return self.stream.read(size)

    def bounded_open(path, *args, **kwargs):
        stream = actual_open(path, *args, **kwargs)
        return StreamingOnly(stream) if path == item.path else stream

    monkeypatch.setattr(Path, "open", bounded_open)
    assert sources.read_sources(item.store, item.sid, "alice") == confirmed
    assert (
        observations.read_source_observation(item.store, item.sid, "alice", receipt["id"])
        == receipt
    )
    assert (
        item.client.get(
            f"{item.base}/source-observations/{receipt['id']}/image.png", headers=item.headers
        ).status_code
        == 200
    )
    response = item.client.post(
        f"{item.base}/sources/{item.artifact['id']}/pages",
        headers=item.headers,
        json={"page_number": 1},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "source_limit"


@pytest.mark.parametrize("change_size", [False, True])
def test_large_retained_tamper_remains_invalid_after_lowering(captured, monkeypatch, change_size):
    item = captured
    receipt = observe(item)
    propose(item, receipt)
    raw = item.raw + b"changed" if change_size else item.raw[:-1] + bytes([item.raw[-1] ^ 1])
    item.path.write_bytes(raw)
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", str(DEFAULT_LIMIT))
    with pytest.raises(observations.SourceObservationError) as failed_observation:
        observations.read_source_observation(item.store, item.sid, "alice", receipt["id"])
    assert failed_observation.value.code == "integrity_error"
    with pytest.raises(observations.SourceObservationError) as failed_claim:
        sources.read_sources(item.store, item.sid, "alice")
    assert failed_claim.value.code == "integrity_error"
    response = item.client.get(
        f"{item.base}/source-observations/{receipt['id']}/image.png", headers=item.headers
    )
    # Existing owner preflight verifies linked source claims through store.get;
    # its workspace access wrapper maps that earlier integrity rejection to 404.
    assert response.status_code == 404
    assert response.json() == {"detail": "workspace not found"}


@pytest.mark.parametrize("field", ["source_limit_bytes", "max_bytes", "capture_policy"])
def test_request_cannot_override_operator_policy(captured, monkeypatch, field):
    item = captured
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", str(DEFAULT_LIMIT))
    monkeypatch.setattr(
        pdf_render, "render_pdf_page", lambda *a, **k: pytest.fail("Worker invoked")
    )
    for path, body in (
        (f"{item.base}/sources/fetch", {"url": SOURCE_URL, field: PRIVATE_LIMIT}),
        (
            f"{item.base}/sources/{item.artifact['id']}/pages",
            {"page_number": 1, field: PRIVATE_LIMIT},
        ),
    ):
        response = item.client.post(path, headers=item.headers, json=body)
        assert response.status_code == 422, response.text
    assert item.calls == [(SOURCE_URL, PRIVATE_LIMIT)]


def test_large_source_owner_is_checked_before_processing(captured, monkeypatch):
    item = captured
    receipt = observe(item)
    monkeypatch.setattr(
        pdf_render, "render_pdf_page", lambda *a, **k: pytest.fail("Worker invoked")
    )
    headers = {"Authorization": "bob"}
    assert (
        item.client.post(
            f"{item.base}/sources/{item.artifact['id']}/pages",
            headers=headers,
            json={"page_number": 1},
        ).status_code
        == 404
    )
    resource = f"{item.base}/source-observations/{receipt['id']}"
    for path in (resource, resource + "/image.png"):
        assert item.client.get(path, headers=headers).status_code == 404
    with pytest.raises(WorkspaceError):
        observations.observe_source_page(item.service, item.sid, "bob", item.artifact["id"], 1)


def test_capture_records_admission_snapshot_if_configuration_changes(captured, monkeypatch):
    item = captured

    def changing_retrieve(url, timeout, cancelled, max_bytes):
        assert max_bytes == PRIVATE_LIMIT
        monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", str(DEFAULT_LIMIT))
        return item.raw, "application/pdf", url

    monkeypatch.setattr(sources, "retrieve", changing_retrieve)
    state = sources.fetch_source(item.service, item.sid, "alice", SOURCE_URL)
    saved = state["artifacts"][-1]
    assert saved["capture_policy"]["source_limit_bytes"] == PRIVATE_LIMIT
    assert saved["raw_pdf"]["bytes"] == len(item.raw)


@pytest.mark.parametrize("captured", [128], indirect=True)
def test_declared_legacy_receipt_without_policy_remains_usable(captured, monkeypatch):
    item = captured
    receipt = observe(item)
    assert len(item.raw) < DEFAULT_LIMIT  # This fixture could exist under the old 2 MiB cap.
    directory = item.store.root / "sessions" / item.sid / "source-observations" / receipt["id"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["metadata"]["source_limit_bytes"]
    serialized = json.dumps(manifest, allow_nan=False)
    legacy_digest = sources._digest(manifest)
    # Explicit fixture construction only: simulate the previous immutable format,
    # not a production migration or an allowed way to edit registered evidence.
    manifest_path.write_text(serialized)
    with item.store.connect() as con:
        con.execute(
            "UPDATE source_observations SET document=?,digest=? WHERE session=? AND id=?",
            (serialized, legacy_digest, item.sid, receipt["id"]),
        )
    monkeypatch.setenv("FORGE_WORKSPACE_SOURCE_BYTES", "512")
    assert len(item.raw) > 512
    historical = observations.read_source_observation(item.store, item.sid, "alice", receipt["id"])
    assert "source_limit_bytes" not in historical["metadata"]
    assert historical["observation_digest"] == legacy_digest
    state = propose(item, historical)
    claim = state["claims"][-1]
    assert claim["confirmation"] is None
    assert claim["source_observation_digest"] == legacy_digest
    confirmed = sources.confirm_claim(item.store, item.sid, "alice", claim["id"], state["version"])
    assert sources.read_sources(item.store, item.sid, "alice") == confirmed
    assert observations.observation_image(item.store, item.sid, "alice", receipt["id"])
    assert manifest_path.read_text() == serialized  # Reads never retrofit policy into old evidence.
