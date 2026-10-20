from __future__ import annotations

import base64
import hashlib
import io

import pytest
from PIL import Image

from forge_mind.workspace.customer_references import (
    CustomerReference,
    decode_reference,
    reference_image,
)
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import Store, WorkspaceError


def reference(**changes):
    buffer = io.BytesIO()
    Image.new("RGB", (20, 10), "blue").save(buffer, format="PNG")
    return {
        "filename": "mount.png",
        "media_type": "image/png",
        "content_base64": base64.b64encode(buffer.getvalue()).decode(),
        "known_dimension": {"label": "Hole center distance", "value": 25.4, "unit": "mm"},
        **changes,
    }


def test_customer_photo_retains_original_source_and_declared_scale(tmp_path):
    store = Store(tmp_path)
    image = reference()
    session = store.create_customer_project(
        "owner", {"request_id": "a" * 32, "prompt": "Match this shape", "reference": image}
    )
    sid = session["id"]
    metadata = session["customer_reference"]
    assert (
        metadata["sha256"] == hashlib.sha256(base64.b64decode(image["content_base64"])).hexdigest()
    )
    assert metadata["known_dimension"] == image["known_dimension"]
    assert "content_base64" not in str(session)
    vision, returned = reference_image(store, sid, "owner")
    assert hashlib.sha256(vision).hexdigest() == returned["vision_sha256"]
    source = read_sources(store, sid, "owner")["artifacts"][0]
    assert source["content_hash"] == metadata["sha256"]
    assert source["kind"] == "customer_reference"
    assert "25.4 mm" in session["instructions"][0]
    assert reference_image(store, sid, "owner")[0] == vision
    with pytest.raises(WorkspaceError, match="not found"):
        reference_image(store, sid, "other")


@pytest.mark.parametrize(
    "changes",
    [
        {"content_base64": "invalid!"},
        {"media_type": "image/jpeg"},
        {"content_base64": base64.b64encode(b"not an image").decode()},
    ],
)
def test_invalid_photo_does_not_admit_a_project(tmp_path, changes):
    store = Store(tmp_path)
    with pytest.raises(WorkspaceError):
        store.create_customer_project(
            "owner", {"request_id": "a" * 32, "prompt": "A part", "reference": reference(**changes)}
        )
    assert store.list_sessions("owner") == []


def test_reference_without_scale_keeps_it_unknown_and_detects_changed_bytes(tmp_path):
    store = Store(tmp_path)
    session = store.create_customer_project(
        "owner",
        {"request_id": "a" * 32, "prompt": "A part", "reference": reference(known_dimension=None)},
    )
    assert session["customer_reference"]["known_dimension"] is None
    sid = session["id"]
    path = (
        store.root
        / "sessions"
        / sid
        / "source-artifacts"
        / session["customer_reference"]["source_artifact_id"]
        / "source.original"
    )
    path.write_bytes(b"changed original")
    with pytest.raises(WorkspaceError, match="changed"):
        reference_image(store, sid, "owner")


def test_pixel_limit_is_checked_before_full_decode(monkeypatch):
    monkeypatch.setattr("forge_mind.workspace.customer_references.MAX_REFERENCE_PIXELS", 199)
    with pytest.raises(WorkspaceError, match="pixel limits"):
        decode_reference(CustomerReference.model_validate(reference()))
