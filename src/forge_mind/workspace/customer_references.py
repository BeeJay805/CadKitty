"""Owned original photo sources and separately hashed visual inputs."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image, ImageOps
from pydantic import Field, field_validator

from forge_mind.workspace.models import Record
from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    import sqlite3

    from forge_mind.workspace.store import Store

MAX_REFERENCE_BYTES = 4 * 1024 * 1024
MAX_REFERENCE_PIXELS = 16000000
REFERENCE_MEDIA_TYPES = ["image/png", "image/jpeg", "image/webp"]
SCOPE = "Owner-supplied visual reference and declared measurement; pixels do not establish exact dimensions or physical fit."


class KnownDimension(Record):
    label: str = Field(min_length=1, max_length=200)
    value: float = Field(gt=0, allow_inf_nan=False)
    unit: Literal["mm", "cm", "in"]


class CustomerReference(Record):
    filename: str = Field(min_length=1, max_length=255)
    media_type: Literal["image/png", "image/jpeg", "image/webp"]
    content_base64: str = Field(min_length=1, max_length=(MAX_REFERENCE_BYTES + 2) // 3 * 4)
    known_dimension: KnownDimension | None = None

    @field_validator("filename")
    @classmethod
    def basename(cls, value: str) -> str:
        if value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00", "\r", "\n")):
            raise ValueError("Reference filename must be a plain filename")
        return value


def decode_reference(reference: CustomerReference) -> tuple[bytes, bytes]:
    try:
        raw = base64.b64decode(reference.content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise WorkspaceError("Reference must contain valid base64 image bytes") from exc
    if not raw or len(raw) > MAX_REFERENCE_BYTES:
        raise WorkspaceError("Reference exceeds its original image byte limit")
    try:
        with Image.open(io.BytesIO(raw)) as image:
            expected = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}
            if image.format != expected[reference.media_type]:
                raise WorkspaceError("Reference image bytes differ from the declared media type")
            if (
                image.width * image.height > MAX_REFERENCE_PIXELS
                or getattr(image, "n_frames", 1) != 1
            ):
                raise WorkspaceError("Reference image exceeds pixel limits or is animated")
            image.load()
            normalized_image = ImageOps.exif_transpose(image)
            normalized_image.thumbnail((2048, 2048))
            normalized = io.BytesIO()
            normalized_image.convert("RGB").save(normalized, format="PNG")
    except WorkspaceError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise WorkspaceError("Reference image could not be decoded within image limits") from exc
    return raw, normalized.getvalue()


def _rollback_reference(directory: Path, failure: Exception) -> None:
    """Remove only this failed capture's paths while preserving its original error."""
    try:
        if directory.is_symlink() or directory.resolve() != directory:
            failure.add_note("Reference cleanup skipped a redirected artifact directory")
            return
    except (OSError, RuntimeError) as exc:
        failure.add_note(f"Reference cleanup could not verify its artifact directory: {exc}")
        return
    for name in ("source.original", "vision.png"):
        try:
            (directory / name).unlink(missing_ok=True)
        except OSError as exc:
            failure.add_note(f"Reference cleanup could not remove {name}: {exc}")
    try:
        directory.rmdir()
    except OSError as exc:
        failure.add_note(f"Reference cleanup could not remove its artifact directory: {exc}")


def retain_reference(
    store: Store,
    con: sqlite3.Connection,
    sid: str,
    reference: CustomerReference,
    content: tuple[bytes, bytes],
) -> tuple[dict[str, Any], Path]:
    """Called before job admission while the owned project transaction is held."""
    from forge_mind.workspace.product_sources import _digest

    raw, vision = content
    aid = uuid.uuid4().hex
    directory = store.root / "sessions" / sid / "source-artifacts" / aid
    metadata = {
        "source_artifact_id": aid,
        "filename": reference.filename,
        "media_type": reference.media_type,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "vision_sha256": hashlib.sha256(vision).hexdigest(),
        "known_dimension": reference.known_dimension.model_dump(mode="json")
        if reference.known_dimension
        else None,
        "scope": SCOPE,
    }
    artifact = {
        "id": aid,
        "kind": "customer_reference",
        "version": 1,
        "content_hash": metadata["sha256"],
        **metadata,
    }
    event = {"kind": "artifact", "data": artifact}
    directory.mkdir(parents=True, exist_ok=False)
    try:
        (directory / "source.original").write_bytes(raw)
        (directory / "vision.png").write_bytes(vision)
        con.execute(
            "INSERT INTO product_source_events VALUES (?,?,?,?)",
            (sid, 1, json.dumps(event), _digest(event)),
        )
    except Exception as exc:
        _rollback_reference(directory, exc)
        raise
    return metadata, directory


def reference_image(store: Store, sid: str, owner: str) -> tuple[bytes, dict[str, Any]]:
    from forge_mind.workspace.product_sources import read_sources

    session = store.session_record(sid, owner)
    metadata = session.get("customer_reference")
    if not isinstance(metadata, dict):
        raise WorkspaceError("Customer reference not found")
    artifact = next(
        (
            item
            for item in read_sources(store, sid, owner)["artifacts"]
            if item["id"] == metadata["source_artifact_id"]
        ),
        None,
    )
    if (
        artifact is None
        or artifact.get("content_hash") != metadata["sha256"]
        or any(artifact.get(key) != value for key, value in metadata.items())
    ):
        raise WorkspaceError("Customer reference differs from its source record")
    directory = store.root / "sessions" / sid / "source-artifacts" / metadata["source_artifact_id"]
    original, normalized = directory / "source.original", directory / "vision.png"
    if not regular_file(original, MAX_REFERENCE_BYTES) or not regular_file(
        normalized, MAX_REFERENCE_PIXELS * 4
    ):
        raise WorkspaceError("Customer reference files are unavailable")
    raw, vision = original.read_bytes(), normalized.read_bytes()
    if (
        len(raw) != metadata["bytes"]
        or hashlib.sha256(raw).hexdigest() != metadata["sha256"]
        or hashlib.sha256(vision).hexdigest() != metadata["vision_sha256"]
    ):
        raise WorkspaceError("Customer reference bytes changed")
    return vision, metadata
