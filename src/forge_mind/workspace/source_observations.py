"""Owned immutable PDF page derivatives, separate from source-event identity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge_mind.workspace.processing import regular_file
from forge_mind.workspace.product_sources import _access, _digest, _read
from forge_mind.workspace.source_capacity import source_byte_limit
from forge_mind.workspace.store import Store, WorkspaceError

SCOPE = "Untrusted source page observation; text and pixels do not confirm physical dimensions or applicability."


class SourceObservationError(WorkspaceError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _invalid(message: str = "Source observation integrity check failed") -> SourceObservationError:
    return SourceObservationError("integrity_error", message)


def _json(raw: bytes | str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise _invalid()
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise _invalid()

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _directory(root: Path, sid: str, kind: str, identity: str) -> Path:
    if not all(re.fullmatch(r"[a-f0-9]{32}", value) for value in (sid, identity)):
        raise SourceObservationError("invalid_request", "Invalid source observation identity")
    path = root / "sessions" / sid / kind / identity
    for component in (path, *path.parents):
        if component == root:
            break
        if component.is_symlink() or component.resolve() != component.absolute():
            raise _invalid("Source observation directory is linked or redirected")
    return path


def _bytes(path: Path, maximum: int) -> bytes:
    if not regular_file(path, maximum):
        raise _invalid("Source evidence is missing, linked, nonregular, or oversized")
    try:
        before = path.lstat()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise _invalid()
            result = stream.read(maximum + 1)
    except OSError as exc:
        raise _invalid("Source evidence could not be read") from exc
    if len(result) != before.st_size or len(result) > maximum:
        raise _invalid()
    return result


def _write(path: Path, raw: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def retain_pdf(store: Store, sid: str, artifact_id: str, raw: bytes) -> Path:
    directory = _directory(store.root, sid, "source-artifacts", artifact_id)
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "source.pdf"
    try:
        _write(directory / "capture.tmp", raw)
        os.replace(directory / "capture.tmp", target)
    except BaseException:
        (directory / "capture.tmp").unlink(missing_ok=True)
        directory.rmdir()
        raise
    return target


def _table(store: Store) -> None:
    with store.connect() as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS source_observations (session TEXT NOT NULL, id TEXT NOT NULL, document TEXT NOT NULL, digest TEXT NOT NULL, PRIMARY KEY(session,id), FOREIGN KEY(session) REFERENCES sessions(id))"
        )


def _source(
    root: Path, con: Any, sid: str, artifact_id: str, *, source_limit_bytes: int | None = None
) -> tuple[dict[str, Any], bytes | None]:
    if not con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='product_source_events'"
    ).fetchone():
        raise SourceObservationError("source_not_found", "Captured PDF source not found")
    state = _read(con, sid, verify_observations=False)
    source = next((item for item in state["artifacts"] if item["id"] == artifact_id), None)
    if (
        source is None
        or source.get("kind") != "public_fetch"
        or source.get("mime_type") != "application/pdf"
    ):
        raise SourceObservationError("source_not_found", "Captured PDF source not found")
    binding = source.get("raw_pdf")
    if binding is None:
        raise SourceObservationError(
            "source_bytes_unavailable",
            "This legacy PDF source has no retained bytes; capture it again to inspect pages",
        )
    if (
        not isinstance(binding, dict)
        or set(binding) != {"name", "sha256", "bytes"}
        or binding["name"] != "source.pdf"
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or not isinstance(binding["sha256"], str)
        or not re.fullmatch(r"[a-f0-9]{64}", binding["sha256"])
        or binding["sha256"] != source["content_hash"]
    ):
        raise _invalid()
    size = binding["bytes"]
    if source_limit_bytes is not None:
        if type(source_limit_bytes) is not int or source_limit_bytes <= 0:
            raise SourceObservationError("invalid_request", "Source byte limit must be positive")
        if size > source_limit_bytes:
            raise SourceObservationError(
                "source_limit",
                f"Captured source exceeds current byte limit ({source_limit_bytes} bytes)",
            )
    path = _directory(root, sid, "source-artifacts", artifact_id) / "source.pdf"
    if not regular_file(path, size):
        raise _invalid("Retained PDF is missing, linked, nonregular, or changed")
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if source_limit_bytes is not None else None
    total = 0
    try:
        before = path.lstat()
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            if before.st_size != size or (before.st_dev, before.st_ino, before.st_size) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
            ):
                raise _invalid()
            while total <= size:
                chunk = stream.read(min(65536, size + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                digest.update(chunk)
                if chunks is not None:
                    chunks.append(chunk)
    except OSError as exc:
        raise _invalid("Retained PDF could not be read") from exc
    if total != size or digest.hexdigest() != binding["sha256"]:
        raise _invalid("Retained PDF differs from its captured source")
    return source, b"".join(chunks) if chunks is not None else None


def _metadata(
    metadata: dict[str, Any],
    source: dict[str, Any],
    text: str,
    image: bytes,
    page: int,
    scale: float,
) -> None:
    expected = {
        "schema_version": 1,
        "mode": "source_page",
        "source_sha256": source["content_hash"],
        "source_bytes": source["raw_pdf"]["bytes"],
        "page_number": page,
        "scale": scale,
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "text_bytes": len(text.encode()),
        "image_sha256": hashlib.sha256(image).hexdigest(),
        "image_bytes": len(image),
    }
    allowed = set(expected) | {
        "page_count",
        "page_bounds",
        "page_size",
        "rotation",
        "crop",
        "bitmap_width",
        "bitmap_height",
        "backend",
        "worker_sha256",
        "text_truncated",
        "text_available",
        "text_method",
        "scope",
    }
    if set(metadata) not in (allowed, allowed | {"source_limit_bytes"}) or any(
        metadata.get(key) != value for key, value in expected.items()
    ):
        raise _invalid("PDF page output does not match its request or bytes")
    if "source_limit_bytes" in metadata and (
        type(metadata["source_limit_bytes"]) is not int
        or metadata["source_limit_bytes"] < expected["source_bytes"]
    ):
        raise _invalid("PDF page source policy is invalid")
    if (
        type(metadata["page_count"]) is not int
        or not 1 <= page <= metadata["page_count"]
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise _invalid()
    if type(page) is not int or any(
        type(metadata[key]) is not int
        for key in (
            "schema_version",
            "source_bytes",
            "page_number",
            "text_bytes",
            "image_bytes",
            "bitmap_width",
            "bitmap_height",
            "rotation",
        )
    ):
        raise _invalid()
    if (
        not all(0 < metadata[key] <= 8192 for key in ("bitmap_width", "bitmap_height"))
        or metadata["bitmap_width"] * metadata["bitmap_height"] > 16_000_000
    ):
        raise _invalid()
    if (
        len(text) > 120000
        or len(image) > 16 * 1024 * 1024
        or type(metadata["text_truncated"]) is not bool
        or type(metadata["text_available"]) is not bool
        or metadata["text_available"] != bool(text)
    ):
        raise _invalid()
    for key, count in (("page_bounds", 4), ("page_size", 2), ("crop", 4)):
        if (
            not isinstance(metadata[key], list)
            or len(metadata[key]) != count
            or not all(
                type(value) in (int, float) and math.isfinite(value) for value in metadata[key]
            )
        ):
            raise _invalid()
    if (
        metadata["crop"] != [0, 0, 0, 0]
        or metadata["rotation"] not in (0, 90, 180, 270)
        or metadata["text_method"] != "pdfium_get_text_range_ucs2_no_ocr"
        or not re.fullmatch(r"[a-f0-9]{64}", metadata["worker_sha256"])
    ):
        raise _invalid()
    backend = metadata["backend"]
    if (
        not isinstance(backend, dict)
        or set(backend) != {"pypdfium2", "pdfium", "pillow", "flags"}
        or not all(
            isinstance(backend[key], str) and 0 < len(backend[key]) <= 1000
            for key in ("pypdfium2", "pdfium", "pillow")
        )
        or not isinstance(backend["flags"], list)
        or not all(isinstance(flag, str) and len(flag) <= 1000 for flag in backend["flags"])
    ):
        raise _invalid()
    if not isinstance(metadata["scope"], str) or len(metadata["scope"]) > 2000:
        raise _invalid()
    _digest(metadata)  # Reject nonfinite or unserializable metadata.


def _read_at_root(
    root: Path, con: Any, sid: str, owner: str, observation_id: str
) -> tuple[dict[str, Any], bytes]:
    _access(con, sid, owner)
    directory = _directory(root, sid, "source-observations", observation_id)
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_observations'"
    ).fetchone()
    row = (
        con.execute(
            "SELECT document,digest FROM source_observations WHERE session=? AND id=?",
            (sid, observation_id),
        ).fetchone()
        if exists
        else None
    )
    if row is None:
        raise SourceObservationError("observation_not_found", "Source page observation not found")
    try:
        manifest = _json(row[0])
        if (
            _digest(manifest) != row[1]
            or _json(_bytes(directory / "manifest.json", 32768)) != manifest
        ):
            raise _invalid()
        if (
            set(manifest)
            != {
                "schema_version",
                "id",
                "source_artifact",
                "source_sha256",
                "source_bytes",
                "page_number",
                "scale",
                "metadata",
                "artifacts",
                "mode",
                "scope",
            }
            or manifest["id"] != observation_id
            or manifest["mode"] != "source_page"
            or manifest["schema_version"] != 1
            or manifest["scope"] != SCOPE
        ):
            raise _invalid()
        source, _ = _source(root, con, sid, manifest["source_artifact"])
        text_raw = _bytes(directory / "page.txt", 480000)
        image = _bytes(directory / "page.png", 16 * 1024 * 1024)
        text = text_raw.decode("utf-8")
        _metadata(
            manifest["metadata"], source, text, image, manifest["page_number"], manifest["scale"]
        )
        if (
            manifest["source_sha256"] != source["content_hash"]
            or manifest["source_bytes"] != source["raw_pdf"]["bytes"]
            or manifest["artifacts"]
            != {
                "image": {
                    "name": "page.png",
                    "sha256": hashlib.sha256(image).hexdigest(),
                    "bytes": len(image),
                },
                "text": {
                    "name": "page.txt",
                    "sha256": hashlib.sha256(text_raw).hexdigest(),
                    "bytes": len(text_raw),
                },
            }
        ):
            raise _invalid()
        return {**manifest, "text": text, "observation_digest": row[1]}, image
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise _invalid() from exc


def _read_observation(
    store: Store, con: Any, sid: str, owner: str, observation_id: str
) -> dict[str, Any]:
    return _read_at_root(store.root, con, sid, owner, observation_id)[0]


def validate_source_claims(con: Any, sid: str, state: dict[str, Any]) -> None:
    linked = [claim for claim in state["claims"] if claim.get("source_observation_id")]
    if not linked:
        return
    root = Path(
        next(row[2] for row in con.execute("PRAGMA database_list") if row[1] == "main")
    ).parent.resolve()
    owner = con.execute("SELECT owner FROM sessions WHERE id=?", (sid,)).fetchone()[0]
    verified: dict[str, dict[str, Any]] = {}
    for claim in linked:
        observation_id = claim["source_observation_id"]
        if observation_id not in verified:
            verified[observation_id] = _read_at_root(root, con, sid, owner, observation_id)[0]
        observation = verified[observation_id]
        if (
            observation["source_artifact"] != claim["source_artifact"]
            or observation["observation_digest"] != claim.get("source_observation_digest")
            or claim["exact_quote"] not in observation["text"]
        ):
            raise _invalid("Source observation claim binding changed")


def read_source_observation(
    store: Store, sid: str, owner: str, observation_id: str
) -> dict[str, Any]:
    with store.connect() as con:
        return _read_observation(store, con, sid, owner, observation_id)


def observation_image(store: Store, sid: str, owner: str, observation_id: str) -> bytes:
    with store.connect() as con:
        return _read_at_root(store.root, con, sid, owner, observation_id)[1]


def observe_source_page(
    service: Any,
    sid: str,
    owner: str,
    artifact_id: str,
    page_number: int,
    *,
    scale: float = 2.0,
    timeout: float = 30,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    from forge_mind.workspace.pdf_render import PdfObservationError, render_pdf_page

    if (
        type(page_number) is not int
        or page_number < 1
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise SourceObservationError("invalid_request", "Page number and scale must be positive")
    store = service.store
    limit = source_byte_limit()
    with store.connect() as con:
        _access(con, sid, owner)
        source, raw = _source(store.root, con, sid, artifact_id, source_limit_bytes=limit)
    assert raw is not None
    try:
        metadata, text, image = render_pdf_page(
            raw,
            page_number,
            scale=scale,
            timeout=timeout,
            cancelled=cancelled,
            source_limit_bytes=limit,
        )
    except PdfObservationError as exc:
        raise SourceObservationError(exc.code, str(exc)) from exc
    try:
        _metadata(metadata, source, text, image, page_number, scale)
        if metadata.get("source_limit_bytes") != limit:
            raise _invalid("PDF page policy differs from its requested allowance")
    except (ValueError, TypeError, KeyError) as exc:
        raise _invalid("PDF page output metadata is invalid") from exc
    identity = uuid.uuid4().hex
    manifest = {
        "schema_version": 1,
        "id": identity,
        "source_artifact": artifact_id,
        "source_sha256": source["content_hash"],
        "source_bytes": len(raw),
        "page_number": page_number,
        "scale": scale,
        "metadata": metadata,
        "artifacts": {
            "image": {"name": "page.png", "sha256": metadata["image_sha256"], "bytes": len(image)},
            "text": {
                "name": "page.txt",
                "sha256": metadata["text_sha256"],
                "bytes": len(text.encode()),
            },
        },
        "mode": "source_page",
        "scope": SCOPE,
    }
    _table(store)
    directory = _directory(store.root, sid, "source-observations", identity)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        _write(directory / "page.txt", text.encode())
        _write(directory / "page.png", image)
        _write(directory / "manifest.json", json.dumps(manifest, allow_nan=False).encode())
        with store.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            _access(con, sid, owner)
            _source(store.root, con, sid, artifact_id)  # Recheck after worker execution.
            if cancelled is not None and cancelled():
                raise SourceObservationError("cancelled", "Source page observation cancelled")
            con.execute(
                "INSERT INTO source_observations VALUES (?,?,?,?)",
                (sid, identity, json.dumps(manifest, allow_nan=False), _digest(manifest)),
            )
            result = _read_observation(store, con, sid, owner, identity)
        return result
    except BaseException:
        # An uncertain commit keeps its files. Never remove registered evidence.
        with store.connect() as con:
            registered = con.execute(
                "SELECT 1 FROM source_observations WHERE session=? AND id=?", (sid, identity)
            ).fetchone()
        if registered is None:
            for name in ("page.txt", "page.png", "manifest.json"):
                (directory / name).unlink(missing_ok=True)
            directory.rmdir()
        raise
