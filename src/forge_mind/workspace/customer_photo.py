"""Owned ChatGPT photo input over existing atomic customer project admission."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, field_validator

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.customer_photo_authority import PROTOCOL, photo_digest
from forge_mind.workspace.customer_photo_fetch import (
    MAX_BYTES,
    PhotoFetchError,
    approved_hosts,
    target,
)
from forge_mind.workspace.customer_projects import (
    CustomerProjectRequest,
    customer_capabilities,
    hosted_customer_limits,
)
from forge_mind.workspace.customer_references import CustomerReference, KnownDimension
from forge_mind.workspace.models import Record
from forge_mind.workspace.store import WorkspaceError

HOSTS_ENV = "FORGE_CUSTOMER_PHOTO_DOWNLOAD_HOSTS"
FETCH_SECONDS = 10


class ChatGPTPhotoFile(Record):
    """Official top-level file input. File ID is provenance, never authentication."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, hide_input_in_errors=True)

    # Validate URL content in the fetch boundary, whose errors omit input text.
    # Pydantic length errors can otherwise echo a confidential signed URL.
    download_url: str
    file_id: str = Field(min_length=1, max_length=1000)
    mime_type: str = Field(default="", max_length=120)
    file_name: str = Field(default="", max_length=255)

    @field_validator("file_name")
    @classmethod
    def plain_filename(cls, value: str) -> str:
        if value and (
            value in {".", ".."} or any(c in value for c in ("/", "\\", "\x00", "\r", "\n"))
        ):
            raise ValueError("File name must be a plain filename")
        return value


def photo_capability() -> dict[str, Any]:
    try:
        configured = bool(approved_hosts(os.environ.get(HOSTS_ENV, "")))
    except PhotoFetchError:
        configured = False
    return {
        "configured": configured,
        "status": "configured_unqualified" if configured else "unavailable",
        "maximum_bytes": MAX_BYTES,
        "maximum_seconds": FETCH_SECONDS,
        "actual_chatgpt_file_delivery": "unqualified",
        "scope": "Private operator-approved CDN hosts only. Configuration does not establish signed ChatGPT file compatibility or parser security.",
    }


def fetch_photo(url: str) -> bytes:
    hosts = approved_hosts(os.environ.get(HOSTS_ENV, ""))
    target(url, hosts)  # Refuse before spawning; no URL appears in errors or argv.
    deadline = time.monotonic() + FETCH_SECONDS
    environment = {
        key: os.environ[key] for key in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP") if key in os.environ
    }
    process = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).with_name("customer_photo_fetch.py"))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=environment,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired("photo_fetch", FETCH_SECONDS)
        output, _ = process.communicate(
            input=json.dumps({"url": url, "allowed_hosts": ",".join(hosts)}).encode(),
            timeout=remaining,
        )
        if process.returncode != 0 or len(output) > ((MAX_BYTES + 2) // 3) * 4 + 256:
            raise PhotoFetchError("photo_fetch_refused")
        result = json.loads(output)
        if result.get("status") != "ok":
            raise PhotoFetchError("photo_fetch_refused")
        raw = base64.b64decode(result["content_base64"], validate=True)
        if not raw or len(raw) > MAX_BYTES:
            raise PhotoFetchError("photo_fetch_refused")
        return raw
    except subprocess.TimeoutExpired:
        raise PhotoFetchError("photo_fetch_deadline") from None
    except (ValueError, KeyError, TypeError):
        raise PhotoFetchError("photo_fetch_refused") from None
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=2)


def actual_media_type(raw: bytes) -> Literal["image/png", "image/jpeg", "image/webp"]:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    raise PhotoFetchError("photo_type_refused")


def create_from_photo(
    service: WorkspaceService,
    principal: str,
    *,
    request_id: str,
    prompt: str,
    file: ChatGPTPhotoFile,
    known_dimension: KnownDimension | None,
    approval_required: bool,
    printable: bool,
) -> dict[str, Any]:
    if not customer_capabilities(service.executor)["enabled"]:
        raise WorkspaceError(
            "Customer delivery requires a qualified worker and explicit host budgets"
        )
    # Validate stable ordinary input before network access. URL is intentionally absent.
    request = CustomerProjectRequest(
        request_id=request_id,
        prompt=prompt,
        approval_required=approval_required,
        printable=printable,
    )
    stable = photo_digest(
        {
            "prompt": prompt,
            "approval_required": approval_required,
            "printable": printable,
            "known_dimension": known_dimension.model_dump(mode="json") if known_dimension else None,
        }
    )
    file_id = hashlib.sha256(file.file_id.encode()).hexdigest()
    try:
        existing = service.store.customer_request(principal, request_id)
    except WorkspaceError:
        existing = None
    if existing is not None:
        identity = existing.get("customer_photo_identity") or {}
        if (
            identity.get("protocol") != PROTOCOL
            or identity.get("file_id_sha256") != file_id
            or identity.get("stable_input_sha256") != stable
        ):
            raise WorkspaceError("Request identity already belongs to a different photo request")
        reference = existing.get("customer_reference") or {}
        if reference.get("sha256") != identity.get("original_image_sha256"):
            raise WorkspaceError("Saved photo identity differs from its original source")
        # Shared reader verifies both original/normalized retained bytes and ownership.
        from forge_mind.workspace.customer_references import reference_image

        reference_image(service.store, existing["id"], principal)
        if file.mime_type and file.mime_type != reference.get("media_type"):
            raise PhotoFetchError("photo_type_refused")
        return existing
    if file.mime_type and file.mime_type not in {
        "image/png",
        "image/jpeg",
        "image/webp",
    }:
        raise PhotoFetchError("photo_type_refused")
    raw = fetch_photo(file.download_url)
    media_type = actual_media_type(raw)
    if file.mime_type and media_type != file.mime_type:
        raise PhotoFetchError("photo_type_refused")
    extension = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[media_type]
    request = request.model_copy(
        update={
            "reference": CustomerReference(
                filename=file.file_name or "reference." + extension,
                media_type=media_type,
                content_base64=base64.b64encode(raw).decode(),
                known_dimension=known_dimension,
            )
        }
    )
    identity = {
        "protocol": PROTOCOL,
        "file_id_sha256": file_id,
        "stable_input_sha256": stable,
        "original_image_sha256": hashlib.sha256(raw).hexdigest(),
    }
    result = service.store.create_customer_project(
        principal,
        request.model_dump(mode="json"),
        hosted_limits=hosted_customer_limits(),
        server_metadata=identity,
    )
    service.start()
    return result
