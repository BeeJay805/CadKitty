"""Opt-in, short-lived capabilities for exact owned package bytes only."""

from __future__ import annotations

import hashlib
import io
import math
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

from forge_mind.workspace.store import WorkspaceError

MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
PackageReader = Callable[[str, str, str, str], tuple[bytes, dict[str, Any], list[dict[str, Any]]]]


@dataclass(frozen=True)
class DownloadGrant:
    owner: str = field(repr=False)
    kind: str
    project_id: str
    package_id: str
    package_hash: str
    member_id: str | None
    archive_path: str | None
    sha256: str
    size: int
    filename: str
    expires_at: float


def project_package_reader(service: Any) -> PackageReader:
    """Reuse owned package readback and its complete CAD membership checks."""
    from forge_mind.workspace.project_delivery import read_project_export
    from forge_mind.workspace.project_members import captured_cad_files

    def read(
        owner: str, project: str, package: str, checksum: str
    ) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
        content, receipt = read_project_export(service, project, owner, package, checksum)
        return content, receipt, captured_cad_files(content, receipt)

    return read


def print_package_reader(service: Any) -> PackageReader:
    """Use only the immutable print-package reader and its exact STL inventory."""
    from forge_mind.workspace.print_setup_service import read_print_export

    def read(
        owner: str, project: str, package: str, checksum: str
    ) -> tuple[bytes, dict[str, Any], list[dict[str, Any]]]:
        content, receipt = read_print_export(service, project, owner, package, checksum)
        members = [
            {
                "file_id": "print-" + item["part_id"],
                "archive_path": name,
                "sha256": item["sha256"],
                "bytes": item["bytes"],
            }
            for name, item in receipt["files"].items()
        ]
        return content, receipt, members

    return read


class ConnectorDownloads:
    """Process-local grants expire on restart; URLs are confidential bearer capabilities."""

    def __init__(
        self,
        *,
        origin: str,
        readers: dict[str, PackageReader],
        ttl_seconds: int = 300,
        capacity: int = 1024,
        clock: Callable[[], float] = time.time,
    ) -> None:
        parsed = urlsplit(origin)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or '"' in origin
        ):
            raise ValueError("Downloads require one configured plain HTTPS origin")
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 900:
            raise ValueError("Download lifetime must be between 1 and 900 seconds")
        if type(capacity) is not int or not 1 <= capacity <= 4096:
            raise ValueError("Download grant capacity must be between 1 and 4096")
        if not readers or any(not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", kind) for kind in readers):
            raise ValueError("Require explicitly registered package readers")
        self.origin = f"{parsed.scheme}://{parsed.netloc}"
        self.readers = dict(readers)
        self.ttl_seconds, self.capacity, self.clock = ttl_seconds, capacity, clock
        self._grants: dict[str, DownloadGrant] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _key(token: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", token):
            raise WorkspaceError("Download is unavailable")
        return hashlib.sha256(token.encode("ascii")).hexdigest()

    def _read_package(
        self, kind: str, owner: str, project: str, package: str, checksum: str
    ) -> tuple[bytes, list[dict[str, Any]]]:
        if (
            kind not in self.readers
            or not owner
            or not re.fullmatch(r"[a-f0-9]{32}", project)
            or not re.fullmatch(r"[a-f0-9]{32}", package)
            or not re.fullmatch(r"[a-f0-9]{64}", checksum)
        ):
            raise WorkspaceError("Download is unavailable")
        content, receipt, members = self.readers[kind](owner, project, package, checksum)
        if (
            not isinstance(content, bytes)
            or not 0 < len(content) <= MAX_DOWNLOAD_BYTES
            or self._hash(content) != checksum
            or receipt.get("sha256") != checksum
            or receipt.get("project_id") != project
            or receipt.get("package_id") != package
            or receipt.get("bytes") != len(content)
        ):
            raise WorkspaceError("Download source changed")
        return content, members

    def mint(
        self,
        *,
        owner: str,
        project_id: str,
        package_id: str,
        package_hash: str,
        kind: str = "project",
        individual_files: bool = True,
        authorization_expires_at: float,
    ) -> dict[str, Any]:
        """Call only after request authentication; readers recheck owner and membership."""
        if not math.isfinite(authorization_expires_at):
            raise WorkspaceError("Download authorization expired")
        content, members = self._read_package(kind, owner, project_id, package_id, package_hash)
        expiry = min(self.clock() + self.ttl_seconds, authorization_expires_at)
        if expiry <= self.clock():
            raise WorkspaceError("Download authorization expired")
        rows: list[tuple[str | None, str | None, str, int, str]] = [
            (
                None,
                None,
                package_hash,
                len(content),
                "print-files.zip" if kind == "print" else "project.zip",
            )
        ]
        if individual_files:
            seen: set[str] = set()
            with ZipFile(io.BytesIO(content)) as archive:
                if len(archive.namelist()) != len(set(archive.namelist())):
                    raise WorkspaceError("Download package has duplicate members")
                for member in members:
                    identifier, path = member.get("file_id"), member.get("archive_path")
                    if (
                        not isinstance(identifier, str)
                        or not re.fullmatch(r"[A-Za-z0-9_-]{1,160}", identifier)
                        or identifier in seen
                        or not isinstance(path, str)
                        or not re.fullmatch(r"[A-Za-z0-9_./-]{1,240}", path)
                        or any(part in ("", ".", "..") for part in path.split("/"))
                    ):
                        raise WorkspaceError("Download member identity is invalid")
                    seen.add(identifier)
                    data = self._member_bytes(archive, member)
                    filename = path.replace("/", "-")
                    rows.append((identifier, path, self._hash(data), len(data), filename))
        issued: list[tuple[str, DownloadGrant]] = []
        for identifier, path, checksum, size, filename in rows:
            token = secrets.token_urlsafe(32)
            issued.append(
                (
                    token,
                    DownloadGrant(
                        owner,
                        kind,
                        project_id,
                        package_id,
                        package_hash,
                        identifier,
                        path,
                        checksum,
                        size,
                        filename,
                        expiry,
                    ),
                )
            )
        with self._lock:
            self._grants = {
                key: grant for key, grant in self._grants.items() if grant.expires_at > self.clock()
            }
            if len(self._grants) + len(issued) > self.capacity:
                raise WorkspaceError("Download grant capacity reached")
            self._grants.update({self._key(token): grant for token, grant in issued})
        values = [
            {
                "file_id": grant.member_id,
                "archive_path": grant.archive_path,
                "sha256": grant.sha256,
                "bytes": grant.size,
                "expires_at": grant.expires_at,
                "download_url": self.origin + "/downloads/" + token,
            }
            for token, grant in issued
        ]
        return {
            "package": values[0],
            "individual_files": values[1:],
            "scope": "Short-lived bearer links; keep private. Readback rechecks current ownership and exact saved package bytes.",
        }

    @staticmethod
    def _member_bytes(archive: ZipFile, member: dict[str, Any]) -> bytes:
        path = member["archive_path"]
        item = archive.getinfo(path)
        size = member.get("bytes")
        if type(size) is not int or not 0 < size <= MAX_DOWNLOAD_BYTES or item.file_size != size:
            raise WorkspaceError("Download member size changed")
        with archive.open(item) as stream:
            data = stream.read(size + 1)
        if len(data) != size or hashlib.sha256(data).hexdigest() != member.get("sha256"):
            raise WorkspaceError("Download member changed")
        return data

    def read(
        self, token: str, *, authenticated_owner: str | None = None
    ) -> tuple[bytes, DownloadGrant]:
        key = self._key(token)
        with self._lock:
            grant = self._grants.get(key)
        if (
            grant is None
            or grant.expires_at <= self.clock()
            or (authenticated_owner is not None and authenticated_owner != grant.owner)
        ):
            raise WorkspaceError("Download is unavailable")
        content, members = self._read_package(
            grant.kind, grant.owner, grant.project_id, grant.package_id, grant.package_hash
        )
        if grant.member_id is not None:
            matches = [
                member
                for member in members
                if member.get("file_id") == grant.member_id
                and member.get("archive_path") == grant.archive_path
            ]
            if len(matches) != 1:
                raise WorkspaceError("Download membership changed")
            with ZipFile(io.BytesIO(content)) as archive:
                content = self._member_bytes(archive, matches[0])
        if len(content) != grant.size or self._hash(content) != grant.sha256:
            raise WorkspaceError("Download source changed")
        with self._lock:
            if self._grants.get(key) != grant or grant.expires_at <= self.clock():
                raise WorkspaceError("Download is unavailable")
        return content, grant

    def revoke(self, token: str) -> None:
        with self._lock:
            self._grants.pop(self._key(token), None)

    def revoke_owner(self, owner: str) -> None:
        with self._lock:
            self._grants = {
                key: grant for key, grant in self._grants.items() if grant.owner != owner
            }

    def revoke_package(self, owner: str, project: str, package: str) -> None:
        with self._lock:
            self._grants = {
                key: grant
                for key, grant in self._grants.items()
                if (grant.owner, grant.project_id, grant.package_id) != (owner, project, package)
            }


def register_download_route(
    app: Any,
    downloads: ConnectorDownloads,
    *,
    verifier: Any,
    issuer: str,
    resource: str,
    owner_transform: Callable[[str], str] | None = None,
) -> None:
    """Add only explicitly enabled capability reads; OAuth discovery routes stay intact."""
    import asyncio

    from starlette.requests import Request
    from starlette.responses import Response

    from forge_mind.workspace.connector_auth import ConnectorAuthError, require_owner

    headers = {
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
    }

    async def download(request: Request) -> Response:
        try:
            if request.headers.get("host", "").lower() != urlsplit(downloads.origin).netloc.lower():
                raise WorkspaceError("Download is unavailable")
            owner = None
            authorization = request.headers.get("authorization")
            if authorization is not None:
                if not authorization.startswith("Bearer "):
                    raise WorkspaceError("Download is unavailable")
                token = await verifier.verify_token(authorization[7:])
                owner = require_owner(token, issuer=issuer, resource=resource)
                if owner_transform is not None:
                    owner = owner_transform(owner)
            content, grant = await asyncio.to_thread(
                downloads.read, request.path_params["token"], authenticated_owner=owner
            )
        except (ConnectorAuthError, WorkspaceError, ValueError, OSError, KeyError, BadZipFile):
            return Response("Download is unavailable", status_code=404, headers=headers)
        return Response(
            content,
            media_type="application/zip" if grant.member_id is None else "application/octet-stream",
            headers={**headers, "Content-Disposition": f'attachment; filename="{grant.filename}"'},
        )

    app.add_route("/downloads/{token}", download, methods=["GET"])
