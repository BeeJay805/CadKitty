"""Off-the-shelf STEP part sourcing (approach copied from text-to-cad's step-parts skill, MIT).

For a FUNCTIONAL assembly a real bolt / bearing / motor should be a verified catalog STEP, not a
hand-modeled stand-in. This searches the hosted step.parts catalog, downloads the canonical STEP,
and VERIFIES it against the record's SHA-256 before trusting it; verified bytes are content-addressed
in a local cache so the same part is fetched once. The discipline on a MISS is explicit: record the
miss and let the caller fall back to a documented envelope placeholder -- never silently substitute.

step.parts is a third-party hosted API (api.step.parts) Forge-Mind does not control, so the HTTP
fetcher is INJECTABLE: this is fully testable offline and the provider can be swapped or cached.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_ORIGIN = "https://api.step.parts"
_USER_AGENT = "forge-mind-sourcing/1.0"

# (url, timeout_s) -> raw bytes. The seam that makes sourcing testable + provider-swappable.
Fetcher = Callable[[str, float], bytes]


class SourcingError(RuntimeError):
    """A part could not be sourced or failed checksum verification."""


@dataclass(frozen=True)
class PartRecord:
    """One catalog hit. ``step_url`` + ``sha256`` are what make a download trustworthy."""

    id: str
    name: str
    step_url: str | None
    sha256: str | None
    category: str | None = None
    family: str | None = None
    standard: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> PartRecord:
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            step_url=d.get("stepUrl"),
            sha256=d.get("sha256"),
            category=d.get("category"),
            family=d.get("family"),
            standard=d.get("standard"),
            attributes=d.get("attributes") or {},
        )


@dataclass(frozen=True)
class SourcedPart:
    """A downloaded, checksum-checked STEP on disk."""

    record: PartRecord
    path: Path
    sha256: str
    checksum_verified: bool


def _http_fetch(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def _slug(text: str) -> str:
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in text.strip().lower()]
    return ("".join(keep)[:64] or "part").strip("_") or "part"


class StepPartsClient:
    """Search + download verified STEP parts from a step.parts-shaped catalog."""

    def __init__(
        self,
        *,
        origin: str = DEFAULT_ORIGIN,
        fetch: Fetcher | None = None,
        cache_dir: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._origin = origin.rstrip("/")
        self._fetch = fetch or _http_fetch
        self._cache_dir = cache_dir
        self._timeout = timeout

    def _json(self, path: str, params: list[tuple[str, str]] | None = None) -> Any:
        url = f"{self._origin}{path}"
        if params:
            url = f"{url}?{urlencode(params)}"
        return json.loads(self._fetch(url, self._timeout).decode("utf-8"))

    def search(
        self,
        query: str | None = None,
        *,
        limit: int = 10,
        tags: Iterable[str] = (),
        category: Iterable[str] = (),
        family: Iterable[str] = (),
        standard: Iterable[str] = (),
    ) -> list[PartRecord]:
        """Search the catalog. Returns [] on no match (a recorded miss for the caller)."""
        params: list[tuple[str, str]] = [("limit", str(limit))]
        if query:
            params.append(("q", query))
        for name, values in (
            ("tag", tags),
            ("category", category),
            ("family", family),
            ("standard", standard),
        ):
            params.extend((name, v) for v in values)
        data = self._json("/v1/parts", params)
        return [PartRecord.from_json(p) for p in data.get("items", [])]

    def get(self, part_id: str) -> PartRecord:
        return PartRecord.from_json(self._json(f"/v1/parts/{part_id}"))

    def _cache_path(self, sha256: str) -> Path | None:
        if self._cache_dir is None or not sha256:
            return None
        return self._cache_dir / sha256[:2] / f"{sha256}.step"

    def download(
        self, record: PartRecord, dest_dir: Path, *, overwrite: bool = False
    ) -> SourcedPart:
        """Download ``record``'s STEP into ``dest_dir`` and verify its SHA-256.

        Raises SourcingError if the record has no STEP url or the checksum does not match. Verified
        bytes are content-addressed in the cache, so a re-source of the same part skips the network.
        """
        if not record.step_url:
            raise SourcingError(f"part {record.id!r} has no STEP url")
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{_slug(record.id)}.step"
        if dest.exists() and not overwrite:
            data = dest.read_bytes()
        else:
            cache = self._cache_path(record.sha256 or "")
            if cache is not None and cache.exists():
                data = cache.read_bytes()  # cache hit: no network
            else:
                url = record.step_url
                if url.startswith("/"):
                    url = f"{self._origin}{url}"
                data = self._fetch(url, self._timeout)
            dest.write_bytes(data)
        actual = hashlib.sha256(data).hexdigest()
        if record.sha256 is not None and record.sha256 != actual:
            raise SourcingError(
                f"checksum mismatch for {record.id!r}: expected {record.sha256}, got {actual}"
            )
        cache = self._cache_path(actual)
        if cache is not None and not cache.exists():
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(data)
        return SourcedPart(
            record=record, path=dest, sha256=actual, checksum_verified=record.sha256 is not None
        )


def source_or_envelope(
    query: str, dest_dir: Path, *, client: StepPartsClient | None = None
) -> SourcedPart | None:
    """Try to source a real verified part for ``query``; return None on ANY miss or error.

    None is the signal to the caller to model a documented ENVELOPE placeholder instead -- the
    no-silent-substitution discipline. Never raises: a catalog outage degrades to "model it yourself".
    """
    client = client or StepPartsClient()
    try:
        hits = client.search(query, limit=1)
        if not hits:
            return None
        return client.download(hits[0], dest_dir)
    except Exception:
        return None


__all__ = [
    "DEFAULT_ORIGIN",
    "Fetcher",
    "PartRecord",
    "SourcedPart",
    "SourcingError",
    "StepPartsClient",
    "source_or_envelope",
]
