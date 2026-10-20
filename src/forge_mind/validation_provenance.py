"""Content identity for verdict-producing code and geometry artifacts."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_VALIDATION_SOURCES = (
    "fabrication.py",
    "materials.py",
    "agentic/accept_consistency.py",
    "agentic/critic_refutation.py",
    "agentic/feature_presence.py",
    "agentic/spatial.py",
    "agentic/validate.py",
    "assembly/fits.py",
    "validator/feature_checks.py",
)


def file_sha256(path: Path) -> str:
    """SHA-256 of one persisted artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def validation_fingerprint() -> str:
    """Stable digest of every source file that participates in the cached verdict."""
    digest = hashlib.sha256()
    for relative in _VALIDATION_SOURCES:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\x00")
        try:
            digest.update((_ROOT / relative).read_bytes())
        except OSError:
            digest.update(b"<unavailable>")
        digest.update(b"\x00")
    return digest.hexdigest()


__all__ = ["file_sha256", "validation_fingerprint"]
