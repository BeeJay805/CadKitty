"""Canonical owner identity shared by authenticated website and assistant entrypoints."""

from __future__ import annotations

import hashlib
import json

from forge_mind.workspace.hosting import HostingError, https_origin


def canonical_principal(issuer: str, subject: str) -> str:
    """Call only after signature/session validation; email is never an identity key."""
    suffix = "/auth/v1"
    if not issuer.endswith(suffix):
        raise HostingError("A validated canonical Supabase issuer is required")
    issuer = https_origin(issuer[: -len(suffix)]) + suffix
    if (
        not isinstance(subject, str)
        or not subject
        or len(subject) > 512
        or subject.strip() != subject
    ):
        raise HostingError("A validated nonempty subject is required")
    value = json.dumps([issuer, subject], separators=(",", ":"), ensure_ascii=True)
    return "principal:v1:" + hashlib.sha256(value.encode()).hexdigest()
