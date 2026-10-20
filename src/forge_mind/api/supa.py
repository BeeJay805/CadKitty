"""Supabase integration for the API: token verification (auth), credit metering, and durable
artifact storage. Everything here is OPTIONAL and env-gated -- with the env vars unset the helpers
report "disabled" and the server runs open / local (dev mode), so nothing breaks without Supabase.

Environment:
  SUPABASE_URL               project URL, e.g. https://abcd.supabase.co
  SUPABASE_ANON_KEY          anon/publishable key -- token verification + per-user credit RPC
  SUPABASE_SERVICE_ROLE_KEY  service-role key -- session store + artifact storage (bypasses RLS)
  FORGE_STORAGE_BUCKET       storage bucket for artifacts (default "parts")
"""

from __future__ import annotations

import contextlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
STORAGE_BUCKET = os.environ.get("FORGE_STORAGE_BUCKET", "parts").strip()


def auth_enabled() -> bool:
    """True when token verification + credit metering can run (URL + anon key present)."""
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def service_enabled() -> bool:
    """True when the service-role features (session store, durable storage) can run."""
    return bool(SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY)


class InsufficientCreditsError(Exception):
    """The user's balance is below the cost of the requested generation."""


@lru_cache(maxsize=1)
def _anon_client() -> Any:
    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


@lru_cache(maxsize=1)
def service_client() -> Any | None:
    """Cached service-role client (bypasses RLS). None when not configured."""
    if not service_enabled():
        return None
    from supabase import create_client

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def verify_token(token: str) -> str | None:
    """Return the user id for a valid Supabase access token, else None. No-op (None) if auth is off."""
    if not auth_enabled():
        return None
    try:
        resp = _anon_client().auth.get_user(token)
    except Exception:
        return None
    user = getattr(resp, "user", None)
    return str(user.id) if user is not None and getattr(user, "id", None) else None


def charge_credit(token: str, amount: int, generation_id: str) -> None:
    """Deduct ``amount`` credits AS the token's user (atomic via the deduct_credits RPC). Raises
    InsufficientCreditsError when the balance is too low. No-op if auth is off.

    A fresh client is used per call: setting the per-user JWT mutates the client, which would race
    across concurrent requests on a shared instance."""
    if not auth_enabled():
        return
    from supabase import create_client

    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    client.postgrest.auth(token)  # run the RPC AS the user so deduct_credits' auth.uid() resolves
    try:
        client.rpc(
            "deduct_credits",
            {"p_amount": amount, "p_reason": "generate", "p_generation_id": generation_id},
        ).execute()
    except Exception as exc:
        if "insufficient_credits" in str(exc).lower():
            raise InsufficientCreditsError() from exc
        raise


def refund_credit(user_id: str, amount: int) -> None:
    """Refund ``amount`` credits to a user via the service role (grant_credits). Best-effort: a failed
    refund must never turn a successful request into an error, so this swallows exceptions."""
    client = service_client()
    if client is None:
        return
    with contextlib.suppress(Exception):
        client.rpc(
            "grant_credits",
            {"p_user_id": user_id, "p_amount": amount, "p_reason": "generate_refund"},
        ).execute()


def _mime(name: str) -> str:
    if name.endswith(".glb"):
        return "model/gltf-binary"
    if name.endswith(".stl"):
        return "model/stl"
    return "application/octet-stream"


def upload_artifacts(job_id: str, files: dict[str, Path]) -> None:
    """Upload {filename: path} to the storage bucket under ``_api/<job>/<name>`` (service role).
    Best-effort durability: a failed upload must not fail the request (the local copy still serves
    within this instance's lifetime)."""
    client = service_client()
    if client is None:
        return
    bucket = client.storage.from_(STORAGE_BUCKET)
    for name, path in files.items():
        if not path.is_file():
            continue
        key = f"_api/{job_id}/{name}"
        with contextlib.suppress(Exception):
            bucket.upload(
                key,
                path.read_bytes(),
                {"content-type": _mime(name), "upsert": "true"},
            )


def download_artifact(job_id: str, name: str) -> bytes | None:
    """Fetch ``_api/<job>/<name>`` from storage (service role). None if missing / storage is off.
    The /files fallback path: serves an artifact when the local copy is gone (restart / other
    instance)."""
    client = service_client()
    if client is None:
        return None
    try:
        data = client.storage.from_(STORAGE_BUCKET).download(f"_api/{job_id}/{name}")
    except Exception:
        return None
    return bytes(data) if data else None


__all__ = [
    "STORAGE_BUCKET",
    "InsufficientCreditsError",
    "auth_enabled",
    "charge_credit",
    "download_artifact",
    "refund_credit",
    "service_client",
    "service_enabled",
    "upload_artifacts",
    "verify_token",
]
