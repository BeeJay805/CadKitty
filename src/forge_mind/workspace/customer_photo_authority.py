"""Server-owned photo request identity; signed download URLs are never identity."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

PROTOCOL = "customer_photo_v1"
IDENTITY_KEYS = {"protocol", "file_id_sha256", "stable_input_sha256", "original_image_sha256"}


def photo_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_photo_identity(
    value: dict[str, str] | None, raw: bytes | None
) -> dict[str, str] | None:
    if value is None:
        return None
    if set(value) != IDENTITY_KEYS or value.get("protocol") != PROTOCOL:
        raise ValueError("Invalid server-owned photo identity")
    if any(
        not isinstance(value[key], str) or re.fullmatch(r"[a-f0-9]{64}", value[key]) is None
        for key in IDENTITY_KEYS - {"protocol"}
    ):
        raise ValueError("Invalid server-owned photo digest")
    if raw is None or hashlib.sha256(raw).hexdigest() != value["original_image_sha256"]:
        raise ValueError("Server-owned photo identity differs from original image bytes")
    return dict(value)
