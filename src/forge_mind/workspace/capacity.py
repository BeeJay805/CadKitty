"""Optional operator resource policies; model validity has no fixed size ceiling."""

from __future__ import annotations

import os


def optional_limit(name: str) -> int | None:
    """Read a positive integer policy. Unset/empty means no configured limit."""
    key = "FORGE_WORKSPACE_" + name
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive integer or unset") from exc
    if value <= 0:
        raise ValueError(f"{key} must be a positive integer or unset")
    return value


def enforce_limit(count: int, limit: int | None, resource: str) -> None:
    if limit is not None and count > limit:
        raise ValueError(
            f"Operator resource policy exceeded: {resource} requires {count}, limit is {limit}"
        )


def completion_policy() -> dict[str, dict[str, int | str | None]]:
    """Discover completion admission limits separately from evidence quality."""
    return {
        name.lower(): {
            "value": optional_limit(name),
            "environment": "FORGE_WORKSPACE_" + name,
            "unit": "bytes" if name.endswith("_BYTES") else "count",
        }
        for name in (
            "COMPLETION_SUBSYSTEM_LIMIT",
            "COMPLETION_CHECK_LIMIT",
            "COMPLETION_PLAN_BYTES",
        )
    }
