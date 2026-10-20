"""Operator admission policy for new public-source transfers and PDF page work."""

from forge_mind.workspace.capacity import optional_limit

DEFAULT_SOURCE_BYTES = 2 * 1024 * 1024


def source_byte_limit() -> int:
    """Unset or empty retains the bounded public default; overrides must be positive."""
    configured = optional_limit("SOURCE_BYTES")
    return DEFAULT_SOURCE_BYTES if configured is None else configured


def source_policy() -> dict[str, int | str]:
    return {
        "value": source_byte_limit(),
        "default": DEFAULT_SOURCE_BYTES,
        "environment": "FORGE_WORKSPACE_SOURCE_BYTES",
        "unit": "bytes",
        "scope": "Admission for new source transfers and PDF page processing; independent worker memory, time, pixel and output limits still apply. Historical evidence verification is separate.",
    }
