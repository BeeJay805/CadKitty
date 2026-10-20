"""``FORGE_PHYSICS_REPORT`` gates whether the PhysicsReport is attached to a response. Default OFF →
``/generate`` and ``/validate-upload`` are BYTE-IDENTICAL (the physics block is never written). Mirrors
the verified ``invention.lifecycle.invention_enabled`` pattern exactly."""

from __future__ import annotations

import os

_PHYSICS_REPORT_ENV = "FORGE_PHYSICS_REPORT"


def physics_report_enabled() -> bool:
    """True iff ``FORGE_PHYSICS_REPORT=1`` — the report is attached. Default off → byte-identical."""
    return os.environ.get(_PHYSICS_REPORT_ENV, "").strip() == "1"


__all__ = ["physics_report_enabled"]
