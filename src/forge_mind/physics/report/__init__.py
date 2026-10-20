"""Physics-validation report — the FROZEN CONTRACT v1 projection (ADR-0063).

A READ-ONLY projection that turns the validator's already-finished report into the ``PhysicsReport`` the
frontend renders. It never runs a check or gates anything (the validator is the sole verdict authority,
ADR-0056); the report verdict is a pure relabel of the validator's verdict. Attached to a response only
when ``FORGE_PHYSICS_REPORT=1`` (default off → byte-identical responses).
"""

from __future__ import annotations

from forge_mind.physics.report.compose import compose_physics_report
from forge_mind.physics.report.flag import physics_report_enabled
from forge_mind.physics.report.schema import Intent, PhysicsReport

__all__ = [
    "Intent",
    "PhysicsReport",
    "compose_physics_report",
    "physics_report_enabled",
]
