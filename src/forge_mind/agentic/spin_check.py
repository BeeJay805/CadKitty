"""Balance advisory for the agentic validator -- the first wiring of the functional-mechanism
spine into generation. For a ROTATING-POWER part it measures balance about the shaft bore and,
when the part is unbalanced, tells the author to fix it (add counterweights / symmetric mass).

Gated, fail-closed, advisory:
  - GATED to the ``rotating_power`` class (crankshaft, flywheel, impeller, turbine...) via the
    existing deterministic classify_part, so a bored bracket -- which does not spin -- gets no
    balance noise.
  - FAILS CLOSED: no shaft bore (a journal-shaft rotor, or no hole), no B-rep, or an
    unverifiable mesh -> no advisory. The honest "not balanced" service disclaimer still stands;
    we never invent an axis (see validator.balance on why a mass-inferred axis is a false PASS).
  - ADVISORY ONLY: balance-for-service is a fabrication claim, never a geometry gate -- this
    feeds the author's revision loop but does not flip the verdict.
"""

from __future__ import annotations

from typing import Any

from forge_mind.agentic.brief import DesignBrief
from forge_mind.agentic.confidence import classify_part
from forge_mind.validator.balance import rotational_balance
from forge_mind.validator.geometry import main_bore_axis


def balance_feedback(brief: DesignBrief, part: Any, mesh: Any) -> list[str]:
    """One advisory line when a rotating-power part is UNBALANCED about its shaft bore; empty
    otherwise (wrong class, no bore, balanced, or unverifiable -- all fail-closed to no claim)."""
    if classify_part(brief.object_name, brief.summary) != "rotating_power":
        return []
    if part is None:
        return []
    axis = main_bore_axis(part)
    if axis is None:
        return []
    point, direction = axis
    report = rotational_balance(mesh, point, direction)
    if report.balanced is not False:  # balanced (True) or unverifiable (None) -> no advisory
        return []
    return [
        "BALANCE (advisory): this rotating part is UNBALANCED about its shaft bore "
        f"(static {report.static_ratio:.3f}, dynamic {report.dynamic_ratio:.3f}) -- a spinning "
        "part must be mass-symmetric about its axis. Add counterweights or symmetric material so "
        "the center of mass sits on the bore axis."
    ]


__all__ = ["balance_feedback"]
