"""The measured-refutation override (L72 design, founder-approved 2026-07-22).

Generations 4 and 5 measured the last blocker on the manifold brief: the vision critic
vetoes validator-green, audit-clean parts with fixes the instruments REFUTE (demanding
walls be thickened "to 2.5 mm" on walls measured AT 2.5). Text levers on the critic are
measured out (L72); this is the deterministic answer, and it is deliberately narrow:

  The override fires ONLY when every prioritized fix the critic names is individually
  refuted by a measurement:
    * wall/thickness demands  -> refuted when the wall gates PASSED (validation.passed
      carries the material-floor and area-significance verdicts);
    * stated-dimension demands -> refuted only when the interface audit is ON for this
      run (FORGE_INTERFACE_AUDIT=1) and passed with the validation;
    * ANY other fix (add a feature, redesign, reroute) is NOT refutable -> no override.
  No fixes listed = nothing verifiable = no override (fail-closed both directions).

The critic is not silenced: its critique rides the record verbatim, the acceptance is
marked as override-accepted, and the flywheel row's label_env carries the flag so the
training corpus can filter these ships. Flag: FORGE_CRITIC_REFUTATION_OVERRIDE=1
(default OFF; promotion needs its one-lever arm like every lever).
"""

from __future__ import annotations

import os
import re
from typing import Any

OVERRIDE_MARKER = "[ACCEPTED BY MEASURED REFUTATION: every named fix contradicts a measured value]"

_WALL_DEMAND = re.compile(
    r"thicken|wall[s]?\s+(?:to|of|thickness)|thickness of \d|to at least \d+(?:\.\d+)?\s*mm",
    re.IGNORECASE,
)
_DIM_DEMAND = re.compile(
    r"to match the specified|as specified|specified dimensions?|stated (?:diameter|dimension)",
    re.IGNORECASE,
)
# Gen-6 measured the third refutable class: ADD/CREATE demands for features the presence
# and count audits GATED as present ("add eight M6 clearance holes" on a part whose
# hole-count audit measured all twelve). Only demands NAMING a brief-declared feature
# qualify; flourishes the brief never declared stay unrefutable.
_ADD_DEMAND = re.compile(r"^\s*(?:add|create|include|convert)\b", re.IGNORECASE)


def refutation_override_enabled() -> bool:
    return os.environ.get("FORGE_CRITIC_REFUTATION_OVERRIDE") == "1"


def _interface_audit_on() -> bool:
    return os.environ.get("FORGE_INTERFACE_AUDIT") == "1"


def _presence_audits_on() -> bool:
    return os.environ.get("FORGE_FEATURE_PRESENCE") == "1"


def _named_features(fix: str, brief: Any) -> list[Any]:
    """Declared features specifically named by one critic add-demand.

    A shared generic token is insufficient. For multi-token names the fix must name
    at least two informative tokens and at least half of the name, which keeps
    ``head-flange bolt holes`` distinct from an unrelated ``outlet flange`` while
    still matching normal critic prose.
    """
    from forge_mind.agentic.feature_presence import feature_name_tokens

    lowered = fix.lower()
    matched: list[Any] = []
    for feature in getattr(brief, "features", None) or []:
        tokens = feature_name_tokens(str(getattr(feature, "name", "")))
        hits = sum(token in lowered for token in tokens)
        required = 1 if len(tokens) <= 1 else max(2, (len(tokens) + 1) // 2)
        if tokens and hits >= required:
            matched.append(feature)
    return matched


def _fix_refuted(fix: str, validation: Any, brief: Any) -> bool:
    """One fix -> is it contradicted by measurement? Unknown demand classes never are."""
    if _WALL_DEMAND.search(fix):
        # The wall gates (material floor + area significance) are inside `passed`; a
        # passing validation measured the walls the fix wants thickened.
        return bool(validation.passed)
    if _DIM_DEMAND.search(fix):
        # Dimension conformance is only MEASURED when the interface audit gated this run.
        return bool(validation.passed) and _interface_audit_on()
    if _ADD_DEMAND.search(fix):
        # Every specifically named feature must have participated in a deterministic
        # instrument. A green aggregate plus an enabled presence flag does not prove a
        # slot/pocket/vent exists because those negative features are excluded by design.
        from forge_mind.agentic.feature_presence import feature_was_measured

        named = _named_features(fix, brief)
        return (
            bool(validation.passed)
            and _presence_audits_on()
            and bool(named)
            and all(feature_was_measured(feature) for feature in named)
        )
    return False


def refutation_override(validation: Any, critique: Any, brief: Any = None) -> bool:
    """True only when the flag is on, the part is validator-green, the critic declined,
    and EVERY prioritized fix is individually refuted by a measurement."""
    if not refutation_override_enabled():
        return False
    if not getattr(validation, "passed", False):
        return False
    if critique is None or getattr(critique, "looks_right", False):
        return False
    fixes = [f for f in (getattr(critique, "prioritized_fixes", None) or []) if str(f).strip()]
    if not fixes:
        return False
    return all(_fix_refuted(str(f), validation, brief) for f in fixes)


__all__ = ["OVERRIDE_MARKER", "refutation_override", "refutation_override_enabled"]
