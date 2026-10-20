"""The accept-side consistency gate (the L78 desk_grommet incident; ADR-0122 filed it).

The measured incident: the critic scored a solid disc 4 (ship) while its OWN prioritized
fixes demanded brief-declared features by name (outer ring, cover flap, hinge slot) and its
resemblance read 'weak'. The accept predicate reads only the scalar score, and every named
feature sat outside the instruments' measured vocabulary (all declared volumes unparseable
-> the presence audit's expectation was None -> nothing measured), so nothing contradicted
the critique's missing-feature findings. Score said ship; findings said missing; the
approval was self-contradictory and a solid disc shipped as a grommet.

This is the symmetric twin of ``critic_refutation`` (ADR-0121): the override discards a
self-contradictory VETO (every fix refuted by a measurement); this gate refuses a
self-contradictory APPROVAL (a fix the instruments could NOT refute stands as a live
missing-declared-feature claim at accept time).

Deliberately narrow (the exact incident mechanism, BOTH conjuncts required):
  BLOCK a score>=4 accept iff
  (1) the accepting critique carries an ADD/CREATE/INCLUDE/CONVERT demand naming a
      brief-declared feature that no instrument MEASURED this run. A named feature
      counts as measured when
        * it is a through-hole class feature ("hole"/"bore" in its name): the count/
          stack audits fold into ``passed`` via ``features_missing``, always False at
          an accept site; or
        * it is an additive feature whose declared volume PARSED (it participated in
          the presence audit's measured volume expectation).
      Unparseable additive features and blind negatives (slot/pocket/recess/vent) are
      measured by nothing today -> the demand stands; AND
  (2) the same critique rates resemblance at the BOTTOM of the ordinal ('weak'/'none',
      or missing): the critic's own aggregate fidelity signal agrees the part does not
      read as the object.
  Retro-calibration over all 79 archived accepted rows (2026-07-22) demanded conjunct
  (2): with (1) alone, 6 legitimate ships false-FAIL (chamfer-polish add-fixes on
  strong-resemblance accepts x5; the founder-eyeballed gen-7 manifold at partial);
  with both, exactly the desk_grommet false-PASS blocks. A blocked accept keeps
  iterating with the critique's fixes riding the revision as usual.

Aesthetic fixes stay non-triggers (the unrefutable text classes, measured out in L72).
Flag: FORGE_ACCEPT_CONSISTENCY=1 (default OFF; flag off = accept byte-identical;
promotion needs its one-lever arm like every lever).
"""

from __future__ import annotations

import os
from typing import Any

from forge_mind.agentic.critic_refutation import _ADD_DEMAND
from forge_mind.agentic.feature_presence import feature_name_tokens, feature_was_measured

# The bottom of both resemblance ordinals (default and FORGE_CRITIC_ORDINAL). A missing/
# unparseable resemblance also qualifies -- an accepting critique that never claimed
# fidelity cannot contradict its own missing-feature findings.
_BOTTOM_RESEMBLANCE = ("weak", "none", "")


def accept_consistency_enabled() -> bool:
    return os.environ.get("FORGE_ACCEPT_CONSISTENCY") == "1"


def _measured(feature: Any) -> bool:
    """Did ANY instrument measure this declared feature this run? (At an accept site
    ``validation.passed`` already holds, so a measured channel that gates == measured.)"""
    return feature_was_measured(feature)


def accept_consistency_block(critique: Any, brief: Any) -> str | None:
    """Reason string when a score>=4 accept must be REFUSED as self-contradictory;
    None = accept proceeds. Called only at the accept site (validator green)."""
    if not accept_consistency_enabled():
        return None
    if critique is None or brief is None:
        return None
    resemblance = str(getattr(critique, "resemblance", "") or "").strip().lower()
    if resemblance not in _BOTTOM_RESEMBLANCE:
        # Conjunct (2): a strong/partial-resemblance accept's add-demands are polish, not
        # a standing missing-feature claim (calibrated: 6/79 archived ships say so).
        return None
    fixes = [str(f) for f in (getattr(critique, "prioritized_fixes", None) or []) if str(f).strip()]
    if not fixes:
        return None
    unmeasured: list[str] = []
    for feature in getattr(brief, "features", None) or []:
        tokens = feature_name_tokens(str(getattr(feature, "name", "")))
        if not tokens or _measured(feature):
            continue
        for fix in fixes:
            if _ADD_DEMAND.search(fix) and any(t in fix.lower() for t in tokens):
                unmeasured.append(str(getattr(feature, "name", "")))
                break
    if not unmeasured:
        return None
    return (
        "accept refused (self-contradictory approval): the critique demands brief-declared "
        "feature(s) no instrument measured present this run: " + ", ".join(sorted(set(unmeasured)))
    )


__all__ = ["accept_consistency_block", "accept_consistency_enabled"]
