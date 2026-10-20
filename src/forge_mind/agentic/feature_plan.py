"""Feature-plan-then-author (lever f, ADR-0045): a deterministic FEATURE CHECKLIST the
author must map to its own code BEFORE the build is attempted.

The measured failure this targets: the author silently DROPS a declared feature (a hole
group, a rib, a boss) and the loop only discovers it post-build, after paying for the
sandbox + validator + render. With FORGE_FEATURE_PLAN=1 the author returns, alongside the
script, a plan-to-code MAPPING (one entry per checklist feature naming the code that
builds it); the loop audits that mapping textually -- cheap, pre-build -- and REFUSES to
even sandbox a draft whose mapping misses a declared feature. The refusal feedback names
the exact gaps, so the next attempt arrives complete.

Honesty boundaries:
  - The audit is a PRE-build completeness check on the author's own claim. It never
    replaces the validator: the genus gate / measured-hole audit still measure the real
    geometry post-build (a mapping can lie; the validator cannot).
  - Refusals are BOUNDED (``_REFUSAL_CAP``): a part that keeps failing the mapping audit
    eventually proceeds to the sandbox anyway, where the measured gates judge it -- the
    lever can never deadlock a part on mapping bureaucracy.
  - OPT-IN via FORGE_FEATURE_PLAN (unset = deployed default, byte-identical prompts).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Total plan-refusals allowed per part before the loop stops refusing and lets the
# sandbox + validator judge the draft. Each refusal costs one author call (no sandbox,
# no render, no vision) -- cheap, but bounded so a stubborn mapping never starves the
# real build of iterations.
_REFUSAL_CAP = 2


def enabled() -> bool:
    """Is the feature-plan lever ON? (FORGE_FEATURE_PLAN=1; unset/0 = deployed default)."""
    return os.environ.get("FORGE_FEATURE_PLAN", "").strip().lower() in {"1", "true", "yes"}


def refusal_cap() -> int:
    return _REFUSAL_CAP


@dataclass(frozen=True)
class ChecklistItem:
    """One declared feature the author must account for: name x count x key dims."""

    name: str
    count: int
    dims: str
    standard: str | None


def feature_checklist(brief: Any) -> list[ChecklistItem]:
    """Derive the deterministic checklist from the brief: every feature, its instance
    COUNT, and its key dims verbatim. Pure transcription -- nothing inferred, so the
    checklist can never disagree with the brief the author also reads."""
    items: list[ChecklistItem] = []
    for f in getattr(brief, "features", []) or []:
        name = str(getattr(f, "name", "") or "").strip()
        if not name:
            continue
        items.append(
            ChecklistItem(
                name=name,
                count=max(1, int(getattr(f, "count", 1) or 1)),
                dims=str(getattr(f, "approx_dims_mm", "") or "").strip(),
                standard=(getattr(f, "standard", None) or None),
            )
        )
    return items


def checklist_block(items: list[ChecklistItem]) -> str:
    """The prompt block instructing the author to return a plan-to-code mapping.
    '' when there is nothing to map (so the prompt stays clean)."""
    if not items:
        return ""
    lines = [
        "",
        "",
        "FEATURE CHECKLIST (mandatory plan-to-code mapping):",
        "Your response includes a `feature_map` with EXACTLY one entry per checklist item "
        "below. Each entry's `feature` copies the checklist name VERBATIM; its `where` quotes "
        "the exact identifier or code fragment in YOUR script that builds that feature (the "
        "fragment must appear in the script verbatim). A draft whose mapping misses a declared "
        "feature is refused BEFORE it is built -- the build is not attempted.",
    ]
    lines.extend(
        f"  - {it.name} x{it.count}"
        + (f" | {it.dims}" if it.dims else "")
        + (f" | standard: {it.standard}" if it.standard else "")
        for it in items
    )
    return "\n".join(lines)


def audit_feature_map(
    items: list[ChecklistItem], mapping: list[Any] | None, code: str
) -> list[str]:
    """Audit the author's plan-to-code mapping against the checklist. Returns the list of
    GAPS ([] = complete): a checklist feature absent from the mapping, or a mapped `where`
    fragment that does not appear in the script (a claim about code that is not there).

    Matching is by feature NAME, case-insensitive, whitespace-normalized -- the author is
    told to copy names verbatim, so anything fuzzier would mask a dropped feature."""
    if not items:
        return []

    def norm(s: str) -> str:
        return " ".join(str(s).lower().split())

    mapped: dict[str, str] = {}
    for entry in mapping or []:
        feature = norm(getattr(entry, "feature", "") or "")
        where = str(getattr(entry, "where", "") or "").strip()
        if feature and feature not in mapped:
            mapped[feature] = where

    gaps: list[str] = []
    for it in items:
        claimed = mapped.get(norm(it.name))
        if claimed is None:
            gaps.append(
                f"feature {it.name!r} (x{it.count}) has NO feature_map entry -- map it to the "
                "code that builds it (and build it)."
            )
        elif not claimed:
            gaps.append(
                f"feature {it.name!r} is mapped to an EMPTY `where` -- quote the code fragment "
                "that builds it."
            )
        elif claimed not in code:
            gaps.append(
                f"feature {it.name!r} is mapped to {claimed!r}, but that fragment does not appear "
                "in the script -- the mapping must quote code that is actually there."
            )
    return gaps


def refusal_feedback(gaps: list[str]) -> str:
    """The pre-build refusal message the author revises from (names the exact gaps)."""
    lines = [
        "FEATURE-PLAN REFUSAL (pre-build; the build was NOT attempted): your plan-to-code "
        "mapping is incomplete, so the draft was refused before the sandbox. Fix the mapping "
        "AND the code together -- every checklist feature must be built and mapped:",
    ]
    lines.extend(f"  - {g}" for g in gaps)
    lines.append(
        "Return the COMPLETE script plus a feature_map entry for EVERY checklist item, each "
        "`where` quoting a fragment that appears verbatim in the script."
    )
    return "\n".join(lines)


__all__ = [
    "ChecklistItem",
    "audit_feature_map",
    "checklist_block",
    "enabled",
    "feature_checklist",
    "refusal_cap",
    "refusal_feedback",
]
