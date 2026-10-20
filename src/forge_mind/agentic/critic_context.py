"""The critique's measured-context block (ADR-0119/0120 lineage, extracted for testability).

What the deterministic instruments verified, formatted for the vision critic. The L71
addition rides here: the brief's STATED dimensions travel with the block, so spec-true
slender geometry reads as CONFORMANT rather than fragile. Generation 4 measured the need:
the critic held approval at score 3 demanding thicker walls on tubes MEASURED at their
stated 2.5 mm -- and obeying it broke validation twice (i5/i6) while the loop rightly
kept the green i4. The critic must know the spec it is grading against.

Flag: FORGE_CRITIC_MEASURED=1 (unchanged; None when off = byte-identical critique).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge_mind.agentic.brief import DesignBrief


def critic_measured_block(validation: Any, sandbox: Any, brief: DesignBrief | None) -> str | None:
    """The measured-context block for ``critique_part(measured=...)``, or None when off."""
    if os.environ.get("FORGE_CRITIC_MEASURED") != "1":
        return None
    m = [
        "MEASURED BY THE VALIDATOR (deterministic B-rep instruments -- "
        "trust these over what the render seems to show):"
    ]
    extents = getattr(sandbox, "extents", None)
    if extents:
        m.append(f"- built envelope: {extents[0]:.0f} x {extents[1]:.0f} x {extents[2]:.0f} mm")
    m.append(
        f"- volume {validation.volume_mm3:.0f} mm^3; bodies "
        f"{validation.body_count}; watertight {validation.watertight}"
    )
    if validation.issues and not validation.passed:
        m.append("- OPEN validator complaints (the real deficits -- anchor your fixes HERE):")
        m.extend(f"    * {i[:160]}" for i in validation.issues[:5])
    elif validation.issues:
        # ADR-0120: on a PASSED validation the leftover lines are sub-threshold residuals --
        # feeding them as "OPEN complaints" burned iteration 5 of the L68 run.
        m.append(
            "- validator: PASSED. Lines below are SUB-THRESHOLD residuals "
            "(advisory, not blocking) -- do NOT chase them; judge "
            "proportions and spec fidelity of what is measured present:"
        )
        m.extend(f"    * {i[:160]}" for i in validation.issues[:3])
    else:
        m.append("- validator complaints: NONE (declared features measured present)")
    # L71: the stated dims ARE the grading rubric. Without them the critic taxes correct
    # slender geometry by eye ("thicken the 2.5mm walls") and steers revisions into red.
    feats = list(getattr(brief, "features", []) or []) if brief is not None else []
    stated = [
        (f.name, str(getattr(f, "approx_dims_mm", "") or "").strip())
        for f in feats
        if str(getattr(f, "approx_dims_mm", "") or "").strip()
    ][:8]
    if stated:
        m.append(
            "THE SPEC (the user's stated dimensions -- geometry MEASURED at these values "
            "is CORRECT even where it looks slender or delicate in the render; never "
            "request thickening or resizing beyond the spec):"
        )
        m.extend(f"    * {name}: {dims[:80]}" for name, dims in stated)
    m.append(
        "Do NOT re-request features absent from the complaint list; judge the "
        "proportions and fidelity of what is measured present."
    )
    return "\n".join(m)


__all__ = ["critic_measured_block"]
