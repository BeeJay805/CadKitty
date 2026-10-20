"""Grow-don't-reject envelope policy (the #1 askability fix).

A deterministic interpreter derives a part's real footprint from its interfaces/op. The
``envelope_mm`` it is handed is an LLM ESTIMATE, not a user limit -- and the LLM systematically
UNDER-sizes it, so a perfectly buildable part was being rejected (``LayoutError`` -> silent
direct-DSL fallback). A derived footprint is a COMPUTED CONSTRAINT, so the right move is to GROW
the envelope to the derived minimum, not reject.

The ONE exception is a user-stated HARD maximum (``intent.max_envelope_mm``, plumbed onto the
spec): that is a real limit. If the part cannot fit it, we do NOT grow -- we raise a clean
"cannot fit within the requested max envelope" (an honest refusal, not a silent fallback).

This never weakens a check: the grown part still goes through every validation tier against its
grown bounding box. Growth only ever RAISES an axis, so a part whose envelope already fits its
footprint is byte-identical (no growth triggered).
"""

from __future__ import annotations

from forge_mind.layout.interpreter import InfeasibleRequestError
from forge_mind.types import Vec3

_EPS_MM = 1e-6  # float-noise tolerance for the hard-max comparison


def grow_envelope(requested: Vec3, required: Vec3, hard_max: Vec3 | None = None) -> Vec3:
    """Return an envelope that fits ``required``, growing ``requested`` per-axis where it is too
    small. With a ``hard_max``: refuse (LayoutError) if ``required`` exceeds it on any axis;
    otherwise the result is clamped into ``[required, hard_max]`` per axis (honour a user max even
    if the LLM over-picked the working envelope)."""
    if hard_max is not None:
        over = [
            f"{ax} needs {req:.2f}mm > max {mx:.2f}mm"
            for ax, req, mx in (
                ("X", required.x, hard_max.x),
                ("Y", required.y, hard_max.y),
                ("Z", required.z, hard_max.z),
            )
            if req > mx + _EPS_MM
        ]
        if over:
            raise InfeasibleRequestError(
                "cannot fit within the requested max envelope: " + "; ".join(over)
            )

    def _axis(required_axis: float, requested_axis: float, max_axis: float | None) -> float:
        grown = max(required_axis, requested_axis)
        return grown if max_axis is None else min(grown, max_axis)

    mx = hard_max
    return Vec3(
        x=_axis(required.x, requested.x, None if mx is None else mx.x),
        y=_axis(required.y, requested.y, None if mx is None else mx.y),
        z=_axis(required.z, requested.z, None if mx is None else mx.z),
    )


__all__ = ["grow_envelope"]
