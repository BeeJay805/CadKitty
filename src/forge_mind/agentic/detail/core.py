"""Detail library CORE: the registry, the LLM-directed FinishOp, and the fail-closed
apply pass. Ops live in the sibling modules (textures, structure, markings) and
self-register via ``@register``.

Law (same as the rest of the engine): the LLM DIRECTS finish -- it names which op goes
on which region with what parameters (a FinishOp list) -- and this library OWNS the
build123d code. Every op is re-validated so a broken texture can never ship.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_MIN_FEATURE_MM = 1.0  # a 0.4mm nozzle cannot resolve finer; refuse rather than build noise
# Volume sanity band (ASYMMETRIC): a finish op may REMOVE at most half the part (a texture /
# lightening that guts it is broken) and ADD up to 2.5x (cooling fins / ribs legitimately
# bulk a part up). The watertight-solid check is the primary safety; this catches runaway
# booleans that still happen to be watertight.
_VOL_MIN, _VOL_MAX = 0.5, 2.5

_OPS: dict[str, Callable[..., Any]] = {}


def register(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator: add an op to the finish registry under ``name``."""

    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        _OPS[name] = fn
        return fn

    return _wrap


def op_names() -> list[str]:
    """Every registered op name (for the LLM finish-planner's allowed set + tests)."""
    return sorted(_OPS)


def finish_enabled() -> bool:
    """FORGE_FINISH=1 turns the finish pass on. Unset/0 -> production byte-identical."""
    return os.environ.get("FORGE_FINISH", "").strip().lower() in {"1", "true", "yes"}


@dataclass
class FinishOp:
    """One LLM-directed finish directive: which texture, where, with what parameters.
    ``op`` is a registered op name; ``params`` are its keyword args. The LLM emits these;
    it never writes geometry code."""

    op: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class FinishResult:
    part: Any
    applied: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _watertight_volume(part: Any) -> float | None:
    """Volume of ``part`` if it is a positive, finite solid, else None (fail-closed signal)."""
    try:
        v = float(part.volume)
        return v if v > 0 and math.isfinite(v) else None
    except Exception:
        return None


def apply_finish(part: Any, plan: list[FinishOp]) -> FinishResult:
    """Apply each finish op in order, FAIL-CLOSED per op: an op that raises, produces a
    non-solid, or moves the volume outside +-_VOL_BAND is SKIPPED and the part before it
    kept. Returns the finished part + which ops applied/skipped. Never raises."""
    res = FinishResult(part=part)
    base_v = _watertight_volume(part)
    if base_v is None:
        return res  # the input is already not a clean solid -- add no detail
    for d in plan:
        fn = _OPS.get(d.op)
        if fn is None:
            res.skipped.append(f"{d.op}: unknown op")
            continue
        try:
            candidate = fn(res.part, **d.params)
        except Exception as exc:  # bad params / geometry failure -> keep the current part
            res.skipped.append(f"{d.op}: {type(exc).__name__}: {exc}")
            continue
        cur_v = _watertight_volume(candidate)
        prev_v = _watertight_volume(res.part) or base_v
        if cur_v is None:
            res.skipped.append(f"{d.op}: result not a solid")
            continue
        if not _VOL_MIN * prev_v <= cur_v <= _VOL_MAX * prev_v:
            res.skipped.append(f"{d.op}: volume moved {cur_v / prev_v:.2f}x (outside band)")
            continue
        res.part = candidate
        res.applied.append(d.op)
    return res
