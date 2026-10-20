"""Pure private scheduling policy; reserved opportunities do not guarantee success.

Allocation can become configurable later. This policy does not extend supplied
budgets, select CAD work, qualify evidence, or decide whether delivery is ready.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_PHASES = ("construction", "handoff", "instructions")


def _phase_rank(phase: str) -> int:
    try:
        return _PHASES.index(phase)
    except ValueError:
        raise ValueError(f"Unknown handoff phase: {phase!r}") from None


def _nonnegative(value: float, label: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


@dataclass(frozen=True)
class HandoffBudget:
    """Keep construction, handoff, and instruction stages inside author budgets."""

    author_deadline: float | None
    author_seconds: float | None
    action_limit: int | None

    def __post_init__(self) -> None:
        if self.author_deadline is not None and not math.isfinite(self.author_deadline):
            raise ValueError("author_deadline must be finite or None")
        if self.author_seconds is not None:
            _nonnegative(self.author_seconds, "author_seconds")
        if self.action_limit is not None and self.action_limit < 0:
            raise ValueError("action_limit must be nonnegative or None")

    def deadline(self, phase: str) -> float | None:
        """Return a phase boundary without creating an absent wall-clock cutoff."""
        rank = _phase_rank(phase)
        if self.author_deadline is None:
            return None
        handoff = 0.0
        if self.author_seconds is not None:
            handoff = min(600.0, self.author_seconds * 0.4)
        if rank == 0:
            return self.author_deadline - handoff
        if rank == 1:
            return self.author_deadline - handoff / 3.0
        return self.author_deadline

    def inference_deadline(self, phase: str) -> float | None:
        """Let construction inference cross its soft cue, never the instruction cutoff.

        Explicit inferred CAD retains this cutoff for its work. Automatic work
        and early checks still use deadline() for their active phase.
        No absent deadline or additional action is created.
        """
        _phase_rank(phase)
        return self.deadline("handoff" if phase == "construction" else phase)

    def correction_deadline(self) -> float | None:
        """Leave half the existing instruction window for a final protected review."""
        if self.author_deadline is None:
            return None
        reserve = (
            min(600.0, self.author_seconds * 0.4) / 6.0 if self.author_seconds is not None else 0.0
        )
        return self.author_deadline - reserve

    def can_correct(self, index: int, now: float) -> bool:
        """Use existing slots only, retaining the final slot and review time."""
        if self.action_limit is not None and index >= self.action_limit - 1:
            return False
        cutoff = self.correction_deadline()
        return cutoff is None or now < cutoff

    def phase(self, index: int, now: float, previous: str = "construction") -> str:
        """Advance on either declared budget, retaining any previous advancement."""
        rank = _phase_rank(previous)
        if self.action_limit is not None:
            if index >= self.action_limit - 1:
                rank = 2
            elif self.action_limit >= 3:
                slots = min(16, max(2, self.action_limit // 4))
                if index >= self.action_limit - slots:
                    rank = max(rank, 1)
            elif self.action_limit == 2:
                rank = max(rank, 1)
        construction = self.deadline("construction")
        handoff = self.deadline("handoff")
        if handoff is not None and now >= handoff:
            rank = 2
        elif construction is not None and now >= construction:
            rank = max(rank, 1)
        return _PHASES[rank]


def capture_deadline(
    overall_deadline: float | None,
    reserved_seconds: float | None,
) -> float | None:
    """Leave capture time inside an existing overall budget; None stays unbounded.

    No declared reserve leaves the supplied overall deadline unchanged. The caller
    chooses where this validation deadline applies; this function starts no work.
    """
    if overall_deadline is None:
        return None
    if not math.isfinite(overall_deadline):
        raise ValueError("overall_deadline must be finite or None")
    reserve = 0.0 if reserved_seconds is None else reserved_seconds
    _nonnegative(reserve, "reserved_seconds")
    return overall_deadline - min(60.0, reserve / 5.0)
