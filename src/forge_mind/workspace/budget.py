"""Optional work budgets: None means no automatic time cutoff."""

from __future__ import annotations

import math
import time
from collections.abc import Iterable
from itertools import count


def end_time(seconds: float | None) -> float | None:
    if seconds is None:
        return None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Time budget must be finite and nonnegative, or None")
    return time.monotonic() + seconds


def time_left(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def expired(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def share_time(deadline: float | None, parts: int = 1) -> float | None:
    remaining = time_left(deadline)
    return None if remaining is None else remaining / parts


def action_indices(limit: int | None) -> Iterable[int]:
    return count() if limit is None else range(limit)
