"""Bounded inference retries and output correction; never replay CAD actions."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from forge_mind.llm.client import LLMError
from forge_mind.llm.output_contract import OutputContractError

T = TypeVar("T")
CAPACITY_ERROR = (
    "Codex structured call failed (exit=1): Selected model is at capacity. "
    "Please try a different model."
)


def run_model_call(
    operation: Callable[[], T],
    *,
    native: bool,
    allow_timeout_retry: bool,
    remaining: Callable[[], float | None],
    cancelled: Callable[[], bool],
    configure_timeout: Callable[[float | None], None],
    on_retry: Callable[[str, int, float], None],
    correction_factory: Callable[[OutputContractError], Callable[[], T] | None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> T:
    """Share one attempt cap across capacity, timeout and one output correction."""
    attempts = 3 if native else 2 if allow_timeout_retry else 1
    timeout_retried = False
    correction_used = False
    active_operation = operation

    def check_budget() -> float | None:
        if cancelled():
            raise LLMError("Authoring call cancelled before the next attempt")
        left = remaining()
        if left is not None and left <= 0:
            raise LLMError("Authoring deadline reached; reserved validation starts now")
        return left

    for index in range(attempts):
        configure_timeout(check_budget())
        check_budget()
        try:
            return active_operation()
        except LLMError as error:
            left = remaining()
            if index + 1 == attempts or cancelled() or (left is not None and left <= 0):
                raise
            if isinstance(error, OutputContractError):
                if (
                    correction_used
                    or correction_factory is None
                    or not error.retryable
                    or error.stage not in {"json_decode", "pydantic_validation"}
                    or (left is not None and left < 45)
                ):
                    raise
                correction_used = True
                corrected = correction_factory(error)
                if corrected is None:
                    raise
                left = check_budget()
                if left is not None and left < 45:
                    raise
                active_operation = corrected
                reason, delay = "output_contract", 0.0
            elif native and str(error).strip() == CAPACITY_ERROR:
                reason, delay = "capacity", float(2 ** (index + 1))
                if left is not None and left <= delay:
                    raise
            elif (
                allow_timeout_retry
                and not timeout_retried
                and (left is None or left >= 45)
                and any(
                    word in str(error).lower()
                    for word in ("timeout", "timed out", "wall-clock limit")
                )
            ):
                reason, delay = "timeout", 0.0
                timeout_retried = True
            else:
                raise
            on_retry(reason, index + 2, delay)
            wake_at = clock() + delay
            while clock() < wake_at:
                left = check_budget()
                pause = min(0.1, wake_at - clock())
                if left is not None:
                    pause = min(pause, left)
                if pause > 0:
                    sleep(pause)
    raise RuntimeError("Unreachable model retry state")
