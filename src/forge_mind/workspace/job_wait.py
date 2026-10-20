"""Bounded, read-only waiting on one job's current owned project snapshot.

This does not query historical job completion or cancel CAD execution. The callback
must enforce ownership on every read and honor its remaining request timeout.
"""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_WAIT_SECONDS = 20.0
ACTIVE_STATES = frozenset({"queued", "running", "cancelling"})
# WorkspaceService / document_service finish states, plus Store startup recovery.
TERMINAL_STATES = frozenset(
    {
        "contract_satisfied",
        "needs_review",
        "needs_input",
        "needs_approval",
        "budget_exhausted",
        "error",
        "cancelled",
        "interrupted",
    }
)
ReadState = Callable[[float], Awaitable[dict[str, Any]]]
Outcome = Literal["terminal", "timeout", "superseded", "unknown"]


class JobWaitSnapshot(BaseModel):
    """Revision pointers from the most recent authorized project observation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str | None = None
    design_revision: str | None = None
    selected_revision: str | None = None
    accepted_revision: str | None = None
    input_version: int | None = Field(default=None, ge=0)


class JobWaitResult(BaseModel):
    """Execution state only; no geometry, acceptance, or historical success claim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_job_id: str
    current_job_id: str | None
    state: str | None
    outcome: Outcome
    elapsed_seconds: float = Field(ge=0, allow_inf_nan=False)
    snapshot: JobWaitSnapshot


def summary_text(result: JobWaitResult) -> str:
    """Describe the outcome for clients that expose only MCP text content."""
    meanings = {
        "terminal": "The requested job stopped. Inspect evidence before judging or accepting.",
        "timeout": "The wait deadline expired. The last observed state is shown; waiting does not cancel CAD.",
        "superseded": "The current job differs. Completion of the requested job is unknown.",
        "unknown": "The observed state is not recognized. Completion is unknown.",
    }
    return (
        f"{result.outcome}: target_job_id={result.target_job_id}; "
        f"current_job_id={result.current_job_id}; state={result.state}. "
        f"{meanings[result.outcome]} Snapshot pointers describe the current observation: "
        f"{result.snapshot.model_dump_json()}."
    )


async def wait_for_job(
    read_state: ReadState, job_id: str, timeout_seconds: float = MAX_WAIT_SECONDS
) -> JobWaitResult:
    """Wait at most 20 seconds, including reads and sleeps, without any write.

    An initial deadline raises TimeoutError because no authorized snapshot exists.
    Later deadlines return the last authorized observation. Callback exceptions and
    request cancellation propagate; a transport timeout is not a wait outcome.
    """
    if not isinstance(job_id, str) or not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("job_id must contain 32 lowercase hexadecimal characters")
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or not 0 < timeout_seconds <= MAX_WAIT_SECONDS
    ):
        raise ValueError("timeout_seconds must be finite, greater than zero, and at most 20")
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + timeout_seconds
    snapshot: JobWaitSnapshot | None = None
    current_job_id: str | None = None
    state: str | None = None

    def result(outcome: Outcome) -> JobWaitResult:
        assert snapshot is not None
        return JobWaitResult(
            target_job_id=job_id,
            current_job_id=current_job_id,
            state=state,
            outcome=outcome,
            elapsed_seconds=loop.time() - started,
            snapshot=snapshot,
        )

    deadline_scope = asyncio.timeout_at(deadline)
    try:
        async with deadline_scope:
            delay = 0.25
            while True:
                observed = await read_state(min(timeout_seconds, max(0.0, deadline - loop.time())))
                snapshot = JobWaitSnapshot.model_validate(
                    {name: observed.get(name) for name in JobWaitSnapshot.model_fields}
                )
                current_job_id = observed.get("job_id")
                state = observed.get("state")
                if current_job_id != job_id:
                    return result("superseded")
                if state in TERMINAL_STATES:
                    return result("terminal")
                if state not in ACTIVE_STATES:
                    return result("unknown")
                await asyncio.sleep(min(delay, max(0.0, deadline - loop.time())))
                delay = min(1.0, delay * 2)
    except TimeoutError:
        if not deadline_scope.expired() or snapshot is None:
            raise
        return result("timeout")
