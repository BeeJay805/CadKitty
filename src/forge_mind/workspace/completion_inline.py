"""Early prepared probes reuse immutable evidence, never author-supplied verdicts."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.budget import expired, share_time
from forge_mind.workspace.completion import _numeric_selector, _snapshot, _structure
from forge_mind.workspace.completion_readonly import prepared_binding as _binding
from forge_mind.workspace.completion_readonly import read_prepared_receipt
from forge_mind.workspace.completion_readonly import reusable_result as _reusable_result
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def _read_cached(
    service: WorkspaceService,
    sid: str,
    snapshot: dict[str, Any],
    definition: dict[str, Any],
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """A cached row is only a pointer. Read and verify the actual receipt every time."""
    try:
        with service.store.connect() as con:
            return read_prepared_receipt(con, sid, snapshot, definition, entry)
    except Exception:
        return None


def execute_prepared(
    service: WorkspaceService,
    sid: str,
    owner: str,
    snapshot: dict[str, Any],
    definition: dict[str, Any],
    timeout: float | None,
    cancelled: Callable[[], bool],
) -> tuple[dict[str, Any], bool]:
    """Execute or reuse one exact probe; final validation shares this same path."""
    from forge_mind.workspace.product_service import run_tool

    if cancelled():
        raise WorkspaceError("Prepared check cancelled")
    if (
        _numeric_selector(definition["request"])
        and definition.get("reference_revision_id") != snapshot["revision_id"]
    ):
        raise WorkspaceError("Numeric body selectors require the exact reference revision")
    if _snapshot(service, sid, owner, snapshot["revision_id"]) != snapshot:
        raise WorkspaceError("Prepared check inputs changed")
    cache = service.store.get(sid, owner).get("prepared_check_cache") or {}
    receipt = _read_cached(service, sid, snapshot, definition, cache.get(definition["id"], {}))
    reused = receipt is not None
    if receipt is None:
        receipt = run_tool(
            service, sid, owner, snapshot["revision_id"], definition["request"], timeout, cancelled
        )
    if cancelled() or _snapshot(service, sid, owner, snapshot["revision_id"]) != snapshot:
        raise WorkspaceError("Prepared check inputs changed or cancelled")
    cache[definition["id"]] = {
        "binding": _binding(snapshot, definition),
        "receipt_id": receipt["id"],
    }
    service.store.update(sid, prepared_check_cache=cache)
    return receipt, reused


def author_checkpoint(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str | None,
    job_id: str,
    deadline: float | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    """Measure one ready subsystem before another author call, leaving repair time.

    Other independent subsystems remain available. An unavailable probe is attempted
    once per job and binding, with a visible reason; a changed input or resumed job
    permits another attempt. Finalization still retries unavailable probes.
    """
    from forge_mind.workspace.completion import validate_plan

    session = service.store.get(sid, owner)
    plan = session.get("design_plan") or {}
    revision = service._load_revision(sid, rid) if rid else {}
    if not revision.get("measurement", {}).get("geometry_valid") or not plan.get("subsystems"):
        return {"status": "unverified", "subsystems": [], "reason": "No measurable subsystem plan"}
    validate_plan(plan, revision.get("contract") or session.get("contract"))
    snapshot = _snapshot(service, sid, owner, rid)
    rows = _structure(plan, revision)
    by_id = {row["id"]: row for row in rows}
    definitions = {c["id"]: c for s in plan["subsystems"] for c in s.get("checks", [])}
    cache = session.get("prepared_check_cache") or {}
    old = session.get("author_checkpoint") or {}
    attempts = old.get("attempts", {}) if old.get("job_id") == job_id else {}
    pending: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ready = (
            row["status"] != "missing"
            and row["geometry_declared"]
            and all(
                by_id[d]["status"] != "missing" and by_id[d]["geometry_declared"]
                for d in row["dependencies"]
            )
        )
        for check in row["checks"]:
            definition = definitions[check["id"]]
            receipt = _read_cached(service, sid, snapshot, definition, cache.get(check["id"], {}))
            if receipt is not None:
                check.update(
                    status=receipt["result"]["status"],
                    receipt_id=receipt["id"],
                    reason="Current exploratory receipt",
                    result=receipt["result"],
                )
            elif ready and (
                attempts.get(check["id"], {}).get("binding") != _binding(snapshot, definition)
                or attempts.get(check["id"], {}).get("receipt_id")
            ):
                pending.setdefault(row["id"], []).append(check)
            elif ready:
                check["reason"] = attempts[check["id"]].get("reason", "Probe unavailable")
            else:
                check["reason"] = "Subsystem or dependency geometry is missing or undeclared"
    target = next(iter(pending), None)
    if target is not None:
        for check in pending[target]:
            if cancelled() or expired(deadline):
                break
            definition = definitions[check["id"]]
            try:
                receipt, reused = execute_prepared(
                    service,
                    sid,
                    owner,
                    snapshot,
                    definition,
                    share_time(deadline, 2 * len(pending[target])),
                    lambda: cancelled() or expired(deadline),
                )
                check.update(
                    status=receipt["result"]["status"],
                    receipt_id=receipt["id"],
                    reason="Early exploratory probe completed",
                    result=receipt["result"],
                    reused=reused,
                )
                if not _reusable_result(receipt["result"]):
                    check["reason"] = (
                        f"Probe unavailable: {str(receipt['result'].get('results'))[:300]}"
                    )
            except Exception as exc:
                check.update(status="unverified", reason=f"{type(exc).__name__}: {str(exc)[:300]}")
            attempts[check["id"]] = {
                "binding": _binding(snapshot, definition),
                "reason": check["reason"],
                "receipt_id": check.get("receipt_id")
                if _reusable_result(check.get("result", {}))
                else None,
            }
            # Persist after each check so interruption does not discard attempts.
            service.store.update(
                sid,
                author_checkpoint={
                    **snapshot,
                    "job_id": job_id,
                    "attempts": attempts,
                    "subsystems": rows,
                },
            )
    fresh = (
        not cancelled()
        and not expired(deadline)
        and _snapshot(service, sid, owner, rid) == snapshot
    )
    for row in rows:
        statuses = [c["status"] for c in row["checks"]]
        if row["status"] != "missing":
            row["status"] = (
                "checked_fail"
                if fresh and "fail" in statuses
                else (
                    "checked_pass"
                    if fresh
                    and statuses
                    and all(s == "pass" for s in statuses)
                    and row["geometry_declared"]
                    and not row["unresolved"]
                    else "unverified"
                )
            )
        if not fresh:
            for check in row["checks"]:
                check.update(status="unverified", reason="Checkpoint inputs changed or cancelled")
    report = {
        **snapshot,
        "job_id": job_id,
        "attempts": attempts,
        "subsystems": rows,
        "stale": not fresh,
        "status": "exploratory",
        "measured_subsystem": target,
        "scope": "Early scoped evidence guides repair. No frozen acceptance or physical qualification.",
    }
    service.store.update(sid, author_checkpoint=report)
    return report
