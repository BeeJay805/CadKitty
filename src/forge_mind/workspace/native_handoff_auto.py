"""Claim one host preparation opportunity; existing services still own every check."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from forge_mind.workspace.native_handoff_budget import HandoffBudget
from forge_mind.workspace.store import Store, WorkspaceError


def claim_automatic_preparation(
    store: Store,
    sid: str,
    owner: str,
    jid: str,
    *,
    context: dict[str, Any],
    published_revision_id: str | None,
    index: int,
    schedule: HandoffBudget | None,
    phase: str,
    now: float,
    enabled: bool,
) -> dict[str, Any] | None:
    """Persist at most one attempt for this job, before any preparation work.

    Eligibility never establishes construction or requirement satisfaction. The
    unchanged preparation helper rechecks live authority and all work blockers.
    """
    if (
        enabled is not True
        or schedule is None
        or phase != "construction"
        or not published_revision_id
        or type(index) is not int
        or index < 0
        or not math.isfinite(now)
        or schedule.phase(index, now, phase) != "construction"
    ):
        return None
    # Charge this slot and retain both an editable review and the final slot.
    if schedule.action_limit is not None and index + 2 >= schedule.action_limit:
        return None
    cutoff = schedule.deadline("construction")
    if cutoff is not None and now >= cutoff:
        return None
    review_time = now if cutoff is None else cutoff
    if schedule.phase(index + 1, review_time, phase) == "instructions":
        return None
    if (
        any(
            context.get(name)
            for name in (
                "failed_submission",
                "failed_attempt",
                "plan_draft",
                "proposal",
                "proposed_contract",
            )
        )
        or (context.get("request_attribution") or {}).get("status") == "needs_input"
    ):
        return None
    selected = context.get("selected") or {}
    preflight = context.get("handoff_preflight") or {}
    preparation = preflight.get("print_preparation") or {}
    guide = context.get("delivery_guidance") or {}
    basis = guide.get("basis") or {}
    plan_hash = preflight.get("plan_hash")
    reasons = preflight.get("work_reasons")
    if (
        selected.get("id") != published_revision_id
        or preflight.get("revision_id") != published_revision_id
        or basis.get("revision_id") != published_revision_id
        or not isinstance(plan_hash, str)
        or not plan_hash
        or basis.get("plan_hash") != plan_hash
        or guide.get("status") == "unavailable"
        or not isinstance(guide.get("basis_hash"), str)
        or not guide["basis_hash"]
        or preparation.get("required") is not True
        or preparation.get("supported") is not True
        or preparation.get("status") != "missing"
        or not isinstance(reasons, list)
        or len(reasons) != 1
    ):
        return None
    with store.connect() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT document FROM sessions WHERE id=? AND owner=?", (sid, owner)
        ).fetchone()
        if row is None:
            return None
        doc = json.loads(row[0])
        try:
            store._check_plan_writer(con, sid, doc, jid)
        except WorkspaceError:
            return None
        if (
            doc.get("authoring_mode") != "native_operations"
            or doc.get("selected_revision") != published_revision_id
            or doc.get("plan_hash") != plan_hash
            or doc.get("authority_conflict_job_id") == jid
            or doc.get("proposal")
            or (doc.get("intent_attribution") or {}).get("status") == "needs_input"
            or (doc.get("automatic_handoff_attempt") or {}).get("job_id") == jid
        ):
            return None
        draft = con.execute(
            "SELECT admitted FROM plan_drafts WHERE session=? ORDER BY version DESC LIMIT 1", (sid,)
        ).fetchone()
        if draft is not None and not draft[0]:
            return None
        marker = {
            "job_id": jid,
            "sid": sid,
            "revision_id": published_revision_id,
            "basis_hash": guide["basis_hash"],
            "plan_hash": plan_hash,
            "action_index": index,
            "phase": phase,
            "origin": "host",
            "claimed_at": now,
        }
        doc["automatic_handoff_attempt"] = marker
        con.execute(
            "UPDATE sessions SET document=? WHERE id=? AND owner=?", (json.dumps(doc), sid, owner)
        )
    return deepcopy(marker)
