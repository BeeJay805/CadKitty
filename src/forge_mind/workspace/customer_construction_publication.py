"""Read the existing measured publication without granting another build."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from forge_mind.workspace import measure
from forge_mind.workspace.models import Measurement

if TYPE_CHECKING:
    from forge_mind.workspace.store import Store


def publication_current(
    store: Store,
    con: Any,
    sid: str,
    doc: dict[str, Any],
    record: dict[str, Any],
    binding: dict[str, Any],
) -> bool:
    execution = record.get("execution") or {}
    rid = execution.get("revision_id")
    if (
        execution.get("phase") != "published"
        or not isinstance(execution.get("job_id"), str)
        or not rid
        or rid != doc.get("selected_revision")
        or execution.get("publication_binding") != binding
    ):
        return False
    row = con.execute(
        "SELECT document FROM revisions WHERE session=? AND id=?", (sid, rid)
    ).fetchone()
    revision = json.loads(row[0]) if row else {}
    measured = Measurement.model_validate(revision.get("measurement") or {})
    command = record["command"]
    return bool(
        revision.get("document") == command["document"]
        and revision.get("contract") == command["contract"]
        and revision.get("parent_id") == command["parent"]
        and (revision.get("operation_record") or {}).get("plan_snapshot")
        == command["target_snapshot"]
        and measured.revision_id == rid
        and measured.geometry_valid is True
        and measured.contract_hash == command["target_snapshot"]["contract_hash"]
        and measured.engine_fingerprint == measure.engine_fingerprint()
        and measure.verify_artifacts(store.directory(sid, rid), measured)
    )
