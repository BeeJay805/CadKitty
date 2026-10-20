"""Native author access to scoped design context and measured improvement experiments."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.action_payloads import (
    EmptyArguments,
    decode_json_object,
    read_design_arguments,
    validate_action_payload,
)
from forge_mind.workspace.design_study_api import study_schema
from forge_mind.workspace.design_study_models import (
    AddStudyCandidate,
    CreateDesignStudy,
    RunDesignStudy,
    SearchDesignStudy,
)
from forge_mind.workspace.design_study_service import (
    _fresh,
    _study,
    add_candidate,
    create_study,
    list_studies,
    read_study,
    run_study_job,
)
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.inspection import query_document
from forge_mind.workspace.models import Action
from forge_mind.workspace.store import WorkspaceError

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def execute_study_action(
    service: WorkspaceService,
    sid: str,
    owner: str,
    job: dict[str, Any],
    action: Action,
    current: str | None,
    cancelled: Callable[[], bool],
) -> dict[str, Any]:
    payload = decode_json_object(action.document_json or "{}")
    if action.kind == "read_design":
        arguments = read_design_arguments(action.query or "parts", payload)
        revision_id = action.revision_id or current
        if not revision_id:
            raise WorkspaceError("Build or select a design before reading its pages")
        revision = service._load_revision(sid, revision_id)
        return query_document(
            DesignDocument.model_validate(revision.get("document")),
            revision_id,
            arguments.collection,
            **arguments.model_dump(exclude={"collection"}),
        )
    query, _, identifier = action.query.partition(":")
    if query == "schema":
        validate_action_payload(EmptyArguments, payload)
        return study_schema()
    if query == "list":
        validate_action_payload(EmptyArguments, payload)
        return list_studies(service, sid, owner)
    if query == "create":
        return create_study(
            service, sid, owner, validate_action_payload(CreateDesignStudy, payload), cancelled
        )
    if query == "read":
        validate_action_payload(EmptyArguments, payload)
        return read_study(service, sid, owner, identifier)
    if query == "candidate":
        return add_candidate(
            service, sid, owner, identifier, validate_action_payload(AddStudyCandidate, payload)
        )
    if query in {"run", "search"}:
        search = (
            validate_action_payload(SearchDesignStudy, payload).model_dump()
            if query == "search"
            else None
        )
        retry_failed = (
            validate_action_payload(RunDesignStudy, payload, strict=True).retry_failed
            if query == "run"
            else False
        )
        study = _study(service, sid, owner, identifier)
        _fresh(service, sid, owner, study)
        run = {
            "id": uuid.uuid4().hex,
            "job_id": job["id"],
            "study_id": identifier,
            "state": "running",
            "search": search,
            "rounds_completed": 0,
            "created_at": time.time(),
        }
        with service.store.connect() as con:
            con.execute(
                "INSERT INTO design_study_runs VALUES (?,?,?,?)",
                (run["id"], identifier, json.dumps(run), digest(run)),
            )
        run_study_job(
            service,
            {
                **job,
                "study_id": identifier,
                "run_id": run["id"],
                "search": search,
                "retry_failed": retry_failed,
            },
            cancelled,
        )
        return read_study(service, sid, owner, identifier)
    raise WorkspaceError(
        "Use design_study schema, list, create, read:<id>, candidate:<id>, run:<id> or search:<id>"
    )
