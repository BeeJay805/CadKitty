"""Bounded investigation actions for the built-in author, without fabricated measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from forge_mind.workspace.action_payloads import (
    EmptyArguments,
    ExplainDesignArguments,
    decode_json_object,
    validate_action_payload,
)
from forge_mind.workspace.design_reasoning import (
    ChallengeRequest,
    RationaleRequest,
    add_rationale,
    explain_design,
    preview_challenge,
)
from forge_mind.workspace.experiment_models import CalibrationRequest, PredictionRequest
from forge_mind.workspace.experiment_service import calibrate_experiment, commit_prediction
from forge_mind.workspace.lab_records import list_records, read_record, record_summary
from forge_mind.workspace.lab_service import create_comparison
from forge_mind.workspace.store import WorkspaceError
from forge_mind.workspace.synthesis_models import SynthesisRequest

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService


def run_lab_action(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    operation: str,
    document_json: str,
) -> dict[str, Any]:
    # An author can investigate supplied evidence, but cannot create physical observations.
    body = decode_json_object(document_json or "{}")
    if operation == "records":
        validate_action_payload(EmptyArguments, body)
        return {"records": [record_summary(r) for r in list_records(service, sid, owner)]}
    if operation.startswith("record:"):
        validate_action_payload(EmptyArguments, body)
        return read_record(service, sid, owner, operation.removeprefix("record:"))
    if operation == "explain":
        arguments = validate_action_payload(ExplainDesignArguments, body, "lab.explain")
        return explain_design(service, sid, owner, rid, **arguments.model_dump())
    if operation == "compare":
        return create_comparison(
            service,
            sid,
            owner,
            rid,
            service.store.engineering(sid)["version"],
            validate_action_payload(SynthesisRequest, body),
        )
    if operation in {"rationale", "challenge"}:
        body.update(revision_id=rid, input_version=service.store.engineering(sid)["version"])
        if operation == "rationale":
            return add_rationale(
                service, sid, owner, validate_action_payload(RationaleRequest, body)
            )
        return preview_challenge(
            service, sid, owner, validate_action_payload(ChallengeRequest, body)
        )
    if operation == "calibrate":
        return calibrate_experiment(
            service, sid, owner, validate_action_payload(CalibrationRequest, body)
        )
    if operation == "predict":
        return commit_prediction(
            service, sid, owner, validate_action_payload(PredictionRequest, body)
        )
    raise WorkspaceError(
        "Supported lab actions: records, record:<id>, explain, compare, rationale, challenge, calibrate, predict. Observations must be supplied outside the author loop."
    )
