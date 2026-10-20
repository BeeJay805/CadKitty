"""Connect screened mechanisms and experimental hypotheses to editable candidates."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.lab_records import capture_binding, read_record, save_record
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.product_sources import read_sources
from forge_mind.workspace.store import WorkspaceError
from forge_mind.workspace.synthesis_models import SynthesisRequest

if TYPE_CHECKING:
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.lab_api import CalibratedDesignRequest


def create_comparison(
    service: WorkspaceService,
    sid: str,
    owner: str,
    rid: str,
    input_version: int,
    request: SynthesisRequest,
) -> dict[str, Any]:
    from forge_mind.workspace.synthesis import compare_mechanisms

    binding = capture_binding(service, sid, owner, rid, input_version)
    source_set = read_sources(service.store, sid, owner)
    claim_ids = {claim["id"] for claim in source_set["claims"]}
    if any(not set(c.source_claim_ids) <= claim_ids for c in request.candidates):
        raise WorkspaceError("Mechanism comparison references an unknown source claim")
    result = compare_mechanisms(request)
    return save_record(
        service,
        sid,
        owner,
        "comparison",
        binding,
        {
            **result,
            "status": "screened",
            "physical_qualification": "unverified",
            "source_scope": "References identify owned source claims; inferred numerical mappings and supplier ratings are not certified",
        },
    )


def realize_comparison(
    service: WorkspaceService,
    sid: str,
    owner: str,
    identifier: str,
    candidate_id: str,
    name: str,
) -> dict[str, Any]:
    from forge_mind.workspace.synthesis import candidate_document

    record = read_record(service, sid, owner, identifier, "comparison")
    if record["stale"]:
        raise WorkspaceError("Comparison is stale; rerun with the current conditions")
    request = SynthesisRequest.model_validate(record["request"])
    selected = next((c for c in record["candidates"] if c["id"] == candidate_id), None)
    if selected is None or not selected.get("document_hash"):
        raise WorkspaceError("Candidate has no executable feature document")
    document = candidate_document(request, candidate_id)
    if document.fingerprint() != selected["document_hash"]:
        raise WorkspaceError("Realized document no longer matches its mechanism screening")
    requirements = [
        Requirement(
            id="functional_brief",
            text=request.brief,
            source="explicit",
            source_quote=request.brief,
            metric="unmeasured",
            critical=True,
        )
    ]
    for identifier_key, value, unit in (
        ("opening_min_mm", request.task.opening_min_mm, "mm"),
        ("opening_max_mm", request.task.opening_max_mm, "mm"),
        ("required_force_per_jaw_n", request.task.required_force_per_jaw_n, "N per jaw"),
        ("max_close_time_s", request.task.max_close_time_s, "s"),
        ("max_width_mm", request.task.max_width_mm, "mm"),
    ):
        if value is not None:
            text = f"{identifier_key}: {value} {unit}"
            requirements.append(
                Requirement(
                    id=identifier_key,
                    text=text,
                    source="explicit",
                    source_quote=text,
                    metric="unmeasured",
                    value=value,
                    critical=True,
                )
            )
    contract = Contract(requirements=requirements)
    project = service.store.create(owner, request.brief, contract.model_dump())
    service.store.update(project["id"], title=name.strip() or document.name)
    job_id = service.store.enqueue(
        project["id"],
        owner,
        {
            "document": document.model_dump(),
            "contract": contract.model_dump(),
            "preview_only": True,
            "explanation": f"Realize screened {selected['name']} architecture; physical obligations remain unresolved",
            "lab_realization": {
                "project_id": sid,
                "comparison_id": record["id"],
                "receipt_sha256": record["receipt_sha256"],
                "candidate_id": candidate_id,
                "document_hash": selected["document_hash"],
                "task_hash": record["task_hash"],
            },
        },
    )
    service.start()
    return {
        "id": project["id"],
        "project_id": project["id"],
        "job_id": job_id,
        "state": "queued",
        "source_project_id": sid,
        "comparison_id": record["id"],
    }


def check_lab_job(service: WorkspaceService, job: dict[str, Any]) -> None:
    """Repeat source checks in the worker, not just at HTTP admission."""
    sid, owner = job["session"], job["owner"]
    if job.get("lab_binding"):
        binding = job["lab_binding"]
        # Queue admission already compared the design head transactionally. Preview
        # admission then restores the accepted head and clears the candidate pointer.
        # Requiring that cleared pointer here would reject edits of unaccepted CAD.
        current = capture_binding(
            service,
            sid,
            owner,
            binding["revision_id"],
            binding["input_version"],
            require_current=False,
        )
        if (
            job.get("expected_current_revision") != binding["revision_id"]
            or job.get("revision_id") != binding["revision_id"]
        ):
            raise WorkspaceError("Queued challenge is not bound to its admitted source revision")
        if current != binding:
            raise WorkspaceError("Challenge evidence changed after queue admission")
        record = read_record(service, sid, owner, job["lab_challenge_id"], "challenge")
        if digest(job["document"]) != record["candidate_document_hash"]:
            raise WorkspaceError("Queued challenge document diverges from its proposal")
    if job.get("lab_realization"):
        source = job["lab_realization"]
        record = read_record(
            service, source["project_id"], owner, source["comparison_id"], "comparison"
        )
        if record["stale"] or record["receipt_sha256"] != source["receipt_sha256"]:
            raise WorkspaceError("Mechanism comparison changed or became stale while queued")
        if digest(job["document"]) != source["document_hash"]:
            raise WorkspaceError("Queued mechanism differs from its screened document")


def calibrated_design(
    service: WorkspaceService,
    sid: str,
    owner: str,
    identifier: str,
    request: CalibratedDesignRequest,
) -> dict[str, Any]:
    from forge_mind.workspace.design_reasoning import ChallengeRequest, preview_challenge
    from forge_mind.workspace.experiment_models import ExperimentProtocol
    from forge_mind.workspace.experiments import recommend_thickness

    calibration = read_record(service, sid, owner, identifier, "calibration")
    if calibration["stale"]:
        raise WorkspaceError("Calibration is stale; create a current investigation")
    protocol = ExperimentProtocol.model_validate(calibration["protocol"])
    revision = service._load_revision(sid, calibration["binding"]["revision_id"])
    document = DesignDocument.model_validate(revision["document"])
    if request.variable_name not in document.variables or not math.isclose(
        document.variables[request.variable_name],
        protocol.thickness_mm,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise WorkspaceError(
            "Map an existing design variable whose value equals the protocol thickness"
        )
    # A beam formula requires an actual rectangular section, not a coincidentally equal bore radius.
    if document.assembly:
        part = next((p for p in document.assembly.parts if p.id == request.part_id), None)
        if part is None:
            raise WorkspaceError("Choose the rectangular specimen part definition")
        features, outputs = part.features, [part.output]
    else:
        if request.part_id is not None:
            raise WorkspaceError("Standalone specimen has no part definition ID")
        features, outputs = document.features, document.outputs
    feature = next((f for f in features if f.id == request.feature_id), None)
    if (
        feature is None
        or feature.kind != "box"
        or outputs != [feature.id]
        or feature.bindings.get(request.thickness_parameter) != request.variable_name
    ):
        raise WorkspaceError(
            "Map thickness to an output rectangular box dimension bound to this variable; modified sections need another model"
        )
    dimensions = {
        **feature.params,
        **{key: document.values()[value] for key, value in feature.bindings.items()},
    }
    other_sides = [dimensions[key] for key in ("x", "y", "z") if key != request.thickness_parameter]
    if not any(math.isclose(side, protocol.width_mm, rel_tol=1e-9) for side in other_sides):
        raise WorkspaceError("Specimen width does not match the declared rectangular CAD section")
    if not any(
        math.isclose(other_sides[i], protocol.width_mm, rel_tol=1e-9)
        and other_sides[1 - i] >= request.condition.span_mm
        for i in range(2)
    ):
        raise WorkspaceError("Declared test span extends beyond the rectangular CAD specimen")
    if not request.mapping_basis.strip():
        raise WorkspaceError("State how this variable controls the measured specimen thickness")
    fit = next(f for f in calibration["fits"] if f["hypothesis"] == request.hypothesis)
    prediction = recommend_thickness(
        protocol,
        fit,
        request.condition,
        request.target_displacement_mm,
        request.min_thickness_mm,
        request.max_thickness_mm,
    )
    proposed = document.model_dump()
    proposed["variables"][request.variable_name] = prediction["recommended_thickness_mm"]
    new_values = DesignDocument.model_validate(proposed).values()
    new_dimensions = {
        **feature.params,
        **{key: new_values[value] for key, value in feature.bindings.items()},
    }
    if any(
        not math.isclose(new_dimensions[key], dimensions[key], rel_tol=1e-10, abs_tol=1e-10)
        for key in ("x", "y", "z")
        if key != request.thickness_parameter
    ):
        raise WorkspaceError(
            "Thickness variable also changes another specimen dimension; coupled geometry requires a different response model"
        )
    proposal = preview_challenge(
        service,
        sid,
        owner,
        ChallengeRequest(
            revision_id=calibration["binding"]["revision_id"],
            input_version=calibration["binding"]["input_version"],
            question=f"Revise {request.variable_name} using {request.hypothesis}; target deflection {request.target_displacement_mm} mm under the declared test",
            variables={request.variable_name: prediction["recommended_thickness_mm"]},
        ),
    )
    return save_record(
        service,
        sid,
        owner,
        "design_recommendation",
        calibration["binding"],
        {
            "challenge_id": proposal["id"],
            "calibration_id": calibration["id"],
            "mapping_basis": request.mapping_basis,
            "specimen_mapping": {
                "part_id": request.part_id,
                "feature_id": request.feature_id,
                "thickness_parameter": request.thickness_parameter,
                "scope": "Direct output rectangular feature binding; fixture location and material transfer remain declared",
            },
            "variable_name": request.variable_name,
            "prediction": prediction,
            "status": "proposal",
            "provenance": calibration["provenance"],
            "physical_qualification": "unverified",
            "scope": "Conditional section change under a selected calibrated hypothesis and owner-declared CAD mapping; rebuild and a new prospective test are required",
        },
        parents=(identifier, proposal["id"]),
    )
