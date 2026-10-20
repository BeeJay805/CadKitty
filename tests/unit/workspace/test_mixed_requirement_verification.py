"""Mixed scalar and unsupported duties retain exact postcapture status, without CAD."""

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_requirement_verification import digital, prepared

from forge_mind.workspace import completion, delivery_guidance, native_delivery
from forge_mind.workspace import requirement_verification as verification
from forge_mind.workspace.models import Contract, Finding, Requirement
from forge_mind.workspace.native_delivery import current_delivery_binding
from forge_mind.workspace.project_access import project_overview
from forge_mind.workspace.project_delivery import create_project_export
from forge_mind.workspace.store import Store, WorkspaceError


def mixed_contract():
    """Replay the actual candidate's 21-duty ordering and metric shape, without assets."""
    texts = [
        "Provide a small, machinable assembly.",
        "Provide four cylinders.",
        "Provide a hand-turnable crankshaft.",
        "Include a camshaft in the assembly.",
        "Include pistons in the assembly.",
        "Include rods in the assembly.",
        "Include bearings in the assembly.",
        "Provide lubrication paths.",
        "Provide removable service covers.",
        "Keep the assembled X extent at or below 200 mm.",
        "Keep the assembled Y extent at or below 120 mm.",
        "Keep the assembled Z extent at or below 160 mm.",
        "Arrange the cylinders in a row along X.",
        "Orient the cylinder axes along Z.",
        "Provide a machinable block.",
        "Use aluminum for the block.",
        "Use steel shafts.",
        "Provide replaceable bushings for the shafts.",
        "Use bronze for the bushings.",
        "Choose the remaining dimensions for a desktop mechanical prototype.",
        "Use conventional milling, turning and drilling.",
    ]
    dimensions = {10: ("extent_x", 200), 11: ("extent_y", 120), 12: ("extent_z", 160)}
    return Contract(
        requirements=[
            Requirement(
                id=f"r{index}",
                text=text,
                source_quote=text,
                **(
                    {
                        "metric": dimensions[index][0],
                        "comparator": "le",
                        "value": dimensions[index][1],
                    }
                    if index in dimensions
                    else {}
                ),
            )
            for index, text in enumerate(texts, 1)
        ]
    )


@pytest.mark.parametrize(
    "target,eligible",
    [
        (Contract(), False),
        (
            Contract(
                requirements=[
                    Requirement(
                        id="size", text="Size", source_quote="Size", metric="extent_x", value=10
                    )
                ]
            ),
            False,
        ),
        (Contract(requirements=[Requirement(id="fit", text="Fit", source_quote="Fit")]), True),
        (mixed_contract(), True),
        (Contract(requirements=[digital()]), True),
    ],
)
def test_status_eligibility_keeps_empty_and_all_scalar_legacy_paths(target, eligible):
    assert verification.has_requirement_status(target.model_dump()) is eligible


def native_case(tmp_path, monkeypatch, scalar_status):
    # Create native authority at the normal creation boundary; never change an existing mode.
    create = Store.create

    def create_native(self, owner, prompt, contract, authoring_mode=None):
        return create(self, owner, prompt, contract, authoring_mode or "native_operations")

    monkeypatch.setattr(Store, "create", create_native)
    original_guidance = delivery_guidance.guidance_context
    service, sid, jid, target, _, _, _ = prepared(tmp_path, monkeypatch, target=mixed_contract())
    rid = service.store.get(sid, "owner")["selected_revision"]
    revision = service.store.revision(sid, rid)
    revision["measurement"]["findings"] = [
        Finding(
            requirement_id=r.id,
            status="unverified" if r.metric == "unmeasured" else scalar_status,
            method="fixture measurement",
            reason=f"Exact saved scalar finding: {r.id}",
        ).model_dump()
        for r in target.requirements
    ]
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(revision), sid, rid),
        )
    with monkeypatch.context() as capture:
        capture.setattr(delivery_guidance, "guidance_context", original_guidance)
        receipt = create_project_export(service, sid, "owner", rid, include_stl=True)
    delivery = {
        **current_delivery_binding(service, sid, "owner"),
        "job_id": jid,
        "status": "delivered",
        "readback_verified": True,
        "receipt": receipt,
    }
    monkeypatch.setattr(service, "_run_agent_job", lambda job: "needs_review")
    monkeypatch.setattr(completion, "finalize_completion", lambda *args: None)
    monkeypatch.setattr(
        native_delivery,
        "capture_native_delivery",
        lambda *args: service.store.update(sid, automatic_delivery=deepcopy(delivery)),
    )
    monkeypatch.setattr(
        service.executor,
        "process_observation",
        lambda *args, **kwargs: pytest.fail("Mixed legacy findings cannot invoke a CAD worker"),
    )
    return service, sid, jid, target, revision


@pytest.mark.parametrize("scalar_status", ["pass", "fail", "unverified"])
def test_native_postcapture_report_covers_mixed_duties_and_current_projection(
    tmp_path, monkeypatch, scalar_status
):
    service, sid, jid, target, original = native_case(tmp_path, monkeypatch, scalar_status)
    service.run_job({"id": jid, "session": sid, "owner": "owner", "max_seconds": 120})
    proof = verification.read_contract_verification(service, sid, "owner")
    assert proof["status"] == ("fail" if scalar_status == "fail" else "unverified")
    assert proof["requirements_satisfied"] is False
    assert proof["physical_readiness"] == "unverified"
    assert [r["requirement_id"] for r in proof["results"]] == [r.id for r in target.requirements]
    findings = {r["requirement_id"]: r for r in original["measurement"]["findings"]}
    for requirement, result in zip(target.requirements, proof["results"], strict=True):
        assert result["criterion_hash"] is None
        assert result["status"] == (
            "unverified" if requirement.metric == "unmeasured" else scalar_status
        )
        if requirement.metric != "unmeasured":
            assert result["facts"]["finding"] == findings[requirement.id]
    assert service.store.revision(sid, original["id"]) == original
    saved = service.store.get(sid, "owner")
    assert saved["contract"] == target.model_dump() and saved["accepted_revision"] is None
    overview = project_overview(service.store, sid, "owner")
    assert overview["current_completion_status"] == "unverified"
    assert overview["job_outcome_state"] == overview["state"] == "needs_review"
    assert overview["contract_verification"]["finding_counts"] == {
        "pass": 3 if scalar_status == "pass" else 0,
        "fail": 3 if scalar_status == "fail" else 0,
        "unverified": 21 if scalar_status == "unverified" else 18,
    }
    with pytest.raises(WorkspaceError, match="not found"):
        verification.read_contract_verification(service, sid, "other-owner")
    service.store.update(sid, plan_version=saved["plan_version"] + 1)
    with pytest.raises(WorkspaceError, match="basis changed"):
        verification.read_contract_verification(service, sid, "owner")
    stale = project_overview(service.store, sid, "owner")
    assert stale["current_completion_status"] == "unverified"
    assert stale["contract_verification"]["status"] == "unverified"
    assert "basis changed" in stale["contract_verification"]["reason"]


def test_scalar_only_and_empty_legacy_projection_do_not_gain_new_status_fields():
    for requirements in (
        [],
        [Requirement(id="size", text="Size", source_quote="Size", metric="extent_x", value=10)],
    ):
        session = {
            "state": "contract_satisfied",
            "authoring_mode": "native_operations",
            "contract": Contract(requirements=requirements).model_dump(),
        }
        assert verification.completion_projection(session, {"status": "unavailable"}) == {
            "state": "contract_satisfied"
        }
