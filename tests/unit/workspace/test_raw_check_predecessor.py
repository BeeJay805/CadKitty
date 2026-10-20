"""Legacy raw request identities survive typed patch preparation unchanged."""

from copy import deepcopy

import pytest

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.check_amendment import request_hash, validate_check_amendments
from forge_mind.workspace.completion_models import PreparedCheck, Subsystem
from forge_mind.workspace.edit_service import patch_plan
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, DesignPlan, Requirement
from forge_mind.workspace.operation_plan import snapshot
from forge_mind.workspace.plan_patch import PlanPatch, apply_plan_patch
from forge_mind.workspace.store import Store


def legacy_plan():
    request = {
        "tool": "fastener_stack",
        "units": "mm",
        "screw_length": {"low": 12.0, "high": 12.0},
        "grip_layers": [{"low": 5.0, "high": 5.0}],
        "required_engagement": 3.0,
        "available_depth": {"low": 8.0, "high": 8.0},
    }
    return DesignPlan(
        objective="Check nominal engagement",
        subsystems=[
            Subsystem(
                id="attachment",
                name="Attachment",
                requirement_ids=["fit"],
                checks=[PreparedCheck(id="stack", request=request)],
            )
        ],
    )


def first_check(plan):
    return plan["subsystems"][0]["checks"][0]


@pytest.fixture
def raw_predecessor(tmp_path):
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    contract = Contract(
        requirements=[
            Requirement(id="fit", text="Check engagement", source_quote="Check engagement")
        ]
    )
    original = legacy_plan()
    raw = original.model_dump(mode="json")
    # Explicit valid persisted compatibility state, not an observed emission
    # from today's Store/model stack. JSON 0 is accepted by the numeric schema.
    first_check(raw)["request"]["minimum_tip_clearance"] = 0
    assert isinstance(first_check(raw)["request"]["minimum_tip_clearance"], int)
    typed = DesignPlan.model_validate(raw).model_dump(mode="json")
    assert isinstance(first_check(typed)["request"]["minimum_tip_clearance"], float)
    assert request_hash(first_check(raw)["request"]) != request_hash(first_check(typed)["request"])
    sid = service.store.create("owner", "Check engagement", contract.model_dump())["id"]
    service.store.set_design_plan(sid, "owner", original, 0)
    # Simulate a valid legacy persisted plan. Do not rewrite its fingerprint
    # or manufacture new receipt identity merely because typed loading changes 0.
    binding = snapshot(original, 1, contract)
    binding["plan"] = deepcopy(raw)
    binding["plan_hash"] = request_hash(raw)
    service.store.update(
        sid,
        design_plan=deepcopy(raw),
        plan_hash=binding["plan_hash"],
        operation_plan_snapshot=deepcopy(binding),
    )
    return service, sid, raw, typed


def amended_subsystem(raw, previous_hash):
    subsystem = deepcopy(raw["subsystems"][0])
    check = subsystem["checks"][0]
    check["request"]["required_engagement"] = 4.0
    check["criterion_amendment"] = {
        "previous_request_hash": previous_hash,
        "reason": "Correct the nominal engagement target",
        "basis": "Explicit engagement criterion; threads and physical fit remain unverified",
    }
    return Subsystem.model_validate(subsystem)


def test_service_patch_preserves_exact_raw_predecessor_hash(raw_predecessor):
    service, sid, raw, typed = raw_predecessor
    exact = request_hash(first_check(raw)["request"])
    changed = amended_subsystem(raw, exact)
    candidate = deepcopy(raw)
    candidate["subsystems"] = [changed.model_dump(mode="json")]
    validate_check_amendments(candidate, raw)
    with pytest.raises(ValueError, match="predecessor request hash"):
        validate_check_amendments(candidate, typed)
    result = patch_plan(
        service,
        sid,
        "owner",
        PlanPatch(basis=service.store.edit_basis(sid, "owner"), subsystems=[changed]),
    )
    assert result["applied"] and result["plan_applied"] and not result["cad_applied"]
    saved = service.store.get(sid, "owner")
    assert (
        first_check(saved["design_plan"])["criterion_amendment"]["previous_request_hash"] == exact
    )
    assert saved["design_plan_history"][-1]["plan"] == raw
    assert isinstance(
        first_check(saved["design_plan_history"][-1]["plan"])["request"]["minimum_tip_clearance"],
        int,
    )


def test_service_refuses_normalized_hash_alias_for_legacy_raw_predecessor(raw_predecessor):
    service, sid, raw, typed = raw_predecessor
    before = service.store.get(sid, "owner")
    wrong = request_hash(first_check(typed)["request"])
    with pytest.raises(ValueError, match="predecessor request hash"):
        patch_plan(
            service,
            sid,
            "owner",
            PlanPatch(
                basis=service.store.edit_basis(sid, "owner"),
                subsystems=[amended_subsystem(raw, wrong)],
            ),
        )
    assert service.store.get(sid, "owner") == before


def test_raw_previous_cannot_replace_the_typed_baseline_with_another_criterion(raw_predecessor):
    service, sid, raw, _ = raw_predecessor
    forged = deepcopy(raw)
    first_check(forged)["request"]["required_engagement"] = 3.5
    basis = service.store.edit_basis(sid, "owner").model_copy(
        update={"plan_hash": request_hash(forged)}
    )
    with pytest.raises(ValueError, match="Raw previous plan differs"):
        apply_plan_patch(
            DesignPlan.model_validate(raw),
            PlanPatch(basis=basis, verification=["Unrelated prose"]),
            Contract.model_validate(service.store.get(sid, "owner")["contract"]),
            previous_raw=forged,
        )
