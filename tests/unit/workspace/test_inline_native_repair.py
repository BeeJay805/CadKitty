"""Real STEP misfit measured and repaired through the actual author job loop."""

import json

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, DesignPlan, Requirement
from forge_mind.workspace.product_service import read_tool_receipt
from forge_mind.workspace.store import Store


def test_real_misfit_reaches_author_then_repair_is_measured_before_finish(tmp_path):
    store = Store(tmp_path)
    contract = Contract(
        requirements=[
            Requirement(
                id="gap",
                text="80 mm gap and verified real fit",
                source_quote="80 mm gap and verified real fit",
            )
        ]
    )
    sid = store.create("owner", "Repair this fixture gap", contract.model_dump())["id"]
    seen = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            check = context["early_checks"]["subsystems"][0]["checks"][0]
            seen.append(check)
            if len(seen) == 1:
                assert check["status"] == "fail"
                assert check["result"]["results"]["ordered_separation_mm"] == pytest.approx(79)
                assert context["completion"]["next_target"] == "jaws"
                action = Action(
                    kind="parameters",
                    explanation="Repair measured 1 mm gap error",
                    parameters={"right_center": 85},
                )
            else:
                assert len(seen) == 2
                assert check["status"] == "pass"
                assert check["result"]["results"]["ordered_separation_mm"] == pytest.approx(80)
                action = Action(
                    kind="finish",
                    explanation="Nominal gap checked; physical fit remains unresolved",
                )
            return action, LLMUsage("test", 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    document = DesignDocument.model_validate(
        {
            "name": "Controlled early repair",
            "variables": {"right_center": 84},
            "assembly": {
                "parts": [
                    {
                        "id": "jaw",
                        "name": "Jaw",
                        "features": [
                            {
                                "id": "body",
                                "name": "Body",
                                "kind": "box",
                                "params": {"x": 10, "y": 8, "z": 6},
                            }
                        ],
                        "output": "body",
                    }
                ],
                "roots": [
                    {"id": "left", "definition": "jaw", "position": [-5, 0, 0]},
                    {"id": "right", "definition": "jaw", "position": ["right_center", 0, 0]},
                ],
            },
        }
    )
    initial = build_document(
        service, sid, "owner", document, contract, None, "Deliberate 79 mm gap", 120, lambda: False
    )
    assert initial["measurement"]["geometry_valid"], initial["measurement"]["errors"]
    plan = DesignPlan.model_validate(
        {
            "objective": "Repair nominal gap while retaining physical obligations",
            "subsystems": [
                {
                    "id": "jaws",
                    "name": "Jaws",
                    "part_ids": ["jaw"],
                    "requirement_ids": ["gap"],
                    "unresolved": ["Physical fit and process repeatability"],
                    "checks": [
                        {
                            "id": "gap_check",
                            "request": {
                                "tool": "projected_pair",
                                "units": "mm",
                                "a": {"occurrence_path": "left"},
                                "b": {"occurrence_path": "right"},
                                "axis": [1, 0, 0],
                                "datum": [0, 0, 0],
                                "expected_separation_mm": 80,
                                "tolerance_mm": 0.00001,
                            },
                        }
                    ],
                }
            ],
        }
    )
    store.update(
        sid,
        selected_revision=initial["id"],
        accepted_revision=None,
        best_revision=initial["id"],
        design_plan=plan.model_dump(),
        plan_contract_hash=contract.fingerprint(),
    )
    store.enqueue(sid, "owner", {"max_seconds": 180, "max_actions": 3, "preview_only": True})
    service.run_job(store.claim())
    session = store.get(sid, "owner")
    assert len(seen) == 2, [(e["kind"], e.get("message")) for e in session["events"]]
    assert seen[0]["receipt_id"] != seen[1]["receipt_id"]
    for check in seen:
        receipt = read_tool_receipt(service, sid, "owner", check["receipt_id"])
        assert receipt["result"]["status"] == check["status"]
        assert receipt["contract"] == contract.model_dump()
        assert receipt["result"]["qualification_status"] == "unverified"
    assert session.get("accepted_revision") is None
    assert session["contract"] == contract.model_dump()
    assert session["completion_report"]["status"] == "unverified"
    assert session["completion_report"]["subsystems"][0]["checks"][0]["reused"]
