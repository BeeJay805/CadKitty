"""Discovery lists exactly the subjects already permitted by frozen admission."""

import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_inventory_verification_admission import target as inventory_contract
from tests.unit.workspace.test_verification_admission import box_contract, pattern_contract

from forge_mind.llm.client import LLMError
from forge_mind.llm.cost import LLMUsage
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Action, Contract, DesignPlan, Requirement
from forge_mind.workspace.native_authoring import native_bootstrap
from forge_mind.workspace.native_review_context import compact_review_context
from forge_mind.workspace.native_stages import construction_context
from forge_mind.workspace.store import Store
from forge_mind.workspace.verification_admission import validate_verification_bindings


def mixed_contract():
    photo = Requirement.model_validate(
        {
            "id": "photo",
            "text": "Recorded nominal photo profile",
            "source_quote": "Make the photo profile",
            "verification": {
                "version": 1,
                "kind": "nominal_photo_solid",
                "subject_id": "photo",
                "claims": ["visible_profile"],
                "size_mm": [40.0, 30.0, 4.0],
                "size_fact_ids": ["width", "height", "depth"],
                "source_quote": "Make the photo profile",
            },
        }
    )
    requirements = [
        *pattern_contract().requirements,
        *box_contract().requirements,
        *inventory_contract().requirements,
        photo,
    ]
    return Contract(
        requirements=[
            requirement.model_copy(update={"id": f"catalog_{index}"})
            for index, requirement in enumerate(requirements)
        ]
    )


def plan(subject):
    return DesignPlan(
        objective="Build the requested geometry",
        verification_bindings=[
            {"subject_id": subject, "part_id": "part", "frame": "definition_local"}
        ],
    )


def test_catalog_matches_all_existing_admitted_subject_families_and_is_detached():
    from forge_mind.workspace.verification_admission import verification_binding_subject_ids

    contract = mixed_contract()
    original = deepcopy(contract.model_dump())
    subjects = verification_binding_subject_ids(contract)
    assert subjects == {"plate", "plates", "spacers", "photo"}
    catalog = native_bootstrap(contract)["discovery"]["verification_subject_catalog"]
    assert catalog["subject_ids"] == sorted(subjects)
    assert catalog["contract_hash"] == contract.fingerprint()
    for subject in catalog["subject_ids"]:
        validate_verification_bindings(plan(subject), contract)
    with pytest.raises(ValueError, match="unknown frozen subject"):
        validate_verification_bindings(plan("attribution_only"), contract)
    catalog["subject_ids"].append("invented")
    subjects.add("also_invented")
    assert native_bootstrap(contract)["discovery"]["verification_subject_catalog"][
        "subject_ids"
    ] == ["photo", "plate", "plates", "spacers"]
    assert contract.model_dump() == original


def test_no_typed_subjects_means_no_binding_despite_scalar_requirements_or_subject_names():
    contract = Contract(
        requirements=[
            Requirement(
                id="tray",
                text="Make the attributed tray",
                source_quote="Make the attributed tray",
                metric="extent_x",
                value=40.0,
            )
        ]
    )
    catalog = native_bootstrap(contract)["discovery"]["verification_subject_catalog"]
    assert catalog["subject_ids"] == []
    validate_verification_bindings(
        DesignPlan(objective="Build without invented bindings"), contract
    )
    with pytest.raises(ValueError, match="unknown frozen subject"):
        validate_verification_bindings(plan("tray"), contract)
    discovery = native_bootstrap(contract)["discovery"]
    assert "attribution" in discovery["requirements"].lower()
    assert "omit verification_bindings" in discovery["construction_assignments"]


def test_default_bootstrap_cannot_advertise_proposed_or_unknown_subjects():
    assert "verification_subject_catalog" not in native_bootstrap()["discovery"]
    assert "verification_subject_catalog" not in native_bootstrap(None)["discovery"]


@pytest.mark.parametrize("projection", [construction_context, compact_review_context])
def test_existing_stage_projections_retain_the_exact_eligible_catalog(projection):
    context = {
        "native_authoring": native_bootstrap(mixed_contract()),
        "authoritative_schemas": {},
        "history": [],
    }
    original = deepcopy(context)
    projected = projection(context)
    assert (
        projected["native_authoring"]["discovery"]["verification_subject_catalog"]
        == original["native_authoring"]["discovery"]["verification_subject_catalog"]
    )
    assert context == original


def test_controller_passes_only_current_frozen_contract_to_native_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "0")
    contract = mixed_contract()
    store = Store(tmp_path)
    sid = store.create(
        "owner", "Continue the saved design", contract.model_dump(), "native_operations"
    )["id"]
    seen = []

    class Client:
        model = "scripted-subject-catalog"

        def generate_structured(self, **kwargs):
            seen.append(json.loads(kwargs["user"]))
            return Action(kind="schema", query="feature:box", explanation="Read syntax"), LLMUsage(
                self.model, 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": 120})
    assert service._run_agent_job(store.claim()) != "error"
    assert (
        seen[0]["native_authoring"]["discovery"]["verification_subject_catalog"]["contract_hash"]
        == contract.fingerprint()
    )


def test_pending_intake_never_passes_proposed_contract_as_frozen_authority(tmp_path, monkeypatch):
    from tests.unit.workspace.test_source_contract_draft import deliveries
    from tests.unit.workspace.test_source_contract_draft_lifecycle import source_wire

    from forge_mind.workspace import native_authoring

    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    draft, source = deliveries()
    store = Store(tmp_path)
    sid = store.create("owner", source, None, "native_operations")["id"]
    observed, calls = [], []
    bootstrap = native_authoring.native_bootstrap

    def record(contract=None):
        observed.append(contract)
        return bootstrap(contract)

    monkeypatch.setattr(native_authoring, "native_bootstrap", record)

    class Client:
        model = "scripted-provisional-catalog"

        def generate_structured(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return kwargs["output_model"].model_validate(source_wire(draft)), LLMUsage(
                    self.model, 0, 0
                )
            context = json.loads(kwargs["user"])
            assert context["frozen_contract"] is None
            assert context["proposed_contract"]
            raise LLMError("Controlled stop after provisional context observation")

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(service, "build", lambda *args, **kwargs: pytest.fail("No CAD may run"))
    store.enqueue(sid, "owner", {"max_actions": 1, "max_seconds": 120})
    service._run_agent_job(store.claim())
    assert len(calls) == 2 and observed == [None]
