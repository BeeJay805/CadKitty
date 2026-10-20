"""Fresh author calls can inspect failed source without selecting invalid CAD."""

import json

import pytest

from forge_mind.llm.cost import LLMUsage
from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.generation_policy import author_model
from forge_mind.workspace.models import Action, Contract, Measurement
from forge_mind.workspace.store import Store


def test_actual_failed_script_is_available_to_next_fresh_author_call(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Make an editable block", Contract().model_dump())["id"]
    broken = (
        "from build123d import Box\nPARAMS={'width':10.0}\nraise ValueError('bad construction')"
    )
    fixed = "from build123d import Box\nPARAMS={'width':10.0}\nresult=Box(PARAMS['width'],8,4)"
    contexts = []

    class Client:
        model = "gpt-6.1-sol"

        def generate_structured(self, **kwargs):
            assert kwargs["model"] == self.model
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind="build", explanation="Controlled failed construction", code=broken
                )
            elif len(contexts) == 2:
                failed = context["failed_attempt"]
                assert failed["code"] == broken
                assert not failed["measurement"]["geometry_valid"]
                assert "bad construction" in str(failed["measurement"]["errors"])
                assert context["selected"] is None
                action = Action(kind="build", explanation="Repair retained source", code=fixed)
            else:
                assert context["selected"]["code"] == fixed
                assert "failed_attempt" not in context
                action = Action(
                    kind="finish", explanation="Geometry measured; physical work unverified"
                )
            return action, LLMUsage(self.model, 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    service._run_agent_job(store.claim())
    assert len(contexts) == 3, store.get(sid, "owner")["events"]
    session = store.get(sid, "owner")
    assert session["author_model_requested"] == "gpt-6.1-sol"
    assert session["author_model_reported"] == "gpt-6.1-sol"
    assert not session.get("accepted_revision")
    failed = store.revisions(sid)[0]
    assert service._load_revision(sid, failed["id"])["code"] == broken


@pytest.mark.parametrize("configured", ["gpt-6.1-sol", "gpt-6-astra"])
def test_model_setting_applies_to_injected_clients_without_fixed_model(monkeypatch, configured):
    monkeypatch.setenv("FORGE_CODEX_MODEL", configured)
    assert author_model(object()) == configured


def test_fixed_client_model_takes_priority_over_environment(monkeypatch):
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-6-astra")

    class Client:
        model = "gpt-6.1-sol"

    assert author_model(Client()) == "gpt-6.1-sol"


def test_tampered_failed_source_is_reported_unavailable_without_becoming_selected(tmp_path):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Repair retained CAD", contract.model_dump())["id"]
    rid = "a" * 32
    directory = store.directory(sid, rid)
    directory.mkdir(parents=True)
    source = directory / "program.py"
    source.write_text("raise ValueError('original failure')", encoding="utf-8")
    measurement = Measurement(
        revision_id=rid,
        contract_hash=contract.fingerprint(),
        engine_fingerprint="test",
        artifact_hashes={"program.py": file_sha256(source)},
        geometry_valid=False,
        errors=["original failure"],
    )
    store.add_revision(
        sid,
        {
            "id": rid,
            "code": source.read_text(),
            "measurement": measurement.model_dump(),
            "contract": contract.model_dump(),
        },
    )
    source.write_text("changed after measurement", encoding="utf-8")
    seen = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            seen.append(context)
            assert context["selected"] is None
            failed = context["failed_attempt"]
            assert failed["id"] == rid
            assert "stale evidence" in failed["unavailable"]
            assert "code" not in failed and "measurement" not in failed
            return Action(kind="finish", explanation="Retained source integrity failed"), LLMUsage(
                "test", 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    store.enqueue(sid, "owner", {"max_actions": 1})
    assert service._run_agent_job(store.claim()) == "needs_review"
    assert len(seen) == 1
    assert not store.get(sid, "owner").get("selected_revision")


@pytest.mark.parametrize("mismatch", ["parent", "contract"])
def test_failed_attempt_from_another_branch_or_contract_does_not_drive_repair(
    tmp_path, monkeypatch, mismatch
):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Continue this candidate", contract.model_dump())["id"]
    current, failed = "a" * 32, "b" * 32
    for rid, valid in ((current, True), (failed, False)):
        store.add_revision(
            sid,
            {
                "id": rid,
                "parent_id": "c" * 32 if not valid and mismatch == "parent" else current,
                "code": "retained source",
                "measurement": Measurement(
                    revision_id=rid,
                    contract_hash="old contract"
                    if not valid and mismatch == "contract"
                    else contract.fingerprint(),
                    engine_fingerprint="test",
                    artifact_hashes={},
                    geometry_valid=valid,
                ).model_dump(),
            },
        )
    store.update(sid, selected_revision=current)
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            assert context["selected"]["id"] == current
            assert "failed_attempt" not in context
            assert "failed_attempt_revision_id" not in context["repair_focus"]
            return Action(kind="finish", explanation="Selected branch retained"), LLMUsage(
                "test", 0, 0
            )

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda sid, rid, **kw: store.revision(sid, rid))
    store.enqueue(sid, "owner", {"max_actions": 1, "revision_id": current})
    assert service._run_agent_job(store.claim()) == "needs_review"
    assert len(contexts) == 1
    assert store.get(sid, "owner")["selected_revision"] == current


@pytest.mark.parametrize("offset_mm", [1000, 0])
def test_real_section_preserves_viewer_fields_and_continues_author_job(tmp_path, offset_mm):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Inspect this block", contract.model_dump())["id"]
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind="section", explanation="Check the plane", axis="z", offset_mm=offset_mm
                )
            else:
                section = next(row["section"] for row in context["history"] if "section" in row)
                if offset_mm:
                    assert section["loops"] == 0
                    assert section["message"] == "Plane does not intersect geometry"
                    assert "image" not in section
                    assert context["image_observation"]["mode"] != "section"
                else:
                    assert section["loops"] > 0
                    assert section["image"]
                    assert context["image_observation"]["mode"] == "section"
                action = Action(
                    kind="finish", explanation="Empty section; original CAD remains usable"
                )
            return action, LLMUsage("test", 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    original = service.build(
        sid,
        "from build123d import Box\nPARAMS={'width':10.0}\nresult=Box(PARAMS['width'],8,4)",
        contract,
        None,
        "Actual exported block",
        60,
        lambda: False,
    )
    assert original["measurement"]["geometry_valid"]
    store.update(sid, selected_revision=original["id"])
    store.enqueue(sid, "owner", {"max_actions": 2, "max_seconds": 60})
    assert service._run_agent_job(store.claim()) in {"needs_review", "contract_satisfied"}
    assert len(contexts) == 2
    session = store.get(sid, "owner")
    event = next(event for event in session["events"] if event["kind"] == "section")
    assert event["message"] == "Cross-section computed"
    assert event["axis"] == event["section"]["axis"] == "z"
    assert event["offset_mm"] == event["section"]["offset_mm"] == offset_mm
    assert event["loops"] == event["section"]["loops"]
    # Match the existing CadWorkspace viewer's event selection contract.
    visible_sections = [
        row
        for row in session["events"]
        if row["kind"] == "section"
        and row.get("image")
        and row.get("revision_id") == original["id"]
    ]
    if offset_mm:
        assert event["section"]["message"] == "Plane does not intersect geometry"
        assert visible_sections == []
    else:
        assert visible_sections == [event]
        assert event["image"] == event["section"]["image"]
        assert (store.root / "sections" / sid / event["image"]).is_file()
    assert session["selected_revision"] == original["id"]
    assert service._load_revision(sid, original["id"])["measurement"] == original["measurement"]


def test_actual_typed_validation_failure_can_be_repaired_by_a_stateless_author(tmp_path):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Make an editable block", contract.model_dump())["id"]
    invalid = json.dumps(
        {
            "name": "Retain this exact construction",
            "features": [
                {
                    "id": "body",
                    "name": "Body",
                    "kind": "not_real_kind",
                    "params": {"x": 10, "y": 8, "z": 4},
                }
            ],
            "outputs": ["body"],
        },
        indent=2,
    )
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind="document", explanation="Rejected typed input", document_json=invalid
                )
            elif len(contexts) == 2:
                failure = context["failed_submission"]
                assert failure["submitted_action"]["document_json"] == invalid
                assert failure["based_on_revision"] is None
                assert failure["contract_hash"] == contract.fingerprint()
                assert failure["status"] == "validation_failed"
                assert not any(
                    failure[key]
                    for key in ("geometry_produced", "evidence_produced", "revision_created")
                )
                assert "not_real_kind" in failure["validation_error"]
                assert (
                    context["repair_focus"]["input_validation_error"] == failure["validation_error"]
                )
                assert context["selected"] is None
                assert store.revisions(sid) == []
                assert "failed_attempt" not in context
                repaired = json.loads(failure["submitted_action"]["document_json"])
                repaired["features"][0]["kind"] = "box"
                action = Action(
                    kind="document",
                    explanation="Repair the exact input",
                    document_json=json.dumps(repaired),
                )
            else:
                assert "failed_submission" not in context
                assert context["selected"]["document"]["name"] == "Retain this exact construction"
                assert context["selected"]["measurement"]["geometry_valid"]
                action = Action(kind="finish", explanation="Measured editable block")
            return action, LLMUsage("test", 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    store.enqueue(sid, "owner", {"max_actions": 3, "max_seconds": 120})
    assert service._run_agent_job(store.claim()) in {"needs_review", "contract_satisfied"}
    assert len(contexts) == 3, store.get(sid, "owner")["events"]
    revisions = store.revisions(sid)
    assert len(revisions) == 1
    assert revisions[0]["measurement"]["geometry_valid"]
    assert (store.directory(sid, revisions[0]["id"]) / "document.json").is_file()
    assert store.get(sid, "owner")["failed_submission"] is None


@pytest.mark.parametrize("oversized", [False, True])
def test_rejected_typed_input_survives_next_job_with_explicit_retention_limit(tmp_path, oversized):
    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Repair this editable document", contract.model_dump())["id"]
    invalid = json.dumps(
        {
            "name": "x" * 40000 if oversized else "Exact original",
            "features": [{"id": "body", "name": "Body", "kind": "not_real_kind"}],
        }
    )
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind="document", explanation="Rejected input", document_json=invalid
                )
            else:
                failure = context["failed_submission"]
                assert failure == store.get(sid, "owner")["failed_submission"]
                if oversized:
                    assert "submitted_action" not in failure
                    assert "source was not retained" in failure["unavailable"]
                    assert invalid not in kwargs["user"]
                else:
                    assert failure["submitted_action"]["document_json"] == invalid
                assert context["selected"] is None and store.revisions(sid) == []
                action = Action(kind="finish", explanation="No geometry exists")
            return action, LLMUsage("test", 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    for _ in range(2):
        store.enqueue(sid, "owner", {"max_actions": 1})
        job = store.claim()
        state = service._run_agent_job(job)
        store.finish(job["id"], sid, state)
    assert len(contexts) == 2
    assert store.revisions(sid) == []


def test_corrected_plan_clears_rejected_input_but_schema_read_keeps_it(tmp_path):
    store = Store(tmp_path)
    sid = store.create("owner", "Make a useful block", Contract().model_dump())["id"]
    invalid = '{"objective": ""}'
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(kind="plan", explanation="Invalid plan", document_json=invalid)
            elif len(contexts) == 2:
                assert context["failed_submission"]["submitted_action"]["document_json"] == invalid
                action = Action(kind="schema", explanation="Read valid plan fields", query="plan")
            elif len(contexts) == 3:
                assert context["failed_submission"]["submitted_action"]["document_json"] == invalid
                action = Action(
                    kind="plan",
                    explanation="Correct the plan",
                    document_json='{"objective": "Make a useful block"}',
                )
            else:
                assert "failed_submission" not in context
                assert context["design_plan"]["objective"] == "Make a useful block"
                action = Action(kind="finish", explanation="No geometry exists yet")
            return action, LLMUsage("test", 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    store.enqueue(sid, "owner", {"max_actions": 4})
    service._run_agent_job(store.claim())
    assert len(contexts) == 4, store.get(sid, "owner")["events"]
    assert store.get(sid, "owner")["failed_submission"] is None
    assert store.revisions(sid) == []


@pytest.mark.parametrize("kind", ["assembly_edit", "edit_part"])
def test_rejected_scoped_edit_is_available_to_next_stateless_call_without_building(
    tmp_path, monkeypatch, kind
):
    from forge_mind.workspace.document import DesignDocument

    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Repair the scoped assembly edit", contract.model_dump())["id"]
    rid = "a" * 32
    document = DesignDocument.model_validate(
        {
            "name": "Existing assembly",
            "assembly": {
                "parts": [
                    {
                        "id": "body",
                        "name": "Body",
                        "features": [
                            {
                                "id": "block",
                                "name": "Block",
                                "kind": "box",
                                "params": {"x": 10, "y": 8, "z": 4},
                            }
                        ],
                        "output": "block",
                    }
                ],
                "roots": [{"id": "body_instance", "definition": "body"}],
            },
        }
    )
    store.add_revision(
        sid,
        {
            "id": rid,
            "code": "",
            "document": document.model_dump(),
            "measurement": Measurement(
                revision_id=rid,
                contract_hash=contract.fingerprint(),
                engine_fingerprint="test",
                geometry_valid=True,
                artifact_hashes={},
            ).model_dump(),
        },
    )
    store.update(sid, selected_revision=rid)
    invalid = '{"unrecognized_field": "exact failed scoped construction"}'
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind=kind,
                    part_id="body",
                    explanation="Rejected scoped input",
                    document_json=invalid,
                )
            else:
                failure = context["failed_submission"]
                assert failure["submitted_action"]["kind"] == kind
                assert failure["submitted_action"]["document_json"] == invalid
                assert failure["submitted_action"]["part_id"] == "body"
                assert failure["based_on_revision"] == rid
                assert failure["contract_hash"] == contract.fingerprint()
                assert not any(
                    failure[key]
                    for key in ("geometry_produced", "evidence_produced", "revision_created")
                )
                assert context["selected"]["id"] == rid
                assert "failed_attempt" not in context
                assert [row["id"] for row in store.revisions(sid)] == [rid]
                action = Action(
                    kind="finish", explanation="Invalid edit did not change the candidate"
                )
            return action, LLMUsage("test", 0, 0)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda sid, rid, **kw: store.revision(sid, rid))
    store.enqueue(sid, "owner", {"max_actions": 2})
    service._run_agent_job(store.claim())
    assert len(contexts) == 2, store.get(sid, "owner")["events"]
    assert store.get(sid, "owner")["selected_revision"] == rid
    assert store.revision(sid, rid)["document"] == document.model_dump(mode="json")


@pytest.mark.parametrize("rejection", ["assembly_root", "part_id", "part_port"])
def test_semantic_scoped_rejection_retains_exact_action_until_stateless_repair(
    tmp_path, monkeypatch, rejection
):
    """Exercise the real transaction validator, with a controlled build boundary."""
    from forge_mind.workspace import document_service
    from forge_mind.workspace.document import DesignDocument

    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Repair this scoped edit", contract.model_dump())["id"]
    body = {
        "id": "body",
        "name": "Body",
        "features": [
            {"id": "block", "name": "Block", "kind": "box", "params": {"x": 10, "y": 8, "z": 4}}
        ],
        "output": "block",
        "ports": [{"name": "shaft_axis", "kind": "shaft", "diameter": 2}],
    }
    sibling = {**body, "id": "sibling", "name": "Unchanged sibling"}
    document = DesignDocument.model_validate(
        {
            "name": "Retained assembly",
            "assembly": {
                "parts": [body, sibling],
                "roots": [
                    {"id": "body_instance", "definition": "body"},
                    {"id": "sibling_instance", "definition": "sibling"},
                ],
            },
        }
    )
    document.assembly.parts[0].production.role = "make"
    document.assembly.parts[0].production.process = "additive"
    initial = None if rejection == "assembly_root" else "a" * 32

    def revision(rid, doc, parent):
        return {
            "id": rid,
            "parent_id": parent,
            "code": "",
            "document": doc.model_dump(mode="json"),
            "measurement": Measurement(
                revision_id=rid,
                contract_hash=contract.fingerprint(),
                engine_fingerprint="unit transaction control",
                geometry_valid=True,
                artifact_hashes={},
            ).model_dump(),
        }

    if initial:
        store.add_revision(sid, revision(initial, document, None))
        store.update(sid, selected_revision=initial)
    if rejection == "assembly_root":
        payload = {"name": document.name, **document.assembly.model_dump(mode="json")}
        payload["roots"][0]["definition"] = "mistyped_body"
        kind = "assembly_edit"
    else:
        payload = json.loads(json.dumps(body))
        payload["features"][0]["params"]["x"] = 12
        if rejection == "part_id":
            payload["id"] = "mistyped_body"
        else:
            payload["ports"] = []
        kind = "edit_part"
    submitted = Action(
        kind=kind,
        part_id="body",
        explanation="Retain exact rejected scoped action",
        document_json=json.dumps(payload, indent=2),
    )
    builds = []

    def controlled_build(service, sid, owner, doc, contract, parent, *args):
        builds.append(doc.model_dump(mode="json"))
        assert service.store.get(sid, owner)["failed_submission"] is None
        row = revision("b" * 32, doc, parent)
        store.add_revision(sid, row)
        return row

    monkeypatch.setattr(document_service, "build_document", controlled_build)
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = submitted
            elif len(contexts) in {2, 3}:
                failure = context["failed_submission"]
                assert failure["submitted_action"] == submitted.model_dump(mode="json")
                assert failure["based_on_revision"] == initial
                assert failure["contract_hash"] == contract.fingerprint()
                assert not any(
                    failure[key]
                    for key in ("geometry_produced", "evidence_produced", "revision_created")
                )
                assert builds == []
                assert len(store.revisions(sid)) == (1 if initial else 0)
                if len(contexts) == 2:
                    action = Action(kind="schema", explanation="Read scoped fields", query=kind)
                else:
                    action = Action.model_validate(failure["submitted_action"])
                    repaired = json.loads(action.document_json)
                    if rejection == "assembly_root":
                        repaired["roots"][0]["definition"] = "body"
                    elif rejection == "part_id":
                        repaired["id"] = "body"
                    else:
                        repaired["ports"] = body["ports"]
                    action = action.model_copy(update={"document_json": json.dumps(repaired)})
            else:
                assert "failed_submission" not in context
                assert len(builds) == 1
                assert context["selected"]["id"] == "b" * 32
                selected = builds[0]
                assert selected["assembly"]["parts"][1] == document.assembly.parts[1].model_dump(
                    mode="json"
                )
                assert (
                    selected["assembly"]["roots"]
                    == document.assembly.model_dump(mode="json")["roots"]
                )
                repaired_part = selected["assembly"]["parts"][0]
                assert repaired_part["id"] == "body"
                assert (
                    repaired_part["ports"]
                    == document.assembly.parts[0].model_dump(mode="json")["ports"]
                )
                assert repaired_part["production"] == document.assembly.parts[
                    0
                ].production.model_dump(mode="json")
                if initial:
                    assert repaired_part["features"][0]["params"]["x"] == 12
                action = Action(kind="finish", explanation="Transaction control completed")
            return action, LLMUsage("test", 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda sid, rid, **kw: store.revision(sid, rid))
    store.enqueue(sid, "owner", {"max_actions": 4})
    assert service._run_agent_job(store.claim()) == "needs_review", store.get(sid, "owner")[
        "events"
    ]
    assert len(contexts) == 4, store.get(sid, "owner")["events"]
    assert len(builds) == 1
    assert len(store.revisions(sid)) == (2 if initial else 1)
    assert store.get(sid, "owner")["failed_submission"] is None


@pytest.mark.parametrize("verified", [None, False, True])
def test_author_guidance_uses_actual_executor_qualification(tmp_path, verified):
    store = Store(tmp_path)
    sid = store.create("owner", "Make an editable block", Contract().model_dump())["id"]

    class QualifiedExecutor(Executor):
        @property
        def typed_container_verified(self):
            return verified

    class LegacyExecutor:
        mode = "docker"

    executor = LegacyExecutor() if verified is None else QualifiedExecutor("docker")
    systems = []

    class Client:
        def generate_structured(self, **kwargs):
            systems.append(kwargs["system"])
            return Action(
                kind="finish", explanation="No build in this capability control"
            ), LLMUsage("test", 0, 0)

    service = WorkspaceService(store, executor, lambda **kw: Client())
    store.enqueue(sid, "owner", {"max_actions": 1})
    assert service._run_agent_job(store.claim()) == "needs_review"
    assert len(systems) == 1
    assert ("Typed document actions are unavailable" in systems[0]) == (verified is not True)


@pytest.mark.parametrize("kind", ["assembly_edit", "edit_part"])
def test_postbuild_measurement_error_is_not_retained_as_rejected_scoped_input(
    tmp_path, monkeypatch, kind
):
    from forge_mind.workspace import document_service
    from forge_mind.workspace.document import DesignDocument

    store = Store(tmp_path)
    contract = Contract()
    sid = store.create("owner", "Edit this assembly", contract.model_dump())["id"]
    rid = "a" * 32
    body = {
        "id": "body",
        "name": "Body",
        "features": [
            {"id": "block", "name": "Block", "kind": "box", "params": {"x": 10, "y": 8, "z": 4}}
        ],
        "output": "block",
    }
    document = DesignDocument.model_validate(
        {
            "name": "Retained assembly",
            "assembly": {
                "parts": [body],
                "roots": [{"id": "body_instance", "definition": "body"}],
            },
        }
    )
    store.add_revision(
        sid,
        {
            "id": rid,
            "code": "",
            "document": document.model_dump(mode="json"),
            "measurement": Measurement(
                revision_id=rid,
                contract_hash=contract.fingerprint(),
                engine_fingerprint="unit boundary control",
                geometry_valid=True,
                artifact_hashes={},
            ).model_dump(),
        },
    )
    store.update(sid, selected_revision=rid)
    builds = []

    def malformed_build(*args):
        builds.append(args[3])
        return {"id": "b" * 32, "measurement": {"unexpected": "invalid output"}}

    monkeypatch.setattr(document_service, "build_document", malformed_build)
    contexts = []

    class Client:
        def generate_structured(self, **kwargs):
            context = json.loads(kwargs["user"])
            contexts.append(context)
            if len(contexts) == 1:
                action = Action(
                    kind=kind,
                    part_id="body",
                    explanation="Valid scoped transaction",
                    document_json=json.dumps(
                        {"parts": [body]} if kind == "assembly_edit" else body
                    ),
                )
            else:
                assert "failed_submission" not in context
                assert len(builds) == 1
                assert "validation error" in str(context["history"])
                action = Action(kind="finish", explanation="Postbuild evidence remains unresolved")
            return action, LLMUsage("test", 0, 0)

        def generate_structured_with_image(self, **kwargs):
            return self.generate_structured(**kwargs)

    service = WorkspaceService(store, Executor("trusted_local"), lambda **kw: Client())
    monkeypatch.setattr(service, "_load_revision", lambda sid, rid, **kw: store.revision(sid, rid))
    store.enqueue(sid, "owner", {"max_actions": 2})
    assert service._run_agent_job(store.claim()) == "needs_review", store.get(sid, "owner")[
        "events"
    ]
    assert len(contexts) == 2
    assert store.get(sid, "owner")["selected_revision"] == rid
    assert store.get(sid, "owner")["failed_submission"] is None
