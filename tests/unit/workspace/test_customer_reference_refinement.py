from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest
from PIL import Image
from tests.unit.workspace.test_author_images import register
from tests.unit.workspace.test_author_images import revision_files as revision_files

from forge_mind.workspace.author_images import (
    AuthorImages,
    invoke_author_images,
    prepare_author_images,
    with_customer_reference,
)
from forge_mind.workspace.customer_references import reference_image
from forge_mind.workspace.store import Store, WorkspaceError


def png(color):
    output = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="PNG")
    return output.getvalue()


def owned(tmp_path):
    store = Store(tmp_path / "store")
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "a" * 32,
            "prompt": "Copy the visible profile",
            "reference": {
                "filename": "profile.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(png("blue")).decode(),
            },
        },
    )
    return store, session["id"]


@pytest.mark.parametrize("multiple", [True, False])
def test_two_revisions_retain_source_and_every_required_view_with_real_receipts(tmp_path, multiple):
    store, sid = owned(tmp_path)
    photo, binding = reference_image(store, sid, "owner")
    before = store.session_record(sid, "owner")
    calls = []

    class Client:
        def generate_structured_with_images(self, **kwargs):
            calls.append(kwargs["images_png"])
            return "response"

        def generate_structured_with_image(self, **kwargs):
            calls.append((kwargs["image_png"],))
            return "response"

    for index, color in enumerate(("red", "green"), 1):
        directory = tmp_path / str(index)
        directory.mkdir()
        render = png(color)
        (directory / "render.png").write_bytes(render)
        rid = str(index) * 32
        revision = {
            "id": rid,
            "measurement": {"artifact_hashes": {"render.png": hashlib.sha256(render).hexdigest()}},
        }
        current = prepare_author_images(directory, revision, multiple=True)
        # A requested section is retained as an additional required observation.
        section = png("yellow")
        current = AuthorImages(
            (*current.images, section),
            (*current.labels, "Requested section view"),
            rid,
            current.mode,
            current.supplemental_status,
        )
        images = with_customer_reference(store, sid, "owner", current, multiple=multiple)
        assert images.revision_id == rid
        assert images.source_binding["customer_reference"] == binding
        if multiple:
            assert images.images == (render, section, photo)
        else:
            assert (
                images.images
                == with_customer_reference(store, sid, "owner", current, multiple=False).images
            )
            components = images.source_binding["contact_sheet_components"]
            assert [row["sha256"] for row in components] == [
                hashlib.sha256(data).hexdigest() for data in (render, section, photo)
            ]
            with Image.open(io.BytesIO(images.images[0])) as sheet:
                assert sheet.size == (1024, 2400)
        invoke_author_images(
            Client(),
            images,
            {},
            observation_root=directory / "observations",
            event=lambda *args, **kwargs: None,
        )
        receipt = json.loads(next((directory / "observations").glob("*.json")).read_text())
        assert receipt["source_binding"] == images.source_binding
        assert receipt["images"] == images.records()
    assert len(calls) == 2
    assert store.session_record(sid, "owner") == before


def test_reference_owner_and_original_bytes_are_rechecked_on_each_refinement(tmp_path):
    store, sid = owned(tmp_path)
    images = AuthorImages((png("red"),), ("Complete view",), "b" * 32, "complete", "none")
    with pytest.raises(WorkspaceError, match="not found"):
        with_customer_reference(store, sid, "other", images, multiple=True)
    with_customer_reference(store, sid, "owner", images, multiple=True)
    metadata = store.session_record(sid, "owner")["customer_reference"]
    source = (
        store.root
        / "sessions"
        / sid
        / "source-artifacts"
        / metadata["source_artifact_id"]
        / "source.original"
    )
    source.write_bytes(b"tampered")
    with pytest.raises(WorkspaceError, match="changed"):
        with_customer_reference(store, sid, "owner", images, multiple=True)


def test_image_limit_refuses_instead_of_dropping_required_views(tmp_path):
    store, sid = owned(tmp_path)
    current = AuthorImages((png("red"),) * 6, ("Required view",) * 6, None, "complete", "none")
    with pytest.raises(WorkspaceError, match="envelope"):
        with_customer_reference(store, sid, "owner", current, multiple=True)


def test_legacy_canonical_correction_retains_exact_owned_photo(tmp_path, monkeypatch):
    from tests.unit.workspace.test_source_contract_draft import deliveries
    from tests.unit.workspace.test_source_contract_draft_lifecycle import source_wire

    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.intent_attribution import ContractDraft
    from forge_mind.workspace.measure import ground_contract
    from forge_mind.workspace.models import Requirement
    from forge_mind.workspace.source_delivery_contract_draft import PhotoSourceContractDraftV2

    initial, source = deliveries()
    source = source[:-1] + " for my Ender 3. Its size is 80 mm in local X."
    source += " Use this photo only as context."
    initial.contract.requirements.append(
        Requirement(
            id="width",
            text="Keep the stated width",
            source_quote="80 mm in local X",
            metric="extent_x",
            value=80,
        )
    )
    corrected = initial.model_copy(deep=True)
    corrected.contract.requirements.append(
        Requirement(id="machine", text="Retain the Ender 3 duty", source_quote="for my Ender 3")
    )
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    service = WorkspaceService(Store(tmp_path), Executor("trusted_local"))
    session = service.store.create_customer_project(
        "owner",
        {
            "request_id": "f" * 32,
            "prompt": source,
            "reference": {
                "filename": "profile.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(png("blue")).decode(),
            },
        },
    )
    sid = session["id"]
    calls = []
    initial_wire = source_wire(initial)
    initial_wire.pop("nominal_targets", None)
    initial_wire.pop("photo_nominal_claims", None)
    initial_wire.update(
        photo_reference={"role": "context", "source_quote": "Use this photo only as context."},
        photo_requirements=[],
    )
    responses = [initial_wire, corrected.model_dump(mode="json")]

    class Client:
        model = "scripted-canonical-image-correction"
        timeout_s = None

        def generate_structured(self, **kwargs):
            pytest.fail("Canonical correction dropped its retained source image")

        def generate_structured_with_image(self, **kwargs):
            calls.append(kwargs)
            assert len(calls) <= 2, "Only one correction is authorized"
            return kwargs["output_model"].model_validate(responses[len(calls) - 1]), LLMUsage(
                self.model, 13, 8
            )

    service.client_factory = lambda **kwargs: Client()

    def forbidden(*args, **kwargs):
        pytest.fail("Source correction cannot execute CAD or native workers")

    monkeypatch.setattr(service, "build", forbidden)
    monkeypatch.setattr(service.executor, "build", forbidden)
    monkeypatch.setattr(service.executor, "run_typed_tool", forbidden)
    job = service.store.claim()
    service._run_agent_job({**job, "max_actions": 0, "max_seconds": 100})
    saved = service.store.get(sid, "owner")
    assert len(calls) == 2
    assert [call["output_model"] for call in calls] == [
        PhotoSourceContractDraftV2,
        ContractDraft,
    ]
    assert (
        calls[0]["image_png"]
        == calls[1]["image_png"]
        == reference_image(service.store, sid, "owner")[0]
    )
    assert "Owned visual reference metadata:" in calls[1]["user"]
    assert saved["intent_compilation"]["status"] == "ready"
    assert len(saved["intent_compilation"]["attempts"]) == len(saved["usage"]) == 2
    assert saved["customer_reference"]["known_dimension"] is None
    assert (
        saved["contract"] is None and not saved["revisions"] and not saved.get("accepted_revision")
    )
    grounded = ground_contract(corrected.contract, source, attribution=corrected.attribution)
    assert saved["intent_attribution"]["proposed_contract"] == grounded.model_dump(mode="json")
    assert saved["intent_compilation"]["attempts"][1]["raw_draft"] == corrected.model_dump(
        mode="json"
    )
    assert next(row for row in grounded.requirements if row.id == "width").value == 80


def test_actual_author_loop_keeps_reference_after_current_cad_selection(
    revision_files, monkeypatch
):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace import agent, completion, photo_nominal
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Action, Contract, Measurement

    # This legacy fixture isolates image continuity after CAD selection. Fresh
    # authenticated role admission and mandatory copying have separate tests.
    monkeypatch.setattr(photo_nominal, "photo_reference_role", lambda session: "context")

    directory, revision = revision_files
    for part in revision["document"]["assembly"]["parts"]:
        part.update(
            name=part["id"],
            features=[
                {"id": "body", "name": "Body", "kind": "box", "params": {"x": 1, "y": 1, "z": 1}}
            ],
            output="body",
        )
    revision["document"]["assembly"]["roots"] = [
        {"id": name, "definition": part}
        for name, part in (("base", "fixture"), ("keepout", "reservation"), ("item", "unknown"))
    ]
    register(directory, revision, "document.json", json.dumps(revision["document"]).encode())
    inspection = json.loads((directory / "render-inspection.json").read_bytes())
    inspection["document_sha256"] = revision["measurement"]["artifact_hashes"]["document.json"]
    register(directory, revision, "render-inspection.json", json.dumps(inspection).encode())
    register(directory, revision, "program.py", b"retained program")
    register(directory, revision, "part.stl", b"fixed STL")
    contract = Contract()
    revision["contract"] = contract.model_dump()
    revision["code"] = "retained code"
    revision["measurement"] = Measurement(
        **revision["measurement"],
        revision_id=revision["id"],
        contract_hash=contract.fingerprint(),
        geometry_valid=True,
    ).model_dump()
    store = Store(directory / "store")
    session = store.create_customer_project(
        "owner",
        {
            "request_id": "c" * 32,
            "prompt": "Copy this visible shape",
            "contract": contract.model_dump(),
            "printable": False,
            "reference": {
                "filename": "profile.png",
                "media_type": "image/png",
                "content_base64": base64.b64encode(png("blue")).decode(),
            },
        },
    )
    sid = session["id"]
    store.add_revision(sid, revision)
    store.update(sid, selected_revision=revision["id"], best_revision=revision["id"])
    monkeypatch.setattr(store, "directory", lambda *_: directory)
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "0")
    calls = []

    class Client:
        def generate_structured_with_images(self, **kwargs):
            calls.append(kwargs)
            action = (
                Action(kind="schema", explanation="Read the photo tool", query="photo_profile")
                if len(calls) == 1
                else Action(kind="finish", explanation="Read-only scripted inspection ends")
            )
            return action, LLMUsage("test", 0, 0)

    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kwargs: Client())
    monkeypatch.setattr(agent, "engine_fingerprint", lambda: "frozen-engine")
    monkeypatch.setattr(completion, "completion_context", lambda *args: {"premature_finish": False})
    job = store.claim()
    state = service._run_agent_job({**job, "max_actions": 3})
    assert len(calls) == 2, (state, store.get(sid, "owner")["events"])
    photo, binding = reference_image(store, sid, "owner")
    for call in calls:
        assert call["images_png"] == (b"complete PNG", b"inspection PNG", photo)
        context = json.loads(call["user"])
        assert context["image_observation"]["source_binding"]["customer_reference"] == binding
        assert (
            context["native_authoring"]["discovery"]["photo_profile"]["source"]["vision_sha256"]
            == binding["vision_sha256"]
        )
