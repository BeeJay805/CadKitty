from __future__ import annotations

import hashlib
import json

import pytest

from forge_mind.llm.client import LLMError
from forge_mind.workspace import author_images
from forge_mind.workspace.author_images import (
    INSPECTION_SCOPE,
    invoke_author_images,
    prepare_author_images,
)
from forge_mind.workspace.store import WorkspaceError


def hashed(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def register(directory, revision, name, data):
    (directory / name).write_bytes(data)
    revision["measurement"]["artifact_hashes"][name] = hashed(data)


def metadata(directory, revision, **changes):
    path = directory / "render-inspection.json"
    value = json.loads(path.read_bytes())
    value.update(changes)
    register(directory, revision, path.name, json.dumps(value).encode())


@pytest.fixture
def revision_files(tmp_path):
    document = {
        "assembly": {
            "parts": [
                {"id": "fixture", "production": {"role": "make"}},
                {"id": "reservation", "production": {"role": "reference"}},
                {"id": "unknown"},
            ]
        }
    }
    revision = {
        "id": "a" * 32,
        "document": document,
        "measurement": {
            "engine_fingerprint": "frozen-engine",
            "artifact_hashes": {},
            "facts": {
                "solid_count": 3,
                "assembly": {
                    "occurrences": [
                        {"path": "base", "definition": "fixture", "body_index": 0},
                        {"path": "keepout", "definition": "reservation", "body_index": 1},
                        {"path": "item", "definition": "unknown", "body_index": 2},
                    ]
                },
            },
        },
    }
    for name, data in {
        "document.json": json.dumps(document).encode(),
        "part.step": b"fixed STEP",
        "render.png": b"complete PNG",
        "render-inspection.png": b"inspection PNG",
    }.items():
        register(tmp_path, revision, name, data)
    hashes = revision["measurement"]["artifact_hashes"]
    register(
        tmp_path,
        revision,
        "render-inspection.json",
        json.dumps(
            {
                "schema_version": 1,
                "revision_id": revision["id"],
                "engine_fingerprint": "frozen-engine",
                "step_sha256": hashes["part.step"],
                "document_sha256": hashes["document.json"],
                "primary": {"file": "render.png", "sha256": hashes["render.png"]},
                "inspection": {
                    "file": "render-inspection.png",
                    "sha256": hashes["render-inspection.png"],
                },
                "selection": {
                    "status": "reference_excluded",
                    "body_count": 3,
                    "included_body_indices": [0, 2],
                    "omitted_body_indices": [1],
                    "omitted_occurrence_ids": ["keepout"],
                    "omitted_count": 1,
                    "diagnostic": "References omitted",
                },
                "render": {
                    "linear_deflection": 0.15,
                    "angular_deflection": 0.35,
                    "frame_bounds_mm": [[0, 0, 0], [10, 10, 10]],
                    "shared_framing": True,
                },
                "scope": INSPECTION_SCOPE,
            }
        ).encode(),
    )
    return tmp_path, revision


def test_fresh_images_keep_complete_first_and_unknown_roles_visible(revision_files):
    images = prepare_author_images(*revision_files, multiple=True)
    assert images.images == (b"complete PNG", b"inspection PNG")
    assert images.mode == "complete_and_inspection"
    assert images.labels == ("Complete view: every body", INSPECTION_SCOPE)
    assert images.records()[0]["sha256"] == hashed(b"complete PNG")
    assert images.context()["supplemental_status"] == "reference_excluded"


@pytest.mark.parametrize(
    "name", ["render-inspection.png", "render-inspection.json", "part.step", "document.json"]
)
@pytest.mark.parametrize("change", ["missing", "unregistered", "bytes"])
def test_stale_optional_binding_retains_complete(revision_files, name, change):
    directory, revision = revision_files
    if change == "missing":
        (directory / name).unlink()
    elif change == "unregistered":
        revision["measurement"]["artifact_hashes"].pop(name)
    else:
        (directory / name).write_bytes(b"different")
    images = prepare_author_images(directory, revision, multiple=True)
    assert images.images == (b"complete PNG",)
    assert images.supplemental_status == "inspection_unavailable_or_stale"


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"schema_version": True},
        {"revision_id": "b" * 32},
        {"engine_fingerprint": "different"},
        {"step_sha256": "0" * 64},
        {"document_sha256": "0" * 64},
        {"scope": "Looks safe"},
        {"primary": {"file": "other.png", "sha256": "0" * 64}},
        {"inspection": None},
        {"render": {"shared_framing": False}},
    ],
)
def test_rehashed_wrong_metadata_cannot_qualify(revision_files, changes):
    metadata(*revision_files, **changes)
    assert len(prepare_author_images(*revision_files, multiple=True).images) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "no_reference"},
        {"status": "all_reference"},
        {"status": "full_context_fallback"},
        {"body_count": 4},
        {"included_body_indices": [0]},
        {"included_body_indices": [False, 2]},
        {"omitted_body_indices": [1, 2]},
        {"omitted_occurrence_ids": ["base"]},
        {"omitted_count": 2},
    ],
)
def test_wrong_cached_body_partition_cannot_qualify(revision_files, changes):
    directory, revision = revision_files
    current = json.loads((directory / "render-inspection.json").read_bytes())
    metadata(directory, revision, selection=current["selection"] | changes)
    assert len(prepare_author_images(*revision_files, multiple=True).images) == 1


def test_duplicate_json_fields_do_not_qualify(revision_files):
    directory, revision = revision_files
    original = (directory / "render-inspection.json").read_bytes()
    register(directory, revision, "render-inspection.json", b'{"schema_version": 2,' + original[1:])
    assert len(prepare_author_images(*revision_files, multiple=True).images) == 1


def test_stored_document_or_body_mapping_cannot_be_substituted(revision_files):
    directory, revision = revision_files
    revision["measurement"]["facts"]["assembly"]["occurrences"][0]["body_index"] = 1
    assert len(prepare_author_images(directory, revision, multiple=True).images) == 1
    revision["document"]["assembly"]["parts"][0]["production"]["role"] = "reference"
    assert len(prepare_author_images(directory, revision, multiple=True).images) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_primary_stale_fails_closed(revision_files, missing):
    directory, revision = revision_files
    if missing:
        (directory / "render.png").unlink()
    else:
        (directory / "render.png").write_bytes(b"changed primary")
    with pytest.raises(WorkspaceError, match="Primary render"):
        prepare_author_images(directory, revision, multiple=True)


def test_provider_fallback_and_explicit_section_keep_their_intent(revision_files):
    directory, revision = revision_files
    assert prepare_author_images(directory, revision).images == (b"complete PNG",)
    section = directory / "section.png"
    section.write_bytes(b"requested section")
    result = prepare_author_images(directory, revision, section=section, multiple=True)
    assert result.images == (b"requested section",)
    assert result.mode == "section"
    assert len(prepare_author_images(directory, revision, multiple=True).images) == 2


def test_section_observation_binds_its_actual_historical_target_and_instrument(revision_files):
    directory, revision = revision_files
    section = directory / "section.png"
    section.write_bytes(b"historical section")
    result = prepare_author_images(
        directory,
        revision,
        section=section,
        section_revision_id="b" * 32,
        section_instrument={"axis": "z", "offset_mm": 7.0},
        multiple=True,
    )
    assert result.images == (b"historical section",)
    assert result.revision_id == "b" * 32
    assert result.context()["instrument"] == {"axis": "z", "offset_mm": 7.0}


@pytest.mark.parametrize(
    "artifact",
    ["render.png", "render-inspection.png", "render-inspection.json", "part.step", "document.json"],
)
def test_actual_hardlinked_artifact_is_rejected_before_read(revision_files, artifact):
    import os

    directory, revision = revision_files
    os.link(directory / artifact, directory / "extra-link")
    if artifact == "render.png":
        with pytest.raises(WorkspaceError, match="Primary render"):
            prepare_author_images(directory, revision, multiple=True)
    else:
        assert prepare_author_images(directory, revision, multiple=True).images == (
            b"complete PNG",
        )


def test_absent_revision_and_unregistered_primary_do_not_promote_supplement(revision_files):
    assert not prepare_author_images(None, None, multiple=True).images
    directory, revision = revision_files
    revision["measurement"]["artifact_hashes"].pop("render.png")
    assert not prepare_author_images(directory, revision, multiple=True).images


class Client:
    def __init__(self, *, fail=False, launched=True, legacy=False):
        self.calls = []
        self.fail = fail
        self.launched = launched
        self.legacy = legacy
        if not legacy:
            self.last_input_images = ()

    def generate_structured_with_images(self, **kwargs):
        self.calls.append(kwargs)
        if not self.legacy:
            self.last_input_images = (
                tuple(
                    {"index": i, "sha256": hashed(data), "bytes": len(data), "label": label}
                    for i, (data, label) in enumerate(
                        zip(kwargs["images_png"], kwargs["image_labels"], strict=True), 1
                    )
                )
                if self.launched
                else ()
            )
        if self.fail:
            raise LLMError("Timed out after possible request delivery; private provider detail")
        return "response", "usage"

    def generate_structured_with_image(self, **kwargs):
        image = kwargs.pop("image_png")
        return self.generate_structured_with_images(
            **kwargs, images_png=(image,), image_labels=("Primary image",)
        )


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("launched", [False, True])
def test_one_invocation_records_exact_submission_even_after_timeout(
    revision_files, multiple, fail, launched
):
    directory, revision = revision_files
    images = prepare_author_images(directory, revision, multiple=multiple)
    client = Client(fail=fail, launched=launched)
    events = []
    root = directory / "observations"
    arguments = {"system": "private system prompt", "user": "private request"}

    def invoke():
        return invoke_author_images(
            client,
            images,
            arguments,
            observation_root=root,
            event=lambda *a, **kw: events.append(kw),
        )

    if fail:
        with pytest.raises(LLMError, match="Timed out"):
            invoke()
    else:
        assert invoke() == ("response", "usage")
    assert len(client.calls) == 1
    records = list(root.glob("*.json"))
    assert len(records) == 1
    content = records[0].read_text()
    receipt = json.loads(content)
    assert receipt["images"] == images.records()
    assert receipt["submission_status"] == (
        "submitted_to_cli" if launched else "submitted_to_client"
    )
    assert receipt["response_status"] == ("failed_or_unknown" if fail else "complete")
    assert events[-1] == receipt
    assert "private" not in content
    assert "reasoning" not in content


def test_legacy_provider_never_claims_cli_submission(revision_files):
    directory, revision = revision_files
    invoke_author_images(
        Client(legacy=True),
        prepare_author_images(directory, revision),
        {},
        observation_root=directory / "obs",
        event=lambda *a, **kw: None,
    )
    receipt = json.loads(next((directory / "obs").glob("*.json")).read_text())
    assert receipt["submission_status"] == "submitted_to_client"


@pytest.mark.parametrize("fail_create", [False, True])
def test_receipt_sharing_failures_are_visible_diagnostics_not_model_retries(
    revision_files, monkeypatch, fail_create
):
    directory, revision = revision_files
    images = prepare_author_images(directory, revision, multiple=True)
    original = author_images._write_receipt

    def sharing(path, record, *, create):
        if create == fail_create:
            raise PermissionError("Windows sharing violation")
        original(path, record, create=create)

    monkeypatch.setattr(author_images, "_write_receipt", sharing)
    events = []
    client = Client()
    assert invoke_author_images(
        client,
        images,
        {},
        observation_root=directory / "obs",
        event=lambda *args, **kw: events.append(args),
    ) == ("response", "usage")
    assert len(client.calls) == 1
    assert any(item[0] == "image_observation_diagnostic" for item in events)
    if not fail_create:
        receipt = json.loads(next((directory / "obs").glob("*.json")).read_text())
        assert receipt["response_status"] == "unknown"
        assert receipt["submission_status"] == "prepared"


def test_receipt_exists_with_unknown_outcome_before_provider_is_called(revision_files):
    directory, revision = revision_files
    root = directory / "obs"

    class CrashAwareClient(Client):
        def generate_structured_with_images(self, **kwargs):
            receipt = json.loads(next(root.glob("*.json")).read_text())
            assert receipt["response_status"] == "unknown"
            assert receipt["submission_status"] == "prepared"
            return super().generate_structured_with_images(**kwargs)

    invoke_author_images(
        CrashAwareClient(),
        prepare_author_images(directory, revision, multiple=True),
        {},
        observation_root=root,
        event=lambda *a, **kw: None,
    )


@pytest.mark.parametrize("name", ["render-inspection.png", "render-inspection.json"])
def test_api_uses_existing_owner_and_artifact_hash_guards(revision_files, monkeypatch, name):
    from fastapi import FastAPI, Header
    from fastapi.testclient import TestClient

    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.api import router
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.store import Store

    directory, revision = revision_files
    store = Store(directory / "store")
    sid = store.create("alice", "fixture", None)["id"]
    store.add_revision(sid, revision)
    monkeypatch.setattr(store, "directory", lambda *_: directory)
    service = WorkspaceService(store, Executor())

    def owner(authorization: str = Header()) -> str:
        return authorization

    app = FastAPI()
    app.include_router(router(store, service, owner))
    client = TestClient(app)
    url = f"/workspace/sessions/{sid}/revisions/{revision['id']}/{name}"
    assert client.get(url, headers={"Authorization": "bob"}).status_code == 404
    result = client.get(url, headers={"Authorization": "alice"})
    assert result.status_code == 200
    assert result.headers["cache-control"] == "private, no-store"
    assert result.content == (directory / name).read_bytes()
    (directory / name).write_bytes(b"changed")
    assert client.get(url, headers={"Authorization": "alice"}).status_code == 409
    revision["measurement"]["artifact_hashes"].pop(name)
    monkeypatch.setattr(store, "revision", lambda *_: revision)
    assert client.get(url, headers={"Authorization": "alice"}).status_code == 409


@pytest.mark.parametrize("multiple", [False, True])
@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("optional_missing,native", [(False, False), (True, False), (True, True)])
def test_actual_author_loop_supplies_ordered_images_once_with_durable_receipt(
    revision_files, monkeypatch, multiple, failure, optional_missing, native
):
    from forge_mind.llm.cost import LLMUsage
    from forge_mind.workspace import agent, completion
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Action, Contract, Measurement
    from forge_mind.workspace.store import Store

    directory, revision = revision_files
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
    sid = store.create(
        "owner",
        "Inspect the fixture",
        contract.model_dump(),
        authoring_mode="native_operations" if native else "hybrid",
    )["id"]
    store.add_revision(sid, revision)
    if optional_missing:
        (directory / "render-inspection.png").unlink()
    store.update(sid, selected_revision=revision["id"], best_revision=revision["id"])
    monkeypatch.setattr(store, "directory", lambda *_: directory)
    calls = []

    class AuthorClient:
        last_input_images = ()

        def generate_structured_with_image(self, **kwargs):
            return self.finish(kwargs, (kwargs["image_png"],))

        def finish(self, kwargs, images):
            calls.append(kwargs)
            self.last_input_images = tuple(
                {"index": i, "sha256": hashed(data), "bytes": len(data)}
                for i, data in enumerate(images, 1)
            )
            if failure:
                raise LLMError("Timed out after submission")
            return Action(
                kind="finish", explanation="Inspection complete; review remains"
            ), LLMUsage("test", 0, 0)

    client = AuthorClient()
    if multiple:
        client.generate_structured_with_images = lambda **kwargs: client.finish(
            kwargs, kwargs["images_png"]
        )
    service = agent.WorkspaceService(store, Executor("trusted_local"), lambda **kw: client)
    monkeypatch.setattr(agent, "engine_fingerprint", lambda: "frozen-engine")
    monkeypatch.setattr(completion, "completion_context", lambda *a: {"premature_finish": False})
    job_id = store.enqueue(sid, "owner", {"max_actions": 2})
    service._run_agent_job(store.claim())
    assert len(calls) == 1
    if multiple and not optional_missing:
        assert calls[0]["images_png"] == (b"complete PNG", b"inspection PNG")
    else:
        assert calls[0]["image_png"] == b"complete PNG"
    context = json.loads(calls[0]["user"])
    if native:
        assert context["plan_reconciliation"]["status"] == "unavailable"
        assert context["plan_reconciliation"]["reason"]
        assert context["delivery_guidance"]["status"] == "unavailable"
        assert "basis_hash" not in context["delivery_guidance"]
        assert "basis" not in context["delivery_guidance"]
        assert "artifact" in context["delivery_guidance"]["reason"].lower()
    assert len(context["image_observation"]["images"]) == (
        2 if multiple and not optional_missing else 1
    )
    if multiple and optional_missing:
        assert (
            context["image_observation"]["supplemental_status"] == "inspection_unavailable_or_stale"
        )
    receipt = json.loads(
        next((store.root / "observations" / sid / job_id).glob("*.json")).read_text()
    )
    assert receipt["submission_status"] == "submitted_to_cli"
    assert receipt["response_status"] == ("failed_or_unknown" if failure else "complete")


@pytest.mark.parametrize(
    "artifact",
    ["render-inspection.png", "render-inspection.json", "render.png", "part.step", "document.json"],
)
def test_author_load_excludes_only_optional_display_artifacts(
    revision_files, monkeypatch, artifact
):
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.executor import Executor
    from forge_mind.workspace.models import Contract, Measurement
    from forge_mind.workspace.store import Store

    directory, revision = revision_files
    register(directory, revision, "program.py", b"retained program")
    register(directory, revision, "part.stl", b"fixed STL")
    revision["measurement"] = Measurement(
        **revision["measurement"],
        revision_id=revision["id"],
        contract_hash=Contract().fingerprint(),
        geometry_valid=True,
    ).model_dump()
    store = Store(directory / "store")
    sid = store.create("owner", "fixture", None)["id"]
    store.add_revision(sid, revision)
    monkeypatch.setattr(store, "directory", lambda *_: directory)
    service = WorkspaceService(store, Executor())
    (directory / artifact).write_bytes(b"changed")
    if artifact.startswith("render-inspection"):
        with pytest.raises(WorkspaceError, match="stale evidence"):
            service._load_revision(sid, revision["id"])
        selected = service._load_revision(sid, revision["id"], allow_optional_display_loss=True)
        result = prepare_author_images(directory, selected, multiple=True)
        assert result.images == (b"complete PNG",)
        assert result.context()["supplemental_status"] == "inspection_unavailable_or_stale"
    else:
        with pytest.raises(WorkspaceError, match="stale evidence"):
            service._load_revision(sid, revision["id"])
