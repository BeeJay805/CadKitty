"""Saved guidance through actual SDK transports; fixture CAD bytes are never parsed."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import shutil
import time
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile

import httpx
import httpx2
import pytest
from fastapi import FastAPI
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace.api import router
from forge_mind.workspace.engineering_api import engineering_router
from forge_mind.workspace.handoff import json_bytes
from forge_mind.workspace.mcp_server import WorkspaceAdapter
from forge_mind.workspace.mcp_server import create_server as local_server
from forge_mind.workspace.product_api import product_router
from forge_mind.workspace.project_access import project_access_router
from forge_mind.workspace.remote_mcp import create_app as remote_app

ISSUER = "https://guidance.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
LEGACY = "c" * 32
GUIDANCE = {
    "print": "Place the declared mounting face on the bed.\nPrototype choice: PLA ü.\n```text\nDo not execute <script>.\n```",
    "installation": "Use the existing bolt part and quantity in the purchasing list.\nWall substrate and anchor specification remain unknown.",
    "use_and_removal": "Support the device, then lift it out.\nCheck the actual removal path before use.",
    "unresolved": "Printing, physical fit, loads and anchors remain untested.",
}
PLAN_ONLY = "CURRENT PLAN ONLY: never substitute these steps for saved customer instructions"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_record(service, sid, rid, record):
    with service.store.connect() as con:
        con.execute(
            "UPDATE revisions SET document=? WHERE session=? AND id=?",
            (json.dumps(record), sid, rid),
        )


def fixture(tmp_path, monkeypatch):
    service, sid, directory = saved_project(tmp_path, monkeypatch)
    record = service.store.revision(sid, RID)
    raw = json.loads((directory / "document.json").read_bytes())
    raw["delivery_guidance"] = deepcopy(GUIDANCE)
    # Preserve literal saved source, including omissions, rather than expanding it.
    document_bytes = json_bytes(raw)
    (directory / "document.json").write_bytes(document_bytes)
    manifest = json.loads((directory / "definition-meshes.json").read_bytes())
    manifest["document_sha256"] = sha(document_bytes)
    (directory / "definition-meshes.json").write_bytes(json_bytes(manifest))
    record["document"] = raw
    record["measurement"]["artifact_hashes"].update(
        {
            name: sha((directory / name).read_bytes())
            for name in ("document.json", "definition-meshes.json")
        }
    )
    save_record(service, sid, RID, record)
    legacy_directory = service.store.directory(sid, LEGACY)
    shutil.copytree(directory, legacy_directory)
    legacy = deepcopy(record)
    legacy["id"] = LEGACY
    legacy["based_on"] = RID
    legacy["document"].pop("delivery_guidance")
    legacy["measurement"]["revision_id"] = LEGACY
    (legacy_directory / "document.json").write_bytes(json_bytes(legacy["document"]))
    manifest["revision_id"] = LEGACY
    manifest["document_sha256"] = sha((legacy_directory / "document.json").read_bytes())
    (legacy_directory / "definition-meshes.json").write_bytes(json_bytes(manifest))
    legacy["measurement"]["artifact_hashes"].update(
        {
            name: sha((legacy_directory / name).read_bytes())
            for name in ("document.json", "definition-meshes.json")
        }
    )
    service.store.add_revision(sid, legacy)
    for rid, saved in ((RID, record), (LEGACY, legacy)):
        source = service.store.directory(sid, rid)
        (source / "contract.json").write_bytes(json_bytes(saved["contract"]))
        (source / "evidence.json").write_bytes(json_bytes(saved["measurement"]))
    service.store.update(
        sid,
        selected_revision=LEGACY,
        design_plan={
            "objective": "A newer planning context",
            "assembly_sequence": [PLAN_ONLY],
            "unresolved": [PLAN_ONLY],
        },
    )
    return service, sid


class Verifier:
    def __init__(self, *, write=False):
        self.write = write

    async def verify_token(self, token):
        if token not in {"owner", "foreign"}:
            return None
        return AccessToken(
            token=token,
            client_id="guidance-sdk-test",
            subject=token,
            scopes=["cad:read"] + (["cad:write"] if self.write else []),
            resource=RESOURCE,
            expires_at=int(time.time()) + 600,
            claims={"iss": ISSUER},
        )


@asynccontextmanager
async def connection(service, tmp_path, surface, *, foreign=False, write=False):
    if surface == "local":
        app = FastAPI()

        def owner():
            return "foreign" if foreign else "owner"

        app.include_router(router(service.store, service, owner))
        app.include_router(engineering_router(service.store, service, owner))
        app.include_router(product_router(service.store, service, owner))
        app.include_router(project_access_router(service.store, service, owner))
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.ASGITransport(app=app)
        )
        async with Client(local_server(adapter)) as sdk:
            yield sdk
    else:
        app = remote_app(
            service, token_verifier=Verifier(write=write), issuer_url=ISSUER, resource_url=RESOURCE
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                trust_env=False,
                headers={"Authorization": "Bearer " + ("foreign" if foreign else "owner")},
            ) as http,
            Client(streamable_http_client(RESOURCE, http_client=http), mode="legacy") as sdk,
        ):
            yield sdk
    assert service.worker is None
    assert not service.executor.typed_container_verified


def snapshot(service, sid):
    with service.store.connect() as con:
        rows = {
            table: con.execute(f"SELECT * FROM {table}").fetchall()
            for table in ("sessions", "revisions", "jobs")
        }
    sources = {
        str(p.relative_to(service.store.directory(sid, RID).parent)): sha(p.read_bytes())
        for p in service.store.directory(sid, RID).parent.rglob("*")
        if p.is_file()
    }
    return rows, sources


async def delivered_zip(sdk, response, surface):
    assert not response.is_error
    receipt = response.structured_content
    if surface == "local":
        content = Path(receipt["path"]).read_bytes()
    else:
        result = await sdk.read_resource(receipt["uri"])
        assert result.contents[0].mime_type == "application/zip"
        content = base64.b64decode(result.contents[0].blob, validate=True)
    assert len(content) == receipt["bytes"] and sha(content) == receipt["sha256"]
    assert receipt["manufacturing_ready"] is False
    assert receipt["physical_readiness"] == "unverified"
    return receipt, content


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_sdk_historical_guidance_and_legacy_export_exact_saved_bytes(
    tmp_path, monkeypatch, surface
):
    service, sid = fixture(tmp_path, monkeypatch)
    before = snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            for rid in (RID, LEGACY):
                source = service.store.directory(sid, rid)
                response = await sdk.call_tool(
                    "get_revision_source", {"project_id": sid, "revision_id": rid}
                )
                assert not response.is_error
                saved = json.loads((source / "document.json").read_bytes())
                assert response.structured_content["document"] == saved
                assert response.structured_content["document_sha256"] == sha(
                    (source / "document.json").read_bytes()
                )
                assert (saved.get("delivery_guidance") == GUIDANCE) is (rid == RID)
                if rid == LEGACY:
                    assert "delivery_guidance" not in saved
                response = await sdk.call_tool(
                    "export_project",
                    {"project_id": sid, "revision_id": rid, "include_stl": True, "preview": "none"},
                )
                receipt, content = await delivered_zip(sdk, response, surface)
                assert receipt["revision_id"] == rid
                assert receipt["guidance_status"] == "unbound"
                with ZipFile(io.BytesIO(content)) as archive:
                    sidecar = json.loads(archive.read("customer-guidance.json"))
                    assert sidecar["status"] == "unbound"
                    assert sidecar["basis"]["revision_id"] == rid
                    assert sidecar["instruction_binding"]["status"] == "legacy_unverified"
                    assert sidecar["instruction_binding"]["steps"] == []
                    assert archive.read("document.json") == (source / "document.json").read_bytes()
                    assert archive.read("assembly/part.step") == (source / "part.step").read_bytes()
                    for part in ("base", "cap"):
                        for ext in ("step", "stl"):
                            name = f"definition_{part}.{ext}"
                            assert archive.read("make/" + name) == (source / name).read_bytes()
                    readme = archive.read("README.md").decode("utf-8")
                    assert "## Guidance acknowledged against the current saved facts" not in readme
                    assert "## Print, install and use" in readme
                    assert "Instruction binding status: legacy_unverified." in readme
                    assert "No bound customer steps are supplied." in readme
                    assert "## Unverified author notes and history" in readme
                    assert "[customer-guidance.json](customer-guidance.json)" in readme
                    assert "Instruction reference state: unbound" in readme
                    assert PLAN_ONLY not in readme
                    assert json.loads(archive.read("inspection-intent.json"))[
                        "assembly_sequence"
                    ] == [PLAN_ONLY]
                    assert "## Legacy customer guidance: unbound reference state" not in readme
                    if rid == RID:
                        assert sidecar["legacy_guidance"] == GUIDANCE
                        for name, text in GUIDANCE.items():
                            assert text not in readme, name
                    else:
                        assert sidecar["legacy_guidance"] is None
            schema = await sdk.call_tool("get_schema", {"kind": "document"})
            assert not schema.is_error and "delivery_guidance" in json.dumps(
                schema.structured_content
            )
        async with connection(service, tmp_path, surface, foreign=True) as sdk:
            for name in ("get_revision_source", "export_project"):
                refusal = await sdk.call_tool(name, {"project_id": sid, "revision_id": RID})
                assert refusal.is_error and not refusal.structured_content

    asyncio.run(scenario())
    assert snapshot(service, sid) == before
    assert before[0]["jobs"] == []


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("fault", ["document", "step", "stl"])
def test_sdk_refuses_tampered_guide_or_saved_geometry(tmp_path, monkeypatch, surface, fault):
    service, sid = fixture(tmp_path, monkeypatch)
    source = service.store.directory(sid, RID)
    if fault == "document":
        raw = json.loads((source / "document.json").read_bytes())
        raw["delivery_guidance"]["installation"] = "tampered after measurement"
        (source / "document.json").write_bytes(json_bytes(raw))
    else:
        (source / f"definition_base.{fault}").write_bytes(b"changed after measurement")
    before = snapshot(service, sid)

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            if fault == "document":
                response = await sdk.call_tool(
                    "get_revision_source", {"project_id": sid, "revision_id": RID}
                )
                assert response.is_error and not response.structured_content
            response = await sdk.call_tool(
                "export_project",
                {"project_id": sid, "revision_id": RID, "include_stl": True, "preview": "none"},
            )
            assert response.is_error and not response.structured_content

    asyncio.run(scenario())
    assert snapshot(service, sid) == before
    assert not list((service.store.root / "project-exports").glob("*/*/project.zip"))
    assert not list((tmp_path / "exports").glob("*.zip"))


@pytest.mark.parametrize("status", [None, "qualified", {"status": "current"}])
def test_remote_sdk_refuses_invalid_guidance_receipt_status(tmp_path, monkeypatch, status):
    from forge_mind.workspace import project_delivery

    service, sid = fixture(tmp_path, monkeypatch)
    before = snapshot(service, sid)
    original = project_delivery.create_project_export

    def poisoned(*args, **kwargs):
        receipt = original(*args, **kwargs)
        return {**receipt, "guidance_status": status}

    monkeypatch.setattr(project_delivery, "create_project_export", poisoned)

    async def scenario():
        async with connection(service, tmp_path, "remote") as sdk:
            response = await sdk.call_tool(
                "export_project",
                {"project_id": sid, "revision_id": RID, "include_stl": True, "preview": "none"},
            )
            assert response.is_error and not response.structured_content

    asyncio.run(scenario())
    assert snapshot(service, sid) == before
    assert not list((service.store.root / "project-exports").glob("*/*/project.zip"))


def guidance_update(context, text):
    return {
        "revision_id": context["basis"]["revision_id"],
        "basis_hash": context["basis_hash"],
        "reviewed_current_facts": True,
        "guidance": text,
    }


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize("accepted_baseline", [False, True])
def test_sdk_pending_candidate_guidance_and_complete_files_before_acceptance(
    tmp_path, monkeypatch, surface, accepted_baseline
):
    from forge_mind.workspace import delivery_guidance

    service, sid = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(delivery_guidance, "engine_fingerprint", lambda: "fixture-engine")
    from forge_mind.workspace import project_delivery

    monkeypatch.setattr(project_delivery, "engine_fingerprint", lambda: "fixture-engine")
    baseline = LEGACY if accepted_baseline else None
    service.store.update(sid, selected_revision=baseline, accepted_revision=baseline)
    candidate = service.store.revision(sid, RID)
    proposal = {
        "revision_id": RID,
        "based_on": baseline,
        "eligible": False,
        "input_version": 0,
        "contract": candidate["contract"],
    }
    baseline_text = {field: "BASELINE ONLY: " + text for field, text in GUIDANCE.items()}
    pending_text = {field: "PENDING CANDIDATE: " + text for field, text in GUIDANCE.items()}
    initial_session = service.store.get(sid, "owner")
    before_rows, before_sources = snapshot(service, sid)
    baseline_record = deepcopy(service.store.revision(sid, LEGACY))

    async def scenario():
        async with connection(service, tmp_path, surface, write=True) as sdk:
            if accepted_baseline:
                context = await sdk.call_tool("get_guidance", {"project_id": sid})
                assert not context.is_error
                response = await sdk.call_tool(
                    "update_guidance",
                    {
                        "project_id": sid,
                        "update": guidance_update(context.structured_content, baseline_text),
                    },
                )
                assert not response.is_error, response.content
                assert response.structured_content["status"] == "current"
            service.store.update(sid, proposal=proposal)
            project = await sdk.call_tool("get_project", {"project_id": sid})
            assert not project.is_error
            assert project.structured_content["design_revision"] == RID
            assert project.structured_content["selected_revision"] == baseline
            assert project.structured_content["accepted_revision"] == baseline
            default = await sdk.call_tool("get_guidance", {"project_id": sid})
            if accepted_baseline:
                assert not default.is_error
                assert default.structured_content["basis"]["revision_id"] == LEGACY
                assert default.structured_content["record"]["guidance"] == baseline_text
            else:
                assert default.is_error
            context = await sdk.call_tool("get_guidance", {"project_id": sid, "revision_id": RID})
            assert not context.is_error, context.content
            facts = context.structured_content
            assert facts["basis"]["revision_id"] == facts["design_revision"] == RID
            assert facts["target_kind"] == "pending_candidate"
            assert facts["selected_revision"] == facts["accepted_revision"] == baseline
            response = await sdk.call_tool(
                "update_guidance",
                {"project_id": sid, "update": guidance_update(facts, pending_text)},
            )
            assert not response.is_error, response.content
            assert response.structured_content["status"] == "current"
            assert response.structured_content["target_kind"] == "pending_candidate"
            assert (
                response.structured_content["instruction_binding"]["status"] == "legacy_unverified"
            )
            assert response.structured_content["instruction_binding"]["steps"] == []
            response = await sdk.call_tool(
                "export_project",
                {
                    "project_id": sid,
                    "revision_id": RID,
                    "include_stl": True,
                    "include_individual_files": True,
                    "preview": "none",
                },
            )
            receipt, content = await delivered_zip(sdk, response, surface)
            assert receipt["revision_id"] == RID
            assert receipt["guidance_status"] == "current"
            assert receipt["context"]["matches_accepted_revision"] is False
            source = service.store.directory(sid, RID)
            with ZipFile(io.BytesIO(content)) as archive:
                readme = archive.read("README.md").decode("utf-8")
                for field, text in pending_text.items():
                    assert text not in readme, field
                assert "BASELINE ONLY:" not in readme
                assert "## Print, install and use" in readme
                assert "Instruction binding status: legacy_unverified." in readme
                assert "No bound customer steps are supplied." in readme
                assert "[customer-guidance.json](customer-guidance.json)" in readme
                assert PLAN_ONLY not in readme
                saved = json.loads(archive.read("customer-guidance.json"))
                assert saved["status"] == "current"
                assert saved["basis"]["revision_id"] == RID
                assert saved["record"]["guidance"] == pending_text
                assert saved["legacy_guidance"] == GUIDANCE
                assert saved["basis"] == facts["basis"]
                assert saved["basis_hash"] == facts["basis_hash"]
                assert saved["instruction_binding"]["status"] == "legacy_unverified"
                assert saved["instruction_binding"]["steps"] == []
                assert archive.read("document.json") == (source / "document.json").read_bytes()
                assert archive.read("assembly/part.step") == (source / "part.step").read_bytes()
                members = {member["archive_path"]: member for member in receipt["individual_files"]}
                expected = {"assembly/part.step"}
                expected.update(
                    f"make/definition_{part}.{extension}"
                    for part in ("base", "cap")
                    for extension in ("step", "stl")
                )
                assert set(members) == expected
                for name, member in members.items():
                    if surface == "local":
                        raw = Path(member["path"]).read_bytes()
                    else:
                        resource = await sdk.read_resource(member["uri"])
                        raw = base64.b64decode(resource.contents[0].blob, validate=True)
                        assert receipt["package_id"] in member["uri"]
                        assert receipt["sha256"] in member["uri"]
                    assert raw == archive.read(name)
                    assert sha(raw) == member["sha256"]
                    assert len(raw) == member["bytes"]
                    assert member["quantity"] == (2 if member["part_id"] == "cap" else 1)
            if accepted_baseline:
                context = await sdk.call_tool(
                    "get_guidance", {"project_id": sid, "revision_id": LEGACY}
                )
                assert not context.is_error
                assert context.structured_content["status"] == "current"
                assert context.structured_content["record"]["guidance"] == baseline_text
                refused = await sdk.call_tool(
                    "update_guidance",
                    {
                        "project_id": sid,
                        "update": guidance_update(context.structured_content, baseline_text),
                    },
                )
                assert refused.is_error
                baseline_export = await sdk.call_tool(
                    "export_project",
                    {
                        "project_id": sid,
                        "revision_id": LEGACY,
                        "include_stl": True,
                        "preview": "none",
                    },
                )
                old_receipt, old_content = await delivered_zip(sdk, baseline_export, surface)
                assert old_receipt["guidance_status"] == "current"
                assert old_receipt["context"]["matches_accepted_revision"] is True
                with ZipFile(io.BytesIO(old_content)) as archive:
                    readme = archive.read("README.md").decode("utf-8")
                    assert all(text not in readme for text in baseline_text.values())
                    assert "PENDING CANDIDATE:" not in readme
                    assert "## Print, install and use" in readme
                    assert "Instruction binding status: legacy_unverified." in readme
                    assert "No bound customer steps are supplied." in readme
                    assert "[customer-guidance.json](customer-guidance.json)" in readme
                    saved_baseline = json.loads(archive.read("customer-guidance.json"))
                    assert saved_baseline["status"] == "current"
                    assert saved_baseline["basis"]["revision_id"] == LEGACY
                    assert saved_baseline["record"]["guidance"] == baseline_text
                    assert saved_baseline["instruction_binding"]["status"] == "legacy_unverified"
                    assert saved_baseline["instruction_binding"]["steps"] == []
        async with connection(service, tmp_path, surface, foreign=True, write=True) as sdk:
            denied = await sdk.call_tool("get_guidance", {"project_id": sid, "revision_id": RID})
            assert denied.is_error

    asyncio.run(scenario())
    after = service.store.get(sid, "owner")
    assert after["selected_revision"] == baseline
    assert after["accepted_revision"] == baseline
    assert after["proposal"] == proposal
    for field in ("contract", "design_plan", "plan_hash"):
        assert after.get(field) == initial_session.get(field), field
    assert service.store.revision(sid, LEGACY) == baseline_record
    after_rows, after_sources = snapshot(service, sid)
    assert after_sources == before_sources
    assert after_rows["jobs"] == before_rows["jobs"] == []
    assert after_rows["revisions"] == before_rows["revisions"]
