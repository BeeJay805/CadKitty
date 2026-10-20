"""Actual SDK/owned HTTP transport checks. Saved bytes are fixtures, not native proof."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import httpx2
import pytest
from mcp import Client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.auth.provider import AccessToken
from mcp.shared.exceptions import MCPError

from forge_mind.server.workspace_app import create_app as local_app
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.definition_meshes import MANIFEST_NAME
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.models import Measurement
from forge_mind.workspace.remote_mcp import create_app as remote_app
from forge_mind.workspace.store import Store

ISSUER = "https://fixture.example.test"
RESOURCE = "http://127.0.0.1:8765/mcp"
RID = "b" * 32


def sha(value):
    return hashlib.sha256(value).hexdigest()


def encoded(value):
    return json.dumps(value, allow_nan=False).encode()


def saved_fixture(tmp_path, *, selected=True, status="available", legacy=False):
    service = WorkspaceService(Store(tmp_path / "data"), Executor("docker"))

    def forbidden_start():
        raise AssertionError("No read or export may start a native worker")

    service.start = forbidden_start
    sid = service.store.create("local-owner", "saved mesh fixture", None)["id"]
    document = {
        "name": "Selected make mesh",
        "assembly": {
            "parts": [
                {
                    "id": "tray",
                    "name": "Tray",
                    "production": {
                        "role": "make",
                        **({"deliver_stl": selected} if not legacy else {}),
                    },
                    "features": [
                        {
                            "id": "shape",
                            "kind": "box",
                            "name": "Shape",
                            "params": {"x": 10, "y": 8, "z": 3},
                        }
                    ],
                    "output": "shape",
                }
            ],
            "roots": [
                {"id": "tray_a", "definition": "tray"},
                {"id": "tray_b", "definition": "tray", "position": [20, 0, 0]},
            ],
        },
    }
    data = {
        "document.json": encoded(document),
        "definition_tray.step": b"saved native STEP fixture",
        "definition_tray.stl": b"saved native STL fixture",
    }
    entry = {
        "status": status,
        "role": "make",
        "quantity": 2,
        "step_file": "definition_tray.step",
        "step_sha256": sha(data["definition_tray.step"]),
    }
    if status == "available":
        entry.update(
            stl_file="definition_tray.stl",
            stl_sha256=sha(data["definition_tray.stl"]),
            stl_bytes=len(data["definition_tray.stl"]),
            body_count=1,
            bounds_mm=[[-5.0, -4.0, -1.5], [5.0, 4.0, 1.5]],
            tessellation={
                "linear_deflection": 0.001,
                "linear_deflection_is_relative": True,
                "angular_deflection": 0.1,
                "collapsed_triangles_removed": 0,
            },
            export_consistency={
                "volume_within_relative_0_001": True,
                "bounds_within_0_01_mm": True,
                "maximum_sampled_bounds_difference_mm": 0.0,
                "reference_tessellation": {
                    "triangle_inventory_matches": True,
                    "linear_deflection_setting": 0.001,
                    "linear_deflection_is_relative": True,
                    "angular_deflection_rad": 0.1,
                    "scope": "Independent tessellation of the final STEP with identical exporter settings; exact oriented float32 triangle multiset including multiplicity. Sampled curved bounds are not a dimensional or manufacturing tolerance.",
                },
                "scope": "Numerical export consistency only. Exact dimensions and requirement checks use STEP; sampled tessellation bounds do not define manufacturing precision.",
            },
        )
    else:
        entry["error"] = "Selected mesh failed topology check"
        data.pop("definition_tray.stl")
    manifest = {
        "schema_version": 1,
        "revision_id": RID,
        "document_sha256": sha(data["document.json"]),
        "frame": "definition_local",
        "units": "mm",
        "parts": {"tray": entry},
    }
    if legacy:
        data.pop("definition_tray.stl", None)
    else:
        data[MANIFEST_NAME] = encoded(manifest)
    directory = service.store.directory(sid, RID)
    directory.mkdir(parents=True)
    for name, content in data.items():
        (directory / name).write_bytes(content)
    measurement = Measurement(
        revision_id=RID,
        geometry_valid=True,
        contract_hash="fixture-contract",
        engine_fingerprint="fixture-engine",
        artifact_hashes={name: sha(content) for name, content in data.items()},
    ).model_dump()
    service.store.add_revision(sid, {"id": RID, "measurement": measurement})
    (directory / "evidence.json").write_bytes(encoded(measurement))
    return service, sid, data, manifest


class Verifier:
    async def verify_token(self, token):
        if token not in {"owned", "foreign", "readonly"}:
            return None
        return AccessToken(
            token=token,
            client_id="scratch-sdk",
            subject="foreign" if token == "foreign" else "local-owner",
            scopes=["cad:read"] if token == "readonly" else ["cad:read", "cad:write"],
            resource=RESOURCE,
            expires_at=int(time.time()) + 60,
            claims={"iss": ISSUER},
        )


@asynccontextmanager
async def connection(service, tmp_path, surface, foreign=False, request_paths=None):
    if surface == "local":
        app = local_app(service=service, local=True)
        if foreign:
            owners = {
                dep.call
                for route in app.routes
                if hasattr(route, "dependant")
                for dep in route.dependant.dependencies
                if getattr(dep.call, "__name__", None) == "owner"
            }
            assert len(owners) == 1
            app.dependency_overrides[next(iter(owners))] = lambda: "foreign"

        class CountingTransport(httpx.ASGITransport):
            async def handle_async_request(self, request):
                if request_paths is not None:
                    request_paths.append(request.url.path)
                return await super().handle_async_request(request)

        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=CountingTransport(app=app)
        )
        async with Client(create_server(adapter)) as sdk:
            yield sdk
    else:
        app = remote_app(
            service, token_verifier=Verifier(), issuer_url=ISSUER, resource_url=RESOURCE
        )
        async with (
            app.router.lifespan_context(app),
            httpx2.AsyncClient(
                transport=httpx2.ASGITransport(app=app),
                headers={"Authorization": "Bearer " + ("foreign" if foreign else "readonly")},
                trust_env=False,
            ) as http,
            Client(streamable_http_client(RESOURCE, http_client=http), mode="legacy") as sdk,
        ):
            yield sdk
    assert service.worker is None and not service.executor.typed_container_verified


@pytest.mark.parametrize("surface", ["local", "remote"])
def test_sdk_mesh_export_exact_saved_binding_and_unchanged_step_default(tmp_path, surface):
    service, sid, data, manifest = saved_fixture(tmp_path)
    args = {"project_id": sid, "revision_id": RID, "part_id": "tray"}

    async def scenario():
        paths = []
        async with connection(service, tmp_path, surface, request_paths=paths) as sdk:
            assert sdk.server_info.version == ("0.0.19" if surface == "local" else "0.0.20")
            tool = {t.name: t for t in (await sdk.list_tools()).tools}["export_part"]
            assert tool.input_schema["properties"]["format"]["default"] == "step"
            assert tool.input_schema["properties"]["format"]["enum"] == ["step", "stl"]
            assert "deliver_stl" in sdk.instructions
            default = await sdk.call_tool("export_part", args)
            explicit = await sdk.call_tool("export_part", {**args, "format": "step"})
            assert not default.is_error and not explicit.is_error
            old, same = default.structured_content.copy(), explicit.structured_content.copy()
            old.pop("path", None)
            same.pop("path", None)
            assert old == same
            assert "mesh_manifest" not in old and old["artifact_name"] == "definition_tray.step"
            assert old["sha256"] == sha(data["definition_tray.step"])
            expected = {
                "part_id": "tray",
                "artifact_name": "definition_tray.step",
                "frame": "definition_local",
                "production_role": "make",
                "document_sha256": sha(data["document.json"]),
                "status": "draft_candidate",
                "revision_id": RID,
                "sha256": sha(data["definition_tray.step"]),
                "bytes": len(data["definition_tray.step"]),
            }
            if surface == "local":
                expected.update(
                    revision_geometry_valid=True,
                    claims="Hash-verified STEP definition in local part coordinates; no posed occurrence, manufacturing or functional approval.",
                )
            else:
                expected.update(
                    uri=f"cadkitty://projects/{sid}/revisions/{RID}/definition_tray.step",
                    claims="Hash-verified STEP definition; no posed occurrence, manufacturing or functional approval.",
                )
            assert old == expected
            before_mesh = len(paths)
            mesh = await sdk.call_tool("export_part", {**args, "format": "stl"})
            assert not mesh.is_error, mesh
            exported = mesh.structured_content
            assert exported["mesh_manifest"] == manifest["parts"]["tray"]
            assert exported["mesh_manifest_sha256"] == sha(data[MANIFEST_NAME])
            assert exported["document_sha256"] == sha(data["document.json"])
            assert exported["step_sha256"] == sha(data["definition_tray.step"])
            assert exported["sha256"] == sha(data["definition_tray.stl"])
            assert exported["units"] == "mm" and exported["frame"] == "definition_local"
            if surface == "local":
                assert (
                    sum(path.endswith("/definition_tray.stl") for path in paths[before_mesh:]) == 1
                )
                assert Path(exported["path"]).read_bytes() == data["definition_tray.stl"]
                assert len(list((tmp_path / "exports").iterdir())) == 3
            else:
                result = await sdk.read_resource(exported["uri"])
                assert base64.b64decode(result.contents[0].blob) == data["definition_tray.stl"]
                # Binding is checked again on immutable resource retrieval, not just export.
                (service.store.directory(sid, RID) / MANIFEST_NAME).write_bytes(b"changed manifest")
                with pytest.raises(MCPError):
                    await sdk.read_resource(exported["uri"])
            invalid = await sdk.call_tool("export_part", {**args, "format": "glb"})
            assert invalid.is_error
        async with connection(service, tmp_path, surface, foreign=True) as sdk:
            denied = await sdk.call_tool("export_part", {**args, "format": "stl"})
            assert denied.is_error

    asyncio.run(scenario())


@pytest.mark.parametrize("surface", ["local", "remote"])
@pytest.mark.parametrize(
    "failure",
    [
        "legacy",
        "optout",
        "failed",
        "document",
        "manifest",
        "revision",
        "quantity",
        "step",
        "stl",
        "unknown",
        "size",
        "hardlink",
    ],
)
def test_sdk_mesh_refusals_never_export_partial_or_regenerate(
    tmp_path, monkeypatch, surface, failure
):
    service, sid, data, manifest = saved_fixture(
        tmp_path,
        selected=failure != "optout",
        status="failed" if failure == "failed" else "available",
        legacy=failure == "legacy",
    )
    directory = service.store.directory(sid, RID)
    args = {"project_id": sid, "revision_id": RID, "part_id": "tray", "format": "stl"}
    if failure in {"document", "manifest", "step", "stl"}:
        name = {
            "document": "document.json",
            "manifest": MANIFEST_NAME,
            "step": "definition_tray.step",
            "stl": "definition_tray.stl",
        }[failure]
        (directory / name).write_bytes(data[name] + b" changed")
    elif failure in {"revision", "quantity"}:
        changed = copy.deepcopy(manifest)
        if failure == "revision":
            changed["revision_id"] = "c" * 32
        else:
            changed["parts"]["tray"]["quantity"] = 1
        raw = encoded(changed)
        (directory / MANIFEST_NAME).write_bytes(raw)
        record = service.store.revision(sid, RID)
        record["measurement"]["artifact_hashes"][MANIFEST_NAME] = sha(raw)
        with service.store.connect() as con:
            con.execute("UPDATE revisions SET document=? WHERE id=?", (json.dumps(record), RID))
        (directory / "evidence.json").write_bytes(encoded(record["measurement"]))
    elif failure == "unknown":
        args["part_id"] = "tray_a"
    elif failure == "size":
        if surface == "local":
            monkeypatch.setenv("FORGE_WORKSPACE_MCP_EXPORT_BYTES", "1")
        else:
            monkeypatch.setattr("forge_mind.workspace.remote_mcp.MAX_ARTIFACT_BYTES", 1)
    elif failure == "hardlink":
        os.link(directory / "definition_tray.stl", directory / "linked.stl")

    async def scenario():
        async with connection(service, tmp_path, surface) as sdk:
            result = await sdk.call_tool("export_part", args)
            assert result.is_error, result
            if failure == "legacy":
                assert "legacy" in result.content[0].text.lower()
            if failure == "failed":
                assert "topology" in result.content[0].text.lower()
        assert not (tmp_path / "exports").exists()

    asyncio.run(scenario())
