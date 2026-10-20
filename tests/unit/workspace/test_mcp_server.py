from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest


def test_bridge_module_exists() -> None:
    assert importlib.util.find_spec("forge_mind.workspace.mcp_server") is not None


def test_explicit_operation_retry_forwards_only_original_job_id(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    requests = []

    def backend(request):
        requests.append(request)
        return httpx.Response(202, json={"id": "a" * 32, "job_id": "c" * 32, "state": "queued"})

    async def scenario():
        server = create_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
            )
        )
        async with Client(server) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            retry = tools["retry_operation"]
            assert set(retry.input_schema["properties"]) == {"project_id", "job_id"}
            assert retry.annotations.read_only_hint is False
            result = await client.call_tool(
                "retry_operation", {"project_id": "a" * 32, "job_id": "b" * 32}
            )
            assert not result.is_error
            assert result.structured_content["job_id"] == "c" * 32

    asyncio.run(scenario())
    assert len(requests) == 1
    assert requests[0].method == "POST"
    assert requests[0].url.path == f"/workspace/sessions/{'a' * 32}/retry-operation"
    assert json.loads(requests[0].content) == {"job_id": "b" * 32}


def test_design_page_preserves_encoded_cursor_and_feature_scope(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    def backend(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/page")
        assert request.url.params["cursor"] == "cursor+with/slash="
        assert request.url.params["feature_id"] == "source"
        assert request.url.params["scope"] == "component"
        return httpx.Response(200, json={"items": [1], "next_cursor": None})

    async def scenario():
        server = create_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
            )
        )
        async with Client(server) as client:
            result = await client.call_tool(
                "read_design_page",
                {
                    "project_id": "a" * 32,
                    "revision_id": "b" * 32,
                    "collection": "features",
                    "scope": "component",
                    "feature_id": "source",
                    "cursor": "cursor+with/slash=",
                },
            )
            assert not result.is_error
            assert result.structured_content["items"] == [1]

    asyncio.run(scenario())


def test_mcp_response_capacity_is_an_explicit_policy(tmp_path: Path, monkeypatch) -> None:
    from mcp.server.mcpserver.exceptions import ToolError

    from forge_mind.workspace.mcp_server import WorkspaceAdapter

    adapter = WorkspaceAdapter(
        "http://127.0.0.1:4321",
        tmp_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"data": "x" * 100})
        ),
    )
    monkeypatch.delenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", raising=False)
    assert asyncio.run(adapter.json("GET", "/workspace/test"))["data"] == "x" * 100
    monkeypatch.setenv("FORGE_WORKSPACE_MCP_RESPONSE_BYTES", "10")
    with pytest.raises(ToolError, match="size limit"):
        asyncio.run(adapter.json("GET", "/workspace/test"))


def test_structure_read_and_compact_acceptance_protocol(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    seen = []
    sid, rid = "a" * 32, "b" * 32

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/structure"):
            return httpx.Response(
                200, json={"revision_id": rid, "roots": [{"id": "jaw", "position": [3, 4, 5]}]}
            )
        assert request.url.params["compact"] == "true"
        return httpx.Response(200, json={"id": sid, "accepted_revision": rid, "candidate": None})

    async def scenario():
        server = create_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
            )
        )
        async with Client(server) as client:
            structure = await client.call_tool(
                "read_assembly_structure", {"project_id": sid, "revision_id": rid}
            )
            assert not structure.is_error
            assert structure.structured_content["roots"][0]["position"] == [3, 4, 5]
            accepted = await client.call_tool(
                "accept_candidate", {"project_id": sid, "revision_id": rid, "input_version": 0}
            )
            assert not accepted.is_error
            assert accepted.structured_content["accepted_revision"] == rid

    asyncio.run(scenario())
    assert len(seen) == 2


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.2:4321",
        "http://localhost.evil:4321",
        "http://user:secret@localhost:4321",
        "http://localhost:4321/path",
        "http://localhost:4321?x=y",
        "http://localhost:4321#fragment",
        "file:///tmp",
    ],
)
def test_backend_config_rejects_non_loopback_and_ambiguous_urls(tmp_path: Path, url: str) -> None:
    from forge_mind.workspace.mcp_server import WorkspaceAdapter

    with pytest.raises(ValueError, match="loopback"):
        WorkspaceAdapter(url, tmp_path)


def test_sdk_protocol_schema_validation_and_queued_state(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    requests: list[httpx.Request] = []
    sid = "a" * 32

    def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"id": sid, "state": "queued"})

    server = create_server(
        WorkspaceAdapter("http://localhost:4321", tmp_path, transport=httpx.MockTransport(backend))
    )

    async def scenario() -> None:
        async with Client(server, raise_exceptions=True) as client:
            assert client.server_info.name == "CadKitty"
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert {
                "create_project",
                "submit_document",
                "edit_variables",
                "export_artifact",
                "ai_create_project",
                "ai_edit_project",
                "get_project",
                "cancel_job",
                "accept_candidate",
                "run_engineering_review",
                "get_engineering_report",
            } <= tools.keys()
            invalid = await client.call_tool("get_project", {"project_id": "../outside"})
            assert invalid.is_error
            assert not requests
            queued = await client.call_tool("ai_create_project", {"prompt": "a plate"})
            assert not queued.is_error
            assert queued.structured_content["state"] == "queued"
            assert requests[-1].url.host == "127.0.0.1"
            assert requests[-1].url.path == "/workspace/sessions"
            before = len(requests)
            bad_document = await client.call_tool(
                "submit_document",
                {
                    "project_id": sid,
                    "document": {"features": [], "code": "unsafe"},
                },
            )
            assert bad_document.is_error
            assert len(requests) == before

    asyncio.run(scenario())


def test_redirects_and_backend_failures_are_tool_errors(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    for status in (302, 401, 409, 500):
        requests: list[httpx.Request] = []

        def backend(request: httpx.Request, requests=requests, status=status) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                status,
                headers={"Location": "http://example.com/secret"},
                json={"detail": "refused"},
            )

        server = create_server(
            WorkspaceAdapter(
                "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
            )
        )

        async def scenario(server=server, status=status) -> None:
            async with Client(server) as client:
                result = await client.call_tool("list_projects")
                assert result.is_error
                if status == 409:
                    assert "refused" in str(result.content)

        asyncio.run(scenario())
        assert len(requests) == 1


def test_deterministic_variable_edit_uses_verified_document_without_ai(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    sid, rid = "a" * 32, "b" * 32
    doc = {"variables": {"width": 20}, "features": [], "outputs": []}
    requests: list[httpx.Request] = []

    def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            if request.url.path.endswith("overview"):
                return httpx.Response(200, json={"selected_revision": rid, "input_version": 3})
            return httpx.Response(200, json=doc)
        payload = json.loads(request.content)
        assert request.url.path == f"/workspace/sessions/{sid}/document"
        assert payload["document"]["variables"]["width"] == 40
        assert payload["revision_id"] == rid
        assert payload["expected_current_revision"] == rid
        assert payload["expected_input_version"] == 3
        assert "instruction" not in payload
        return httpx.Response(202, json={"id": sid, "state": "queued"})

    server = create_server(
        WorkspaceAdapter("http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend))
    )

    async def scenario() -> None:
        async with Client(server) as client:
            response = await client.call_tool(
                "edit_variables",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "variables": {"width": 40},
                },
            )
            assert not response.is_error
            assert response.structured_content["state"] == "queued"
            assert requests[0].url.path.endswith(f"/revisions/{rid}/document.json")
            response = await client.call_tool(
                "edit_variables",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "variables": {"unknown": 40},
                },
            )
            assert response.is_error
            assert sum(r.method == "POST" for r in requests) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize("tamper", [False, True])
def test_export_verifies_hash_and_stays_in_configured_directory(
    tmp_path: Path, tamper: bool
) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    sid, rid = "a" * 32, "b" * 32
    content = b"ISO-10303-21; example"
    digest = hashlib.sha256(content).hexdigest()

    def backend(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("evidence.json"):
            return httpx.Response(
                200,
                json={
                    "revision_id": rid,
                    "artifact_hashes": {"part.step": digest},
                    "geometry_valid": False,
                },
            )
        return httpx.Response(200, content=content + (b"tampered" if tamper else b""))

    server = create_server(
        WorkspaceAdapter("http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend))
    )

    async def scenario() -> None:
        async with Client(server) as client:
            response = await client.call_tool(
                "export_artifact",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "format": "step",
                },
            )
            assert bool(response.is_error) == tamper
            if not tamper:
                data = response.structured_content
                path = Path(data["path"])
                assert path.parent == tmp_path
                assert path.read_bytes() == content
                assert data["sha256"] == digest
                assert data["geometry_valid"] is False
            else:
                assert not list(tmp_path.iterdir())

    asyncio.run(scenario())


def test_real_stdio_initialize_list_and_schema_call(tmp_path: Path) -> None:
    from mcp import Client, StdioServerParameters

    from forge_mind.workspace.document import DesignDocument

    class Backend(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path == "/workspace/document-schema"
            content = json.dumps({"document": DesignDocument.model_json_schema()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args: object) -> None:
            pass

    backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
    thread = threading.Thread(target=backend.serve_forever, daemon=True)
    thread.start()

    async def scenario() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "forge_mind.workspace.mcp_server"],
            env={
                "FORGE_MCP_EXPORT_DIR": str(tmp_path),
                "FORGE_MCP_WORKSPACE_URL": f"http://127.0.0.1:{backend.server_port}",
            },
        )
        async with Client(params) as client:
            assert client.server_info.name == "CadKitty"
            assert (await client.list_tools()).tools
            response = await client.call_tool("get_schema", {"kind": "document"})
            assert not response.is_error
            assert "features" in response.structured_content["document"]["properties"]

    try:
        asyncio.run(scenario())
    finally:
        backend.shutdown()
        backend.server_close()
        thread.join()


def test_scoped_tools_keep_state_guards_and_preserve_omitted_metadata(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    sid, rid = "a" * 32, "b" * 32
    requests = []

    def backend(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"state": "queued"})

    server = create_server(
        WorkspaceAdapter("http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend))
    )

    async def scenario() -> None:
        async with Client(server) as client:
            result = await client.call_tool(
                "edit_assembly",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "input_version": 4,
                    "edit": {"variables": {"gap": 30}},
                },
            )
            assert not result.is_error
            payload = json.loads(requests[-1].content)
            assert payload == {
                "revision_id": rid,
                "input_version": 4,
                "edit": {"variables": {"gap": 30}},
            }
            result = await client.call_tool(
                "run_motion_check", {"project_id": sid, "revision_id": rid}
            )
            assert not result.is_error
            assert requests[-1].url.path.endswith("/motion-check")
            result = await client.call_tool(
                "read_part", {"project_id": sid, "revision_id": rid, "part_id": "../../secrets"}
            )
            assert result.is_error and len(requests) == 2

    asyncio.run(scenario())


def test_structural_and_qualification_tools_use_scoped_compact_api(tmp_path: Path) -> None:
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
    from forge_mind.workspace.structural_models import StructuralStudy

    sid, rid, study_id, job_id = (c * 32 for c in "abcd")
    study = {
        "name": "Declared cantilever case",
        "target": {"occurrence_path": "support"},
        "material": {
            "name": "Declared test material",
            "source": "Explicit test assumption",
            "density_kg_m3": 2700,
            "elastic_modulus_mpa": 70000,
            "poisson_ratio": 0.3,
        },
        "force_n": 100,
        "direction": [1, 0, 0],
        "basis": "Idealized component load only, no clamp-force inference",
    }
    seen = []

    def backend(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        path = request.url.path
        if request.method == "POST":
            assert path == f"/workspace/sessions/{sid}/structural-studies"
            payload = json.loads(request.content)
            assert payload == {
                "revision_id": rid,
                "input_version": 3,
                "study": StructuralStudy.model_validate(study).model_dump(mode="json"),
            }
            return httpx.Response(
                202, json={"study_id": study_id, "job_id": job_id, "state": "queued"}
            )
        assert request.method == "GET"
        if path == "/workspace/structural-schema":
            return httpx.Response(200, json={"study": StructuralStudy.model_json_schema()})
        if path == f"/workspace/sessions/{sid}/analysis-geometry/{rid}":
            assert dict(request.url.params) == {"body_index": "2"}
            return httpx.Response(
                200, json={"body_index": 2, "planar_faces": [{"signature": "f" * 64}]}
            )
        if path == f"/workspace/sessions/{sid}/qualification/{rid}":
            return httpx.Response(
                200, json={"revision_id": rid, "physical_readiness": "unverified"}
            )
        if path == f"/workspace/sessions/{sid}/structural-studies/{study_id}":
            return httpx.Response(
                200, json={"id": study_id, "revision_id": rid, "qualification_status": "unverified"}
            )
        assert path == f"/workspace/sessions/{sid}/structural-studies"
        assert dict(request.url.params) in ({}, {"revision_id": rid})
        return httpx.Response(200, json={"studies": [{"id": study_id}]})

    async def scenario() -> None:
        async with Client(
            create_server(
                WorkspaceAdapter(
                    "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
                )
            )
        ) as client:
            definitions = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert definitions["get_structural_study"].annotations.read_only_hint is True
            assert definitions["run_structural_study"].annotations.open_world_hint is False
            for name, arguments, expected_key in [
                ("get_schema", {"kind": "structural"}, "study"),
                (
                    "read_analysis_geometry",
                    {"project_id": sid, "revision_id": rid, "body_index": 2},
                    "planar_faces",
                ),
                (
                    "get_qualification",
                    {"project_id": sid, "revision_id": rid},
                    "physical_readiness",
                ),
                (
                    "run_structural_study",
                    {"project_id": sid, "revision_id": rid, "input_version": 3, "study": study},
                    "job_id",
                ),
                (
                    "get_structural_study",
                    {"project_id": sid, "study_id": study_id},
                    "qualification_status",
                ),
                ("list_structural_studies", {"project_id": sid, "revision_id": rid}, "studies"),
                ("list_structural_studies", {"project_id": sid}, "studies"),
            ]:
                result = await client.call_tool(name, arguments)
                assert not result.is_error, result.content
                assert expected_key in result.structured_content
            before = len(seen)
            for invalid_study in [
                {**study, "direction": [0, 0, 0]},
                {**study, "force_n": -1},
                {**study, "mesh_factors": [0.1, 0.2]},
                {**study, "target": {"body_index": 0, "occurrence_path": "support"}},
                {**study, "script": "arbitrary source"},
            ]:
                invalid = await client.call_tool(
                    "run_structural_study",
                    {
                        "project_id": sid,
                        "revision_id": rid,
                        "input_version": 3,
                        "study": invalid_study,
                    },
                )
                assert invalid.is_error
            for invalid_index in [-1, True]:
                invalid = await client.call_tool(
                    "read_analysis_geometry",
                    {"project_id": sid, "revision_id": rid, "body_index": invalid_index},
                )
                assert invalid.is_error
            assert len(seen) == before

    asyncio.run(scenario())
    assert len(seen) == 7
    assert not any("field" in request.url.path for request in seen)


def _named_export_document():
    return {
        "name": "Named export fixture",
        "assembly": {
            "parts": [
                {
                    "id": identity,
                    "name": identity,
                    "production": {"role": role},
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
                for identity, role in (("tray", "make"), ("marker", "reference"))
            ],
            "roots": [
                {"id": "tray_a", "definition": "tray"},
                {"id": "tray_b", "definition": "tray", "position": [20, 0, 0]},
                {"id": "marker_a", "definition": "marker"},
            ],
        },
    }


@pytest.mark.parametrize("failure", [None, "document", "step", "missing_hash", "unknown"])
def test_named_part_export_exact_definition_and_local_role(tmp_path, failure):
    from mcp import Client

    from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server

    sid, rid = "a" * 32, "b" * 32
    data = {
        "document.json": json.dumps(_named_export_document()).encode(),
        "definition_tray.step": b"immutable tray definition",
        "definition_marker.step": b"immutable marker definition",
    }
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in data.items()}
    if failure == "missing_hash":
        del hashes["definition_tray.step"]
    paths = []

    def backend(request):
        paths.append(request.url.path)
        name = request.url.path.rsplit("/", 1)[-1]
        if name == "evidence.json":
            return httpx.Response(
                200, json={"revision_id": rid, "artifact_hashes": hashes, "geometry_valid": True}
            )
        content = data[name]
        if name == {"document": "document.json", "step": "definition_tray.step"}.get(failure):
            content += b"changed"
        return httpx.Response(200, content=content)

    async def scenario():
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as client:
            tools = {tool.name: tool for tool in (await client.list_tools()).tools}
            assert tools["export_part"].annotations.read_only_hint is False
            response = await client.call_tool(
                "export_part",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "part_id": "ghost" if failure == "unknown" else "tray",
                },
            )
            assert bool(response.is_error) == bool(failure)
            if failure:
                assert not list(tmp_path.iterdir())
                return
            result = response.structured_content
            path = Path(result["path"])
            assert path.parent == tmp_path
            assert path.read_bytes() == data["definition_tray.step"]
            assert result["frame"] == "definition_local"
            assert result["part_id"] == "tray" and result["production_role"] == "make"
            assert result["document_sha256"] == hashes["document.json"]
            assert result["status"] == "draft_candidate"
            assert "geometry_valid" not in result and result["revision_geometry_valid"] is True
            reference = await client.call_tool(
                "export_part", {"project_id": sid, "revision_id": rid, "part_id": "marker"}
            )
            assert not reference.is_error
            assert reference.structured_content["production_role"] == "reference"
            assert (
                Path(reference.structured_content["path"]).read_bytes()
                == data["definition_marker.step"]
            )
            again = await client.call_tool(
                "export_part", {"project_id": sid, "revision_id": rid, "part_id": "tray"}
            )
            assert not again.is_error
            assert again.structured_content["path"] != result["path"]
            assert len(list(tmp_path.iterdir())) == 3
            invalid = await client.call_tool(
                "export_part", {"project_id": sid, "revision_id": rid, "part_id": "../tray"}
            )
            assert invalid.is_error

    asyncio.run(scenario())
    assert not any(path.endswith("/part.step") or "definition_ghost" in path for path in paths)


def test_named_definition_rejects_occurrence_id_and_nonassembly_document():
    from mcp.server.mcpserver.exceptions import ToolError

    from forge_mind.workspace.mcp_server import definition_export_metadata

    data = json.dumps(_named_export_document()).encode()
    with pytest.raises(ToolError, match="saved assembly definition"):
        definition_export_metadata(data, "tray_a")
    with pytest.raises(ToolError, match="saved assembly definition"):
        definition_export_metadata(b'{"name":"single part"}', "tray")
    with pytest.raises(ToolError, match="Invalid part"):
        definition_export_metadata(data, "../tray")


def test_named_export_preserves_existing_file_on_filename_collision(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from mcp.server.mcpserver.exceptions import ToolError

    import forge_mind.workspace.mcp_server as module

    document = json.dumps(_named_export_document()).encode()
    content = b"immutable definition"
    hashes = {
        "document.json": hashlib.sha256(document).hexdigest(),
        "definition_tray.step": hashlib.sha256(content).hexdigest(),
    }

    def backend(request):
        if request.url.path.endswith("evidence.json"):
            return httpx.Response(200, json={"revision_id": "b" * 32, "artifact_hashes": hashes})
        return httpx.Response(
            200, content=document if request.url.path.endswith("document.json") else content
        )

    monkeypatch.setattr(module.uuid, "uuid4", lambda: SimpleNamespace(hex="c" * 32))
    target = tmp_path / f"{'a' * 32}-{'b' * 32}-tray-{'c' * 12}.step"
    target.write_bytes(b"preserve")
    adapter = module.WorkspaceAdapter(
        "http://127.0.0.1:4321", tmp_path, transport=httpx.MockTransport(backend)
    )
    with pytest.raises(ToolError, match="Cannot write"):
        asyncio.run(adapter.export("a" * 32, "b" * 32, "step", part_id="tray"))
    assert target.read_bytes() == b"preserve"
