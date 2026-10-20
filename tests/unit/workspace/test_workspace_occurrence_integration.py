"""Real postprocessor, immutable Store, HTTP and SDK occurrence inspection chain."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.api import router
from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
    PartPort,
)
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.mcp_server import WorkspaceAdapter, create_server
from forge_mind.workspace.models import Contract
from forge_mind.workspace.processing import process_revision
from forge_mind.workspace.store import Store


def test_real_processing_to_sdk_isolates_body_and_explains_axial_datum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = WorkspaceService(Store(tmp_path / "store"), Executor("trusted_local"))
    monkeypatch.setattr(service, "start", lambda: pytest.fail("Read inspection must not enqueue"))
    sid = service.store.create("alice", "Measured ring occurrences", None)["id"]
    rid = "a" * 32
    directory = service.store.directory(sid, rid)
    directory.mkdir(parents=True)
    part = PartDefinition(
        id="ring",
        name="Guide ring",
        features=[
            Feature(
                id="body",
                name="Ring body",
                kind="bearing",
                params={"diameter": 20, "bore": 12.6, "width": 16.8},
            )
        ],
        output="body",
        ports=[
            PartPort(name="bad", kind="bore", diameter=12.6, origin=(0, 0, -9)),
            PartPort(name="good", kind="bore", diameter=12.6, origin=(0, 0, -7)),
        ],
    )
    document = DesignDocument(
        assembly=AssemblySpec(
            parts=[part],
            roots=[
                Occurrence(
                    id="head", definition="ring", position=(30, 40, 50), rotation=(0, 90, 0)
                ),
                Occurrence(id="other", definition="ring"),
            ],
        )
    )
    (directory / "document.json").write_text(document.model_dump_json(), encoding="utf-8")
    measurement = process_revision(
        directory,
        {
            "revision_id": rid,
            "document": document.model_dump(),
            "contract": Contract().model_dump(),
        },
        90,
        lambda: False,
    )
    assert measurement["geometry_valid"]
    assembly = measurement["facts"]["assembly"]
    assert assembly["ports"]["head.bad"]["status"] == "fail"
    assert assembly["ports"]["head.good"]["status"] == "pass"
    selected = next(row for row in assembly["occurrences"] if row["path"] == "head")
    assert selected["cylindrical_surfaces"]
    service.store.add_revision(sid, {"id": rid, "measurement": measurement})
    before_files = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    before_record = service.store.get(sid, "alice")
    app = FastAPI()
    app.include_router(router(service.store, service, lambda: "alice"))
    client = TestClient(app)

    def backend(request: httpx.Request) -> httpx.Response:
        response = client.request(
            request.method,
            request.url.path,
            headers={"Content-Type": "application/json"},
            content=request.content,
        )
        return httpx.Response(response.status_code, content=response.content)

    async def inspect() -> None:
        adapter = WorkspaceAdapter(
            "http://127.0.0.1:4321", tmp_path / "exports", transport=httpx.MockTransport(backend)
        )
        async with Client(create_server(adapter)) as sdk:
            response = await sdk.call_tool(
                "inspect_occurrence",
                {
                    "project_id": sid,
                    "revision_id": rid,
                    "occurrence_path": "head",
                    "include_image": True,
                },
            )
            assert not response.is_error, response.content
            observed = response.structured_content
            assert observed is not None
            assert json.loads(response.content[0].text) == observed
            facts, receipt = observed["facts"], observed["receipt"]
            assert facts["occurrence"]["body_index"] == selected["body_index"]
            assert facts["frame"] == "cad_z_up_mm_world"
            ports = {row["port_id"]: row for row in facts["ports"]}
            bad = next(
                row
                for row in ports["head.bad"]["surface_comparisons"]
                if row["predicate_matches"]["material_side"]
            )
            assert bad["axial_outside_distance_mm"] == pytest.approx(0.6)
            assert bad["axial_origin_projection_mm"] == pytest.approx(21)
            assert ports["head.bad"]["declaration"]["status"] == "fail"
            assert ports["head.good"]["matching_surface_count"] == 1
            assert receipt["body_index"] == selected["body_index"]
            assert receipt["selected_body_count"] == 1
            assert receipt["omitted_body_count"] == 1
            assert receipt["omitted_body_indices"] == [1 - selected["body_index"]]
            for name, key in (
                ("part.step", "step_sha256"),
                ("document.json", "document_sha256"),
                ("topology.json", "topology_sha256"),
            ):
                assert receipt[key] == hashlib.sha256(before_files[name]).hexdigest()
            assert receipt["frame_bounds"]["min"] == pytest.approx(
                selected["bbox_min_mm"], abs=1e-5
            )
            assert receipt["frame_bounds"]["max"] == pytest.approx(
                selected["bbox_max_mm"], abs=1e-5
            )
            image = base64.b64decode(response.content[1].data)
            assert image.startswith(b"\x89PNG\r\n\x1a\n")
            assert hashlib.sha256(image).hexdigest() == receipt["image"]["sha256"]
            assert "display" in receipt and "not manufacturing" in receipt["display"]["purpose"]

    asyncio.run(inspect())
    assert service.store.get(sid, "alice") == before_record
    assert service.store.revision(sid, rid)["measurement"] == measurement
    assert {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    } == before_files
    with service.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
