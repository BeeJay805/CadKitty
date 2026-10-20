"""Exercise general design iteration through real stdio MCP, HTTP and exported CAD."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


async def run(args: argparse.Namespace) -> None:
    from mcp import Client, StdioServerParameters

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Keep generated artifacts outside the source repository")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "model_calls": 0,
        "physical_measurements": 0,
        "calls": [],
        "scope": "Native software workflow control, not aerodynamic or physical qualification",
    }

    def save() -> None:
        (output / "design-study-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "forge_mind.workspace.mcp_server"],
        cwd=str(root),
        env={
            **os.environ,
            "FORGE_MCP_WORKSPACE_URL": args.api,
            "FORGE_MCP_EXPORT_DIR": str(output / "exports"),
        },
    )
    async with Client(params) as client:

        async def call(tool_name: str, **arguments: Any) -> Any:
            result = await client.call_tool(tool_name, arguments)
            if result.is_error:
                raise RuntimeError(f"{tool_name}: {result.content}")
            data = result.structured_content
            if data is None:
                data = json.loads(next(c.text for c in result.content if hasattr(c, "text")))
            if isinstance(data, dict) and set(data) == {"result"}:
                data = data["result"]
            report["calls"].append(
                {"tool": tool_name, "response_bytes": len(json.dumps(data).encode())}
            )
            save()
            return data

        async def settled(sid: str) -> dict[str, Any]:
            while True:
                state = await call("get_project", project_id=sid)
                if state["state"] not in {"queued", "running", "cancelling"}:
                    if state["state"] in {"error", "failed", "cancelled", "interrupted"}:
                        raise RuntimeError(f"Native job failed: {state}")
                    return state
                await asyncio.sleep(2)

        schema = await call("get_design_study_schema")
        assert {e["kind"] for e in schema["evaluators"]} == {"geometry", "structure", "flow"}
        report["resource_policy"] = schema["resource_policy"]
        project = await call("create_project", name="Design studies | measured material search")
        sid = report["project_id"] = project["id"]
        document = {
            "name": "Material search control",
            "variables": {"width": 8, "thickness": 8},
            "features": [
                {
                    "id": "body",
                    "name": "Constant-span specimen",
                    "kind": "box",
                    "params": {"x": 100},
                    "bindings": {"y": "width", "z": "thickness"},
                }
            ],
            "outputs": ["body"],
        }
        contract = {
            "requirements": [
                {
                    "id": "span",
                    "text": "Preserve the 100 mm span",
                    "source_quote": "Preserve the 100 mm span",
                    "source": "explicit",
                    "metric": "extent_x",
                    "value": 100,
                    "tolerance": 0.001,
                }
            ],
            "coverage_notes": [
                "Geometry-only benchmark. No strength or product qualification claim."
            ],
        }
        await call("submit_document", project_id=sid, document=document, contract=contract)
        state = await settled(sid)
        rid = report["baseline_revision_id"] = state["design_revision"]
        initial = report["initial_evidence"] = await call(
            "get_evidence", project_id=sid, revision_id=rid
        )
        assert initial["geometry_valid"]
        first = await call(
            "read_design_page", project_id=sid, revision_id=rid, collection="variables", limit=1
        )
        second = await call(
            "read_design_page",
            project_id=sid,
            revision_id=rid,
            collection="variables",
            limit=1,
            cursor=first["next_cursor"],
        )
        assert (
            first["items"][0]["name"] != second["items"][0]["name"]
            and second["next_cursor"] is None
        )
        study = await call(
            "create_design_study",
            project_id=sid,
            request={
                "revision_id": rid,
                "input_version": 0,
                "spec": {
                    "name": "Less material with frozen span and volume constraints",
                    "objectives": [{"metric": "geometry.volume_mm3", "direction": "minimize"}],
                    "constraints": [
                        {"metric": "geometry.volume_mm3", "operator": "ge", "value": 3000},
                        {"metric": "geometry.extent_x_mm", "operator": "ge", "value": 100},
                    ],
                    "variables": {"width": [2, 8], "thickness": [2, 8]},
                    "evaluators": [{"kind": "geometry"}],
                    "comparison_tolerance": 0.001,
                },
            },
        )
        study_id = report["study_id"] = study["id"]
        started = time.monotonic()
        await call(
            "search_design_study",
            project_id=sid,
            study_id=study_id,
            request={"rounds": 2, "step_fraction": 0.5},
        )
        await settled(sid)
        study = report["study"] = await call("get_design_study", project_id=sid, study_id=study_id)
        report["search_seconds"] = time.monotonic() - started
        assert study["status"] == "completed", study
        assert study["comparison"]["conclusion"] == "improved", study
        assert any(c["feasible"] is False for c in study["candidates"])
        best = next(
            c for c in study["candidates"] if c["id"] == study["comparison"]["best_candidate_id"]
        )
        assert 3000 <= best["metrics"]["geometry.volume_mm3"] < 6400
        assert abs(best["metrics"]["geometry.extent_x_mm"] - 100) < 1e-6
        preserved = {
            c["id"]: c["evaluations"] for c in study["candidates"] if c["status"] == "complete"
        }
        await call("run_design_study", project_id=sid, study_id=study_id)
        await settled(sid)
        resumed = await call("get_design_study", project_id=sid, study_id=study_id)
        assert all(
            c["evaluations"] == preserved[c["id"]]
            for c in resumed["candidates"]
            if c["id"] in preserved
        )
        before = await call("get_project", project_id=sid)
        assert before["design_revision"] == rid
        await call(
            "propose_study_candidate",
            project_id=sid,
            study_id=study_id,
            request={"candidate_id": best["id"], "expected_current_revision": rid},
        )
        proposed = report["proposal"] = await settled(sid)
        evidence = report["proposed_evidence"] = await call(
            "get_evidence", project_id=sid, revision_id=proposed["design_revision"]
        )
        assert evidence["geometry_valid"] and evidence["contract_hash"] == initial["contract_hash"]
        report["all_native_controls_passed"] = True
        save()
        print(
            json.dumps(
                {
                    "report": str(output / "design-study-report.json"),
                    "project_id": sid,
                    "study_id": study_id,
                    "candidates": len(study["candidates"]),
                    "baseline_volume_mm3": 6400,
                    "best_volume_mm3": best["metrics"]["geometry.volume_mm3"],
                    "search_seconds": report["search_seconds"],
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:4321")
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
