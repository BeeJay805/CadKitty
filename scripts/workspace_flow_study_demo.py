"""Run a real two-shape CFD design study through MCP; retain numerical failures honestly."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
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
    if (output / "flow-study-report.json").exists():
        raise ValueError(
            "Choose a fresh output directory; an existing native report must be preserved"
        )
    settings = json.loads(args.settings.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "model_calls": 0,
        "physical_measurements": 0,
        "scope": "Two actual constant-volume CAD candidates under one frozen numerical protocol. Not an aircraft qualification.",
        "calls": [],
    }

    def save() -> None:
        (output / "flow-study-report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "forge_mind.workspace.mcp_server"],
        cwd=str(root),
        env={**os.environ, "FORGE_MCP_WORKSPACE_URL": args.api},
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
            report["calls"].append({"tool": tool_name, "time": time.time()})
            save()
            return data

        async def settled(sid: str) -> dict[str, Any]:
            while True:
                state = await call("get_project", project_id=sid)
                if state["state"] not in {"queued", "running", "cancelling"}:
                    return state
                await asyncio.sleep(5)

        project = await call("create_project", name="Flow study | constant-volume shape comparison")
        sid = report["project_id"] = project["id"]
        document = {
            "name": "External-flow shape control",
            "variables": {"aspect": 1},
            "features": [
                {
                    "id": "body",
                    "name": "Constant-volume smooth body",
                    "kind": "python",
                    "bindings": {"aspect": "aspect"},
                    "code": "a = params['aspect']\nbase = b.Sphere(1000)\nresult = base if a == 1 else b.scale(base, by=(a,a**-0.5,a**-0.5))\nif a != 1:\n    result = b.scale(result, by=(base.volume/result.volume)**(1/3))",
                }
            ],
            "outputs": ["body"],
        }
        contract = {
            "requirements": [
                {
                    "id": "single_solid",
                    "text": "One closed solid",
                    "source_quote": "One closed solid",
                    "source": "explicit",
                    "metric": "solid_count",
                    "value": 1,
                }
            ],
            "coverage_notes": [
                "Numerical flow comparison only; no manufacture or flight qualification"
            ],
        }
        await call("submit_document", project_id=sid, document=document, contract=contract)
        state = await settled(sid)
        rid = report["baseline_revision_id"] = state["design_revision"]
        assert (await call("get_evidence", project_id=sid, revision_id=rid))["geometry_valid"]
        nominal_volume = 4 * math.pi * 1000**3 / 3
        study = await call(
            "create_design_study",
            project_id=sid,
            request={
                "revision_id": rid,
                "input_version": 0,
                "spec": {
                    "name": "Compare drag while retaining nominal solid volume",
                    "objectives": [{"metric": "flow.drag_n", "direction": "minimize"}],
                    "constraints": [
                        {
                            "metric": "geometry.volume_mm3",
                            "operator": "ge",
                            "value": nominal_volume * 0.9999,
                        },
                        {
                            "metric": "geometry.volume_mm3",
                            "operator": "le",
                            "value": nominal_volume * 1.0001,
                        },
                    ],
                    "variables": {"aspect": [1, 2]},
                    "evaluators": [{"kind": "geometry"}, {"kind": "flow", "settings": settings}],
                    "comparison_tolerance": 0.02,
                },
            },
        )
        study_id = report["study_id"] = study["id"]
        # Validate every CAD candidate before spending time on any flow solve.
        await call("edit_variables", project_id=sid, revision_id=rid, variables={"aspect": 1.5})
        state = await settled(sid)
        candidate_rid = report["candidate_revision_id"] = state["design_revision"]
        assert candidate_rid != rid, "Candidate build did not produce a new revision"
        candidate_evidence = await call("get_evidence", project_id=sid, revision_id=candidate_rid)
        assert candidate_evidence["geometry_valid"], "Candidate CAD preflight failed"
        report["cad_preflight_passed"] = True
        save()
        await call(
            "add_design_candidate",
            project_id=sid,
            study_id=study_id,
            request={
                "label": "Longer, narrower, equal nominal volume",
                "revision_id": candidate_rid,
            },
        )
        await call("run_design_study", project_id=sid, study_id=study_id)
        print(
            json.dumps({"project_id": sid, "study_id": study_id, "stage": "evaluating"}), flush=True
        )
        if args.resume_control:
            while True:
                progress = await call(
                    "get_design_study_progress", project_id=sid, study_id=study_id
                )
                snapshots = progress.get("flow_progress", [])
                if any(
                    p.get("completed_stages", 0) >= 1 and p.get("phase") == "solving"
                    for p in snapshots
                ):
                    report["before_interruption"] = progress
                    await call("cancel_job", project_id=sid)
                    await settled(sid)
                    report["interrupted_study"] = await call(
                        "get_design_study", project_id=sid, study_id=study_id
                    )
                    await call(
                        "run_design_study", project_id=sid, study_id=study_id, retry_failed=True
                    )
                    report["resume_requested"] = True
                    save()
                    break
                if progress["status"] not in {"queued", "running"}:
                    report["resume_control_unavailable"] = (
                        "Run stopped before a second live solver stage"
                    )
                    save()
                    break
                await asyncio.sleep(2)
        await settled(sid)
        study = report["study"] = await call("get_design_study", project_id=sid, study_id=study_id)
        assert len(study["candidates"]) == 2
        # A negative numerical outcome is retained as evidence, never asserted to be a winner.
        best = study["comparison"]["best_candidate_id"]
        if best:
            winner = next(c for c in study["candidates"] if c["id"] == best)
            assert winner["eligible"] is True and winner["feasible"] is True
        report["workflow_completed"] = study["status"] == "completed"
        report["final_progress"] = await call(
            "get_design_study_progress", project_id=sid, study_id=study_id
        )
        if report.get("resume_requested"):
            assert any(
                p.get("reused_stages", 0) >= 1
                for p in report["final_progress"].get("flow_progress", [])
            ), "Completed native stage was not reused after interruption"
        save()
        print(
            json.dumps(
                {
                    "report": str(output / "flow-study-report.json"),
                    "project_id": sid,
                    "comparison": study["comparison"],
                    "candidate_states": [
                        {
                            "label": c["label"],
                            "status": c["status"],
                            "eligible": c["eligible"],
                            "error": c.get("error"),
                        }
                        for c in study["candidates"]
                    ],
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:4321")
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume-control",
        action="store_true",
        help="Cancel once during a second solver stage, then verify completed-stage reuse",
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
