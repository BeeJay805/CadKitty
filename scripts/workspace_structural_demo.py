"""Exercise real queued structural studies through MCP, without any model calls.

The default coupon is a numerical verification problem with explicitly hypothetical
material and criteria, not a qualified manufactured product. --document and --studies
can instead replay a saved assembly and independently prepared component load cases.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.structural_models import StructuralStudy
from forge_mind.workspace.structural_service import validate_field


def coupon() -> dict[str, Any]:
    return {
        "name": "Linear-elastic verification coupon",
        "units": "mm",
        "variables": {},
        "features": [
            {
                "id": "beam",
                "name": "12 x 8 x 100 mm beam",
                "kind": "box",
                "params": {"x": 12, "y": 8, "z": 100},
            }
        ],
        "outputs": ["beam"],
        "production": {
            "role": "reference",
            "material": {"name": "Hypothetical numerical verification material"},
        },
    }


def cases() -> list[dict[str, Any]]:
    base = {
        "target": {"body_index": 0},
        "axis": [0, 0, 1],
        "fixed_end": "minimum",
        "material": {
            "name": "Numerical verification material",
            "source": "Controlled analytic benchmark, not a supplier material qualification",
            "density_kg_m3": 2700,
            "elastic_modulus_mpa": 70000,
            "poisson_ratio": 0,
            "yield_strength_mpa": 200,
        },
        "mesh_factors": [0.16, 0.1],
        "convergence_tolerance": 0.1,
        "max_deflection_mm": 0.1,
        "min_safety_factor": 2,
        "basis": "Independent numerical test: full lower planar cap fixed and full upper cap uniformly loaded. Hypothetical material and comparison limits do not qualify a physical product.",
    }
    return [
        {**base, "name": name, "force_n": force, "direction": direction}
        for name, force, direction in [
            ("Axial reference | 100 N", 100, [0, 0, 1]),
            ("Force scaling | 200 N", 200, [0, 0, 1]),
            ("Bending | declared deflection limit exceeded", 100, [1, 0, 0]),
        ]
    ]


async def run(args: argparse.Namespace) -> None:
    from mcp import Client, StdioServerParameters

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("--output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    document = DesignDocument.model_validate(
        json.loads(args.document.read_text(encoding="utf-8")) if args.document else coupon()
    ).model_dump()
    studies = [
        StructuralStudy.model_validate(s).model_dump()
        for s in (json.loads(args.studies.read_text(encoding="utf-8")) if args.studies else cases())
    ]
    if args.mesher:
        for study in studies:
            study["mesher"] = args.mesher
    contract = (
        json.loads(args.contract.read_text(encoding="utf-8"))
        if args.contract
        else {
            "requirements": [],
            "coverage_notes": [
                "Numerical solver and workflow verification; no physical engineering approval"
            ],
        }
    )
    report: dict[str, Any] = {
        "kind": "structural_mcp_verification",
        "model_calls": 0,
        "document": document,
        "cases": studies,
        "studies": [],
        "calls": [],
    }

    def save() -> None:
        (output / "structural-demo-report.json").write_text(
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
    async with Client(params) as client, httpx.AsyncClient(timeout=60, trust_env=False) as http:

        async def call(tool_name: str, **arguments: Any) -> Any:
            response = await client.call_tool(tool_name, arguments)
            if response.is_error:
                raise RuntimeError(f"{tool_name}: {response.content}")
            value = response.structured_content
            if value is None:
                value = json.loads(next(c.text for c in response.content if hasattr(c, "text")))
            if isinstance(value, dict) and set(value) == {"result"}:
                value = value["result"]
            report["calls"].append(
                {"tool": tool_name, "response_bytes": len(json.dumps(value).encode())}
            )
            save()
            return value

        async def poll(sid: str) -> dict[str, Any]:
            start = time.monotonic()
            while True:
                state = await call("get_project", project_id=sid)
                if state["state"] not in {"queued", "running", "cancelling"}:
                    if state["state"] in {"error", "failed", "cancelled", "interrupted"}:
                        raise RuntimeError(
                            f"Native job ended {state['state']}: {state['recent_events']}"
                        )
                    return state
                if args.timeout and time.monotonic() - start > args.timeout:
                    raise TimeoutError(
                        "Demonstration observer timed out; native job remains inspectable"
                    )
                await asyncio.sleep(2)

        await call("get_schema", kind="structural")
        created = await call("create_project", name=args.name)
        sid = report["project_id"] = created["id"]
        print(f"Created structural demonstration {sid}", flush=True)
        await call("submit_document", project_id=sid, document=document, contract=contract)
        state = await poll(sid)
        rid = report["revision_id"] = state["design_revision"]
        before = await call("get_evidence", project_id=sid, revision_id=rid)
        if not before["geometry_valid"]:
            raise RuntimeError("Demonstration geometry did not validate")
        report["initial_evidence"] = before
        for case in studies:
            print("Solving " + case["name"], flush=True)
            started = time.monotonic()
            job = await call(
                "run_structural_study",
                project_id=sid,
                revision_id=rid,
                input_version=state["input_version"],
                study=case,
            )
            state = await poll(sid)
            receipt = await call("get_structural_study", project_id=sid, study_id=job["study_id"])
            report["studies"].append(
                {"receipt": receipt, "elapsed_seconds": time.monotonic() - started}
            )
            save()
            if receipt["status"] != "computed" or receipt["stale"]:
                raise RuntimeError(f"Study did not produce current computed evidence: {receipt}")
            response = await http.get(
                f"{args.api}/workspace/sessions/{sid}/structural-studies/{job['study_id']}/field.json"
            )
            response.raise_for_status()
            if hashlib.sha256(response.content).hexdigest() != receipt["hashes"]["field.json"]:
                raise RuntimeError(
                    "Downloaded structural field failed independent hash verification"
                )
            field = response.json()
            validate_field(field)
            (output / (job["study_id"] + "-field.json")).write_bytes(response.content)
            print(
                json.dumps(
                    {
                        "name": case["name"],
                        "summary": receipt["summary"],
                        "quality": receipt["numerical_quality"]["status"],
                        "field_bytes": len(response.content),
                    }
                ),
                flush=True,
            )
        after = await call("get_evidence", project_id=sid, revision_id=rid)
        if after["findings"] != before["findings"]:
            raise RuntimeError("Structural study changed frozen findings")
        report["qualification"] = await call("get_qualification", project_id=sid, revision_id=rid)
        report["final_project"] = await call("get_project", project_id=sid)
        report["exports"] = [
            await call("export_artifact", project_id=sid, revision_id=rid, format=fmt)
            for fmt in ("step", "glb")
        ]
        if not args.document and not args.studies:
            axial, doubled, bending = [s["receipt"] for s in report["studies"]]
            expected_d = 100 * 100 / (96 * 70000)
            expected_stress = 100 / 96
            ratios = {
                "axial_displacement_error": abs(
                    axial["summary"]["max_displacement_mm"] / expected_d - 1
                ),
                "axial_stress_error": abs(
                    axial["summary"]["peak_von_mises_mpa"] / expected_stress - 1
                ),
                "force_displacement_ratio": doubled["summary"]["max_displacement_mm"]
                / axial["summary"]["max_displacement_mm"],
                "force_stress_ratio": doubled["summary"]["peak_von_mises_mpa"]
                / axial["summary"]["peak_von_mises_mpa"],
            }
            report["analytic_verification"] = ratios
            if max(ratios["axial_displacement_error"], ratios["axial_stress_error"]) > 0.03 or any(
                abs(ratios[k] - 2) > 0.04
                for k in ("force_displacement_ratio", "force_stress_ratio")
            ):
                raise RuntimeError(f"Independent analytic/scaling verification failed: {ratios}")
            if bending["comparisons"][0]["status"] != "fail":
                raise RuntimeError("Known excessive-deflection case was not detected")
        save()
        print(
            json.dumps(
                {
                    "report": str(output / "structural-demo-report.json"),
                    "project_id": sid,
                    "revision_id": rid,
                    "physical_qualification": "unverified",
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:4321")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--name", default="Structural proof | analytic and overloaded cases")
    parser.add_argument("--document", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--studies", type=Path)
    parser.add_argument("--mesher", choices=("surface", "cad"))
    parser.add_argument("--timeout", type=float)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
