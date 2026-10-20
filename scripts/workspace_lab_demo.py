"""Reproducible native MCP engineering lab exercise. All observations are synthetic controls."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from forge_mind.workspace.experiment_service import synthetic_tutorial
from forge_mind.workspace.synthesis_models import example_synthesis_request


def specimen() -> dict[str, Any]:
    return {
        "name": "Experimental specimen | declared rectangular beam",
        "variables": {"thickness": 4},
        "features": [
            {
                "id": "beam",
                "name": "Rectangular test section",
                "kind": "box",
                "params": {"x": 180, "y": 20, "z": 4},
                "bindings": {"z": "thickness"},
            }
        ],
        "outputs": ["beam"],
        "production": {"role": "reference", "material": {"name": "Synthetic experiment specimen"}},
    }


def observation(identifier: str, span: float, force: float) -> dict[str, Any]:
    # Independent manufactured-data control. This function never reads a fitted prediction.
    return {
        "observation_id": identifier,
        "span_mm": span,
        "force_n": force,
        "force_uncertainty_n": 0.01,
        "displacement_uncertainty_mm": 0.001,
        "displacement_mm": force * (4 * span**3 / (70000 * 20 * 4**3) + 0.005),
    }


async def run(args: argparse.Namespace) -> None:
    from mcp import Client, StdioServerParameters

    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Keep native artifacts outside the source repository")
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "kind": "engineering_lab_native_mcp",
        "model_calls": 0,
        "physical_measurements": 0,
        "provenance": "synthetic",
        "records": {},
        "realizations": [],
        "calls": [],
    }

    def save() -> None:
        (output / "lab-demo-report.json").write_text(
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
            value = result.structured_content
            if value is None:
                value = json.loads(next(c.text for c in result.content if hasattr(c, "text")))
            if isinstance(value, dict) and set(value) == {"result"}:
                value = value["result"]
            report["calls"].append(
                {"tool": tool_name, "response_bytes": len(json.dumps(value).encode())}
            )
            save()
            return value

        async def settled(sid: str) -> dict[str, Any]:
            while True:
                state = await call("get_project", project_id=sid)
                if state["state"] not in {"queued", "running", "cancelling"}:
                    if state["state"] in {"error", "failed", "cancelled", "interrupted"}:
                        raise RuntimeError(f"Native CAD job failed: {state}")
                    return state
                await asyncio.sleep(2)

        contract = {
            "requirements": [
                {
                    "id": "section_width",
                    "text": "Keep the section 20 mm wide",
                    "source_quote": "Keep the section 20 mm wide",
                    "source": "explicit",
                    "metric": "extent_y",
                    "value": 20,
                    "tolerance": 0.001,
                },
                {
                    "id": "deflection",
                    "text": "Observe at most 0.8 mm deflection at span 160 mm and force 10 N",
                    "source_quote": "Observe at most 0.8 mm deflection at span 160 mm and force 10 N",
                    "source": "explicit",
                    "metric": "unmeasured",
                    "value": 0.8,
                },
            ],
            "coverage_notes": [
                "Independent synthetic workflow controls; no physical qualification"
            ],
        }
        created = await call("create_project", name="Engineering Lab | challenge, test, improve")
        sid = report["project_id"] = created["id"]
        await call("submit_document", project_id=sid, document=specimen(), contract=contract)
        state = await settled(sid)
        rid = report["revision_id"] = state["design_revision"]
        baseline = await call("get_evidence", project_id=sid, revision_id=rid)
        assert baseline["geometry_valid"]
        report["initial_evidence"] = baseline
        print(
            json.dumps({"stage": "initial_cad", "project_id": sid, "revision_id": rid}), flush=True
        )

        tutorial = synthetic_tutorial()
        training = [observation(f"span80_{i}", 80, force) for i, force in enumerate((5, 10, 15))]

        async def dataset(
            title: str, rows: list[dict[str, Any]], project_id: str = sid, revision_id: str = rid
        ) -> dict[str, Any]:
            return await call(
                "import_experiment",
                project_id=project_id,
                experiment={
                    "revision_id": revision_id,
                    "expected_input_version": 0,
                    "title": title,
                    "provenance": "synthetic",
                    "protocol": tutorial["protocol"],
                    "observations": rows,
                },
            )

        async def fit(data: dict[str, Any], project_id: str = sid) -> dict[str, Any]:
            return await call(
                "calibrate_experiment",
                project_id=project_id,
                calibration={
                    "experiment_id": data["id"],
                    "observation_ids": [r["observation_id"] for r in data["observations"]],
                },
            )

        async def predict(fitted: dict[str, Any], hypothesis: str, span: float) -> dict[str, Any]:
            return await call(
                "register_prediction",
                project_id=sid,
                prediction={
                    "calibration_id": fitted["id"],
                    "hypothesis": hypothesis,
                    "condition": {"span_mm": span, "force_n": 10, "force_uncertainty_n": 0.01},
                    "comparison_tolerance_mm": 0.01,
                    "planned_displacement_uncertainty_mm": 0.001,
                    "rationale": "Prospective changed-span control; hidden response evaluated only after commitment",
                },
            )

        initial = await dataset("Ambiguous single-span experiment | synthetic", training)
        ambiguous = await fit(initial)
        assert ambiguous["status"] == "ambiguous"
        wrong = await predict(ambiguous, "material_stiffness", 160)
        actual160 = observation("first_heldout_160", 160, 10)
        failed = await call(
            "record_experiment_outcome",
            project_id=sid,
            outcome={
                "prediction_id": wrong["id"],
                "provenance": "synthetic",
                "observation": actual160,
                "acquisition_note": "Independent synthetic fixture-compliance model; no hardware measurement",
            },
        )
        assert failed["status"] == "outside_tolerance"
        corrected_data = await dataset(
            "Discriminating span experiment | synthetic", [*training, actual160]
        )
        corrected = await fit(corrected_data)
        assert corrected["selected_hypothesis"] == "fixture_translation", corrected
        prospective = await predict(corrected, "fixture_translation", 140)
        successful = await call(
            "record_experiment_outcome",
            project_id=sid,
            outcome={
                "prediction_id": prospective["id"],
                "provenance": "synthetic",
                "observation": observation("second_heldout_140", 140, 10),
                "acquisition_note": "Independent synthetic changed-span control evaluated after commitment",
            },
        )
        assert successful["status"] == "within_tolerance"
        assert (
            wrong["sequence"]
            < failed["sequence"]
            < prospective["sequence"]
            < successful["sequence"]
        )
        report["records"].update(
            initial=initial,
            ambiguous=ambiguous,
            wrong_prediction=wrong,
            failed_outcome=failed,
            corrected_data=corrected_data,
            corrected_calibration=corrected,
            prediction=prospective,
            outcome=successful,
        )
        print(
            json.dumps(
                {
                    "stage": "hypothesis_correction",
                    "first_error_mm": failed["error_mm"],
                    "next_error_mm": successful["error_mm"],
                    "selected": corrected["selected_hypothesis"],
                }
            ),
            flush=True,
        )

        rationale = await call(
            "record_design_rationale",
            project_id=sid,
            rationale={
                "revision_id": rid,
                "input_version": 0,
                "target": {"feature_id": "beam"},
                "purpose": "A rectangular specimen distinguishes material stiffness from fixture motion when tested at multiple spans",
                "variable_names": ["thickness"],
                "requirement_ids": ["deflection"],
                "evidence_ids": ["lab:" + corrected["id"]],
            },
        )
        report["records"]["rationale"] = rationale
        improve_request = {
            "variable_name": "thickness",
            "hypothesis": "fixture_translation",
            "condition": {"span_mm": 160, "force_n": 10, "force_uncertainty_n": 0.01},
            "target_displacement_mm": 0.8,
            "min_thickness_mm": 4,
            "max_thickness_mm": 6,
            "mapping_basis": "Direct output beam box z dimension; y=20 mm and x=180 mm remain fixed",
            "feature_id": "beam",
            "thickness_parameter": "z",
        }
        recommendation = await call(
            "propose_calibrated_design",
            project_id=sid,
            calibration_id=corrected["id"],
            request=improve_request,
        )
        report["records"]["design_recommendation"] = recommendation
        report["explanation"] = await call(
            "explain_design_part", project_id=sid, revision_id=rid, body_index=0
        )

        request = example_synthesis_request().model_dump()
        comparison = await call(
            "compare_mechanism_architectures",
            project_id=sid,
            comparison={"revision_id": rid, "input_version": 0, "request": request},
        )
        assert all(c["status"] == "fail" for c in comparison["candidates"])
        improved_request = copy.deepcopy(request)
        improved_request["candidates"][0]["lead_mm"] = 8
        improved = await call(
            "compare_mechanism_architectures",
            project_id=sid,
            comparison={"revision_id": rid, "input_version": 0, "request": improved_request},
        )
        assert improved["task_hash"] == comparison["task_hash"]
        screw = improved["candidates"][0]
        assert screw["status"] == "pass", screw
        assert math.isclose(screw["metrics"]["close_time_s"], 3.75, abs_tol=1e-9)
        report["records"].update(comparison=comparison, improved_comparison=improved)
        for candidate in improved["candidates"]:
            started = time.monotonic()
            queued = await call(
                "realize_mechanism_candidate",
                project_id=sid,
                comparison_id=improved["id"],
                candidate_id=candidate["id"],
                name="Lab alternative | " + candidate["name"],
            )
            built = await settled(queued["project_id"])
            evidence = await call(
                "get_evidence",
                project_id=queued["project_id"],
                revision_id=built["design_revision"],
            )
            assert evidence["geometry_valid"] and not evidence["requirements_satisfied"]
            export = await call(
                "export_artifact",
                project_id=queued["project_id"],
                revision_id=built["design_revision"],
                format="step",
            )
            report["realizations"].append(
                {
                    "candidate_id": candidate["id"],
                    "project_id": queued["project_id"],
                    "revision_id": built["design_revision"],
                    "evidence": evidence,
                    "export": export,
                    "elapsed_seconds": time.monotonic() - started,
                }
            )
            print(
                json.dumps(
                    {
                        "stage": "realized",
                        "candidate": candidate["id"],
                        "solids": evidence["facts"]["solid_count"],
                    }
                ),
                flush=True,
            )

        # Exercise real challenge admission/build separately, retaining the main notebook on its original geometry.
        fork = await call("create_project", name="Lab specimen | rebuilt conditional section")
        fsid = fork["id"]
        await call("submit_document", project_id=fsid, document=specimen(), contract=contract)
        original = await settled(fsid)
        frid = original["design_revision"]
        data = await dataset(
            "Supplied synthetic two-span calibration", [*training, actual160], fsid, frid
        )
        fitted = await fit(data, fsid)
        proposal = await call(
            "propose_calibrated_design",
            project_id=fsid,
            calibration_id=fitted["id"],
            request=improve_request,
        )
        await call("apply_design_challenge", project_id=fsid, challenge_id=proposal["challenge_id"])
        changed = await settled(fsid)
        evidence = await call(
            "get_evidence", project_id=fsid, revision_id=changed["design_revision"]
        )
        assert evidence["geometry_valid"]
        assert math.isclose(
            evidence["facts"]["extents_mm"][2],
            proposal["prediction"]["recommended_thickness_mm"],
            rel_tol=1e-8,
        )
        assert evidence["contract_hash"] == baseline["contract_hash"]
        stale = await call("get_lab_record", project_id=fsid, record_id=fitted["id"])
        assert stale["stale"]
        report["rebuilt_specimen"] = {
            "project_id": fsid,
            "revision_id": changed["design_revision"],
            "evidence": evidence,
            "proposal": proposal,
            "historical_calibration_stale": stale["stale"],
        }
        after = await call("get_evidence", project_id=sid, revision_id=rid)
        assert after["findings"] == baseline["findings"]
        report["final_project"] = await call("get_project", project_id=sid)
        report["physical_qualification"] = "unverified"
        save()
        print(
            json.dumps(
                {
                    "report": str(output / "lab-demo-report.json"),
                    "project_id": sid,
                    "physical_measurements": 0,
                    "all_native_controls_passed": True,
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
