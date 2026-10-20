"""Reproducible assistant-authored MCP acceptance fixture, not a generation benchmark.

The opposed-hand drive, bought fasteners and bearings are clearance envelopes.
Neither thread engagement, clamping force, retention nor physical manufacture is
qualified. No built-in model call occurs. Outputs must be outside the repository.
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

from forge_mind.workspace.document import DesignDocument


class Part:
    def __init__(self, identifier: str, name: str, role: str = "make") -> None:
        self.identifier, self.name, self.role = identifier, name, role
        self.features: list[dict[str, Any]] = []

    def feature(self, kind: str, inputs: list[str] | None = None, **params: float) -> str:
        identifier = f"f{len(self.features)}"
        self.features.append(
            {
                "id": identifier,
                "name": identifier,
                "kind": kind,
                "inputs": inputs or [],
                "params": params,
            }
        )
        return identifier

    def box(self, width: float, depth: float, height: float, **pose: float) -> str:
        result = self.feature("box", x=width, y=depth, z=height)
        return self.feature("transform", [result], **pose) if pose else result

    def cylinder(self, radius: float, height: float, **pose: float) -> str:
        result = self.feature("cylinder", radius=radius, height=height)
        return self.feature("transform", [result], **pose) if pose else result

    def ring(self, outer: float, inner: float, length: float) -> str:
        return self.feature("cut", [self.cylinder(outer, length), self.cylinder(inner, length + 2)])

    def document(self) -> dict[str, Any]:
        return {
            "id": self.identifier,
            "name": self.name,
            "features": self.features,
            "output": self.features[-1]["id"],
            "production": {
                "role": self.role,
                "process": "cnc" if self.role == "make" else "unspecified",
                "material": {
                    "name": "Aluminum, proposed"
                    if self.role == "make"
                    else "Steel, proposed; supplier and grade unconfirmed"
                },
                "appearance": {"color": "#7f9bad" if self.role == "make" else "#b4bbc0"},
            },
            "source": "Scripted acceptance fixture. Nominal geometry only; "
            "manufacture, procurement and mechanical function unverified.",
        }


def fixture_document() -> dict[str, Any]:
    """Build 42 separately bound occurrences with deliberate nominal clearances."""
    parts: list[dict[str, Any]] = []

    def finish(part: Part) -> None:
        parts.append(part.document())

    base = Part("base", "Mounting base")
    body = base.box(272, 112, 12)
    holes = [base.cylinder(2.75, 16, x=x, y=y) for x in (-116, 116) for y in (-42, 42)]
    holes += [base.cylinder(3.5, 16, x=x, y=y) for x in (-95, 95) for y in (-45, 45)]
    base.feature("cut", [body, *holes])
    finish(base)

    end = Part("end_plate", "Guide and drive support end plate")
    body = end.box(12, 100, 44)
    holes = [end.cylinder(6.1, 16, ry=90, y=y, z=-2) for y in (-30, 30)]
    holes += [end.cylinder(14.1, 16, ry=90, z=-2)]
    holes += [end.cylinder(2.75, 48, y=y) for y in (-42, 42)]
    end.feature("cut", [body, *holes])
    finish(end)

    rod = Part("guide_rod", "Purchased diameter 12 guide rod envelope", "buy")
    rod.cylinder(6, 260)
    finish(rod)
    bushing = Part("bushing", "Purchased plain bushing 18 OD / 12.2 ID / 20 long envelope", "buy")
    bushing.ring(9, 6.1, 20)
    finish(bushing)

    jaw = Part("jaw", "Reusable carriage and integral jaw tower")
    carriage = jaw.box(24, 84, 24)
    tower = jaw.box(16, 50, 24, x=-4, z=24)
    body = jaw.feature("union", [carriage, tower])
    holes = [jaw.cylinder(9.1, 28, ry=90, y=y) for y in (-30, 30)]
    holes += [jaw.cylinder(10.1, 28, ry=90)]
    holes += [jaw.cylinder(1.75, 28, ry=90, y=y, z=28) for y in (-16, 16)]
    jaw.feature("cut", [body, *holes])
    finish(jaw)

    pad = Part("pad", "Replaceable jaw pad with recessed fastener heads")
    body = pad.box(4, 50, 16)
    holes = [pad.cylinder(1.75, 6, ry=90, y=y) for y in (-16, 16)]
    holes += [pad.cylinder(3.75, 3.5, ry=90, x=-0.25, y=y) for y in (-16, 16)]
    pad.feature("cut", [body, *holes])
    finish(pad)
    parts[-1]["production"]["material"] = {"name": "Polymer jaw pad, grade unconfirmed"}

    screw = Part(
        "drive_envelope", "TR12x3 opposed-hand screw CLEARANCE ENVELOPE; threads absent", "buy"
    )
    screw.cylinder(6, 288)
    finish(screw)
    for handed in ("left", "right"):
        nut = Part(
            f"drive_nut_{handed}",
            f"{handed}-hand drive nut CLEARANCE ENVELOPE; engagement unverified",
            "buy",
        )
        nut.ring(10, 6.1, 18)
        finish(nut)
    bearing = Part("drive_bearing", "Drive bearing 28 OD / 12.2 ID envelope", "buy")
    bearing.ring(14, 6.1, 8)
    finish(bearing)

    wheel = Part("handwheel", "Handwheel with unqualified shaft retention")
    wheel.ring(30, 6.1, 8)
    finish(wheel)
    grip = Part("grip", "Handwheel grip")
    grip.cylinder(6, 40)
    finish(grip)

    bolt = Part("end_bolt", "M5x65 end support bolt minor-diameter envelope", "buy")
    stem = bolt.cylinder(2.4, 65, z=32.5)
    head = bolt.cylinder(4.25, 5, z=-2.5)
    bolt.feature("union", [stem, head])
    finish(bolt)
    washer = Part("washer", "M5 washer clearance envelope", "buy")
    washer.ring(5, 2.75, 1)
    finish(washer)
    nut = Part("end_nut", "M5 nut clearance envelope; threads absent", "buy")
    nut.ring(4.5, 2.75, 4)
    finish(nut)
    pad_bolt = Part("pad_bolt", "M4 pad screw minor-diameter envelope", "buy")
    head = pad_bolt.cylinder(3.5, 3.2, z=1.6)
    stem = pad_bolt.cylinder(1.6, 14, z=10.2)
    pad_bolt.feature("union", [head, stem])
    finish(pad_bolt)
    collar = Part("collar", "Drive collar envelope; set-screw retention unverified", "buy")
    body = collar.ring(11, 6.1, 6)
    bore = collar.cylinder(1.7, 26, rx=90)
    collar.feature("cut", [body, bore])
    finish(collar)
    set_screw = Part("set_screw", "Collar set screw envelope; no modeled thread", "buy")
    set_screw.cylinder(1.5, 4.8)
    finish(set_screw)
    guard = Part("side_guard", "Fixed guide channel side guard")
    guard.box(220, 4, 16)
    finish(guard)

    roots: list[dict[str, Any]] = []
    moving: dict[str, list[str]] = {"left": [], "right": []}

    def occurrence(
        identifier: str,
        definition: str,
        position: list[Any],
        rotation: list[float] | None = None,
        side: str | None = None,
    ) -> None:
        roots.append(
            {
                "id": identifier,
                "definition": definition,
                "position": position,
                "rotation": rotation or [0, 0, 0],
            }
        )
        if side:
            moving[side].append(identifier)

    occurrence("base", "base", [0, 0, 6])
    for side, sign in (("left", -1), ("right", 1)):
        occurrence(f"end_{side}", "end_plate", [116 * sign, 0, 34])
        occurrence(f"bearing_{side}", "drive_bearing", [116 * sign, 0, 32], [0, 90, 0])
        occurrence(f"collar_{side}", "collar", [127 * sign, 0, 32], [0, 90, 0])
        occurrence(f"set_screw_{side}", "set_screw", [127 * sign, 8.6, 32], [90, 0, 0])
        for index, y in enumerate((-42, 42)):
            occurrence(f"end_bolt_{side}_{index}", "end_bolt", [116 * sign, y, 0])
            occurrence(f"washer_{side}_{index}", "washer", [116 * sign, y, 56.5])
            occurrence(f"nut_{side}_{index}", "end_nut", [116 * sign, y, 59])
        occurrence(
            f"jaw_{side}", "jaw", [f"jaw_{side}_x", 0, 32], [0, 0, 180 if sign < 0 else 0], side
        )
        occurrence(
            f"pad_{side}", "pad", [f"pad_{side}_x", 0, 60], [0, 0, 180 if sign < 0 else 0], side
        )
        occurrence(
            f"drive_nut_{side}", f"drive_nut_{side}", [f"jaw_{side}_x", 0, 32], [0, 90, 0], side
        )
        for index, y in enumerate((-30, 30)):
            occurrence(
                f"bushing_{side}_{index}", "bushing", [f"jaw_{side}_x", y, 32], [0, 90, 0], side
            )
        for index, y in enumerate((-16, 16)):
            occurrence(
                f"pad_bolt_{side}_{index}",
                "pad_bolt",
                [f"pad_bolt_{side}_x", y, 60],
                [0, 90 * sign, 0],
                side,
            )
    for index, y in enumerate((-30, 30)):
        occurrence(f"guide_{index}", "guide_rod", [0, y, 32], [0, 90, 0])
    for index, y in enumerate((-52, 52)):
        occurrence(f"guard_{index}", "side_guard", [0, y, 20])
    occurrence("drive", "drive_envelope", [0, 0, 32], [0, 90, 0])
    occurrence("wheel", "handwheel", [148, 0, 32], [0, 90, 0])
    occurrence("grip", "grip", [172, 22, 32], [0, 90, 0])

    relations = [{"name": "half_gap", "operation": "divide", "a": "jaw_opening", "b": 2}]
    for component, offset in (("jaw", 16), ("pad", 2), ("pad_bolt", 0.2)):
        relations += [
            {"name": f"{component}_right_x", "operation": "add", "a": "half_gap", "b": offset},
            {
                "name": f"{component}_left_x",
                "operation": "multiply",
                "a": f"{component}_right_x",
                "b": -1,
            },
        ]
    document = {
        "name": "MCP acceptance fixture: opposed-jaw clamping concept",
        "variables": {"jaw_opening": 60},
        "relations": relations,
        "assembly": {"parts": parts, "roots": roots},
        "motion_rig": {
            "drivers": [
                {
                    "id": "opening",
                    "name": "Jaw opening",
                    "unit": "mm",
                    "minimum": 20,
                    "maximum": 100,
                    "reference": "jaw_opening",
                }
            ],
            "groups": [
                {
                    "id": side,
                    "name": f"{side.title()} carriage",
                    "driver": "opening",
                    "kind": "slide",
                    "axis": [1, 0, 0],
                    "ratio": sign * 0.5,
                    "occurrences": moving[side],
                }
                for side, sign in (("left", -1), ("right", 1))
            ],
        },
    }
    assert len(roots) == 42
    return DesignDocument.model_validate(document).model_dump()


def acceptance_contract() -> dict[str, Any]:
    requirements = [
        {
            "id": "occurrences",
            "text": "42 separate physical occurrences",
            "source_quote": "42 separate physical occurrences",
            "source": "explicit",
            "metric": "solid_count",
            "comparator": "eq",
            "value": 42,
        }
    ]
    for axis, maximum in (("x", 350), ("y", 120), ("z", 80)):
        requirements.append(
            {
                "id": f"extent_{axis}",
                "text": f"Envelope {axis} <= {maximum} mm",
                "source_quote": f"Envelope {axis} <= {maximum} mm",
                "source": "explicit",
                "metric": f"extent_{axis}",
                "comparator": "le",
                "value": maximum,
            }
        )
    requirements.append(
        {
            "id": "physical_clamping",
            "text": "Physical clamping and thread engagement",
            "source_quote": "Physical clamping and thread engagement",
            "source": "explicit",
            "metric": "unmeasured",
        }
    )
    return {
        "requirements": requirements,
        "coverage_notes": [
            "Scripted assistant-authored fixture, not a general autonomous generation benchmark.",
            "Drive and purchased fasteners are clearance envelopes; threads and retention are unqualified.",
            "Symmetric jaw motion is prescribed; drive rotation, forces and contact are not solved.",
        ],
    }


def measured_targets(evidence: dict[str, Any], opening: float) -> dict[str, Any]:
    """Check exported measurements against independent fixture acceptance numbers."""
    facts = evidence["facts"]
    assembly = facts["assembly"]
    occurrences = {item["path"]: item for item in assembly["occurrences"]}
    faces = [occurrences["pad_left"]["bbox_max_mm"][0], occurrences["pad_right"]["bbox_min_mm"][0]]
    expected = [-opening / 2, opening / 2]
    checks = {
        "42_occurrences": len(occurrences) == 42,
        "pad_faces": all(abs(a - b) < 0.001 for a, b in zip(faces, expected, strict=True)),
        "symmetric_midpoint": abs(sum(faces)) < 0.001,
        "frame_extents": all(
            abs(a - b) < 0.01 for a, b in zip(facts["extents_mm"], [336, 112, 73], strict=True)
        ),
        "unqualified_clamping_retained": any(
            row["requirement_id"] == "physical_clamping" and row["status"] == "unverified"
            for row in evidence["findings"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Independent exported-geometry targets failed: {checks}, faces={faces}")
    return {"checks": checks, "pad_faces_x": faces, "build": assembly["build"]}


async def run(args: argparse.Namespace) -> None:
    from mcp import Client, StdioServerParameters

    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[1]
    if output == repo or repo in output.parents:
        raise ValueError("--output must be outside the repository")
    output.mkdir(parents=True, exist_ok=True)
    document = fixture_document()
    (output / "fixture-document.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    if args.document_only:
        print(json.dumps({"document": str(output / "fixture-document.json"), "occurrences": 42}))
        return
    report: dict[str, Any] = {
        "kind": "scripted_mcp_acceptance_fixture",
        "model_calls": 0,
        "claims": acceptance_contract()["coverage_notes"],
        "tool_calls": [],
    }
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "forge_mind.workspace.mcp_server"],
        cwd=str(repo),
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
                value = json.loads(
                    next(item.text for item in result.content if hasattr(item, "text"))
                )
            if isinstance(value, dict) and set(value) == {"result"}:
                value = value["result"]
            report["tool_calls"].append(
                {"tool": tool_name, "response_bytes": len(json.dumps(value).encode())}
            )
            (output / "mcp-demo-report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            if tool_name == "get_project" and len(json.dumps(value).encode()) > 16384:
                raise RuntimeError("Project polling exceeded the 16 KiB acceptance budget")
            return value

        async def poll(project_id: str) -> dict[str, Any]:
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                project = await call("get_project", project_id=project_id)
                if project["state"] not in {"queued", "running", "cancelling"}:
                    if project["state"] in {"failed", "error", "cancelled", "interrupted"}:
                        raise RuntimeError(f"Fixture job ended {project['state']}: {project}")
                    return project
                await asyncio.sleep(2)
            raise TimeoutError("Fixture job still active at the demonstration timeout")

        def revision(project: dict[str, Any]) -> str:
            proposal = project.get("proposal") or {}
            latest = project.get("latest_revision") or {}
            candidate = project.get("candidate") or {}
            found = (
                proposal.get("revision_id")
                or candidate.get("revision_id")
                or (latest.get("id") if isinstance(latest, dict) else latest)
                or project.get("selected_revision")
            )
            if not found:
                raise RuntimeError(f"No candidate revision in project overview: {project}")
            return str(found)

        project = await call(
            "create_project", name="Self-centering vise | 42-component MCP demonstration"
        )
        sid = project["id"]
        report["project_id"] = sid
        print(f"Created MCP fixture project {sid}", flush=True)
        await call(
            "submit_document", project_id=sid, document=document, contract=acceptance_contract()
        )
        initial = await poll(sid)
        initial_rid = revision(initial)
        initial_evidence = await call("get_evidence", project_id=sid, revision_id=initial_rid)
        report["initial"] = {"revision_id": initial_rid, "evidence": initial_evidence}
        report["initial_measurements"] = measured_targets(initial_evidence, 60)
        await call("get_design_relationships", project_id=sid, revision_id=initial_rid)
        report["scoped_part"] = await call(
            "read_part", project_id=sid, revision_id=initial_rid, part_id="jaw"
        )
        await call(
            "edit_variables", project_id=sid, revision_id=initial_rid, variables={"jaw_opening": 80}
        )
        edited = await poll(sid)
        edited_rid = revision(edited)
        evidence = await call("get_evidence", project_id=sid, revision_id=edited_rid)
        report["edited"] = {"revision_id": edited_rid, "evidence": evidence}
        report["edited_measurements"] = measured_targets(evidence, 80)
        if not initial_evidence.get("geometry_valid") or not evidence.get("geometry_valid"):
            raise RuntimeError("Fixture did not produce valid measured geometry")
        # Narrow one reusable pad by 2 mm through a scoped assembly edit. This
        # exercises a real shape change without resubmitting the other definitions.
        scoped = await call("read_part", project_id=sid, revision_id=edited_rid, part_id="pad")
        replacement = scoped["part"]
        replacement["features"][0]["params"]["y"] = 48
        await call(
            "edit_assembly",
            project_id=sid,
            revision_id=edited_rid,
            input_version=edited["input_version"],
            edit={"parts": [replacement]},
        )
        surgical = await poll(sid)
        surgical_rid = revision(surgical)
        surgical_evidence = await call("get_evidence", project_id=sid, revision_id=surgical_rid)
        report["surgical_measurements"] = measured_targets(surgical_evidence, 80)
        previous_rows = {row["path"]: row for row in evidence["facts"]["assembly"]["occurrences"]}
        surgical_rows = {
            row["path"]: row for row in surgical_evidence["facts"]["assembly"]["occurrences"]
        }
        for identifier, row in surgical_rows.items():
            if identifier.startswith("pad_") and not identifier.startswith("pad_bolt_"):
                if abs(row["bbox_max_mm"][1] - row["bbox_min_mm"][1] - 48) > 0.001:
                    raise RuntimeError("Scoped pad edit did not measure 48 mm width")
            elif any(
                abs(row[field] - previous_rows[identifier][field]) > 0.001
                for field in ("volume_mm3",)
            ):
                raise RuntimeError(f"Scoped pad edit changed unrelated occurrence {identifier}")
        if surgical_evidence["facts"]["assembly"]["build"]["built"] != 1:
            raise RuntimeError("Scoped pad edit must rebuild exactly one definition")
        report["surgical"] = {"revision_id": surgical_rid, "evidence": surgical_evidence}
        edited_rid = surgical_rid
        structure = await call("read_assembly_structure", project_id=sid, revision_id=edited_rid)
        report["assembly_structure"] = structure
        probe = await call(
            "run_product_check",
            project_id=sid,
            revision_id=edited_rid,
            request={
                "tool": "projected_pair",
                "units": "mm",
                "a": {"occurrence_path": "pad_left"},
                "b": {"occurrence_path": "pad_right"},
                "axis": [1, 0, 0],
                "datum": [0, 0, 0],
                "expected_separation_mm": 80,
                "expected_midpoint_mm": 0,
                "tolerance_mm": 0.001,
            },
        )
        await poll(sid)
        receipt = await call("get_product_check", project_id=sid, check_id=probe["check_id"])
        report["projected_pair"] = receipt
        result = receipt["result"]
        if (
            receipt["revision_id"] != edited_rid
            or receipt["stale"]
            or not receipt["matches_current_design"]
            or result["status"] != "pass"
            or result["qualification_status"] != "unverified"
            or abs(result["results"]["ordered_separation_mm"] - 80) > 0.001
            or abs(result["results"]["boundary_midpoint_mm"]) > 0.001
        ):
            raise RuntimeError("Directional BRep probe did not establish the requested geometry")
        after_probe = await call("get_evidence", project_id=sid, revision_id=edited_rid)
        if after_probe["findings"] != surgical_evidence["findings"]:
            raise RuntimeError("Exploratory probe changed frozen acceptance findings")
        for fmt in ("step", "stl", "glb"):
            artifact = await call(
                "export_artifact", project_id=sid, revision_id=edited_rid, format=fmt
            )
            if (
                hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
                != artifact["sha256"]
            ):
                raise RuntimeError("Downloaded artifact failed independent SHA-256 verification")
            report.setdefault("exports", []).append(artifact)
        report["motion_request"] = await call(
            "run_motion_check", project_id=sid, revision_id=edited_rid
        )
        final = await poll(sid)
        report["final_overview"] = final
        check_ids = final.get("motion_check_ids", [])
        if not check_ids:
            raise RuntimeError("Motion job produced no check record")
        motion = await call("get_motion_check", project_id=sid, check_id=check_ids[-1])
        report["motion"] = motion
        if (
            motion.get("errors")
            or motion.get("collision_count") != 0
            or motion.get("completed_pairs") != motion.get("total_pairs")
            or motion.get("coverage") != "discrete_only"
        ):
            raise RuntimeError("Sampled motion did not finish all pairs with zero intersections")
        if args.accept:
            version = final.get("input_version", final.get("engineering", {}).get("version", 0))
            report["acceptance"] = await call(
                "accept_candidate",
                project_id=sid,
                revision_id=edited_rid,
                input_version=version,
                expected_revision=final.get("accepted_revision"),
            )
        report["independent_targets"] = {
            "occurrences": 42,
            "initial_pad_faces_x": [-30, 30],
            "edited_pad_faces_x": [-40, 40],
            "pad_translation_x": [-10, 10],
            "pad_face_midpoint_x": 0,
            "reference_extents_mm": [336, 112, 73],
            "guide_axis_spacing_mm": 60,
            "bushing_radial_clearance_mm": 0.1,
            "motion_opening_range_mm": [20, 100],
            "maximum_opening_jaw_to_end_plate_clearance_mm": 32,
            "physical_clamping": "unverified",
        }
    path = output / "mcp-demo-report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "project_id": sid,
                "report": str(path),
                "revision_id": edited_rid,
                "accepted": args.accept,
                "physical_clamping": "unverified",
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:4321")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument(
        "--accept", action="store_true", help="Explicitly accept the reviewed candidate"
    )
    parser.add_argument("--document-only", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
