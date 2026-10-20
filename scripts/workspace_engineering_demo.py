"""Build reproducible private workspace examples; optionally exercise a real Astra edit.

Run with --root pointing to a new evidence/demo directory. No publishing or deployment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.hardware import search
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store


def f(
    fid: str,
    kind: str,
    params: dict[str, float] | None = None,
    inputs: list[str] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": fid,
        "name": fid.replace("_", " ").title(),
        "kind": kind,
        "params": params or {},
        "inputs": inputs or [],
        **extra,
    }


def run(service: WorkspaceService, sid: str, payload: dict[str, Any]) -> dict[str, Any]:
    service.store.enqueue(sid, "local-owner", payload)
    job = service.store.claim()
    assert job is not None
    service.run_job(job)
    result = service.store.get(sid, "local-owner")
    if result["state"] in {"error", "interrupted", "cancelled", "budget_exhausted"}:
        raise RuntimeError(json.dumps(result["events"][-4:]))
    return result


def accept(service: WorkspaceService, session: dict[str, Any]) -> dict[str, Any]:
    proposal = session["proposal"]
    assert proposal and proposal["eligible"], session["events"][-3:]
    service.store.accept_proposal(
        session["id"],
        "local-owner",
        proposal["revision_id"],
        proposal["based_on"],
        proposal["input_version"],
    )
    return service.store.revision(session["id"], proposal["revision_id"])


def seed(
    service: WorkspaceService,
    name: str,
    features: list[dict[str, Any]],
    outputs: list[str],
    inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sid = service.store.create("local-owner", name, Contract().model_dump())["id"]
    document = DesignDocument.model_validate(
        {"name": name, "features": features, "outputs": outputs}
    )
    session = run(
        service, sid, {"document": document.model_dump(), "preview_only": True, "max_seconds": 180}
    )
    revision = accept(service, session)
    settings = dict(inputs or {})
    settings["assembly_revision"] = revision["id"]
    service.store.save_engineering(
        sid, "local-owner", 0, EngineeringInputs.model_validate(settings).model_dump()
    )
    reviewed = run(
        service,
        sid,
        {"engineering_operation": "review", "revision_id": revision["id"], "max_seconds": 300},
    )
    result = {
        "session_id": sid,
        "revision_id": revision["id"],
        "name": name,
        "valid": revision["measurement"]["geometry_valid"],
        "reports": len(reviewed.get("reports", [])),
    }
    print(json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--astra", action="store_true")
    args = parser.parse_args()
    if (args.root / "workspace.sqlite3").exists():
        raise SystemExit("Use a new evidence directory; existing designs are preserved")
    service = WorkspaceService(Store(args.root), Executor("trusted_local"))
    outcomes = []
    process = {
        "process": "fdm",
        "source": "Demonstration assumptions: 220 mm printer, 1.2 mm minimum wall",
        "build_volume_mm": [220, 220, 250],
        "min_wall_mm": 1.2,
        "max_unsupported_angle_deg": 45,
    }
    outcomes.append(
        seed(
            service,
            "Manual bracket: editable features and interfaces",
            [
                f("foot", "box", {"x": 60, "y": 40, "z": 6}),
                f("upright_blank", "box", {"x": 60, "y": 6, "z": 35}),
                f("upright", "transform", {"y": 17, "z": 14.5}, ["upright_blank"]),
                f("bracket", "union", inputs=["foot", "upright"]),
                f("bore_blank", "cylinder", {"radius": 2.5, "height": 12}),
                f("left_bore", "transform", {"x": -20, "y": -10}, ["bore_blank"]),
                f("right_bore", "transform", {"x": 20, "y": -10}, ["bore_blank"]),
                f("mounting_bracket", "cut", inputs=["bracket", "left_bore", "right_bore"]),
            ],
            ["mounting_bracket"],
            {"process": process},
        )
    )
    outcomes.append(
        seed(
            service,
            "Enclosure and removable lid: assembly and service",
            [
                f("outside", "box", {"x": 80, "y": 60, "z": 30}),
                f("cavity_blank", "box", {"x": 74, "y": 54, "z": 30}),
                f("cavity", "transform", {"z": 3}, ["cavity_blank"]),
                f("housing", "cut", inputs=["outside", "cavity"]),
                f("lid_blank", "box", {"x": 80, "y": 60, "z": 3}),
                f("lid", "transform", {"z": 22}, ["lid_blank"]),
            ],
            ["housing", "lid"],
            {
                "process": process,
                "joints": [
                    {
                        "id": "lid_slide",
                        "kind": "slider",
                        "parent": 0,
                        "child": 1,
                        "parent_origin": [0, 0, 22],
                        "child_origin": [0, 0, 22],
                        "value": 5,
                        "lower": 0,
                        "upper": 50,
                    }
                ],
                "service_paths": [
                    {
                        "id": "remove_lid",
                        "body": 1,
                        "travel": [0, 0, 40],
                        "task": "Lift lid for access",
                        "samples": 10,
                    }
                ],
            },
        )
    )
    bearing = search("bearing")[0]
    outcomes.append(
        seed(
            service,
            "Catalog bearing pair: sourced hardware and BOM",
            [
                f("bearing", "hardware", catalog_id=bearing["id"]),
                f("bearing_pair", "pattern", {"count": 2, "x": 45}, ["bearing"]),
            ],
            ["bearing_pair"],
        )
    )
    material = {
        "name": "Demonstration isotropic aluminum",
        "source": "Declared demonstration inputs, not a material certificate",
        "density_kg_m3": 2700,
        "elastic_modulus_mpa": 70000,
        "poisson_ratio": 0.3,
        "yield_strength_mpa": 240,
    }
    outcomes.append(
        seed(
            service,
            "Cantilever: analytic and two-mesh FEA comparison",
            [f("beam", "box", {"x": 5, "y": 5, "z": 30})],
            ["beam"],
            {
                "material": material,
                "loads": [
                    {
                        "id": "analytic",
                        "method": "rectangular_cantilever",
                        "force_n": 1,
                        "max_deflection_mm": 0.1,
                        "min_safety_factor": 2,
                        "source": "Demonstration 1 N transverse load",
                    },
                    {
                        "id": "fea",
                        "method": "linear_fea",
                        "force_n": 1,
                        "max_deflection_mm": 0.1,
                        "min_safety_factor": 2,
                        "mesh_factors": [0.2, 0.15],
                        "convergence_tolerance": 0.15,
                        "source": "Demonstration 1 N transverse load",
                    },
                ],
            },
        )
    )
    profile = {
        "points": [
            {"id": "a", "x": 6, "y": 0},
            {"id": "b", "x": 12, "y": 0},
            {"id": "c", "x": 12, "y": 18},
            {"id": "d", "x": 6, "y": 18},
        ],
        "entities": [
            {"id": pair, "kind": "line", "points": list(pair)} for pair in ("ab", "bc", "cd", "da")
        ],
        "constraints": [{"id": "fix_" + p, "kind": "fix", "points": [p]} for p in "abcd"],
    }
    outcomes.append(
        seed(
            service,
            "Revolved sleeve: constrained profile and tolerance stack",
            [
                f("profile", "sketch", sketch=profile),
                f("sleeve", "revolve", {"angle": 360}, ["profile"]),
            ],
            ["sleeve"],
            {
                "fits": [
                    {
                        "id": "shaft_fit",
                        "hole": {
                            "name": "Sleeve bore",
                            "nominal_mm": 12,
                            "minus_mm": 0.02,
                            "plus_mm": 0.03,
                            "source": "Demonstration machining tolerance",
                        },
                        "shaft": {
                            "name": "Mating shaft",
                            "nominal_mm": 11.9,
                            "minus_mm": 0.02,
                            "plus_mm": 0.02,
                            "source": "Demonstration shaft tolerance",
                        },
                        "min_clearance_mm": 0.04,
                        "max_clearance_mm": 0.16,
                        "fit_class": "clearance",
                        "source": "Demonstration functional allowance",
                    }
                ]
            },
        )
    )
    if args.astra:
        prompt = "Create an editable parametric mounting plate, centered at the origin: 60 mm along X, 40 mm along Y, and 6 mm along Z. Two 5 mm diameter through bores run along Z, centered at X=-20,Y=0 and X=20,Y=0. Use a typed feature document with separate bore features so one bore can be edited independently. No other features."
        sid = service.store.create("local-owner", prompt, None)["id"]
        session = run(service, sid, {"preview_only": True, "max_actions": 8, "max_seconds": 480})
        first = accept(service, session)
        assert first.get("document"), "Astra did not use editable features"
        topology = json.loads(
            (service.store.directory(sid, first["id"]) / "topology.json").read_text()
        )
        face = next(
            e
            for e in topology["entities"]
            if e["kind"] == "face"
            and str(e.get("surface", "")).endswith("CYLINDER")
            and e["center"][0] > 0
        )
        base = next(f for f in first["document"]["features"] if f["kind"] == "box")
        reference = {
            "revision_id": first["id"],
            "entity_id": face["id"],
            "signature": face["signature"],
            "body_signature": face["body_signature"],
            "kind": "face",
            "point": face["center"],
        }
        settings = EngineeringInputs.model_validate(
            {
                "annotations": [
                    {
                        "id": "edit_note",
                        "text": "Enlarge only the positive-X bore to 7 mm diameter",
                        "reference": reference,
                        "marks": [face["center"]],
                    }
                ],
                "protections": [
                    {
                        "id": "keep_plate",
                        "revision_id": first["id"],
                        "feature_id": base["id"],
                        "reason": "Keep the plate and its dependency closure unchanged",
                    }
                ],
            }
        )
        service.store.save_engineering(sid, "local-owner", 0, settings.model_dump())
        edited = run(
            service,
            sid,
            {
                "preview_only": True,
                "revision_id": first["id"],
                "instruction": "Enlarge only the annotated bore at X=20,Y=0 along Z to 7 mm diameter. Preserve the plate, the other bore and all other dimensions. Keep the existing feature document and IDs.",
                "max_actions": 8,
                "max_seconds": 480,
            },
        )
        second = accept(service, edited)
        assert second["measurement"]["geometry_valid"]
        facts = second["measurement"]["facts"]
        assert all(abs(a - b) < 1e-6 for a, b in zip(facts["extents_mm"], [60, 40, 6], strict=True))
        holes = facts["holes"]
        assert len(holes) == 2
        for x, diameter in ((-20, 5), (20, 7)):
            matching = [
                h for h in holes if abs(h["point"][0] - x) < 1e-6 and abs(h["point"][1]) < 1e-6
            ]
            assert len(matching) == 1 and abs(matching[0]["diameter_mm"] - diameter) < 1e-6
        assert all(c["status"] == "pass" for c in edited["proposal"]["protection_checks"])
        run(
            service,
            sid,
            {"engineering_operation": "review", "revision_id": second["id"], "max_seconds": 180},
        )
        outcomes.append(
            {
                "name": "Real Astra structured generation and annotated protected edit",
                "session_id": sid,
                "first": first["id"],
                "revision_id": second["id"],
                "findings": second["measurement"]["findings"],
                "protection_checks": edited["proposal"]["protection_checks"],
            }
        )
        print(json.dumps(outcomes[-1]), flush=True)
    (args.root / "demo-evidence.json").write_text(json.dumps(outcomes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
