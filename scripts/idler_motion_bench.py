"""Reproduce the 29-part prescribed idler motion and bounded collision check."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scripts.idler_mechanical_revision import revised

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.assembly import expand
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract
from forge_mind.workspace.motion_rig import MotionDriver, MotionGroup, MotionRig
from forge_mind.workspace.motion_service import check_path, run_motion_job
from forge_mind.workspace.store import Store


def motion_document(source: DesignDocument) -> DesignDocument:
    doc = revised(source)
    roller = [
        "moving/idler",
        "moving/retainer_positive",
        "moving/retainer_negative",
        *["moving/retainer_screw_" + s for s in ("pp", "pn", "np", "nn")],
    ]
    carriage = [
        r["path"]
        for r in expand(doc)
        if r["path"].startswith("moving/") and r["path"] not in roller
    ]
    doc.motion_rig = MotionRig(
        drivers=[
            MotionDriver(
                id="adjustment",
                name="Carriage adjustment",
                unit="mm",
                minimum=-15,
                maximum=15,
                reference="carriage_x",
            ),
            MotionDriver(
                id="roller", name="Idler rotation", unit="deg", minimum=0, maximum=360, reference=0
            ),
        ],
        groups=[
            MotionGroup(
                id="carriage",
                name="Carriage and axle",
                driver="adjustment",
                kind="slide",
                axis=(1, 0, 0),
                occurrences=carriage,
            ),
            MotionGroup(
                id="screw",
                name="Lead screw and handwheel",
                driver="adjustment",
                kind="rotate",
                axis=(1, 0, 0),
                pivot=(0, 0, 18),
                ratio=-360 / 1.25,
                occurrences=[
                    "adjuster",
                    "adjuster_retention",
                    "handwheel",
                    "hub_crossbolt",
                    "hub_crossnut",
                ],
            ),
            MotionGroup(
                id="idler",
                name="Idler and rotating retainers",
                parent="carriage",
                driver="roller",
                kind="rotate",
                axis=(0, 1, 0),
                pivot=("carriage_x", 0, 45),
                occurrences=roller,
            ),
        ],
    )
    doc.name = "Belt idler with interactive adjustment and rotation"
    return DesignDocument.model_validate(doc.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also compare exhaustive and optimized checks at identical configurations",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "rebuild.json").exists():
        raise ValueError("Use a fresh output directory")
    raw = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs/benchmarks/engineering-idler/source-revision.json"
        ).read_text(encoding="utf-8")
    )
    doc = motion_document(DesignDocument.model_validate(raw["document"]))
    # Same four-location scope as the preceding mechanical benchmark: new holes
    # must not invalidate the original mounting pattern's independent targets.
    contract = Contract.model_validate(raw["contract"])
    total = next(r for r in contract.requirements if r.id == "base_holes")
    contract.requirements.remove(total)
    for index, (x, y) in enumerate([(-50, -30), (-50, 30), (50, -30), (50, 30)]):
        contract.requirements.append(
            total.model_copy(
                update={"id": f"base_hole_{index}", "center": (float(x), float(y)), "value": 1.0}
            )
        )
    store = Store(args.store or args.output / "store")
    sid = store.create("local-owner", "Belt idler: interactive motion", contract.model_dump())["id"]
    store.update(sid, title="Belt idler: interactive motion")
    service = WorkspaceService(store, Executor("trusted_local"))
    result = build_document(
        service,
        sid,
        "local-owner",
        doc,
        contract,
        None,
        "Prescribed slide, screw lead and idler rotation; contact physics unverified",
        240,
        lambda: False,
    )
    (args.output / "rebuild.json").write_text(json.dumps(result), encoding="utf-8")
    if not result["measurement"]["geometry_valid"]:
        raise ValueError(str(result["measurement"]["errors"]))
    store.update(sid, selected_revision=result["id"], state="needs_review")
    job = {"session": sid, "owner": "local-owner", "revision_id": result["id"], "settings": {}}
    run_motion_job(service, job, lambda: False)
    receipt = store.get(sid, "local-owner")["motion_checks"][-1]
    check = json.loads(
        check_path(service, sid, "local-owner", receipt["id"]).read_text(encoding="utf-8")
    )
    (args.output / "motion-check.json").write_text(json.dumps(check), encoding="utf-8")
    start = time.perf_counter()
    run_motion_job(service, job, lambda: False)
    summary = {
        "session": sid,
        "revision": result["id"],
        "geometry_valid": True,
        "check": check,
        "cached_seconds": time.perf_counter() - start,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.compare:
        import build123d as b
        import numpy as np
        import psutil

        from forge_mind.workspace.motion_check import (
            MotionCheckSettings,
            configurations,
            review_poses,
        )
        from forge_mind.workspace.motion_rig import group_matrices

        process = psutil.Process()
        original_affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None
        if original_affinity:
            process.cpu_affinity(original_affinity[:2])
        try:
            motion = result["measurement"]["facts"]["motion_rig"]
            solids = list(
                b.import_step(str(store.directory(sid, result["id"]) / "part.step")).solids()
            )
            bindings = sorted(motion["bindings"], key=lambda r: r["body_index"])
            poses = []
            for values in configurations(motion["rig"], MotionCheckSettings()):
                matrices = group_matrices(motion["rig"], values)
                poses.append((values, [matrices.get(r["group"], np.eye(4)) for r in bindings]))
            comparisons = {}
            for name, optimized in [("exhaustive", False), ("optimized", True)]:
                comparisons[name] = review_poses(
                    solids,
                    [r["path"] for r in bindings],
                    [r["group"] for r in bindings],
                    poses,
                    optimized=optimized,
                    seconds=240,
                )
                (args.output / (name + ".json")).write_text(
                    json.dumps(comparisons[name]), encoding="utf-8"
                )
            if (
                comparisons["exhaustive"]["coverage"] != "discrete_only"
                or comparisons["optimized"]["coverage"] != "discrete_only"
                or comparisons["exhaustive"]["collisions"] != comparisons["optimized"]["collisions"]
            ):
                raise ValueError("Comparison incomplete or collision witnesses differ")
        finally:
            if original_affinity:
                process.cpu_affinity(original_affinity)
    print(json.dumps({k: v for k, v in summary.items() if k != "check"}), flush=True)


if __name__ == "__main__":
    main()
