"""Rebuild the recorded idler engineering revision through normal workspace validation."""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.engineering_service import report_path, run_report_job
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract, DesignPlan
from forge_mind.workspace.parameter_study import parameter_study, summarize_report
from forge_mind.workspace.store import Store
from idler_mechanical_revision import revised


def located_mount_contract(contract: Contract) -> Contract:
    """Count the original four mounting locations, allowing new attachment holes.

    The independent 100x60 pattern and 6.6 diameter requirements are unchanged.
    This instrument adjustment is recorded explicitly in the run output.
    """
    result = contract.model_copy(deep=True)
    total = next(r for r in result.requirements if r.id == "base_holes")
    result.requirements.remove(total)
    for index, (x, y) in enumerate([(-50, -30), (-50, 30), (50, -30), (50, 30)]):
        result.requirements.append(
            total.model_copy(
                update={"id": f"base_hole_{index}", "center": (float(x), float(y)), "value": 1.0}
            )
        )
    return Contract.model_validate(result.model_dump())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", action="store_true")
    parser.add_argument("--corners", action="store_true")
    parser.add_argument("--store", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "rebuild.json").exists():
        raise ValueError("Use a new output directory; existing evidence is immutable")
    raw = json.loads(args.source.read_text(encoding="utf-8"))
    document = revised(DesignDocument.model_validate(raw["document"]))
    contract = located_mount_contract(Contract.model_validate(raw["contract"]))
    (args.output / "source-contract.json").write_text(
        json.dumps(raw["contract"], indent=2), encoding="utf-8"
    )
    (args.output / "contract.json").write_text(contract.model_dump_json(indent=2), encoding="utf-8")
    store = Store(args.store or args.output / "store")
    owner = "local-owner"
    sid = store.create(
        owner,
        "Engineering revision of the saved natural-language belt idler",
        contract.model_dump(),
    )["id"]
    service = WorkspaceService(store, Executor("trusted_local"))
    plan_path = (
        Path(__file__).resolve().parents[1]
        / "docs/benchmarks/engineering-idler/mechanical-plan.json"
    )
    plan = DesignPlan.model_validate_json(plan_path.read_text(encoding="utf-8"))
    store.update(sid, design_plan=plan.model_dump())
    result = build_document(
        service,
        sid,
        owner,
        document,
        contract,
        None,
        "Mechanical closure revision; mounting count now scoped to the four original locations",
        240,
        lambda: False,
    )
    (args.output / "rebuild.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "session": sid,
                "revision": result["id"],
                "geometry_valid": result["measurement"]["geometry_valid"],
                "errors": result["measurement"]["errors"],
                "findings": result["measurement"]["findings"],
            }
        ),
        flush=True,
    )
    if result["measurement"]["geometry_valid"]:
        store.update(sid, selected_revision=result["id"], state="needs_review")
        run_report_job(
            service,
            {"session": sid, "owner": owner, "revision_id": result["id"], "max_seconds": 240},
            lambda: False,
        )
        if args.study:
            study = parameter_study(
                service,
                sid,
                owner,
                result["id"],
                contract,
                "carriage_x",
                [-15, 0, 15],
                time.monotonic() + 600,
                lambda: False,
            )
            (args.output / "study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")
            print(json.dumps({"study": study["status"], "coverage": study["coverage"]}), flush=True)
        if args.corners:
            corners = []
            for endplay, carriage, closed in itertools.product(
                [0.15, 0.35], [-15.0, 15.0], [False, True]
            ):
                candidate = document.model_copy(deep=True)
                candidate.variables.update(
                    screw_endplay=endplay,
                    carriage_x=carriage,
                    screw_float=-endplay if closed else 0.0,
                )
                revision = build_document(
                    service,
                    sid,
                    owner,
                    candidate,
                    contract,
                    result["id"],
                    f"Endplay corner {candidate.variables}",
                    240,
                    lambda: False,
                )
                measurement = revision["measurement"]
                sample = {
                    "revision_id": revision["id"],
                    "variables": candidate.variables,
                    "geometry_valid": measurement["geometry_valid"],
                    "errors": measurement["errors"],
                    "findings": measurement["findings"],
                    "interfaces": measurement["facts"].get("assembly", {}).get("mates", []),
                }
                if measurement["geometry_valid"]:
                    run_report_job(
                        service,
                        {
                            "session": sid,
                            "owner": owner,
                            "revision_id": revision["id"],
                            "max_seconds": 240,
                        },
                        lambda: False,
                    )
                    receipt = store.get(sid, owner)["reports"][-1]
                    report = json.loads(
                        report_path(service, sid, owner, receipt["id"], "report.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    sample["interference"] = summarize_report(report, measurement["facts"])
                corners.append(sample)
                (args.output / "corners.json").write_text(
                    json.dumps(corners, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            "corner": candidate.variables,
                            "valid": measurement["geometry_valid"],
                            "interference": sample.get("interference", {}).get("status"),
                        }
                    ),
                    flush=True,
                )
    (args.output / "session.json").write_text(
        json.dumps(store.get(sid, owner), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
