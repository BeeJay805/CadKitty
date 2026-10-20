"""Bounded assembly regression benchmark. Templates and real AI trials are reported separately."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.document_service import build_document
from forge_mind.workspace.engineering_service import release_package, run_report_job
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.steering_template import SteeringTemplate, steering_document
from forge_mind.workspace.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ai", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model": "gpt-6-astra",
        "provider": "codex subscription" if args.ai else "deterministic",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "engine": engine_fingerprint(),
        "started": time.time(),
        "scope": "Template/perturbation engineering benchmark. AI edit trial is separate from unconstrained generation.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    store = Store(args.output / "data")
    contract = Contract(
        requirements=[
            Requirement(
                id="function",
                text="Editable steering reference, preserved parts and interfaces; vehicle performance unverified",
                source_quote="steering",
                source="explicit",
            )
        ]
    )
    sid = store.create(
        "local-owner", "Editable steering development reference", contract.model_dump()
    )["id"]
    service = WorkspaceService(store, Executor("trusted_local"))
    rows: list[dict[str, Any]] = []
    parent = None
    original = steering_document(SteeringTemplate())
    last = None
    for name, variables in [
        ("reference", {}),
        ("wheel_220", {"wheel_diameter": 220}),
        ("track_650", {"track_width": 650}),
        ("steer_15", {"steering_angle": 15}),
    ]:
        doc = (
            original.model_copy(deep=True)
            if last is None
            else DesignDocument.model_validate(last["document"])
        )
        doc.variables.update(variables)
        rev = build_document(
            service, sid, "local-owner", doc, contract, parent, name, 120, lambda: False
        )
        facts = rev["measurement"]["facts"]
        rows.append(
            {
                "case": name,
                "revision": rev["id"],
                "seconds": rev["execution_seconds"],
                "geometry_valid": rev["measurement"]["geometry_valid"],
                "errors": rev["measurement"]["errors"],
                "assembly": facts.get("assembly"),
                "motion_status": facts.get("mechanism", {}).get("status"),
            }
        )
        (args.output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(json.dumps({k: v for k, v in rows[-1].items() if k != "assembly"}), flush=True)
        if not rev["measurement"]["geometry_valid"]:
            break
        parent, last = rev["id"], rev
    if last:
        store.update(sid, selected_revision=last["id"])
        run_report_job(
            service,
            {"session": sid, "owner": "local-owner", "revision_id": last["id"], "max_seconds": 300},
            lambda: False,
        )
        final = store.get(sid, "local-owner")
        package = release_package(service, sid, "local-owner", last["id"])
        (args.output / "package.json").write_text(
            json.dumps(
                {"path": str(package), "session": sid, "report": final["reports"][-1]}, indent=2
            ),
            encoding="utf-8",
        )
        if args.ai:
            store.enqueue(
                sid,
                "local-owner",
                {
                    "instruction": "Increase wheel_diameter to 240 mm. Preserve all part and occurrence identities and interfaces. Use the shared design variable, inspect the measured result and finish without claiming vehicle safety.",
                    "revision_id": last["id"],
                    "preview_only": True,
                    "max_actions": 4,
                    "max_seconds": 600,
                },
            )
            job = store.claim()
            assert job
            service.run_job(job)
        (args.output / "session.json").write_text(
            json.dumps(store.get(sid, "local-owner"), indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
