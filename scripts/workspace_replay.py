"""Rebuild saved pilot sources with current instruments into a usable workspace.

No model calls. This verifies retained CAD source and final artifact processing, while
the original generation traces remain unchanged in the input run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    store = Store(args.output)
    service = WorkspaceService(store, Executor("trusted_local"))
    report = {"engine_fingerprint": engine_fingerprint(), "started": time.time(), "cases": []}
    for row in json.loads((args.input_run / "results.json").read_text(encoding="utf-8")):
        original = json.loads(
            (args.input_run / row["case"] / "session.json").read_text(encoding="utf-8")
        )
        final = next(r for r in original["revisions"] if r["id"] == original["selected_revision"])
        c = Contract.model_validate(final["contract"])
        sid = store.create("local-owner", original["prompt"], c.model_dump())["id"]
        revision = service.build(
            sid,
            final["code"],
            c,
            None,
            f"Rebuilt and remeasured saved Astra {row['case'].replace('_', ' ')} source",
            120,
            lambda: False,
        )
        m = revision["measurement"]
        store.update(
            sid,
            selected_revision=revision["id"],
            best_revision=revision["id"] if m["geometry_valid"] else None,
            state="contract_satisfied" if m["requirements_satisfied"] else "needs_review",
        )
        observed = {
            "case": row["case"],
            "source_revision": final["id"],
            "session_id": sid,
            "revision_id": revision["id"],
            "geometry_valid": m["geometry_valid"],
            "passed": sum(f["status"] == "pass" for f in m["findings"]),
            "unverified": sum(f["status"] == "unverified" for f in m["findings"]),
            "errors": m["errors"],
        }
        report["cases"].append(observed)
        (args.output / "replay_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(json.dumps(observed), flush=True)


if __name__ == "__main__":
    main()
