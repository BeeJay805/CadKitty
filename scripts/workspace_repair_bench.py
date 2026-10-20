"""Seed two known defects in a real generated plate and measure autonomous repair.

The source comes from a saved workspace pilot, not a CAD template. The intentional
perturbations remain in an immutable first revision; both dimensions must be repaired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge_mind.agentic.params import extract_params, override_params
from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_session.read_text(encoding="utf-8"))
    original = next(r for r in source["revisions"] if r["id"] == source["selected_revision"])
    code = original["code"]
    params = extract_params(code)
    assert params and {"length_x", "hole_diameter"} <= set(params.float_names)
    broken = override_params(code, params, {"length_x": 72.0, "hole_diameter": 4.5})
    c = Contract.model_validate(original["contract"])
    store = Store(args.output)
    sid = store.create("local-owner", source["prompt"], c.model_dump())["id"]
    service = WorkspaceService(store, Executor("trusted_local"))
    seed = service.build(
        sid,
        broken,
        c,
        None,
        "Controlled defects: length 72 instead of 80 mm, bores 4.5 instead of 5.5 mm",
        60,
        lambda: False,
    )
    store.update(sid, selected_revision=seed["id"])
    print("BEFORE", seed["measurement"], flush=True)
    store.enqueue(
        sid,
        "local-owner",
        {
            "instruction": "Repair the measured defects while preserving every correct feature. Do not weaken or remove any requirements.",
            "contract": c.model_dump(),
            "max_seconds": 300,
            "max_actions": 5,
        },
    )
    job = store.claim()
    assert job
    service.run_job(job)
    final = store.get(sid, "local-owner")
    (args.output / "session.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    selected = next((r for r in final["revisions"] if r["id"] == final["selected_revision"]), None)
    print("AFTER", final["state"], selected["measurement"] if selected else None, flush=True)


if __name__ == "__main__":
    main()
