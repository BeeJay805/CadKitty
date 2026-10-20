"""Run a frozen natural-language engineering job on the Codex subscription.

No geometry or dedicated template is supplied by this runner. A supplied contract
is frozen independently; otherwise the normal requirement extractor runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.store import Store


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-actions", type=int, help="Optional positive action budget")
    parser.add_argument("--max-seconds", type=int, help="Optional positive total execution budget")
    args = parser.parse_args(argv)
    for option in ("max_actions", "max_seconds"):
        value = getattr(args, option)
        if value is not None and value <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive when supplied")
    return args


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prompt = args.brief.read_text(encoding="utf-8")
    contract = (
        Contract.model_validate_json(args.contract.read_text(encoding="utf-8")).model_dump()
        if args.contract
        else None
    )
    (args.output / "brief.txt").write_text(prompt, encoding="utf-8")
    provenance = {
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "contract": contract,
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "engine_fingerprint": engine_fingerprint(),
        "provider": "codex subscription",
        "model": "gpt-6-astra",
        "max_actions": args.max_actions,
        "max_seconds": args.max_seconds,
        "scope": "Private development trial; no production-provider or broad generality claim.",
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    store = Store(args.output / "data")
    sid = store.create("local-owner", prompt, contract)["id"]
    store.enqueue(
        sid,
        "local-owner",
        {
            "max_actions": args.max_actions,
            "max_seconds": args.max_seconds,
            "preview_only": True,
        },
    )
    start = time.monotonic()
    WorkspaceService(store, Executor("trusted_local")).run_job(store.claim())
    session = store.get(sid, "local-owner")
    (args.output / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    result = {
        "session": sid,
        "state": session["state"],
        "seconds": time.monotonic() - start,
        "revisions": len(session["revisions"]),
        "template_calls": sum(e.get("action") == "template" for e in session["events"]),
        "selected_revision": session.get("selected_revision"),
        "proposal": session.get("proposal"),
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
