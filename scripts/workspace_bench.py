"""Reproducible real-Astra pilot with independently frozen numeric contracts.

Run: python scripts/workspace_bench.py --output <external-directory> [--cases plate,...]
Each case retains the first attempt, every repair, final evidence, timing and source hashes.
This measures within-run changes, not causal improvement over the old production engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from forge_mind.workspace.agent import WorkspaceService
from forge_mind.workspace.executor import Executor
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract, Requirement
from forge_mind.workspace.store import Store


def contract(case: dict[str, Any]) -> Contract:
    requirements = [
        Requirement(
            id=f"extent_{axis}",
            text=f"Overall {axis.upper()} extent {value} mm",
            source_quote=case["prompt"],
            source="explicit",
            metric=f"extent_{axis}",
            value=value,
        )
        for axis, value in zip("xyz", case["extents"], strict=True)
    ]
    requirements.append(
        Requirement(
            id="solids",
            text=f"Exactly {case['solids']} solids",
            source_quote=case["prompt"],
            source="explicit",
            metric="solid_count",
            value=case["solids"],
        )
    )
    for i, hole in enumerate(case.get("holes", [])):
        requirements.append(
            Requirement(
                id=f"hole_count_{i}",
                text=f"{hole['count']} bores along {hole['axis']}",
                source_quote=case["prompt"],
                source="explicit",
                metric="hole_count",
                axis=hole["axis"],
                value=hole["count"],
            )
        )
    for i, bore in enumerate(case.get("bores", [])):
        requirements.append(
            Requirement(
                id=f"bore_{i}",
                text=f"Bore {bore['diameter']} mm at {bore['center']}",
                source_quote=case["prompt"],
                source="explicit",
                metric="hole_diameter",
                axis=bore["axis"],
                center=bore["center"],
                value=bore["diameter"],
            )
        )
    requirements.append(
        Requirement(
            id="functional_intent",
            text="All remaining construction, appearance, wall, slot, opening and fit requirements in the prompt",
            source_quote=case["prompt"],
            source="explicit",
            metric="unmeasured",
        )
    )
    return Contract(
        requirements=requirements,
        coverage_notes=[
            "Numeric pilot only. Full functional intent is retained and requires separate review."
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="")
    parser.add_argument("--seconds", type=int, default=300)
    parser.add_argument("--actions", type=int, default=5)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    corpus = repo / "benchmarks/workspace_v1.json"
    cases = json.loads(corpus.read_text(encoding="utf-8"))["cases"]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases.split(",")]
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "model": "gpt-6-astra",
        "provider": "codex subscription",
        "mode": "trusted_local",
        "started": time.time(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "engine_fingerprint": engine_fingerprint(),
        "max_actions": args.actions,
        "max_seconds": args.seconds,
        "dependencies": {n: version(n) for n in ("build123d", "trimesh", "numpy")},
        "sources": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo / "src/forge_mind/workspace").glob("*.py")
        },
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    results = []
    for case in cases:
        store = Store(args.output / case["id"])
        c = contract(case)
        session = store.create("local-owner", case["prompt"], c.model_dump())
        store.enqueue(
            session["id"], "local-owner", {"max_actions": args.actions, "max_seconds": args.seconds}
        )
        job = store.claim()
        assert job
        print(f"START {case['id']} {session['id']}", flush=True)
        started = time.monotonic()
        WorkspaceService(store, Executor("trusted_local")).run_job(job)
        final = store.get(session["id"], "local-owner")
        (args.output / case["id"] / "session.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8"
        )

        def score(
            rev: dict[str, Any] | None, numeric_count: int = len(c.requirements) - 1
        ) -> dict[str, Any]:
            if rev is None:
                return {"geometry_valid": False, "passed": 0, "total": numeric_count}
            m = rev["measurement"]
            return {
                "geometry_valid": m["geometry_valid"],
                "passed": sum(f["status"] == "pass" for f in m["findings"]),
                "total": numeric_count,
                "revision_id": rev["id"],
                "errors": m["errors"],
            }

        revs = final["revisions"]
        selected = next((r for r in revs if r["id"] == final["selected_revision"]), None)
        row = {
            "case": case["id"],
            "session_id": session["id"],
            "state": final["state"],
            "seconds": round(time.monotonic() - started, 2),
            "builds": len(revs),
            "first": score(revs[0] if revs else None),
            "selected": score(selected),
            "usage": final["usage"],
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
