"""Exemplar A/B — does a retrieved reference build help the author (ADR-0012 E1 follow-up)?

The honest experimental design, forced by the reconcile (2026-07-07): a pass@1 lift A/B has NO
headroom population — the parts the engine can construct it already passes (saturated), and the
parts it fails, it fails on print-DFM (thin walls) that construction exemplars do not address. The
metric WITH headroom is ITERATIONS-TO-PASS: a good reference can make a part pass on draft 1 instead
of after N revisions, even when pass@1 is already 100%. That is what this measures.

Method (leak-guarded):
  * Library = non-frozen PASSING builds, RE-VETTED through the current (E1) validator (they passed
    an OLDER, weaker gate; the council's rule is never retrieve through a stale reward). The frozen
    benchmark is denied at harvest (memorization guard, exemplars.benchmark_deny_list).
  * Test = K held-out library-source prompts, REMOVED from the library for their own run (a part
    never sees itself). Retrieval similarity is logged so near-duplicate pulls are visible, not
    hidden.
  * For each test prompt: run the loop with FORGE_AUTHOR_EXEMPLARS off vs on, record
    iterations-to-first-pass (and whether it passed at all). Same prompt, same settings, only the
    exemplar block differs.

Subscription only ($0): set FORGE_LLM_PROVIDER=claude_code. This is DIRECTIONAL (small n, stochastic
generation) — it reports the split and the caveats, never a headline claim.

Run:
  $env:FORGE_LLM_PROVIDER='claude_code'
  uv run --no-sync python -u scripts/exemplar_ab.py [n_test] [seed_offset]
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.agentic.exemplars import (  # noqa: E402
    ExemplarRecord,
    _is_denied,
    _norm,
    benchmark_deny_list,
    load_all,
    store,
)
from forge_mind.agentic.loop import run_agentic_pipeline  # noqa: E402
from forge_mind.agentic.validate import validate_authored_part  # noqa: E402


def _passing_nonfrozen_builds(deny: set[str]) -> list[dict]:
    """Every passing, non-frozen build with a program + brief; newest run wins per prompt."""
    by_prompt: dict[str, dict] = {}
    for results in sorted(REPO.glob("benchmarks/results/*/results.json")):
        try:
            data = json.loads(results.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in data.get("results", []):
            if not rec.get("passed"):
                continue
            prompt = str(rec.get("prompt", ""))
            if _is_denied(prompt, deny) or not prompt.strip():
                continue
            # Two persisted layouts: the arbitrary bench writes {label}_v{variant}; the
            # pass@k runner writes {label}_k{sample}. The k-dirs DO carry full artifacts --
            # the original pool starved on exactly this naming mismatch (2026-07-07).
            label, var = str(rec.get("label", "")), rec.get("variant", 0)
            job = results.parent / f"{label}_v{var}"
            if not (job / "program.py").exists():
                job = results.parent / f"{label}_k{var}"
            if not (job / "program.py").exists() or not (job / "part.stl").exists():
                continue
            by_prompt[_norm(prompt)] = {
                "label": str(rec.get("label", "")),
                "prompt": prompt,
                "job": job,
            }
    return list(by_prompt.values())


def _revet(job: Path, prompt: str) -> bool:
    """Does this build still PASS the current E1 validator? (Re-vet: it passed an older gate.)"""
    try:
        from forge_mind.agentic.brief import DesignBrief

        brief = DesignBrief.model_validate_json((job / "brief.json").read_text(encoding="utf-8"))
        code = (job / "program.py").read_text(encoding="utf-8")
        val = validate_authored_part(
            job / "part.stl", job / "part.step", brief, author_code=code, prompt=prompt
        )
        return bool(val.passed)
    except Exception:
        return False


def _iters_to_pass(work: Path, prompt: str) -> tuple[bool, int, int]:
    """Run one generation; return (passed, iterations_to_first_pass, total_iterations)."""
    res = run_agentic_pipeline(prompt, work, max_iterations=4, n_drafts=1)
    first = 0
    for i, it in enumerate(res.iterations, 1):
        if getattr(it, "validation_passed", False):
            first = i
            break
    passed = bool(res.geometry_passed)
    return passed, (first or len(res.iterations)), len(res.iterations)


def main() -> int:
    n_test = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    targets_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "benchmarks" / "results" / f"exemplar_ab_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    deny = benchmark_deny_list(REPO)
    builds = _passing_nonfrozen_builds(deny)
    # re-vet the whole pool through the current validator (deterministic, $0)
    vetted = [b for b in builds if _revet(b["job"], b["prompt"])]
    print(
        f"[pool] {len(builds)} passing non-frozen builds -> {len(vetted)} still pass E1", flush=True
    )

    if targets_json is not None:
        # FOREIGN targets (the clean design): a fresh-seed holdout set, disjoint from the
        # library by seed rotation -- the library keeps every vetted build, and no target
        # can be its own exemplar by construction.
        data = json.loads(targets_json.read_text(encoding="utf-8"))
        test = [
            {"label": str(p["id"]), "prompt": str(p["prompt"]), "job": None}
            for p in data["prompts"][:n_test]
        ]
        test_norms: set[str] = set()
        if len(vetted) < 10:
            print("[stop] library too thin for a leak-guarded A/B — report, do not fake")
            return 2
    else:
        if len(vetted) < n_test + 5:
            print("[stop] pool too thin for a leak-guarded A/B — report, do not fake")
            return 2
        # deterministic held-out test split from the pool itself (fallback design)
        vetted.sort(key=lambda b: b["label"])
        test = [vetted[(i * 7) % len(vetted)] for i in range(n_test)]
        test_norms = {_norm(b["prompt"]) for b in test}

    lib_dir = out / "library"
    for b in vetted:
        if _norm(b["prompt"]) in test_norms:
            continue  # a part never sees itself
        store(
            ExemplarRecord(
                object_name=b["label"].replace("_", " "),
                prompt=b["prompt"],
                code=(b["job"] / "program.py").read_text(encoding="utf-8", errors="replace"),
                source=str(b["job"]),
                harvested_at=stamp,
            ),
            lib_dir,
        )
    print(f"[library] {len(load_all(lib_dir))} exemplars (test parts excluded)", flush=True)

    rows = []
    for b in test:
        row: dict = {"label": b["label"], "prompt": b["prompt"][:120]}
        for arm in ("off", "on"):
            if arm == "on":
                os.environ["FORGE_AUTHOR_EXEMPLARS"] = str(lib_dir)
            else:
                os.environ.pop("FORGE_AUTHOR_EXEMPLARS", None)
            t0 = time.monotonic()
            passed, first, total = _iters_to_pass(out / f"{b['label']}_{arm}", b["prompt"])
            row[arm] = {
                "passed": passed,
                "iters_to_pass": first,
                "total_iters": total,
                "secs": round(time.monotonic() - t0),
            }
            print(
                f"[{b['label']}|{arm}] passed={passed} iters_to_pass={first} ({row[arm]['secs']}s)",
                flush=True,
            )
        rows.append(row)
        (out / "ab_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    # directional summary (never a headline: small n, stochastic)
    on_pass = sum(1 for r in rows if r["on"]["passed"])
    off_pass = sum(1 for r in rows if r["off"]["passed"])
    on_it = [r["on"]["iters_to_pass"] for r in rows if r["on"]["passed"]]
    off_it = [r["off"]["iters_to_pass"] for r in rows if r["off"]["passed"]]
    off_mean = (sum(off_it) / len(off_it)) if off_it else float("nan")
    on_mean = (sum(on_it) / len(on_it)) if on_it else float("nan")
    print(f"\n=== exemplar A/B (DIRECTIONAL, n={len(rows)}) ===")
    print(f"passed: off {off_pass}/{len(rows)}  on {on_pass}/{len(rows)}")
    print(f"mean iters-to-pass: off {off_mean:.2f}  on {on_mean:.2f}")
    print(f"results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
