"""Known-good corpus loading for the seeded-defect audit (ADR-0083).

Two persisted layouts feed the audit:

* BENCH runs (frozen breadth bench): ``<run>/results.json`` + ``<label>_v<N>/`` job dirs.
  Known-good = the run's results.json says ``passed`` (engine gate AND independent rubric)
  -- the strongest "truly good" label the repo has.
* GOLDEN problem-mode builds (problems_v1): ``<run>/golden_<id>/`` dirs. Known-good = the
  engine's own report says the loop accepted it CLEAN: ``validation.passed`` true,
  ``stopped_reason == "accepted"``, ``disposition == "shipped"``, and ZERO open issues.
  The row-38 lesson (TEST_LOG) is binding here: a golden that shipped WITH issues (the
  hdd35 false-PASS ship) must never seed the corpus -- mutants inherit their baseline's
  "known-good" label, so a dirty baseline mislabels every trial derived from it.

The goldens matter because they are the PRINT-mode brackets/trays/ducts the product ships
today -- without them the audit measures reward trust on last month's parts only.

Newest run wins per label across all matching dirs (sorted glob, later overwrites).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from forge_mind.agentic.brief import DesignBrief

_NEEDED = ("brief.json", "program.py", "part.stl", "part.step")


def _entry(label: str, job: Path, prompt: str) -> dict[str, Any] | None:
    """One corpus record from a job dir; None when artifacts or the brief are unusable."""
    if not all((job / f).exists() for f in _NEEDED):
        return None
    try:
        brief = DesignBrief.model_validate_json((job / "brief.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "label": label,
        "job_dir": job,
        "brief": brief,
        "prompt": prompt,
        "code": (job / "program.py").read_text(encoding="utf-8"),
    }


def _load_bench_run(run: Path, parts: dict[str, dict[str, Any]]) -> None:
    """Known-goods from a frozen-bench results dir (results.json layout)."""
    try:
        records = json.loads((run / "results.json").read_text(encoding="utf-8")).get("results", [])
    except Exception:
        return
    for rec in records:
        if not rec.get("passed"):
            continue
        label = str(rec.get("label", ""))
        job = run / f"{label}_v{int(rec.get('variant', 0))}"
        entry = _entry(label, job, str(rec.get("prompt", "")))
        if entry is not None:
            parts[label] = entry


def _load_golden_dir(job: Path, parts: dict[str, dict[str, Any]]) -> None:
    """A golden problem-mode build, admitted ONLY when it shipped clean (see module doc)."""
    try:
        rep = json.loads((job / "report.json").read_text(encoding="utf-8"))
    except Exception:
        return
    val = rep.get("validation") or {}
    clean = (
        val.get("passed") is True
        and rep.get("stopped_reason") == "accepted"
        and rep.get("disposition") == "shipped"
        and not (val.get("issues") or [])
    )
    if not clean:
        return
    label = job.name.removeprefix("golden_")
    entry = _entry(label, job, str(rep.get("prompt", "")))
    if entry is not None:
        parts[label] = entry


def load_corpus(repo: Path, glob_pat: str) -> list[dict[str, Any]]:
    """Known-good parts across all runs matching the COMMA-SEPARATED glob (e.g.
    ``arbitrary_*,audit_fixtures_v1,problems_v1_*``), both layouts, newest run wins."""
    parts: dict[str, dict[str, Any]] = {}
    run_dirs: list[Path] = []
    for pat in glob_pat.split(","):
        run_dirs.extend(sorted(repo.glob(pat.strip())))
    for run in run_dirs:
        if (run / "results.json").exists():
            _load_bench_run(run, parts)
        for golden in sorted(run.glob("golden_*")):
            if golden.is_dir():
                _load_golden_dir(golden, parts)
    return [parts[k] for k in sorted(parts)]
