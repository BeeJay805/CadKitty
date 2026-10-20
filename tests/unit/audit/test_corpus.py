"""Audit corpus loading (ADR-0083 / PrintCell E0): bench + golden layouts, money-free.

The load-bearing case is the golden CLEAN filter: a golden that shipped WITH issues (the
row-38 hdd35 false-PASS ship) must never seed the corpus -- mutants inherit their
baseline's "known-good" label, so a dirty baseline mislabels every derived trial.
"""

from __future__ import annotations

import json
from pathlib import Path

from forge_mind.audit.corpus import load_corpus

_BRIEF = {
    "object_name": "test plate",
    "summary": "a plate",
    "visual_description": "a flat plate",
    "overall_envelope_mm": "60 x 60 x 5 mm",
    "features": [
        {
            "name": "mounting holes",
            "purpose": "bolts",
            "approx_dims_mm": "M4 clearance",
            "method": "hole",
            "count": 4,
            "standard": "M4",
        }
    ],
    "min_wall_mm": 2.0,
    "print_orientation_note": "flat",
}


def _write_job(job: Path, code: str = "result = 1") -> None:
    job.mkdir(parents=True, exist_ok=True)
    (job / "brief.json").write_text(json.dumps(_BRIEF), encoding="utf-8")
    (job / "program.py").write_text(code, encoding="utf-8")
    (job / "part.stl").write_bytes(b"solid")
    (job / "part.step").write_bytes(b"ISO-10303")


def _write_golden(
    run: Path,
    prompt_id: str,
    *,
    passed: bool = True,
    stopped: str = "accepted",
    disposition: str = "shipped",
    issues: list[str] | None = None,
    code: str = "result = 1",
) -> None:
    job = run / f"golden_{prompt_id}"
    _write_job(job, code=code)
    (job / "report.json").write_text(
        json.dumps(
            {
                "prompt": f"prompt for {prompt_id}",
                "disposition": disposition,
                "stopped_reason": stopped,
                "validation": {"passed": passed, "issues": issues or []},
            }
        ),
        encoding="utf-8",
    )


def _write_bench(run: Path, label: str, *, passed: bool = True) -> None:
    job = run / f"{label}_v0"
    _write_job(job)
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.json").write_text(
        json.dumps({"results": [{"label": label, "variant": 0, "passed": passed, "prompt": "p"}]}),
        encoding="utf-8",
    )


def test_clean_golden_loads(tmp_path: Path) -> None:
    _write_golden(tmp_path / "problems_v1_a", "nr200_ssd")
    corpus = load_corpus(tmp_path, "problems_v1_*")
    assert [p["label"] for p in corpus] == ["nr200_ssd"]
    assert corpus[0]["prompt"] == "prompt for nr200_ssd"
    assert corpus[0]["code"] == "result = 1"


def test_dirty_goldens_never_seed_the_corpus(tmp_path: Path) -> None:
    run = tmp_path / "problems_v1_a"
    # the row-38 shape: shipped=true, passed=true, but OPEN issues -> poison, excluded
    _write_golden(run, "hdd35_issues", issues=["MEASURED BORES vs PLAN: still MISSING 4"])
    _write_golden(run, "hdd35_caveats", disposition="shipped_with_caveats")
    _write_golden(run, "hdd35_rejected", disposition="rejected", stopped="plateau")
    _write_golden(run, "hdd35_failed", passed=False)
    assert load_corpus(tmp_path, "problems_v1_*") == []


def test_newest_golden_run_wins(tmp_path: Path) -> None:
    _write_golden(tmp_path / "problems_v1_a", "nr200_ssd", code="result = 'old'")
    _write_golden(tmp_path / "problems_v1_b", "nr200_ssd", code="result = 'new'")
    corpus = load_corpus(tmp_path, "problems_v1_*")
    assert len(corpus) == 1
    assert corpus[0]["code"] == "result = 'new'"


def test_bench_layout_still_loads_alongside_goldens(tmp_path: Path) -> None:
    _write_bench(tmp_path / "arbitrary_x", "bracket", passed=True)
    _write_bench(tmp_path / "arbitrary_y", "flange", passed=False)  # not known-good
    _write_golden(tmp_path / "problems_v1_a", "nr200_ssd")
    corpus = load_corpus(tmp_path, "arbitrary_*,problems_v1_*")
    assert [p["label"] for p in corpus] == ["bracket", "nr200_ssd"]


def test_missing_artifacts_skip_the_part(tmp_path: Path) -> None:
    run = tmp_path / "problems_v1_a"
    _write_golden(run, "nr200_ssd")
    (run / "golden_nr200_ssd" / "part.step").unlink()
    assert load_corpus(tmp_path, "problems_v1_*") == []
