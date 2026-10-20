"""Seeded-defect validator audit runner (ADR-0083) -- measured false-PASS / false-FAIL.

Materials: job dirs persisted by the frozen bench (brief.json + program.py + part.stl/step),
filtered to KNOWN-GOOD = the run's results.json says passed (engine gate AND independent
rubric). Two arms:

  * false-FAIL arm: re-validate every known-good's PERSISTED artifacts through the SAME
    production entry the loop uses (validate_authored_part). Any FAIL is a false-FAIL
    (validator overstrictness/flake -- the build is not re-run, so author variance is out).
  * false-PASS arm: inject one labeled defect per trial (audit.mutations), rebuild in the
    production sandbox, CONFIRM the defect materialized (audit.confirmers), then take the
    production verdict. PASS on a confirmed-defective part = false-PASS (an escape, pinned
    in the results with its evidence).

$0 and deterministic: no LLM anywhere. Kill-safe: results JSON rewritten after every
trial; pass a previous results path to RESUME (finished trials skip).

Args: [corpus_glob] [resume_json]
  corpus_glob  default "benchmarks/results/arbitrary_*,benchmarks/results/audit_fixtures_v1,
               benchmarks/results/problems_v1_*" (bench + fixtures + print-mode goldens;
               golden dirs admit ONLY builds that shipped CLEAN -- see audit.corpus)
  resume_json  a previous audit_results.json to resume into a NEW stamped dir

Env: FORGE_AUDIT_WORKERS (default 2) | FORGE_AUDIT_PER_CLASS per part (default 2)
     FORGE_AUDIT_MAX_GOODS (default 40)

Run:  uv run --no-sync python -u scripts/audit_validator.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.agentic.brief import DesignBrief  # noqa: E402
from forge_mind.agentic.mapped_dims import stated_bore_positions  # noqa: E402
from forge_mind.agentic.sandbox import run_build123d_code  # noqa: E402
from forge_mind.agentic.stated_dims import stated_numbers  # noqa: E402
from forge_mind.agentic.validate import validate_authored_part  # noqa: E402
from forge_mind.audit.confirmers import confirm  # noqa: E402
from forge_mind.audit.corpus import load_corpus  # noqa: E402
from forge_mind.audit.mutations import (  # noqa: E402
    MutantCandidate,
    blind_hole_mutants,
    dropped_feature_mutants,
    envelope_mutants,
    off_footprint_mutants,
    stated_dim_mutants,
    subfloor_wall_mutants,
    undersized_clearance_bore_mutants,
)
from forge_mind.fabrication import FabricationMode  # noqa: E402

_WORKERS = max(1, int(os.environ.get("FORGE_AUDIT_WORKERS", "2")))
_PER_CLASS = max(1, int(os.environ.get("FORGE_AUDIT_PER_CLASS", "2")))
_MAX_GOODS = max(1, int(os.environ.get("FORGE_AUDIT_MAX_GOODS", "40")))

_HOLE_WORDS = ("hole", "bore", "clearance", "mount", "bolt", "screw", "pivot", "shaft")


def _git_provenance() -> dict[str, Any]:
    """Exact source identity for one scorecard run."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except Exception:
        sha, dirty = "unavailable", None
    return {"git_sha": sha, "git_dirty": dirty}


def _audit_provenance(corpus: list[dict[str, Any]], glob_pat: str) -> dict[str, Any]:
    """Fingerprint validator code, flags, and every admitted corpus artifact."""
    from forge_mind.validation_provenance import file_sha256, validation_fingerprint

    audit_digest = hashlib.sha256()
    for path in (
        Path(__file__),
        REPO / "src/forge_mind/audit/confirmers.py",
        REPO / "src/forge_mind/audit/corpus.py",
        REPO / "src/forge_mind/audit/mutations.py",
        REPO / "src/forge_mind/audit/scorecard.py",
    ):
        audit_digest.update(path.relative_to(REPO).as_posix().encode("utf-8"))
        audit_digest.update(path.read_bytes())
    corpus_manifest: list[dict[str, Any]] = []
    for part in corpus:
        job = Path(part["job_dir"])
        corpus_manifest.append(
            {
                "label": part["label"],
                "material": part["brief"].material,
                "artifacts": {
                    name: file_sha256(job / name)
                    for name in ("brief.json", "program.py", "part.stl", "part.step")
                },
            }
        )
    return {
        **_git_provenance(),
        "audit_code_fingerprint": audit_digest.hexdigest(),
        "validation_fingerprint": validation_fingerprint(),
        "corpus_glob": glob_pat,
        "corpus_manifest": corpus_manifest,
        "forge_env": {k: os.environ[k] for k in sorted(os.environ) if k.startswith("FORGE_")},
        "workers": _WORKERS,
        "per_class_cap": _PER_CLASS,
        "max_goods": _MAX_GOODS,
    }


def wilson(k: int, n: int) -> tuple[float, float]:
    """95% Wilson interval (lo, hi) for k/n; (0,1) when n=0."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _feature_labels(brief: DesignBrief) -> tuple[list[str], list[str]]:
    """(hole-ish labels, repeated-feature labels) from the brief's planned features."""
    holes: list[str] = []
    repeated: list[str] = []
    for f in brief.features:
        name = f.name
        low = name.lower()
        if any(w in low for w in _HOLE_WORDS) or f.standard is not None:
            holes.append(name)
        if f.count >= 2:
            repeated.append(name)
    return holes, repeated


def _blind_depth_hint(part: dict[str, Any]) -> float | None:
    """A cut depth guaranteed to stop SHORT of piercing, turning a through-bore blind. v2.1: capped
    at 1.2 mm, not 0.45x the whole-part smallest extent -- a bolt hole crosses a LOCAL feature (an
    ear, a wall) far thinner than the part's overall min extent, so 0.45x still pierced (measured:
    9 mm through a ~4 mm ear). 1.2 mm is below any printable wall (>= min_wall ~2 mm), so the cut is
    blind in any real feature while leaving a measurable floor. None on any failure (bare-Hole idiom
    then skipped, never guessed)."""
    try:
        import trimesh

        m: Any = trimesh.load(str(part["job_dir"] / "part.stl"), force="mesh")
        lo, hi = m.bounds
        return round(min(0.45 * float(min(hi[i] - lo[i] for i in range(3))), 1.2), 3)
    except Exception:
        return None


def _candidates(part: dict[str, Any]) -> list[MutantCandidate]:
    """Up to _PER_CLASS candidates per defect class, deterministic order.

    off_footprint_bore is anchor-filtered (ADR-0012 E1, measurement honesty): a mutant counts
    only when the drifted position is a USER-STATED, bore-MEASURABLE position -- i.e. the mapping
    check has an owning spec for it. Drifting a MODEL-INVENTED hole pattern (clip_tube's plate
    holes, hook_j's screws, pulley's aux holes -- none in the prompt) or a non-bore spacing (slot,
    rib, blade angle) makes a different valid part, not a defect, so counting it as a false-PASS
    escape was measuring noise (the reconcile, 2026-07-07)."""
    brief: DesignBrief = part["brief"]
    code: str = part["code"]
    stated = stated_numbers(part["prompt"], brief)
    holes, repeated = _feature_labels(brief)
    from forge_mind.agentic.spatial import _planned_hole_specs

    clearance_targets = [
        float(spec.target_d_mm)
        for spec in _planned_hole_specs(brief)
        if spec.target_d_mm is not None and spec.fit_nominal_d_mm is not None
    ]
    pos_values = [s.value_mm for s in stated_bore_positions(brief, part["prompt"], stated)]
    footprint = [
        m
        for m in off_footprint_mutants(code)
        if any(abs(float(m.meta.get("old", -1.0)) - v) <= 0.05 for v in pos_values)
    ]
    per_class = [
        stated_dim_mutants(code, stated),
        subfloor_wall_mutants(code, brief.min_wall_mm),
        dropped_feature_mutants(code, repeated),
        blind_hole_mutants(code, holes, depth_hint_mm=_blind_depth_hint(part)),
        footprint,
        envelope_mutants(code),
        undersized_clearance_bore_mutants(code, clearance_targets),
    ]
    out: list[MutantCandidate] = []
    for group in per_class:
        out.extend(group[:_PER_CLASS])
    return out


# ------------------------------------------------------------------------ trials ---


def _verdict(
    stl: Path,
    step: Path,
    brief: DesignBrief,
    code: str,
    mode: FabricationMode = FabricationMode.GENERAL,
    prompt: str | None = None,
) -> dict[str, Any]:
    """The production verdict, mirrored from the loop's call (loop.py validate site). v2 captures
    the FULL issue list + features_missing so the surfaced detector can see every finding (v1
    truncated to 6 and read only `passed`). The prompt enables the stated-dim owning check
    (ADR-0012 E1-1) exactly as the live loop passes it."""
    val = validate_authored_part(
        stl, step, brief, fabrication_mode=mode, author_code=code, prompt=prompt
    )
    return {
        "passed": bool(val.passed),
        "failed_checks": val.report.failed_check_ids() if val.report is not None else [],
        "issues": list(val.issues),
        "features_missing": bool(val.features_missing),
    }


def _surfaced_by(defect_class: str, verdict: dict[str, Any]) -> bool | None:
    """Did the validator SURFACE this defect anywhere -- a gate FAIL or an advisory issue (ADR-0083
    v2)? This is the reward-readiness measure: an RL reward is the graded loss + check verdicts, so
    what matters is whether the signal SEES the defect, NOT whether `passed` gates it (v1 measured
    only `passed`, which is advisory-rich, so it under-counted the moat). None = the class has no
    owning check by design (stated-dim), where "surfaced" is N/A, not a miss."""
    issues = " ".join(verdict["issues"]).upper()
    failed = " ".join(verdict["failed_checks"]).lower()
    fm = bool(verdict.get("features_missing"))
    if defect_class == "off_brief_stated_dim":
        # The owning check (ADR-0012 E1-1) surfaces as STATED DIM; the pre-existing partial
        # coverage (bore ID windows -> SIZE MISMATCH, coarse drift -> ENVELOPE) still counts.
        return (
            ("STATED DIM" in issues)
            or ("SIZE MISMATCH" in issues)
            or ("MEASURED BORES" in issues)
            or ("ENVELOPE" in issues)
            or ("bbox" in failed)
        )
    if defect_class == "envelope_violation":
        return (
            ("ENVELOPE" in issues)
            or ("STATED DIM" in issues)
            or ("bbox" in failed)
            or ("envelope" in failed)
        )
    if defect_class == "sub_floor_wall":
        return (
            ("min_wall" in failed)
            or ("WALL GATE" in issues)
            or ("THIN" in issues)
            or ("UNMEASURABLE" in issues)
        )
    if defect_class == "undersized_clearance_bore":
        return "BORE FIT" in issues or "SIZE MISMATCH" in issues
    if defect_class in ("blind_where_through", "dropped_named_feature"):
        return fm or ("MEASURED BORES" in issues) or ("MISSING" in issues)
    if defect_class == "off_footprint_bore":
        return (
            ("MAPPED POSITION" in issues)
            or fm
            or ("MEASURED BORES" in issues)
            or ("MISSING" in issues)
            or ("SIZE MISMATCH" in issues)
        )
    return None


def _run_mutant(part: dict[str, Any], cand: MutantCandidate, work_root: Path) -> dict[str, Any]:
    label = part["label"]
    key = f"{label}::{cand.defect_class}::{cand.description}"
    rec: dict[str, Any] = {
        "key": key,
        "label": label,
        "defect_class": cand.defect_class,
        "description": cand.description,
    }
    site = hashlib.md5(cand.description.encode("utf-8")).hexdigest()[:8]
    job = work_root / label / cand.defect_class / site
    sb = run_build123d_code(cand.code, job)
    if not sb.ok or sb.stl_path is None or sb.step_path is None:
        rec.update(valid=False, why="mutant did not build", error=str(sb.error_type))
        return rec
    conf = confirm(
        cand,
        baseline_stl=part["job_dir"] / "part.stl",
        baseline_step=part["job_dir"] / "part.step",
        mutant_stl=sb.stl_path,
        mutant_step=sb.step_path,
        min_wall_mm=part["brief"].min_wall_mm,
    )
    if not conf.confirmed:
        rec.update(valid=False, why="defect not confirmed", evidence=conf.evidence)
        return rec
    gen = _verdict(
        sb.stl_path,
        sb.step_path,
        part["brief"],
        cand.code,
        FabricationMode.GENERAL,
        prompt=part["prompt"],
    )
    pr = _verdict(
        sb.stl_path,
        sb.step_path,
        part["brief"],
        cand.code,
        FabricationMode.PRINT_3D,
        prompt=part["prompt"],
    )
    rec.update(
        valid=True,
        evidence=conf.evidence,
        verdict_passed=gen["passed"],
        false_pass=gen["passed"],  # GENERAL passed-escape (matches v1)
        print_passed=pr["passed"],
        print_false_pass=pr["passed"],  # does the stricter PRINT gate catch it? (wall class)
        surfaced=_surfaced_by(cand.defect_class, gen),  # gate OR advisory -- the reward signal
        failed_checks=gen["failed_checks"],
        issues=gen["issues"][:8],
    )
    return rec


def _run_good(part: dict[str, Any]) -> dict[str, Any]:
    label = part["label"]
    v = _verdict(
        part["job_dir"] / "part.stl",
        part["job_dir"] / "part.step",
        part["brief"],
        part["code"],
        prompt=part["prompt"],
    )
    return {
        "key": f"{label}::known_good",
        "label": label,
        "defect_class": "known_good",
        "valid": True,
        "verdict_passed": v["passed"],
        "false_fail": not v["passed"],
        "failed_checks": v["failed_checks"],
        "issues": v["issues"],
    }


# ----------------------------------------------------------------------- reporting ---


def _rates(records: list[dict[str, Any]]) -> dict[str, Any]:
    mutants = [r for r in records if r["defect_class"] != "known_good"]
    valid = [r for r in mutants if r.get("valid")]
    goods = [r for r in records if r["defect_class"] == "known_good"]
    escapes = [r for r in valid if r.get("false_pass")]
    ffails = [r for r in goods if r.get("false_fail")]
    per_class: dict[str, Any] = {}
    for r in valid:
        c = per_class.setdefault(
            r["defect_class"],
            {"n": 0, "escapes": 0, "print_escapes": 0, "surf_n": 0, "surfaced": 0},
        )
        c["n"] += 1
        c["escapes"] += int(bool(r.get("false_pass")))
        c["print_escapes"] += int(bool(r.get("print_false_pass")))
        s = r.get("surfaced")
        if (
            s is not None
        ):  # None = no owning check by design; excluded from the surfaced denominator
            c["surf_n"] += 1
            c["surfaced"] += int(bool(s))
    for c in per_class.values():
        c["fp_rate"] = c["escapes"] / c["n"] if c["n"] else None
        c["print_fp_rate"] = c["print_escapes"] / c["n"] if c["n"] else None
        # surfaced = the validator SAW the defect (gate or advisory) -- the reward-readiness measure
        c["surfaced_rate"] = c["surfaced"] / c["surf_n"] if c["surf_n"] else None
        c["wilson95"] = wilson(c["escapes"], c["n"])
    # top-level surfaced rate over classes that HAVE an owning check (excludes stated-dim)
    surf_all = [r for r in valid if r.get("surfaced") is not None]
    surf_seen = [r for r in surf_all if r.get("surfaced")]
    print_escapes = [r for r in valid if r.get("print_false_pass")]
    fp = len(escapes) / len(valid) if valid else None
    fn = len(ffails) / len(goods) if goods else None
    return {
        "n_valid_mutants": len(valid),
        "n_invalid_mutants": len(mutants) - len(valid),
        "n_known_goods": len(goods),
        "false_pass_rate": fp,
        "false_pass_wilson95": wilson(len(escapes), len(valid)),
        "print_false_pass_rate": (len(print_escapes) / len(valid) if valid else None),
        "surfaced_rate": (len(surf_seen) / len(surf_all) if surf_all else None),
        "surfaced_note": "fraction of defects (with an owning check) the validator SAW (gate or advisory) -- the reward-readiness measure, vs false_pass_rate which is passed-gate-only",
        "false_fail_rate": fn,
        "false_fail_wilson95": wilson(len(ffails), len(goods)),
        "per_class": per_class,
        "escapes_pinned": [
            {k: r[k] for k in ("label", "defect_class", "description", "evidence")} for r in escapes
        ],
        "false_fails_pinned": [
            {k: r[k] for k in ("label", "failed_checks", "issues")} for r in ffails
        ],
    }


_DEFAULT_GLOB = (
    "benchmarks/results/arbitrary_*,benchmarks/results/audit_fixtures_v1,"
    "benchmarks/results/problems_v1_*"
)


def main() -> int:
    glob_pat = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_GLOB
    resume = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO / "benchmarks" / "results" / f"audit_{stamp}"
    out_json = out_dir / "audit_results.json"

    previous: dict[str, Any] | None = None
    if resume is not None and resume.exists():
        previous = json.loads(resume.read_text(encoding="utf-8"))

    corpus = load_corpus(REPO, glob_pat)[:_MAX_GOODS]
    print(f"[corpus] {len(corpus)} known-good parts from {glob_pat!r}", flush=True)
    if not corpus:
        print("no corpus parts found -- run the bench first or fix the glob", flush=True)
        return 2
    provenance = _audit_provenance(corpus, glob_pat)
    done: dict[str, dict[str, Any]] = {}
    if previous is not None:
        if previous.get("provenance") != provenance:
            print(
                "resume refused: validator, flags, git state, or corpus artifacts changed",
                flush=True,
            )
            return 2
        done = {r["key"]: r for r in previous.get("records", [])}
        print(f"[resume] {len(done)} finished trials carried over", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    trials: list[tuple[str, Any]] = []
    for part in corpus:
        gk = f"{part['label']}::known_good"
        if gk not in done:
            trials.append(("good", part))
        for cand in _candidates(part):
            key = f"{part['label']}::{cand.defect_class}::{cand.description}"
            if key not in done:
                trials.append(("mutant", (part, cand)))
    print(f"[plan] {len(trials)} trials to run ({len(done)} resumed)", flush=True)

    records: list[dict[str, Any]] = list(done.values())

    def _save() -> None:
        payload = {
            "protocol": "ADR-0083 seeded-defect audit v1",
            "stamp": stamp,
            "corpus_glob": glob_pat,
            "per_class_cap": _PER_CLASS,
            "provenance": provenance,
            "records": records,
            "rates": _rates(records),
        }
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _save()
    work_root = out_dir / "work"
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futs = {}
        for kind, payload in trials:
            if kind == "good":
                futs[pool.submit(_run_good, payload)] = ("good", payload["label"])
            else:
                part, cand = payload
                futs[pool.submit(_run_mutant, part, cand, work_root)] = (
                    cand.defect_class,
                    part["label"],
                )
        for i, fut in enumerate(as_completed(futs), 1):
            kind, label = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # a crashed trial is recorded, never silently dropped
                rec = {
                    "key": f"{label}::{kind}::crash",
                    "label": label,
                    "defect_class": kind,
                    "valid": False,
                    "why": f"trial crashed: {type(exc).__name__}: {exc}",
                }
            records.append(rec)
            _save()
            status = (
                "ESCAPE"
                if rec.get("false_pass")
                else (
                    "FALSE-FAIL"
                    if rec.get("false_fail")
                    else ("ok" if rec.get("valid") else "invalid")
                )
            )
            print(f"[{i}/{len(trials)}] {label} {rec['defect_class']}: {status}", flush=True)

    rates = _rates(records)
    print(
        json.dumps(
            {
                k: rates[k]
                for k in (
                    "n_valid_mutants",
                    "n_invalid_mutants",
                    "n_known_goods",
                    "false_pass_rate",
                    "false_fail_rate",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    # The standing scorecard: per-run copy + the committed current reading (audit.scorecard).
    from forge_mind.audit.scorecard import scorecard_markdown

    md = scorecard_markdown(rates, stamp=stamp, corpus_glob=glob_pat)
    (out_dir / "scorecard.md").write_text(md, encoding="utf-8")
    (REPO / "docs" / "VALIDATOR_SCORECARD.md").write_text(md, encoding="utf-8")
    print(f"[done] full results: {out_json}", flush=True)
    print(f"[done] scorecard: {out_dir / 'scorecard.md'} (+ docs/VALIDATOR_SCORECARD.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
