"""Arbitrary single-part benchmark scorer v1 -- the GENERAL agentic author, pass@1.

THE instrument for the breadth objective: ~30 frozen, diverse, arbitrary single-part
prompts (the long tail no deterministic lane owns), scored pass@1 with
min-across-paraphrases. Drives ``agentic.result.produce_part`` DIRECTLY -- the general
author loop the deployed /generate falls through to -- so deterministic lanes never
touch the number.

Protocol (mirrors scripts/bench_breadth.py; measurement code copied verbatim so the
instrument is self-contained and a shared-lib change can never silently alter it):
  * Content lives in benchmarks/arbitrary_v1/arbitrary_v1.json, authored by a
    NON-author model (sonnet) and FROZEN: MANIFEST.sha256 pins it and this script
    REFUSES to run if the hash drifted. The CAD author model never sees the rubric.
  * pass@1: every variant runs ONCE -- no retries, no best-of. A prompt's score is the
    MIN across its variants; the headline is the mean over prompts.
  * A variant PASSES iff the engine's own gate holds (validation.passed) AND every
    rubric expectation verifies AGAINST THE ARTIFACTS, re-measured independently
    (meshes via trimesh; STEP text census; holes measured on the reimported B-rep).
  * Build cache force-disabled (a cache hit would fake pass@1).
  * Kill-safe: the results JSON is rewritten after EVERY variant; pass a previous
    results path as the 4th arg to RESUME (already-run variants are skipped).

Args: [arm] [variants] [n_prompts] [resume_json] [shard_ix] [shard_n]
  arm       sonnet|opus|fable (default opus -- the production author default)
  variants  1 = base prompts only (iteration), 3 = base + 2 paraphrases (the headline)
  shard     run only prompts where index % shard_n == shard_ix (parallel wall-clock;
            each shard writes its own results dir; merge with merge_bench_results.py)
Run:  $env:FORGE_LLM_PROVIDER='claude_code'; uv run --no-sync python -u scripts/bench_arbitrary.py opus 3
Writes benchmarks/results/arbitrary_<arm>_<stamp>[_s<ix>]/ (per-variant job dirs +
results JSON + render gallery).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The bench measures the engine without persistence/billing; a build-cache hit would
# replay an old artifact and fake pass@1, so force it off before importing the engine.
# FORGE_BUILD_CACHE is a cache DIRECTORY PATH (falsy = disabled): setting it to "0"
# ENABLED a cache at literal path ./0 -- caught live when the BEFORE run's first
# variant replayed a prior smoke build as `it=0 cached`. Empty string = off.
os.environ["FORGE_BUILD_CACHE"] = ""

from forge_mind.agentic.result import produce_part
from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.config import FABLE, OPUS, SONNET
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.claude_code_client import subscription_client_if_selected

_PROVIDER_CLIENT = subscription_client_if_selected()

# FORGE_BENCH_VERSION selects the frozen content: "v1" (default -- byte-identical
# historical behavior) or "v1.1" (the errata-corrected successor; see its MANIFEST +
# errata_from_v1). Both stay committed; scores only compare WITHIN a version.
_BENCH_VERSION = os.environ.get("FORGE_BENCH_VERSION", "v1").strip() or "v1"
_BENCH_SLUG = {"v1": "arbitrary_v1", "v1.1": "arbitrary_v1_1", "v1_1": "arbitrary_v1_1"}.get(
    _BENCH_VERSION
)
if _BENCH_SLUG is None:
    raise SystemExit(f"unknown FORGE_BENCH_VERSION {_BENCH_VERSION!r}; use v1 or v1.1")
BENCH_DIR = Path("benchmarks") / _BENCH_SLUG
BENCH_FILE = BENCH_DIR / f"{_BENCH_SLUG}.json"
MANIFEST = BENCH_DIR / "MANIFEST.sha256"

_MODELS = {"sonnet": SONNET, "opus": OPUS, "fable": FABLE}
_MAX_ITERATIONS = 8  # the candidate live config (dynamic early-stop usually halts sooner)
_N_DRAFTS = 1
_PART_BUDGET_USD = 1.0  # inert on subscription (cost records 0); bounds API-provider runs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_frozen() -> dict[str, Any]:
    """Load the benchmark; hard-fail unless the committed manifest matches the content."""
    if not BENCH_FILE.exists():
        raise SystemExit(f"missing {BENCH_FILE} -- the frozen benchmark is required")
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST} -- refuse to run an unfrozen benchmark")
    want: dict[str, str] = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        want[name.strip()] = digest
    for name, digest in want.items():
        p = BENCH_DIR / name
        if not p.exists() or _sha256(p) != digest:
            raise SystemExit(f"FROZEN-CONTENT MISMATCH for {name}: refusing to run")
    data = json.loads(BENCH_FILE.read_text(encoding="utf-8"))
    if not isinstance(data.get("prompts"), list) or not data["prompts"]:
        raise SystemExit("benchmark file has no prompts")
    return data


# ------------------------------------------------- independent artifact measurement
# (copied from scripts/bench_breadth.py so this instrument is self-contained)


def _step_census(step_path: Path) -> dict[str, Any]:
    """Pure ISO-10303-21 text census: product entities, B-rep markers, product names."""
    try:
        text = step_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {"products": 0, "is_brep": False, "tessellated": True, "product_names": []}
    products = re.findall(r"=\s*PRODUCT\s*\(\s*'([^']*)'", text)
    return {
        "products": len(products),
        "is_brep": ("MANIFOLD_SOLID_BREP" in text and "ADVANCED_FACE" in text),
        "tessellated": ("TESSELLATED" in text or "TRIANGULATED" in text),
        "product_names": products,
    }


def _hole_stacks(step_paths: list[Path]) -> list[dict[str, Any]]:
    """Measured concave cylinder stacks (physical hole walls) across the part STEPs."""
    out: list[dict[str, Any]] = []
    try:
        from build123d import import_step

        from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders
    except Exception as exc:  # measurement unavailable -> measured nothing, never a guess
        print(f"  [measure] B-rep utilities unavailable: {exc}", flush=True)
        return out
    for sp in step_paths:
        try:
            part = import_step(str(sp))
            stacks = cluster_coaxial(extract_cylinders(part))
        except Exception as exc:
            print(f"  [measure] {sp.name}: {exc}", flush=True)
            continue
        for s in stacks:
            if s.concave is not True:
                continue
            out.append(
                {
                    "diameter_mm": 2.0 * float(s.radius),
                    "depth_mm": (float(s.depth_mm) if s.depth_mm is not None else None),
                    # distinct penetrations sharing this axis (a 180-degree opposed
                    # set-screw pair is ONE stack but TWO segments)
                    "segments": int(getattr(s, "segments", 1) or 1),
                    "step": sp.name,
                }
            )
    return out


def _mesh_stats(stl_paths: list[Path]) -> dict[str, Any]:
    try:
        import trimesh
    except Exception:
        return {"watertight_all": None, "bbox_mm": None, "body_count": None}
    watertight = True
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    bodies = 0
    any_mesh = False
    for p in stl_paths:
        try:
            m = trimesh.load(str(p), force="mesh")
        except Exception:
            watertight = False
            continue
        any_mesh = True
        watertight = watertight and bool(m.is_watertight)
        bodies += int(m.body_count)
        b = m.bounds
        lo = [min(lo[i], float(b[0][i])) for i in range(3)]
        hi = [max(hi[i], float(b[1][i])) for i in range(3)]
    if not any_mesh:
        return {"watertight_all": None, "bbox_mm": None, "body_count": None}
    return {
        "watertight_all": watertight,
        "bbox_mm": [hi[i] - lo[i] for i in range(3)],
        "body_count": bodies,
    }


# ------------------------------------------------------------------------ evaluation


def _evaluate(
    expect: dict[str, Any],
    job_dir: Path,
    hole_semantics: str = "stacks",
    bbox_semantics: str = "per_axis",
) -> tuple[dict[str, bool], dict[str, Any]]:
    """Every rubric key -> measured verdict against the artifacts on disk.

    Both semantics switches are declared by the frozen CONTENT, so v1 scoring stays
    byte-identical: ``hole_semantics`` (v1 "stacks" -- the historical count; v1.1
    "penetrations" -- each segment of a coaxial stack counts, so an opposed 180-degree
    pair of real holes is 2) and ``bbox_semantics`` (v1 "per_axis" -- world-axis compare;
    v1.1 "sorted" -- orientation-agnostic: an envelope cap tests SIZE, not pose)."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    stls = sorted(job_dir.glob("*.stl"))
    steps = sorted(job_dir.glob("*.step"))

    mesh = _mesh_stats(stls)
    detail["bbox_mm"] = mesh["bbox_mm"]
    detail["body_count"] = mesh["body_count"]
    if expect.get("watertight"):
        checks["watertight"] = mesh["watertight_all"] is True
    if "min_parts" in expect:
        have = max(len(stls), len(steps))
        checks["min_parts"] = have >= int(expect["min_parts"])
    if "bbox_max_mm" in expect:
        bb = mesh["bbox_mm"]
        lim = [float(x) for x in expect["bbox_max_mm"]]
        if bbox_semantics == "sorted" and bb is not None:
            bb, lim = sorted(bb), sorted(lim)
        checks["bbox_max_mm"] = bb is not None and all(bb[i] <= lim[i] for i in range(3))
    if expect.get("named_step"):
        census = _step_census(steps[0]) if steps else None
        detail["step_census"] = census
        checks["named_step"] = bool(
            census
            and census["is_brep"]
            and census["product_names"]
            and any(n and n.upper() != "SOLID" for n in census["product_names"])
        )
    if "holes" in expect:
        holes = _hole_stacks(steps)
        detail["holes_measured"] = [
            {"d": round(h["diameter_mm"], 2), "depth": h["depth_mm"]} for h in holes
        ]
        ok = True
        for h_exp in expect["holes"]:
            d = float(h_exp["diameter_mm"])
            tol = float(h_exp.get("tol_mm", 0.4))
            need = int(h_exp.get("min_count", 1))
            if hole_semantics == "penetrations":
                n = sum(
                    int(h.get("segments", 1)) for h in holes if abs(h["diameter_mm"] - d) <= tol
                )
            else:
                n = sum(1 for h in holes if abs(h["diameter_mm"] - d) <= tol)
            ok = ok and (n >= need)
            detail[f"holes_d{d:g}"] = n
        checks["holes"] = ok
    return checks, detail


def _run_variant(
    label: str,
    variant_ix: int,
    prompt: str,
    expect: dict[str, Any],
    job_dir: Path,
    arm_id: str,
    hole_semantics: str = "stacks",
    bbox_semantics: str = "per_axis",
    fabrication_mode: FabricationMode = FabricationMode.GENERAL,
) -> dict[str, Any]:
    job_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    try:
        result = produce_part(
            prompt,
            job_dir,
            n_drafts=_N_DRAFTS,
            max_iterations=_MAX_ITERATIONS,
            budget_usd=_PART_BUDGET_USD,
            fabrication_mode=fabrication_mode,
            author_model=arm_id,
            client=_PROVIDER_CLIENT,
        )
    except Exception as exc:  # an engine crash is a FAIL with a recorded reason, never a hole
        return {
            "label": label,
            "variant": variant_ix,
            "prompt": prompt,
            "passed": False,
            "engine_passed": False,
            "checks": {},
            "crash": f"{type(exc).__name__}: {exc}",
            "started": started,
        }
    loop = result.loop
    val = loop.validation
    engine_passed = bool(val.passed) if val is not None else bool(loop.geometry_passed)
    if expect.get("expect_refusal"):
        # The intentional infeasibility probe (v1.1 cam_indexing): the CORRECT outcome is a
        # pre-authoring refusal naming the arithmetic (ADR-0084). Building anything -- however
        # tidy -- means the contradiction went undetected, and that is the failure.
        refused = result.disposition == "refused_infeasible"
        checks = {"refused_infeasible": refused}
        detail: dict[str, Any] = {"refusal_reasons": list(loop.refusal_reasons)}
        passed = refused
    else:
        checks, detail = _evaluate(expect, job_dir, hole_semantics, bbox_semantics)
        passed = engine_passed and all(checks.values())
    features_missing = bool(val.features_missing) if val is not None else None
    missing_detail = next(
        (i for i in (val.issues if val is not None else []) if i.startswith("MISSING FEATURES")),
        "",
    )
    failed = val.report.failed_check_ids() if (val is not None and val.report is not None) else []
    return {
        "label": label,
        "variant": variant_ix,
        "prompt": prompt,
        "passed": passed,
        "engine_passed": engine_passed,
        "checks": checks,
        "detail": detail,
        "disposition": result.disposition,
        "stopped_reason": loop.stopped_reason,
        "iterations": len(loop.iterations),
        "failed_checks": failed,
        "features_missing": features_missing,
        "missing_detail": missing_detail,
        "repaired": bool(getattr(loop, "repaired_features", False)),
        "cost_usd": round(result.total_cost_usd, 4),
        # produce_part copies the best render into the (kept) job dir as part.png.
        "render_src": (
            str(job_dir / "part.png")
            if (job_dir / "part.png").exists()
            else (str(loop.final_render) if loop.final_render else None)
        ),
        "started": started,
    }


def _headline(records: list[dict[str, Any]], n_variants: int) -> dict[str, Any]:
    """pass@1, min-across-variants per prompt; headline = mean over prompts with a
    complete variant set (partial prompts reported separately, never extrapolated)."""
    by_label: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_label.setdefault(str(r["label"]), []).append(r)
    complete = {k: v for k, v in by_label.items() if len(v) >= n_variants}
    prompt_pass = {k: all(r["passed"] for r in v) for k, v in complete.items()}
    n = len(complete)
    p = sum(1 for v in prompt_pass.values() if v)
    return {
        "prompts_complete": n,
        "prompts_passed": p,
        "pass_at_1": (p / n if n else 0.0),
        "wilson95_lb": (wilson_lower_bound(p, n) if n else 0.0),
        "per_prompt": {
            k: {
                "min_pass": prompt_pass[k],
                "variants": [{"v": r["variant"], "passed": r["passed"]} for r in complete[k]],
            }
            for k in sorted(complete)
        },
    }


def main() -> int:
    arm = sys.argv[1] if len(sys.argv) > 1 else "opus"
    n_variants = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    n_prompts = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = all
    resume_path = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    shard_ix = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    shard_n = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    arm_id = _MODELS.get(arm)
    if arm_id is None:
        raise SystemExit(f"unknown arm {arm!r}; use sonnet|opus|fable")

    data = _verify_frozen()
    prompts = data["prompts"][: n_prompts or len(data["prompts"])]
    if shard_n > 1:
        prompts = [p for i, p in enumerate(prompts) if i % shard_n == shard_ix]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_s{shard_ix}" if shard_n > 1 else ""
    # v1 keeps its historical dir prefix (tooling globs arbitrary_<arm>_*); other versions
    # get the slug so runs are never mistaken for v1 numbers.
    prefix = "arbitrary" if _BENCH_SLUG == "arbitrary_v1" else _BENCH_SLUG
    out_dir = Path("benchmarks/results") / f"{prefix}_{arm}_{stamp}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "results.json"
    render_dir = out_dir / "renders"
    render_dir.mkdir(exist_ok=True)

    records: list[dict[str, Any]] = []
    done: set[tuple[str, int]] = set()
    if resume_path is not None and resume_path.exists():
        prev = json.loads(resume_path.read_text(encoding="utf-8"))
        records = list(prev.get("results", []))
        done = {(str(r["label"]), int(r["variant"])) for r in records}
        print(f"-- resuming: {len(records)} variants already recorded --", flush=True)

    def dump() -> None:
        out_json.write_text(
            json.dumps(
                {
                    "benchmark": data["version"],
                    "manifest_sha": _sha256(BENCH_FILE),
                    "arm": arm,
                    "arm_model": arm_id,
                    "provider": ("claude_code" if _PROVIDER_CLIENT is not None else "api"),
                    "author_max_tokens_env": os.environ.get("FORGE_AUTHOR_MAX_TOKENS"),
                    "n_variants": n_variants,
                    "max_iterations": _MAX_ITERATIONS,
                    "headline": _headline(records, n_variants),
                    "engine_pass_rate_all_variants": (
                        sum(1 for r in records if r["passed"]) / len(records) if records else 0.0
                    ),
                    "total_cost_usd": round(sum(float(r.get("cost_usd", 0)) for r in records), 2),
                    "results": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for i, entry in enumerate(prompts):
        label = str(entry["id"])
        variants = [str(entry["prompt"]), *[str(p) for p in entry.get("paraphrases", [])]]
        variants = variants[:n_variants]
        expect = dict(entry.get("expect", {}))
        for vix, vprompt in enumerate(variants):
            if (label, vix) in done:
                continue
            rec = _run_variant(
                label,
                vix,
                vprompt,
                expect,
                out_dir / f"{label}_v{vix}",
                arm_id,
                hole_semantics=str(data.get("hole_count_semantics", "stacks")),
                bbox_semantics=str(data.get("bbox_semantics", "per_axis")),
            )
            records.append(rec)
            src = rec.get("render_src")
            if isinstance(src, str) and Path(src).exists():
                verdict = "PASS" if rec["passed"] else "FAIL"
                dst = render_dir / f"{i + 1:02d}_{label}_v{vix}_{verdict}.png"
                try:
                    shutil.copy2(src, dst)
                    rec["render"] = str(dst)
                except OSError:
                    pass
            failed_keys = [k for k, v in rec.get("checks", {}).items() if not v]
            why = (
                rec.get("crash")
                or (", ".join(failed_keys) if failed_keys else "")
                or ("engine_fail: " + ",".join(rec.get("failed_checks", [])))
                if not rec["passed"]
                else ""
            )
            print(
                f"[{i + 1:2}/{len(prompts)} v{vix}] {label:28} "
                f"{'PASS' if rec['passed'] else 'FAIL':4} "
                f"it={rec.get('iterations', '-')} {rec.get('stopped_reason', 'crash'):11} {why}",
                flush=True,
            )
            dump()  # kill-safe: every variant lands immediately

    h = _headline(records, n_variants)
    print(
        f"\n=== arbitrary_v1 pass@1 ({arm}, min-across-{n_variants}-variants): "
        f"{h['prompts_passed']}/{h['prompts_complete']} = {h['pass_at_1'] * 100:.0f}%  "
        f"Wilson95 LB {h['wilson95_lb'] * 100:.0f}% ===",
        flush=True,
    )
    print(f"wrote {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
