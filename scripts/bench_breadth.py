"""Breadth benchmark scorer v1 (PAID) -- the FROZEN pre-registered breadth set, pass@1.

Protocol (Phase 0 of the arm+breadth objective; see benchmarks/breadth_v1/README.md):
  * Content (prompts + rubric + paraphrases) lives in benchmarks/breadth_v1/breadth_v1.json,
    authored by a NON-author model (sonnet) and FROZEN: MANIFEST.sha256 pins it and this
    script REFUSES to run if the hash drifted. The CAD author model never sees the rubric.
  * pass@1: every variant (base + 2 paraphrases) runs ONCE -- no retries, no best-of.
    A prompt's score is the MIN across its variants; the headline is the mean over prompts.
  * A variant PASSES iff the engine's own gate holds (status ok, validation.passed,
    confidence.basis == "validated") AND every rubric expectation verifies AGAINST THE
    ARTIFACTS. Artifact checks re-measure independently: the GLB is parsed at the byte level
    (node tree + ck_joint extras), the STEP gets a text census (=PRODUCT( entities, B-rep
    schema markers), holes/mounts are measured on the reimported B-rep (validator geometry
    utilities -- measurement code, not engine self-claims), meshes via trimesh.
  * Build cache is force-disabled for the run (a cache hit would fake pass@1).

Args: [cap_usd] [only_prefix]
Run:  uv run --no-sync python -u scripts/bench_breadth.py 30
Writes benchmarks/results/breadth_<stamp>/ (per-variant job dirs + breadth_<stamp>.json,
rewritten after EVERY variant so a kill never loses finished work).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import struct
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The bench measures the product without the persistence/billing layers; a build-cache hit
# would replay an old artifact and fake pass@1, so force it off before importing the engine.
# FORGE_BUILD_CACHE is a cache DIRECTORY PATH (falsy = disabled): "0" ENABLED a cache at
# literal path ./0 (the arbitrary_v1 BEFORE run caught it live). Empty string = off.
os.environ["FORGE_BUILD_CACHE"] = ""

from forge_mind.api.core import generate_part
from forge_mind.llm.claude_code_client import subscription_client_if_selected

# FORGE_LLM_PROVIDER=claude_code bills the AGENTIC engine's calls to the subscription
# (iteration runs, $0 API). The deterministic lanes' small spec-pick calls still use
# the API key (~$0.02-0.10/prompt). Provider is recorded in the results JSON; only
# api-provider runs are citable for swap decisions.
_PROVIDER_CLIENT = subscription_client_if_selected()

BENCH_DIR = Path("benchmarks/breadth_v1")
BENCH_FILE = BENCH_DIR / "breadth_v1.json"
MANIFEST = BENCH_DIR / "MANIFEST.sha256"
PATTERNS_FILE = BENCH_DIR / "mount_patterns.json"

# Refusals/errors don't surface their LLM spend in the API result; bill them at a
# conservative flat estimate so the cap can never be dodged by a string of refusals.
_UNPRICED_VARIANT_USD = 0.25


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


# ----------------------------------------------------------------------------- GLB census


def _glb_json(glb_path: Path) -> dict[str, Any] | None:
    """Parse the JSON chunk straight out of the GLB container (no loader 'help')."""
    try:
        raw = glb_path.read_bytes()
        magic, _version, _length = struct.unpack_from("<4sII", raw, 0)
        if magic != b"glTF":
            return None
        off = 12
        while off + 8 <= len(raw):
            clen, ctype = struct.unpack_from("<II", raw, off)
            off += 8
            if ctype == 0x4E4F534A:  # 'JSON'
                return json.loads(raw[off : off + clen].decode("utf-8"))
            off += clen
        return None
    except Exception:
        return None


def _glb_census(glb_path: Path) -> dict[str, Any]:
    """Mesh-node count + every ck_joint extras payload, read independently of the engine."""
    gltf = _glb_json(glb_path)
    if gltf is None:
        return {"mesh_nodes": 0, "joints": []}
    nodes = gltf.get("nodes", []) or []
    mesh_nodes = sum(1 for n in nodes if isinstance(n, dict) and "mesh" in n)
    joints = [
        n["extras"]["ck_joint"]
        for n in nodes
        if isinstance(n, dict)
        and isinstance(n.get("extras"), dict)
        and isinstance(n["extras"].get("ck_joint"), dict)
    ]
    return {"mesh_nodes": mesh_nodes, "joints": joints}


# ---------------------------------------------------------------------------- STEP census


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


# ------------------------------------------------------------------------ B-rep measuring


def _hole_stacks(step_paths: list[Path]) -> list[dict[str, Any]]:
    """Measured concave cylinder stacks (physical hole walls) across every part STEP.
    Uses the validator's measurement utilities (shared measuring code, not self-claims)."""
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
                    "point": tuple(float(c) for c in s.axis_point),
                    "axis": tuple(float(c) for c in s.axis_dir),
                    "depth_mm": (float(s.depth_mm) if s.depth_mm is not None else None),
                    "step": sp.name,
                }
            )
    return out


def _pairwise(points: list[tuple[float, float, float]]) -> list[float]:
    return sorted(
        ((ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2) ** 0.5
        for (ax, ay, az), (bx, by, bz) in itertools.combinations(points, 2)
    )


def _match_pattern(holes: list[dict[str, Any]], pattern: dict[str, Any], need: int) -> int:
    """Count DISJOINT hole groups matching a mount pattern's diameter range + pairwise-distance
    fingerprint (greedy). A single-bore pattern counts matching stacks directly."""
    d_lo, d_hi = pattern["hole_diameter_mm"]
    cands = [h for h in holes if d_lo <= h["diameter_mm"] <= d_hi]
    n_holes = int(pattern.get("holes", 1))
    if n_holes <= 1:
        min_depth = float(pattern.get("min_depth_mm", 0.0))
        return sum(1 for h in cands if (h["depth_mm"] or 0.0) >= min_depth)
    want = sorted(float(x) for x in pattern["pairwise_mm"])
    tol = float(pattern.get("tol_mm", 0.8))
    found = 0
    pool = list(cands)
    while found < need and len(pool) >= n_holes:
        hit: tuple[int, ...] | None = None
        for combo in itertools.combinations(range(len(pool)), n_holes):
            got = _pairwise([pool[i]["point"] for i in combo])
            if len(got) == len(want) and all(
                abs(g - w) <= tol for g, w in zip(got, want, strict=True)
            ):
                hit = combo
                break
        if hit is None:
            break
        found += 1
        pool = [h for i, h in enumerate(pool) if i not in set(hit)]
    return found


# ------------------------------------------------------------------------------ artifacts


def _collect_artifacts(job_dir: Path) -> dict[str, Any]:
    stls = sorted(job_dir.glob("*.stl")) + sorted((job_dir / "parts").glob("**/*.stl"))
    steps = sorted(job_dir.glob("*.step")) + sorted((job_dir / "parts").glob("**/*.step"))
    glbs = sorted(job_dir.glob("*.glb"))
    return {"stl": stls, "step": steps, "glb": glbs}


def _mesh_stats(stl_paths: list[Path]) -> dict[str, Any]:
    try:
        import trimesh
    except Exception:
        return {"watertight_all": None, "bbox_mm": None}
    watertight = True
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    any_mesh = False
    for p in stl_paths:
        try:
            m = trimesh.load(str(p), force="mesh")
        except Exception:
            watertight = False
            continue
        any_mesh = True
        watertight = watertight and bool(m.is_watertight)
        b = m.bounds
        lo = [min(lo[i], float(b[0][i])) for i in range(3)]
        hi = [max(hi[i], float(b[1][i])) for i in range(3)]
    if not any_mesh:
        return {"watertight_all": None, "bbox_mm": None}
    return {"watertight_all": watertight, "bbox_mm": [hi[i] - lo[i] for i in range(3)]}


# -------------------------------------------------------------------------------- scoring


def _evaluate(
    expect: dict[str, Any],
    res: Any,
    job_dir: Path,
    patterns: dict[str, Any],
) -> dict[str, Any]:
    """Every rubric key -> measured verdict. The variant passes iff ALL checks hold."""
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}

    ok = getattr(res, "ok", None)
    conf = getattr(ok, "confidence", None) if ok is not None else None
    basis = getattr(conf, "basis", None) if conf is not None else None
    checks["engine_ok"] = res.status == "ok"
    checks["engine_validated"] = bool(
        ok is not None and ok.validation.passed and basis == "validated"
    )
    detail["status"] = res.status
    detail["gen_path"] = getattr(ok, "gen_path", None)
    detail["disposition"] = getattr(ok, "disposition", None)
    detail["basis"] = basis

    art = _collect_artifacts(job_dir)
    glb = _glb_census(art["glb"][0]) if art["glb"] else {"mesh_nodes": 0, "joints": []}
    step = (
        _step_census(art["step"][0])
        if art["step"]
        else {"products": 0, "is_brep": False, "tessellated": True, "product_names": []}
    )
    parts_detected = max(glb["mesh_nodes"], step["products"], 1 if art["stl"] else 0)
    detail["parts_detected"] = parts_detected
    detail["glb_mesh_nodes"] = glb["mesh_nodes"]
    detail["step_products"] = step["products"]
    detail["joints_found"] = glb["joints"]

    if "min_parts" in expect:
        checks["min_parts"] = parts_detected >= int(expect["min_parts"])
    if "joints" in expect:
        for j_exp in expect["joints"]:
            jt, need = str(j_exp["type"]), int(j_exp["min_count"])
            have = [j for j in glb["joints"] if j.get("type") == jt]
            checks[f"joints_{jt}"] = len(have) >= need
            if jt == "revolute" and "min_sweep_deg" in expect:
                span_ok = [
                    j
                    for j in have
                    if isinstance(j.get("limits_deg"), list)
                    and len(j["limits_deg"]) == 2
                    and (j["limits_deg"][1] - j["limits_deg"][0]) >= float(expect["min_sweep_deg"])
                ]
                checks["min_sweep_deg"] = len(span_ok) >= need
    if expect.get("collision_checked") == "swept":
        checks["collision_checked_swept"] = bool(glb["joints"]) and all(
            j.get("checked") == "swept" for j in glb["joints"]
        )

    needs_brep = "holes" in expect or "mounts" in expect
    holes = _hole_stacks(art["step"]) if needs_brep else []
    if needs_brep:
        detail["holes_measured"] = [
            {"d": round(h["diameter_mm"], 2), "depth": h["depth_mm"]} for h in holes
        ]
    if "holes" in expect:
        hole_ok = True
        for h_exp in expect["holes"]:
            d, tol = float(h_exp["diameter_mm"]), float(h_exp.get("tol_mm", 0.3))
            n = sum(1 for h in holes if abs(h["diameter_mm"] - d) <= tol)
            hole_ok = hole_ok and n >= int(h_exp["min_count"])
        checks["holes"] = hole_ok
    if "mounts" in expect:
        mounts_ok = True
        for m_exp in expect["mounts"]:
            key, need = str(m_exp["component"]), int(m_exp["min_count"])
            pat = patterns.get(key)
            if pat is None:
                mounts_ok = False
                detail[f"mount_{key}"] = "NO FROZEN PATTERN -- counted as unmet"
                continue
            got = _match_pattern(holes, pat, need)
            detail[f"mount_{key}"] = got
            mounts_ok = mounts_ok and got >= need
        checks["mounts"] = mounts_ok

    mesh = _mesh_stats(art["stl"]) if ("watertight" in expect or "bbox_max_mm" in expect) else {}
    if expect.get("watertight"):
        checks["watertight"] = mesh.get("watertight_all") is True
    if "bbox_max_mm" in expect:
        bb = mesh.get("bbox_mm")
        cap = [float(x) for x in expect["bbox_max_mm"]]
        checks["bbox"] = bb is not None and all(bb[i] <= cap[i] for i in range(3))
        detail["bbox_mm"] = bb
    if expect.get("named_step"):
        names = [n for n in step["product_names"] if n and n.upper() != "SOLID"]
        checks["named_step"] = step["is_brep"] and not step["tessellated"] and bool(names)

    return {"passed": all(checks.values()), "checks": checks, "detail": detail}


def _dump(out_json: Path, payload: dict[str, Any]) -> None:
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def main() -> int:
    cap_usd = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    only = sys.argv[2] if len(sys.argv) > 2 else ""
    bench = _verify_frozen()
    patterns: dict[str, Any] = {}
    if PATTERNS_FILE.exists():
        patterns = json.loads(PATTERNS_FILE.read_text(encoding="utf-8"))

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = Path("benchmarks/results") / f"breadth_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    out_json = root / f"breadth_{stamp}.json"

    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    stopped = ""
    prompts = [p for p in bench["prompts"] if str(p["id"]).startswith(only)]
    for pi, entry in enumerate(prompts):
        variants = [("base", entry["prompt"])] + [
            (f"p{i + 1}", t) for i, t in enumerate(entry.get("paraphrases", []))
        ]
        for vkey, text in variants:
            if total_cost >= cap_usd:
                stopped = f"spend cap ${cap_usd:.2f} reached"
                break
            job = root / f"{entry['id']}__{vkey}"
            job.mkdir(parents=True, exist_ok=True)
            t0 = datetime.now(UTC)
            try:
                res = generate_part(
                    text,
                    job,
                    quality="standard",
                    fabrication_mode="general",
                    client=_PROVIDER_CLIENT,
                )
            except Exception as exc:  # never lose the run to one crash
                res = type("R", (), {"status": "error", "ok": None})()
                (job / "CRASH.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
            dt = (datetime.now(UTC) - t0).total_seconds()
            cost = float(res.ok.cost_usd) if getattr(res, "ok", None) else _UNPRICED_VARIANT_USD
            total_cost += cost
            verdict = _evaluate(entry.get("expect", {}), res, job, patterns)
            rows.append(
                {
                    "id": entry["id"],
                    "class": entry["class"],
                    "variant": vkey,
                    "prompt": text,
                    "passed": verdict["passed"],
                    "checks": verdict["checks"],
                    "detail": verdict["detail"],
                    "cost_usd": round(cost, 4),
                    "duration_s": round(dt, 1),
                }
            )
            failed = [k for k, v in verdict["checks"].items() if not v]
            print(
                f"[{pi + 1:2}/{len(prompts)} {entry['id']:28} {vkey:4}] "
                f"{'PASS' if verdict['passed'] else 'FAIL':4} ${cost:5.2f} {dt:5.0f}s "
                f"{('miss: ' + ','.join(failed)) if failed else ''}",
                flush=True,
            )
            # Aggregate after every variant: min-across-variants per prompt, mean over prompts.
            by_id: dict[str, list[bool]] = {}
            for r in rows:
                by_id.setdefault(r["id"], []).append(bool(r["passed"]))
            per_prompt = {k: int(all(v)) for k, v in by_id.items()}
            by_class: dict[str, list[int]] = {}
            for r in rows:
                if r["id"] in per_prompt:
                    by_class.setdefault(r["class"], [])
            for k, v in per_prompt.items():
                cls = next(r["class"] for r in rows if r["id"] == k)
                by_class[cls].append(v)
            _dump(
                out_json,
                {
                    "benchmark": bench["version"],
                    "provider": ("claude_code" if _PROVIDER_CLIENT is not None else "api"),
                    "manifest": MANIFEST.read_text(encoding="utf-8"),
                    "protocol": "pass@1; min across base+2 paraphrases; mean over prompts",
                    "stamp": stamp,
                    "n_prompts_scored": len(per_prompt),
                    "pass_at_1": round(sum(per_prompt.values()) / max(len(per_prompt), 1), 4),
                    "per_class": {
                        k: {"n": len(v), "passed": sum(v)} for k, v in sorted(by_class.items())
                    },
                    "per_prompt": per_prompt,
                    "total_cost_usd": round(total_cost, 2),
                    "stopped": stopped,
                    "rows": rows,
                },
            )
        if stopped:
            break

    print(f"\nwrote {out_json}  (total ${total_cost:.2f}) {stopped}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
