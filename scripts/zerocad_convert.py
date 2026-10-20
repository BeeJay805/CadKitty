"""ZERO-TO-CAD-1M ingestion: convert Autodesk's ~1M CadQuery dataset into VERIFIED build123d
training pairs (founder-directed 2026-07-15 -- "use premade data instead of generating it all").

Why this beats pure self-harvest: Zero-To-CAD-1M covers REAL construction (revolve, loft, sweep,
fillet, chamfer, shell, patterns -- not just extrude), kernel-validated, Apache-2.0, ~1M samples.
Confirmed 15/15 local samples pass our OWN fidelity gate. The only gaps: it's CadQuery (our engine
speaks build123d) and has no text prompts (captioned separately, later). This script closes the
language gap with LLM TRANSLATION (a far easier task than authoring from scratch -- the exact ops
and args are already known) verified GEOMETRICALLY against the dataset's own ground-truth STEP, so
a bad translation can never contaminate the training set.

Pipeline per sample (local qwen-coder, $0, does not contend with the Claude teacher harvest -- a
different resource pool):
  CadQuery code --translate--> build123d code --sandbox build--> STEP'
  compare (volume, bbox) against the dataset's ground-truth STEP -- keep iff it matches.
One retry with the mismatch as feedback before giving up on a sample.

STREAMING + OUT-OF-ONEDRIVE (the lesson from the harvest's OneDrive-sync-contention incident): each
parquet (~114MB, ~416 samples) is downloaded, fully processed, then DELETED -- disk footprint stays
~1 file. All heavy I/O (raw cache, per-sample sandbox scratch, the converted dataset, stats) lives
under ~/.forge-train/ (outside both the repo and OneDrive), mirroring the existing parts.jsonl tap.

Resumable: a file-level manifest (done parquet files) + a uuid-level done-set (from the stats log)
so a kill mid-file re-downloads that file but skips already-verified rows. Self-healing PID lock (one
instance). Fail-open: any error on a sample is logged and skipped, never crashes the run.

Run: python scripts/zerocad_convert.py [max_files]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home()  # C:\Users\brenn -- outside OneDrive
CACHE = HOME / ".forge-train" / "zerocad_raw"
WORK = HOME / ".forge-train" / "zerocad_work"
CONVERTED = (
    HOME / ".forge-train" / "zerocad_converted.jsonl"
)  # verified (cq_code, b123d_code) pairs
STATS = (
    HOME / ".forge-train" / "zerocad_stats.jsonl"
)  # every attempt, pass/fail, for yield tracking
MANIFEST = HOME / ".forge-train" / "zerocad_manifest.jsonl"  # which parquet files are fully done
LOCKDIR = HOME / ".forge-train" / ".zerocad.lock"

HF_API = "https://huggingface.co/api/datasets/ADSKAILab/Zero-To-CAD-1m"
HF_RESOLVE = "https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m/resolve/main/"

VOL_TOL = 0.05  # 5% relative volume tolerance
BBOX_TOL = 0.05  # 5% relative per-axis bbox tolerance

_CHEATSHEET = """You translate CadQuery Python (fluent Workplane API) into build123d Python
(builder mode). Both run on the SAME OpenCASCADE kernel via OCP -- this is a faithful 1:1 geometric
translation, not a redesign. Preserve every dimension EXACTLY.

THE #1 MISTAKE TO AVOID: build123d sketch/line ops are OBJECTS you instantiate, NEVER bare method
calls like CadQuery's cursor-based moveTo/lineTo/close/circle/rect. There is no `moveTo()`,
`lineTo()`, `close()`, `circle()`, or `rect()` FUNCTION in build123d -- calling them raises
NameError. Use `Line(p1, p2)`, `ThreePointArc(p1, p2, p3)`, `Circle(radius)`, `Rectangle(w, h)`
as objects; a sketch closes automatically when the wire is continuous (no explicit close() call).

WORKED EXAMPLE 1 -- profile + extrude + hole pattern + fillet (study this structure exactly):
```python
from build123d import *

length, width, thickness = 80.0, 40.0, 8.0
hole_d, hole_spacing, hole_count = 4.5, 20.0, 3
start_offset = -(hole_spacing * (hole_count - 1)) / 2

with BuildPart() as bp:
    with BuildSketch():
        Rectangle(length, width)
    extrude(amount=thickness)

    top = bp.faces().sort_by(Axis.Z)[-1]
    with BuildSketch(top):
        with GridLocations(hole_spacing, 0, hole_count, 1):
            Circle(hole_d / 2)
    extrude(amount=-thickness, mode=Mode.SUBTRACT)

    fillet(bp.edges().filter_by(Axis.Z), radius=1.0)

result = bp.part
```

WORKED EXAMPLE 2 -- an arc profile (NOT moveTo/lineTo/threePointArc as bare calls):
```python
with BuildPart() as bp:
    with BuildSketch():
        with BuildLine():
            Line((0, 0), (40, 0))
            ThreePointArc((40, 0), (46, 12), (40, 24))
            Line((40, 24), (0, 24))
            Line((0, 24), (0, 0))
        make_face()
    extrude(amount=12.0)
result = bp.part
```

FURTHER MAPPINGS:
  .revolve(angle, axisStart, axisEnd) -> profile in BuildSketch, then
                                          revolve(axis=Axis((p0),(p1)), revolution_arc=angle)
  .loft()                             -> loft([sketch1, sketch2, ...])
  .shell(t)                           -> offset(bp.part, amount=-t, openings=<faces>)
  .chamfer(d)                         -> chamfer(bp.edges().filter_by(...), length=d)
  .cboreHole(d,cd,cd_depth,depth)     -> two stacked Circle+extrude(SUBTRACT) passes (wide then narrow)
  .cskHole(d,cd,angle,depth)          -> a plain hole, then chamfer() its top edge
  .polarArray(r,start,angle,count)    -> with PolarLocations(r, count, start_angle=start,
                                               angular_range=angle):
  .translate((x,y,z)) / .rotate(...)  -> Pos(x,y,z) * shape   /   Rot(x,y,z) * shape

HARD RULES:
1. `from build123d import *` at the top.
2. There is NO `shell()` function in build123d (CadQuery's `.shell(t)` has no same-named
   equivalent) -- use `offset(bp.part, amount=-t, openings=<faces to remove>)` instead.
3. There is NO `.add()` method on a BuildPart object. To union a second solid into the same part,
   author it as another sketch+extrude step INSIDE the same `with BuildPart() as bp:` block (each
   extrude defaults to Mode.ADD, i.e. it fuses automatically) -- do not call bp.add(...).
4. Never call moveTo/lineTo/close/circle/rect/threePointArc/polyline/rotate/translate/move/val/wire
   as bare functions -- always the object form (Line/Circle/Rectangle/ThreePointArc/Polyline) shown
   in the examples above, or `Pos(x,y,z) * shape` / `Rot(x,y,z) * shape` for translate/rotate.
5. Do NOT wrap the part in a custom class. Just `with BuildPart() as bp: ...` then, at module level
   (NOT indented inside anything), the single line: `result = bp.part`
6. Every `with GridLocations(...):`, `with PolarLocations(...):`, `with Locations(...):` is used
   directly as a context manager -- never nested inside another Locations unless the source genuinely
   nests two patterns.
7. Keep every numeric literal IDENTICAL to the source. Do not invent features the source lacks.
8. Output ONLY the complete runnable script, no explanation."""


class TranslatedCode(BaseModel):
    code: str


def _normalize_preamble_indent(code: str) -> str:
    """Fix a recurring qwen artifact: a stray 1-3 space indent on the first few top-level
    variable-assignment lines (before the first `with`/`def`/`for`/`if`/`class` block opens).
    Only touches lines strictly BEFORE the first block starts, so real block indentation inside
    BuildPart/BuildSketch is never altered -- safe by construction, not a general re-indenter."""
    lines = code.split("\n")
    out: list[str] = []
    hit_block = False
    for line in lines:
        if hit_block:
            out.append(line)
            continue
        s = line.strip()
        if s.startswith(("with ", "for ", "if ", "def ", "class ")):
            hit_block = True
            out.append(s if not line.startswith(s) else line)
        else:
            out.append(s)
    return "\n".join(out)


def _translate(client: Any, cq_code: str, ops_summary: str, feedback: str = "") -> str | None:
    user = (
        f"Translate this CadQuery script to build123d.\n\nCADQUERY SOURCE:\n```python\n{cq_code}\n```"
        f"\n\nPARSED OPERATIONS (structure hint): {ops_summary}"
    )
    if feedback:
        user += f"\n\nYOUR PREVIOUS ATTEMPT DID NOT MATCH THE TARGET GEOMETRY: {feedback}\nFix it."
    try:
        inst, _usage = client.generate_structured(
            system=_CHEATSHEET,
            user=user,
            output_model=TranslatedCode,
            model="qwen3-coder:30b",
            max_tokens=4000,
        )
        code = inst.code.strip()
        if code.startswith("```"):
            code = code.split("```")[1]
            code = code[code.find("\n") + 1 :] if code.startswith("python") else code
        return _normalize_preamble_indent(code.strip())
    except Exception:
        return None


def _ground_truth_metrics(
    step_bytes: bytes, tmp_dir: Path
) -> tuple[float, tuple[float, float, float]] | None:
    """Load the dataset's own ground-truth STEP -> (volume_mm3, (dx,dy,dz)). None if unreadable."""
    try:
        from build123d import import_step

        from forge_mind.agentic.brep_checks import brep_volume

        p = tmp_dir / "gt.step"
        p.write_bytes(step_bytes)
        part = import_step(str(p))
        vol = brep_volume(part)
        bb = part.bounding_box()
        ext = (
            float(bb.max.X - bb.min.X),
            float(bb.max.Y - bb.min.Y),
            float(bb.max.Z - bb.min.Z),
        )
        p.unlink(missing_ok=True)
        if vol is None or vol <= 0:
            return None
        return vol, ext
    except Exception:
        return None


def _translated_metrics(step_path: Path) -> tuple[float, tuple[float, float, float]] | None:
    try:
        from build123d import import_step

        from forge_mind.agentic.brep_checks import brep_volume

        part = import_step(str(step_path))
        vol = brep_volume(part)
        bb = part.bounding_box()
        ext = (
            float(bb.max.X - bb.min.X),
            float(bb.max.Y - bb.min.Y),
            float(bb.max.Z - bb.min.Z),
        )
        if vol is None or vol <= 0:
            return None
        return vol, ext
    except Exception:
        return None


def _geometry_matches(
    gt: tuple[float, tuple[float, float, float]], got: tuple[float, tuple[float, float, float]]
) -> tuple[bool, str]:
    gt_vol, gt_ext = gt
    got_vol, got_ext = got
    vol_err = abs(got_vol - gt_vol) / max(gt_vol, 1e-9)
    if vol_err > VOL_TOL:
        return (
            False,
            f"volume mismatch: target {gt_vol:.1f} mm^3, got {got_vol:.1f} ({vol_err:.1%} off)",
        )
    gt_sorted = sorted(gt_ext)
    got_sorted = sorted(got_ext)
    for i, (a, b) in enumerate(zip(gt_sorted, got_sorted, strict=False)):
        err = abs(b - a) / max(a, 1e-6)
        if err > BBOX_TOL:
            return False, (
                f"bbox mismatch on axis rank {i}: target {gt_sorted} mm, got {got_sorted} mm"
            )
    return True, "ok"


def _list_parquet_files(http: httpx.Client) -> list[str]:
    r = http.get(f"{HF_API}", timeout=30.0)
    r.raise_for_status()
    sibs = r.json().get("siblings", [])
    files = sorted(s["rfilename"] for s in sibs if s.get("rfilename", "").endswith(".parquet"))
    return files


def _done_files() -> set[str]:
    if not MANIFEST.exists():
        return set()
    out = set()
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "done":
                out.add(r["file"])
    return out


def _done_uuids() -> set[str]:
    if not STATS.exists():
        return set()
    out = set()
    for line in STATS.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if not r.get("infra_fail"):
                out.add(r["uuid"])
    return out


def main() -> int:
    max_files = int(sys.argv[1]) if len(sys.argv) > 1 else 999999

    LOCKDIR.parent.mkdir(parents=True, exist_ok=True)
    if not _acquire_lock():
        print("ANOTHER LIVE zerocad_convert HOLDS THE LOCK -- exiting.", file=sys.stderr)
        return 3
    try:
        _run(max_files)
    finally:
        _release_lock()
    return 0


def _acquire_lock() -> bool:
    try:
        LOCKDIR.mkdir()
    except FileExistsError:
        holder = (
            (LOCKDIR / "pid").read_text(encoding="utf-8").strip()
            if (LOCKDIR / "pid").exists()
            else ""
        )
        if holder and _pid_alive(int(holder)):
            return False
        shutil.rmtree(LOCKDIR, ignore_errors=True)
        try:
            LOCKDIR.mkdir()
        except FileExistsError:
            return False
    (LOCKDIR / "pid").write_text(str(os.getpid()), encoding="utf-8")
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return True  # unsupported on this platform -- assume alive, fail safe (never double-run)


def _release_lock() -> None:
    shutil.rmtree(LOCKDIR, ignore_errors=True)


def _run(max_files: int) -> None:
    from forge_mind.llm.ollama_client import OllamaClient

    CACHE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    CONVERTED.parent.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(model="qwen3-coder:30b")
    http = httpx.Client(timeout=120.0, follow_redirects=True)

    files = _list_parquet_files(http)
    done_files = _done_files()
    todo = [f for f in files if f not in done_files][:max_files]
    done_uuids = _done_uuids()
    print(
        f"zerocad_convert: {len(files)} total parquet files, {len(done_files)} already done, "
        f"{len(todo)} to process this run, {len(done_uuids)} uuids already verified/logged",
        flush=True,
    )

    import pyarrow.parquet as pq

    n_pass = n_fail = n_seen = 0
    t_start = time.perf_counter()

    for fi, fname in enumerate(todo):
        local = CACHE / Path(fname).name
        try:
            with http.stream("GET", HF_RESOLVE + fname) as resp:
                resp.raise_for_status()
                with local.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
        except Exception as e:
            print(f"[{fi + 1}/{len(todo)}] DOWNLOAD FAILED {fname}: {str(e)[:100]}", flush=True)
            continue

        try:
            table = (
                pq.ParquetFile(str(local))
                .read(
                    columns=[
                        "uuid",
                        "cadquery_file",
                        "cadquery_ops_json",
                        "cadquery_ops_count",
                        "num_faces",
                        "step_file",
                    ]
                )
                .to_pydict()
            )
        except Exception as e:
            print(f"[{fi + 1}/{len(todo)}] PARQUET READ FAILED {fname}: {str(e)[:100]}", flush=True)
            local.unlink(missing_ok=True)
            continue

        n_rows = len(table["uuid"])
        for i in range(n_rows):
            uuid = table["uuid"][i]
            if uuid in done_uuids:
                continue
            n_seen += 1
            cq_code = table["cadquery_file"][i]
            cq_code = (
                cq_code.decode("utf-8", "ignore")
                if isinstance(cq_code, (bytes, bytearray))
                else str(cq_code)
            )
            ops_json = table["cadquery_ops_json"][i]
            step_bytes = table["step_file"][i]
            step_bytes = bytes(step_bytes) if not isinstance(step_bytes, bytes) else step_bytes

            t0 = time.perf_counter()
            row = _process_one(uuid, cq_code, ops_json, step_bytes, client)
            row["seconds"] = round(time.perf_counter() - t0, 1)
            with STATS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            if row.get("verdict") == "pass":
                n_pass += 1
                with CONVERTED.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "uuid": uuid,
                                "cadquery_code": cq_code,
                                "build123d_code": row["b123d_code"],
                                "num_faces": table["num_faces"][i],
                                "fid_verdict": row.get("fid_verdict"),
                                "fid_score": row.get("fid_score"),
                            }
                        )
                        + "\n"
                    )
            elif not row.get("infra_fail"):
                n_fail += 1
            tag = (
                "PASS"
                if row.get("verdict") == "pass"
                else ("INFRA" if row.get("infra_fail") else "fail")
            )
            elapsed = time.perf_counter() - t_start
            rate = n_seen / elapsed * 3600 if elapsed > 0 else 0
            print(
                f"[{fi + 1}/{len(todo)} file, row {i + 1}/{n_rows}] {tag:5} {row['seconds']:5.1f}s  "
                f"{uuid[:8]}  total pass={n_pass} fail={n_fail}  ({rate:.0f}/hr)",
                flush=True,
            )

        local.unlink(missing_ok=True)
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"file": fname, "status": "done"}) + "\n")

    print(f"\nzerocad_convert DONE this run: seen={n_seen} pass={n_pass} fail={n_fail}", flush=True)


def _process_one(
    uuid: str, cq_code: str, ops_json: str, step_bytes: bytes, client: Any
) -> dict[str, Any]:
    from forge_mind.agentic.fidelity import fidelity_report
    from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code

    tmp = Path(tempfile.mkdtemp(dir=str(WORK)))
    try:
        gt = _ground_truth_metrics(step_bytes, tmp)
        if gt is None:
            return {
                "uuid": uuid,
                "verdict": "fail",
                "infra_fail": True,
                "reason": "ground truth unreadable",
            }

        try:
            ops = json.loads(ops_json)
            ops_summary = ", ".join(sorted({o.get("op_name", "?") for o in ops}))
        except Exception:
            ops_summary = "(unavailable)"

        feedback = ""
        for attempt in range(2):
            b123d_code = _translate(client, cq_code, ops_summary, feedback)
            if not b123d_code:
                continue
            job_dir = tmp / f"attempt_{attempt}"
            result = run_build123d_code(
                b123d_code, job_dir, part_name="result", limits=SandboxLimits(timeout_s=90)
            )
            if not result.ok or result.step_path is None:
                feedback = f"the code failed to build: {result.error_type} {str(result.error_message)[:200]}"
                continue
            got = _translated_metrics(result.step_path)
            if got is None:
                feedback = "the built STEP could not be re-loaded for verification"
                continue
            ok, reason = _geometry_matches(gt, got)
            if ok:
                fid = fidelity_report(result.step_path, b123d_code, "part")
                return {
                    "uuid": uuid,
                    "verdict": "pass",
                    "infra_fail": False,
                    "attempt": attempt,
                    "b123d_code": b123d_code,
                    "fid_verdict": fid.verdict,
                    "fid_score": fid.score,
                }
            feedback = reason
        return {"uuid": uuid, "verdict": "fail", "infra_fail": False, "reason": feedback[:200]}
    except Exception as e:
        return {"uuid": uuid, "verdict": "fail", "infra_fail": True, "reason": str(e)[:200]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
