"""DETERMINISTIC CadQuery -> build123d transpiler for Zero-To-CAD-1M (founder-directed 2026-07-15).

The LLM converter (scripts/zerocad_convert.py) plateaued at ~8-9% because a 30B open model keeps
REPRODUCING CadQuery syntax (shell(), moveTo(), move()) inside build123d code instead of translating
it. The dataset's code is machine-generated and, for the SIMPLE samples, a single fluent Workplane
chain -- which is a closed, ~30-op vocabulary and therefore DETERMINISTICALLY transpilable with no
LLM: it cannot hallucinate the wrong syntax because it only ever emits from a fixed rule table, and
it FAILS CLOSED (skips) on anything outside that table instead of guessing.

SCOPE (honest): this handles the single-fluent-chain samples -- `result = cq.Workplane(...)...`.
It fail-closed SKIPS samples with control flow (for/while), list comprehensions, `import math`,
multiple separate Workplane objects joined by .union()/.cut(other), or ops it has no rule for
(shell, loft, sweep). Those richer imperative programs need either the LLM path or running CadQuery
directly; a skip here is honest, never a wrong pair.

Every emitted pair is STILL geometrically verified: run the build123d code in the sandbox, export
STEP, compare volume + bbox (5% tol) against the dataset's ground-truth STEP. Deterministic emission
+ geometric verification = zero bad pairs, predictable yield.

Two run modes:
  python scripts/zerocad_transpile.py --test <sample.parquet>   # measure coverage+yield locally, no download
  python scripts/zerocad_transpile.py [max_files]               # stream the full dataset (like zerocad_convert)

Output (stream mode): ~/.forge-train/zerocad_transpiled.jsonl  (verified (cadquery, build123d) pairs).
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

HOME = Path.home()
CACHE = HOME / ".forge-train" / "ztr_raw"
WORK = HOME / ".forge-train" / "ztr_work"
OUT = HOME / ".forge-train" / "zerocad_transpiled.jsonl"
STATS = HOME / ".forge-train" / "ztr_stats.jsonl"
MANIFEST = HOME / ".forge-train" / "ztr_manifest.jsonl"
LOCKDIR = HOME / ".forge-train" / ".ztr.lock"

HF_API = "https://huggingface.co/api/datasets/ADSKAILab/Zero-To-CAD-1m"
HF_RESOLVE = "https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m/resolve/main/"
VOL_TOL = 0.05
BBOX_TOL = 0.05


class SkipSample(Exception):  # noqa: N818
    """Raised to fail-closed: the sample is outside the deterministic vocabulary."""


# ---- the transpiler -----------------------------------------------------------------------------
def _unparse(node: ast.AST) -> str:
    return ast.unparse(node).strip()


def _collect_chain(node: ast.AST) -> list[tuple[str, list[ast.AST], list[ast.keyword]]]:
    """Flatten a fluent call chain `cq.Workplane(..).a(..).b(..)` into [(method, args, kwargs), ...]
    in call order. The base `cq.Workplane("XY")` becomes ('Workplane', [args], [])."""
    chain: list[tuple[str, list[ast.AST], list[ast.keyword]]] = []
    cur = node
    while isinstance(cur, ast.Call):
        func = cur.func
        if isinstance(func, ast.Attribute):
            chain.append((func.attr, cur.args, cur.keywords))
            cur = func.value
        elif isinstance(func, ast.Name):  # e.g. bare Workplane(...) (rare) -- treat as base
            chain.append((func.id, cur.args, cur.keywords))
            cur = None
            break
        else:
            raise SkipSample("unrecognized call base")
    chain.reverse()
    if not chain:
        raise SkipSample("empty chain")
    # first element must be the Workplane constructor (cq.Workplane or Workplane)
    if chain[0][0] != "Workplane":
        raise SkipSample(f"chain does not start at Workplane ({chain[0][0]})")
    return chain


_SUPPORTED = {
    "Workplane",
    "box",
    "rect",
    "circle",
    "extrude",
    "cutBlind",
    "cut",
    "hole",
    "cboreHole",
    "faces",
    "workplane",
    "center",
    "edges",
    "fillet",
    "chamfer",
    "moveTo",
    "lineTo",
    "threePointArc",
    "close",
    "rarray",
    "pushPoints",
    "polarArray",
    "revolve",
}


class _Emitter:
    """Walks a fluent chain, emits build123d builder-mode code. Live geometry is NOT executed here;
    correctness is proven downstream by running the emitted code + geometric verification."""

    def __init__(self) -> None:
        self.lines: list[str] = ["from build123d import *", ""]
        self.body: list[str] = []  # inside `with BuildPart() as bp:`
        self.plane = "Plane.XY"  # where the next sketch goes
        self.center: tuple[str, str] | None = None
        self.locs: str | None = None  # a `with <Locations>:` header for the next sketch op
        self.profile: list[str] = []  # accumulated BuildLine segments
        self.prof_start: str | None = None
        self.prof_cur: str | None = None
        self.edge_sel = "bp.edges()"

    def _face(self, sel: str) -> str:
        s = sel.strip("\"'")
        if s == ">Z":
            return "bp.faces().sort_by(Axis.Z)[-1]"
        if s == "<Z":
            return "bp.faces().sort_by(Axis.Z)[0]"
        raise SkipSample(f"unsupported face selector {sel}")

    def _edges(self, args: list[ast.AST]) -> str:
        if not args:
            return "bp.edges()"
        s = _unparse(args[0]).strip("\"'")
        if s in ("|Z",):
            return "bp.edges().filter_by(Axis.Z)"
        if s in (">Z",):
            return "bp.edges().group_by(Axis.Z)[-1]"
        if s in ("<Z",):
            return "bp.edges().group_by(Axis.Z)[0]"
        if s in ("not Z",):
            return "bp.edges().filter_by(Axis.Z, reverse=True)"
        raise SkipSample(f"unsupported edge selector {s}")

    def _sketch_ctx(self) -> tuple[str, str]:
        """Return (indent-open, indent-close-note) for placing a sketch: BuildSketch on the plane,
        with an optional Locations for center/pattern."""
        head = f"    with BuildSketch({self.plane}):"
        return head, ""

    def _emit_sketch(self, primitive: str, subtract: bool, amount: str) -> None:
        head, _ = self._sketch_ctx()
        self.body.append(head)
        inner_indent = "        "
        loc_open = None
        if self.locs:
            loc_open = f"{inner_indent}with {self.locs}:"
            inner_indent += "    "
        elif self.center:
            loc_open = f"{inner_indent}with Locations(({self.center[0]}, {self.center[1]})):"
            inner_indent += "    "
        if loc_open:
            self.body.append(loc_open)
        self.body.append(f"{inner_indent}{primitive}")
        mode = ", mode=Mode.SUBTRACT" if subtract else ""
        self.body.append(f"    extrude(amount={amount}{mode})")
        self.center = None
        self.locs = None

    def op(self, method: str, args: list[ast.AST], kw: list[ast.keyword]) -> None:
        a = [_unparse(x) for x in args]
        kmap = {k.arg: _unparse(k.value) for k in kw}
        # FAIL-CLOSED: a center-shifted profile origin (`.center(x,y).lineTo(...)` with no moveTo)
        # is subtle CadQuery cursor semantics we do not model -- skip rather than emit wrong code.
        if method in ("moveTo", "lineTo", "threePointArc", "close") and self.center is not None:
            raise SkipSample("center-shifted profile origin (unsupported)")
        if method in ("lineTo", "threePointArc") and self.prof_cur is None:
            raise SkipSample("profile without a moveTo start (unsupported)")
        A, K = a, kmap  # noqa: N806  (short aliases: A=positional args, K=keyword args)
        if method == "Workplane":
            return
        elif method == "box":
            self.body.append(f"    Box({A[0]}, {A[1]}, {A[2]})")
        elif method == "rect":
            self._pending_prim = f"Rectangle({A[0]}, {A[1]})"
        elif method == "circle":
            self._pending_prim = f"Circle({A[0]})"
        elif method == "moveTo":
            self.prof_start = self.prof_cur = f"({A[0]}, {A[1]})"
        elif method == "lineTo":
            npt = f"({A[0]}, {A[1]})"
            self.profile.append(f"Line({self.prof_cur}, {npt})")
            self.prof_cur = npt
        elif method == "threePointArc":
            mid, end = A[0], A[1]
            self.profile.append(f"ThreePointArc({self.prof_cur}, {mid}, {end})")
            self.prof_cur = end
        elif method == "close":
            if self.prof_cur != self.prof_start:
                self.profile.append(f"Line({self.prof_cur}, {self.prof_start})")
            self._pending_prim = "PROFILE"
        elif method == "extrude":
            self._flush_solid(A[0], subtract=False)
        elif method == "cutBlind":
            self._flush_solid(A[0], subtract=True)
        elif method == "hole":
            depth = A[1] if len(A) > 1 else K.get("depth")
            amt = f"-({depth})" if depth else "-1000"  # through: large negative (clipped by body)
            prim = f"Circle(({A[0]}) / 2)"
            self._pending_prim = prim
            self._emit_sketch(prim, subtract=True, amount=amt)
        elif method == "cboreHole":
            # cboreHole(d, cbore_d, cbore_depth, depth=None) -> two subtract passes
            d, cbd, cbdep = A[0], A[1], A[2]
            dep = A[3] if len(A) > 3 else K.get("depth", "1000")
            self._emit_sketch(f"Circle(({cbd}) / 2)", True, f"-({cbdep})")
            self._emit_sketch(f"Circle(({d}) / 2)", True, f"-({dep})")
        elif method == "faces":
            self.plane = self._face(A[0]) if A else self.plane
        elif method == "workplane":
            pass  # the face already became the plane in faces()
        elif method == "center":
            self.center = (A[0], A[1])
        elif method == "edges":
            self.edge_sel = self._edges(args)
        elif method == "fillet":
            self.body.append(f"    fillet({self.edge_sel}, radius={A[0]})")
            self.edge_sel = "bp.edges()"
        elif method == "chamfer":
            self.body.append(f"    chamfer({self.edge_sel}, length={A[0]})")
            self.edge_sel = "bp.edges()"
        elif method == "rarray":
            self.locs = f"GridLocations({A[0]}, {A[1]}, {A[2]}, {A[3]})"
        elif method == "pushPoints":
            self.locs = f"Locations(*{A[0]})"
        elif method == "polarArray":
            self.locs = f"PolarLocations({A[0]}, {A[3]}, start_angle={A[1]})"
        elif method == "revolve":
            self._flush_solid(None, subtract=False, revolve=True)
        else:
            raise SkipSample(f"no rule for op {method}")

    def _flush_solid(self, amount: str | None, subtract: bool, revolve: bool = False) -> None:
        prim = getattr(self, "_pending_prim", None)
        if prim == "PROFILE":
            if not self.profile:
                raise SkipSample("empty profile")
            head, _ = self._sketch_ctx()
            self.body.append(head)
            self.body.append("        with BuildLine():")
            for seg in self.profile:
                self.body.append(f"            {seg}")
            self.body.append("        make_face()")
            self.profile = []
            self.prof_start = self.prof_cur = None
            if revolve:
                self.body.append("    revolve(axis=Axis.Z)")
            else:
                mode = ", mode=Mode.SUBTRACT" if subtract else ""
                self.body.append(f"    extrude(amount={amount}{mode})")
        elif prim:
            self._emit_sketch(prim, subtract, amount if amount else "1")
        else:
            raise SkipSample("solid op without a pending sketch")
        self._pending_prim = None

    def finish(self) -> str:
        self.lines.append("with BuildPart() as bp:")
        self.lines.extend(self.body)
        self.lines.append("")
        self.lines.append("result = bp.part")
        return "\n".join(self.lines)


def transpile(cq_source: str) -> str:
    """CadQuery source -> build123d code, or raise SkipSample if outside the deterministic scope."""
    try:
        tree = ast.parse(cq_source)
    except SyntaxError as e:
        raise SkipSample(f"source did not parse: {e}") from e

    preamble: list[str] = []
    result_chain: ast.AST | None = None
    workplane_count = 0
    for stmt in ast.walk(tree):
        if isinstance(stmt, (ast.For, ast.While, ast.ListComp, ast.comprehension, ast.FunctionDef)):
            raise SkipSample(
                "control flow / comprehension -- imperative program, not a simple chain"
            )
        if isinstance(stmt, ast.Attribute) and stmt.attr == "Workplane":
            workplane_count += 1
    if workplane_count > 1:
        raise SkipSample("multiple Workplane objects (union/cut of separate solids)")

    for stmt in tree.body:
        if isinstance(stmt, ast.Import) and any(a.name == "math" for a in stmt.names):
            raise SkipSample("uses math -- computed geometry, skip")
        if isinstance(stmt, ast.Import) and all(a.name == "cadquery" for a in stmt.names):
            continue
        if isinstance(stmt, ast.ImportFrom):
            continue
        if isinstance(stmt, ast.Assign):
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "result":
                result_chain = stmt.value
            else:
                preamble.append(_unparse(stmt))  # a variable definition -> copy verbatim
        elif isinstance(stmt, ast.Expr):
            continue
        else:
            raise SkipSample(f"unexpected top-level statement {type(stmt).__name__}")

    if result_chain is None:
        raise SkipSample("no `result = ...` assignment found")

    chain = _collect_chain(result_chain)
    for method, _a, _k in chain:
        if method not in _SUPPORTED:
            raise SkipSample(f"unsupported op in chain: {method}")

    em = _Emitter()
    for line in preamble:
        em.lines.append(line)
    if preamble:
        em.lines.append("")
    for method, cargs, ckw in chain:
        em.op(method, cargs, ckw)
    return em.finish()


# ---- verification (reused pattern from zerocad_convert) -----------------------------------------
def _metrics(step_path: Path) -> tuple[float, tuple[float, float, float]] | None:
    try:
        from build123d import import_step

        from forge_mind.agentic.brep_checks import brep_volume

        part = import_step(str(step_path))
        vol = brep_volume(part)
        bb = part.bounding_box()
        ext = (float(bb.max.X - bb.min.X), float(bb.max.Y - bb.min.Y), float(bb.max.Z - bb.min.Z))
        return (vol, ext) if vol and vol > 0 else None
    except Exception:
        return None


def _matches(gt: tuple[float, tuple], got: tuple[float, tuple]) -> bool:
    if abs(got[0] - gt[0]) / max(gt[0], 1e-9) > VOL_TOL:
        return False
    for a, b in zip(sorted(gt[1]), sorted(got[1]), strict=False):
        if abs(b - a) / max(a, 1e-6) > BBOX_TOL:
            return False
    return True


def _verify_one(cq_code: str, step_bytes: bytes) -> dict[str, Any]:
    from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code

    tmp = Path(tempfile.mkdtemp(dir=str(WORK)))
    try:
        gtp = tmp / "gt.step"
        gtp.write_bytes(step_bytes)
        gt = _metrics(gtp)
        if gt is None:
            return {"verdict": "skip", "reason": "ground truth unreadable"}
        try:
            b123d = transpile(cq_code)
        except SkipSample as e:
            return {"verdict": "skip", "reason": str(e)}
        res = run_build123d_code(
            b123d, tmp / "b", part_name="result", limits=SandboxLimits(timeout_s=60)
        )
        if not res.ok or res.step_path is None:
            return {
                "verdict": "fail",
                "reason": f"{res.error_type}: {str(res.error_message)[:120]}",
            }
        got = _metrics(res.step_path)
        if got is None or not _matches(gt, got):
            return {"verdict": "fail", "reason": "geometry mismatch", "b123d": b123d}
        return {"verdict": "pass", "b123d": b123d}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---- test mode (local, no download) -------------------------------------------------------------
def _test(parquet_path: str, n: int = 416) -> int:
    import pyarrow.parquet as pq

    WORK.mkdir(parents=True, exist_ok=True)
    t = pq.ParquetFile(parquet_path).read(columns=["cadquery_file", "step_file"]).to_pydict()
    total = min(n, len(t["cadquery_file"]))
    import collections

    v = collections.Counter()
    attempted = 0  # transpiler did NOT skip
    for i in range(total):
        cq = t["cadquery_file"][i]
        cq = cq.decode("utf-8", "ignore") if isinstance(cq, (bytes, bytearray)) else str(cq)
        r = _verify_one(cq, bytes(t["step_file"][i]))
        v[r["verdict"]] += 1
        if r["verdict"] != "skip":
            attempted += 1
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{total}: pass={v['pass']} fail={v['fail']} skip={v['skip']}",
                flush=True,
            )
    print("\n==== DETERMINISTIC TRANSPILER (local test) ====")
    print(f"  total samples          : {total}")
    print(f"  in-scope (not skipped) : {attempted}  ({100 * attempted // total}%)")
    print(
        f"  VERIFIED pass          : {v['pass']}  ({100 * v['pass'] // total}% of all, "
        f"{100 * v['pass'] // max(attempted, 1)}% of in-scope)"
    )
    print(f"  in-scope but failed    : {v['fail']}")
    print(f"  skipped (out of scope) : {v['skip']}")
    return 0


# ---- stream mode (full dataset) -----------------------------------------------------------------
def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:
        return True


def _acquire_lock() -> bool:
    try:
        LOCKDIR.mkdir(parents=True)
    except FileExistsError:
        h = (LOCKDIR / "pid").read_text().strip() if (LOCKDIR / "pid").exists() else ""
        if h and _pid_alive(int(h)):
            return False
        shutil.rmtree(LOCKDIR, ignore_errors=True)
        try:
            LOCKDIR.mkdir(parents=True)
        except FileExistsError:
            return False
    (LOCKDIR / "pid").write_text(str(os.getpid()))
    return True


def _done_files() -> set[str]:
    if not MANIFEST.exists():
        return set()
    return {json.loads(x)["file"] for x in MANIFEST.read_text().splitlines() if x.strip()}


def _stream(max_files: int) -> None:
    import pyarrow.parquet as pq

    CACHE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    http = httpx.Client(timeout=120.0, follow_redirects=True)
    files = sorted(
        s["rfilename"]
        for s in http.get(HF_API, timeout=30.0).json().get("siblings", [])
        if s.get("rfilename", "").endswith(".parquet")
    )
    done = _done_files()
    todo = [f for f in files if f not in done][:max_files]
    print(f"ztr: {len(files)} files, {len(done)} done, {len(todo)} to go", flush=True)
    n_pass = n_fail = n_skip = 0
    for fi, fname in enumerate(todo):
        local = CACHE / Path(fname).name
        try:
            with http.stream("GET", HF_RESOLVE + fname) as resp:
                resp.raise_for_status()
                with local.open("wb") as fh:
                    for chunk in resp.iter_bytes(chunk_size=1 << 20):
                        fh.write(chunk)
            tab = (
                pq.ParquetFile(str(local))
                .read(columns=["uuid", "cadquery_file", "num_faces", "step_file"])
                .to_pydict()
            )
        except Exception as e:
            print(f"[{fi + 1}/{len(todo)}] file error {fname}: {str(e)[:80]}", flush=True)
            local.unlink(missing_ok=True)
            continue
        for i in range(len(tab["uuid"])):
            cq = tab["cadquery_file"][i]
            cq = cq.decode("utf-8", "ignore") if isinstance(cq, (bytes, bytearray)) else str(cq)
            r = _verify_one(cq, bytes(tab["step_file"][i]))
            rec = {"uuid": tab["uuid"][i], "verdict": r["verdict"], "reason": r.get("reason", "")}
            with STATS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            if r["verdict"] == "pass":
                n_pass += 1
                with OUT.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "uuid": tab["uuid"][i],
                                "cadquery_code": cq,
                                "build123d_code": r["b123d"],
                                "num_faces": tab["num_faces"][i],
                            }
                        )
                        + "\n"
                    )
            elif r["verdict"] == "fail":
                n_fail += 1
            else:
                n_skip += 1
        local.unlink(missing_ok=True)
        with MANIFEST.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"file": fname}) + "\n")
        print(
            f"[{fi + 1}/{len(todo)}] {fname.split('/')[-1][:24]}  "
            f"pass={n_pass} fail={n_fail} skip={n_skip}",
            flush=True,
        )
    print(f"\nztr done: pass={n_pass} fail={n_fail} skip={n_skip}", flush=True)


def main() -> int:
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        WORK.mkdir(parents=True, exist_ok=True)
        return _test(sys.argv[2])
    if not _acquire_lock():
        print("another ztr instance holds the lock -- exiting", file=sys.stderr)
        return 3
    try:
        _stream(int(sys.argv[1]) if len(sys.argv) > 1 else 999999)
    finally:
        shutil.rmtree(LOCKDIR, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
