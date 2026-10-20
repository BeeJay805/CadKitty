"""Sandbox CHILD harness: execute model-authored build123d code, export STL + STEP.

Runs only as a subprocess (``python -m forge_mind.agentic._sandbox_runner <job.json>``);
it is never imported into the parent. It reads a job spec, execs the author's code
with a restricted import set, exports the bound ``result`` part, and writes a
structured ``result.json``. EVERY failure becomes a structured error in that file,
never an unreadable crash -- the parent reads the JSON to build a SandboxResult.

The geometry kernel (build123d / OCP / numpy) is imported BEFORE the import guard
is installed, so denying network / subprocess modules to the author's code does not
break the kernel's own (broad, legitimate) transitive imports.
"""

from __future__ import annotations

import builtins
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Top-level modules the sandboxed author code may NOT import: network, process-spawning, and
# the OS / FFI / import machinery that would let it shell out, spawn, or read the environment.
# The kernel is pre-imported (before the guard installs), so build123d/OCP do not need any of
# these, and no author idiom in the cheatsheet imports them (verified: 0 hits).
#
# NOT an airtight jail. It contains BUGGY, non-adversarial code (the author is Opus) and raises
# the bar against the obvious escapes. Determined Python can still reach a module via an allowed
# object's __globals__, and builtins.open() is unrestricted -- true isolation needs a
# container/seccomp (see SECURITY.md). The child env is whitelisted (sandbox.py) so even a
# reached os.environ holds no secrets.
_DENIED_IMPORTS = frozenset(
    {
        # network
        "socket",
        "ssl",
        "http",
        "urllib",
        "urllib2",
        "urllib3",
        "ftplib",
        "smtplib",
        "telnetlib",
        "poplib",
        "imaplib",
        "nntplib",
        "requests",
        "httpx",
        "aiohttp",
        "xmlrpc",
        # process spawning / shell / FFI / import machinery / filesystem tooling
        "subprocess",
        "multiprocessing",
        "os",
        "sys",
        "ctypes",
        "importlib",
        "shutil",
        "tempfile",
        "webbrowser",
    }
)


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


_GUARD_INSTALLED = False


def _install_import_guard() -> None:
    """Block the denylisted top-level modules for any import the author code makes.
    Idempotent: loop mode (warm workers) processes many jobs in one process, and re-wrapping
    ``__import__`` per job would grow an unbounded delegation chain."""
    global _GUARD_INSTALLED
    if _GUARD_INSTALLED:
        return
    real_import = builtins.__import__

    def guarded(name: str, *args: Any, **kwargs: Any) -> Any:
        top = name.split(".", 1)[0]
        if top in _DENIED_IMPORTS:
            raise ImportError(f"import of {top!r} is blocked in the Forge-Mind sandbox")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded
    _GUARD_INSTALLED = True


def _weld_stl_seams(stl_path: Path) -> None:
    """Weld exact-duplicate vertices in the exported STL (bench baseline row 17: both no_build
    parts were single-body, correct-volume solids whose STLs read non-watertight ONLY because
    OCC triangulates each B-rep face independently and leaves coincident seam vertices
    unmerged -- 8 iterations each failed the coherence gate on an export artifact).

    Honesty bound: merging EXACT duplicates is lossless and cannot bridge a real crack (real
    gaps have distinct coordinates), so this fixes the ARTIFACT, never the verdict -- a part
    that is genuinely open stays non-watertight. Rewrites the file only when welding flips it
    to watertight; fails soft (any error keeps the original export)."""
    try:
        import trimesh

        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        if bool(mesh.is_watertight):
            return
        mesh.merge_vertices()
        mesh.process(validate=True)
        if bool(mesh.is_watertight):
            mesh.export(str(stl_path))
    except Exception:
        return


def _find_part(ns: dict[str, Any]) -> Any:
    """Locate the authored solid. Convention: bind it to a top-level ``result``."""
    for key in ("result", "part", "PART", "model"):
        value = ns.get(key)
        if value is not None:
            return value
    raise NameError("no part found: bind the final solid to a top-level variable named `result`")


# ---- per-op measurement transcript (ADR-0119, FORGE_OP_TRACE) ---------------------------


def _instrument_for_optrace(code: str) -> str | None:
    """Insert ``__fm_op_probe(<lineno>, <builder>)`` after every statement in each
    ``with BuildPart() as <builder>:`` body, so the trace records what each op DID.
    Statement granularity on purpose: a nested Locations block is one op to its builder.
    Returns the instrumented source, or None when the code cannot be transformed
    (fail-soft: the caller runs the original, untraced)."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    class _Probe(ast.NodeTransformer):
        def visit_With(self, node: ast.With) -> ast.With:
            self.generic_visit(node)
            builder = None
            for item in node.items:
                ctx = item.context_expr
                name = ""
                if isinstance(ctx, ast.Call):
                    name = (
                        ctx.func.id
                        if isinstance(ctx.func, ast.Name)
                        else getattr(ctx.func, "attr", "")
                    )
                if name == "BuildPart" and isinstance(item.optional_vars, ast.Name):
                    builder = item.optional_vars.id
            if builder is None:
                return node
            new_body: list[ast.stmt] = []
            for stmt in node.body:
                new_body.append(stmt)
                new_body.append(
                    ast.Expr(
                        ast.Call(
                            func=ast.Name("__fm_op_probe", ast.Load()),
                            args=[
                                ast.Constant(getattr(stmt, "lineno", 0)),
                                ast.Name(builder, ast.Load()),
                            ],
                            keywords=[],
                        )
                    )
                )
            node.body = new_body
            return node

    try:
        tree = _Probe().visit(tree)
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)
    except Exception:
        return None


def _make_op_probe(trace: list[dict[str, Any]]) -> Any:
    """The probe the instrumented code calls. Best-effort per entry; a probe failure
    records the error name and never disturbs the build. Capped so a pathological
    script cannot balloon result.json."""

    def __fm_op_probe(line: int, builder: Any) -> None:
        if len(trace) >= 64:
            return
        entry: dict[str, Any] = {"line": int(line)}
        try:
            p = getattr(builder, "part", None)
            if p is not None:
                entry["volume"] = round(float(p.volume), 1)
                entry["solids"] = len(p.solids())
                bb = p.bounding_box()
                entry["extents"] = [
                    round(float(bb.max.X - bb.min.X), 1),
                    round(float(bb.max.Y - bb.min.Y), 1),
                    round(float(bb.max.Z - bb.min.Z), 1),
                ]
        except Exception as exc:
            entry["err"] = type(exc).__name__
        trace.append(entry)

    return __fm_op_probe


def _run_selfcheck(
    part: Any,
    extents: list[float] | None,
    spec: dict[str, Any],
    stl_path: Any = None,
) -> dict[str, Any]:
    """The self-check epilogue (FORGE_AUTHOR_SELFCHECK): measure the JUST-built part against
    the brief's own targets -- bbox extents vs the envelope, measured concave bores vs the
    planned hole groups -- and return the structured deltas for ``result.json``. FEEDBACK
    ONLY (the validator still gates); best-effort (a failure returns an ``error`` note,
    never fails the build)."""
    out: dict[str, Any] = {}
    try:
        target = spec.get("envelope")
        if target and extents:
            built = sorted(float(e) for e in extents)
            want = sorted(float(t) for t in target)
            out["envelope"] = {
                "built_sorted": [round(b, 2) for b in built],
                "target_sorted": [round(w, 2) for w in want],
                "delta_mm": [round(b - w, 2) for b, w in zip(built, want, strict=False)],
                "delta_pct": [
                    round(100.0 * (b - w) / w, 1) if w > 0 else None
                    for b, w in zip(built, want, strict=False)
                ],
            }
        hole_specs = spec.get("holes") or []
        if hole_specs:
            # validator.geometry was pre-imported BEFORE the import guard (harness code).
            from forge_mind.validator.geometry import cluster_coaxial, extract_cylinders

            stacks = [s for s in cluster_coaxial(extract_cylinders(part)) if s.concave is True]

            # A stack counts for its SEGMENTS: two clevis ears on one pivot axis are one
            # coaxial stack but TWO physical bores (mirrors spatial.measured_hole_feedback).
            def _seg(s: Any) -> int:
                return max(1, int(getattr(s, "segments", 1)))

            out["bores"] = [
                {
                    "d_mm": round(2.0 * s.radius, 2),
                    "at": [round(float(c), 1) for c in s.axis_point],
                    "segments": _seg(s),
                }
                for s in stacks[:16]
            ]
            tol = 0.6  # mirrors spatial._DIA_TOL_MM: a bore within this of target counts
            remaining = list(stacks)
            holes_out: list[dict[str, Any]] = []
            for h in hole_specs:
                want_n = int(h.get("count") or 1)
                d = h.get("d_mm")
                if isinstance(d, (int, float)):
                    matched = [s for s in remaining if abs(2.0 * s.radius - float(d)) <= tol]
                else:
                    matched = list(remaining)
                got = 0
                for s in matched:
                    if got >= want_n:
                        break
                    remaining.remove(s)
                    got += _seg(s)
                holes_out.append(
                    {
                        "label": str(h.get("label") or "holes"),
                        "want": want_n,
                        "d_mm": float(d) if isinstance(d, (int, float)) else None,
                        "matched": min(got, want_n),
                    }
                )
            out["holes"] = holes_out
    except Exception as exc:  # the epilogue must NEVER fail the build
        out["error"] = f"{type(exc).__name__}: {exc}"
    # WALL floor (2026-07-27, the L102 thin-wall class): measure with the validator GATE'S OWN
    # function, against the spec's EFFECTIVE floor, on the just-exported STL -- so the author's
    # own number is the number the gate will read. feature_checks + trimesh were pre-imported
    # before the guard (harness code). Its OWN try block, OUTSIDE the one above (review-caught):
    # sharing it meant a B-rep hole-traversal failure jumped past the wall measurement and
    # silently dropped this feedback, even though the STL was perfectly measurable.
    try:
        floor = spec.get("wall_floor_mm")
        if isinstance(floor, (int, float)) and float(floor) > 0 and stl_path is not None:
            from forge_mind.validator.feature_checks import measure_wall_fraction

            mesh = sys.modules["trimesh"].load(str(stl_path), force="mesh")
            measured = measure_wall_fraction(mesh, float(floor))
            if measured is not None:
                out["wall"] = {
                    "floor_mm": float(floor),
                    "min_area_mm2": float(spec.get("wall_min_area_mm2") or float(floor) ** 2),
                    # RAW, unrounded (review-caught): the consumer compares this against the
                    # gate's area threshold, and rounding 0.029991 -> 0.0300 would claim a
                    # failure the gate accepts. Rounding happens at DISPLAY time only.
                    "thin_fraction": float(measured.thin_fraction),
                    "thinnest_mm": (
                        round(measured.thinnest_mm, 2) if measured.thinnest_mm is not None else None
                    ),
                    "thin_area_mm2": measured.thin_area_mm2,
                    "thin_sample_count": measured.thin_sample_count,
                    "valid_sample_count": measured.valid_sample_count,
                }
    except Exception as exc:
        out["wall_error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    if sys.argv[1] == "--loop":
        return _loop()
    return _process_job(Path(sys.argv[1]))


def _loop() -> int:
    """Warm-worker mode (engine research idea 1): process one job per stdin LINE in this single
    process, so evaluation #2..#N of the parametric fit skips the ~7 s kernel cold-import tax
    (measured 2026-07-03). Used ONLY for re-running code that already passed the one-shot
    sandbox once -- crash containment is preserved (the parent kills + respawns this worker on
    any timeout/crash); only the per-eval import price is dropped. Each job still executes in a
    FRESH namespace; the parent's leakage canary re-runs baseline params and requires identical
    geometry."""
    print("READY", flush=True)
    for line in sys.stdin:
        job = line.strip()
        if not job:
            continue
        if job == "EXIT":
            break
        try:
            _process_job(Path(job))
        except Exception:
            # _process_job writes structured errors itself; anything escaping is a harness bug --
            # report and keep the worker alive (the parent treats a missing result.json as harness).
            traceback.print_exc(file=sys.stderr)
        print(f"DONE {job}", flush=True)
    return 0


def _plumbing_note(ns: dict[str, Any], part: Any) -> str | None:
    """Detect the shipped-the-wrong-solid class (measured 2026-07-18: bottle_cap built a
    genuine threaded cap and bound ``result`` to its 0.5 mm printing base plate). Scan the
    top-level namespace for OTHER build123d solids whose volume dwarfs the exported one and
    say so. ADVISORY string for the revision feedback -- never gates, fail-soft."""
    try:
        pv = float(part.volume)
    except Exception:
        return None
    best_name, best_vol = None, 0.0
    for name, val in ns.items():
        if name.startswith("_") or val is part:
            continue
        cand = getattr(val, "part", val)  # a BuildPart context exposes .part
        if cand is part:
            continue
        try:
            v = float(cand.volume)
        except Exception:
            continue
        if v > best_vol:
            best_name, best_vol = name, v
    if best_name is not None and pv < 0.5 * best_vol:
        return (
            f"RESULT PLUMBING: `result` is bound to a solid of {pv:.0f} mm^3, but your script "
            f"also built `{best_name}` at {best_vol:.0f} mm^3 -- you exported the SMALLER piece. "
            "If the larger body is the actual part, fuse everything that belongs to it and bind "
            "THAT to `result`."
        )
    return None


def _process_job(job_path: Path) -> int:
    job: dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["out_dir"])
    part_name = str(job["part_name"])
    code = str(job["code"])
    result_path = out_dir / "result.json"

    def _write(payload: dict[str, Any]) -> None:
        # Atomic: write a sibling temp file then os.replace it onto result.json. The parent polls
        # for result.json after the child exits; a non-atomic write could leave the parent reading a
        # half-written (truncated / invalid-JSON) file if it raced the child. os.replace is atomic on
        # the same filesystem (POSIX rename + Windows MoveFileEx), so the parent only ever sees the
        # old absence or the complete new file -- never a partial one.
        tmp_path = result_path.with_name(result_path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, result_path)

    # 1. Pre-import the kernel (no guard yet). A failure here is a HARNESS problem,
    #    not the author's code.
    try:
        import build123d  # noqa: F401
        from build123d import export_step, export_stl

        b123d_version = _pkg_version("build123d")
        ocp_version = _pkg_version("OCP") or _pkg_version("cadquery-ocp")
    except Exception as exc:
        _write(
            {
                "ok": False,
                "error_type": "kernel_import",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 0

    # 2. Seed the namespace with the kernel + (optional) helper lib, THEN lock imports.
    ns: dict[str, Any] = {"__name__": "__sandbox__"}
    with contextlib.suppress(Exception):
        exec("from build123d import *", ns)
    with contextlib.suppress(Exception):
        exec("from forge_mind.agentic.helpers import *", ns)  # may not exist yet
    with contextlib.suppress(Exception):
        exec(
            "from forge_mind.agentic.gears import spur_gear, ring_gear, toothed_rim", ns
        )  # deterministic teeth
    with contextlib.suppress(Exception):
        exec(
            "from forge_mind.agentic.gears import center_distance, mesh_clock_deg, "
            "planet_carrier_radius, planet_clock_deg, planetary_gearset",
            ns,
        )  # meshing/clocking + planetary (build123d + pure math; import-safe)
    with contextlib.suppress(Exception):
        exec("from forge_mind.agentic.standards_helpers import *", ns)  # exact ISO/DIN dimensions
    # Pre-import the B-rep measurement module for the SELF-CHECK epilogue (harness code, not
    # author-reachable): it must be loaded BEFORE the import guard locks down, because the
    # epilogue runs after the author's code under the active guard.
    with contextlib.suppress(Exception):
        import forge_mind.validator.geometry
    # The shared WALL measurement the epilogue reports (same guard reason). Its OWN suppress
    # block on purpose: sharing one with the import above would let a geometry failure
    # silently disable the wall self-check too (the silent-darkening class).
    with contextlib.suppress(Exception):
        import forge_mind.validator.feature_checks  # noqa: F401
    # Pre-import trimesh for the seam-weld epilogue (same guard trap as above: trimesh's
    # import chain touches denylisted modules, so importing it lazily under the guard
    # silently no-ops the weld -- caught rebuilding the coat hook, bench row 17).
    with contextlib.suppress(Exception):
        import trimesh  # noqa: F401
    # Pre-import bd_warehouse (standard SOLID screws / bearings / threads) BEFORE the guard:
    # its __init__ pulls importlib.metadata, which the guard DENIES, so the documented
    # "import it on demand" path could never actually succeed under the guard -- the catalog
    # screw/bearing idioms crashed at generation time (exposed by the idiom execution test).
    # Pre-imported, an authored `from bd_warehouse.fastener import ...` resolves from
    # sys.modules without touching the denied import machinery. suppress: a venv without
    # bd_warehouse degrades to the same structured exec error as before.
    with contextlib.suppress(Exception):
        import bd_warehouse.bearing
        import bd_warehouse.fastener
        import bd_warehouse.thread  # noqa: F401
    _install_import_guard()

    # 3. Execute the author's code. With the job's optrace flag (FORGE_OP_TRACE), run the
    # probe-instrumented source instead -- fail-soft: an untransformable script runs
    # untraced, byte-identical to the flag-off path.
    op_trace: list[dict[str, Any]] = []
    # ADR-0133 fix 3: drain stale edge-finish events BEFORE the job runs (warm workers
    # must never leak another part's fillet outcomes). sys.modules lookup dodges the
    # import guard installed above; helpers was star-imported pre-guard.
    with contextlib.suppress(Exception):
        sys.modules["forge_mind.agentic.helpers"].drain_edge_finish_events()
    exec_code = code
    if job.get("optrace"):
        instrumented = _instrument_for_optrace(code)
        if instrumented is not None:
            exec_code = instrumented
            ns["__fm_op_probe"] = _make_op_probe(op_trace)
    try:
        exec(compile(exec_code, "<authored_part>", "exec"), ns)
    except Exception as exc:
        _write(
            {
                "ok": False,
                "error_type": "exec",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
                # the partial trace shows WHERE the build died -- the most valuable case
                "optrace": op_trace or None,
            }
        )
        return 0

    # 4. Find + export the part.
    try:
        part = _find_part(ns)
    except NameError as exc:
        _write(
            {
                "ok": False,
                "error_type": "no_result",
                "error_message": str(exc),
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
            }
        )
        return 0

    stl_path = out_dir / f"{part_name}.stl"
    step_path = out_dir / f"{part_name}.step"
    try:
        # Name the STEP product (bench baseline row 17: five valid parts failed the rubric's
        # named_step check because an unlabeled Part exports as PRODUCT('SOLID')). The label is
        # metadata only -- geometry is untouched.
        with contextlib.suppress(Exception):
            if not getattr(part, "label", ""):
                part.label = part_name
        export_stl(part, str(stl_path))
        _weld_stl_seams(stl_path)
        export_step(part, str(step_path))
    except Exception as exc:
        _write(
            {
                "ok": False,
                "error_type": "export",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
            }
        )
        return 0

    # 5. Bounding box (mirror compiler.compile._bbox_of). Best-effort.
    bbox: list[list[float]] | None = None
    extents: list[float] | None = None
    try:
        bb = part.bounding_box()
        bbox = [
            [float(bb.min.X), float(bb.min.Y), float(bb.min.Z)],
            [float(bb.max.X), float(bb.max.Y), float(bb.max.Z)],
        ]
        extents = [
            float(bb.max.X - bb.min.X),
            float(bb.max.Y - bb.min.Y),
            float(bb.max.Z - bb.min.Z),
        ]
    except Exception:
        pass

    # 6. Self-check epilogue (only when the job carries a spec -- FORGE_AUTHOR_SELFCHECK).
    selfcheck: dict[str, Any] | None = None
    sc_spec = job.get("selfcheck")
    if isinstance(sc_spec, dict):
        selfcheck = _run_selfcheck(part, extents, sc_spec, stl_path)

    plumbing = None
    with contextlib.suppress(Exception):
        plumbing = _plumbing_note(ns, part)

    # ADR-0133 fix 3: the safe_fillet/safe_chamfer outcome log rides result.json ONLY
    # when the job carries the flag -- flag-off artifacts stay byte-identical.
    edge_finish: list[dict[str, Any]] | None = None
    if job.get("edge_finish"):
        with contextlib.suppress(Exception):
            edge_finish = sys.modules["forge_mind.agentic.helpers"].drain_edge_finish_events()

    payload: dict[str, Any] = {
        "ok": True,
        "stl_path": str(stl_path),
        "step_path": str(step_path),
        "bbox": bbox,
        "extents": extents,
        "selfcheck": selfcheck,
        "plumbing_note": plumbing,
        "build123d_version": b123d_version,
        "ocp_version": ocp_version,
        "optrace": op_trace or None,
    }
    if job.get("edge_finish"):
        payload["edge_finish"] = edge_finish or []
    _write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
