"""Sandbox CHILD harness for CadQuery code (ADR-0112: the CAD-Coder bridge).

Runs only as a subprocess (``python -m forge_mind.agentic._cadquery_runner <job.json>``),
never imported into the parent. It is the CadQuery sibling of ``_sandbox_runner`` and is
driven by the SAME parent (``sandbox.run_build123d_code(..., runner_module=...)``), so it
inherits that driver's timeout / memory-watch / native-crash detection unchanged.

The one thing it does differently: CAD-Coder emits *CadQuery* (``import cadquery as cq``),
not build123d. So the child execs the CadQuery, then BRIDGES the bound ``result`` onto the
shared OCP kernel via ``build123d.Solid(shape.wrapped)`` -- verified byte-identical geometry
in commit a54ba2b -- and exports STL + STEP through the SAME build123d exporters the
build123d path uses. Everything downstream (validator, fidelity gate, slicer) then operates
on a plain build123d solid and never knows the source was CadQuery. "Bridge once per part,
at exec."

Isolation caveat is identical to ``_sandbox_runner``: NOT an airtight jail (non-adversarial
author code, env whitelisted by the parent), but network / process-spawn / import machinery
are denied to the authored code.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Reuse the proven helpers from the build123d child (importing the module only DEFINES
# functions -- its ``main`` runs solely under ``__main__``). Keeps the denied-import set,
# the atomic-write discipline, the seam-weld, and the SELF-CHECK epilogue in ONE place.
from forge_mind.agentic._sandbox_runner import (
    _install_import_guard,
    _pkg_version,
    _run_selfcheck,
    _weld_stl_seams,
)


def _to_build123d_solid(obj: Any) -> Any:
    """Bridge a CadQuery result onto the shared OCP kernel as a build123d ``Solid``.

    Accepts what CAD-Coder actually binds to ``result``: a ``Workplane`` (``.val()`` yields
    the shape), a bare CadQuery ``Shape``/``Solid`` (already has ``.wrapped``), or -- as a
    courtesy -- something already build123d-native (has ``.wrapped`` and exports directly).
    Raises a plain ``TypeError`` on anything else so the parent reports a clean ``export``
    error rather than an opaque crash.
    """
    import build123d as bd

    # CadQuery Workplane: the current shape is obj.val() (or the last of obj.vals()).
    val = getattr(obj, "val", None)
    shape = obj.val() if callable(val) else obj
    wrapped = getattr(shape, "wrapped", None)
    if wrapped is None:
        raise TypeError(
            f"result of type {type(obj).__name__!r} is not a CadQuery/OCP shape "
            "(no `.wrapped` OCP handle to bridge)"
        )
    # If it is already a build123d object, keep it; else wrap the raw OCP shape.
    if isinstance(shape, (bd.Solid, bd.Part, bd.Compound)):
        return shape
    return bd.Solid(wrapped)


def _is_cadquery_obj(value: Any) -> bool:
    """A CadQuery Workplane (has ``.val``) or a shape/solid (has ``.wrapped``)."""
    return callable(getattr(value, "val", None)) or getattr(value, "wrapped", None) is not None


def _find_result(ns: dict[str, Any]) -> Any:
    """Locate the explicitly named final CadQuery object.

    Falling back to the last shape can export a subtraction tool or other temporary
    binding as the finished part. Authored code must make ownership explicit with one
    of the documented conventional names.
    """
    for key in ("result", "part", "PART", "model"):
        value = ns.get(key)
        if value is not None:
            return value
    raise NameError("no final CadQuery object named result, part, PART, or model")


def _process_job(job_path: Path) -> int:
    job: dict[str, Any] = json.loads(job_path.read_text(encoding="utf-8"))
    out_dir = Path(job["out_dir"])
    part_name = str(job["part_name"])
    code = str(job["code"])
    result_path = out_dir / "result.json"

    def _write(payload: dict[str, Any]) -> None:
        # Atomic sibling-write + replace (identical rationale to _sandbox_runner): the parent
        # polls for result.json and must never read a half-written file. ``os`` is imported at
        # module top (before the guard installs) -- re-importing it here would be DENIED.
        tmp_path = result_path.with_name(result_path.name + ".tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp_path, result_path)

    # 1. Pre-import both kernels (no guard yet). A failure here is a HARNESS problem.
    try:
        import build123d  # noqa: F401
        import cadquery  # noqa: F401
        from build123d import export_step, export_stl

        cq_version = _pkg_version("cadquery")
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

    # 2. Seed the namespace with CadQuery, THEN lock imports (guard denies os/sys/subprocess/...).
    ns: dict[str, Any] = {"__name__": "__sandbox__"}
    with contextlib.suppress(Exception):
        exec("import cadquery as cq", ns)
    with contextlib.suppress(Exception):
        exec("from cadquery import *", ns)
    # Pre-import the B-rep measurement module for the SELF-CHECK epilogue (harness code, not
    # author-reachable): it must load BEFORE the guard locks down (same trap as _sandbox_runner).
    with contextlib.suppress(Exception):
        import forge_mind.validator.geometry
    # The shared WALL measurement the epilogue reports (its OWN suppress: sharing one with the
    # import above would let a geometry failure silently disable the wall self-check too).
    with contextlib.suppress(Exception):
        import forge_mind.validator.feature_checks  # noqa: F401
    with contextlib.suppress(Exception):
        import trimesh  # noqa: F401
    _install_import_guard()

    # 3. Execute the CAD-Coder-authored CadQuery.
    try:
        exec(compile(code, "<cadcoder_part>", "exec"), ns)
    except Exception as exc:
        _write(
            {
                "ok": False,
                "error_type": "exec",
                "error_message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "cadquery_version": cq_version,
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
            }
        )
        return 0

    # 4. Find the bound result and bridge it to a build123d solid.
    try:
        part = _to_build123d_solid(_find_result(ns))
    except (NameError, TypeError) as exc:
        _write(
            {
                "ok": False,
                "error_type": "no_result",
                "error_message": str(exc),
                "cadquery_version": cq_version,
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
            }
        )
        return 0

    # 5. Export through the build123d exporters (the bridged solid is native from here).
    stl_path = out_dir / f"{part_name}.stl"
    step_path = out_dir / f"{part_name}.step"
    try:
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
                "cadquery_version": cq_version,
                "build123d_version": b123d_version,
                "ocp_version": ocp_version,
            }
        )
        return 0

    # 6. Bounding box (best-effort, mirrors _sandbox_runner).
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

    # Self-check epilogue (when the job carries a spec): measure the JUST-built part against the
    # brief's targets -- envelope + concave bores + the WALL floor (the exported STL is passed so
    # the wall half runs on this lane too; review-caught, it was silently envelope-only here).
    # The bridged ``part`` is a build123d solid, so the shared ``_run_selfcheck`` works unchanged.
    # Feedback only; the validator still gates.
    selfcheck: dict[str, Any] | None = None
    sc_spec = job.get("selfcheck")
    if isinstance(sc_spec, dict):
        selfcheck = _run_selfcheck(part, extents, sc_spec, stl_path)

    _write(
        {
            "ok": True,
            "stl_path": str(stl_path),
            "step_path": str(step_path),
            "bbox": bbox,
            "extents": extents,
            "selfcheck": selfcheck,
            "cadquery_version": cq_version,
            "build123d_version": b123d_version,
            "ocp_version": ocp_version,
        }
    )
    return 0


def main() -> int:
    return _process_job(Path(sys.argv[1]))


if __name__ == "__main__":
    sys.exit(main())
