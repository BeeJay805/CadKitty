"""Parent-side CadQuery bridge (ADR-0112): run CAD-Coder's CadQuery, hand the result to
the EXISTING validator as a plain build123d solid.

Two functions, both reusable by the eventual generator backend (not just the smoke bench):

* ``run_cadquery_code`` -- exec CadQuery in the isolated child (``_cadquery_runner``) via the
  shared sandbox driver, producing STL + STEP on disk. Never raises on authored-code failure.
* ``load_validation_model`` -- re-import the exported STEP into a ``CADModel`` so
  ``validator.run.validate`` operates on it unchanged. STEP is the geometry exchange format, so
  the re-imported B-rep is the same solid the child built -- and re-importing (rather than
  passing the live object across the process boundary) is exactly what a production backend does.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.agentic.sandbox import SandboxLimits, SandboxResult, run_build123d_code
from forge_mind.compiler.compile import CADModel
from forge_mind.compiler.topology_tags import TopologyIndex
from forge_mind.types import BBox, Vec3

_CADQUERY_RUNNER = "forge_mind.agentic._cadquery_runner"


def run_cadquery_code(
    code: str,
    job_dir: Path,
    *,
    part_name: str = "part",
    limits: SandboxLimits | None = None,
) -> SandboxResult:
    """Execute CadQuery ``code`` (must bind the final object to ``result``) in the sandbox,
    bridging it to a build123d solid and exporting STL + STEP. Delegates to the same crash-
    isolated driver the build123d path uses -- only the child harness differs."""
    return run_build123d_code(
        code,
        job_dir,
        part_name=part_name,
        limits=limits,
        runner_module=_CADQUERY_RUNNER,
    )


def load_validation_model(step_path: Path, stl_path: Path) -> CADModel:
    """Re-import the exported STEP into a ``CADModel`` for ``validator.run.validate``.

    Uses trimesh on the STL for the bounding box (the validator's own mesh tiers read the STL
    directly, so this only needs a coarse bbox). ``topology_index`` is left empty -- the
    interface/feature tiers introspect ``part``/``stl_path`` directly and do not depend on it
    being populated (matches the agentic path, ``agentic/validate.py``)."""
    from build123d import import_step

    part = import_step(str(step_path))

    lo = (0.0, 0.0, 0.0)
    hi = (0.0, 0.0, 0.0)
    try:
        import trimesh

        mesh = trimesh.load(str(stl_path), force="mesh")
        lo_b, hi_b = mesh.bounds
        lo = (float(lo_b[0]), float(lo_b[1]), float(lo_b[2]))
        hi = (float(hi_b[0]), float(hi_b[1]), float(hi_b[2]))
    except Exception:
        pass

    bbox = BBox(
        min=Vec3(x=lo[0], y=lo[1], z=lo[2]),
        max=Vec3(x=hi[0], y=hi[1], z=hi[2]),
    )
    return CADModel(
        part=part,
        stl_path=stl_path,
        step_path=step_path,
        topology_index=TopologyIndex(),
        bounding_box=bbox,
    )


__all__ = ["load_validation_model", "run_cadquery_code"]
