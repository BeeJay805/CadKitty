"""STEP export alongside STL (editable B-rep output).

The compiler emits a ``.step`` next to every ``.stl`` (additive). These tests prove:
  * ``CADModel.step_path`` exists and the file is non-empty;
  * a STEP round-trip (reimport via build123d) matches the STL's bbox + volume
    within a small tolerance, for a primitive AND a real fixture part;
  * the STL output is behaviour-preserving (unchanged volume) after adding STEP.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import trimesh
from build123d import import_step

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box
from forge_mind.dsl.program import Metadata, PartProgram

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "handwritten" / "nema17_mount_good.partdsl.json"

# The hand-coded Phase 0 ground truth (examples/handwritten_nema17.py); the STL must
# still match it after the STEP addition (behaviour-preserving).
PHASE_0_VOLUME_MM3 = 11252.89


def _box_program() -> PartProgram:
    return PartProgram(
        metadata=Metadata(name="just_a_box"),
        operations=[Box(op_id="b", width=10.0, depth=10.0, height=4.0)],
    )


def test_step_emitted_alongside_stl(tmp_path: Path) -> None:
    model = compile_program(_box_program(), tmp_path)
    assert model.step_path.exists(), "no .step emitted"
    assert model.step_path.stat().st_size > 0, ".step is empty"
    assert model.step_path.name == "just_a_box.step"
    # The STL is still produced next to it (additive, not a replacement).
    assert model.stl_path.exists() and model.stl_path.stat().st_size > 0


def test_step_roundtrip_matches_stl_primitive(tmp_path: Path) -> None:
    model = compile_program(_box_program(), tmp_path)
    stl = trimesh.load(str(model.stl_path), force="mesh")
    assert isinstance(stl, trimesh.Trimesh)

    reimported = import_step(str(model.step_path))
    step_size = reimported.bounding_box().size
    stl_ext = stl.extents  # (x, y, z)

    # A box is exact in both representations -> tight tolerance.
    assert math.isclose(reimported.volume, float(stl.volume), rel_tol=5e-3)
    for axis, step_dim, stl_dim in zip(
        "XYZ", (step_size.X, step_size.Y, step_size.Z), stl_ext, strict=True
    ):
        assert math.isclose(step_dim, float(stl_dim), abs_tol=0.1), f"{axis} bbox mismatch"


def test_step_roundtrip_matches_stl_fixture(tmp_path: Path) -> None:
    """Round-trip the real NEMA17 fixture (curved holes): STEP vs STL within FDM-mesh
    tolerance (the STL tessellates cylinders, so volume drifts ~<1%)."""
    program = PartProgram.model_validate(json.loads(FIXTURE.read_text()))
    model = compile_program(program, tmp_path)
    assert model.step_path.exists() and model.step_path.stat().st_size > 0

    stl = trimesh.load(str(model.stl_path), force="mesh")
    assert isinstance(stl, trimesh.Trimesh)
    reimported = import_step(str(model.step_path))
    step_size = reimported.bounding_box().size
    stl_ext = stl.extents

    assert math.isclose(reimported.volume, float(stl.volume), rel_tol=1e-2), (
        f"STEP volume {reimported.volume:.2f} vs STL {stl.volume:.2f} beyond 1%"
    )
    for axis, step_dim, stl_dim in zip(
        "XYZ", (step_size.X, step_size.Y, step_size.Z), stl_ext, strict=True
    ):
        assert math.isclose(step_dim, float(stl_dim), abs_tol=0.2), f"{axis} bbox mismatch"


def test_stl_unchanged_after_step_addition(tmp_path: Path) -> None:
    """Behaviour-preserving: the STL the compiler emits is byte-stable across runs and
    its geometry still matches the Phase 0 golden -> adding STEP did not perturb STL."""
    program = PartProgram.model_validate(json.loads(FIXTURE.read_text()))
    a = compile_program(program, tmp_path / "a")
    b = compile_program(program, tmp_path / "b")
    assert a.stl_path.read_bytes() == b.stl_path.read_bytes(), "STL no longer byte-deterministic"

    stl = trimesh.load(str(a.stl_path), force="mesh")
    assert isinstance(stl, trimesh.Trimesh)
    err_pct = abs(stl.volume - PHASE_0_VOLUME_MM3) / PHASE_0_VOLUME_MM3 * 100
    assert err_pct < 0.5, (
        f"STL volume drifted from Phase 0 golden ({stl.volume:.2f}); err {err_pct:.3f}%"
    )
