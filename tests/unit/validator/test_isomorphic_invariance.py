"""Isomorphic-invariance property tests for the validator's mesh tier (LLMs-Gaming-Verifiers idea: a
sound verifier must not flip its verdict under a meaning-preserving transform). Watertight / solid /
winding / single-body / positive-volume are properties of the SHAPE, so a rigid transform (the same
part moved or turned) or sub-tolerance vertex noise must leave every mesh-tier verdict identical.

Scope note (honest): only the mesh tier is asserted, because it is the tier whose verdict is provably
rigid-invariant. Full-report invariance is deliberately NOT claimed -- printability/overhang is
orientation-dependent BY DESIGN (a part flat on the bed prints differently than on its side) and
interface detection is name-semantic, so neither is rotation- nor relabel-invariant. Extending a
translation-only check to the whole report is a follow-up, not a property that holds today.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from trimesh.transformations import rotation_matrix, translation_matrix

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, Cylinder, OpMode
from forge_mind.dsl.program import Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.mesh_checks import run_mesh_checks


def _program() -> PartProgram:
    # A clean plate with two through-holes; no max_bbox declared, so every mesh check is one of the
    # rigid-invariant five (the bbox check is the only position/orientation-sensitive one).
    return PartProgram(
        metadata=Metadata(name="plate_two_holes"),
        operations=[
            Box(op_id="plate", width=40.0, depth=20.0, height=4.0, position=Vec3(z=2.0)),
            Cylinder(
                op_id="h1",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(x=-10.0, z=2.0),
            ),
            Cylinder(
                op_id="h2",
                mode=OpMode.SUBTRACT,
                radius=1.6,
                height=4.0,
                position=Vec3(x=10.0, z=2.0),
            ),
        ],
    )


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):  # type: ignore[no-untyped-def]
    program = _program()
    model = compile_program(program, tmp_path_factory.mktemp("iso"))
    statuses = {c.id: c.status for c in run_mesh_checks(model.stl_path, program)}
    return model.stl_path, program, statuses


def _statuses_after(src_stl, program, matrix, tmp_path, jitter=0.0):  # type: ignore[no-untyped-def]
    mesh = trimesh.load(str(src_stl), force="mesh")
    assert isinstance(mesh, trimesh.Trimesh)
    if matrix is not None:
        mesh.apply_transform(matrix)
    if jitter:
        rng = np.random.default_rng(0)
        mesh.vertices = mesh.vertices + rng.normal(0.0, jitter, mesh.vertices.shape)
    dst = tmp_path / "transformed.stl"
    mesh.export(str(dst))
    return {c.id: c.status for c in run_mesh_checks(dst, program)}


def test_baseline_is_all_pass(baseline) -> None:  # type: ignore[no-untyped-def]
    _, _, base = baseline
    assert base, "expected the mesh tier to run at least one check"
    assert all(s.value == "pass" for s in base.values())


def test_translation_leaves_every_mesh_verdict_unchanged(baseline, tmp_path) -> None:  # type: ignore[no-untyped-def]
    src, program, base = baseline
    got = _statuses_after(src, program, translation_matrix([123.0, -45.0, 67.0]), tmp_path)
    assert got == base


def test_arbitrary_rotation_leaves_every_mesh_verdict_unchanged(baseline, tmp_path) -> None:  # type: ignore[no-untyped-def]
    src, program, base = baseline
    matrix = rotation_matrix(np.radians(37.0), [0.3, 0.7, 0.64])
    got = _statuses_after(src, program, matrix, tmp_path)
    assert got == base


def test_subtolerance_jitter_leaves_every_mesh_verdict_unchanged(baseline, tmp_path) -> None:  # type: ignore[no-untyped-def]
    src, program, base = baseline
    got = _statuses_after(src, program, None, tmp_path, jitter=1e-4)
    assert got == base
