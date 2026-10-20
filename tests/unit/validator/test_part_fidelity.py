"""Part-fidelity tier (founder 2026-07-12, the Merlin valves): robust watertightness +
role-keyed external-massing. Both sides are load-bearing: a real defect must never pass,
AND a legitimately-simple part (bracket, pin, stem) must never be flagged. Thresholds
were measured on the real probe corpus (ADR-0091)."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.validator.manufacturability import scope_certificate
from forge_mind.validator.part_fidelity import fidelity_report


def _detailed_valve() -> trimesh.Trimesh:
    """Body + THROUGH-bore + flange + actuator: a low-poly but topologically honest
    valve (genus 1 -- it can pass flow)."""
    body = trimesh.creation.cylinder(radius=15, height=40)
    flange = trimesh.creation.cylinder(radius=20, height=4)
    flange.apply_translation([0, 0, -18])
    actuator = trimesh.creation.box(extents=[10, 30, 10])
    actuator.apply_translation([0, 20, 0])
    bore = trimesh.creation.cylinder(radius=5, height=60)
    part = trimesh.boolean.difference([trimesh.boolean.union([body, flange, actuator]), bore])
    assert isinstance(part, trimesh.Trimesh)
    return part


def test_detailed_valve_passes() -> None:
    r = fidelity_report([("lox ball valve", _detailed_valve())])
    assert r.verdict == "pass" and r.findings == []


def test_bare_sphere_valve_is_caveated_external_massing() -> None:
    """The founder's Merlin case: a 'valve' that is a watertight blob with no
    through-passage is external massing -- caveat, never a silent pass."""
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=20)
    r = fidelity_report([("rp1 ball valve", sphere)])
    assert r.verdict == "caveats"
    assert any(
        f.check == "external_massing" and f.severity == "caveat" and "massing" in f.detail
        for f in r.findings
    )


def test_solid_convex_housing_is_caveated() -> None:
    """A 'gearbox housing' that is a solid near-convex brick has no cavity for what it
    houses -- massing. (A real shell housing is strongly non-convex and passes.)"""
    r = fidelity_report([("gearbox housing", trimesh.creation.box(extents=[60, 40, 30]))])
    assert r.verdict == "caveats"
    assert any(f.check == "external_massing" for f in r.findings)


def test_simple_bracket_and_pin_are_not_flagged() -> None:
    """The false-FAIL side: roles that do not imply internals are never checked, and a
    solid-subpart word (stem/pin) negates a role word -- a valve STEM is a solid."""
    bracket = trimesh.creation.box(extents=[40, 20, 5])
    pin = trimesh.creation.cylinder(radius=3, height=30)
    stem = trimesh.creation.cylinder(radius=4, height=50)
    r = fidelity_report([("corner bracket", bracket), ("hinge pin", pin), ("valve stem", stem)])
    assert r.verdict == "pass" and r.findings == []


def test_open_boundary_mesh_is_a_blocker() -> None:
    """A mesh missing real surface (open boundary that welding cannot close) is not a
    solid -- the one unambiguous BLOCKER."""
    box = trimesh.creation.box(extents=[40, 40, 40])
    keep = box.face_normals[:, 2] < 0.9  # drop the +Z cap: a 4-edge open square
    holed = trimesh.Trimesh(vertices=box.vertices.copy(), faces=box.faces[keep], process=False)
    r = fidelity_report([("bearing block", holed)])
    assert r.verdict == "fail"
    assert any(f.check == "solidity" and f.severity == "blocker" for f in r.findings)


def test_raw_stl_seam_artifact_is_not_flagged() -> None:
    """Raw STL geometry duplicates vertices per triangle, so ``is_watertight`` reads
    False on a perfectly good solid (the Merlin false flag) -- welding must clear it."""
    box = trimesh.creation.box(extents=[30, 30, 30])
    soup = trimesh.Trimesh(
        vertices=box.triangles.reshape(-1, 3),
        faces=np.arange(len(box.faces) * 3).reshape(-1, 3),
        process=False,
    )
    assert not soup.is_watertight  # the artifact this check must see through
    r = fidelity_report([("widget", soup)])
    assert r.verdict == "pass" and r.findings == []


def test_non_manifold_fused_geometry_is_reported_not_gated() -> None:
    """Wrong-euler geometry with NO open boundary (overlapping shells fused in place)
    was measured on shipped-and-printable parts -- report it, never block on it."""
    box = trimesh.creation.box(extents=[30, 30, 30])
    fused = trimesh.Trimesh(
        vertices=box.vertices.copy(),
        faces=np.vstack([box.faces, box.faces[:1]]),
        process=False,
    )
    assert not fused.is_watertight
    r = fidelity_report([("spacer", fused)])
    assert r.verdict == "caveats"  # caveats never break the clean-ship fold
    assert any(
        f.check == "solidity" and f.severity == "caveat" and "non-manifold" in f.detail
        for f in r.findings
    )


def test_empty_input_is_unchecked() -> None:
    r = fidelity_report([])
    assert r.verdict == "unchecked" and r.findings == []


def test_certificate_carries_the_fidelity_tier() -> None:
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=20)
    rep = fidelity_report([("ball valve", sphere)])
    cert = scope_certificate(
        disposition="shipped_with_caveats",
        material_process="fdm",
        fit_verdict="seated",
        motion_status="pass",
        worst_case=None,
        dfm=None,
        expert_blockers=0,
        fidelity=rep,
    )
    assert cert["fidelity"] == "caveats"
    assert "part fidelity (caveats)" in cert["scope"]


def test_certificate_marks_fidelity_unchecked_when_absent() -> None:
    cert = scope_certificate(
        disposition="shipped",
        material_process="fdm",
        fit_verdict=None,
        motion_status=None,
        worst_case=None,
        dfm=None,
        expert_blockers=0,
    )
    assert cert["fidelity"] is None
    assert "part fidelity" in cert["scope"] and "NOT CHECKED" in cert["scope"]
