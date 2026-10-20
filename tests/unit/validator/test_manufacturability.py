"""Metal DFM profile + scoped certificate (founder 2026-07-12: 'not every part is 3D
printable'). The DFM tier is process-keyed and deterministic: it flags features a 3-axis
mill cannot make (sharp internal corners, sub-minimum walls). The certificate names its
scope so 'certified' never over-claims."""

from __future__ import annotations

import trimesh

from forge_mind.materials import resolve_material
from forge_mind.validator.manufacturability import (
    check_metal_dfm,
    dfm_for_material,
    scope_certificate,
)


def test_filleted_block_is_machinable() -> None:
    """A plain box has only convex external edges -- no sharp INTERNAL corners, thick
    walls: a clean machined-metal pass."""
    box = trimesh.creation.box(extents=[40, 30, 20])
    r = check_metal_dfm(box)
    assert r.verdict == "pass" and r.findings == []


def test_sharp_internal_pocket_is_flagged_as_caveat() -> None:
    """A pocket milled into a block leaves sharp internal vertical corners a round cutter
    cannot reach -- the canonical metal DFM defect. Caveat (add a fillet), not a fail."""
    block = trimesh.creation.box(extents=[40, 40, 20])
    pocket = trimesh.creation.box(extents=[20, 20, 12])
    pocket.apply_translation([0, 0, 6])
    part = trimesh.boolean.difference([block, pocket])
    assert isinstance(part, trimesh.Trimesh)
    r = check_metal_dfm(part)
    assert r.verdict == "caveats"
    assert any(f.check == "sharp_internal_corner" and f.severity == "caveat" for f in r.findings)


def test_thin_wall_is_a_blocker() -> None:
    """A wall below the 3-axis minimum chatters -- a hard blocker (unmakeable)."""
    shell = trimesh.boolean.difference(
        [trimesh.creation.box(extents=[40, 40, 40]), trimesh.creation.box(extents=[39.3, 39.3, 60])]
    )
    assert isinstance(shell, trimesh.Trimesh)
    r = check_metal_dfm(shell)
    assert r.verdict == "fail"
    assert any(f.check == "min_wall" and f.severity == "blocker" for f in r.findings)


def test_dfm_is_process_keyed() -> None:
    """METAL material -> the metal profile runs; FDM/RESIN -> None (printability owns
    those, per-part). None means 'no metal profile applies', never 'passed'."""
    box = trimesh.creation.box(extents=[40, 30, 20])
    assert dfm_for_material([box], resolve_material("steel")) is not None
    assert dfm_for_material([box], resolve_material("PLA")) is None
    assert dfm_for_material([box], resolve_material("resin")) is None


def test_certificate_names_its_scope() -> None:
    box = trimesh.creation.box(extents=[40, 30, 20])
    metal_dfm = dfm_for_material([box], resolve_material("steel"))
    cert = scope_certificate(
        disposition="shipped_with_caveats",
        material_process="metal",
        fit_verdict="seated",
        motion_status="pass",
        worst_case=None,
        dfm=metal_dfm,
        expert_blockers=1,
    )
    assert cert["geometry_kinematics"] == "caveats"
    assert cert["manufacturability"]["tier"] == "machined-metal"
    assert "CERTIFIED:" in cert["scope"] and "NOT CHECKED:" in cert["scope"]
    assert "machined-metal" in cert["scope"]
    assert cert["expert_blockers"] == 1


def test_certificate_marks_machinability_unchecked_for_printed_parts() -> None:
    """A PLA part gets FDM printability, and the certificate says machinability is NOT
    checked -- the honest scoping the founder asked for."""
    cert = scope_certificate(
        disposition="shipped",
        material_process="fdm",
        fit_verdict="seated",
        motion_status="pass",
        worst_case=None,
        dfm=None,
        expert_blockers=0,
    )
    assert cert["geometry_kinematics"] == "pass"
    assert "machinability" in cert["scope"] and "NOT CHECKED" in cert["scope"]
    assert cert["manufacturability"]["process"] == "fdm"
