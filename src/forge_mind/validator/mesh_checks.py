"""Tier 1: mesh checks. Fast, hard pass/fail, no B-rep needed.

Operates on the STL via trimesh. Catches the catastrophic failures (non-solid,
multi-body, oversized) before the more expensive tiers run.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from forge_mind.dsl.program import PartProgram
from forge_mind.fabrication import FabricationMode
from forge_mind.validator.report import Check, CheckStatus, Tier

# Absolute minimum realizable thickness (mm). A part whose SMALLEST overall dimension is below one thin
# FDM layer is physically unbuildable -- enforced UNCONDITIONALLY in a printable mode, independent of any
# declared min_wall. The feature tier only measures thickness when min_wall IS declared, so a part that
# declares none had NO thickness floor at all (the degenerate-sheet false-PASS the adversarial probe found).
MIN_PRINTABLE_THICKNESS_MM = 0.1


def _check(id: str, ok: bool, message: str, expected: str, actual: str) -> Check:
    return Check(
        id=id,
        tier=Tier.MESH,
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        message=message,
        expected=expected,
        actual=actual,
    )


def run_mesh_checks(
    stl_path: Path,
    program: PartProgram,
    *,
    fabrication_mode: FabricationMode = FabricationMode.PRINT_3D,
) -> list[Check]:
    mesh = trimesh.load(str(stl_path), force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        return [
            Check(
                id="mesh.loadable",
                tier=Tier.MESH,
                status=CheckStatus.FAIL,
                message=f"STL did not load as a single mesh (got {type(mesh).__name__}).",
            )
        ]

    checks: list[Check] = [
        _check(
            "mesh.watertight",
            bool(mesh.is_watertight),
            "Mesh must be watertight (no boundary edges).",
            expected="watertight",
            actual="watertight" if mesh.is_watertight else "has holes",
        ),
        _check(
            "mesh.is_volume",
            bool(mesh.is_volume),
            "Mesh must enclose a valid solid volume.",
            expected="solid",
            actual="solid" if mesh.is_volume else "not a solid",
        ),
        _check(
            "mesh.winding_consistent",
            bool(mesh.is_winding_consistent),
            "Mesh face winding must be consistent.",
            expected="consistent",
            actual="consistent" if mesh.is_winding_consistent else "inconsistent",
        ),
        _check(
            "mesh.single_body",
            mesh.body_count == 1,
            "Mesh must be a single connected shell.",
            expected="1 body",
            actual=f"{mesh.body_count} bodies",
        ),
        _check(
            "mesh.positive_volume",
            float(mesh.volume) > 1e-6,
            "Mesh must have positive volume.",
            expected="> 0 mm^3",
            actual=f"{float(mesh.volume):.3f} mm^3",
        ),
    ]

    # Absolute printable-thickness floor (printable mode only): a part whose smallest overall dimension
    # is under one thin FDM layer is unbuildable. UNCONDITIONAL -- unlike the feature tier it needs no
    # declared min_wall -- so it closes the degenerate-sheet escape the adversarial probe found.
    if fabrication_mode.enforces_printability:
        min_extent = float(min(mesh.bounding_box.extents))
        checks.append(
            _check(
                "mesh.min_printable_thickness",
                min_extent >= MIN_PRINTABLE_THICKNESS_MM,
                "The part's smallest dimension must be at least one printable layer.",
                expected=f">= {MIN_PRINTABLE_THICKNESS_MM} mm",
                actual=f"{min_extent:.3f} mm",
            )
        )

    # Bounding-box constraint (only if declared).
    max_bbox = program.constraints.max_bbox
    if max_bbox is not None:
        extents = mesh.bounding_box.extents  # (x, y, z)
        within = (
            extents[0] <= max_bbox.x + 1e-3
            and extents[1] <= max_bbox.y + 1e-3
            and extents[2] <= max_bbox.z + 1e-3
        )
        checks.append(
            _check(
                "mesh.bbox_within_max",
                bool(within),
                "Bounding box must fit within the declared maximum.",
                expected=f"<= ({max_bbox.x}, {max_bbox.y}, {max_bbox.z}) mm",
                actual=f"({extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}) mm",
            )
        )

        # Scale lower bound (ADR-0035 B1, advisory-first INFO): the DSL lane's envelopes are
        # derived tight to the part's footprint, so a body whose EVERY extent is under half the
        # declared envelope is almost certainly built at the wrong scale (the 0.1x-part class
        # that passes every other tier). INFO gates nothing; promotion per ADR-0035.
        if max_bbox.x > 0 and max_bbox.y > 0 and max_bbox.z > 0:
            shrunken = (
                extents[0] < 0.5 * max_bbox.x
                and extents[1] < 0.5 * max_bbox.y
                and extents[2] < 0.5 * max_bbox.z
            )
            checks.append(
                Check(
                    id="mesh.bbox_scale",
                    tier=Tier.MESH,
                    status=CheckStatus.INFO,
                    message=(
                        "The body is much smaller than its declared envelope -- possible "
                        "wrong-scale build."
                        if shrunken
                        else "Body extents are in proportion to the declared envelope."
                    ),
                    expected=f"each extent >= 50% of ({max_bbox.x}, {max_bbox.y}, {max_bbox.z}) mm",
                    actual=f"({extents[0]:.2f}, {extents[1]:.2f}, {extents[2]:.2f}) mm"
                    + (" [SHRUNKEN]" if shrunken else ""),
                )
            )

    return checks


__all__ = ["run_mesh_checks"]
