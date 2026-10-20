"""Isolated Gmsh worker: selected STEP solid to a dimension-checked SI fluid domain."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.flow_models import FlowMeshSettings, FlowSettings


def _layer_heights(mesh: FlowMeshSettings, reference: float, body_length: float) -> list[float]:
    """Refine wall spacing without shrinking the declared physical prism coverage."""
    if not mesh.wall_layers:
        return []
    growth = mesh.layer_growth
    log_growth = math.log(growth)
    first = reference * mesh.first_layer_size_factor
    try:
        coverage = first * math.expm1(mesh.wall_layers * log_growth) / (growth - 1)
    except OverflowError as exc:
        raise ValueError("Requested wall layer coverage is not finite") from exc
    if not math.isfinite(coverage) or coverage >= body_length * 0.5:
        raise ValueError("Requested wall layers exceed half the body length; revise layer settings")
    if mesh.refinement == "coarse":
        return [
            first * math.expm1((i + 1) * log_growth) / (growth - 1) for i in range(mesh.wall_layers)
        ]
    requested_first = first * mesh.fine_scale
    count = max(
        mesh.wall_layers,
        math.ceil(math.log1p(coverage * (growth - 1) / requested_first) / log_growth),
    )
    # Rescale a geometric series instead of leaving a very thin final sliver.
    # The actual first height is at most the requested refined height.
    denominator = math.expm1(count * log_growth)
    return [coverage * math.expm1((i + 1) * log_growth) / denominator for i in range(count)]


def _grading_expression(
    distance: int, threshold: int, zone: float, reference: float, rate: float, scale: float
) -> str:
    """Fixed physical field: a C1 outer ramp with asymptotic slope rate * scale."""
    beyond = f"((F{distance}-{zone:.17g})+Abs(F{distance}-{zone:.17g}))/2"
    return f"F{threshold}+{rate * scale:.17g}*({beyond})^2/(({beyond})+{reference:.17g})"


def mesh_flow(step_path: Path, directory: Path, settings: FlowSettings) -> dict[str, Any]:
    import gmsh
    import numpy as np
    from build123d import import_step

    source = import_step(step_path)
    solids = list(source.solids())
    index = settings.solid_index
    if index is None:
        if len(solids) != 1:
            raise ValueError("Flow requires one solid or an explicit exported solid index")
        index = 0
    if index >= len(solids):
        raise ValueError("Selected flow solid index does not exist")
    # Export the selected CAD kernel solid, avoiding assumptions about Gmsh import order.
    from build123d import export_step

    selected = solids[index]
    if not selected.is_valid or selected.volume <= 0:
        raise ValueError("Flow boundary must be a valid closed positive-volume CAD solid")
    box = selected.bounding_box()
    expected = np.asarray([tuple(box.min), tuple(box.max)], dtype=float) * 0.001
    export_step(selected, directory / "selected.step")
    gmsh.initialize()
    gmsh.logger.start()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", settings.resources.threads)
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        # OCC's default bounds can enclose spline control points far outside the
        # actual body. Triangulated bounds plus independently matched CAD volume
        # check SI conversion without rejecting ordinary spline/loft surfaces.
        gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
        gmsh.model.add("external_viscous_flow")
        imported = gmsh.model.occ.importShapes(str(directory / "selected.step"))
        gmsh.model.occ.synchronize()
        if len(imported) != 1 or imported[0][0] != 3:
            raise ValueError("Selected STEP did not import as exactly one solid volume")
        bounds = np.asarray(gmsh.model.occ.getBoundingBox(*imported[0])).reshape(2, 3)
        length = float(max(expected[1] - expected[0]))
        imported_volume = float(gmsh.model.occ.getMass(*imported[0]))
        if not np.allclose(
            bounds, expected, atol=max(2e-6, length * 0.002), rtol=0
        ) or not math.isclose(imported_volume, float(selected.volume) * 1e-9, rel_tol=1e-6):
            raise ValueError(
                "STEP-to-fluid-domain dimensions disagree after millimetre-to-metre conversion"
            )
        extent = expected[1] - expected[0]
        length = float(max(extent))
        if not math.isfinite(length) or length <= 0:
            raise ValueError("Selected flow body has invalid dimensions")
        reynolds = (
            settings.density_kg_m3
            * math.hypot(*settings.velocity_m_s)
            * length
            / settings.dynamic_viscosity_pa_s
        )
        reference = settings.mesh.reference_length_m or length
        padding = settings.mesh.domain_padding_factor * reference
        outer_min = expected[0] - padding
        outer_size = extent + 2 * padding
        outer = gmsh.model.occ.addBox(*outer_min, *outer_size)
        fluid, _ = gmsh.model.occ.cut([(3, outer)], imported)
        gmsh.model.occ.synchronize()
        if len(fluid) != 1 or fluid[0][0] != 3:
            raise ValueError("External fluid domain is disconnected or invalid")
        fluid_volume = float(gmsh.model.occ.getMass(*fluid[0]))
        wall: list[int] = []
        far: list[int] = []
        tolerance = max(2e-6, length * 1e-5)
        for dim, tag in gmsh.model.getBoundary(fluid, oriented=False):
            sb = np.asarray(gmsh.model.getBoundingBox(dim, tag)).reshape(2, 3)
            on_body = bool(
                np.all(sb[0] >= expected[0] - tolerance)
                and np.all(sb[1] <= expected[1] + tolerance)
            )
            (wall if on_body else far).append(tag)
        if not wall or len(far) != 6:
            raise ValueError("Could not identify complete body and enclosing farfield surfaces")
        scale = settings.mesh.fine_scale if settings.mesh.refinement == "fine" else 1.0
        near_size = reference * settings.mesh.surface_size_factor * scale
        far_size = reference * settings.mesh.farfield_size_factor * scale
        refinement_distance = reference * settings.mesh.refinement_distance_factor
        transition_start = (
            reference * settings.mesh.surface_size_factor
            if settings.mesh.reference_length_m is not None or settings.mesh.grading_rate > 0
            else near_size
        )
        layer_heights = _layer_heights(settings.mesh, reference, length)
        fluid_tags = [t for _, t in fluid]
        if settings.mesh.wall_layers:
            gmsh.model.occ.remove(fluid, recursive=False)
            gmsh.model.occ.synchronize()
            # Gmsh's native topological extrusion preserves the source CAD wall and
            # connects prismatic layers to the unstructured outer tetrahedral mesh.
            layers = gmsh.model.geo.extrudeBoundaryLayer(
                [(2, t) for t in wall], [1] * len(layer_heights), layer_heights, True
            )
            top = [layers[i - 1][1] for i in range(1, len(layers)) if layers[i][0] == 3]
            layer_volumes = [t for d, t in layers if d == 3]
            if not top or not layer_volumes:
                raise ValueError("Could not extrude a complete CAD wall layer")
            outer_loop = gmsh.model.geo.addSurfaceLoop(far)
            inner_loop = gmsh.model.geo.addSurfaceLoop(top)
            outer_fluid = gmsh.model.geo.addVolume([outer_loop, inner_loop])
            gmsh.model.geo.synchronize()
            fluid_tags = [*layer_volumes, outer_fluid]
        gmsh.model.addPhysicalGroup(2, wall, name="body")
        gmsh.model.addPhysicalGroup(2, far, name="farfield")
        gmsh.model.addPhysicalGroup(3, fluid_tags, name="fluid")
        distance = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", wall)
        threshold = gmsh.model.mesh.field.add("Threshold")
        for key, value in {
            "InField": distance,
            "SizeMin": near_size,
            "SizeMax": far_size,
            "DistMin": transition_start,
            "DistMax": refinement_distance,
        }.items():
            gmsh.model.mesh.field.setNumber(threshold, key, value)
        background = threshold
        grading_expression = None
        if settings.mesh.grading_rate > 0:
            grading_expression = _grading_expression(
                distance,
                threshold,
                refinement_distance,
                reference,
                settings.mesh.grading_rate,
                scale,
            )
            background = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(background, "F", grading_expression)
        gmsh.model.mesh.field.setAsBackgroundMesh(background)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20 / scale)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        # MeshAdapt also handles closed periodic spline surfaces with degenerate
        # poles, for which Frontal-Delaunay can silently emit no body elements.
        gmsh.option.setNumber("Mesh.Algorithm", 1)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.model.mesh.generate(3)
        nodes = len(gmsh.model.mesh.getNodes()[0])
        if (
            settings.resources.max_mesh_nodes is not None
            and nodes > settings.resources.max_mesh_nodes
        ):
            raise ValueError(f"Mesh resource policy exceeded: {nodes} nodes")
        types, tags, _ = gmsh.model.mesh.getElements(3)
        if not set(types) <= {4, 5, 6, 7} or 4 not in types or (layer_heights and 6 not in types):
            raise ValueError(
                f"Flow domain requires linear tetrahedra and requested prismatic layers, got element types {list(types)}"
            )
        element_tags = np.concatenate(tags)
        quality = np.asarray(gmsh.model.mesh.getElementQualities(element_tags, "minSICN"))
        if not np.isfinite(quality).all() or float(quality.min()) <= 0:
            raise ValueError("Fluid mesh contains nonpositive or invalid elements")
        gmsh.write(str(directory / "mesh.su2"))
        receipt: dict[str, Any] = {
            "mesher": "gmsh",
            "mesher_version": gmsh.__version__,
            "surface_algorithm": "MeshAdapt",
            "frame": "cad_xyz_m",
            "source_step_sha256": file_sha256(step_path),
            "selected_step_sha256": file_sha256(directory / "selected.step"),
            "mesh_sha256": file_sha256(directory / "mesh.su2"),
            "selected_solid_index": index,
            "source_solid_count": len(solids),
            "body_bounds_m": expected.tolist(),
            "body_length_m": length,
            "mesh_reference_length_m": reference,
            "body_volume_m3": float(selected.volume) * 1e-9,
            "reynolds_body_length": reynolds,
            "declared_regime": settings.regime,
            "regime_applicability": "unverified",
            "regime_note": (
                "The selected steady laminar or SA model is a declared assumption. "
                "Body-length Reynolds number alone does not establish transition, "
                "steadiness or turbulence-model applicability for this geometry."
            ),
            "fluid_volume_m3": fluid_volume,
            "domain_bounds_m": [outer_min.tolist(), (outer_min + outer_size).tolist()],
            "body_surface_tags": wall,
            "farfield_surface_tags": far,
            "nodes": nodes,
            "tetrahedra": sum(len(t) for k, t in zip(types, tags, strict=True) if k == 4),
            "prisms": sum(len(t) for k, t in zip(types, tags, strict=True) if k == 6),
            "hexahedra": sum(len(t) for k, t in zip(types, tags, strict=True) if k == 5),
            "pyramids": sum(len(t) for k, t in zip(types, tags, strict=True) if k == 7),
            "minimum_scaled_jacobian": float(quality.min()),
            "surface_size_m": near_size,
            "farfield_size_m": far_size,
            "refinement_distance_m": refinement_distance,
            "transition_start_m": transition_start,
            "domain_padding_m": padding,
            "domain_rule": "body bounding box plus physical reference padding",
            "grading_policy": "fixed-distance smooth outer growth v1"
            if grading_expression
            else "uniform beyond refinement distance",
            "grading_expression": grading_expression,
            "grading_rate_scaled": settings.mesh.grading_rate * scale,
            "boundary_layer": "prismatic CAD-normal extrusion"
            if layer_heights
            else "isotropic tetrahedra; no prismatic wall layers",
            "layer_heights_m": layer_heights,
            "actual_wall_layers": len(layer_heights),
            "actual_first_layer_height_m": layer_heights[0] if layer_heights else None,
            "wall_layer_coverage_m": layer_heights[-1] if layer_heights else 0.0,
            "settings": settings.mesh.model_dump(),
        }
        (directory / "mesh_receipt.json").write_text(
            json.dumps(receipt, allow_nan=False), encoding="utf-8"
        )
        return receipt
    finally:
        (directory / "mesher_messages.log").write_text(
            "\n".join(gmsh.logger.get()), encoding="utf-8"
        )
        gmsh.logger.stop()
        gmsh.finalize()


def main() -> None:
    directory = Path(sys.argv[1]).resolve()
    settings = FlowSettings.model_validate_json(
        (directory / "settings.json").read_text(encoding="utf-8")
    )
    mesh_flow(directory / "source.step", directory, settings)


if __name__ == "__main__":
    main()
