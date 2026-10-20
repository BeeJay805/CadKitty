"""Complete mesh placement controls; no native STEP or model calls."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import trimesh

from forge_mind.workspace.print_setup import (
    arithmetic_guard,
    binary_stl,
    inspect_mesh,
    oriented_stl,
    sha,
    stl_triangles,
    validate_transform,
)
from forge_mind.workspace.print_setup_models import PrintPlane, PrintSource


def box(center=(0, 0, 0), extents=(20, 30, 10)):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return np.asarray(mesh.triangles).copy()


def source(triangles, *, bodies=1):
    data = binary_stl(triangles)
    return PrintSource(
        revision_id="a" * 32,
        part_id="body",
        quantity=2,
        document_sha256="b" * 64,
        manifest_sha256="c" * 64,
        step_file="definition_body.step",
        step_sha256="d" * 64,
        stl_file="definition_body.stl",
        stl_sha256=sha(data),
        stl_bytes=len(data),
        body_count=bodies,
    )


def plane(point=(0, 0, -5), normal=(0, 0, -1), index=0):
    return PrintPlane(
        face_index=index,
        point_mm=list(point),
        outward_normal=list(normal),
        area_mm2=600.0,
        kernel_tolerance_mm=1e-7,
    )


@pytest.mark.parametrize("normal,point", [((0, 0, -1), (0, 0, -5)), ((0, 0, 1), (0, 0, 5))])
def test_bottom_and_180_degree_top_pose_preserve_every_triangle(normal, point):
    triangles = box()
    binding = source(triangles)
    inspected = inspect_mesh(triangles, binding, [plane(point, normal)])
    candidate = inspected.candidates[0]
    assert candidate.status == "measured"
    assert inspected.recommended_candidate_hash == candidate.candidate_hash
    matrix = np.asarray(candidate.transform)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1)
    assert matrix[:3, :3] @ np.asarray(normal) == pytest.approx([0, 0, -1])
    data, receipt = oriented_stl(triangles, binding, candidate)
    delivered = stl_triangles(data)
    assert delivered[:, :, 2].min() == 0
    assert delivered[:, :, 2].max() == 10
    assert (delivered - matrix[:3, 3]) @ matrix[:3, :3] == pytest.approx(triangles)
    assert receipt["triangle_inventory_preserved"] is True
    assert receipt["quantity"] == 2
    assert receipt["source_mesh"]["volume_mm3"] == pytest.approx(6000)
    assert receipt["derived_mesh"]["volume_mm3"] == pytest.approx(6000)
    assert receipt["physical_readiness"] == "unverified"


def test_oblique_measured_plane_has_explicit_float32_uncertainty():
    angle = np.deg2rad(23)
    rotation = np.array(
        [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    )
    analytic = box() @ rotation.T
    triangles = stl_triangles(binary_stl(analytic))
    bottom = np.array([0, 0, -5]) @ rotation.T
    normal = np.array([0, 0, -1]) @ rotation.T
    binding = source(triangles)
    inspected = inspect_mesh(triangles, binding, [plane(bottom, normal)])
    assert inspected.status == "measured"
    data, receipt = oriented_stl(triangles, binding, inspected.candidates[0])
    assert receipt["float32_maximum_rounding_mm"] <= receipt["float32_half_ulp_mm"] + 1e-10
    assert receipt["inverse_maximum_coordinate_error_mm"] <= receipt["inverse_numeric_guard_mm"]
    assert len(stl_triangles(data)) == len(triangles)


def test_any_protrusion_below_selected_plane_blocks_the_candidate():
    triangles = np.concatenate([box(center=(0, 0, 5)), box(center=(40, 0, -2), extents=(2, 2, 2))])
    inspected = inspect_mesh(triangles, source(triangles, bodies=2), [plane(point=(0, 0, 0))])
    assert inspected.status == "unresolved"
    observation = inspected.candidates[0].observations["whole_mesh"]
    assert observation["minimum_plate_z_mm"] == -3
    assert observation["below_plate_vertex_count"] > 0
    assert inspected.recommended_candidate_hash is None


def test_floating_second_body_cannot_use_first_body_contact_hull():
    triangles = np.concatenate([box(center=(0, 0, 5)), box(center=(0, 0, 25))])
    inspected = inspect_mesh(triangles, source(triangles, bodies=2), [plane(point=(0, 0, 0))])
    assert inspected.status == "unresolved"
    observations = inspected.candidates[0].observations
    assert observations["whole_mesh"]["centroid_inside_contact_hull"] is True
    assert any(body["contact_area_mm2"] == 0 for body in observations["bodies"])


def test_no_flat_native_face_leaves_sphere_unresolved():
    triangles = np.asarray(trimesh.creation.icosphere(subdivisions=1, radius=10).triangles)
    inspected = inspect_mesh(triangles, source(triangles), [])
    assert inspected.status == "unresolved"
    assert inspected.candidates == []
    assert inspected.recommended_candidate_hash is None


@pytest.mark.parametrize("change", ["reflection", "scale", "shear", "nonfinite", "last_row"])
def test_nonrigid_or_invalid_transform_is_refused(change):
    matrix = np.eye(4)
    if change == "reflection":
        matrix[0, 0] = -1
    elif change == "scale":
        matrix[0, 0] = 1.001
    elif change == "shear":
        matrix[0, 1] = 0.2
    elif change == "nonfinite":
        matrix[0, 3] = np.inf
    else:
        matrix[3, 0] = 1
    with pytest.raises(ValueError, match=r"finite|proper rigid"):
        validate_transform(matrix)


def test_changed_candidate_cannot_be_exported_under_old_identity():
    triangles = box()
    binding = source(triangles)
    candidate = inspect_mesh(triangles, binding, [plane()]).candidates[0]
    changed = deepcopy(candidate)
    changed.transform[2][3] += 1
    with pytest.raises(ValueError, match="candidate hash"):
        oriented_stl(triangles, binding, changed)


def test_changed_input_triangles_cannot_use_an_old_measured_candidate():
    triangles = box()
    binding = source(triangles)
    candidate = inspect_mesh(triangles, binding, [plane()]).candidates[0]
    changed = triangles.copy()
    changed[:, :, 0] += 1
    with pytest.raises(ValueError, match="source triangle inventory"):
        oriented_stl(changed, binding, candidate)


def test_open_or_badly_wound_saved_mesh_is_not_repaired():
    triangles = box()
    with pytest.raises(ValueError, match="watertight"):
        inspect_mesh(triangles[:-1], source(triangles[:-1]), [plane()])
    triangles[0] = triangles[0, ::-1]
    with pytest.raises(ValueError, match="winding"):
        inspect_mesh(triangles, source(triangles), [plane()])


def test_collapsed_triangle_and_truncated_inventory_are_refused():
    triangles = box()
    data = binary_stl(triangles)
    with pytest.raises(ValueError):
        stl_triangles(data[:-1])
    triangles[0, 0] = triangles[0, 1]
    with pytest.raises(ValueError, match="collapsed"):
        binary_stl(triangles)


def test_precision_guard_scales_with_actual_coordinates():
    small = arithmetic_guard(box(), plane())
    shifted = box(center=(1e8, 1e8, 1e8), extents=(64, 64, 64))
    large = arithmetic_guard(shifted, plane(point=(1e8, 1e8, 1e8 - 32)))
    assert (
        large["source_float32_half_ulp_projection_mm"]
        > small["source_float32_half_ulp_projection_mm"]
    )
    assert large["arithmetic_mm"] > small["arithmetic_mm"]


def test_rank_is_deterministic_area_then_native_face_index():
    triangles = box()
    binding = source(triangles)
    large = plane(index=2)
    equal = plane(index=1)
    narrow = plane(point=(10, 0, 0), normal=(1, 0, 0), index=0)
    inspected = inspect_mesh(triangles, binding, [large, narrow, equal])
    assert inspected.recommended_candidate_hash == inspected.candidates[2].candidate_hash
