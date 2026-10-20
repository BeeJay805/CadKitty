"""Bulk display selection retains world vertices and validates omitted topology."""

from copy import deepcopy

import numpy as np
import pytest

from forge_mind.workspace import occurrence_render as render


def topology_fixture(count=3):
    entities, meshes = [], []
    for index in range(count):
        offset = 10 * index
        identity = f"body_{index}"
        signature = f"{index + 1:064x}"
        entities.append(
            {
                "id": identity,
                "kind": "body",
                "body_index": index,
                "signature": signature,
                "body_signature": signature,
                "min": [offset, 0, 0],
                "max": [offset + 1, 1, 0],
            }
        )
        meshes.append(
            {
                "id": identity,
                "body_index": index,
                "vertices": [[offset, 0, 0], [offset + 1, 0, 0], [offset, 1, 0]],
                "triangles": [[0, 1, 2]],
            }
        )
    return {
        "frame": "cad_z_up_mm",
        "selection_scope": "body",
        "topology_counts": {"bodies": count},
        "entities": entities,
        "meshes": meshes,
        "pick_tessellation": {
            "purpose": "Selection preview only; not manufacturing geometry",
            "relative_linear_deflection": 0.01,
            "angular_deflection_radians": 0.1,
        },
    }


def test_bulk_matches_single_wrapper_in_sorted_world_order_without_mutation():
    source = topology_fixture()
    before = deepcopy(source)
    mesh, total, discarded = render.selected_topology_meshes(source, [2, 0])
    expected = np.concatenate([render._selected_mesh(source, i)[0].triangles for i in [0, 2]])
    assert np.array_equal(mesh.triangles, expected)
    assert total == 3 and discarded == 0 and source == before
    assert mesh.bounds.tolist() == [[0, 0, 0], [21, 1, 0]]


@pytest.mark.parametrize("indices", [[], [True], [False], [-1], [3], [0, 0], (0,)])
def test_invalid_selection_refuses(indices):
    with pytest.raises(ValueError):
        render.selected_topology_meshes(topology_fixture(), indices)


@pytest.mark.parametrize(
    "mutation", ["nan", "bounds", "signature", "mesh_id", "index", "duplicate", "coverage"]
)
def test_invalid_omitted_data_refuses(mutation):
    source = topology_fixture()
    if mutation == "nan":
        source["meshes"][2]["vertices"][0][0] = float("nan")
    elif mutation == "bounds":
        source["meshes"][2]["vertices"][0][0] = 500
    elif mutation == "signature":
        source["entities"][2]["body_signature"] = "0" * 64
    elif mutation == "mesh_id":
        source["meshes"][2]["id"] = "unknown"
    elif mutation == "index":
        source["meshes"][2]["triangles"][0][1] = True
    elif mutation == "duplicate":
        source["meshes"].append(deepcopy(source["meshes"][2]))
    else:
        source["meshes"].pop(0)  # Selected coverage cannot be inferred from total bounds.
    with pytest.raises(ValueError):
        render.selected_topology_meshes(source, [0])


def test_global_limit_includes_omitted_meshes(monkeypatch):
    monkeypatch.setattr(render, "MAX_TRIANGLES", 2)
    with pytest.raises(render.TopologyMeshLimitError):
        render.selected_topology_meshes(topology_fixture(), [0])


def test_mesh_and_entity_collections_are_not_rescanned_for_each_selected_body():
    class Counted(list):
        passes = 0

        def __iter__(self):
            self.passes += 1
            return super().__iter__()

    source = topology_fixture()
    source["meshes"] = Counted(source["meshes"])
    source["entities"] = Counted(source["entities"])
    render.selected_topology_meshes(source, [0, 1, 2])
    assert source["meshes"].passes == 1
    assert source["entities"].passes == 1


def test_face_mode_requires_all_selected_trimmed_face_meshes():
    source = topology_fixture(1)
    source["selection_scope"] = "face_edge_body"
    source["meshes"][0]["id"] = "face_0"
    for identity in ("face_0", "face_1"):
        row = {**source["entities"][0], "id": identity, "kind": "face"}
        source["entities"].append(row)
    with pytest.raises(ValueError, match="coverage"):
        render.selected_topology_meshes(source, [0])
    source["meshes"].append({**deepcopy(source["meshes"][0]), "id": "face_1"})
    mesh, _, _ = render.selected_topology_meshes(source, [0])
    assert len(mesh.faces) == 2
