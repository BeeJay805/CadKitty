"""Pure contract controls; no mesh, renderer, CAD or optional library imports."""

from copy import deepcopy

import pytest

from forge_mind.workspace import product_photo_contract as contract


def request():
    return {
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "mesh_sha256": "c" * 64,
        "preset": "metal",
        "frame": contract.FRAME,
        "version": 1,
    }


def receipt(value):
    return {
        **value,
        "scope": contract.SCOPE,
        "image": {"file": "photo.png", "sha256": "d" * 64, "width": 1024, "height": 1024},
        "rendering": {
            "versions": {"pyvista": "0.48.4", "vtk": "9.3.1"},
            "camera": {
                "position": [[1.5, -2, 3], [0, 0, 0], [0, 0, 1]],
                "view_angle_degrees": 35.0,
                "distance_mm": 178.8,
                "maximum_absolute_ndc": 0.833,
                "projection": "perspective",
            },
            "lights": [{"position": [1, 2, 3], "intensity": 0.05}],
            "texture": {"tile_mm": 12, "seed": 8712},
            "ssao": {"radius_mm": 4.2, "blur": True},
            "shadow_method": "Native shadow maps with directional samples",
            "geometry": {
                "triangles": 5068,
                "triangle_sha256": "e" * 64,
                "triangle_positions_unchanged": True,
                "bounds_mm": [[-40, -25, -3], [40, 25, 3]],
            },
        },
    }


@pytest.mark.parametrize("preset", contract.PRESETS)
def test_exact_request_and_v4_facts_are_preserved_without_mutation(preset):
    wanted = request()
    wanted["preset"] = preset
    observed = receipt(wanted)
    before = deepcopy((wanted, observed))
    assert contract.validated_request(wanted) == wanted
    assert contract.validate_receipt(observed, wanted) == observed
    assert (wanted, observed) == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("project_id", "foreign"),
        ("revision_id", "b" * 31),
        ("mesh_sha256", "C" * 64),
        ("preset", "galvanized"),
        ("frame", "gltf_y_up"),
        ("version", True),
        ("version", 1.0),
    ],
)
def test_invalid_request_fields_refuse(key, value):
    wanted = request()
    wanted[key] = value
    with pytest.raises(ValueError):
        contract.validated_request(wanted)


def test_unknown_request_and_receipt_fields_refuse():
    wanted = request()
    with pytest.raises(ValueError):
        contract.validated_request({**wanted, "camera_script": "not allowed"})
    with pytest.raises(ValueError):
        contract.validate_receipt({**receipt(wanted), "engineering_pass": True}, wanted)


@pytest.mark.parametrize("key", ["project_id", "revision_id", "mesh_sha256", "preset"])
def test_valid_but_different_source_echo_is_rejected(key):
    wanted = request()
    observed = receipt(wanted)
    observed[key] = {
        "project_id": "f" * 32,
        "revision_id": "f" * 32,
        "mesh_sha256": "f" * 64,
        "preset": "plastic",
    }[key]
    with pytest.raises(ValueError, match="source"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("count", [0, -1, True, 1.0, 500_001])
def test_invalid_geometry_counts_refuse(count):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["geometry"]["triangles"] = count
    with pytest.raises(ValueError, match="triangle"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("unchanged", [False, 1, "true", None])
def test_unchanged_geometry_must_be_literal_true(unchanged):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["geometry"]["triangle_positions_unchanged"] = unchanged
    with pytest.raises(ValueError, match="triangle"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("count", [1, 500_000])
def test_both_geometry_count_boundaries_are_available(count):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["geometry"]["triangles"] = count
    assert contract.validate_receipt(observed, wanted) == observed


@pytest.mark.parametrize(
    "key,value",
    [("file", "../photo.png"), ("sha256", "invalid"), ("width", True), ("height", 512)],
)
def test_wrong_image_metadata_refuses(key, value):
    wanted = request()
    observed = receipt(wanted)
    observed["image"][key] = value
    with pytest.raises(ValueError, match="PNG"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_nested_render_facts_refuse(number):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["camera"]["position"][0][0] = number
    with pytest.raises(ValueError, match="finite"):
        contract.validate_receipt(observed, wanted)


def test_oversized_rendering_and_non_json_types_refuse():
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["graphics_context"] = "x" * contract.MAX_RECEIPT_BYTES
    with pytest.raises(ValueError, match="byte"):
        contract.validate_receipt(observed, wanted)
    observed["rendering"]["graphics_context"] = object()
    with pytest.raises(ValueError, match="JSON"):
        contract.validate_receipt(observed, wanted)


def test_wrong_scope_and_invalid_triangle_hash_refuse():
    wanted = request()
    observed = receipt(wanted)
    observed["scope"] = "Certified material"
    with pytest.raises(ValueError, match="scope"):
        contract.validate_receipt(observed, wanted)
    observed["scope"] = contract.SCOPE
    observed["rendering"]["geometry"]["triangle_sha256"] = "invalid"
    with pytest.raises(ValueError, match="triangle"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize(
    "key", ["versions", "camera", "lights", "texture", "ssao", "shadow_method"]
)
def test_partial_rendering_receipt_refuses(key):
    wanted = request()
    observed = receipt(wanted)
    del observed["rendering"][key]
    with pytest.raises(ValueError):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize(
    "key,value",
    [
        ("versions", {"vtk": 9.3}),
        ("camera", {"position": [[1, 2, 3]]}),
        ("lights", []),
        ("texture", {}),
        ("ssao", {}),
        ("shadow_method", ""),
    ],
)
def test_malformed_rendering_facts_refuse(key, value):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"][key] = value
    with pytest.raises(ValueError):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("key", ["view_angle_degrees", "distance_mm", "maximum_absolute_ndc"])
def test_camera_numeric_metadata_cannot_be_boolean(key):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["camera"][key] = True
    with pytest.raises(ValueError, match="finite"):
        contract.validate_receipt(observed, wanted)


@pytest.mark.parametrize("bounds", [[[1, 0, 0], [0, 1, 1]], [[0, 0, 0]], [[0, 0, True], [1, 1, 1]]])
def test_ordered_finite_geometry_bounds_required(bounds):
    wanted = request()
    observed = receipt(wanted)
    observed["rendering"]["geometry"]["bounds_mm"] = bounds
    with pytest.raises(ValueError, match="bounds"):
        contract.validate_receipt(observed, wanted)
