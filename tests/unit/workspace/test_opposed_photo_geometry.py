"""Pure camera/contract and staged-worker dispatch checks, not visual qualification."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
import zlib
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from forge_mind.workspace import product_photo_render as worker
from forge_mind.workspace.product_photo_contract import (
    ASSEMBLY_SCOPE,
    ASSEMBLY_SOURCE_KEYS,
    CAPTURED_SCOPE,
    FRAME,
    OPPOSED_VIEW,
    SCOPE,
    legacy_request,
    opposed_request,
    validate_receipt,
    validated_request,
)


def request(family="mesh", preset="metal"):
    value = {
        "version": 1,
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "preset": preset,
        "frame": FRAME,
        "mesh_sha256": "c" * 64,
    }
    if family in ("captured", "assembly"):
        value.update(package_id="d" * 32, package_sha256="e" * 64)
    if family == "captured":
        value.update(
            part_id="holder", file_id="part-holder-stl", quantity=2, frame="definition_local"
        )
    if family == "assembly":
        value.pop("mesh_sha256")
        value.update(
            source_kind="saved_topology_made_assembly",
            source=dict.fromkeys(ASSEMBLY_SOURCE_KEYS, "c" * 64),
            selection={
                "selected": [
                    {
                        "occurrence_path": "holder_inst",
                        "part_id": "holder",
                        "role": "make",
                        "body_index": 0,
                    }
                ],
                "omitted": [],
                "total_bodies": 1,
            },
        )
    return value


def points(size=(80, 50, 6), shift=(0, 0, 0)):
    return np.array(list(itertools.product(*[(-x / 2, x / 2) for x in size]))) + shift


def render_facts(vertices, view=None):
    return {
        "versions": {"renderer": "pure fixture, no graphics context"},
        "camera": worker.fitted_camera(vertices, view=view),
        "lights": [{"intensity": 1}],
        "texture": {"fixture": True},
        "ssao": {"fixture": True},
        "shadow_method": "fixture",
        "geometry": {
            "triangles": 1,
            "triangle_sha256": "f" * 64,
            "triangle_positions_unchanged": True,
            "bounds_mm": [vertices.min(axis=0).tolist(), vertices.max(axis=0).tolist()],
        },
    }


def receipt(value, vertices=None):
    vertices = points() if vertices is None else vertices
    scope = (
        ASSEMBLY_SCOPE if "selection" in value else CAPTURED_SCOPE if "part_id" in value else SCOPE
    )
    facts = render_facts(vertices, value.get("view"))
    if "selection" in value:
        facts["source_tessellation"] = {
            "purpose": "saved display",
            "relative_linear_deflection": 0.01,
            "angular_deflection_radians": 0.1,
            "selected_body_count": 1,
            "discarded_zero_area_triangles": 0,
            "vertex_precision": worker.DISPLAY_VERTEX_PRECISION,
        }
    return {
        **deepcopy(value),
        "scope": scope,
        "image": {"file": "photo.png", "sha256": "f" * 64, "width": 1024, "height": 1024},
        "rendering": facts,
    }


@pytest.mark.parametrize("family", ["mesh", "captured", "assembly"])
@pytest.mark.parametrize("preset", ["metal", "plastic"])
def test_three_families_roundtrip_without_rewriting_or_aliasing_sources(family, preset):
    primary = request(family, preset)
    before = deepcopy(primary)
    alternate = opposed_request(primary)
    assert alternate == {**primary, "version": 2, "view": OPPOSED_VIEW}
    assert legacy_request(alternate) == primary == before
    assert legacy_request(primary) == primary
    assert validate_receipt(receipt(alternate), alternate)["view"] == OPPOSED_VIEW
    if family == "assembly":
        alternate["selection"]["selected"][0]["body_index"] = 9
        assert primary == before
        normalized = legacy_request(opposed_request(primary))
        normalized["source"]["topology_sha256"] = "0" * 64
        assert primary == before
    assert "view" not in primary


@pytest.mark.parametrize(
    "mutation",
    [
        {"version": 2},
        {"version": True},
        {"view": OPPOSED_VIEW},
        {"version": 2, "view": "rear"},
        {"version": 2, "view": None},
        {"version": 2, "view": OPPOSED_VIEW, "camera": [1, 2, 3]},
        {"version": 3, "view": OPPOSED_VIEW},
    ],
)
def test_only_exact_v1_or_opposed_v2_requests_are_admitted(mutation):
    value = {**request(), **mutation}
    with pytest.raises(ValueError):
        validated_request(value)
    with pytest.raises(ValueError):
        legacy_request(value)


def test_opposed_helper_does_not_reinterpret_an_alternate_as_primary():
    with pytest.raises(ValueError):
        opposed_request(opposed_request(request()))


def test_legacy_receipt_camera_validation_remains_unchanged():
    primary = request()
    value = receipt(primary)
    # Historical v1 only requires finite camera facts, not the new opposed basis.
    value["rendering"]["camera"] = {
        "position": [[0, 0, 0]] * 3,
        "view_angle_degrees": 0,
        "distance_mm": 0,
        "maximum_absolute_ndc": 2,
        "projection": "historical finite fixture",
    }
    before = deepcopy(value)
    assert validate_receipt(value, primary) == before
    assert value == before
    alternate = opposed_request(primary)
    value.update(version=2, view=OPPOSED_VIEW)
    with pytest.raises(ValueError):
        validate_receipt(value, alternate)


@pytest.mark.parametrize(
    "defect",
    [
        "primary_direction",
        "target",
        "up",
        "distance",
        "angle",
        "orthographic",
        "margin",
        "nonfinite",
        "boolean",
        "bounds",
    ],
)
def test_opposed_receipt_rejects_wrong_actual_camera_despite_correct_label(defect):
    query = opposed_request(request())
    value = receipt(query)
    camera = value["rendering"]["camera"]
    if defect == "primary_direction":
        value["rendering"]["camera"] = worker.fitted_camera(points())
    elif defect == "target":
        for vector in camera["position"][:2]:
            vector[0] += 1
    elif defect == "up":
        camera["position"][2] = [0, 0, 1]
    elif defect == "distance":
        camera["distance_mm"] *= 2
    elif defect == "angle":
        camera["view_angle_degrees"] = 34
    elif defect == "orthographic":
        camera["projection"] = "parallel"
    elif defect == "margin":
        camera["maximum_absolute_ndc"] = 0.9
    elif defect == "nonfinite":
        camera["position"][0][0] = float("nan")
    elif defect == "boolean":
        camera["distance_mm"] = True
    elif defect == "bounds":
        value["rendering"]["geometry"]["bounds_mm"][1][0] += 2
    with pytest.raises(ValueError):
        validate_receipt(value, query)


@pytest.mark.parametrize(
    "size,shift",
    [
        ((80, 50, 6), (0, 0, 0)),
        ((1000, 2, 1), (0, 0, 0)),
        ((2, 1, 1000), (0, 0, 0)),
        ((32, 55, 210), (200000, -80000, 1200)),
    ],
)
@pytest.mark.parametrize("view", [None, OPPOSED_VIEW])
def test_both_cameras_have_independent_positive_depth_and_perspective_margin(size, shift, view):
    vertices = points(size, shift)
    before = vertices.tobytes()
    camera = worker.fitted_camera(vertices, view=view)
    eye, target, up = np.asarray(camera["position"])
    direction = eye - target
    expected = np.array([1.25, -1.65, 1.6] if view is None else [-1.25, 1.65, 1.6])
    assert direction / np.linalg.norm(direction) == pytest.approx(
        expected / np.linalg.norm(expected)
    )
    assert target == pytest.approx(np.asarray(shift))
    forward = -direction / np.linalg.norm(direction)
    right = np.cross(forward, up)
    assert np.linalg.norm(up) == pytest.approx(1)
    assert np.linalg.norm(right) == pytest.approx(1)
    assert np.dot(up, forward) == pytest.approx(0, abs=1e-12)
    offsets = vertices - eye
    depth = offsets @ forward
    assert np.all(depth > 0)
    ndc = np.column_stack((offsets @ right, offsets @ up)) / (
        depth[:, None] * math.tan(math.radians(35 / 2))
    )
    assert np.max(np.abs(ndc)) <= 1 / 1.2 + 1e-10
    assert vertices.tobytes() == before
    if view is None:
        assert worker.fitted_camera(vertices) == camera
    else:
        validate_receipt(receipt(opposed_request(request()), vertices), opposed_request(request()))


@pytest.mark.parametrize(
    "vertices",
    [
        [],
        [[1, 2, 3]],
        [[0, 0, float("inf")], [1, 1, 1]],
        [[-1e308, 0, 0], [1e308, 1, 1]],
        [[1, 2], [3, 4]],
    ],
)
def test_opposed_camera_refuses_invalid_derived_or_source_geometry(vertices):
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        worker.fitted_camera(vertices, view=OPPOSED_VIEW)


def test_unknown_view_refused_before_optional_render_import_or_mesh_parse():
    for function, data in [
        (worker.fitted_camera, points()),
        (worker._render, b"not an STL"),
        (worker._render_triangles, []),
    ]:
        with pytest.raises(ValueError, match="view"):
            if function is worker.fitted_camera:
                function(data, view="arbitrary_camera")
            else:
                function(data, "metal", view="arbitrary_camera")


def png():
    def chunk(kind, content):
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content))
        )

    return (
        worker.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 1024, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\x80" * 3072) * 1024))
        + chunk(b"IEND", b"")
    )


@pytest.mark.parametrize("family", ["mesh", "captured", "assembly"])
@pytest.mark.parametrize("view", [None, OPPOSED_VIEW])
@pytest.mark.parametrize("preset", ["metal", "plastic"])
def test_staged_worker_dispatches_same_bound_source_and_echoes_exact_version(
    tmp_path, monkeypatch, family, view, preset
):
    primary = request(family, preset)
    vertices = points()
    image = png()
    seen = []
    facts = render_facts(vertices, view)
    if family == "assembly":
        from forge_mind.workspace import occurrence_render

        source = json.dumps(
            {
                "topology_counts": {"bodies": 1},
                "pick_tessellation": {
                    "purpose": "saved display",
                    "relative_linear_deflection": 0.01,
                    "angular_deflection_radians": 0.1,
                },
            }
        ).encode()
        primary["source"]["topology_sha256"] = hashlib.sha256(source).hexdigest()
        (tmp_path / "topology.json").write_bytes(source)
        triangles = np.array([vertices[[0, 1, 2]]])

        def selector(topology, indices):
            assert indices == [0]
            assert topology["topology_counts"]["bodies"] == 1
            return SimpleNamespace(triangles=triangles), 1, 0

        monkeypatch.setattr(occurrence_render, "selected_topology_meshes", selector)

        def render(value, chosen_preset, **kwargs):
            assert np.array_equal(value, triangles)
            seen.append((chosen_preset, kwargs))
            return image, deepcopy(facts)

        monkeypatch.setattr(worker, "_render_triangles", render)
    else:
        source = b"staged immutable mesh fixture"
        primary["mesh_sha256"] = hashlib.sha256(source).hexdigest()
        (tmp_path / "part.stl").write_bytes(source)

        def render(value, chosen_preset, **kwargs):
            assert value == source
            seen.append((chosen_preset, kwargs))
            return image, deepcopy(facts)

        monkeypatch.setattr(worker, "_render", render)
    query = primary if view is None else opposed_request(primary)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(query), encoding="utf-8")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = worker.render_product_photo(path)
    assert seen == [(preset, {} if view is None else {"view": OPPOSED_VIEW})]
    assert {key: result[key] for key in query} == query
    assert result["image"]["sha256"] == hashlib.sha256(image).hexdigest()
    assert legacy_request({key: result[key] for key in query}) == primary
    for name, content in before.items():
        assert (tmp_path / name).read_bytes() == content
    assert (tmp_path / "photo.png").read_bytes() == image
    assert json.loads((tmp_path / "receipt.json").read_bytes()) == result


@pytest.mark.parametrize("family", ["mesh", "captured", "assembly"])
@pytest.mark.parametrize("view", [None, OPPOSED_VIEW])
def test_service_stages_correct_source_family_for_each_view(monkeypatch, family, view):
    from threading import Event

    from forge_mind.workspace import product_photo

    source = b"Bound source bytes; the isolated worker is mocked in this test."
    primary = request(family)
    source_hash = hashlib.sha256(source).hexdigest()
    if family == "assembly":
        primary["source"]["topology_sha256"] = source_hash
    else:
        primary["mesh_sha256"] = source_hash
    query = primary if view is None else opposed_request(primary)
    expected_name = "topology.json" if family == "assembly" else "part.stl"
    image = png()
    observed = receipt(query)
    observed["image"]["sha256"] = hashlib.sha256(image).hexdigest()
    directories = []

    def execute(command, directory, timeout, cancelled, **options):
        directories.append(directory)
        assert not cancelled() and 0 < timeout <= 5
        assert {p.name for p in directory.iterdir()} == {expected_name, "request.json"}
        assert (directory / expected_name).read_bytes() == source
        assert json.loads((directory / "request.json").read_bytes()) == query
        (directory / "photo.png").write_bytes(image)
        (directory / "receipt.json").write_text(json.dumps(observed))

    monkeypatch.setattr(product_photo, "bounded_process", execute)
    service = SimpleNamespace(executor=SimpleNamespace(mode="trusted_local"), stop=Event())
    result = product_photo.render_mesh(service, source, query, timeout=5)
    assert result["receipt"] == observed
    assert len(directories) == 1 and not directories[0].exists()
