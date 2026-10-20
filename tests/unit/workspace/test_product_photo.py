"""Owned saved-byte product-photo plumbing with mocked workers, never native rendering."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from forge_mind.workspace import product_photo as photo
from forge_mind.workspace import product_photo_contract as contract

SID, RID = "a" * 32, "b" * 32


def png(width=1024, height=1024):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = zlib.compress((b"\0" + b"\0" * (width * 3)) * height)
    return (
        photo.PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


PNG = png()


def write_result(scratch, request, *, image=PNG, mutate=None):
    observed = {
        **request,
        "scope": contract.SCOPE,
        "image": {
            "file": "photo.png",
            "sha256": hashlib.sha256(image).hexdigest(),
            "width": 1024,
            "height": 1024,
        },
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
                "triangles": 5,
                "triangle_sha256": "c" * 64,
                "triangle_positions_unchanged": True,
                "bounds_mm": [[-40, -25, -3], [40, 25, 3]],
            },
        },
    }
    (scratch / "photo.png").write_bytes(image)
    if mutate:
        mutate(observed, scratch)
    (scratch / "receipt.json").write_text(json.dumps(observed), encoding="utf-8")


def fixture_service(tmp_path, monkeypatch, *, mode="trusted_local", qualified=False):
    directory = tmp_path / SID / RID
    directory.mkdir(parents=True)
    mesh = b"Complete saved single/assembly mesh fixture; not parsed by mock"
    (directory / "part.stl").write_bytes(mesh)
    # No document, STEP or assembly cache is needed for a uniform saved-mesh photo.
    measurement = {
        "revision_id": RID,
        "artifact_hashes": {"part.stl": hashlib.sha256(mesh).hexdigest()},
        "geometry_valid": False,  # Rendering does not add a geometry acceptance gate.
    }
    record = {"id": RID, "measurement": measurement}
    calls = {"owner": [], "revision": [], "worker": [], "scratch": []}

    def owned(sid, owner):
        calls["owner"].append((sid, owner))
        if owner != "owner":
            raise ValueError("Foreign owner")
        return {"id": sid}

    def revision(sid, rid):
        calls["revision"].append((sid, rid))
        return record

    service = SimpleNamespace(
        store=SimpleNamespace(
            session_record=owned, revision=revision, directory=lambda *args: directory
        ),
        executor=SimpleNamespace(
            _docker_path=lambda discover: discover("docker"),
            mode=mode,
            typed_container_verified=qualified,
            typed_container_image="qualified@sha256:fixture",
        ),
        stop=Event(),
    )

    def worker(command, scratch, timeout, cancelled, **kwargs):
        calls["worker"].append((command, timeout, kwargs))
        calls["scratch"].append(scratch)
        assert not cancelled()
        assert 0 < timeout <= 60
        assert (scratch / "part.stl").read_bytes() == mesh
        assert not (scratch / "document.json").exists()
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()))

    monkeypatch.setattr(photo, "bounded_process", worker)
    return service, directory, mesh, record, calls, worker


@pytest.mark.parametrize("preset", contract.PRESETS)
def test_binds_complete_saved_mesh_rechecks_owner_and_preserves_records(
    tmp_path, monkeypatch, preset
):
    service, directory, mesh, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    result = photo.get_product_photo(service, SID, "owner", RID, preset)
    assert result["receipt"]["preset"] == preset
    assert result["receipt"]["mesh_sha256"] == hashlib.sha256(mesh).hexdigest()
    assert result["receipt"]["frame"] == contract.FRAME
    assert result["receipt"]["scope"] == contract.SCOPE
    assert result["mime_type"] == "image/png"
    assert base64.b64decode(result["image_base64"]) == PNG
    assert calls["owner"] == [(SID, "owner"), (SID, "owner")]
    assert len(calls["worker"]) == 1
    assert record == before
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == {"part.stl": mesh}
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_foreign_owner_refuses_before_source_or_worker(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    monkeypatch.setattr(photo, "_artifact", lambda *args: pytest.fail("No source read"))
    with pytest.raises(ValueError, match="Foreign"):
        photo.get_product_photo(service, SID, "foreign", RID, "metal")
    assert calls["revision"] == calls["worker"] == []


def test_optional_renderer_absence_preserves_standalone_value_error(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def unavailable(*args, **kwargs):
        raise contract.PhotoUnavailableError("renderer_unavailable", "Graphics unavailable")

    monkeypatch.setattr(photo, "render_mesh", unavailable)
    with pytest.raises(ValueError, match="Graphics unavailable") as failure:
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert isinstance(failure.value.__cause__, contract.PhotoUnavailableError)
    assert calls["worker"] == []


@pytest.mark.parametrize(
    "preset,timeout",
    [("gold", 30), ("metal", True), ("metal", 0), ("metal", 61), ("metal", float("nan"))],
)
def test_invalid_preset_or_timeout_refuses_before_revision(tmp_path, monkeypatch, preset, timeout):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        photo.get_product_photo(service, SID, "owner", RID, preset, timeout=timeout)
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("mode", ["docker", "unknown"])
def test_unqualified_mode_refuses_before_loading_source(tmp_path, monkeypatch, mode):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch, mode=mode)
    monkeypatch.setattr(photo, "_artifact", lambda *args: pytest.fail("No source read"))
    with pytest.raises(ValueError, match=r"qualified|Unsupported"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert calls["worker"] == []


def test_qualified_container_uses_exact_image_and_mesh_only(tmp_path, monkeypatch):
    service, _, _, _, calls, worker = fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )
    monkeypatch.setattr(photo.shutil, "which", lambda name: "fixture-docker")

    def container(docker, scratch, module, request, timeout, cancelled, *, image):
        assert docker == "fixture-docker"
        assert module == "forge_mind.workspace.product_photo_render"
        assert request == "request.json" and image == "qualified@sha256:fixture"
        worker([module], scratch, timeout, cancelled)

    service.executor._run_container = container
    monkeypatch.setattr(
        photo, "bounded_process", lambda *args, **kwargs: pytest.fail("No host render")
    )
    assert (
        photo.get_product_photo(service, SID, "owner", RID, "plastic")["receipt"]["preset"]
        == "plastic"
    )
    assert len(calls["worker"]) == 1


@pytest.mark.parametrize("change", ["source", "measurement", "owner", "cancel", "deadline"])
def test_post_worker_changes_refuse_and_cleanup(tmp_path, monkeypatch, change):
    service, directory, _, record, calls, worker = fixture_service(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(photo.time, "monotonic", lambda: clock[0])

    def changed(*args, **kwargs):
        worker(*args, **kwargs)
        if change == "source":
            (directory / "part.stl").write_bytes(b"changed")
        elif change == "measurement":
            record["measurement"]["geometry_valid"] = True
        elif change == "owner":
            service.store.session_record = lambda *args: (_ for _ in ()).throw(
                ValueError("Owner changed")
            )
        elif change == "cancel":
            service.stop.set()
        else:
            clock[0] = 31

    monkeypatch.setattr(photo, "bounded_process", changed)
    with pytest.raises(ValueError):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert all(not scratch.exists() for scratch in calls["scratch"])


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "scope",
        "version",
        "geometry",
        "wrong_dimensions",
        "oversize",
        "missing",
        "receipt_nonfinite",
        "not_png",
    ],
)
def test_bad_worker_output_cannot_publish(tmp_path, monkeypatch, change):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def mutate(receipt, scratch):
        if change == "hash":
            receipt["image"]["sha256"] = "0" * 64
        elif change == "scope":
            receipt["scope"] = "Physical proof"
        elif change == "version":
            receipt["version"] = True
        elif change == "geometry":
            receipt["rendering"]["geometry"]["triangle_positions_unchanged"] = False
        elif change == "oversize":
            (scratch / "photo.png").write_bytes(b"x" * (contract.MAX_IMAGE_BYTES + 1))
        elif change == "missing":
            (scratch / "photo.png").unlink()
        elif change == "receipt_nonfinite":
            receipt["rendering"]["bad_number"] = float("inf")
        elif change == "not_png":
            bad_image = b"x" * 40
            (scratch / "photo.png").write_bytes(bad_image)
            receipt["image"]["sha256"] = hashlib.sha256(bad_image).hexdigest()

    def bad(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        image = png(512, 1024) if change == "wrong_dimensions" else PNG
        write_result(
            scratch, json.loads((scratch / "request.json").read_bytes()), image=image, mutate=mutate
        )

    monkeypatch.setattr(photo, "bounded_process", bad)
    with pytest.raises((ValueError, OSError)):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert all(not scratch.exists() for scratch in calls["scratch"])


@pytest.mark.parametrize(
    "error",
    [ValueError("Product photo unavailable: PyVista missing"), TimeoutError("worker deadline")],
)
def test_worker_errors_remain_explicit_without_preview_fallback(tmp_path, monkeypatch, error):
    service, directory, mesh, _, calls, _ = fixture_service(tmp_path, monkeypatch)

    def unavailable(command, scratch, *args, **kwargs):
        calls["scratch"].append(scratch)
        raise error

    monkeypatch.setattr(photo, "bounded_process", unavailable)
    with pytest.raises(type(error), match=str(error)):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == {"part.stl": mesh}
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_stale_source_missing_checksum_and_wrong_revision_refuse(tmp_path, monkeypatch):
    service, directory, _, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    (directory / "part.stl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="stale"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    record["measurement"]["artifact_hashes"].clear()
    with pytest.raises(ValueError, match="checksum"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    record["measurement"]["revision_id"] = "f" * 32
    with pytest.raises(ValueError, match="revision"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert calls["worker"] == []


def test_source_size_refuses_before_open_and_cancel_refuses_before_revision(tmp_path, monkeypatch):
    service, directory, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    monkeypatch.setattr(photo, "MAX_INPUT_BYTES", 1)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == directory / "part.stl":
            pytest.fail("Oversized source must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="oversized"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    calls["revision"].clear()
    service.stop.set()
    with pytest.raises(ValueError, match="cancelled"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert calls["revision"] == calls["worker"] == []


def test_response_budget_refuses_without_persistent_output(tmp_path, monkeypatch):
    service, _, _, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    monkeypatch.setattr(photo, "MAX_RESPONSE_BYTES", 1)
    with pytest.raises(ValueError, match="response byte"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert record == before
    assert all(not scratch.exists() for scratch in calls["scratch"])


def test_linked_source_cannot_be_read(tmp_path, monkeypatch):
    service, directory, mesh, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    source = directory / "part.stl"
    source.unlink()
    external = tmp_path / "external.stl"
    external.write_bytes(mesh)
    try:
        source.symlink_to(external)
    except OSError:
        pytest.skip("Creating symlinks is not available on this test host")
    with pytest.raises(ValueError, match="linked"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert calls["worker"] == []


def test_source_growth_after_stat_is_bounded(tmp_path, monkeypatch):
    service, directory, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    source = directory / "part.stl"
    monkeypatch.setattr(photo, "MAX_INPUT_BYTES", source.stat().st_size)
    original_open = Path.open

    def growth(path, *args, **kwargs):
        if path == source:
            with original_open(path, "ab") as writer:
                writer.write(b"growth beyond the checked size")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", growth)
    with pytest.raises(ValueError, match="oversized"):
        photo.get_product_photo(service, SID, "owner", RID, "metal")
    assert calls["worker"] == []
