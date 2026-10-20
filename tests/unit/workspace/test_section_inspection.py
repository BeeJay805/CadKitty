"""Section ownership/provenance controls with saved bytes and mocked workers only."""

from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from forge_mind.workspace import section_inspection as inspection
from forge_mind.workspace import section_render as render

SID, RID = "a" * 32, "b" * 32
PNG = render.PNG_SIGNATURE + b"controlled PNG payload; geometric image proof is separate"


def fixture_service(tmp_path, monkeypatch, *, mode="trusted_local", qualified=False):
    directory = tmp_path / SID / RID
    directory.mkdir(parents=True)
    contents = {
        "part.stl": b"saved mesh fixture; never parsed",
        "document.json": b'{"name":"fixture"}',
    }
    for name, content in contents.items():
        (directory / name).write_bytes(content)
    measurement = {
        "revision_id": RID,
        "artifact_hashes": {
            name: hashlib.sha256(data).hexdigest() for name, data in contents.items()
        },
        "geometry_valid": True,
    }
    record = {"id": RID, "measurement": measurement}
    calls = {"owner": [], "revision": [], "worker": []}

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
        assert not cancelled()
        assert 0 < timeout <= 60
        request = json.loads((scratch / "request.json").read_bytes())
        assert (scratch / "part.stl").read_bytes() == contents["part.stl"]
        assert (scratch / "document.json").read_bytes() == contents["document.json"]
        write_result(scratch, request)

    monkeypatch.setattr(inspection, "bounded_process", worker)
    return service, directory, contents, record, calls, worker


def write_result(scratch, request, *, empty=False, mutate=None):
    receipt = {
        **request,
        "section_status": "empty" if empty else "intersected",
        "polyline_count": 0 if empty else 2,
        "image": None
        if empty
        else {"file": "section.png", "sha256": hashlib.sha256(PNG).hexdigest()},
        "scope": render.SCOPE,
    }
    if not empty:
        (scratch / "section.png").write_bytes(PNG)
    if mutate:
        mutate(receipt, scratch)
    (scratch / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


@pytest.mark.parametrize("axis", ["x", "y", "z"])
def test_world_plane_section_binds_source_and_rechecks_owner_without_mutation(
    tmp_path, monkeypatch, axis
):
    service, directory, contents, record, calls, _ = fixture_service(tmp_path, monkeypatch)
    before = deepcopy(record)
    result = inspection.inspect_section(service, SID, "owner", RID, axis, -12.5)
    receipt = result["receipt"]
    assert receipt["axis"] == axis and receipt["offset_mm"] == -12.5
    assert receipt["frame"] == "cad_z_up_mm_world"
    assert receipt["project_id"] == SID and receipt["revision_id"] == RID
    assert receipt["mesh_sha256"] == hashlib.sha256(contents["part.stl"]).hexdigest()
    assert receipt["document_sha256"] == hashlib.sha256(contents["document.json"]).hexdigest()
    assert receipt["polyline_count"] == 2 and receipt["section_status"] == "intersected"
    assert result["mime_type"] == "image/png"
    assert "not verified closed cavities" in receipt["scope"]
    assert calls["owner"] == [(SID, "owner"), (SID, "owner")]
    assert len(calls["worker"]) == 1
    assert record == before
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == contents


def test_empty_section_is_truthful_and_has_no_image(tmp_path, monkeypatch):
    service, _, _, _, _, _ = fixture_service(tmp_path, monkeypatch)

    def empty(command, scratch, *args, **kwargs):
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()), empty=True)

    monkeypatch.setattr(inspection, "bounded_process", empty)
    result = inspection.inspect_section(service, SID, "owner", RID, "z", 1000)
    assert result["receipt"]["section_status"] == "empty"
    assert result["receipt"]["polyline_count"] == 0
    assert result["receipt"]["image"] is None
    assert result["image_base64"] is None and result["mime_type"] is None


def test_foreign_owner_is_rejected_before_revision_or_worker(tmp_path, monkeypatch):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Foreign owner"):
        inspection.inspect_section(service, SID, "other", RID, "x", 0)
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize(
    "axis,offset,timeout",
    [
        ("q", 0, 30),
        ("X", 0, 30),
        ("x", True, 30),
        ("x", "0", 30),
        ("x", float("nan"), 30),
        ("x", float("inf"), 30),
        ("x", 10**400, 30),
        ("x", 0, True),
        ("x", 0, 0),
        ("x", 0, 61),
        ("x", 0, float("inf")),
    ],
)
def test_invalid_plane_or_deadline_never_reads_revision(
    tmp_path, monkeypatch, axis, offset, timeout
):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        inspection.inspect_section(service, SID, "owner", RID, axis, offset, timeout=timeout)
    assert calls["revision"] == calls["worker"] == []


@pytest.mark.parametrize("mode", ["docker", "unqualified_mode"])
def test_unqualified_worker_refuses_before_source_loading(tmp_path, monkeypatch, mode):
    service, _, _, _, calls, _ = fixture_service(tmp_path, monkeypatch, mode=mode)
    monkeypatch.setattr(
        inspection, "_artifact", lambda *args: pytest.fail("Source must not be loaded")
    )
    with pytest.raises(ValueError, match=r"qualified|Unsupported"):
        inspection.inspect_section(service, SID, "owner", RID, "x", 0)
    assert calls["worker"] == []


def test_qualified_container_receives_staged_sources_and_immutable_image(tmp_path, monkeypatch):
    service, _, _, _, calls, worker = fixture_service(
        tmp_path, monkeypatch, mode="docker", qualified=True
    )
    monkeypatch.setattr(inspection.shutil, "which", lambda name: "fixture-docker")

    def run_container(docker, scratch, module, request, timeout, cancelled, *, image):
        assert docker == "fixture-docker" and module == "forge_mind.workspace.section_render"
        assert request == "request.json" and image == "qualified@sha256:fixture"
        worker([module], scratch, timeout, cancelled)

    service.executor._run_container = run_container
    monkeypatch.setattr(
        inspection,
        "bounded_process",
        lambda *args, **kwargs: pytest.fail("No host worker"),
    )
    assert inspection.inspect_section(service, SID, "owner", RID, "y", 2)["receipt"]["axis"] == "y"
    assert len(calls["worker"]) == 1


@pytest.mark.parametrize("change", ["source", "measurement", "owner", "cancel", "deadline"])
def test_concurrent_changes_cannot_publish_section(tmp_path, monkeypatch, change):
    service, directory, _, record, _, worker = fixture_service(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(inspection.time, "monotonic", lambda: clock[0])

    def changed(*args, **kwargs):
        worker(*args, **kwargs)
        if change == "source":
            (directory / "part.stl").write_bytes(b"changed")
        elif change == "measurement":
            record["measurement"]["geometry_valid"] = False
        elif change == "owner":
            service.store.session_record = lambda *args: (_ for _ in ()).throw(
                ValueError("Owner changed")
            )
        elif change == "cancel":
            service.stop.set()
        else:
            clock[0] = 31

    monkeypatch.setattr(inspection, "bounded_process", changed)
    with pytest.raises(ValueError):
        inspection.inspect_section(service, SID, "owner", RID, "x", 0)


@pytest.mark.parametrize(
    "change",
    [
        "mesh_sha256",
        "document_sha256",
        "project_id",
        "revision_id",
        "axis",
        "offset_mm",
        "frame",
        "scope",
        "count",
        "image_hash",
        "wrong_status",
        "unknown_field",
        "oversize_image",
        "not_png",
        "empty_image",
    ],
)
def test_worker_output_mismatch_is_refused(tmp_path, monkeypatch, change):
    service, _, _, _, _, _ = fixture_service(tmp_path, monkeypatch)

    def corrupt(receipt, scratch):
        if change in {
            "mesh_sha256",
            "document_sha256",
            "project_id",
            "revision_id",
            "axis",
            "frame",
            "scope",
        }:
            receipt[change] = "changed"
        elif change == "offset_mm":
            receipt[change] += 1
        elif change == "count":
            receipt["polyline_count"] = True
        elif change == "image_hash":
            receipt["image"]["sha256"] = "0" * 64
        elif change == "wrong_status":
            receipt["section_status"] = "empty"
        elif change == "unknown_field":
            receipt["approval"] = True
        elif change == "oversize_image":
            (scratch / "section.png").write_bytes(b"x" * (render.MAX_IMAGE_BYTES + 1))
        elif change == "not_png":
            (scratch / "section.png").write_bytes(b"not a PNG")
        else:
            receipt.update(section_status="empty", polyline_count=0, image=None)

    def bad(command, scratch, *args, **kwargs):
        write_result(scratch, json.loads((scratch / "request.json").read_bytes()), mutate=corrupt)

    monkeypatch.setattr(inspection, "bounded_process", bad)
    with pytest.raises(ValueError):
        inspection.inspect_section(service, SID, "owner", RID, "z", 0)


@pytest.mark.parametrize("offset,boolean_offset", [(0.0, False), (1.0, True)])
def test_boolean_worker_plane_cannot_match_numeric_request(
    tmp_path, monkeypatch, offset, boolean_offset
):
    service, _, _, _, _, _ = fixture_service(tmp_path, monkeypatch)

    def invalid_plane(command, scratch, *args, **kwargs):
        request = json.loads((scratch / "request.json").read_bytes())
        write_result(
            scratch,
            request,
            mutate=lambda receipt, _: receipt.__setitem__("offset_mm", boolean_offset),
        )

    monkeypatch.setattr(inspection, "bounded_process", invalid_plane)
    with pytest.raises(ValueError, match="finite numeric offset"):
        inspection.inspect_section(service, SID, "owner", RID, "z", offset)


def test_source_read_is_bounded_before_open_and_after_growth(tmp_path, monkeypatch):
    path = tmp_path / "bounded.bin"
    path.write_bytes(b"x" * 5)
    original_open = Path.open
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("Oversize opened"))
    with pytest.raises(ValueError, match="oversized"):
        render.read_bytes(path, 4)
    monkeypatch.setattr(Path, "open", original_open)
    sizes = []

    class Growing(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: Growing(b"y" * 20))
    with pytest.raises(ValueError, match="byte budget"):
        render.read_bytes(path, 5)
    assert sizes == [6]


def test_linked_source_and_changed_source_are_refused(tmp_path, monkeypatch):
    service, directory, _, _, calls, _ = fixture_service(tmp_path, monkeypatch)
    (tmp_path / "linked.stl").hardlink_to(directory / "part.stl")
    with pytest.raises(ValueError, match="linked"):
        inspection.inspect_section(service, SID, "owner", RID, "x", 0)
    assert calls["worker"] == []
    (tmp_path / "linked.stl").unlink()
    (directory / "document.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale"):
        inspection.inspect_section(service, SID, "owner", RID, "x", 0)
    assert calls["worker"] == []


@pytest.mark.parametrize("content", [b'{"a":1,"a":2}', b'{"offset_mm":NaN}', b"[]"])
def test_worker_json_refuses_ambiguous_or_nonfinite_objects(content):
    with pytest.raises(ValueError):
        render.read_object(content)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize(
    "requested_offset,observed_offset", [(-3.0, -3.0), (0.0, False), (1.0, True)]
)
def test_worker_reuses_existing_section_method_and_binds_output(
    tmp_path, monkeypatch, empty, requested_offset, observed_offset
):
    from forge_mind.workspace import measure

    contents = {"part.stl": b"mesh fixture", "document.json": b"{}"}
    request = {
        "project_id": SID,
        "revision_id": RID,
        "axis": "y",
        "offset_mm": requested_offset,
        "frame": render.FRAME,
        "mesh_sha256": hashlib.sha256(contents["part.stl"]).hexdigest(),
        "document_sha256": hashlib.sha256(contents["document.json"]).hexdigest(),
    }
    for name, content in contents.items():
        (tmp_path / name).write_bytes(content)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    def section(directory, axis, offset, target):
        assert directory == tmp_path and axis == "y" and offset == requested_offset
        if empty:
            return {"axis": axis, "offset_mm": observed_offset, "loops": 0}
        target.write_bytes(PNG)
        return {"axis": axis, "offset_mm": observed_offset, "loops": 2, "image": target.name}

    monkeypatch.setattr(measure, "section_revision", section)
    if type(observed_offset) is bool:
        with pytest.raises(ValueError, match="finite numeric offset"):
            render.render_section(request_path)
        assert not (tmp_path / "receipt.json").exists()
        return
    receipt = render.render_section(request_path)
    assert receipt["section_status"] == ("empty" if empty else "intersected")
    assert receipt["polyline_count"] == (0 if empty else 2)
    assert json.loads((tmp_path / "receipt.json").read_bytes()) == receipt
    assert (tmp_path / "section.png").exists() is not empty
