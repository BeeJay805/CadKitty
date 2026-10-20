"""Captured photo tests use actual package validation and mocked graphics only."""

from __future__ import annotations

import base64
import json
from copy import deepcopy

import pytest
from tests.unit.workspace.test_product_photo import PNG, write_result
from tests.unit.workspace.test_project_delivery import RID, saved_project

from forge_mind.workspace import executor as execution
from forge_mind.workspace import product_photo as rendering
from forge_mind.workspace import product_photo_contract as contract
from forge_mind.workspace import project_delivery as delivery
from forge_mind.workspace import project_photo as photo


def captured(tmp_path, monkeypatch, *, include_stl=True):
    service, sid, source = saved_project(tmp_path, monkeypatch)
    service.executor.mode = "trusted_local"
    package = delivery.create_project_export(service, sid, "owner", RID, include_stl=include_stl)
    calls = []

    def worker(command, scratch, timeout, cancelled, **kwargs):
        assert not cancelled()
        assert 0 < timeout <= 15
        request = json.loads((scratch / "request.json").read_bytes())
        content, receipt = delivery.read_project_export(
            service, sid, "owner", package["package_id"], package["sha256"]
        )
        expected = photo.read_captured_cad_file(content, receipt, request["file_id"])
        assert (scratch / "part.stl").read_bytes() == expected
        assert {p.name for p in scratch.iterdir()} == {"part.stl", "request.json"}
        calls.append(deepcopy(request))
        write_result(
            scratch, request, mutate=lambda value, _: value.update(scope=contract.CAPTURED_SCOPE)
        )

    monkeypatch.setattr(rendering, "bounded_process", worker)
    return service, sid, source, package, calls, worker


def create(service, sid, package, **kwargs):
    return photo.create_project_photo(
        service, sid, "owner", package["package_id"], package["sha256"], **kwargs
    )


def read(service, sid, package, result, *, owner="owner"):
    return photo.read_project_photo(
        service,
        sid,
        owner,
        package["package_id"],
        package["sha256"],
        result["photo_id"],
        result["receipt"]["image"]["sha256"],
    )


def test_explicit_member_exact_bytes_durable_owner_bound_and_no_source_mutation(
    tmp_path, monkeypatch
):
    service, sid, source, package, calls, _ = captured(tmp_path, monkeypatch)
    source_before = {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    package_before = deepcopy(package)
    original, _ = delivery.read_project_export(
        service, sid, "owner", package["package_id"], package["sha256"]
    )
    result = create(service, sid, package, part_id="cap", preset="metal")
    assert result["status"] == "available"
    assert result["receipt"]["part_id"] == "cap"
    assert result["receipt"]["quantity"] == 2
    assert result["receipt"]["frame"] == "definition_local"
    assert result["receipt"]["package_sha256"] == package["sha256"]
    assert result["receipt"]["scope"] == contract.CAPTURED_SCOPE
    assert calls == [{key: result["receipt"][key] for key in contract.CAPTURED_REQUEST_KEYS}]
    assert base64.b64decode(result["image_base64"]) == PNG
    service.store.update(sid, selected_revision="c" * 32)
    data, receipt = read(service, sid, package, result)
    assert data == PNG and receipt == result["receipt"]
    assert package == package_before
    assert source_before == {p.name: p.read_bytes() for p in source.iterdir() if p.is_file()}
    after, _ = delivery.read_project_export(
        service, sid, "owner", package["package_id"], package["sha256"]
    )
    assert after == original
    photo.discard_project_photo(service, sid, "owner", package["package_id"], result["photo_id"])
    assert not photo._directory(service, sid, package["package_id"], result["photo_id"]).exists()
    assert (
        delivery.read_project_export(
            service, sid, "owner", package["package_id"], package["sha256"]
        )[0]
        == original
    )


def test_structured_auto_cannot_fall_back_to_one_projected_definition(tmp_path, monkeypatch):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch)
    inventory = photo.captured_cad_files
    # A projected member list cannot replace the saved assembly source for automatic photos.
    monkeypatch.setattr(
        photo,
        "captured_cad_files",
        lambda c, r: [row for row in inventory(c, r) if row["part_id"] != "cap"],
    )
    result = create(service, sid, package)
    assert result["status"] == "unavailable" and result["code"] == "source_unavailable"
    assert not calls


@pytest.mark.parametrize(
    "stl,part,code", [(True, None, "source_unavailable"), (False, "base", "missing_stl")]
)
def test_optional_unavailable_preserves_real_package_and_skips_worker(
    tmp_path, monkeypatch, stl, part, code
):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch, include_stl=stl)
    result = create(service, sid, package, part_id=part)
    assert result["status"] == "unavailable" and result["code"] == code
    assert result["candidates"] == ["base", "cap"]
    assert not calls
    assert (
        delivery.read_project_export(
            service, sid, "owner", package["package_id"], package["sha256"]
        )[1]
        == package
    )


@pytest.mark.parametrize("part", ["guide", "bolt", "missing", "../base", True])
def test_invalid_or_nonmake_explicit_selection_refuses_without_render(tmp_path, monkeypatch, part):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        create(service, sid, package, part_id=part)
    assert not calls


@pytest.mark.parametrize("mutation", ["foreign", "hash", "package", "cancel"])
def test_source_and_authority_failures_cannot_degrade_into_unavailable(
    tmp_path, monkeypatch, mutation
):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch)
    owner = "owner"
    if mutation == "foreign":
        owner = "foreign"
    elif mutation == "hash":
        package["sha256"] = "0" * 64
    elif mutation == "package":
        (
            service.store.root / "project-exports" / sid / package["package_id"] / "project.zip"
        ).write_bytes(b"tampered")
    else:
        service.stop.set()
    with pytest.raises((ValueError, OSError)):
        photo.create_project_photo(
            service, sid, owner, package["package_id"], package["sha256"], part_id="base"
        )
    assert not calls


@pytest.mark.parametrize("code", ["renderer_unavailable", "render_failed", "execution_unqualified"])
def test_only_typed_optional_graphics_failures_are_unavailable(tmp_path, monkeypatch, code):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def absent(*args, **kwargs):
        raise contract.PhotoUnavailableError(code, "Optional graphics unavailable")

    monkeypatch.setattr(photo, "render_mesh", absent)
    result = create(service, sid, package, part_id="base")
    assert result == {
        "status": "unavailable",
        "code": code,
        "reason": "Optional graphics unavailable",
        "candidates": ["base", "cap"],
    }


@pytest.mark.parametrize(
    "error",
    [
        ValueError("source invalid"),
        RuntimeError("unclassified worker error"),
        TimeoutError("worker stopped"),
        MemoryError("allocation"),
    ],
)
def test_unclassified_errors_do_not_become_optional_success(tmp_path, monkeypatch, error):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def failed(*args, **kwargs):
        raise error

    monkeypatch.setattr(photo, "render_mesh", failed)
    with pytest.raises(type(error), match=str(error)):
        create(service, sid, package, part_id="base")


@pytest.mark.parametrize("phase", ["available", "unavailable"])
def test_owner_or_cancellation_change_after_worker_refuses(tmp_path, monkeypatch, phase):
    service, sid, _, package, _, worker = captured(tmp_path, monkeypatch)

    def cancelled(command, scratch, *args, **kwargs):
        worker(command, scratch, *args, **kwargs)
        service.stop.set()
        if phase == "unavailable":
            raise contract.PhotoUnavailableError("renderer_unavailable", "Absent")

    monkeypatch.setattr(rendering, "bounded_process", cancelled)
    with pytest.raises(ValueError, match="cancelled"):
        create(service, sid, package, part_id="base")
    assert not list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))


@pytest.mark.parametrize(
    "mutation", ["image", "receipt", "wrong_photo_id", "foreign", "mesh_hash", "quantity_bool"]
)
def test_durable_read_rejects_tampered_or_foreign_photos(tmp_path, monkeypatch, mutation):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    result = create(service, sid, package, part_id="base")
    directory = photo._directory(service, sid, package["package_id"], result["photo_id"])
    owner = "owner"
    if mutation == "image":
        (directory / "photo.png").write_bytes(PNG + b"tamper")
    elif mutation == "foreign":
        owner = "foreign"
    else:
        value = json.loads((directory / "receipt.json").read_bytes())
        if mutation == "receipt":
            value["receipt"]["package_sha256"] = "0" * 64
        elif mutation == "wrong_photo_id":
            value["photo_id"] = "0" * 32
        elif mutation == "mesh_hash":
            value["receipt"]["mesh_sha256"] = "0" * 64
        else:
            value["receipt"]["quantity"] = True
        (directory / "receipt.json").write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, OSError)):
        read(service, sid, package, result, owner=owner)


def test_cancel_during_publication_cleans_only_created_photo(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    original = photo.read_project_export
    calls = 0

    def check(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = original(*args, **kwargs)
        if calls == 3:
            service.stop.set()
        return value

    monkeypatch.setattr(photo, "read_project_export", check)
    with pytest.raises(ValueError, match="cancelled"):
        create(service, sid, package, part_id="base")
    assert not list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))
    assert original(service, sid, "owner", package["package_id"], package["sha256"])[1] == package


@pytest.mark.parametrize(
    "mutation",
    [
        "changed_request",
        "changed_staged_mesh",
        "unclassified_code",
        "contradictory",
        "boolean_quantity",
    ],
)
def test_worker_optional_failure_data_is_strictly_bound(tmp_path, monkeypatch, mutation):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def worker(command, scratch, *args, **kwargs):
        request = json.loads((scratch / "request.json").read_bytes())
        error = {"request": deepcopy(request), "code": "renderer_unavailable", "reason": "Absent"}
        if mutation == "changed_request":
            error["request"]["package_sha256"] = "0" * 64
        elif mutation == "changed_staged_mesh":
            (scratch / "part.stl").write_bytes(b"changed")
        elif mutation == "unclassified_code":
            error["code"] = "tampered"
        elif mutation == "contradictory":
            (scratch / "photo.png").write_bytes(PNG)
        else:
            error["request"]["quantity"] = True
        (scratch / "unavailable.json").write_text(json.dumps(error), encoding="utf-8")

    monkeypatch.setattr(rendering, "bounded_process", worker)
    with pytest.raises(ValueError):
        create(service, sid, package, part_id="base")


def test_captured_contract_rejects_other_frames_and_preserves_legacy_requests():
    legacy = {
        "project_id": "a" * 32,
        "revision_id": "b" * 32,
        "mesh_sha256": "c" * 64,
        "frame": contract.FRAME,
        "preset": "plastic",
        "version": 1,
    }
    before = deepcopy(legacy)
    assert contract.validated_request(legacy) == legacy == before
    local = {
        **legacy,
        "frame": "definition_local",
        "package_id": "d" * 32,
        "package_sha256": "e" * 64,
        "part_id": "knob",
        "file_id": "part-knob-stl",
        "quantity": 1,
    }
    assert contract.validated_request(local) == local
    for field, value in [
        ("frame", contract.FRAME),
        ("quantity", True),
        ("file_id", "assembly-stl"),
        ("package_sha256", "bad"),
    ]:
        with pytest.raises(ValueError):
            contract.validated_request({**local, field: value})


@pytest.mark.parametrize("change", ["timeout", "cancel", "timeout_and_tamper"])
def test_actual_watchdog_callback_distinguishes_expiry_from_cancel_and_tamper(
    tmp_path, monkeypatch, change
):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(rendering.time, "monotonic", lambda: clock[0])
    staged = []

    def watchdog(command, scratch, timeout, cancelled, **kwargs):
        staged.append(scratch)
        clock[0] = 16.0
        if change == "cancel":
            service.stop.set()
        elif change == "timeout_and_tamper":
            (scratch / "part.stl").write_bytes(b"tampered")
        assert cancelled() is True
        # This is the real generic watchdog exception; its text is not classified.
        raise ValueError("Operation cancelled")

    monkeypatch.setattr(rendering, "bounded_process", watchdog)
    if change == "timeout":
        result = create(service, sid, package, part_id="base")
        assert result["status"] == "unavailable" and result["code"] == "render_failed"
    else:
        with pytest.raises(ValueError, match=r"cancelled|changed"):
            create(service, sid, package, part_id="base")
    assert all(not path.exists() for path in staged)
    assert not list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))


def test_elapsed_capture_budget_skips_render_with_explicit_optional_failure(tmp_path, monkeypatch):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(photo.time, "monotonic", lambda: clock[0])
    original = photo.read_captured_cad_file

    def elapsed(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 16.0
        return value

    monkeypatch.setattr(photo, "read_captured_cad_file", elapsed)
    result = create(service, sid, package, part_id="base")
    assert result["status"] == "unavailable" and result["code"] == "render_failed"
    assert calls == []


def test_foreign_owner_after_optional_failure_still_refuses(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def absent(*args, **kwargs):
        service.store.session_record = lambda *a: (_ for _ in ()).throw(ValueError("Owner changed"))
        raise contract.PhotoUnavailableError("renderer_unavailable", "Absent")

    monkeypatch.setattr(photo, "render_mesh", absent)
    with pytest.raises(ValueError, match="Owner changed"):
        create(service, sid, package, part_id="base")


def test_no_make_and_duplicate_target_are_not_photographed():
    row = {"role": "make", "part_id": "knob", "format": "stl"}
    assert photo._target([], None)[1]["code"] == "no_made_parts"
    with pytest.raises(ValueError, match="duplicate"):
        photo._target([row, dict(row)], "knob")


def test_unqualified_execution_preserves_captured_files_without_worker(tmp_path, monkeypatch):
    service, sid, _, package, calls, _ = captured(tmp_path, monkeypatch)
    service.executor.mode = "docker"
    result = create(service, sid, package, part_id="base")
    assert result["code"] == "execution_unqualified" and result["status"] == "unavailable"
    assert calls == []
    assert (
        delivery.read_project_export(
            service, sid, "owner", package["package_id"], package["sha256"]
        )[1]
        == package
    )


def test_linked_photo_directory_refuses_read_and_cleanup(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    result = create(service, sid, package, part_id="base")
    original = photo._directory(service, sid, package["package_id"], result["photo_id"])
    displaced = original.with_name(original.name + "-preserved")
    original.rename(displaced)
    try:
        original.symlink_to(displaced, target_is_directory=True)
    except OSError:
        displaced.rename(original)
        pytest.skip("Directory symlink creation unavailable")
    before = (displaced / "photo.png").read_bytes()
    with pytest.raises(ValueError, match=r"link|reparse"):
        read(service, sid, package, result)
    with pytest.raises(ValueError, match=r"link|reparse"):
        photo.discard_project_photo(
            service, sid, "owner", package["package_id"], result["photo_id"]
        )
    assert (displaced / "photo.png").read_bytes() == before


def test_available_worker_with_malformed_output_never_persists(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def malformed(*args, **kwargs):
        return {
            "receipt": {},
            "image_base64": base64.b64encode(PNG).decode(),
            "mime_type": "image/png",
        }

    monkeypatch.setattr(photo, "render_mesh", malformed)
    with pytest.raises(ValueError):
        create(service, sid, package, part_id="base")
    assert not list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))


def test_actual_container_cleanup_failure_after_photo_deadline_is_not_optional(
    tmp_path, monkeypatch
):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    service.executor.mode = "docker"
    service.executor._typed_container_verified = True
    clock = [0.0]
    monkeypatch.setattr(rendering.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(rendering.shutil, "which", lambda _: "fixture-docker")
    commands = []

    class Process:
        returncode = None

        def poll(self):
            clock[0] = 16.0
            return self.returncode

        def kill(self):
            self.returncode = 0

        def wait(self, *, timeout):
            assert timeout == 10
            return self.returncode

    def popen(command, **kwargs):
        commands.append(command)
        assert command[:2] == ["fixture-docker", "run"]
        return Process()

    def run(command, **kwargs):
        commands.append(command)
        assert command[:3] == ["fixture-docker", "rm", "-f"]
        assert kwargs["timeout"] == 20
        return type("Removal", (), {"returncode": 1, "stderr": b"Removal refused"})()

    monkeypatch.setattr(execution.subprocess, "Popen", popen)
    monkeypatch.setattr(execution.subprocess, "run", run)
    with pytest.raises(ValueError, match="container cleanup failed") as raised:
        create(service, sid, package, part_id="base")
    assert isinstance(raised.value.__cause__, ValueError)
    assert "Operation cancelled" in str(raised.value.__cause__)
    assert len(commands) == 2
    assert not list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))
    assert (
        delivery.read_project_export(
            service, sid, "owner", package["package_id"], package["sha256"]
        )[1]
        == package
    )


@pytest.mark.parametrize("chain", ["cause", "context"])
def test_chained_cleanup_error_after_deadline_propagates_unchanged(tmp_path, monkeypatch, chain):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(rendering.time, "monotonic", lambda: clock[0])
    expected = ValueError("Cleanup failed")

    def watchdog(command, scratch, timeout, cancelled, **kwargs):
        clock[0] = 16.0
        assert cancelled()
        try:
            raise ValueError("Deadline watchdog")
        except ValueError as active:
            if chain == "cause":
                raise expected from active
            raise expected  # noqa: B904 - exercise an implicit cleanup exception chain

    monkeypatch.setattr(rendering, "bounded_process", watchdog)
    with pytest.raises(ValueError, match="Cleanup failed") as raised:
        create(service, sid, package, part_id="base")
    assert raised.value is expected
