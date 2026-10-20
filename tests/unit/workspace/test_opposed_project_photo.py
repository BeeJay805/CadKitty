"""Owned pair orchestration uses real captures and mocked saved-byte graphics."""

from __future__ import annotations

import base64
import json
import math
from copy import deepcopy

import pytest
from tests.unit.workspace.test_product_photo import PNG, write_result
from tests.unit.workspace.test_project_photo import captured, read

from forge_mind.workspace import product_photo_contract as contract
from forge_mind.workspace import project_photo as photo


def pair(service, sid, package, **kwargs):
    return photo.create_project_photo_pair(
        service, sid, "owner", package["package_id"], package["sha256"], part_id="cap", **kwargs
    )


def render_fixture(tmp_path, monkeypatch, *, image=PNG, after=None):
    calls = []

    def render(service, mesh, request, *, timeout):
        calls.append((deepcopy(request), timeout, mesh))
        scratch = tmp_path / ("render-" + str(len(calls)))
        scratch.mkdir()

        def facts(receipt, _):
            receipt["scope"] = contract.CAPTURED_SCOPE
            if request.get("view") == contract.OPPOSED_VIEW:
                vector = [-125.0, 165.0, 160.0]
                length = math.hypot(*vector)
                outward = [value / length for value in vector]
                horizontal = math.hypot(*outward[:2])
                up = [
                    -outward[0] * outward[2] / horizontal,
                    -outward[1] * outward[2] / horizontal,
                    horizontal,
                ]
                receipt["rendering"]["camera"].update(
                    position=[vector, [0, 0, 0], up],
                    distance_mm=length,
                    maximum_absolute_ndc=0.8,
                )
            if after is not None:
                after(receipt, len(calls), service)

        write_result(scratch, request, image=image, mutate=facts)
        return {
            "receipt": json.loads((scratch / "receipt.json").read_bytes()),
            "image_base64": base64.b64encode(image).decode("ascii"),
            "mime_type": "image/png",
        }

    monkeypatch.setattr(photo, "render_mesh", render)
    return calls


def outputs(service, sid):
    return list((service.store.root / "sessions" / sid).glob("project-photos/*/*/photo.png"))


def test_pair_same_capture_bytes_and_quantity_v1_v2_durable_read_and_discard(tmp_path, monkeypatch):
    service, sid, source, package, _, _ = captured(tmp_path, monkeypatch)
    original = {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()}
    calls = render_fixture(tmp_path, monkeypatch)
    result = pair(service, sid, package)
    assert set(result) == {"primary", "alternate"}
    first, second = result["primary"], result["alternate"]
    assert first["receipt"]["version"] == 1 and "view" not in first["receipt"]
    assert second["view"] == second["receipt"]["view"] == contract.OPPOSED_VIEW
    assert second["receipt"]["version"] == 2
    assert first["photo_id"] != second["photo_id"]
    assert first["receipt"]["quantity"] == second["receipt"]["quantity"] == 2
    assert calls[0][2] == calls[1][2]
    assert contract.legacy_request(calls[1][0]) == calls[0][0]
    for item in (first, second):
        assert read(service, sid, package, item) == (PNG, item["receipt"])
        photo.discard_project_photo(service, sid, "owner", package["package_id"], item["photo_id"])
    assert outputs(service, sid) == []
    assert {path.name: path.read_bytes() for path in source.iterdir() if path.is_file()} == original


def test_pair_uses_remaining_total_allowance_and_skips_expired_alternate(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(photo.time, "monotonic", lambda: clock[0])
    calls = render_fixture(
        tmp_path, monkeypatch, after=lambda _, number, __: clock.__setitem__(0, 2 * number)
    )
    result = pair(service, sid, package)
    assert calls[0][1] == 15 and calls[1][1] == 13
    assert result["alternate"]["status"] == "available"
    original = photo.create_project_photo

    def elapsed(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] += 16
        return value

    monkeypatch.setattr(photo, "create_project_photo", elapsed)
    second = pair(service, sid, package)
    assert second["primary"]["status"] == "available"
    assert second["alternate"]["code"] == "render_failed"
    assert len(calls) == 3


def test_unavailable_primary_skips_alternate_without_worker(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch, include_stl=False)
    calls = render_fixture(tmp_path, monkeypatch)
    result = pair(service, sid, package)
    assert result["primary"]["code"] == "missing_stl"
    assert result["alternate"]["code"] == "primary_unavailable"
    assert result["alternate"]["candidates"] == ["base", "cap"]
    assert calls == [] and outputs(service, sid) == []


def test_exhausted_before_alternate_never_passes_nonpositive_deadline(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(photo.time, "monotonic", lambda: clock[0])
    render_fixture(tmp_path, monkeypatch)
    original = photo.create_project_photo
    deadlines = []

    def create(*args, **kwargs):
        deadlines.append(kwargs["timeout"])
        assert kwargs["timeout"] > 0
        result = original(*args, **kwargs)
        clock[0] = 15
        return result

    monkeypatch.setattr(photo, "create_project_photo", create)
    result = pair(service, sid, package)
    assert deadlines == [15]
    assert result["primary"]["status"] == "available"
    assert result["alternate"]["code"] == "render_failed"
    assert len(outputs(service, sid)) == 1


def test_capture_tamper_between_views_is_fatal_and_rolls_back_primary(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    render_fixture(tmp_path, monkeypatch)
    original = photo.render_mesh

    def tamper(*args, **kwargs):
        value = original(*args, **kwargs)
        if args[2].get("view"):
            path = (
                service.store.root / "project-exports" / sid / package["package_id"] / "project.zip"
            )
            path.write_bytes(b"changed capture")
        return value

    monkeypatch.setattr(photo, "render_mesh", tamper)
    with pytest.raises((ValueError, OSError)):
        pair(service, sid, package)
    assert outputs(service, sid) == []


def test_typed_alternate_absence_keeps_primary_but_unclassified_error_rolls_it_back(
    tmp_path, monkeypatch
):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    render_fixture(tmp_path, monkeypatch)
    original = photo.render_mesh

    def unavailable(*args, **kwargs):
        if args[2].get("view"):
            raise contract.PhotoUnavailableError("renderer_unavailable", "Optional runtime absent")
        return original(*args, **kwargs)

    monkeypatch.setattr(photo, "render_mesh", unavailable)
    prior = pair(service, sid, package)
    assert prior["alternate"]["code"] == "renderer_unavailable"
    preserved = outputs(service, sid)

    def fatal(*args, **kwargs):
        if args[2].get("view"):
            raise ValueError("Source integrity failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(photo, "render_mesh", fatal)
    with pytest.raises(ValueError, match="Source integrity failed"):
        pair(service, sid, package)
    assert outputs(service, sid) == preserved
    assert read(service, sid, package, prior["primary"])[0] == PNG


@pytest.mark.parametrize("failure", ["triangles", "cancel"])
def test_alternate_source_mismatch_and_cancellation_refuse_and_clean_pair(
    tmp_path, monkeypatch, failure
):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)

    def mutate(receipt, number, current):
        if number == 2:
            if failure == "triangles":
                receipt["rendering"]["geometry"]["triangle_sha256"] = "d" * 64
            else:
                current.stop.set()

    render_fixture(tmp_path, monkeypatch, after=mutate)
    with pytest.raises(ValueError, match=r"triangles|cancelled"):
        pair(service, sid, package)
    assert outputs(service, sid) == []


def test_http_pair_budget_drops_only_alternate_and_preserves_primary_resource(
    tmp_path, monkeypatch
):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    # Header/hash validation is mocked graphics plumbing, not a raster quality test.
    image = PNG + b"x" * (850_000 - len(PNG))
    render_fixture(tmp_path, monkeypatch, image=image)
    result = pair(service, sid, package)
    assert result["primary"]["status"] == "available"
    assert result["alternate"]["code"] == "delivery_budget_exceeded"
    assert len(outputs(service, sid)) == 1
    assert read(service, sid, package, result["primary"])[0] == image
    assert len(json.dumps(result).encode()) <= contract.MAX_RESPONSE_BYTES


def test_cleanup_failure_remains_fatal_and_cannot_delete_replaced_primary(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    render_fixture(tmp_path, monkeypatch)
    original = photo.render_mesh
    replacements = []

    def replace(*args, **kwargs):
        if args[2].get("view"):
            first = outputs(service, sid)[0]
            first.write_bytes(b"foreign replacement")
            replacements.append(first)
            raise ValueError("Alternate failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(photo, "render_mesh", replace)
    with pytest.raises(ValueError, match=r"cleanup failed.*replaced"):
        pair(service, sid, package)
    assert replacements[0].read_bytes() == b"foreign replacement"


@pytest.mark.parametrize("filename", ["photo.png", "receipt.json"])
def test_successful_alternate_cannot_publish_changed_primary(tmp_path, monkeypatch, filename):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    changed = []

    def tamper(_, number, current):
        if number != 2:
            return
        path = outputs(current, sid)[0].with_name(filename)
        before = path.stat().st_ino
        data = path.read_bytes()
        replacement = data[:-1] + bytes([data[-1] ^ 1])
        with path.open("r+b") as handle:
            handle.write(replacement)
        assert path.stat().st_ino == before
        changed.append((path, replacement))

    calls = render_fixture(tmp_path, monkeypatch, after=tamper)
    with pytest.raises(ValueError, match=r"cleanup failed.*replaced"):
        pair(service, sid, package)
    assert len(calls) == 2
    assert len(outputs(service, sid)) == 1
    assert changed[0][0].read_bytes() == changed[0][1]


def test_v2_tampered_view_and_foreign_owner_still_refuse_read(tmp_path, monkeypatch):
    service, sid, _, package, _, _ = captured(tmp_path, monkeypatch)
    render_fixture(tmp_path, monkeypatch)
    result = pair(service, sid, package)["alternate"]
    with pytest.raises((ValueError, OSError)):
        read(service, sid, package, result, owner="foreign")
    directory = photo._directory(service, sid, package["package_id"], result["photo_id"])
    descriptor = json.loads((directory / "receipt.json").read_bytes())
    descriptor["receipt"]["view"] = "arbitrary-camera"
    (directory / "receipt.json").write_text(json.dumps(descriptor), encoding="utf-8")
    with pytest.raises(ValueError):
        read(service, sid, package, result)
    with pytest.raises(ValueError):
        photo.discard_project_photo(
            service, sid, "owner", package["package_id"], result["photo_id"]
        )
