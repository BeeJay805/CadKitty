"""Pair validation preserves exact capture/source identity and actual SDK byte limits."""

from __future__ import annotations

import base64
import json
import math
from copy import deepcopy

import pytest
from tests.unit.workspace.test_assembly_photo_transport import assembly_photo
from tests.unit.workspace.test_project_export_local import saved_package
from tests.unit.workspace.test_project_photo_delivery_transport import captured_photo

from forge_mind.workspace.product_photo_contract import (
    MAX_IMAGE_BYTES,
    MAX_RESPONSE_BYTES,
    OPPOSED_VIEW,
)
from forge_mind.workspace.project_members import captured_cad_files, captured_photo_context
from forge_mind.workspace.project_photo_transport import (
    PhotoResponseLimitError,
    alternate_unavailable,
    project_export_tool_result,
    validate_project_photo,
    validate_project_photo_pair,
)


def opposed(primary):
    value = deepcopy(primary)
    value.update(view=OPPOSED_VIEW, photo_id="e" * 32)
    value["receipt"].update(view=OPPOSED_VIEW, version=2)
    bounds = value["receipt"]["rendering"]["geometry"]["bounds_mm"]
    center = [(bounds[0][axis] + bounds[1][axis]) / 2 for axis in range(3)]
    vector = [-125, 165, 160]
    length = math.hypot(*vector)
    outward = [item / length for item in vector]
    horizontal = math.hypot(*outward[:2])
    up = [-outward[0] * outward[2] / horizontal, -outward[1] * outward[2] / horizontal, horizontal]
    value["receipt"]["rendering"]["camera"].update(
        position=[[center[axis] + vector[axis] for axis in range(3)], center, up],
        distance_mm=length,
        maximum_absolute_ndc=0.8,
        view_angle_degrees=35,
        projection="perspective",
    )
    return value


def fixture(tmp_path, *, assembly=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    package, data = saved_package(tmp_path, True)
    members = captured_cad_files(data, package)
    context = captured_photo_context(data, package)
    primary = assembly_photo(package, context) if assembly else captured_photo(package, members)
    return package, members, context, {"primary": primary, "alternate": opposed(primary)}


def validate(package, members, context, value, *, assembly=False):
    return validate_project_photo_pair(
        value, package, members, "plastic", None if assembly else "base", capture_context=context
    )


@pytest.mark.parametrize("assembly", [False, True])
def test_same_capture_two_distinct_views_preserve_primary_and_real_image_order(tmp_path, assembly):
    package, members, context, pair = fixture(tmp_path, assembly=assembly)
    before = deepcopy(pair)
    first, png, second, alternate_png = validate(package, members, context, pair, assembly=assembly)
    assert pair == before
    assert first["receipt"] == pair["primary"]["receipt"]
    assert second["view"] == OPPOSED_VIEW and second["receipt"]["version"] == 2
    assert png == alternate_png == base64.b64decode(pair["primary"]["image_base64"])
    metadata = {**package, "preview": first, "preview_alternate": second}
    result = project_export_tool_result(metadata, png, alternate_png=alternate_png)
    assert result.structured_content == metadata == json.loads(result.content[0].text)
    assert [item.type for item in result.content] == ["text", "image", "image"]
    assert base64.b64decode(result.content[1].data) == png
    assert base64.b64decode(result.content[2].data) == alternate_png
    assert len(result.model_dump_json(by_alias=True).encode()) <= MAX_RESPONSE_BYTES


@pytest.mark.parametrize("field", ["view", "preset", "mesh", "triangles", "duplicate_id", "png"])
def test_wrong_view_source_geometry_or_image_is_fatal_not_budget_absence(tmp_path, field):
    package, members, context, pair = fixture(tmp_path)
    alternate = pair["alternate"]
    if field == "view":
        alternate["view"] = "other"
    elif field == "preset":
        alternate["receipt"]["preset"] = "metal"
    elif field == "mesh":
        alternate["receipt"]["mesh_sha256"] = "0" * 64
    elif field == "triangles":
        alternate["receipt"]["rendering"]["geometry"]["triangle_sha256"] = "0" * 64
    elif field == "duplicate_id":
        alternate["photo_id"] = pair["primary"]["photo_id"]
    else:
        alternate["image_base64"] = base64.b64encode(b"not a PNG").decode()
    with pytest.raises(ValueError) as caught:
        validate(package, members, context, pair)
    assert not isinstance(caught.value, PhotoResponseLimitError)


def test_assembly_alternate_cannot_change_saved_source_or_world_body_mapping(tmp_path):
    package, members, context, pair = fixture(tmp_path, assembly=True)
    pair["alternate"]["receipt"]["source"]["saved_document_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="different captured sources"):
        validate(package, members, context, pair, assembly=True)
    # Use the matching second capture to isolate a source-preserving occurrence index change.
    second_package, second_members, second_context, pair = fixture(
        tmp_path / "third", assembly=True
    )
    rows = pair["alternate"]["receipt"]["selection"]["selected"]
    rows[0]["body_index"], rows[1]["body_index"] = rows[1]["body_index"], rows[0]["body_index"]
    with pytest.raises(ValueError, match="different captured sources"):
        validate(second_package, second_members, second_context, pair, assembly=True)


def test_primary_absence_skips_alternate_and_candidate_inventory_is_exact(tmp_path):
    package, members, context, pair = fixture(tmp_path)
    candidates = ["base", "cap"]
    pair["primary"] = {
        "status": "unavailable",
        "code": "renderer_unavailable",
        "reason": "Absent",
        "candidates": candidates,
    }
    pair["alternate"] = alternate_unavailable("primary_unavailable", "Primary absent", candidates)
    first, png, second, alternate = validate(package, members, context, pair)
    assert png is alternate is None and first == pair["primary"] and second == pair["alternate"]
    pair["alternate"]["candidates"] = ["base"]
    with pytest.raises(ValueError, match="inventory"):
        validate(package, members, context, pair)


@pytest.mark.parametrize(
    "code", ["renderer_unavailable", "render_failed", "delivery_budget_exceeded"]
)
def test_valid_primary_survives_only_declared_alternate_absence(tmp_path, code):
    package, members, context, pair = fixture(tmp_path)
    pair["alternate"] = alternate_unavailable(code, "Alternate absent", ["base", "cap"])
    first, png, second, alternate = validate(package, members, context, pair)
    assert first["status"] == "available" and png is not None
    assert second["code"] == code and alternate is None
    pair["alternate"] = alternate_unavailable(
        "primary_unavailable", "Contradiction", ["base", "cap"]
    )
    with pytest.raises(ValueError, match="contradicts"):
        validate(package, members, context, pair)


@pytest.mark.parametrize("bad", ["unknown_code", "unsorted", "duplicate", "long_reason", "boolean"])
def test_alternate_absence_helper_rejects_unbounded_or_unknown_values(bad):
    code, reason, candidates = "render_failed", "Absent", ["base", "cap"]
    if bad == "unknown_code":
        code = "auth_failed"
    elif bad == "unsorted":
        candidates.reverse()
    elif bad == "duplicate":
        candidates.append("cap")
    elif bad == "long_reason":
        reason = "x" * 501
    else:
        candidates = [True]
    with pytest.raises(ValueError):
        alternate_unavailable(code, reason, candidates)


def test_legacy_validator_and_metadata_only_formatter_remain_unchanged(tmp_path):
    package, members, context, pair = fixture(tmp_path)
    primary = pair["primary"]
    metadata, image = validate_project_photo(
        primary, package, members, "plastic", "base", capture_context=context
    )
    assert metadata["receipt"] == primary["receipt"] and image is not None
    legacy = project_export_tool_result(package)
    assert legacy.structured_content == package and len(legacy.content) == 1
    with pytest.raises(ValueError):
        validate_project_photo(
            primary,
            package,
            members,
            "plastic",
            "base",
            capture_context=context,
            expected_view=OPPOSED_VIEW,
        )


def test_only_complete_sdk_size_overflow_has_specific_budget_exception():
    metadata = {"project_id": "a" * 32}
    image = b"x" * 850_000
    assert len(project_export_tool_result(metadata, image).content) == 2
    with pytest.raises(PhotoResponseLimitError):
        project_export_tool_result(metadata, image, alternate_png=image)
    with pytest.raises(ValueError) as invalid:
        project_export_tool_result(metadata, image, alternate_png=b"x" * (MAX_IMAGE_BYTES + 1))
    assert not isinstance(invalid.value, PhotoResponseLimitError)
    with pytest.raises(ValueError) as invalid_json:
        project_export_tool_result({"invalid": float("nan")}, image, alternate_png=image)
    assert not isinstance(invalid_json.value, PhotoResponseLimitError)
    with pytest.raises(ValueError):
        project_export_tool_result(metadata, alternate_png=image)


def test_paired_absence_checks_complete_sdk_metadata_budget_but_legacy_retains_policy():
    large = {"declared_notes": "x" * 1_100_000}
    legacy = project_export_tool_result(large)
    assert legacy.structured_content == large
    absent = alternate_unavailable("primary_unavailable", "Primary absent", ["base"])
    with pytest.raises(PhotoResponseLimitError):
        project_export_tool_result({**large, "preview_alternate": absent})
