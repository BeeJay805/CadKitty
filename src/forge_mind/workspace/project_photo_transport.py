"""Pure captured-photo validation and identical MCP image delivery."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import struct
from typing import Any

from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic_core import to_json

from forge_mind.workspace.product_photo_contract import (
    ASSEMBLY_REQUEST_KEYS,
    CAPTURED_REQUEST_KEYS,
    MAX_IMAGE_BYTES,
    MAX_RECEIPT_BYTES,
    MAX_RESPONSE_BYTES,
    OPPOSED_VIEW,
    legacy_request,
    opposed_request,
    validate_receipt,
)

UNAVAILABLE_CODES = {
    "missing_stl",
    "ambiguous_parts",
    "no_made_parts",
    "renderer_unavailable",
    "execution_unqualified",
    "render_failed",
    "source_unavailable",
    "render_limit",
}


class PhotoResponseLimitError(ValueError):
    """The complete serialized SDK photo delivery exceeds its byte allowance."""


def alternate_unavailable(code: object, reason: object, candidates: object) -> dict[str, Any]:
    """Build a bounded absence using the caller's already validated make inventory."""
    if (
        not isinstance(code, str)
        or code not in UNAVAILABLE_CODES | {"primary_unavailable", "delivery_budget_exceeded"}
        or not isinstance(reason, str)
        or not 1 <= len(reason) <= 500
        or not isinstance(candidates, list)
        or any(
            not isinstance(item, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", item) is None
            for item in candidates
        )
        or candidates != sorted(set(candidates))
    ):
        raise ValueError("Invalid alternate photo absence")
    return {
        "view": OPPOSED_VIEW,
        "status": "unavailable",
        "code": code,
        "reason": reason,
        "candidates": list(candidates),
    }


def _capture_binding(
    context: dict[str, Any], package: dict[str, Any], members: list[dict[str, Any]]
) -> None:
    """Check the verified ZIP projection without trusting the photo's selection."""
    keys = {
        "version",
        "project_id",
        "revision_id",
        "package_id",
        "package_sha256",
        "structured_assembly",
        "captured_document_sha256",
        "captured_production_sha256",
        "assembly_step_sha256",
        "selected",
        "omitted",
        "make_quantities",
    }
    if (
        not isinstance(context, dict)
        or set(context) != keys
        or type(context["version"]) is not int
        or context["version"] != 1
        or type(context["structured_assembly"]) is not bool
        or any(context[key] != package[key] for key in ("project_id", "revision_id", "package_id"))
        or context["package_sha256"] != package["sha256"]
        or context["captured_document_sha256"] != package["source_scope"].get("sha256")
        or context["assembly_step_sha256"] != package["assembly"].get("sha256")
        or any(
            not isinstance(context[key], str) or re.fullmatch(r"[a-f0-9]{64}", context[key]) is None
            for key in (
                "captured_document_sha256",
                "captured_production_sha256",
                "assembly_step_sha256",
            )
        )
    ):
        raise ValueError("Photo capture context does not match the verified package")
    paths: set[str] = set()
    quantities: dict[str, int] = {}
    for section, roles in (
        ("selected", {"make"}),
        ("omitted", {"buy", "reference", "unspecified"}),
    ):
        rows = context[section]
        if not isinstance(rows, list):
            raise ValueError("Invalid photo capture occurrence inventory")
        ordered_paths: list[str] = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or set(row) != {"occurrence_path", "part_id", "role"}
                or not isinstance(row["occurrence_path"], str)
                or not row["occurrence_path"]
                or row["occurrence_path"] in paths
                or not isinstance(row["part_id"], str)
                or re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", row["part_id"]) is None
                or not isinstance(row["role"], str)
                or row["role"] not in roles
            ):
                raise ValueError("Invalid photo capture occurrence identity or role")
            paths.add(row["occurrence_path"])
            ordered_paths.append(row["occurrence_path"])
            if section == "selected":
                quantities[row["part_id"]] = quantities.get(row["part_id"], 0) + 1
        if ordered_paths != sorted(ordered_paths):
            raise ValueError("Photo capture occurrences must be sorted")
    expected = [{"part_id": key, "quantity": quantities[key]} for key in sorted(quantities)]
    supplied = context["make_quantities"]
    if (
        not isinstance(supplied, list)
        or any(
            not isinstance(row, dict)
            or set(row) != {"part_id", "quantity"}
            or type(row["quantity"]) is not int
            or row["quantity"] <= 0
            for row in supplied
        )
        or supplied != expected
    ):
        raise ValueError("Photo capture quantities do not match its occurrences")
    member_quantities: dict[str, int] = {}
    for member in members:
        if member["role"] == "make":
            key, quantity = member["part_id"], member["quantity"]
            if (
                type(quantity) is not int
                or quantity <= 0
                or (key in member_quantities and member_quantities[key] != quantity)
            ):
                raise ValueError("Photo capture has inconsistent CAD member quantities")
            member_quantities[key] = quantity
    if member_quantities != quantities:
        raise ValueError("Photo capture make inventory does not match captured CAD members")


def _assembly_receipt(
    receipt: dict[str, Any],
    package: dict[str, Any],
    context: dict[str, Any],
    preset: str,
    expected_view: str | None = None,
) -> dict[str, Any]:
    """Bind the strict receipt to independently verified production occurrences."""
    if not isinstance(receipt, dict) or not set(receipt) >= ASSEMBLY_REQUEST_KEYS:
        raise ValueError("Invalid made-assembly photo receipt")
    request = {key: receipt[key] for key in ASSEMBLY_REQUEST_KEYS}
    request.update(
        project_id=package["project_id"],
        revision_id=package["revision_id"],
        package_id=package["package_id"],
        package_sha256=package["sha256"],
        preset=preset,
        version=1,
        frame="cad_z_up_mm_world",
        source_kind="saved_topology_made_assembly",
    )
    clean = validate_receipt(receipt, opposed_request(request) if expected_view else request)
    source = clean["source"]
    if (
        source["captured_document_sha256"] != context["captured_document_sha256"]
        or source["assembly_step_sha256"] != context["assembly_step_sha256"]
    ):
        raise ValueError("Assembly photo source does not match captured package bytes")
    selection = clean["selection"]
    for section in ("selected", "omitted"):
        projected = []
        for row in selection[section]:
            if set(row) != {"occurrence_path", "part_id", "role", "body_index"}:
                raise ValueError("Invalid assembly photo occurrence row")
            projected.append({key: row[key] for key in ("occurrence_path", "part_id", "role")})
        if projected != context[section]:
            raise ValueError(
                "Assembly photo selection differs from captured production occurrences"
            )
    return clean


def validate_project_photo(
    value: dict[str, Any],
    package: dict[str, Any],
    members: list[dict[str, Any]],
    preset: str,
    part_id: str | None,
    *,
    capture_context: dict[str, Any] | None = None,
    expected_view: str | None = None,
) -> tuple[dict[str, Any], bytes | None]:
    """Bind the image to exact verified capture evidence, never current project state."""
    if expected_view not in (None, OPPOSED_VIEW):
        raise ValueError("Unsupported expected project photo view")
    candidates = sorted({row["part_id"] for row in members if row["role"] == "make"})
    if capture_context is not None:
        _capture_binding(capture_context, package, members)
    assembly = (
        part_id is None and capture_context is not None and capture_context["structured_assembly"]
    )
    if part_id is not None and part_id not in candidates:
        raise ValueError("Photo target is not a captured made definition")
    chosen = part_id if part_id is not None else candidates[0] if len(candidates) == 1 else None
    selected = [
        row
        for row in members
        if row["role"] == "make" and row["part_id"] == chosen and row["format"] == "stl"
    ]
    if not isinstance(value, dict):
        raise ValueError("Invalid project photo result")
    if value.get("status") == "unavailable":
        if (
            set(value) != {"status", "code", "reason", "candidates"}
            or not isinstance(value["code"], str)
            or value["code"] not in UNAVAILABLE_CODES
            or not isinstance(value["reason"], str)
            or not 1 <= len(value["reason"]) <= 500
            or value["candidates"] != candidates
        ):
            raise ValueError("Invalid optional photo absence")
        if assembly:
            expected = "no_made_parts" if not candidates else None
        else:
            expected = (
                "no_made_parts"
                if not candidates
                else "ambiguous_parts"
                if chosen is None
                else "missing_stl"
                if not selected
                else None
            )
        if value["code"] in {"source_unavailable", "render_limit"} and not assembly:
            raise ValueError("Assembly photo absence requires verified assembly capture context")
        if (expected is not None and value["code"] != expected) or (
            expected is None
            and value["code"] in {"no_made_parts", "ambiguous_parts", "missing_stl"}
        ):
            raise ValueError("Photo absence does not match captured members")
        return dict(value), None
    if (
        set(value) != {"status", "photo_id", "receipt", "image_base64", "mime_type"}
        or value.get("status") != "available"
        or value["mime_type"] != "image/png"
        or not isinstance(value["photo_id"], str)
        or re.fullmatch(r"[a-f0-9]{32}", value["photo_id"]) is None
        or (not assembly and len(selected) != 1)
        or (assembly and not candidates)
    ):
        raise ValueError("Invalid available captured photo")
    receipt = value["receipt"]
    if assembly:
        assert capture_context is not None
        receipt = _assembly_receipt(receipt, package, capture_context, preset, expected_view)
    else:
        row = selected[0]
        request = {
            "project_id": package["project_id"],
            "revision_id": package["revision_id"],
            "package_id": package["package_id"],
            "package_sha256": package["sha256"],
            "part_id": row["part_id"],
            "file_id": row["file_id"],
            "quantity": row["quantity"],
            "mesh_sha256": row["sha256"],
            "frame": row["frame"],
            "preset": preset,
            "version": 1,
        }
        if set(request) != CAPTURED_REQUEST_KEYS:
            raise ValueError("Invalid captured photo request")
        receipt = validate_receipt(receipt, opposed_request(request) if expected_view else request)
    encoded = value["image_base64"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("Captured photo exceeds its image limit")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Invalid captured PNG encoding") from exc
    if (
        not 33 <= len(png) <= MAX_IMAGE_BYTES
        or png[:8] != b"\x89PNG\r\n\x1a\n"
        or struct.unpack(">I", png[8:12])[0] != 13
        or png[12:16] != b"IHDR"
        or struct.unpack(">II", png[16:24]) != (1024, 1024)
        or hashlib.sha256(png).hexdigest() != receipt["image"]["sha256"]
    ):
        raise ValueError("Captured PNG does not match its receipt")
    return {"status": "available", "photo_id": value["photo_id"], "receipt": receipt}, png


def validate_project_photo_pair(
    value: dict[str, Any],
    package: dict[str, Any],
    members: list[dict[str, Any]],
    preset: str,
    part_id: str | None,
    *,
    capture_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bytes | None, dict[str, Any], bytes | None]:
    """Validate two views of the same capture, including source and triangle identity."""
    if not isinstance(value, dict) or set(value) != {"primary", "alternate"}:
        raise ValueError("Invalid project photo pair")
    primary, png = validate_project_photo(
        value["primary"], package, members, preset, part_id, capture_context=capture_context
    )
    alternate = value["alternate"]
    if not isinstance(alternate, dict) or alternate.get("view") != OPPOSED_VIEW:
        raise ValueError("Invalid alternate project photo view")
    raw = {key: item for key, item in alternate.items() if key != "view"}
    candidates = sorted({row["part_id"] for row in members if row["role"] == "make"})
    if raw.get("code") in {"primary_unavailable", "delivery_budget_exceeded"}:
        clean = alternate_unavailable(raw.get("code"), raw.get("reason"), raw.get("candidates"))
        if clean != alternate or raw["candidates"] != candidates:
            raise ValueError("Alternate absence differs from captured make inventory")
        required = "primary_unavailable" if png is None else "delivery_budget_exceeded"
        if raw["code"] != required:
            raise ValueError("Alternate absence contradicts primary availability")
        return primary, png, clean, None
    if png is None:
        raise ValueError("An unavailable primary requires skipped alternate rendering")
    second, second_png = validate_project_photo(
        raw,
        package,
        members,
        preset,
        part_id,
        capture_context=capture_context,
        expected_view=OPPOSED_VIEW,
    )
    if second_png is not None:
        first_receipt, second_receipt = primary["receipt"], second["receipt"]
        first_keys = (
            ASSEMBLY_REQUEST_KEYS if "source_kind" in first_receipt else CAPTURED_REQUEST_KEYS
        )
        second_keys = (
            ASSEMBLY_REQUEST_KEYS if "source_kind" in second_receipt else CAPTURED_REQUEST_KEYS
        )
        first_request = legacy_request({key: first_receipt[key] for key in first_keys})
        second_request = legacy_request(
            {key: second_receipt[key] for key in second_keys | {"view"}}
        )
        if first_request != second_request:
            raise ValueError("Project photo views have different captured sources")
        if first_receipt["rendering"]["geometry"] != second_receipt["rendering"]["geometry"]:
            raise ValueError("Project photo views have different source triangles")
        if primary["photo_id"] == second["photo_id"]:
            raise ValueError("Project photo views must have distinct owned resource identities")
    return primary, png, {"view": OPPOSED_VIEW, **second}, second_png


def project_export_tool_result(
    metadata: dict[str, Any], png: bytes | None = None, *, alternate_png: bytes | None = None
) -> CallToolResult:
    """Keep legacy metadata at the root and append an actual image when available."""
    json.dumps(metadata, allow_nan=False)
    for key in ("preview", "preview_alternate"):
        preview = metadata.get(key)
        if isinstance(preview, dict) and preview.get("status") == "available":
            receipt = preview.get("receipt")
            if not isinstance(receipt, dict) or len(to_json(receipt)) > MAX_RECEIPT_BYTES:
                raise ValueError("Photo receipt exceeds its byte limit or is malformed")
    # This is the SDK's original dictionary-result serialization, including None values.
    text = to_json(metadata).decode("utf-8")
    result = CallToolResult(
        content=[TextContent(type="text", text=text)], structured_content=metadata
    )
    if alternate_png is not None and png is None:
        raise ValueError("An alternate image requires its primary image")
    for image in (png, alternate_png):
        if image is None:
            continue
        if not isinstance(image, bytes) or len(image) > MAX_IMAGE_BYTES:
            raise ValueError("Photo image exceeds its byte limit")
        result.content.append(
            ImageContent(
                type="image", data=base64.b64encode(image).decode("ascii"), mime_type="image/png"
            )
        )
    if (png is not None or "preview_alternate" in metadata) and len(
        result.model_dump_json(by_alias=True).encode("utf-8")
    ) > MAX_RESPONSE_BYTES:
        raise PhotoResponseLimitError("Photo delivery exceeds its response limit")
    # Metadata-only deliveries retain their existing transport capacity policy.
    return result
