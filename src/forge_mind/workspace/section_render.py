"""Render a requested plane from staged saved STL bytes in an isolated worker."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import sys
from pathlib import Path
from typing import Any

FRAME = "cad_z_up_mm_world"
SCOPE = (
    "Whole saved revision STL section in world CAD Z-up millimeters. Polylines are "
    "tessellated display intersections, not verified closed cavities or exact STEP "
    "topology. No full-wall thickness, passage continuity, sealing, strength, "
    "manufacturing suitability or physical approval is established."
)
MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_REQUEST_BYTES = 64 * 1024
MAX_IMAGE_BYTES = 1024 * 1024
MAX_RECEIPT_BYTES = 64 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
REQUEST_KEYS = {
    "project_id",
    "revision_id",
    "axis",
    "offset_mm",
    "frame",
    "mesh_sha256",
    "document_sha256",
}


def read_bytes(path: Path, maximum: int) -> bytes:
    """Check size before opening, then bound growth during the read."""
    info = path.lstat()
    if (
        not stat.S_ISREG(info.st_mode)
        or path.is_symlink()
        or info.st_nlink != 1
        or getattr(info, "st_file_attributes", 0) & 0x400
        or not 0 < info.st_size <= maximum
    ):
        raise ValueError("Section file is missing, linked or oversized")
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if not 0 < len(content) <= maximum:
        raise ValueError("Section file changed beyond its byte budget")
    return content


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate section JSON key")
        value[key] = item
    return value


def _constant(value: str) -> Any:
    raise ValueError("Nonfinite section JSON number: " + value)


def read_object(content: bytes) -> dict[str, Any]:
    value = json.loads(content, object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a section JSON object")
    return value


def finite_number(value: Any) -> float:
    if type(value) not in (int, float):
        raise ValueError("Section plane requires a finite numeric offset")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("Section plane requires a finite numeric offset") from exc
    if not math.isfinite(number):
        raise ValueError("Section plane requires a finite numeric offset")
    return number


def validated_request(value: dict[str, Any]) -> dict[str, Any]:
    if set(value) != REQUEST_KEYS:
        raise ValueError("Unexpected or missing section request fields")
    for key in ("project_id", "revision_id"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{32}", value[key]):
            raise ValueError("Invalid section project or revision identifier")
    for key in ("mesh_sha256", "document_sha256"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[a-f0-9]{64}", value[key]):
            raise ValueError("Invalid section source checksum")
    if value["axis"] not in ("x", "y", "z") or value["frame"] != FRAME:
        raise ValueError("Section requires a world CAD X, Y or Z plane")
    return {**value, "offset_mm": finite_number(value["offset_mm"])}


def render_section(request_path: Path) -> dict[str, Any]:
    request = validated_request(read_object(read_bytes(request_path, MAX_REQUEST_BYTES)))
    directory = request_path.parent
    for name, key in (
        ("part.stl", "mesh_sha256"),
        ("document.json", "document_sha256"),
    ):
        content = read_bytes(directory / name, MAX_INPUT_BYTES)
        if hashlib.sha256(content).hexdigest() != request[key]:
            raise ValueError("Staged section source does not match its receipt")
    # Keep mesh parsing and rendering inside the qualified/bounded worker.
    from forge_mind.workspace.measure import section_revision

    image_path = directory / "section.png"
    observed = section_revision(directory, request["axis"], request["offset_mm"], image_path)
    count = observed.get("loops")
    observed_offset = finite_number(observed.get("offset_mm"))
    if (
        observed.get("axis") != request["axis"]
        or observed_offset != request["offset_mm"]
        or type(count) is not int
        or count < 0
    ):
        raise ValueError("Malformed section renderer observation")
    image = None
    if count == 0:
        if observed.get("image") is not None or image_path.exists():
            raise ValueError("Empty section must not include an image")
    else:
        if observed.get("image") != "section.png":
            raise ValueError("Section renderer returned an unexpected image")
        content = read_bytes(image_path, MAX_IMAGE_BYTES)
        if not content.startswith(PNG_SIGNATURE):
            raise ValueError("Section renderer did not return a PNG")
        image = {"file": "section.png", "sha256": hashlib.sha256(content).hexdigest()}
    receipt = {
        **request,
        "section_status": "intersected" if count else "empty",
        "polyline_count": count,
        "image": image,
        "scope": SCOPE,
    }
    data = json.dumps(receipt, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(data) > MAX_RECEIPT_BYTES:
        raise ValueError("Section receipt exceeds its byte budget")
    with (directory / "receipt.json").open("xb") as output:
        output.write(data)
    return receipt


if __name__ == "__main__":
    render_section(Path(sys.argv[1]))
