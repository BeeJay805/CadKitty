"""Evaluate one hash-bound saved cylindrical access inside a controlled worker."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.geometry_access_contract import (
    MAX_INPUT_BYTES,
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    MAX_TOTAL_INPUT_BYTES,
    SOURCE_NAMES,
    AccessContext,
    AccessResult,
)
from forge_mind.workspace.section_render import read_bytes, read_object

NATIVE_RESULT_KEYS = frozenset(
    {"sweep", "observations", "numerical_guards", "native_completed", "scope"}
)


def _sources(directory: Path, context: AccessContext) -> dict[str, bytes]:
    source: dict[str, bytes] = {}
    for name in sorted(SOURCE_NAMES):
        content = read_bytes(directory / name, MAX_INPUT_BYTES)
        if hashlib.sha256(content).hexdigest() != context.source_hashes[name]:
            raise ValueError("Staged access source does not match saved checksum")
        source[name] = content
    if sum(map(len, source.values())) > MAX_TOTAL_INPUT_BYTES:
        raise ValueError("Staged access source exceeds aggregate byte limit")
    return source


def inspect_request(request_path: Path) -> dict[str, Any]:
    """Validate staged inputs, evaluate once, and exclusively publish result.json."""
    context = AccessContext.model_validate(read_object(read_bytes(request_path, MAX_REQUEST_BYTES)))
    directory = request_path.parent
    result_path = directory / "result.json"
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Access worker refuses an existing result")
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Access worker engine differs from requested query engine")
    source = _sources(directory, context)
    topology = read_object(source["topology.json"])
    read_object(source["document.json"])
    # Native imports occur only in this bounded evaluation branch.
    from forge_mind.workspace.geometry_access_native import inspect_selected_cylindrical_access

    native = inspect_selected_cylindrical_access(
        directory / "part.step",
        topology,
        context.evaluated_cylinder,
        context.evaluated_motion,
        context.obstacles,
        context.inventory_body_count,
    )
    if not isinstance(native, dict) or set(native) != NATIVE_RESULT_KEYS:
        raise ValueError("Native access returned unexpected result fields")
    result = AccessResult.model_validate({**context.model_dump(mode="json"), **native})
    _sources(directory, context)
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Access worker engine changed during evaluation")
    encoded = result.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Access result exceeds its output byte budget")
    with result_path.open("xb") as stream:
        stream.write(encoded)
    return result.model_dump(mode="json")


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Expected one staged request.json argument")
    inspect_request(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
