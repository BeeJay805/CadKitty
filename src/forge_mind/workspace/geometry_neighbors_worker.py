"""Evaluate one hash-bound saved neighbors request inside a controlled native worker."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.geometry_neighbors_contract import (
    MAX_INPUT_BYTES,
    MAX_RESPONSE_BYTES,
    SOURCE_NAMES,
    NeighborContext,
    NeighborResult,
)
from forge_mind.workspace.section_render import read_bytes, read_object

NATIVE_RESULT_KEYS = frozenset(
    {
        "focal",
        "relation",
        "neighbors",
        "total_neighbors",
        "total_incidences",
        "next_offset",
        "native_completed",
        "coverage",
        "scope",
    }
)


def _sources(directory: Path, context: NeighborContext) -> dict[str, bytes]:
    source: dict[str, bytes] = {}
    for name in sorted(SOURCE_NAMES):
        content = read_bytes(directory / name, MAX_INPUT_BYTES)
        if hashlib.sha256(content).hexdigest() != context.source_hashes[name]:
            raise ValueError("Staged neighbors source does not match its saved checksum")
        source[name] = content
    return source


def inspect_request(request_path: Path) -> dict[str, Any]:
    """Validate staged inputs, run native evaluation once, and publish only result.json."""
    context = NeighborContext.model_validate(
        read_object(read_bytes(request_path, MAX_RESPONSE_BYTES))
    )
    directory = request_path.parent
    result_path = directory / "result.json"
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Neighbors worker refuses an existing result")
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Neighbors worker engine differs from requested query engine")
    source = _sources(directory, context)
    topology = read_object(source["topology.json"])
    read_object(source["document.json"])
    # STEP/OCP imports remain in this execution branch, never the service host.
    from forge_mind.workspace.geometry_neighbors_native import inspect_selected_neighbors

    native = inspect_selected_neighbors(
        directory / "part.step", topology, context.selector, context.offset, context.limit
    )
    if not isinstance(native, dict) or set(native) != NATIVE_RESULT_KEYS:
        raise ValueError("Native neighbors returned unexpected result fields")
    result = NeighborResult.model_validate({**context.model_dump(mode="json"), **native})
    _sources(directory, context)
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Neighbors worker engine changed during evaluation")
    encoded = result.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Neighbors result exceeds its output byte budget")
    with result_path.open("xb") as stream:
        stream.write(encoded)
    return result.model_dump(mode="json")


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Expected one staged request.json argument")
    inspect_request(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
