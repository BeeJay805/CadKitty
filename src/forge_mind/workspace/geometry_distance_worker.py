"""Evaluate one hash-bound saved distance request inside a controlled native worker."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from forge_mind.workspace.engine_identity import engine_fingerprint
from forge_mind.workspace.geometry_distance_contract import (
    MAX_INPUT_BYTES,
    MAX_RESPONSE_BYTES,
    SCOPE,
    SOURCE_NAMES,
    DistanceContext,
    DistanceResult,
)
from forge_mind.workspace.section_render import read_bytes, read_object

NATIVE_RESULT_KEYS = frozenset(
    {
        "minimum_distance_mm",
        "witness_a_mm",
        "witness_b_mm",
        "solution_count",
        "native_completed",
        "native_inner_solution",
    }
)


def _sources(directory: Path, context: DistanceContext) -> dict[str, bytes]:
    source: dict[str, bytes] = {}
    for name in sorted(SOURCE_NAMES):
        content = read_bytes(directory / name, MAX_INPUT_BYTES)
        if hashlib.sha256(content).hexdigest() != context.source_hashes[name]:
            raise ValueError("Staged distance source does not match its saved checksum")
        source[name] = content
    return source


def measure_request(request_path: Path) -> dict[str, Any]:
    """Validate staged inputs, run native evaluation once, and publish only result.json."""
    context = DistanceContext.model_validate(
        read_object(read_bytes(request_path, MAX_RESPONSE_BYTES))
    )
    directory = request_path.parent
    result_path = directory / "result.json"
    if result_path.exists() or result_path.is_symlink():
        raise ValueError("Distance worker refuses an existing result")
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Distance worker engine differs from requested query engine")
    source = _sources(directory, context)
    topology = read_object(source["topology.json"])
    read_object(source["document.json"])
    # STEP/OCP imports remain in this execution branch, never the service host.
    from forge_mind.workspace.geometry_distance_native import measure_selected_distance

    native = measure_selected_distance(directory / "part.step", topology, context.a, context.b)
    if not isinstance(native, dict) or set(native) != NATIVE_RESULT_KEYS:
        raise ValueError("Native distance returned unexpected result fields")
    result = DistanceResult.model_validate(
        {**context.model_dump(mode="json"), **native, "scope": SCOPE}
    )
    _sources(directory, context)
    if engine_fingerprint() != context.query_engine_fingerprint:
        raise ValueError("Distance worker engine changed during evaluation")
    encoded = result.model_dump_json().encode("utf-8")
    if len(encoded) > MAX_RESPONSE_BYTES:
        raise ValueError("Distance result exceeds its output byte budget")
    with result_path.open("xb") as stream:
        stream.write(encoded)
    return result.model_dump(mode="json")


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Expected one staged request.json argument")
    measure_request(Path(sys.argv[1]))


if __name__ == "__main__":
    main()
