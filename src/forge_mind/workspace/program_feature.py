"""Source features cross the existing author-process boundary before trusted measurement."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import build123d as b

from forge_mind.workspace.executor import Executor
from forge_mind.workspace.processing import regular_file

_INVALID_GEOMETRY_PREFIX = "Python feature returned invalid geometry"
_FAILURE_MESSAGE_LIMIT = 1400  # The worker error reader retains 1,600 characters.
_FAILURE_SCOPE = (
    "Independently reimported STEP after author execution; "
    "no construction location or cause inferred."
)


def _native_feature_failure_message(shape: Any) -> str:
    """Describe the existing refusal without trusting author-process globals."""
    name = type(shape).__name__
    facts: dict[str, Any] = {
        "stage": "independent_step_reimport",
        "native_valid": None,
        "solid_count": None,
        "shape_type": name[:64],
        "shape_type_truncated": len(name) > 64,
        "scope": _FAILURE_SCOPE,
        "inspection_errors": {},
    }
    inspections: tuple[tuple[str, Callable[[], bool | int]], ...] = (
        ("native_valid", lambda: bool(shape.is_valid)),
        ("solid_count", lambda: len(shape.solids())),
    )
    for key, inspect in inspections:
        try:
            facts[key] = inspect()
        except Exception as exc:
            facts["inspection_errors"][key] = type(exc).__name__[:32]
    message = (
        _INVALID_GEOMETRY_PREFIX + ": " + json.dumps(facts, separators=(",", ":"), allow_nan=False)
    )
    if len(message) > _FAILURE_MESSAGE_LIMIT:
        facts["shape_type"] = None
        facts["shape_type_truncated"] = True
        facts["inspection_errors"] = dict.fromkeys(facts["inspection_errors"], "unavailable")
        message = (
            _INVALID_GEOMETRY_PREFIX
            + ": "
            + json.dumps(facts, separators=(",", ":"), allow_nan=False)
        )
    return message


def compile_program_feature(code: str, parameters: dict[str, float], inputs: list[Any]) -> Any:
    """Only independently reimported CAD crosses back; author globals never do."""
    with tempfile.TemporaryDirectory(prefix="cadkitty-source-feature-") as temporary:
        directory = Path(temporary)
        names = []
        for index, shape in enumerate(inputs):
            name = f"input_{index}.step"
            b.export_step(shape, str(directory / name))
            names.append(name)
        program = "import build123d as b\n"
        program += f"params = {parameters!r}\n"
        program += f"inputs = tuple(b.import_step(name) for name in {names!r})\n"
        program += code
        result = Executor("trusted_local").build(program, directory, None, lambda: False)
        if not result.ok:
            raise ValueError("Python feature author failed: " + str(result.error_message))
        path = directory / "part.step"
        if not regular_file(path):
            raise ValueError("Python feature did not return a regular exported CAD artifact")
        shape = b.import_step(str(path))
        if not shape.is_valid or not shape.solids():
            try:
                message = _native_feature_failure_message(shape)
            except Exception:
                # Diagnostics must not replace the original geometry refusal.
                message = (
                    _INVALID_GEOMETRY_PREFIX
                    + ": "
                    + json.dumps(
                        {
                            "stage": "independent_step_reimport",
                            "native_valid": None,
                            "solid_count": None,
                            "shape_type": None,
                            "scope": _FAILURE_SCOPE,
                            "inspection_errors": {"diagnostic": "unavailable"},
                        },
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
            raise ValueError(message)
        return shape
