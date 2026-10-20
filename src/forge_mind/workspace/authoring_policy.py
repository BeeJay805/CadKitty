"""Enforce authored geometry representation without restricting native operation kinds."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal

from forge_mind.workspace.document import DesignDocument, Feature
from forge_mind.workspace.store import WorkspaceError

AuthoringMode = Literal["hybrid", "native_operations"]


def validate_authoring_mode(value: str | None) -> AuthoringMode:
    """Preserve the legacy default while rejecting unsupported explicit policies."""
    if value is None or value == "hybrid":
        return "hybrid"
    if value == "native_operations":
        return "native_operations"
    raise WorkspaceError("authoring_mode must be hybrid or native_operations")


def _declared_features(document: DesignDocument) -> Iterator[tuple[str, Feature]]:
    """Visit definitions, independent of outputs, suppression and occurrence expansion."""
    for feature in document.features:
        yield f"feature {feature.id}", feature
    if document.assembly is not None:
        for part in document.assembly.parts:
            for feature in part.features:
                yield f"part {part.id} feature {feature.id}", feature


def enforce_document_policy(document: DesignDocument, mode: str) -> None:
    """Reject authored Python anywhere in a native session's resolved document."""
    if validate_authoring_mode(mode) != "native_operations":
        return
    for location, feature in _declared_features(document):
        if feature.kind == "python" or feature.code:
            raise WorkspaceError(
                f"native_operations rejects authored Python or nonempty code in {location}"
            )


def enforce_action_policy(kind: str, mode: str, code: str = "") -> None:
    """Guard author actions; the trusted document execution wrapper is infrastructure."""
    if validate_authoring_mode(mode) != "native_operations":
        return
    if kind == "build" or code:
        raise WorkspaceError(
            f"native_operations rejects action {kind}: authored Python build/code is forbidden; "
            "submit native Features instead"
        )


def source_classification(document: DesignDocument) -> dict[str, Any]:
    """Report declared sources, including retained and imported geometry.

    Counts describe Feature declarations, not output solids or physical occurrences.
    Imports are permitted native-mode inputs, but are identified separately. No trusted
    program wrapper is inspected or counted as authored Python.
    """
    native = python = imported = code = total = 0
    for _, feature in _declared_features(document):
        total += 1
        code += bool(feature.code)
        if feature.kind == "python":
            python += 1
        elif feature.kind == "import":
            imported += 1
        elif not feature.code:
            native += 1
    return {
        "classification": "hybrid" if python or code else "native",
        "native_feature_count": native,
        "python_feature_count": python,
        "imported_feature_count": imported,
        "total_feature_count": total,
        "authored_code_feature_count": code,
    }
