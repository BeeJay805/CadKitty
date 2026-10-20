"""Display selection only. Caller supplies same-invocation verified STEP mapping.

This helper does not authenticate cached evidence or verify geometry. In production,
pass document.model_dump(mode="json") and the result of assembly_evidence for the
same imported STEP used to enumerate the bodies. Keep the full preview alongside
every reference-excluded view; none of these statuses is a physical acceptance.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PreviewSelection:
    included_body_indices: tuple[int, ...]
    omitted_occurrence_ids: tuple[str, ...]
    status: Literal["reference_excluded", "all_reference", "no_reference", "full_context_fallback"]
    diagnostic: str

    @property
    def omitted_count(self) -> int:
        return len(self.omitted_occurrence_ids)


def select_preview_bodies(
    document: Mapping[str, object] | None,
    verified_assembly: Mapping[str, object] | None,
    body_count: int,
) -> PreviewSelection:
    """Select only through explicit production roles and a complete body mapping."""
    if type(body_count) is not int or body_count < 0:
        raise ValueError("body_count must be a nonnegative integer")
    all_bodies = tuple(range(body_count))

    def fallback(reason: str) -> PreviewSelection:
        return PreviewSelection(all_bodies, (), "full_context_fallback", reason)

    if not isinstance(document, Mapping) or not isinstance(document.get("assembly"), Mapping):
        return fallback("No typed assembly role mapping; complete context retained.")
    spec = document["assembly"]
    assert isinstance(spec, Mapping)
    parts = spec.get("parts")
    if not isinstance(parts, list):
        return fallback("Invalid part definitions; complete context retained.")
    references: set[str] = set()
    definitions: set[str] = set()
    for part in parts:
        if not isinstance(part, Mapping):
            return fallback("Invalid part definition; complete context retained.")
        identifier = part.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in definitions:
            return fallback("Missing or duplicate definition identity; complete context retained.")
        definitions.add(identifier)
        production = part.get("production")
        if isinstance(production, Mapping) and production.get("role") == "reference":
            references.add(identifier)
    if not isinstance(verified_assembly, Mapping):
        return fallback("No verified assembly mapping; complete context retained.")
    rows = verified_assembly.get("occurrences")
    if not isinstance(rows, list) or len(rows) != body_count:
        return fallback("Incomplete occurrence coverage; complete context retained.")
    if "occurrence_count" in verified_assembly:
        count = verified_assembly["occurrence_count"]
        if type(count) is not int or count != body_count:
            return fallback("Conflicting occurrence count; complete context retained.")
    paths: set[str] = set()
    indices: set[int] = set()
    omitted: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return fallback("Invalid occurrence mapping; complete context retained.")
        path, definition, index = (
            row.get("path"),
            row.get("definition"),
            row.get("body_index"),
        )
        if not isinstance(path, str) or not path or path in paths:
            return fallback("Missing or duplicate occurrence identity; complete context retained.")
        if not isinstance(definition, str) or definition not in definitions:
            return fallback("Unmapped occurrence definition; complete context retained.")
        if type(index) is not int or not 0 <= index < body_count or index in indices:
            return fallback("Invalid or duplicate body identity; complete context retained.")
        paths.add(path)
        indices.add(index)
        if definition in references:
            omitted[index] = path
    # Unique, in-range indices and exactly body_count rows establish full coverage.
    included = tuple(index for index in all_bodies if index not in omitted)
    omitted_paths = tuple(omitted[index] for index in sorted(omitted))
    if not omitted:
        return PreviewSelection(
            included, (), "no_reference", "No declared reference bodies omitted."
        )
    if not included:
        return PreviewSelection(
            (),
            omitted_paths,
            "all_reference",
            "All bodies are declared references; no inspection geometry.",
        )
    return PreviewSelection(
        included,
        omitted_paths,
        "reference_excluded",
        "Declared reference bodies omitted for inspection only; complete context is a separate view.",
    )
