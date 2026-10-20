"""Named saved-solid interface measurements, without acceptance or physical claims."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from pydantic import Field, StrictStr, field_validator, model_validator

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.models import Record

SCOPE = (
    "Static nominal geometry from the exact saved STEP in CAD world millimetres. "
    "Minimum distance does not distinguish contact, overlap or containment; overlap "
    "volume is measured separately. No continuous-motion, manufacturing, physical-function "
    "or acceptance qualification."
)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Interface paths, basis and requirement IDs must not be blank")
    return value


class InterfacePair(Record):
    id: StrictStr = Field(pattern=r"^[a-z][a-z0-9_]*$")
    a: StrictStr
    b: StrictStr
    basis: StrictStr
    requirement_ids: list[StrictStr] = Field(default_factory=list)

    @field_validator("a", "b", "basis")
    @classmethod
    def nonblank(cls, value: str) -> str:
        return _nonblank(value)

    @model_validator(mode="after")
    def references(self) -> InterfacePair:
        if self.a == self.b:
            raise ValueError("Interface pair requires two distinct occurrence paths")
        if len(set(self.requirement_ids)) != len(self.requirement_ids):
            raise ValueError("Interface pair requirement IDs must be unique")
        for requirement_id in self.requirement_ids:
            _nonblank(requirement_id)
        return self


class InterfaceGeometrySettings(Record):
    pairs: list[InterfacePair] = Field(min_length=1)
    fixed_occurrence_paths: list[StrictStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique(self) -> InterfaceGeometrySettings:
        if len({pair.id for pair in self.pairs}) != len(self.pairs):
            raise ValueError("Interface pair IDs must be unique")
        pairs = {tuple(sorted((pair.a, pair.b))) for pair in self.pairs}
        if len(pairs) != len(self.pairs):
            raise ValueError("Interface occurrence pairs must be unique, including reversed pairs")
        if len(set(self.fixed_occurrence_paths)) != len(self.fixed_occurrence_paths):
            raise ValueError("Fixed occurrence paths must be unique")
        for path in self.fixed_occurrence_paths:
            _nonblank(path)
        return self


def interface_metrics(settings: dict[str, Any] | None) -> list[dict[str, str]]:
    """Discover only the explicit named measurements in the frozen settings."""
    if settings is None:
        return []
    parsed = InterfaceGeometrySettings.model_validate(settings)
    return [
        {
            "key": f"interface_geometry.{pair.id}_{suffix}",
            "label": f"{pair.id}: {label}",
            "unit": unit,
        }
        for pair in parsed.pairs
        for suffix, label, unit in (
            ("distance_mm", "Minimum saved-solid distance", "mm"),
            ("overlap_mm3", "Saved-solid overlap volume", "mm³"),
        )
    ]


def _source_hashes(source: Path) -> dict[str, str]:
    result = {}
    for name in ("part.step", "document.json"):
        path = source / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("Interface geometry requires regular saved " + name)
        result[name] = file_sha256(path)
    return result


def _unchanged(source: Path, expected: dict[str, str]) -> None:
    if _source_hashes(source) != expected:
        raise ValueError("Interface geometry source changed during measurement")


def _placement(location: Any) -> list[list[float]]:
    transform = location.Transformation() if hasattr(location, "Transformation") else location
    values = [[float(transform.Value(row, col)) for col in range(1, 5)] for row in range(1, 4)]
    if not all(math.isfinite(value) for row in values for value in row):
        raise ValueError("Interface occurrence placement is nonfinite")
    return values


def _load(document: Any, source: Path) -> tuple[Any, dict[str, dict[str, Any]], dict[str, str]]:
    """Import once and require an unambiguous named body for every declared occurrence."""
    import build123d as cad
    from OCP.BRepCheck import BRepCheck_Analyzer

    from forge_mind.workspace.assembly import expand, named_solids, occurrence_label
    from forge_mind.workspace.document import DesignDocument, digest
    from forge_mind.workspace.topology import same_exported_solid

    hashes = _source_hashes(source)
    saved = DesignDocument.model_validate_json((source / "document.json").read_bytes())
    parsed = (
        saved
        if document is None
        else (
            document
            if isinstance(document, DesignDocument)
            else DesignDocument.model_validate(document)
        )
    )
    if digest(parsed.model_dump(mode="json")) != digest(saved.model_dump(mode="json")):
        raise ValueError("Interface document does not match its saved source")
    if parsed.assembly is None:
        raise ValueError("Interface geometry requires named assembly occurrences")
    rows = expand(parsed)
    try:
        shape = cad.import_step(source / "part.step")
    except Exception as exc:
        raise ValueError("Interface saved STEP cannot be imported") from exc
    if (
        shape is None
        or shape.wrapped is None
        or shape.wrapped.IsNull()
        or not BRepCheck_Analyzer(shape.wrapped).IsValid()
    ):
        raise ValueError("Interface saved STEP is invalid or empty")
    solids = list(shape.solids())
    named = named_solids(shape)
    labels = [occurrence_label(row["path"]) for row in rows]
    if (
        len(set(labels)) != len(labels)
        or set(named) != set(labels)
        or len(solids) != len(rows)
        or not solids
    ):
        raise ValueError("Interface named occurrence inventory is missing or ambiguous")
    definitions = {part.id: part for part in parsed.assembly.parts}
    result = {}
    used: set[int] = set()
    for row in rows:
        solid = named[occurrence_label(row["path"])]
        matches = [index for index, body in enumerate(solids) if same_exported_solid(solid, body)]
        if len(matches) != 1 or matches[0] in used:
            raise ValueError("Interface saved occurrence correspondence is ambiguous")
        if not BRepCheck_Analyzer(solid.wrapped).IsValid():
            raise ValueError("Interface saved occurrence is invalid")
        used.add(matches[0])
        result[row["path"]] = {
            "definition": row["definition"],
            "role": definitions[row["definition"]].production.role,
            "placement": _placement(row["location"].wrapped),
            "export_placement": _placement(solid.wrapped.Location()),
            "body_index": matches[0],
            "solid": solid,
        }
    _unchanged(source, hashes)
    return parsed, result, hashes


def _selected(
    settings: InterfaceGeometrySettings, occurrences: dict[str, dict[str, Any]]
) -> list[str]:
    paths = sorted({path for pair in settings.pairs for path in (pair.a, pair.b)})
    missing = sorted((set(paths) | set(settings.fixed_occurrence_paths)) - occurrences.keys())
    if missing:
        raise ValueError("Interface occurrence paths are unavailable: " + ", ".join(missing))
    return paths


def bind_interface_targets(document: Any, path: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Freeze selected ownership and all reference/fixed geometry, never acceptance."""
    from forge_mind.workspace.document import digest
    from forge_mind.workspace.topology import signature

    parsed = InterfaceGeometrySettings.model_validate(settings)
    _, occurrences, hashes = _load(document, path)
    selected = _selected(parsed, occurrences)
    references = sorted(key for key, row in occurrences.items() if row["role"] == "reference")
    fixed = sorted(set(references) | set(parsed.fixed_occurrence_paths))
    result = {
        "version": 1,
        "settings_hash": digest(parsed.model_dump(mode="json")),
        "selected": {
            key: {field: occurrences[key][field] for field in ("definition", "role")}
            for key in selected
        },
        "fixed": {
            key: {
                **{
                    field: occurrences[key][field]
                    for field in ("definition", "role", "placement", "export_placement")
                },
                "signature": signature(occurrences[key]["solid"]),
            }
            for key in fixed
        },
        "reference_occurrence_paths": references,
        "source_hashes": hashes,
        "qualification_status": "unverified",
        "scope": SCOPE,
    }
    _unchanged(path, hashes)
    return result


def validate_interface_targets(
    document: Any, path: Path, settings: dict[str, Any], bindings: dict[str, Any]
) -> None:
    """Candidates may change working geometry, but never retarget or move fixed subjects."""
    from forge_mind.workspace.document import digest

    current = bind_interface_targets(document, path, settings)
    if not isinstance(bindings, dict) or set(bindings) != set(current):
        raise ValueError("Interface target bindings are incomplete")
    for field in (
        "version",
        "settings_hash",
        "selected",
        "reference_occurrence_paths",
        "fixed",
        "qualification_status",
        "scope",
    ):
        if digest(bindings[field]) != digest(current[field]):
            raise ValueError("Interface frozen target changed: " + field)
    hashes = bindings["source_hashes"]
    if (
        not isinstance(hashes, dict)
        or set(hashes) != {"part.step", "document.json"}
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes.values()
        )
    ):
        raise ValueError("Interface baseline source hashes are invalid")


def interface_geometry_worker(source: Path, settings: dict[str, Any]) -> dict[str, Any]:
    """Measure all pairs or raise; failure never returns partial scores."""
    from forge_mind.workspace.geometry_distance_native import _distance_operation, _distance_result
    from forge_mind.workspace.geometry_results import boolean_overlap_components, boolean_volume

    parsed = InterfaceGeometrySettings.model_validate(settings)
    _, occurrences, hashes = _load(None, source)
    _selected(parsed, occurrences)
    metrics: dict[str, float] = {}
    pairs = []
    for pair in parsed.pairs:
        left, right = occurrences[pair.a], occurrences[pair.b]
        operation = _distance_operation()
        operation.LoadS1(left["solid"].wrapped)
        operation.LoadS2(right["solid"].wrapped)
        operation.Perform()
        distance = _distance_result(operation)
        common = left["solid"].intersect(right["solid"])
        overlap = boolean_volume(common)
        components = boolean_overlap_components(common)
        a_volume = boolean_volume(left["solid"])
        b_volume = boolean_volume(right["solid"])
        if a_volume <= 0 or b_volume <= 0:
            raise ValueError("Interface saved occurrence has no positive solid volume")
        metrics[f"{pair.id}_distance_mm"] = distance["minimum_distance_mm"]
        metrics[f"{pair.id}_overlap_mm3"] = overlap
        pairs.append(
            {
                **pair.model_dump(mode="json"),
                **distance,
                "a_definition": left["definition"],
                "b_definition": right["definition"],
                "a_role": left["role"],
                "b_role": right["role"],
                "a_body_index": left["body_index"],
                "b_body_index": right["body_index"],
                "a_volume_mm3": a_volume,
                "b_volume_mm3": b_volume,
                "overlap_mm3": overlap,
                "overlap_components": components,
            }
        )
    _unchanged(source, hashes)
    return {
        "status": "complete",
        "eligible": True,
        "metrics": metrics,
        "pairs": pairs,
        "source_hashes": hashes,
        "qualification_status": "unverified",
        "scope": SCOPE,
    }
