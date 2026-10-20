"""Pure binding checks for optional made-definition meshes; no CAD or file reads."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from copy import deepcopy
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

if TYPE_CHECKING:
    from forge_mind.workspace.document import DesignDocument

MANIFEST_NAME = "definition-meshes.json"
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_HASH = re.compile(r"^[a-f0-9]{64}$")
_REVISION = re.compile(r"^[a-f0-9]{32}$")
Hash = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
PartName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]
XYZ = Annotated[list[float], Field(min_length=3, max_length=3)]
Bounds = Annotated[list[XYZ], Field(min_length=2, max_length=2)]


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Tessellation(_Record):
    linear_deflection: float = Field(gt=0)
    linear_deflection_is_relative: Literal[True]
    angular_deflection: float = Field(gt=0)
    collapsed_triangles_removed: StrictInt = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def strict_relative(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("linear_deflection_is_relative") is not True:
            raise ValueError("linear_deflection_is_relative must be true")
        return data


class _BodyReference(_Record):
    triangle_inventory_matches: bool
    scope: str = Field(min_length=1)


class _FullReference(_Record):
    triangle_inventory_matches: bool
    linear_deflection_setting: float = Field(gt=0)
    linear_deflection_is_relative: Literal[True]
    angular_deflection_rad: float = Field(gt=0)
    scope: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def strict_relative(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("linear_deflection_is_relative") is not True:
            raise ValueError("linear_deflection_is_relative must be true")
        return data


class _Consistency(_Record):
    volume_within_relative_0_001: bool
    bounds_within_0_01_mm: bool
    maximum_sampled_bounds_difference_mm: float = Field(ge=0)
    reference_tessellation: _BodyReference | _FullReference | None
    scope: str = Field(min_length=1)


class _Available(_Record):
    status: Literal["available"]
    role: Literal["make"]
    quantity: StrictInt = Field(ge=0)
    step_file: str
    step_sha256: Hash
    stl_file: str
    stl_sha256: Hash
    stl_bytes: StrictInt = Field(gt=0)
    body_count: StrictInt = Field(gt=0)
    bounds_mm: Bounds
    tessellation: _Tessellation
    export_consistency: _Consistency

    @model_validator(mode="after")
    def checks(self) -> _Available:
        if any(high < low for low, high in zip(*self.bounds_mm, strict=True)):
            raise ValueError("Mesh bounds must have ordered minimum and maximum")
        checks = self.export_consistency
        if not (
            checks.volume_within_relative_0_001
            and (
                checks.bounds_within_0_01_mm
                or (
                    checks.reference_tessellation is not None
                    and checks.reference_tessellation.triangle_inventory_matches
                )
            )
        ):
            raise ValueError("Available mesh requires passed measured export-consistency checks")
        if isinstance(checks.reference_tessellation, _FullReference) and (
            checks.reference_tessellation.linear_deflection_setting
            != self.tessellation.linear_deflection
            or checks.reference_tessellation.angular_deflection_rad
            != self.tessellation.angular_deflection
        ):
            raise ValueError("Reference tessellation settings differ from delivered mesh settings")
        return self


class _Failed(_Record):
    status: Literal["failed"]
    role: Literal["make"]
    quantity: StrictInt = Field(ge=0)
    step_file: str
    step_sha256: Hash | None
    error: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def reason(self) -> _Failed:
        if not self.error.strip():
            raise ValueError("Failed mesh requires a nonempty reason")
        return self


class DefinitionMeshesManifest(_Record):
    """Strict saved manifest schema, including explicit failed selected entries."""

    schema_version: Literal[1]
    revision_id: Annotated[str, Field(pattern=r"^[a-f0-9]{32}$")]
    document_sha256: Hash
    frame: Literal["definition_local"]
    units: Literal["mm"]
    parts: dict[PartName, Annotated[_Available | _Failed, Field(discriminator="status")]]

    @model_validator(mode="before")
    @classmethod
    def version_type(cls, data: Any) -> Any:
        if isinstance(data, dict) and type(data.get("schema_version")) is not int:
            raise ValueError("Definition mesh schema_version must be integer 1")
        return data


def _json_object(data: bytes, label: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate {label} key: {key}")
            result[key] = value
        return result

    if type(data) is not bytes or not data:
        raise ValueError(f"{label} is unavailable")
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(f"Malformed {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"{label} contains nonfinite data")
    return value


def _named(records: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError(f"Document {label} must be a list")
    result = {}
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError(f"Malformed document {label} identifier")
        name = item["id"]
        if not _NAME.fullmatch(name) or name in result:
            raise ValueError(f"Invalid or duplicate document {label} identifier: {name}")
        result[name] = item
    return result


def _selected_counts(document: dict[str, Any]) -> dict[str, int]:
    assembly = document.get("assembly")
    if not isinstance(assembly, dict):
        raise ValueError("Named STL delivery requires an assembly document")
    parts = _named(assembly.get("parts"), "parts")
    groups = _named(assembly.get("subassemblies", []), "subassemblies")
    if set(parts) & set(groups):
        raise ValueError("Part and subassembly definition IDs must be unique")
    selected: dict[str, int] = {}
    for name, part in parts.items():
        production = part.get("production", {})
        if not isinstance(production, dict):
            raise ValueError(f"Part {name}: malformed production declaration")
        deliver = production.get("deliver_stl", False)
        if type(deliver) is not bool:
            raise ValueError(f"Part {name}: deliver_stl must be a strict boolean")
        if deliver:
            if production.get("role") != "make":
                raise ValueError(
                    f"Part {name}: deliver_stl requires an explicitly declared make role"
                )
            selected[name] = 0
    roots = _named(assembly.get("roots"), "root occurrences")
    children = {
        name: _named(group.get("children"), f"subassembly {name} children")
        for name, group in groups.items()
    }
    # Memoized quantities avoid enumerating an exponentially reused group graph.
    # Placement and occurrence-path geometry remain the existing assembly engine's work.
    memo: dict[str, dict[str, int]] = {}
    active: set[str] = set()
    pending: list[tuple[Iterator[dict[str, Any]], str | None, dict[str, int]]] = [
        (iter(roots.values()), None, dict(selected))
    ]
    while pending:
        iterator, group, counts = pending[-1]
        occurrence = next(iterator, None)
        if occurrence is None:
            pending.pop()
            if group is not None:
                memo[group] = counts
                active.remove(group)
            if pending:
                parent_counts = pending[-1][2]
                for name, quantity in counts.items():
                    parent_counts[name] += quantity
            else:
                return counts
            continue
        definition = occurrence.get("definition")
        if not isinstance(definition, str):
            raise ValueError("Occurrence requires a definition identifier")
        if definition in parts:
            if definition in selected:
                counts[definition] += 1
        elif definition in groups:
            if definition in active:
                raise ValueError(f"Cyclic assembly definition: {definition}")
            if definition in memo:
                for name, quantity in memo[definition].items():
                    counts[name] += quantity
            else:
                active.add(definition)
                pending.append((iter(children[definition].values()), definition, dict(selected)))
        else:
            raise ValueError(f"Missing assembly definition: {definition}")
    return selected


def selected_definition_meshes(document: DesignDocument) -> dict[str, int]:
    """Selected make definitions and quantities, without native placement expansion."""
    if document.assembly is None:
        return {}
    return _selected_counts(document.model_dump(mode="json"))


def _bound(data: bytes, name: str, hashes: Mapping[str, str]) -> str:
    if not isinstance(hashes, Mapping):
        raise ValueError("Revision artifact hashes must be a mapping")
    expected = hashes.get(name)
    if not isinstance(expected, str) or not _HASH.fullmatch(expected):
        raise ValueError(f"Missing or malformed revision artifact hash: {name}")
    if type(data) is not bytes or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Stale revision artifact bytes: {name}")
    return expected


def definition_mesh_entry(
    document_bytes: bytes,
    manifest_bytes: bytes,
    revision_id: str,
    part_id: str,
    artifact_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Validate saved source/manifest binding and one available selected mesh entry."""
    if not isinstance(revision_id, str) or not _REVISION.fullmatch(revision_id):
        raise ValueError("Invalid definition mesh revision identifier")
    if not isinstance(part_id, str) or not _NAME.fullmatch(part_id):
        raise ValueError("Invalid definition mesh part identifier")
    document_hash = _bound(document_bytes, "document.json", artifact_hashes)
    document = _json_object(document_bytes, "Document")
    selected = _selected_counts(document)
    if part_id not in selected:
        raise ValueError(f"Definition {part_id}: named STL was not selected")
    if not manifest_bytes or MANIFEST_NAME not in artifact_hashes:
        raise ValueError(f"Definition {part_id}: saved mesh manifest is unavailable")
    _bound(manifest_bytes, MANIFEST_NAME, artifact_hashes)
    raw = _json_object(manifest_bytes, "Definition mesh manifest")
    manifest = DefinitionMeshesManifest.model_validate(raw)
    if manifest.revision_id != revision_id or manifest.document_sha256 != document_hash:
        raise ValueError("Stale definition mesh revision or document binding")
    if set(manifest.parts) != set(selected):
        raise ValueError("Definition mesh manifest must cover exactly every selected definition")
    for name, entry in manifest.parts.items():
        if entry.quantity != selected[name]:
            raise ValueError(f"Definition {name}: stale occurrence quantity")
        step_name = f"definition_{name}.step"
        if entry.step_file != step_name:
            raise ValueError(f"Definition {name}: unexpected STEP filename")
        if name == part_id and entry.step_sha256 != artifact_hashes.get(step_name):
            raise ValueError(f"Definition {name}: stale STEP measurement hash")
        if isinstance(entry, _Available):
            if entry.stl_file != f"definition_{name}.stl":
                raise ValueError(f"Definition {name}: unexpected STL filename")
            if name == part_id and entry.stl_sha256 != artifact_hashes.get(entry.stl_file):
                raise ValueError(f"Definition {name}: stale STL measurement hash")
    chosen = manifest.parts[part_id]
    if isinstance(chosen, _Failed):
        raise ValueError(f"Definition {part_id}: STL generation failed: {chosen.error}")
    return deepcopy(raw["parts"][part_id])


def validate_definition_mesh(
    document_bytes: bytes,
    manifest_bytes: bytes,
    revision_id: str,
    part_id: str,
    artifact_hashes: Mapping[str, str],
    step_bytes: bytes,
    stl_bytes: bytes,
) -> dict[str, Any]:
    """Also bind exact STEP/STL bytes and the declared delivered byte count."""
    entry = definition_mesh_entry(
        document_bytes, manifest_bytes, revision_id, part_id, artifact_hashes
    )
    _bound(step_bytes, entry["step_file"], artifact_hashes)
    _bound(stl_bytes, entry["stl_file"], artifact_hashes)
    if len(stl_bytes) != entry["stl_bytes"]:
        raise ValueError(f"Definition {part_id}: stale STL byte count")
    return entry
