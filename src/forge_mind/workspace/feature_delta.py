"""Pure feature edits against an exact document; application belongs to the service."""

from __future__ import annotations

from pydantic import Field, model_validator

from forge_mind.workspace.assembly import preserved_assembly
from forge_mind.workspace.document import DesignDocument, Feature, Name
from forge_mind.workspace.edit_contracts import EditBasis
from forge_mind.workspace.models import Record


class FeatureChange(Record):
    part_id: Name | None = None
    append: list[Feature] = Field(default_factory=list)
    update: list[Feature] = Field(default_factory=list)
    output: Name | None = None

    @model_validator(mode="after")
    def explicit_changes(self) -> FeatureChange:
        if not self.append and not self.update and self.output is None:
            raise ValueError("A feature change must append, update or select an output")
        ids = [feature.id for feature in [*self.append, *self.update]]
        if len(ids) != len(set(ids)):
            raise ValueError("Feature change IDs must be unique across append and update")
        return self


class FeatureDelta(Record):
    basis: EditBasis
    changes: list[FeatureChange] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_scopes(self) -> FeatureDelta:
        scopes = [change.part_id for change in self.changes]
        if len(scopes) != len(set(scopes)):
            raise ValueError("A feature delta must change each part scope only once")
        return self


def _features(previous: list[Feature], change: FeatureChange) -> list[Feature]:
    known = {feature.id for feature in previous}
    appended = {feature.id for feature in change.append}
    updated = {feature.id for feature in change.update}
    if appended & known:
        raise ValueError(f"Appended Feature IDs already exist: {sorted(appended & known)}")
    if updated - known:
        raise ValueError(f"Updated Feature IDs do not exist: {sorted(updated - known)}")
    replacements = {feature.id: feature for feature in change.update}
    return [replacements.get(feature.id, feature).model_copy(deep=True) for feature in previous] + [
        feature.model_copy(deep=True) for feature in change.append
    ]


def _check_outputs(features: list[Feature], outputs: list[str]) -> None:
    """Reject suppressed output dependencies without changing feature order or inputs."""
    by_id = {feature.id: feature for feature in features}
    pending = list(outputs or ([features[-1].id] if features else []))
    seen: set[str] = set()
    while pending:
        feature_id = pending.pop()
        if feature_id in seen:
            continue
        seen.add(feature_id)
        feature = by_id[feature_id]
        if feature.suppressed:
            raise ValueError(f"Output or dependency is suppressed: {feature_id}")
        pending.extend(feature.inputs)


def apply_feature_delta(document: DesignDocument, delta: FeatureDelta) -> DesignDocument:
    """Return a validated copy; revision/head/version comparisons stay service-owned."""
    original = document.model_copy(deep=True)
    delta = FeatureDelta.model_validate(delta.model_dump())
    if delta.basis.revision_id is None or delta.basis.document_hash != original.fingerprint():
        raise ValueError("Feature delta document basis changed; refresh the source revision")
    result = original.model_copy(deep=True)
    for change in delta.changes:
        if result.assembly is None:
            if change.part_id is not None:
                raise ValueError("A flat document feature change requires a null part scope")
            if change.append and not result.outputs and change.output is None:
                raise ValueError("Appending to implicit outputs requires an explicit output")
            result.features = _features(result.features, change)
            if change.output is not None:
                result.outputs = [change.output]
        else:
            part = next((part for part in result.assembly.parts if part.id == change.part_id), None)
            if part is None:
                raise ValueError("An assembly feature change requires an existing part scope")
            part.features = _features(part.features, change)
            if change.output is not None:
                part.output = change.output
    result = DesignDocument.model_validate(result.model_dump())
    if result.assembly is None:
        _check_outputs(result.features, result.outputs)
    else:
        for part in result.assembly.parts:
            _check_outputs(part.features, [part.output])
    retained, reason = preserved_assembly(original.model_dump(), result.model_dump())
    if not retained:
        raise ValueError(reason)
    return result
