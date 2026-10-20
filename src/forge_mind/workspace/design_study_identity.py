"""Stable component targets and external instruments across topology-changing candidates."""

from __future__ import annotations

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.design_study_models import DesignStudySpec, StudyEvaluator
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.store import WorkspaceError


def instrument_identity(spec: DesignStudySpec) -> dict[str, Any]:
    instruments: dict[str, Any] = {}
    for evaluator in spec.evaluators:
        if evaluator.kind == "flow":
            from forge_mind.workspace.flow_solver import solver_executable

            instruments["flow"] = {
                "solver_sha256": file_sha256(solver_executable()),
                "gmsh": version("gmsh"),
            }
        elif evaluator.kind == "structure":
            instruments["structure"] = {
                name: version(name) for name in ("scikit-fem", "scipy", "gmsh")
            }
    return instruments


def _components(
    document: DesignDocument, path: Path
) -> tuple[list[Any], dict[str, tuple[int, str]]]:
    import build123d as b

    from forge_mind.workspace.assembly import expand, named_solids, occurrence_label
    from forge_mind.workspace.topology import same_exported_solid

    shape = b.import_step(str(path))
    solids = list(shape.solids())
    mapping: dict[str, tuple[int, str]] = {}
    if document.assembly:
        names = named_solids(shape)
        for row in expand(document):
            selected = names.get(occurrence_label(row["path"]))
            if selected is None or len(selected.solids()) != 1:
                continue
            matches = [i for i, solid in enumerate(solids) if same_exported_solid(selected, solid)]
            if len(matches) == 1:
                mapping[row["path"]] = (matches[0], row["definition"])
    return solids, mapping


def _interface_targets(
    document: DesignDocument,
    path: Path,
    settings: dict[str, Any],
    bindings: dict[str, Any] | None,
    cancelled: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Native target identity runs in the same cancellable boundary as evaluation."""
    from forge_mind.workspace.document import digest
    from forge_mind.workspace.processing import process_revision

    with TemporaryDirectory(prefix="forge-interface-target-") as directory:
        return process_revision(
            Path(directory),
            {
                "operation": "study_interface_targets",
                "source": str(path.resolve().parent),
                "settings": settings,
                "bindings": bindings,
                "document_hash": digest(document.model_dump()),
                "artifact_hashes": {
                    name: file_sha256(path.parent / name) for name in ("part.step", "document.json")
                },
            },
            None,
            cancelled or (lambda: False),
        )


def bind_targets(
    document: DesignDocument,
    path: Path,
    spec: DesignStudySpec,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for evaluator in spec.evaluators:
        if evaluator.kind == "interface_geometry":
            result[evaluator.kind] = _interface_targets(
                document, path, evaluator.settings, None, cancelled
            )
    physical = [e for e in spec.evaluators if e.kind in {"structure", "flow"}]
    if not physical:
        return result
    solids, mapping = _components(document, path)
    for evaluator in physical:
        target = evaluator.settings.get("target", {})
        index = (
            target.get("body_index")
            if evaluator.kind == "structure"
            else evaluator.settings.get("solid_index")
        )
        occurrence = target.get("occurrence_path")
        if occurrence:
            if occurrence not in mapping:
                raise WorkspaceError(
                    "Study target must resolve uniquely to one exported occurrence"
                )
            index = mapping[occurrence][0]
        if index is None and len(solids) == 1:
            index = 0
        matches = [name for name, pair in mapping.items() if pair[0] == index]
        if len(matches) == 1:
            result[evaluator.kind] = {
                "kind": "occurrence",
                "path": matches[0],
                "definition": mapping[matches[0]][1],
            }
        elif not document.assembly and len(solids) == 1 and index == 0:
            result[evaluator.kind] = {"kind": "whole_model", "outputs": document.outputs}
        else:
            raise WorkspaceError(
                "Index-only multi-body comparisons are ambiguous; select a named assembly occurrence"
            )
    return result


def remap_evaluators(
    document: DesignDocument,
    path: Path,
    spec: DesignStudySpec,
    bindings: dict[str, Any],
    cancelled: Callable[[], bool] | None = None,
) -> list[StudyEvaluator]:
    for evaluator in spec.evaluators:
        if evaluator.kind == "interface_geometry":
            if evaluator.kind not in bindings:
                raise WorkspaceError("Interface study is missing its frozen occurrence targets")
            _interface_targets(
                document, path, evaluator.settings, bindings[evaluator.kind], cancelled
            )
    physical_bindings = {
        key: value for key, value in bindings.items() if key != "interface_geometry"
    }
    if not physical_bindings:
        return spec.evaluators
    solids, mapping = _components(document, path)
    result = []
    for evaluator in spec.evaluators:
        if evaluator.kind not in physical_bindings:
            result.append(evaluator)
            continue
        binding = bindings[evaluator.kind]
        effective = evaluator.model_copy(deep=True)
        if binding["kind"] == "occurrence":
            match = mapping.get(binding["path"])
            if match is None or match[1] != binding["definition"]:
                raise WorkspaceError("Candidate changed or removed the frozen analysis occurrence")
            index = match[0]
        else:
            if document.assembly or len(solids) != 1 or document.outputs != binding["outputs"]:
                raise WorkspaceError(
                    "Candidate changed the identity of the frozen whole-model target"
                )
            index = 0
        if effective.kind == "structure":
            effective.settings["target"] = {"body_index": index, "occurrence_path": None}
        else:
            effective.settings["solid_index"] = index
        result.append(effective)
    return result
