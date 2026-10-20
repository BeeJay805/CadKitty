"""Authoring representation checks do not execute CAD or count wrapper source."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from forge_mind.workspace.document import (
    AssemblySpec,
    DesignDocument,
    Feature,
    Occurrence,
    PartDefinition,
    Subassembly,
)
from forge_mind.workspace.store import WorkspaceError


@pytest.fixture
def policy():
    # Import inside the test so a missing implementation is an assertion failure.
    name = "forge_mind.workspace.authoring_policy"
    assert importlib.util.find_spec(name) is not None, "Authoring policy module is missing"
    return importlib.import_module(name)


def feature(kind="box", *, identifier="body", code="", suppressed=False):
    return Feature(
        id=identifier,
        name=identifier,
        kind=kind,
        code=code,
        suppressed=suppressed,
        params={"x": 10, "y": 8, "z": 4} if kind == "box" else {},
        asset_id="a" * 32 if kind == "import" else None,
    )


def document(*features):
    return DesignDocument(features=list(features), outputs=[features[0].id])


def assembly_document(hidden):
    visible = PartDefinition(id="visible", name="Visible", features=[feature()], output="body")
    retained = PartDefinition(id="retained", name="Retained", features=hidden, output="body")
    # Construct without CAD validation; this suite checks policy, not geometry.
    return DesignDocument.model_construct(
        assembly=AssemblySpec(
            parts=[visible, retained],
            subassemblies=[
                Subassembly(
                    id="group",
                    name="Nested group",
                    children=[Occurrence(id="instance", definition="retained")],
                )
            ],
            roots=[Occurrence(id="visible_instance", definition="visible")],
        )
    )


@pytest.mark.parametrize("value", [None, "hybrid", "native_operations"])
def test_optional_mode_keeps_legacy_default(policy, value):
    assert policy.validate_authoring_mode(value) == ("hybrid" if value is None else value)


@pytest.mark.parametrize("value", ["", "native", "HYBRID", " hybrid ", False, 1, []])
def test_unknown_mode_fails_explicitly(policy, value):
    with pytest.raises(WorkspaceError, match="authoring_mode"):
        policy.validate_authoring_mode(value)


@pytest.mark.parametrize("suppressed", [False, True])
def test_native_rejects_retained_python_and_identifies_feature(policy, suppressed):
    doc = document(
        feature(),
        feature(
            "python", identifier="custom", code="result = b.Box(1, 2, 3)", suppressed=suppressed
        ),
    )
    before = doc.model_dump()
    with pytest.raises(WorkspaceError, match="custom"):
        policy.enforce_document_policy(doc, "native_operations")
    assert doc.model_dump() == before


@pytest.mark.parametrize("code", ["result = b.Box(1, 2, 3)", " ", "\n"])
def test_native_rejects_nonempty_code_even_on_nonpython_feature(policy, code):
    doc = document(feature())
    doc.features[0].code = code
    with pytest.raises(WorkspaceError, match="body"):
        policy.enforce_document_policy(doc, "native_operations")


def test_native_rejects_python_kind_without_source_in_legacy_document(policy):
    doc = DesignDocument.model_construct(features=[feature("python")], outputs=["body"])
    with pytest.raises(WorkspaceError, match="body"):
        policy.enforce_document_policy(doc, "native_operations")


@pytest.mark.parametrize("nested", [False, True])
def test_native_checks_unused_and_nested_assembly_definitions(policy, nested):
    doc = assembly_document([feature("python", code="result = b.Box(1, 2, 3)")])
    if nested:
        doc.assembly.roots.append(Occurrence(id="nested", definition="group"))
    with pytest.raises(WorkspaceError, match=r"retained.*body"):
        policy.enforce_document_policy(doc, "native_operations")


def test_native_checks_nonempty_code_in_retained_assembly_feature(policy):
    doc = assembly_document([feature(), feature(identifier="later", suppressed=True)])
    doc.assembly.parts[1].features[1].code = "pass"
    with pytest.raises(WorkspaceError, match=r"retained.*later"):
        policy.enforce_document_policy(doc, "native_operations")


def test_hybrid_preserves_python_document_and_source(policy):
    doc = document(feature("python", code="result = b.Box(1, 2, 3)"))
    before = doc.model_dump()
    policy.enforce_document_policy(doc, "hybrid")
    assert doc.model_dump() == before


def test_native_allows_imports_and_has_no_feature_count_cap(policy):
    doc = document(feature("import"), *(feature(identifier=f"f{i}") for i in range(600)))
    policy.enforce_document_policy(doc, "native_operations")


def test_policy_does_not_use_an_operation_allowlist(policy):
    doc = document(feature())
    doc.features[0] = doc.features[0].model_copy(update={"kind": "future_native_operation"})
    policy.enforce_document_policy(doc, "native_operations")


@pytest.mark.parametrize("kind", ["build", "edit_part", "document", "inspect"])
def test_native_rejects_any_action_with_authored_code(policy, kind):
    with pytest.raises(WorkspaceError, match=kind):
        policy.enforce_action_policy(kind, "native_operations", code=" ")


def test_native_rejects_whole_program_build_before_execution(policy):
    with pytest.raises(WorkspaceError, match="build"):
        policy.enforce_action_policy("build", "native_operations")


@pytest.mark.parametrize("kind", ["edit_part", "document", "assembly_edit", "inspect", "restore"])
def test_native_allows_structured_and_read_actions(policy, kind):
    policy.enforce_action_policy(kind, "native_operations")


def test_hybrid_allows_authored_build_and_part_routes(policy):
    for kind in ("build", "edit_part"):
        policy.enforce_action_policy(kind, "hybrid", code="result = b.Box(1, 2, 3)")


def test_invalid_policy_never_silently_disables_enforcement(policy):
    with pytest.raises(WorkspaceError, match="authoring_mode"):
        policy.enforce_document_policy(document(feature()), "unsupported")
    with pytest.raises(WorkspaceError, match="authoring_mode"):
        policy.enforce_action_policy("inspect", "unsupported")


def test_classification_reports_imports_as_separate_geometry(policy):
    assert policy.source_classification(
        document(feature(), feature("import", identifier="input"))
    ) == {
        "classification": "native",
        "native_feature_count": 1,
        "python_feature_count": 0,
        "imported_feature_count": 1,
        "total_feature_count": 2,
        "authored_code_feature_count": 0,
    }


def test_classification_counts_declarations_not_occurrences(policy):
    doc = assembly_document(
        [feature(), feature("python", identifier="code", code="pass", suppressed=True)]
    )
    doc.assembly.roots.extend(
        [
            Occurrence(id="second", definition="visible"),
            Occurrence(id="third", definition="visible"),
        ]
    )
    result = policy.source_classification(doc)
    assert result == {
        "classification": "hybrid",
        "native_feature_count": 2,
        "python_feature_count": 1,
        "imported_feature_count": 0,
        "total_feature_count": 3,
        "authored_code_feature_count": 1,
    }


def test_classification_flags_nonpython_authored_source(policy):
    doc = document(feature())
    doc.features[0].code = " "
    result = policy.source_classification(doc)
    assert result["classification"] == "hybrid"
    assert result["native_feature_count"] == 0
    assert result["authored_code_feature_count"] == 1


def test_real_legacy_python_document_is_still_accepted_in_hybrid(policy):
    path = Path(__file__).parent / "fixtures" / "hardware_legacy_document.json"
    doc = DesignDocument.model_validate_json(path.read_text(encoding="utf-8"))
    policy.enforce_document_policy(doc, "hybrid")
    with pytest.raises(WorkspaceError, match="native_operations"):
        policy.enforce_document_policy(doc, "native_operations")
    assert policy.source_classification(doc)["python_feature_count"] > 0
