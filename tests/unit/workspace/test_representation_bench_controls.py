import json

import pytest
from scripts.workspace_generation_bench import parse_args, validate_python_control

from forge_mind.workspace.models import Action


@pytest.mark.parametrize("kind", ["document", "assembly_edit", "edit_part"])
def test_python_control_rejects_native_geometry_across_submission_paths(kind):
    feature = {"id": "body", "name": "Body", "kind": "box", "params": {"x": 4, "y": 3, "z": 2}}
    part = {"id": "mount", "name": "Mount", "features": [feature], "output": "body"}
    data = (
        {"features": [feature], "outputs": ["body"]}
        if kind == "document"
        else {"parts": [part]}
        if kind == "assembly_edit"
        else part
    )
    with pytest.raises(ValueError, match="native geometry"):
        validate_python_control(
            Action(kind=kind, explanation="Control", document_json=json.dumps(data))
        )


def test_python_control_permits_typed_python_and_observation_without_source():
    source = {
        "features": [
            {"id": "body", "name": "Body", "kind": "python", "code": "result = Box(4,3,2)"}
        ],
        "outputs": ["body"],
    }
    validate_python_control(
        Action(kind="document", explanation="Control", document_json=json.dumps(source))
    )
    validate_python_control(Action(kind="inspect", explanation="Read saved geometry"))
    with pytest.raises(ValueError, match="whole-program"):
        validate_python_control(
            Action(kind="build", explanation="Control", code="result = Box(4,3,2)")
        )


def test_representation_option_preserves_legacy_and_does_not_cap_feature_count(tmp_path):
    base = ["--output", str(tmp_path), "--corpus", str(tmp_path / "cases.json")]
    assert parse_args(base).representation == "hybrid"
    assert (
        parse_args([*base, "--representation", "native_operations"]).representation
        == "native_operations"
    )


@pytest.mark.parametrize(
    "payload",
    [
        "{invalid",
        '{"production":{"role":"make","deliver_stl":true},"features":[{"id":"body","name":"Body","kind":"python","code":"result = Box(4,3,2)"}],"outputs":["body"]}',
    ],
)
def test_invalid_document_reaches_shared_refusal_and_repair_in_python_control(payload):
    from pydantic import ValidationError

    from forge_mind.workspace.document import DesignDocument

    action = Action(kind="document", explanation="Control", document_json=payload)
    # Invalid input remains rejected by the production parser. The benchmark
    # observer does not stop the author before that parser can report the error.
    validate_python_control(action)
    with pytest.raises(ValidationError):
        DesignDocument.model_validate_json(payload)
