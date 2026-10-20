"""Typed production intent controls preparation, without CAD execution."""

from copy import deepcopy

import pytest

from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.native_preflight import preflight_context, print_preparation


@pytest.mark.parametrize(
    "role,process,mesh,required",
    [
        ("make", "additive", True, True),
        ("make", "additive", False, True),
        ("make", "cnc", True, False),
        ("make", "unspecified", True, False),
        ("buy", "additive", False, False),
        ("reference", "additive", False, False),
    ],
)
def test_preparation_uses_typed_additive_intent_not_mesh_extension(role, process, mesh, required):
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "made",
                        "name": "Body",
                        "features": [{"id": "base", "name": "Body", "kind": "box"}],
                        "output": "base",
                        "production": {"role": role, "process": process, "deliver_stl": mesh},
                    }
                ],
                "roots": [{"id": "installed", "definition": "made"}],
            },
        }
    )
    before = document.model_dump()
    result = print_preparation(document, {})
    assert result["required"] is required
    assert result["status"] == ("missing" if required else "not_requested")
    assert document.model_dump() == before


def test_unused_additive_definition_does_not_require_printing():
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": part_id,
                        "name": part_id,
                        "features": [{"id": "body", "name": "Body", "kind": "box"}],
                        "output": "body",
                        "production": {"role": "make", "process": process},
                    }
                    for part_id, process in (("active", "cnc"), ("unused", "additive"))
                ],
                "roots": [{"id": "installed", "definition": "active"}],
            }
        }
    )
    assert not print_preparation(document, {})["required"]


def test_frozen_explicit_prepared_package_request_requires_current_files_without_additive_hint():
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "outputs": ["body"],
            "production": {"role": "make", "process": "cnc"},
        }
    )
    assert not print_preparation(document, {})["required"]
    requested = print_preparation(document, {}, explicitly_requested=True)
    assert requested["required"] and requested["part_ids"] == ["part"]
    assert requested["status"] == "conversion_required"


@pytest.mark.parametrize("status", ["current", "stale", "missing"])
def test_preparation_preserves_trusted_receipt_state(status):
    document = DesignDocument.model_validate(
        {
            "assembly": {
                "parts": [
                    {
                        "id": "made",
                        "name": "Body",
                        "features": [{"id": "base", "name": "Body", "kind": "box"}],
                        "output": "base",
                        "production": {"role": "make", "process": "additive"},
                    }
                ],
                "roots": [{"id": "installed", "definition": "made"}],
            },
        }
    )
    guide = {"facts": {"print_setup": {"status": status}}}
    before = deepcopy(guide)
    assert print_preparation(document, guide)["status"] == status
    assert guide == before


def test_flat_additive_preparation_requires_existing_assembly_conversion():
    document = DesignDocument.model_validate(
        {
            "features": [{"id": "base", "name": "Body", "kind": "box"}],
            "outputs": ["base"],
            "production": {"role": "make", "process": "additive"},
        }
    )
    result = print_preparation(document, {})
    assert result["required"] and not result["supported"]
    assert result["status"] == "conversion_required"
    assert "document with plan_patch_json" in result["recovery"]


def test_preflight_retains_blockers_without_copying_catalog_or_check_evidence():
    review = {
        "revision_id": "a" * 32,
        "plan_hash": "b" * 64,
        "work_reasons": ["Current check failed", "Prepare current print files"],
        "instruction_binding": {"facts": ["large catalog"]},
        "requirement_findings": ["complete failed witness"],
    }
    result = preflight_context(review)
    assert result["work_reasons"] == review["work_reasons"]
    assert result["work_reasons"] is not review["work_reasons"]
    assert "instruction_binding" not in result and "requirement_findings" not in result
