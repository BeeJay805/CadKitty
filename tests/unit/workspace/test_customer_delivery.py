"""Customer status is a pure view of receipts, never delivery or physical proof."""

from __future__ import annotations

from copy import deepcopy

import pytest

from forge_mind.workspace.customer_delivery import customer_delivery_view


def context() -> dict:
    return {
        "status": "current",
        "basis": {"revision_id": "a" * 32},
        "basis_hash": "b" * 64,
        "physical_readiness": "approved",  # Presentation never grants physical authority.
        "facts": {
            "package_files": {
                "geometry": {
                    "make/definition_holder.stl": {
                        "part_id": "holder",
                        "role": "make",
                        "quantity": 1,
                        "source_present": True,
                    },
                    "assembly/part.step": {
                        "part_id": None,
                        "role": "overall_review",
                        "quantity": 1,
                        "source_present": True,
                    },
                },
                "metadata_files": ["document.json", "README.md", "manifest.json"],
                "excluded_saved_artifacts": ["definition_remote.step"],
            },
            "print_setup": {
                "status": "current",
                "files": {
                    "print/definition_holder.stl": {
                        "quantity": 1,
                        "sha256": "c" * 64,
                        "frame": "plate_z_up",
                    }
                },
                "package": {"status": "prepared", "sha256": "d" * 64},
            },
            "requirement_findings": [
                {"requirement_id": "thin_wall", "status": "fail", "reason": "Wall is too thin."},
                {
                    "requirement_id": "fit",
                    "status": "unverified",
                    "reason": "Actual fit is unknown.",
                },
            ],
            "functional_interfaces": {
                "rows": [{"id": "mount", "status": "check_failed", "duties": []}],
            },
            "functional_coverage": {
                "external_requirements": [
                    {
                        "requirement_id": "print",
                        "unsupported_checks": ["Prepare files and verify delivery."],
                        "missing_inputs": ["Actual printer profile."],
                        "physical_validation": ["Inspect the first print."],
                    }
                ]
            },
        },
    }


def test_available_source_is_not_included_or_delivered_without_actual_members():
    view = customer_delivery_view(context())
    assert view["project_files"]["status"] == "available"
    assert all(row["included"] is None for row in view["project_files"]["files"])
    assert view["print_files"]["status"] == "current"
    assert view["print_files"]["package"]["status"] == "prepared"
    assert view["physical_readiness"] == "unverified"
    assert set(view) == {
        "project_files",
        "print_files",
        "guidance",
        "instruction_binding",
        "procedure_proposal",
        "checks",
        "physical_readiness",
        "scope",
    }
    assert view["instruction_binding"]["status"] == "legacy_unverified"
    assert view["instruction_binding"]["steps"] == []
    assert view["instruction_binding"]["physical_readiness"] == "unverified"
    assert view["procedure_proposal"]["status"] == "unavailable"
    assert view["procedure_proposal"]["procedures"] == {}
    assert view["checks"]["motion"] == {}


def test_actual_omissions_do_not_change_separate_print_or_guidance_status():
    view = customer_delivery_view(context(), members={"document.json", "assembly/part.step"})
    assert view["project_files"]["status"] == "partial"
    assert view["project_files"]["omitted"] == ["make/definition_holder.stl"]
    assert view["project_files"]["excluded"] == ["definition_remote.step"]
    assert view["print_files"]["status"] == "current"
    assert "print/definition_holder.stl" in view["print_files"]["files"]
    assert view["guidance"]["status"] == "current"
    included = {row["path"]: row["included"] for row in view["project_files"]["files"]}
    assert included == {"assembly/part.step": True, "make/definition_holder.stl": False}


def test_missing_source_and_omitted_metadata_cannot_be_presented_as_complete_inclusion():
    original = context()
    original["facts"]["package_files"]["geometry"]["make/definition_holder.stl"][
        "source_present"
    ] = False
    assert customer_delivery_view(original)["project_files"]["status"] == "incomplete"
    view = customer_delivery_view(original, members={"assembly/part.step"})
    assert view["project_files"]["status"] == "partial"
    assert view["project_files"]["omitted"] == ["document.json", "make/definition_holder.stl"]


def test_inclusion_remains_separate_from_capture_readback_and_requirement_proof():
    original = context()
    original["automatic_delivery"] = {"status": "delivered", "readback_verified": True}
    view = customer_delivery_view(
        original,
        members={"document.json", "assembly/part.step", "make/definition_holder.stl"},
    )
    assert view["project_files"]["status"] == "included"
    assert "automatic_delivery" not in view
    assert "delivered" not in {
        facet.get("status") for facet in view.values() if isinstance(facet, dict)
    }
    assert view["checks"]["failed"] == original["facts"]["requirement_findings"][:1]
    assert view["checks"]["unverified"] == original["facts"]["requirement_findings"][1:]
    assert view["checks"]["interfaces"]["rows"][0]["status"] == "check_failed"
    assert view["checks"]["coverage"] == original["facts"]["functional_coverage"]


@pytest.mark.parametrize("status", ["current", "stale", "missing", "tampered"])
def test_print_status_preserves_server_scope_and_never_upgrades_bad_evidence(status):
    original = context()
    original["facts"]["print_setup"]["status"] = status
    view = customer_delivery_view(original)
    assert view["print_files"]["status"] == ("unverified" if status == "tampered" else status)
    assert view["guidance"]["status"] == "current"  # Independent existing facet, not a waiver.


@pytest.mark.parametrize("status", ["current", "stale", "unbound"])
def test_guidance_identity_and_freshness_are_preserved_without_textual_approval(status):
    original = context()
    original["status"] = status
    view = customer_delivery_view(original)
    assert view["guidance"]["status"] == status
    assert view["guidance"]["revision_id"] == "a" * 32
    assert view["guidance"]["basis_hash"] == "b" * 64
    assert view["physical_readiness"] == "unverified"


def test_missing_context_is_explicit_and_view_does_not_alias_or_mutate_inputs():
    absent = customer_delivery_view(None)
    assert absent["project_files"]["status"] == "unavailable"
    assert absent["print_files"]["status"] == "unavailable"
    assert absent["guidance"]["status"] == "unavailable"
    original = context()
    before = deepcopy(original)
    view = customer_delivery_view(original)
    view["checks"]["failed"][0]["reason"] = "changed view"
    view["checks"]["coverage"]["external_requirements"].clear()
    view["print_files"]["files"].clear()
    view["print_files"]["package"]["status"] = "changed view"
    view["instruction_binding"]["facts"].clear()
    assert original == before
