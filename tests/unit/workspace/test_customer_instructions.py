"""Customer instructions use saved facts and bytes, without CAD or author execution."""

from __future__ import annotations

import copy
import hashlib
import json
from zipfile import ZipFile

import pytest

from forge_mind.workspace.completion import _hash
from forge_mind.workspace.customer_delivery import customer_delivery_view
from forge_mind.workspace.delivery_guidance import SCOPE as GUIDANCE_SCOPE
from forge_mind.workspace.delivery_guidance import _facts
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.handoff import handoff_context, json_bytes, verify_manufacturing_handoff
from forge_mind.workspace.measure import engine_fingerprint
from forge_mind.workspace.models import Contract
from forge_mind.workspace.production import _package_readme, manufacture_bundle, production_report

LEGACY = {
    "print": "Print the old sleeve.",
    "installation": "Use the earlier 0.2 mm radial clearance.",
    "use_and_removal": "Remove the old plug.",
    "unresolved": "Review the first prototype.",
}
CURRENT = {
    "print": "Print a trial sleeve before use. Orientation remains unverified.",
    "installation": "Fit the plug in the 20.8 mm bore with nominal 0.4 mm radial clearance.",
    "use_and_removal": "Withdraw the plug. Physical access remains unverified.",
    "unresolved": (
        "Physical fit, material strength and print settings remain unverified. "
        "Capture the package and check exact readback before finishing."
    ),
}


def document() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "outputs": ["body"],
            "production": {
                "role": "make",
                "process": "additive",
                "material": {"name": "PLA", "grade": "Unspecified"},
            },
            "delivery_guidance": LEGACY,
        }
    )


def context(status: str = "current") -> dict:
    return {
        "status": status,
        "basis": {"revision_id": "a" * 32},
        "record": {"guidance": CURRENT},
        "legacy_guidance": LEGACY,
        "legacy_guidance_status": "unbound",
        "physical_readiness": "unverified",
        "facts": {
            "measured": {
                "extents_mm": [30, 30, 40],
                "volume_mm3": 8000,
                "solid_count": 2,
                "step_valid": True,
                "bbox_min": [-15, -15, 0],
                "hole_count": 100,
            },
            "declared_variables": {
                "inputs": {"technical_bore": 20.8},
                "evaluated": {"technical_clearance": 0.4},
            },
            "requirement_findings": [
                {
                    "requirement_id": "clearance",
                    "status": "pass",
                    "measured": 0.4,
                    "reason": "Nominal saved geometry check passed.",
                }
            ],
        },
    }


def test_current_prose_stays_in_the_sidecar_without_bound_instruction_authority():
    model = document()
    report = production_report(model)
    guidance = context()
    before = copy.deepcopy((model.model_dump(), report, guidance))
    members = {"make/part.step", "assembly/part.step", "customer-guidance.json"}
    text = _package_readme(
        model,
        report,
        members,
        guidance,
    ).decode()

    for field in ("print", "installation", "use_and_removal", "unresolved"):
        assert guidance["record"]["guidance"][field] == CURRENT[field]
        assert CURRENT[field] not in text
    assert "0.2 mm" not in text
    assert "Instruction reference state: current." in text
    assert "Instruction binding status: legacy_unverified." in text
    assert "No bound customer steps are supplied." in text
    assert "## Unverified author notes and history" in text
    assert "Legacy customer guidance" not in text
    assert "Unresolved choices and checks:" not in text
    assert "technical_bore" not in text
    assert "technical_clearance" not in text
    binding = customer_delivery_view(guidance, members=members)["instruction_binding"]
    assert binding["status"] == "legacy_unverified"
    assert binding["steps"] == []
    for fact in binding["facts"]:
        assert fact["text"] in text
        assert fact["scope"] in text
    assert "make/part.step" in text
    assert "No STL print file is included." in text
    assert "PLA" in text
    assert "[customer-guidance.json](customer-guidance.json)" in text
    assert "declared unresolved work and earlier instructions" in text
    assert "Physical fit, printing and load capacity are not verified." in text
    assert "not part of the bound customer steps" in text
    assert "Named input and derived values" not in text
    assert (
        text.index("make/part.step")
        < text.index("Declared material for part:")
        < text.index("## Unverified author notes and history")
    )
    assert "Bound references do not verify adjacent author notes or physical performance." in text
    assert "Files delivered" not in text
    assert (model.model_dump(), report, guidance) == before


def test_current_review_section_preserves_failed_and_unverified_findings():
    guidance = context()
    guidance["facts"]["requirement_findings"] += [
        {
            "requirement_id": "wall_thickness",
            "status": "fail",
            "measured": 0.7,
            "reason": "Wall is below the required thickness.\n```\nKeep this warning literal.",
        },
        {
            "requirement_id": "physical_fit",
            "status": "unverified",
            "measured": None,
            "reason": "The actual plug has not been tested.",
        },
    ]
    model = document()
    text = _package_readme(model, production_report(model), set(), guidance).decode()
    assert "Failed check wall_thickness:" in text
    assert "Saved measured value: 0.7" in text
    assert "Unverified check physical_fit:" in text
    for finding in guidance["facts"]["requirement_findings"][1:]:
        assert finding["reason"] in text
    assert "````text\nWall is below" in text
    assert "Nominal saved geometry check passed." not in text
    assert "Physical fit, printing and load capacity are not verified." in text
    assert "[customer-guidance.json]" not in text  # Never link a missing file.


def test_missing_requirement_findings_do_not_imply_checks_passed():
    guidance = context()
    guidance["facts"].pop("requirement_findings")
    model = document()
    text = _package_readme(model, production_report(model), set(), guidance).decode()
    assert "No saved requirement results are supplied." in text
    assert "Review the saved evidence before manufacture or use." in text
    assert "Physical fit, printing and load capacity are not verified." in text


def test_current_instructions_keep_production_warnings():
    model = document()
    report = production_report(model)
    report["gaps"].append("Required material source is unverified.")
    text = _package_readme(model, report, set(), context()).decode()
    assert "Required material source is unverified." in text
    assert "Declared gaps:" in text
    assert "No production setup gaps are listed." not in text


def test_repeated_diagnostics_keep_raw_obligations_without_promoting_them_to_current_actions():
    guidance = context()
    reason = "Requirement retained, but no reliable measurement is implemented for this claim."
    action = "Check insertion and removal with the actual printed holder."
    limitation = "Static clearance does not establish easy removal."
    guidance["facts"]["requirement_findings"] = [
        {"requirement_id": f"r{i}", "status": "unverified", "measured": None, "reason": reason}
        for i in range(1, 14)
    ]
    guidance["facts"]["functional_interfaces"] = {
        "rows": [{"id": "remote_cradle", "status": "missing_construction", "duties": []}],
        "external_obligations": [
            {
                "interface_id": "remote_cradle",
                "duty_id": "remove_remote",
                "reason": limitation,
                "physical_validation": [action],
            }
        ],
    }
    guidance["facts"]["functional_coverage"] = {
        "external_requirements": [
            {
                "requirement_id": f"r{i}",
                "unsupported_checks": [limitation],
                "physical_validation": [action],
            }
            for i in range(1, 14)
        ]
    }
    before = copy.deepcopy(guidance)
    model = document()
    text = _package_readme(
        model,
        production_report(model),
        {"customer-guidance.json", "engineering-evidence.json", "make/part.step"},
        guidance,
    ).decode()
    assert "13 unverified requirement checks" in text
    assert {
        row["requirement_id"] for row in customer_delivery_view(guidance)["checks"]["unverified"]
    } == {f"r{i}" for i in range(1, 14)}
    assert "Unverified checks (13):" in text
    assert text.count(reason) == 1
    assert action not in text
    assert limitation not in text
    assert "Literal guidance, declared obligations and instruction history:" in text
    assert "not part of the bound customer steps" in text
    assert "Required physical tests:" not in text
    assert "Instruction binding status: legacy_unverified." in text
    assert all(
        row["unsupported_checks"] == [limitation]
        for row in customer_delivery_view(guidance)["checks"]["coverage"]["external_requirements"]
    )
    assert guidance["facts"]["functional_interfaces"]["external_obligations"][0][
        "physical_validation"
    ] == [action]
    assert "remote_cradle" in text and "missing_construction" in text
    assert "[customer-guidance.json](customer-guidance.json)" in text
    assert "[engineering-evidence.json](engineering-evidence.json)" in text
    for value in CURRENT.values():
        assert value not in text
    assert guidance["record"]["guidance"] == CURRENT
    assert guidance == before


def test_more_detail_does_not_link_optional_files_missing_from_this_package():
    model = document()
    text = _package_readme(model, production_report(model), {"make/part.step"}, context()).decode()
    for name in ("assembly/part.step", "customer-guidance.json", "engineering/report.json"):
        assert f"]({name})" not in text


@pytest.mark.parametrize("include_stl", [False, True])
def test_current_package_keeps_authored_unresolved_and_legacy_in_exact_sidecar(
    tmp_path, include_stl
):
    model = document()
    source = tmp_path / "saved"
    source.mkdir()
    saved_step = b"Immutable saved geometry; no CAD execution"
    (source / "part.step").write_bytes(saved_step)
    (source / "part.stl").write_bytes(b"Immutable saved mesh; no CAD execution")
    (source / "document.json").write_bytes(json_bytes(model.model_dump()))
    original = {path.name: path.read_bytes() for path in source.iterdir()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in original.items()}
    rid, sid = "a" * 32, "b" * 32
    engine = engine_fingerprint()
    contract = Contract().model_dump()
    engineering = {"version": 0, "inputs": EngineeringInputs().model_dump()}
    sources = {"fingerprint": "fixture-source", "claims": [], "artifacts": []}
    revision = {
        "id": rid,
        "document": model.model_dump(),
        "contract": contract,
        "measurement": {"engine_fingerprint": engine, "artifact_hashes": hashes, "facts": {}},
    }
    session = {
        "id": sid,
        "selected_revision": rid,
        "accepted_revision": None,
        "engineering": engineering,
        "source_fingerprint": sources["fingerprint"],
    }
    facts = _facts(revision)
    basis = {
        "session_id": sid,
        "revision_id": rid,
        "contract_hash": _hash(contract),
        "session_contract_hash": _hash(contract),
        "engine_fingerprint": engine,
        "source_fingerprint": sources["fingerprint"],
        "input_version": engineering["version"],
        "plan_hash": _hash(None),
        "artifact_hashes": hashes,
        "document_hash": digest(revision["document"]),
        "engineering_inputs_hash": digest(engineering),
        "measured_engine_fingerprint": engine,
        "measured_facts_hash": digest(revision["measurement"]["facts"]),
        "facts_hash": digest(facts),
    }
    current = {**CURRENT, "print": CURRENT["print"] + " Use make/part.stl from the full project."}
    guidance = {
        "status": "current",
        "basis": basis,
        "basis_hash": digest(basis),
        "facts": facts,
        "record": {
            "basis": basis,
            "basis_hash": digest(basis),
            "guidance": current,
            "guidance_hash": digest(current),
            "reviewed_current_facts": True,
            "scope": GUIDANCE_SCOPE,
        },
        "legacy_guidance": LEGACY,
        "legacy_guidance_status": "unbound",
        "physical_readiness": "unverified",
        "scope": GUIDANCE_SCOPE,
    }
    captured = handoff_context(session, revision, engine)
    captured.update(
        guidance_basis=basis,
        guidance_basis_hash=digest(basis),
        guidance_context_hash=digest(guidance),
    )
    package = manufacture_bundle(
        model,
        source,
        tmp_path / "project.zip",
        hashes,
        sources=sources,
        engineering_evidence={"revision_id": rid, "physical_readiness": "unverified"},
        revision_context=captured,
        engineering_inputs=engineering,
        include_assembly_step=True,
        include_stl=include_stl,
        guidance_context=guidance,
    )
    assert verify_manufacturing_handoff(package)["guidance_status"] == "current"
    with ZipFile(package) as archive:
        saved_guidance = json.loads(archive.read("customer-guidance.json"))
        assert saved_guidance == guidance
        assert saved_guidance["record"]["guidance"] == current
        assert saved_guidance["basis"] == basis
        assert saved_guidance["basis_hash"] == digest(basis)
        assert saved_guidance["legacy_guidance"] == LEGACY
        assert archive.read("document.json") == original["document.json"]
        assert json.loads(archive.read("document.json"))["delivery_guidance"] == LEGACY
        text = archive.read("README.md").decode()
        for value in current.values():
            assert value not in text
        assert "## Unverified author notes and history" in text
        assert "Instruction reference state: current." in text
        assert "Instruction binding status: legacy_unverified." in text
        assert "No bound customer steps are supplied." in text
        assert LEGACY["installation"] not in text
        assert ("make/part.stl" in archive.namelist()) is include_stl
        if include_stl:
            assert "Project geometry omitted from this download:" not in text
        else:
            assert "Project geometry omitted from this download:" in text
            assert text.index("Project geometry omitted from this download:") < text.index(
                "## Unverified author notes and history"
            )
            assert "The saved instructions below can require these omitted files." in text
        assert "[customer-guidance.json](customer-guidance.json)" in text
        assert archive.read("assembly/part.step") == saved_step
        assert json.loads(archive.read("manifest.json"))["manufacturing_ready"] is False
    assert guidance["physical_readiness"] == "unverified"
    assert {path.name: path.read_bytes() for path in source.iterdir()} == original


@pytest.mark.parametrize("status", [None, "unbound", "stale"])
def test_omitted_sidecar_keeps_literal_current_and_legacy_notes_in_unverified_history(status):
    model = document()
    guidance = context(status) if status else None
    text = _package_readme(model, production_report(model), set(), guidance).decode()
    assert "## Legacy customer guidance: unbound reference state" in text
    for value in LEGACY.values():
        assert value in text
    assert "Unresolved choices and checks:" in text
    assert "unverified author notes" in text
    if guidance:
        assert "## Unverified author notes and history" in text
        for value in CURRENT.values():
            assert value in text
            assert text.index(value) > text.index("## Unverified author notes and history")
        assert "No bound customer steps are supplied." in text
        assert "technical_bore" in text


@pytest.mark.parametrize("include_import_metadata", [False, True])
def test_readme_distinguishes_prospective_metadata_from_selected_members(include_import_metadata):
    model, guidance = document(), context()
    optional = {"inputs/assets.json", "import-assets.json"}
    guidance["facts"]["package_files"] = {
        "metadata_files": ["README.md", "manifest.json", "document.json", *sorted(optional)]
    }
    members = {"document.json"} | (optional if include_import_metadata else set())
    before = copy.deepcopy(guidance)
    text = _package_readme(model, production_report(model), members, guidance).decode()
    label = "Project metadata omitted from this download:"
    if include_import_metadata:
        assert label not in text
    else:
        omitted = text.split(label, 1)[1].split("```text\n", 1)[1].split("```", 1)[0]
        assert set(omitted.splitlines()) == optional
        assert text.index(label) < text.index(CURRENT["print"])
        assert "Use manifest.json for the exact included files." in text
    for value in CURRENT.values():
        assert value in text
    assert guidance == before


def test_current_preparation_is_separate_from_original_stl_and_old_workflow_declarations():
    model, guidance = document(), context()
    command = "Complete print_setup preparation and verify captured package receipts."
    missing = "Actual wood thickness and purchased screw dimensions."
    physical = "Check retention with the actual remote before routine use."
    guidance["facts"]["print_setup"] = {
        "status": "current",
        "files": {
            "print/definition_part.stl": {
                "quantity": 1,
                "sha256": "c" * 64,
                "frame": "plate_z_up",
            }
        },
        "package": {"status": "prepared", "sha256": "d" * 64},
    }
    guidance["facts"]["functional_coverage"] = {
        "external_requirements": [
            {
                "requirement_id": "print",
                "unsupported_checks": [command],
                "missing_inputs": [missing],
                "physical_validation": [physical],
            }
        ]
    }
    before = copy.deepcopy(guidance)
    text = _package_readme(
        model,
        production_report(model),
        {"make/part.stl", "make/part.step", "customer-guidance.json", "document.json"},
        guidance,
    ).decode()
    assert "Print setup status: current" in text
    assert "print/definition_part.stl: 1 copy" in text
    assert "make/part.stl" in text
    assert "make 1 copy" in text
    assert "1 copies" not in text
    assert "separate print package" in text
    assert "original frames" in text
    assert command not in text
    assert missing in text and physical not in text
    assert "Required physical tests:" not in text
    assert guidance["facts"]["functional_coverage"]["external_requirements"][0] == {
        "requirement_id": "print",
        "unsupported_checks": [command],
        "missing_inputs": [missing],
        "physical_validation": [physical],
    }
    assert "Instruction binding status: legacy_unverified." in text
    assert "[customer-guidance.json](customer-guidance.json)" in text
    assert "delivered" not in text.lower()
    assert "readback verified" not in text.lower()
    assert "Physical fit, printing and load capacity are not verified." in text
    assert guidance == before


@pytest.mark.parametrize("status", ["fail", "failed", "check_failed"])
def test_actual_functional_failure_reason_remains_visible_beside_current_print_files(status):
    model, guidance = document(), context()
    reason = "The measured screw envelope collides with the holder."
    guidance["facts"]["functional_interfaces"] = {
        "rows": [
            {
                "id": "mount",
                "status": status,
                "failed_interface_ids": ["screw_bore"],
                "duties": [{"id": "clearance", "status": status, "reason": reason}],
            }
        ],
    }
    text = _package_readme(
        model, production_report(model), {"customer-guidance.json"}, guidance
    ).decode()
    assert "Failed Interface mount, duty clearance:" in text
    assert "**Saved checks failed." in text
    assert text.index("**Saved checks failed.") < text.index(
        "## Unverified author notes and history"
    )
    assert CURRENT["print"] not in text
    assert reason in text
    assert "screw_bore" in text
    assert "physical" in text.lower()


def test_unsupported_declarations_are_not_lost_when_no_sidecar_is_included():
    model, guidance = document(), context()
    declaration = "No saved method establishes the proposed holding capacity."
    guidance["facts"]["functional_coverage"] = {
        "external_requirements": [{"unsupported_checks": [declaration]}]
    }
    text = _package_readme(model, production_report(model), set(), guidance).decode()
    assert declaration in text
    assert "Authored unsupported_checks (unverified history, not a current task):" in text
    assert text.index(declaration) > text.index("## Unverified author notes and history")
    assert "[customer-guidance.json]" not in text


@pytest.mark.parametrize("status", ["stale", "missing", "tampered"])
def test_unusable_print_receipt_remains_explicit_without_preparation_or_delivery_claim(status):
    model, guidance = document(), context("stale")
    guidance["facts"]["print_setup"] = {"status": status, "files": {}}
    text = _package_readme(model, production_report(model), set(), guidance).decode()
    expected = "unverified" if status == "tampered" else status
    assert f"Print setup status: {expected}" in text
    assert "Instruction reference state: stale" in text
    assert "These instructions have no current saved-facts review" in text
    assert "Print setup status: current" not in text
    assert "delivered" not in text.lower()


@pytest.mark.parametrize("status", ["stale", "unbound"])
def test_current_failures_limits_and_omissions_survive_noncurrent_authored_instructions(status):
    model, guidance = document(), context(status)
    failure = "Current measured wall is below the frozen minimum."
    missing = "Actual wood thickness."
    physical = "Test mounted retention with the actual remote."
    guidance["facts"]["package_files"] = {
        "metadata_files": ["document.json", "production.json", "README.md", "manifest.json"]
    }
    guidance["facts"]["requirement_findings"] = [
        {"requirement_id": "wall", "status": "fail", "reason": failure, "measured": 0.7}
    ]
    guidance["facts"]["functional_interfaces"] = {
        "rows": [
            {
                "id": "support",
                "status": "missing_construction",
                "missing_features": [{"occurrence_path": "holder", "feature_id": "brace"}],
                "duties": [],
            }
        ],
        "external_obligations": [{"missing_inputs": [missing], "physical_validation": [physical]}],
    }
    guidance["facts"]["print_setup"] = {
        "status": "current",
        "files": {"print/definition_part.stl": {"quantity": 1, "sha256": "c" * 64}},
        "package": {"status": "prepared", "sha256": "d" * 64},
    }
    before = copy.deepcopy(guidance)
    text = _package_readme(
        model,
        production_report(model),
        {"customer-guidance.json", "make/part.step", "assembly/part.step"},
        guidance,
    ).decode()
    assert f"Instruction reference state: {status}" in text
    assert "No bound customer steps are supplied." in text
    assert "## Unverified author notes and history" in text
    assert "**Saved checks failed." in text
    assert failure in text and "Saved measured value: 0.7" in text
    assert "missing_construction" in text and "brace" in text
    assert missing in text and physical not in text
    assert "Required physical tests:" not in text
    assert guidance["facts"]["functional_interfaces"]["external_obligations"][0] == {
        "missing_inputs": [missing],
        "physical_validation": [physical],
    }
    assert "[customer-guidance.json](customer-guidance.json)" in text
    assert "Print setup status: current" in text and "print/definition_part.stl: 1 copy" in text
    omitted = text.split("Project geometry omitted from this download:", 1)[1]
    assert "make/part.stl" in omitted
    metadata = text.split("Project metadata omitted from this download:", 1)[1]
    assert "document.json" in metadata and "production.json" in metadata
    assert "Physical fit, printing and load capacity are not verified." in text
    assert guidance == before


def test_current_customer_layout_keeps_hardware_names_but_moves_internal_catalog_detail():
    model, guidance = document(), context()
    report = production_report(model)
    report["buy"] = [
        {
            "part_id": "screws",
            "name": "M4 x 20 pan-head screw",
            "quantity": 2,
            "catalog_id": "internal_catalog_identity",
            "production": {"procurement": {"part_number": "SCREW-4-20"}},
        }
    ]
    before = copy.deepcopy((report, guidance))
    text = _package_readme(
        model,
        report,
        {"customer-guidance.json", "make/part.stl", "make/part.step", "assembly/part.step"},
        guidance,
    ).decode()
    assert "M4 x 20 pan-head screw" in text and "Quantity: 2" in text
    assert "Declared name: M4 x 20 pan-head screw" in text
    assert "SCREW-4-20" in text
    assert "internal_catalog_identity" not in text
    assert "Current saved CAD facts" not in text
    assert (
        text.index("make/part.stl")
        < text.index("M4 x 20 pan-head screw")
        < text.index("## Unverified author notes and history")
    )
    assert CURRENT["print"] not in text
    assert "Instruction binding status: legacy_unverified." in text
    assert "Do not print these parts or the overall assembly." in text
    assert (report, guidance) == before
