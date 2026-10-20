"""Bound customer presentation keeps saved evidence separate from authored prose."""

from __future__ import annotations

from copy import deepcopy

import pytest

from forge_mind.workspace import customer_delivery
from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.instruction_plan import instruction_catalog
from forge_mind.workspace.production import _package_readme, production_report

RID = "a" * 32
AUTHORED = {
    "print": "The future project package is still unconfirmed. ``` Keep this exact text.",
    "installation": "An author suggested unsupported installation work.",
    "use_and_removal": "An author asserted easy removal without a physical trial.",
    "unresolved": "Refresh completion evidence for an old revision, not this saved check.",
}
LEGACY = {
    "print": "Legacy print declaration.",
    "installation": "Legacy installation declaration.",
    "use_and_removal": "Legacy removal declaration.",
    "unresolved": "Legacy unverified limit.",
}
PHYSICAL_NOTE = "Run the old motion check again before declaring the package captured."
UNSUPPORTED_NOTE = "The author proposed waiving a failed collision check."
MISSING_INPUT = "Actual mounting substrate thickness is unknown."


def model() -> DesignDocument:
    return DesignDocument.model_validate(
        {
            "features": [{"id": "body", "name": "Body", "kind": "box"}],
            "outputs": ["body"],
            "production": {"role": "make", "material": {"name": "PLA"}},
            "delivery_guidance": LEGACY,
        }
    )


def context() -> dict:
    facts = {
        "package_files": {
            "geometry": {
                "make/part.stl": {
                    "part_id": "part",
                    "role": "make",
                    "quantity": 1,
                    "source_present": True,
                },
                "make/part.step": {
                    "part_id": "part",
                    "role": "make",
                    "quantity": 1,
                    "source_present": True,
                },
            },
            "metadata_files": ["document.json", "production.json"],
        },
        "instruction_parts": [
            {
                "part_id": "part",
                "declared_name": "Declared body label",
                "role": "make",
                "material": {"name": "PLA"},
                "exported_occurrence_quantity": 1,
            }
        ],
        "requirement_findings": [
            {"requirement_id": "retention", "status": "unverified", "measured": None}
        ],
        "functional_interfaces": {
            "rows": [
                {
                    "id": "removal",
                    "status": "check_failed",
                    "duties": [
                        {
                            "id": "travel",
                            "status": "check_failed",
                            "reason": "Saved collision witness remains failed.",
                        }
                    ],
                }
            ],
            "external_obligations": [
                {
                    "missing_inputs": [MISSING_INPUT],
                    "physical_validation": [PHYSICAL_NOTE],
                    "unsupported_checks": [UNSUPPORTED_NOTE],
                }
            ],
        },
    }
    basis = {"revision_id": RID, "facts_hash": digest(facts)}
    result = {
        "status": "current",
        "basis": basis,
        "basis_hash": digest(basis),
        "facts": facts,
        "physical_readiness": "unverified",
    }
    refs = instruction_catalog(result)["required_refs"]
    plan = {
        "version": 1,
        "fact_refs": list(dict.fromkeys([*refs, "project:make/part.stl"])),
        "steps": [
            {"action": "open_project_file", "target": "project:make/part.stl"},
            {"action": "review_part", "target": "part:part"},
            {"action": "review_check", "target": "checks:summary"},
            {"action": "review_physical_limits", "target": "physical:readiness"},
        ],
    }
    result["record"] = {
        "basis": deepcopy(basis),
        "basis_hash": digest(basis),
        "guidance": deepcopy(AUTHORED),
        "guidance_hash": digest(AUTHORED),
        "reviewed_current_facts": True,
        "instruction_plan": plan,
        "instruction_plan_hash": digest(plan),
    }
    return result


def readme(guidance: dict | None, members: set[str]) -> str:
    document = model()
    return _package_readme(document, production_report(document), members, guidance).decode()


def test_inventory_overlay_does_not_change_the_context_given_to_the_shared_resolver(monkeypatch):
    guidance = context()
    before = deepcopy(guidance)
    members = {"make/other.step"}
    expected = {"status": "test server result"}
    calls = []

    def resolve(supplied, *, members=None):
        calls.append((deepcopy(supplied), members))
        return expected

    monkeypatch.setattr(customer_delivery, "resolve_instruction_plan", resolve)
    view = customer_delivery.customer_delivery_view(
        guidance,
        members=members,
        package_inventory={
            "geometry": {"make/other.step": {"source_present": True, "quantity": 2}}
        },
    )
    assert calls == [(before, members)]
    assert view["instruction_binding"] is expected
    assert view["project_files"]["files"][0]["path"] == "make/other.step"
    assert view["project_files"]["files"][0]["included"] is True
    assert guidance == before


def test_server_facts_and_steps_precede_declared_metadata_and_linked_author_history():
    guidance = context()
    before = deepcopy(guidance)
    members = {"make/part.stl", "make/part.step", "customer-guidance.json", "document.json"}
    binding = customer_delivery.customer_delivery_view(guidance, members=members)[
        "instruction_binding"
    ]
    assert binding["status"] == "bound"
    text = readme(guidance, members)
    for fact in binding["facts"]:
        assert fact["text"] in text
        assert fact["scope"] in text
    for step in binding["steps"]:
        assert step["text"] in text
        assert f"({step['actor']}; {step['state']})" in text
    assert text.index("## Current file and check facts") < text.index("Declared material for part:")
    assert text.index("## Print, install and use") < text.index(
        "## Unverified author notes and history"
    )
    assert "[customer-guidance.json](customer-guidance.json)" in text
    for value in (*AUTHORED.values(), *LEGACY.values(), PHYSICAL_NOTE, UNSUPPORTED_NOTE):
        assert value not in text
    assert MISSING_INPUT in text
    assert "Required physical tests:" not in text
    assert "Saved collision witness remains failed." in text
    assert "**Saved checks failed." in text
    assert "Physical fit, printing and load capacity are not verified." in text
    assert guidance == before


def test_omitted_file_blocks_only_its_step_and_keeps_the_bound_draft_download():
    guidance = context()
    before = deepcopy(guidance)
    members = {"make/part.step", "customer-guidance.json"}
    binding = customer_delivery.customer_delivery_view(guidance, members=members)[
        "instruction_binding"
    ]
    assert binding["status"] == "bound"
    step = next(row for row in binding["steps"] if row["target"] == "project:make/part.stl")
    assert step["state"] == "blocked"
    assert any(row["state"] != "blocked" for row in binding["steps"])
    text = readme(guidance, members)
    assert "Project geometry omitted from this download:" in text
    assert "make/part.stl" in text
    assert "make/part.step" in text
    assert "No STL print file is included." in text
    assert "Instruction binding status: bound." in text
    assert step["text"] in text
    assert guidance == before


@pytest.mark.parametrize("state", ["missing", "stale", "invalid", "blocked"])
def test_unbound_or_blocked_instructions_keep_current_machine_failures_and_missing_inputs(state):
    guidance = context()
    if state == "missing":
        guidance.pop("record")
    elif state == "stale":
        guidance["status"] = "stale"
    elif state == "invalid":
        guidance["record"]["instruction_plan_hash"] = "0" * 64
    else:
        guidance["input_authority"] = {"status": "blocked", "identity": {"pending": "answer"}}
    members = {"customer-guidance.json"}
    binding = customer_delivery.customer_delivery_view(guidance, members=members)[
        "instruction_binding"
    ]
    assert binding["status"] != "bound"
    assert binding["steps"] == []
    text = readme(guidance, members)
    assert "No bound customer steps are supplied." in text
    assert "**Saved checks failed." in text
    assert "check_failed" in text
    assert "Saved collision witness remains failed." in text
    assert MISSING_INPUT in text
    assert "Project geometry omitted from this download:" in text
    assert "unverified requirement check" in text
    assert "Physical fit, printing and load capacity are not verified." in text
    assert PHYSICAL_NOTE not in text and UNSUPPORTED_NOTE not in text


def test_missing_sidecar_preserves_literal_author_notes_history_and_legacy_without_current_tasks():
    guidance = context()
    guidance["record"].pop("instruction_plan")
    historical = "Historical Unicode \u03bc note.\n```\nDo not rewrite <this>."
    guidance["history"] = [{"version": 1, "record": {"guidance": {"print": historical}}}]
    before = deepcopy(guidance)
    text = readme(guidance, set())
    history_start = text.index("## Unverified author notes and history")
    for value in (
        *AUTHORED.values(),
        *LEGACY.values(),
        PHYSICAL_NOTE,
        UNSUPPORTED_NOTE,
        historical,
    ):
        assert value in text
        assert text.index(value) > history_start
    assert "Authored physical_validation (unverified history, not a current task):" in text
    assert "Authored unsupported_checks (unverified history, not a current task):" in text
    assert "Required physical tests:" not in text
    assert "[customer-guidance.json]" not in text
    assert "No bound customer steps are supplied." in text
    assert guidance == before


def test_legacy_document_without_a_guidance_context_retains_partial_draft_notes():
    text = readme(None, set())
    for value in LEGACY.values():
        assert value in text
    assert "Legacy customer guidance: unbound reference state" in text
    assert "unverified author notes" in text
    assert "No STL print file is included." in text
    assert "Physical fit, printing and load capacity are not verified." in text
