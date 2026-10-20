"""Reference binding never certifies authored prose or turns failed checks into approval."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from forge_mind.workspace.instruction_plan import (
    InstructionPlan,
    InstructionStep,
    instruction_catalog,
    resolve_instruction_plan,
    validate_instruction_plan,
)

RID, HASH = "a" * 32, "b" * 64
OLD_NOTE = "Refresh evidence for an old revision; final package inclusion is unconfirmed."


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def context() -> dict:
    facts = {
        "instruction_parts": [
            {
                "part_id": "plate",
                "declared_name": "A claimed qualified plate",
                "role": "make",
                "material": {"name": "PLA"},
                "exported_occurrence_quantity": 2,
            },
            {
                "part_id": "screw",
                "role": "buy",
                "material": None,
                "exported_occurrence_quantity": 4,
            },
            {"part_id": "remote", "role": "reference", "exported_occurrence_quantity": 1},
        ],
        "parts": [{"part_id": "plate"}],  # Bounded legacy summary is not the catalog authority.
        "package_files": {
            "geometry": {
                "make/definition_plate.step": {
                    "part_id": "plate",
                    "role": "make",
                    "quantity": 2,
                    "source_present": True,
                },
                "make/definition_plate.stl": {
                    "part_id": "plate",
                    "role": "make",
                    "quantity": 2,
                    "source_present": True,
                },
                "assembly/part.step": {
                    "part_id": None,
                    "role": "overall_review",
                    "quantity": 1,
                    "source_present": True,
                },
            },
            "metadata_files": ["document.json", "manifest.json"],
            "excluded_saved_artifacts": ["definition_remote.step"],
        },
        "measured": {
            "extents_mm": [45, 20.0, 150],
            "volume_mm3": 1000.0,
            "solid_count": 1,
            "hole_count": 2,
            "step_valid": True,
        },
        "print_setup": {
            "status": "current",
            "files": {
                "print/definition_plate.stl": {
                    "part_id": "plate",
                    "quantity": 2,
                    "frame": "plate_z_up",
                    "units": "mm",
                    "sha256": HASH,
                    "candidate_hash": HASH,
                }
            },
        },
        "requirement_findings": [
            {
                "requirement_id": "fit",
                "status": "unverified",
                "reason": "A declared reason, not proof.",
            }
        ],
        "functional_interfaces": {
            "scope": "Seven sampled configurations only.",
            "rows": [
                {
                    "id": "removal",
                    "status": "check_failed",
                    "scope": "Overall saved motion fails.",
                    "duties": [
                        {
                            "id": "travel",
                            "status": "check_failed",
                            "proof_scope": "sampled_clearance",
                            "reason": "Fixed screw/wood overlap remains a failure.",
                            "physical_validation": [OLD_NOTE],
                        }
                    ],
                }
            ],
        },
    }
    basis = {"revision_id": RID, "facts_hash": digest(facts)}
    guidance = {
        "print": OLD_NOTE,
        "installation": "Author installation proposal.",
        "use_and_removal": "Author use proposal.",
        "unresolved": OLD_NOTE,
    }
    return {
        "status": "current",
        "basis": basis,
        "basis_hash": digest(basis),
        "facts": facts,
        "physical_readiness": "approved",  # Ignored: nominal facts cannot approve physical use.
        "record": {
            "basis": deepcopy(basis),
            "basis_hash": digest(basis),
            "guidance": guidance,
            "guidance_hash": digest(guidance),
            "reviewed_current_facts": True,
        },
    }


def plan_for(value: dict) -> dict:
    refs = instruction_catalog(value)["required_refs"]
    return {
        "version": 1,
        "fact_refs": list(
            dict.fromkeys(
                [
                    *refs,
                    "project:make/definition_plate.stl",
                    "check:duty:removal/travel",
                    "workflow:print_preparation",
                ]
            )
        ),
        "steps": [
            {"action": "open_project_file", "target": "project:make/definition_plate.stl"},
            {"action": "load_print_file", "target": "prepared:print/definition_plate.stl"},
            {"action": "review_part", "target": "part:screw"},
            {"action": "review_check", "target": "check:duty:removal/travel"},
            {"action": "review_physical_limits", "target": "physical:readiness"},
            {"action": "prepare_print_files", "target": "workflow:print_preparation"},
        ],
    }


def bind(value: dict, plan: dict | None = None) -> dict:
    value["basis"]["facts_hash"] = digest(value["facts"])
    if "input_authority" in value:
        authority_hash = digest(value["input_authority"]["identity"])
        value["basis"]["input_authority_hash"] = authority_hash
        value["record"]["input_authority_hash"] = authority_hash
    value["basis_hash"] = digest(value["basis"])
    value["record"].update(basis=deepcopy(value["basis"]), basis_hash=value["basis_hash"])
    normalized = validate_instruction_plan(plan or plan_for(value), value).model_dump()
    value["record"].update(instruction_plan=normalized, instruction_plan_hash=digest(normalized))
    return value


def rows(result: dict) -> dict:
    return {row["id"]: row for row in result["facts"]}


@pytest.mark.parametrize("version", [True, False, 1.0, "1", 2])
def test_plan_version_is_strict(version):
    with pytest.raises(ValidationError):
        InstructionPlan.model_validate(
            {
                "version": version,
                "fact_refs": ["physical:readiness"],
                "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
            }
        )


@pytest.mark.parametrize("extra", ["value", "verdict", "scope", "text", "file_bytes"])
def test_authors_cannot_supply_fact_values_or_step_claims(extra):
    with pytest.raises(ValidationError):
        InstructionStep.model_validate(
            {"action": "review_check", "target": "checks:summary", extra: "pass"}
        )


def test_empty_or_coerced_refs_and_custom_actions_are_rejected():
    for raw in [
        {"fact_refs": [], "steps": [{"action": "review_check", "target": "checks:summary"}]},
        {"fact_refs": [True], "steps": []},
        {
            "fact_refs": ["checks:summary"],
            "steps": [{"action": "custom", "target": "checks:summary"}],
        },
        {"fact_refs": ["checks:summary"], "steps": [{"action": "review_check", "target": 1}]},
    ]:
        with pytest.raises(ValidationError):
            InstructionPlan.model_validate(raw)


def test_catalog_uses_complete_parts_and_exact_namespaces_without_inventing_quantities():
    value = context()
    value["facts"]["instruction_parts"].extend(
        [
            {"part_id": "unknown", "role": "make", "exported_occurrence_quantity": None},
            {"part_id": "absent", "role": "buy", "exported_occurrence_quantity": 0},
        ]
    )
    catalog = instruction_catalog(value)
    assert {
        "checks:summary",
        "physical:readiness",
        "part:plate",
        "part:screw",
        "part:absent",
        "prepared:print/definition_plate.stl",
    } == set(catalog["required_refs"])
    assert catalog["facts"]["part:unknown"]["state"] == "unverified"
    assert catalog["facts"]["part:absent"]["value"]["exported_occurrence_quantity"] == 0
    assert catalog["facts"]["part:absent"]["state"] == "missing"
    assert "part:remote" in catalog["facts"] and "part:remote" not in catalog["required_refs"]
    assert "project:definition_remote.step" not in catalog["facts"]
    assert "measurement:extents_mm" in catalog["facts"]
    assert "unverified literal" in catalog["facts"]["part:plate"]["text"]


@pytest.mark.parametrize(
    "case, expected",
    [
        ("unknown", "project:missing.stl"),
        ("wrong_kind", "part:screw"),
        ("missing", "physical:readiness"),
        ("duplicate", "checks:summary"),
        ("undisclosed", "project:assembly/part.step"),
        ("duplicate_step", "review_check"),
    ],
)
def test_reference_failures_name_exact_targets(case, expected):
    value = context()
    plan = plan_for(value)
    if case == "unknown":
        plan["fact_refs"].append(expected)
    elif case == "wrong_kind":
        plan["steps"][0]["target"] = expected
    elif case == "missing":
        plan["fact_refs"].remove(expected)
    elif case == "duplicate":
        plan["fact_refs"].append(expected)
    elif case == "undisclosed":
        plan["steps"][0]["target"] = expected
    else:
        plan["steps"].append(deepcopy(plan["steps"][3]))
    with pytest.raises(ValueError, match=expected):
        validate_instruction_plan(plan, value)


def test_bound_steps_use_six_server_actions_and_never_promote_notes_or_physical_approval():
    value = bind(context())
    before = deepcopy(value)
    result = resolve_instruction_plan(value)
    assert result["status"] == "bound" and result["physical_readiness"] == "unverified"
    assert len(result["steps"]) == 6
    assert {step["actor"] for step in result["steps"]} == {
        "customer",
        "reviewer",
        "workspace_owner",
    }
    assert result["steps"][-1]["state"] == "complete"
    assert "already prepared" in result["steps"][-1]["text"]
    assert rows(result)["check:duty:removal/travel"]["state"] == "check_failed"
    assert result["steps"][3]["state"] == "check_failed"
    assert rows(result)["checks:summary"]["state"] == "fail"
    rendered = "\n".join(row["text"] for row in result["facts"] + result["steps"])
    assert OLD_NOTE not in rendered and "qualified plate" not in rendered
    assert "only if it is included" in result["steps"][0]["text"]
    assert "plate_z_up" in result["steps"][1]["text"]
    assert rows(result)["check:duty:removal/travel"]["value"]["physical_validation"] == [OLD_NOTE]
    assert value == before  # Every raw failure and authored declaration remains intact.


def test_actual_membership_overlay_blocks_only_omitted_steps_without_rebinding():
    value = bind(context())
    before = deepcopy(value)
    omitted = resolve_instruction_plan(value, members={"document.json", "assembly/part.step"})
    included = resolve_instruction_plan(value, members={"make/definition_plate.stl"})
    assert omitted["status"] == included["status"] == "bound"
    assert omitted["steps"][0]["state"] == "blocked"
    assert included["steps"][0]["state"] == "ready"
    assert (
        "included" in included["steps"][0]["text"]
        or "from this project download" in included["steps"][0]["text"]
    )
    assert omitted["steps"][1]["state"] == "ready"  # Separate print ZIP is not a project member.
    assert value == before


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ("facts", "basis.facts_hash"),
        ("basis", "Stored instruction basis"),
        ("basis_hash", "basis_hash"),
        ("plan", "instruction_plan_hash"),
        ("guidance", "guidance_hash"),
        ("review", "reviewed_current_facts"),
        ("malformed_guidance", "guidance_hash"),
    ],
)
def test_status_alone_cannot_hide_changed_or_invalid_stored_bindings(mutation, expected):
    value = bind(context())
    if mutation == "facts":
        value["facts"]["measured"]["volume_mm3"] = 2000.0
    elif mutation == "basis":
        value["basis"]["revision_id"] = "c" * 32
    elif mutation == "basis_hash":
        value["basis_hash"] = value["record"]["basis_hash"] = "c" * 64
    elif mutation == "plan":
        value["record"]["instruction_plan"]["steps"][0]["target"] = "project:document.json"
    elif mutation == "guidance":
        value["record"]["guidance"]["print"] += " Changed."
    elif mutation == "malformed_guidance":
        value["record"]["guidance"] = {"print": None}
        value["record"]["guidance_hash"] = digest(value["record"]["guidance"])
    else:
        value["record"]["reviewed_current_facts"] = False
    result = resolve_instruction_plan(value)
    assert result["status"] in {"invalid", "stale"}
    assert result["steps"] == []
    assert expected in "; ".join(result["errors"])


def test_missing_plan_is_readable_legacy_and_changed_current_state_is_stale():
    value = context()
    assert resolve_instruction_plan(value)["status"] == "legacy_unverified"
    bind(value)
    value["status"] = "stale"
    result = resolve_instruction_plan(value)
    assert result["status"] == "stale" and result["steps"] == []


@pytest.mark.parametrize(
    "authority",
    [
        None,
        {},
        {"status": "current", "identity": {}},
        {"status": "blocked", "identity": {}},
        {"status": "clear", "identity": None},
    ],
)
def test_explicit_malformed_or_blocked_authority_fails_closed(authority):
    value = bind(context())
    value["input_authority"] = authority
    assert resolve_instruction_plan(value)["status"] == "blocked_input"


def test_same_geometry_input_epoch_change_and_invalid_history_cannot_retain_bound_state():
    value = context()
    value["input_authority"] = {
        "status": "clear",
        "identity": {"revision_id": RID, "application_hash": HASH},
        "reasons": [],
    }
    value["history_status"] = {"status": "verified"}
    bind(value)
    assert resolve_instruction_plan(value)["status"] == "bound"
    value["input_authority"]["identity"]["application_hash"] = "c" * 64
    assert resolve_instruction_plan(value)["status"] == "stale"
    bind(value)
    value["history_status"] = {"status": "unverified", "errors": ["history entry 3 changed"]}
    result = resolve_instruction_plan(value)
    assert result["status"] == "invalid" and "history entry 3 changed" in result["errors"]


@pytest.mark.parametrize(
    ("present", "reasons"),
    [
        (False, None),
        (True, None),
        (True, "pending"),
        (True, {}),
        (True, ["pending"]),
        (True, [""]),
        (True, False),
    ],
)
def test_clear_label_cannot_hide_present_refusal_or_malformed_reasons(present, reasons):
    value = context()
    value["input_authority"] = {
        "status": "clear",
        "identity": {"revision_id": RID, "application_hash": HASH},
        "reasons": [],
    }
    bind(value)
    assert resolve_instruction_plan(value)["status"] == "bound"
    if present:
        value["input_authority"]["reasons"] = reasons
    else:
        del value["input_authority"]["reasons"]
    result = resolve_instruction_plan(value)
    assert result["status"] == "blocked_input"
    assert result["steps"] == []


def test_print_preparation_is_pending_only_when_not_current_and_bad_values_stay_unverified():
    value = context()
    value["facts"].pop("print_setup")
    catalog = instruction_catalog(value)
    assert catalog["facts"]["workflow:print_preparation"]["state"] == "pending"
    plan = {
        "fact_refs": [*catalog["required_refs"], "workflow:print_preparation"],
        "steps": [{"action": "prepare_print_files", "target": "workflow:print_preparation"}],
    }
    result = resolve_instruction_plan(bind(value, plan))
    assert result["steps"][0]["state"] == "pending"
    value = context()
    value["facts"]["print_setup"]["files"]["print/definition_plate.stl"]["quantity"] = True
    value["facts"]["measured"]["solid_count"] = True
    value["facts"]["measured"]["extents_mm"] = [45, "20", 150]
    catalog = instruction_catalog(value)["facts"]
    assert catalog["workflow:print_preparation"]["state"] == "unverified"
    assert catalog["prepared:print/definition_plate.stl"]["state"] == "unverified"
    assert catalog["measurement:solid_count"]["state"] == "unverified"
    assert catalog["measurement:extents_mm"]["state"] == "unverified"


def test_complete_large_part_inventory_has_no_legacy_summary_or_size_cutoff():
    value = context()
    value["facts"]["instruction_parts"] = [
        {"part_id": f"part_{index}", "role": "make", "exported_occurrence_quantity": 1}
        for index in range(1500)
    ]
    catalog = instruction_catalog(value)
    assert "part:part_1499" in catalog["required_refs"]
    assert len([ref for ref in catalog["required_refs"] if ref.startswith("part:")]) == 1500
    plan = {
        "fact_refs": catalog["required_refs"],
        "steps": [{"action": "review_physical_limits", "target": "physical:readiness"}],
    }
    assert resolve_instruction_plan(bind(value, plan))["status"] == "bound"


def test_catalog_ignores_invalid_paths_and_never_guesses_malformed_metadata():
    value = context()
    value["facts"]["package_files"]["metadata_files"] = "not-a-list"
    value["facts"]["package_files"]["geometry"].update(
        {"../secret.step": {}, "/absolute.stl": {}, "C:/file.stl": {}}
    )
    value["facts"]["instruction_parts"].append(
        {"part_id": "bad_role", "role": [], "exported_occurrence_quantity": 1}
    )
    value["facts"]["print_setup"]["status"] = []
    catalog = instruction_catalog(value)
    assert "project:../secret.step" not in catalog["facts"]
    assert "project:/absolute.stl" not in catalog["facts"]
    assert "project:C:/file.stl" not in catalog["facts"]
    assert catalog["facts"]["part:bad_role"]["state"] == "unverified"
    assert catalog["facts"]["workflow:print_preparation"]["state"] == "unverified"


def test_resolver_performs_no_file_io_and_preserves_supplied_scopes(monkeypatch):
    value = bind(context())
    before = deepcopy(value)

    def denied(*_args, **_kwargs):
        raise AssertionError("Pure instruction resolution attempted file IO")

    monkeypatch.setattr("builtins.open", denied)
    result = resolve_instruction_plan(value)
    assert result["status"] == "bound"
    assert rows(result)["check:duty:removal/travel"]["value"]["proof_scope"] == "sampled_clearance"
    assert value == before
