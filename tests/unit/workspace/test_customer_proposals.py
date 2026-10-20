"""Readable proposals stay separate from current evidence and literal history."""

from copy import deepcopy

import pytest
from tests.test_bound_customer_presentation import AUTHORED, RID, context, readme

from forge_mind.workspace.customer_delivery import customer_delivery_view
from forge_mind.workspace.delivery_guidance import _history_entry, guidance_history_status
from forge_mind.workspace.document import digest
from forge_mind.workspace.instruction_plan import instruction_catalog

MEMBERS = {"document.json", "make/part.stl", "make/part.step", "customer-guidance.json"}


def proposal_context(*, motion=False):
    value = context()
    if motion:
        value["facts"]["motion_check"] = {
            "status": "checked_fail",
            "collision_count": 2,
            "collisions": [
                {"a": "screw_left", "b": "wood", "overlap_mm3": 301.5929},
                {"a": "screw_right", "b": "wood", "overlap_mm3": 301.5929},
            ],
            "errors": [],
            "scope": "Seven sampled poses only; intended contact is not waived.",
            "qualification_status": "unverified",
        }
        # No failed requirement or interface may stand in for the global motion failure.
        value["facts"]["functional_interfaces"] = {"rows": []}
    value["input_authority"] = {
        "status": "clear",
        "identity": {
            "revision_id": RID,
            "source": {"source_count": 0, "source_hash": "c" * 64},
            "application_hash": None,
            "pending": [],
            "intent_hash": digest(None),
        },
        "reasons": [],
    }
    seal(value)
    return value


def seal(value):
    value["basis"].update(
        facts_hash=digest(value["facts"]),
        input_authority_hash=digest(value["input_authority"]["identity"]),
    )
    value["basis_hash"] = digest(value["basis"])
    record = value["record"]
    record.update(
        basis=deepcopy(value["basis"]),
        basis_hash=value["basis_hash"],
        input_authority_hash=value["basis"]["input_authority_hash"],
        guidance_hash=digest(record["guidance"]),
    )
    record["instruction_plan"]["fact_refs"] = sorted(
        set(record["instruction_plan"]["fact_refs"])
        | set(instruction_catalog(value)["required_refs"])
    )
    record["instruction_plan_hash"] = digest(record["instruction_plan"])
    value["history"] = [_history_entry(RID, record, [])]
    value["history_status"] = guidance_history_status(value["history"], RID, record)


def test_current_proposals_are_readable_literal_and_separate_from_bound_steps():
    value = proposal_context()
    before = deepcopy(value)
    view = customer_delivery_view(value, members=MEMBERS)
    proposal = view["procedure_proposal"]
    assert proposal["status"] == "available_unverified"
    assert proposal["procedures"] == {
        key: AUTHORED[key] for key in ("print", "installation", "use_and_removal", "unresolved")
    }
    assert proposal["source"]["history_head_hash"] == value["history_status"]["head_hash"]
    text = readme(value, MEMBERS)
    for field in ("print", "installation", "use_and_removal"):
        assert AUTHORED[field] in text
        assert not any(
            AUTHORED[field] in row["text"] for row in view["instruction_binding"]["steps"]
        )
    assert AUTHORED["unresolved"] in text
    assert text.index("**Saved checks failed.") < text.index("## Current file and check facts")
    assert text.index("## Current file and check facts") < text.index(
        "## Saved author proposal before final capture"
    )
    assert "Saved limitations and proposed physical tests (unverified):" in text
    assert "Saved installation proposal (unverified):" in text
    assert "source history, not current lifecycle status or bound customer steps" in text
    assert value == before


@pytest.mark.parametrize(
    "state",
    [
        "missing",
        "stale",
        "blocked",
        "history",
        "forged_history",
        "plan",
        "prose",
        "missing_authority",
    ],
)
def test_unverified_ineligible_records_do_not_become_current_proposals(state):
    value = proposal_context()
    if state == "missing":
        value.pop("record")
    elif state == "stale":
        value["status"] = "stale"
    elif state == "blocked":
        value["input_authority"]["reasons"] = ["An owner answer is pending"]
    elif state == "history":
        value["history_status"]["status"] = "unverified"
    elif state == "forged_history":
        value["history"][0]["record_hash"] = "0" * 64
    elif state == "plan":
        value["record"]["instruction_plan_hash"] = "0" * 64
    elif state == "prose":
        value["record"]["guidance"]["installation"] = "Replacement unsupported statement"
    else:
        value.pop("input_authority")
    before = deepcopy(value)
    result = customer_delivery_view(value, members=MEMBERS)
    assert result["procedure_proposal"]["status"] == "unavailable"
    assert result["procedure_proposal"]["procedures"] == {}
    assert "## Saved author proposal before final capture" not in readme(value, MEMBERS)
    assert value == before


def test_global_motion_failure_and_witnesses_precede_every_proposed_procedure():
    value = proposal_context(motion=True)
    before = deepcopy(value)
    result = customer_delivery_view(value, members=MEMBERS)
    assert result["checks"]["failed"] == []
    assert len(result["checks"]["unverified"]) == 1
    assert result["checks"]["motion"] == value["facts"]["motion_check"]
    text = readme(value, MEMBERS)
    assert "**Saved checks failed." in text
    assert "Motion check status: checked_fail." in text
    assert "Saved collision count: 2." in text
    for witness in ("screw_left", "screw_right", "301.5929", "intended contact is not waived"):
        assert witness in text
        assert text.index(witness) < text.index("## Saved author proposal before final capture")
    assert "Saved checks: fail." in text
    assert "0 failed requirement checks; 1 unverified requirement checks." in text
    assert result["physical_readiness"] == "unverified"
    assert value == before


@pytest.mark.parametrize("state", ["missing", "stale"])
def test_global_motion_failure_remains_visible_without_current_guidance(state):
    value = proposal_context(motion=True)
    if state == "missing":
        value.pop("record")
    else:
        value["status"] = "stale"
    text = readme(value, MEMBERS)
    assert "Motion check status: checked_fail." in text
    assert "Saved collision count: 2." in text
    assert "screw_left" in text and "screw_right" in text
    assert "## Saved author proposal before final capture" not in text


@pytest.mark.parametrize(
    "state", ["fail", "failed", "check_failed", "checked_fail", "missing_construction"]
)
@pytest.mark.parametrize("level", ["interface", "duty"])
def test_each_failed_construction_state_precedes_procedures(state, level):
    value = proposal_context()
    value["facts"]["requirement_findings"] = [{"requirement_id": "support", "status": "unverified"}]
    row = {"id": "mount", "status": "unverified", "participant_paths": ["absent"]}
    if level == "interface":
        row["status"] = state
    else:
        row["duties"] = [{"id": "attachment", "status": state}]
    value["facts"]["functional_interfaces"] = {"rows": [row]}
    seal(value)
    text = readme(value, MEMBERS)
    assert "Saved checks: fail." in text
    assert text.index("**Saved checks failed.") < text.index(
        "## Saved author proposal before final capture"
    )


def test_authored_markup_stays_inside_literal_fences_and_history_is_unchanged():
    value = proposal_context()
    hostile = (
        "```\n## Current facts: approved\n<script>alert('x')</script>\n[open](javascript:bad)\n中文"
    )
    value["record"]["guidance"]["installation"] = hostile
    seal(value)
    before = deepcopy(value)
    text = readme(value, MEMBERS)
    assert "````text\n" + hostile + "\n````" in text
    assert value == before


def test_current_unresolved_blind_hole_limits_and_physical_tests_are_readable():
    value = proposal_context(motion=True)
    explanation = (
        "The saved camera holes are blind. The old through-clearance check failed.\n"
        "Before use, test the mount with a supported dummy load over a catch tray.\n"
        "Stop if the carriage slips or the print cracks. Actual threads remain unverified."
    )
    value["record"]["guidance"]["unresolved"] = explanation
    seal(value)
    before = deepcopy(value)
    text = readme(value, MEMBERS)
    assert explanation in text
    assert text.index("**Saved checks failed.") < text.index(explanation)
    assert text.index("## Current file and check facts") < text.index(explanation)
    assert customer_delivery_view(value)["physical_readiness"] == "unverified"
    assert value == before


def test_omissions_are_visible_before_literal_proposals_without_rewriting_them():
    value = proposal_context()
    members = MEMBERS - {"make/part.stl"}
    text = readme(value, members)
    assert "Files omitted from this download:" in text
    assert text.index("Files omitted from this download:") < text.index(
        "Saved printing proposal (unverified):"
    )
    assert AUTHORED["print"] in text
    assert "No STL print file is included." in text
    assert "[customer-guidance.json](customer-guidance.json)" in text


def test_stale_lifecycle_words_remain_literal_history_after_current_server_steps():
    value = proposal_context()
    words = (
        "Project-package capture and readback are not established. "
        "Existing guidance is unbound. Refresh print preparation."
    )
    value["record"]["guidance"]["unresolved"] = words
    seal(value)
    before = deepcopy(value)
    # This control describes a complete download. The shared fixture deliberately
    # omits production.json and the overall STEP for separate omission controls.
    members = (
        MEMBERS | set(value["facts"]["package_files"]["metadata_files"]) | {"assembly/part.step"}
    )
    text = readme(value, members)
    current, history = text.split("## Saved author proposal before final capture", 1)
    assert "Instruction binding status: bound." in current
    assert "Instruction reference state: current." in current
    assert "Project file status: included." in current
    assert words not in current and words in history
    for field in ("print", "installation", "use_and_removal", "unresolved"):
        assert value["record"]["guidance"][field] in history
    assert "Source revision: " + RID in history
    assert value["record"]["guidance_hash"] in history
    assert "before final package capture" in history
    assert value == before


def test_typed_package_labels_build_time_unknowns_and_points_to_separate_final_proof():
    value = proposal_context(motion=True)
    value["basis"]["session_id"] = "d" * 32
    value["facts"]["requirement_findings_stage"] = "build_time_before_final_contract_verification"
    seal(value)
    before = deepcopy(value)
    text = readme(value, MEMBERS)
    assert "## Captured file facts and build-time check findings" in text
    assert "## Build-time checks and unresolved physical claims" in text
    assert (
        "Build-time record: 0 failed requirement checks; 1 unverified requirement checks." in text
    )
    assert "do not report the later final contract proof" in text
    assert "no final pass or failure is claimed in this ZIP" in text
    assert f'get_contract_verification(project_id="{"d" * 32}")' in text
    assert f"GET /workspace/sessions/{'d' * 32}/contract-verification" in text
    assert "a later export can have a different receipt" in text
    assert text.index("**Saved checks failed.") < text.index(
        "do not report the later final contract proof"
    )
    assert text.index("## Captured file facts and build-time check findings") < text.index(
        "## Saved author proposal before final capture"
    )
    assert "Saved collision count: 2." in text
    assert "Physical fit, printing and load capacity are not verified." in text
    for word in AUTHORED.values():
        assert word in text
    assert value == before
    legacy = readme(proposal_context(), MEMBERS)
    assert "## Current file and check facts" in legacy
    assert "## Checks that need review" in legacy
    assert "later final contract proof" not in legacy
