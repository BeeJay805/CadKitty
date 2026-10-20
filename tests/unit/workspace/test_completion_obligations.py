"""Negative controls for saved physical-interface work, without native CAD."""

from copy import deepcopy

import pytest

from forge_mind.workspace.completion import validate_plan
from forge_mind.workspace.completion_models import FunctionalInterface
from forge_mind.workspace.completion_obligations import interface_obligations
from forge_mind.workspace.completion_work import remaining_work
from forge_mind.workspace.product_tools import _SCOPES


@pytest.fixture
def example():
    request = {
        "tool": "clearance_volume",
        "units": "mm",
        "reservation": {"occurrence_path": "member"},
        "obstacles": [{"occurrence_path": "pin"}],
    }
    contract = {"requirements": [{"id": "connect", "text": "Provide a connected assembly"}]}
    plan = {
        "subsystems": [
            {
                "id": "joint",
                "name": "Joint",
                "requirement_ids": ["connect"],
                "part_ids": ["member", "pin"],
                "checks": [{"id": "space", "request": request}],
            }
        ],
        "functional_interfaces": [
            {
                "id": "joint",
                "requirement_ids": ["connect"],
                "participant_paths": ["member", "pin"],
                "installed_occurrence_paths": ["member", "pin"],
                "required_features": [{"occurrence_path": "member", "feature_id": "body"}],
                "duties": [
                    {
                        "id": "clear",
                        "kind": "clearance",
                        "proof_scope": "static_clearance",
                        "check_ids": ["space"],
                    }
                ],
            }
        ],
    }
    artifacts = {"part.step": "saved-step", "document.json": "saved-document"}
    basis = {
        "source_fingerprint": "source",
        "engine_fingerprint": "engine",
        "artifact_hashes": artifacts,
    }
    revision = {
        "id": "revision",
        "contract": contract,
        "completion_basis": basis,
        "document": {
            "assembly": {
                "parts": [
                    {
                        "id": "member",
                        "features": [{"id": "body", "kind": "box"}],
                        "output": "body",
                        "production": {"role": "make"},
                    },
                    {
                        "id": "pin",
                        "features": [{"id": "pin_body", "kind": "cylinder"}],
                        "output": "pin_body",
                        "production": {"role": "buy"},
                    },
                ]
            }
        },
        "measurement": {
            "geometry_valid": True,
            "artifact_hashes": artifacts,
            "facts": {
                "assembly": {
                    "occurrences": [
                        {"path": "member", "definition": "member", "body_index": 0},
                        {"path": "pin", "definition": "pin", "body_index": 1},
                    ],
                    "ports": {},
                    "mates": [],
                }
            },
        },
    }
    receipt = {
        "id": "receipt",
        "revision_id": "revision",
        "contract": contract,
        **basis,
        "request": request,
        "result": {
            "tool": "clearance_volume",
            "status": "pass",
            "algorithm_scope": _SCOPES["clearance_volume"],
        },
    }
    rows = [
        {
            "id": "joint",
            "checks": [
                {
                    "id": "space",
                    "status": "pass",
                    "receipt_id": "receipt",
                    "verified_evidence": receipt,
                }
            ],
        }
    ]
    return plan, revision, rows


def test_current_clearance_is_nominal_evidence_with_no_physical_pass(example):
    result = interface_obligations(*example)
    assert result["status"] == "nominal_checked"
    assert result["physical_readiness"] == "unverified"
    assert result["next_target"] is None
    assert "binding_diagnostics" not in result["rows"][0]["duties"][0]


@pytest.mark.parametrize("role", ["reference", "unspecified"])
def test_reference_and_unassigned_parts_cannot_supply_installed_hardware(example, role):
    plan, revision, rows = example
    revision["document"]["assembly"]["parts"][1]["production"]["role"] = role
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["missing_installed_occurrence_paths"] == ["pin"]
    assert result["next_work_category"] == "construction"


def test_absent_hardware_is_not_waived_by_external_qualification(example):
    plan, revision, rows = example
    revision["measurement"]["facts"]["assembly"]["occurrences"].pop()
    plan["functional_interfaces"][0]["duties"][0]["physical_validation"] = ["Test the assembly"]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["missing_participant_paths"] == ["pin"]
    assert result["next_work_category"] == "construction"
    assert result["external_obligations"]


def test_current_failed_check_remains_actionable_with_external_qualification(example):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"][0]["physical_validation"] = ["Test the assembly"]
    rows[0]["checks"][0]["status"] = "fail"
    rows[0]["checks"][0]["verified_evidence"]["result"]["status"] = "fail"
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "check_failed"
    assert result["next_work_category"] == "repair"


def test_appended_cutter_outside_the_output_does_not_complete_the_feature(example):
    plan, revision, rows = example
    part = revision["document"]["assembly"]["parts"][0]
    part["features"].append({"id": "hole", "kind": "cylinder"})
    plan["functional_interfaces"][0]["required_features"][0]["feature_id"] = "hole"
    assert (
        interface_obligations(plan, revision, rows)["rows"][0]["status"] == "missing_construction"
    )
    part["features"].append({"id": "cut", "kind": "cut", "inputs": ["body", "hole"]})
    part["output"] = "cut"
    assert interface_obligations(plan, revision, rows)["rows"][0]["status"] == "nominal_checked"
    part["features"][-1]["suppressed"] = True
    assert (
        interface_obligations(plan, revision, rows)["rows"][0]["status"] == "missing_construction"
    )


@pytest.mark.parametrize(
    "kind", ["retention", "holding", "load_transfer", "service_access", "process"]
)
def test_geometric_pass_cannot_launder_an_unsupported_physical_duty(example, kind):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"][0]["kind"] = kind
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "unknown"
    assert result["status"] == "unverified"
    assert result["external_obligations"]
    assert "binding_diagnostics" not in result["rows"][0]["duties"][0]


@pytest.mark.parametrize(
    "field",
    [
        "revision_id",
        "artifact_hashes",
        "source_fingerprint",
        "engine_fingerprint",
        "request",
        "contract",
    ],
)
def test_stale_or_unmatched_receipt_reopens_an_executable_check(example, field):
    plan, revision, rows = example
    receipt = rows[0]["checks"][0]["verified_evidence"] = deepcopy(
        rows[0]["checks"][0]["verified_evidence"]
    )
    receipt[field] = "wrong"
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "check_pending"
    assert result["next_work_category"] == "check"
    assert result["rows"][0]["duties"][0]["binding_diagnostics"] == {
        "binding_source": "prepared_check",
        "code": "current_verified_receipt_required",
        "check_id": "space",
    }


def test_authored_status_without_a_verified_receipt_cannot_close_a_duty(example):
    plan, revision, rows = example
    rows[0]["checks"][0].pop("verified_evidence")
    assert interface_obligations(plan, revision, rows)["rows"][0]["status"] == "check_pending"


@pytest.mark.parametrize("reference", ["revision", "old_revision"])
def test_numeric_body_selection_requires_the_exact_reference_revision(example, reference):
    plan, revision, rows = example
    definition = plan["subsystems"][0]["checks"][0]
    definition["reference_revision_id"] = reference
    definition["request"]["reservation"] = {"body_index": 0}
    definition["request"]["obstacles"] = [{"body_index": 1}]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == (
        "nominal_checked" if reference == "revision" else "check_pending"
    )


def test_check_selecting_an_unrelated_part_cannot_close_the_interface(example):
    plan, revision, rows = example
    plan["subsystems"][0]["checks"][0]["request"] = deepcopy(
        plan["subsystems"][0]["checks"][0]["request"]
    )
    plan["subsystems"][0]["checks"][0]["request"]["reservation"]["occurrence_path"] = "unrelated"
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "review_pending"
    assert result["next_work_category"] == "review"


def test_static_receipt_cannot_close_a_moving_state(example):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["states"] = [{"id": "open", "kind": "motion", "driver_values": {"angle": 30.0}}]
    interface["duties"][0]["state_ids"] = ["open"]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "unknown"
    assert result["rows"][0]["duties"][0]["binding_diagnostics"]["non_reference_state_ids"] == [
        "open"
    ]


def test_required_movement_without_a_rig_or_mechanism_is_actionable(example):
    plan, revision, rows = example
    duty = plan["functional_interfaces"][0]["duties"][0]
    duty.update(kind="travel", proof_scope="physical_function", physical_validation=["Test travel"])
    result = interface_obligations(plan, revision, rows)
    assert result["next_work_category"] == "construction"
    assert result["rows"][0]["status"] == "missing_construction"


def _sampled_motion_example(example, angle=30.0):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["states"] = [{"id": "pose", "kind": "motion", "driver_values": {"angle": angle}}]
    interface["duties"] = [
        {"id": "move", "kind": "travel", "proof_scope": "sampled_clearance", "state_ids": ["pose"]}
    ]
    revision["document"]["motion_rig"] = {"kind": "prescribed_rigid_v1"}
    revision["measurement"]["facts"]["motion_rig"] = {
        "bindings": [{"path": "member", "group": "moving"}, {"path": "pin", "group": None}],
    }
    evidence = {
        "revision_id": "revision",
        "step_sha256": "saved-step",
        "document_sha256": "saved-document",
        "engine_fingerprint": "engine",
        "coverage": "discrete_only",
        "errors": [],
        "requested_configurations": [{"angle": 30.0}],
        "completed_configurations": [{"angle": 30.0}],
    }
    return plan, revision, rows, {"status": "checked", "verified_evidence": evidence}


@pytest.mark.parametrize("angle", [30.0, 45.0])
def test_sampled_motion_covers_only_the_exact_recorded_state(example, angle):
    result = interface_obligations(*_sampled_motion_example(example, angle))
    assert result["rows"][0]["status"] == ("nominal_checked" if angle == 30.0 else "unknown")
    assert result["physical_readiness"] == "unverified"
    duty = result["rows"][0]["duties"][0]
    if angle == 30.0:
        assert "binding_diagnostics" not in duty
    else:
        assert duty["binding_diagnostics"] == {
            "binding_source": "motion_receipt",
            "code": "sampled_motion_binding_incomplete",
            "uncovered_states": [{"state_id": "pose", "driver_values": {"angle": 45.0}}],
        }


def test_arithmetic_check_cannot_discharge_a_geometry_interface(example):
    plan, revision, rows = example
    request = {
        "tool": "fastener_stack",
        "units": "mm",
        "screw_length": {"low": 10.0, "high": 10.0},
        "grip_layers": [{"low": 5.0, "high": 5.0}],
        "required_engagement": 2.0,
        "available_depth": {"low": 6.0, "high": 6.0},
    }
    plan["subsystems"][0]["checks"][0]["request"] = request
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "review_pending"
    assert result["next_work_category"] == "review"
    diagnostic = result["rows"][0]["duties"][0]["binding_diagnostics"]
    assert diagnostic["code"] == "prepared_tool_not_supported"
    assert diagnostic["actual_tool"] == "fastener_stack"
    assert diagnostic["allowed_tools"] == ["clearance_volume"]


def test_real_rig_without_a_verified_motion_receipt_suggests_the_existing_check(example):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["states"] = [{"id": "pose", "kind": "motion", "driver_values": {"angle": 30.0}}]
    interface["duties"] = [
        {"id": "move", "kind": "travel", "proof_scope": "sampled_clearance", "state_ids": ["pose"]}
    ]
    revision["document"]["motion_rig"] = {"kind": "prescribed_rigid_v1"}
    revision["measurement"]["facts"]["motion_rig"] = {
        "bindings": [{"path": "member", "group": "moving"}, {"path": "pin", "group": None}]
    }
    result = interface_obligations(plan, revision, rows)
    assert result["next_work_category"] == "check"
    assert result["suggested_action"] == "motion_check"
    assert result["rows"][0]["duties"][0]["binding_diagnostics"] == {
        "binding_source": "motion_receipt",
        "code": "current_verified_motion_receipt_required",
    }


def test_unchanged_hardware_output_can_supply_legacy_inventory_but_cut_hardware_cannot(example):
    plan, revision, rows = example
    part = revision["document"]["assembly"]["parts"][1]
    part["production"]["role"] = "unspecified"
    part["features"][0]["kind"] = "hardware"
    assert interface_obligations(plan, revision, rows)["rows"][0]["status"] == "nominal_checked"
    part["features"].extend(
        [
            {"id": "tool", "kind": "cylinder"},
            {"id": "modified", "kind": "cut", "inputs": ["pin_body", "tool"]},
        ]
    )
    part["output"] = "modified"
    assert interface_obligations(plan, revision, rows)["rows"][0][
        "missing_installed_occurrence_paths"
    ] == ["pin"]


def test_unrelated_names_do_not_change_a_functional_verdict(example):
    plan, revision, rows = deepcopy(example)
    baseline = interface_obligations(plan, revision, rows)["status"]
    for original, renamed in [("member", "alpha"), ("pin", "beta")]:
        interface = plan["functional_interfaces"][0]
        interface["participant_paths"] = [
            renamed if value == original else value for value in interface["participant_paths"]
        ]
        interface["installed_occurrence_paths"] = [
            renamed if value == original else value
            for value in interface["installed_occurrence_paths"]
        ]
        for feature in interface["required_features"]:
            if feature["occurrence_path"] == original:
                feature["occurrence_path"] = renamed
        for occurrence in revision["measurement"]["facts"]["assembly"]["occurrences"]:
            if occurrence["path"] == original:
                occurrence["path"] = renamed
        request = plan["subsystems"][0]["checks"][0]["request"]
        for selector in [request["reservation"], *request["obstacles"]]:
            if selector["occurrence_path"] == original:
                selector["occurrence_path"] = renamed
    assert interface_obligations(plan, revision, rows)["status"] == baseline


def test_no_new_contract_keeps_legacy_completion_empty(example):
    plan, revision, rows = example
    plan.pop("functional_interfaces")
    result = interface_obligations(plan, revision, rows)
    assert result["status"] == "not_declared"
    assert result["rows"] == []
    assert remaining_work([], [], interface_work=result)["next_target"] is None


def test_closed_contract_rejects_authored_verdicts(example):
    raw = example[0]["functional_interfaces"][0]
    with pytest.raises(ValueError):
        FunctionalInterface.model_validate({**raw, "status": "pass"})


def test_prior_physical_duty_cannot_be_changed_to_an_easy_scope(example):
    plan, revision, _ = example
    before = deepcopy(plan)
    before["functional_interfaces"][0]["duties"][0].update(
        kind="holding", proof_scope="physical_function"
    )
    with pytest.raises(ValueError, match="duty identity"):
        validate_plan(plan, revision["contract"], previous=before)


@pytest.mark.parametrize("scope", ["physical_function", "sampled_clearance"])
def test_unrelated_moving_rig_cannot_supply_requested_participant_travel(example, scope):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["duties"] = [
        {
            "id": "move",
            "kind": "travel",
            "proof_scope": scope,
            "physical_validation": ["Test travel after construction"],
        }
    ]
    revision["document"]["motion_rig"] = {"kind": "prescribed_rigid_v1"}
    revision["measurement"]["facts"]["assembly"]["occurrences"].append(
        {"path": "cover", "definition": "member", "body_index": 2}
    )
    revision["measurement"]["facts"]["motion_rig"] = {
        "bindings": [
            {"path": "member", "group": None},
            {"path": "pin", "group": None},
            {"path": "cover", "group": "moving"},
        ]
    }
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "missing_construction"
    assert result["next_work_category"] == "construction"
    assert result["external_obligations"]
    diagnostic = result["rows"][0]["duties"][0]["binding_diagnostics"]
    assert diagnostic["expected_participant_paths"] == ["member", "pin"]
    assert diagnostic["actual_moving_paths"] == ["cover"]
    assert "missing_moving_paths" not in diagnostic


def test_current_participant_rig_keeps_physical_travel_unknown(example):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"] = [
        {"id": "move", "kind": "travel", "proof_scope": "physical_function"}
    ]
    revision["document"]["motion_rig"] = {"kind": "prescribed_rigid_v1"}
    revision["measurement"]["facts"]["motion_rig"] = {
        "bindings": [{"path": "member", "group": "moving"}, {"path": "pin", "group": None}]
    }
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "unknown"
    assert result["physical_readiness"] == "unverified"
    assert result["next_target"] is None
    assert "binding_diagnostics" not in result["rows"][0]["duties"][0]


@pytest.mark.parametrize("problem", [None, "unrelated", "grounded", "stale_step", "stale_document"])
def test_planar_movement_requires_current_exported_moving_participant_binding(example, problem):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"] = [
        {"id": "move", "kind": "travel", "proof_scope": "physical_function"}
    ]
    revision["document"]["mechanism"] = {"frame": "planar_xy_mm_deg"}
    revision["measurement"]["facts"]["assembly"]["occurrences"].append(
        {"path": "cover", "definition": "member", "body_index": 2}
    )
    mechanism = {
        "step_sha256": "saved-step",
        "document_sha256": "saved-document",
        "graph": {"components": [{"id": "moving", "grounded": problem == "grounded"}]},
        "bindings": [{"component": "moving", "body_index": 2 if problem == "unrelated" else 0}],
    }
    if problem == "stale_step":
        mechanism["step_sha256"] = "old-step"
    if problem == "stale_document":
        mechanism["document_sha256"] = "old-document"
    revision["measurement"]["facts"]["mechanism"] = mechanism
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == ("unknown" if problem is None else "missing_construction")
    assert result["next_work_category"] == (None if problem is None else "construction")
    assert result["physical_readiness"] == "unverified"


@pytest.mark.parametrize("external", [[], ["Physical strength needs a separate test"]])
def test_missing_supported_geometry_check_remains_actionable_with_external_duties(
    example, external
):
    plan, revision, rows = example
    duty = plan["functional_interfaces"][0]["duties"][0]
    duty["check_ids"] = []
    duty["physical_validation"] = external
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "review_pending"
    assert result["next_work_category"] == "review"
    assert bool(result["external_obligations"]) == bool(external)


def test_connection_without_supported_mate_binding_is_actionable(example):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"] = [
        {
            "id": "connect",
            "kind": "connection",
            "proof_scope": "nominal_fit",
            "check_ids": ["space"],
            "physical_validation": ["Test the connected product"],
        }
    ]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "review_pending"
    assert result["next_work_category"] == "review"
    assert result["external_obligations"]
    assert result["rows"][0]["duties"][0]["binding_diagnostics"] == {
        "binding_source": "assembly_mates",
        "code": "nominal_mates_required",
        "required_field": "functional_interfaces[].mate_ids",
        "expected_participant_paths": ["member", "pin"],
        "minimum_participants": 2,
    }


def test_unknown_prepared_id_is_an_actionable_mapping_gap(example):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"][0]["check_ids"] = ["undeclared"]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "review_pending"
    assert result["next_work_category"] == "review"


@pytest.mark.parametrize("connected", [False, True])
def test_one_connection_requires_a_connected_mate_graph(example, connected):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["participant_paths"].extend(["third", "fourth"])
    interface["duties"] = [{"id": "connect", "kind": "connection", "proof_scope": "nominal_fit"}]
    assembly = revision["measurement"]["facts"]["assembly"]
    assembly["occurrences"].extend(
        [
            {"path": "third", "definition": "member", "body_index": 2},
            {"path": "fourth", "definition": "pin", "body_index": 3},
        ]
    )
    assembly["mates"] = [
        {"id": "first_pair", "a": "member.axis", "b": "pin.axis", "status": "pass"},
        {"id": "second_pair", "a": "third.axis", "b": "fourth.axis", "status": "pass"},
    ]
    if connected:
        assembly["mates"].append(
            {"id": "bridge", "a": "pin.axis", "b": "third.axis", "status": "pass"}
        )
    interface["mate_ids"] = [mate["id"] for mate in assembly["mates"]]
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == ("nominal_checked" if connected else "review_pending")
    assert result["next_work_category"] == (None if connected else "review")
    assert result["physical_readiness"] == "unverified"
    duty = result["rows"][0]["duties"][0]
    if connected:
        assert "binding_diagnostics" not in duty
    else:
        diagnostic = duty["binding_diagnostics"]
        assert diagnostic["code"] == "connected_mate_graph_required"
        assert diagnostic["minimum_participants"] == 2
        assert diagnostic["outside_participant_paths"] == []
        assert diagnostic["mate_participant_pairs"] == [
            {"mate_id": "first_pair", "a": "member", "b": "pin"},
            {"mate_id": "second_pair", "a": "third", "b": "fourth"},
        ]


@pytest.mark.parametrize("record", ["port", "mate"])
@pytest.mark.parametrize("status", [None, "unverified", "unknown"])
def test_required_unmeasured_port_or_mate_cannot_close_a_passing_clearance_duty(
    example, record, status
):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    assembly = revision["measurement"]["facts"]["assembly"]
    if record == "port":
        identifier = "member.axis"
        interface["port_ids"] = [identifier]
        assembly["ports"][identifier] = {}
        measured = assembly["ports"][identifier]
    else:
        identifier = "fit_pair"
        interface["mate_ids"] = [identifier]
        measured = {"id": identifier, "a": "member.axis", "b": "pin.axis"}
        assembly["mates"].append(measured)
    if status is not None:
        measured["status"] = status
    result = interface_obligations(plan, revision, rows)
    assert result["rows"][0]["status"] == "check_pending"
    assert result["rows"][0]["pending_interface_ids"] == [identifier]
    assert result["next_work_category"] == "check"


@pytest.mark.parametrize("status", ["pass", "fail", "unverified"])
def test_nominal_connection_keeps_required_mate_measurement_status(example, status):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["duties"] = [{"id": "connect", "kind": "connection", "proof_scope": "nominal_fit"}]
    interface["mate_ids"] = ["fit_pair"]
    revision["measurement"]["facts"]["assembly"]["mates"] = [
        {"id": "fit_pair", "a": "member.axis", "b": "pin.axis", "status": status}
    ]
    result = interface_obligations(plan, revision, rows)
    expected = {"pass": "nominal_checked", "fail": "check_failed", "unverified": "check_pending"}
    assert result["rows"][0]["status"] == expected[status]
    assert result["physical_readiness"] == "unverified"
    duty = result["rows"][0]["duties"][0]
    if status != "pass":
        diagnostic = duty["binding_diagnostics"]
        assert diagnostic["binding_source"] == "assembly_mates"
        assert diagnostic["failed_mate_ids" if status == "fail" else "nonpassing_mate_ids"] == [
            "fit_pair"
        ]


def test_review_target_flows_through_existing_work_selection(example):
    plan, revision, rows = example
    plan["functional_interfaces"][0]["duties"][0]["check_ids"] = []
    projection = interface_obligations(plan, revision, rows)
    work = remaining_work([], [], interface_work=projection)
    assert work["next_target"] == "joint"
    assert work["next_work_category"] == "review"


@pytest.mark.parametrize("problem", ["extra", "unresolved", "uncovered"])
def test_prepared_diagnostics_distinguish_selection_and_coverage(example, problem):
    plan, revision, rows = example
    request = plan["subsystems"][0]["checks"][0]["request"]
    request["obstacles"] = [
        {"body_index": 99}
        if problem == "unresolved"
        else {"occurrence_path": "outside" if problem == "extra" else "member"}
    ]
    rows[0]["checks"][0]["verified_evidence"]["request"] = deepcopy(request)
    result = interface_obligations(plan, revision, rows)
    duty = result["rows"][0]["duties"][0]
    assert duty["status"] == "review_pending"
    diagnostic = duty["binding_diagnostics"]
    if problem == "uncovered":
        assert diagnostic["code"] == "prepared_check_participants_uncovered"
        assert diagnostic["missing_participant_paths"] == ["pin"]
    elif problem == "unresolved":
        assert diagnostic["code"] == "prepared_selector_unresolved"
        assert diagnostic["selected_participant_paths"] is None
        assert diagnostic["unresolved_selector"] is True
        assert "missing_participant_paths" not in diagnostic
    else:
        assert diagnostic["code"] == "prepared_check_participants_outside_interface"
        assert diagnostic["check_id"] == "space"
        assert diagnostic["expected_participant_paths"] == ["member", "pin"]
        assert diagnostic["selected_participant_paths"] == ["member", "outside"]
        assert diagnostic["extra_participant_paths"] == ["outside"]
        assert "missing_participant_paths" not in diagnostic


@pytest.mark.parametrize("problem", ["outside", "single", "missing"])
def test_mate_diagnostics_preserve_endpoint_count_and_missing_measurement_guards(example, problem):
    plan, revision, rows = example
    interface = plan["functional_interfaces"][0]
    interface["duties"] = [{"id": "connect", "kind": "connection", "proof_scope": "nominal_fit"}]
    interface["mate_ids"] = ["fit_pair"]
    if problem == "single":
        interface["participant_paths"] = ["member"]
        interface["installed_occurrence_paths"] = ["member"]
    revision["measurement"]["facts"]["assembly"]["mates"] = (
        []
        if problem == "missing"
        else [
            {
                "id": "fit_pair",
                "a": "member.axis",
                "b": "outside.axis" if problem == "outside" else "member.other",
                "status": "pass",
            }
        ]
    )
    result = interface_obligations(plan, revision, rows)
    duty = result["rows"][0]["duties"][0]
    diagnostic = duty["binding_diagnostics"]
    assert result["physical_readiness"] == "unverified"
    if problem == "missing":
        assert duty["status"] == "check_pending"
        assert diagnostic["nonpassing_mate_ids"] == ["fit_pair"]
    else:
        assert duty["status"] == "review_pending"
        assert diagnostic["code"] == "connected_mate_graph_required"
        assert diagnostic["minimum_participants"] == 2
        assert diagnostic["outside_participant_paths"] == (
            ["outside"] if problem == "outside" else []
        )


@pytest.mark.parametrize(
    "problem",
    [
        "empty_states",
        "reference_state",
        "unbound_fixed",
        "revision_id",
        "step_sha256",
        "document_sha256",
        "engine_fingerprint",
        "coverage",
        "completed_configurations",
        "errors",
    ],
)
def test_motion_diagnostics_preserve_each_current_evidence_guard(example, problem):
    plan, revision, rows, motion = _sampled_motion_example(example)
    interface = plan["functional_interfaces"][0]
    if problem == "empty_states":
        interface["duties"][0]["state_ids"] = []
    elif problem == "reference_state":
        interface["states"] = [{"id": "pose", "kind": "reference", "driver_values": {}}]
    elif problem == "unbound_fixed":
        revision["measurement"]["facts"]["motion_rig"]["bindings"].pop()
    else:
        motion["verified_evidence"][problem] = ["failure"] if problem == "errors" else "wrong"
    result = interface_obligations(plan, revision, rows, motion)
    duty = result["rows"][0]["duties"][0]
    assert duty["status"] == "unknown"
    assert result["physical_readiness"] == "unverified"
    diagnostic = duty["binding_diagnostics"]
    assert "missing_moving_paths" not in diagnostic
    if problem == "empty_states":
        assert diagnostic["empty_state_selection"] is True
    elif problem == "reference_state":
        assert diagnostic["non_motion_state_ids"] == ["pose"]
    elif problem == "unbound_fixed":
        assert diagnostic["unbound_participant_paths"] == ["pin"]
    else:
        assert diagnostic["failed_evidence_fields"] == [problem]


@pytest.mark.parametrize("binding", ["prepared", "mate", "motion"])
def test_binding_details_cannot_close_external_physical_obligations(example, binding):
    if binding == "motion":
        plan, revision, rows, motion = _sampled_motion_example(example)
    else:
        plan, revision, rows = example
        motion = None
    interface = plan["functional_interfaces"][0]
    if binding == "mate":
        interface["duties"] = [
            {"id": "connect", "kind": "connection", "proof_scope": "nominal_fit"}
        ]
        interface["mate_ids"] = ["fit_pair"]
        revision["measurement"]["facts"]["assembly"]["mates"] = [
            {"id": "fit_pair", "a": "member.axis", "b": "pin.axis", "status": "pass"}
        ]
    interface["duties"][0]["physical_validation"] = ["Test the physical product"]
    result = interface_obligations(plan, revision, rows, motion)
    assert result["rows"][0]["duties"][0]["status"] == "unknown"
    assert result["external_obligations"]
    assert result["physical_readiness"] == "unverified"
