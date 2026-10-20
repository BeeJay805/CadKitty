"""Independent equations, actual BRep realizations and failure/unknown controls."""

import math

import numpy as np
import pytest
from pydantic import ValidationError

from forge_mind.workspace.assembly import named_solids, occurrence_label
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.kernel import compile_document
from forge_mind.workspace.mechanism import resolve_mechanism, solve_mechanism
from forge_mind.workspace.synthesis import (
    candidate_document,
    compare_mechanisms,
    linkage_jacobian_bound,
    linkage_kinematics,
)
from forge_mind.workspace.synthesis_models import (
    LinkageCandidate,
    SynthesisRequest,
    example_synthesis_request,
)


@pytest.fixture(scope="module")
def comparison():
    return compare_mechanisms(example_synthesis_request())


def test_screw_and_rack_equations_include_both_jaws(comparison):
    screw, rack, _ = comparison["candidates"]
    assert screw["metrics"]["min_force_per_jaw_n"] == pytest.approx(math.pi * 1000 * 0.65 / 4)
    assert screw["metrics"]["close_time_s"] == pytest.approx(60 / (2 * 4))
    assert rack["metrics"]["min_force_per_jaw_n"] == pytest.approx(1000 * 0.65 / (2 * 15))
    assert rack["metrics"]["close_time_s"] == pytest.approx(60 / (2 * 15 * 2 * math.pi))


def test_all_candidate_curves_conserve_two_jaw_work(comparison):
    for candidate in comparison["candidates"]:
        for row in candidate["curves"]:
            assert row["status"] == "computed"
            assert row["closing_speed_mm_s"] == 2 * row["jaw_speed_mm_s"]
            for bound in ("low", "high"):
                work = 2 * row[f"force_per_jaw_{bound}_n"] * row["jaw_speed_mm_s"] / 1000
                assert work == pytest.approx(row[f"output_power_{bound}_w"], rel=1e-12)
            assert row["force_per_jaw_low_n"] <= row["force_per_jaw_high_n"]
            assert row["required_torque_low_nm"] <= row["required_torque_high_nm"]


def test_linkage_jacobian_matches_independent_finite_difference():
    candidate = example_synthesis_request().candidates[2]
    assert isinstance(candidate, LinkageCandidate)
    for theta in np.linspace(0.15, 2.9, 13):

        def gap(angle):
            x = candidate.crank_mm * math.cos(angle)
            y = candidate.crank_mm * math.sin(angle)
            return 2 * (x + math.sqrt(candidate.rod_mm**2 - y * y) - candidate.jaw_offset_mm)

        opening, derivative = linkage_kinematics(candidate, float(theta))
        assert opening == pytest.approx(gap(theta))
        assert derivative == pytest.approx((gap(theta + 1e-6) - gap(theta - 1e-6)) / 2e-6, rel=1e-7)


def test_linkage_force_screen_includes_between_sample_extremum():
    raw = example_synthesis_request().model_dump()
    raw["samples"] = 3
    result = compare_mechanisms(SynthesisRequest.model_validate(raw))["candidates"][2]
    candidate = example_synthesis_request().candidates[2]
    assert isinstance(candidate, LinkageCandidate)
    derivatives = []
    for theta in np.linspace(math.radians(5), math.radians(175), 50001):
        gap, derivative = linkage_kinematics(candidate, float(theta))
        if 20 <= gap <= 80:
            derivatives.append(abs(derivative))
    assert result["metrics"]["estimated_min_force_per_jaw_n"] == pytest.approx(
        650 / max(derivatives), rel=1e-7
    )
    assert result["metrics"]["min_force_per_jaw_n"] <= 650 / max(derivatives)
    assert result["metrics"]["min_force_per_jaw_n"] <= min(
        c["force_per_jaw_low_n"] for c in result["curves"]
    )


@pytest.mark.parametrize("ratio", [1.000001, 1.01, 1.2, 2, 4, 8])
def test_interval_jacobian_bound_covers_dense_arbitrary_linkage_ratios(ratio):
    raw = example_synthesis_request().candidates[2].model_dump()
    raw.update(crank_mm=20, rod_mm=20 * ratio)
    candidate = LinkageCandidate.model_validate(raw)
    for start, end in [(0, math.pi), (0.03, 1.1), (1.4, 1.8), (1.8, 3.1), (2.7, 2.701)]:
        # Independent Cartesian slider expression differentiated with numpy.
        angles = np.linspace(start, end, 10001)
        sine, cosine = np.sin(angles), np.cos(angles)
        exact = 40 * sine * (1 + 20 * cosine / np.sqrt((20 * ratio) ** 2 - (20 * sine) ** 2))
        bound = linkage_jacobian_bound(candidate, start, end)
        assert bound >= float(np.max(np.abs(exact)))
        assert math.isfinite(bound)


def test_linkage_force_bound_gap_is_unknown_and_evaluated_violation_is_fail(comparison):
    metrics = comparison["candidates"][2]["metrics"]
    lower = metrics["min_force_per_jaw_n"]
    estimate = metrics["estimated_min_force_per_jaw_n"]
    assert lower < estimate
    assert metrics["max_jacobian_bound_mm_per_rad"] > metrics["max_jacobian_numerical_mm_per_rad"]
    for target, expected in [
        (lower * 0.99, "pass"),
        ((lower + estimate) / 2, "unverified"),
        (estimate * 1.01, "fail"),
    ]:
        raw = example_synthesis_request().model_dump()
        raw["task"]["required_force_per_jaw_n"] = target
        result = compare_mechanisms(SynthesisRequest.model_validate(raw))["candidates"][2]
        check = next(c for c in result["checks"] if c["id"] == "force_per_jaw")
        assert check["status"] == expected
        assert check["method"] == "interval_factor_upper_jacobian_bound"
        assert result["qualification_status"] == "unverified"
        if expected == "fail":
            witness = result["metrics"]["force_witness"]
            assert 20 <= witness["opening_mm"] <= 80
            assert witness["force_per_jaw_low_n"] < target
            candidate = example_synthesis_request().candidates[2]
            assert isinstance(candidate, LinkageCandidate)
            _, derivative = linkage_kinematics(candidate, math.radians(witness["input_angle_deg"]))
            assert witness["force_per_jaw_low_n"] == pytest.approx(650 / abs(derivative))


@pytest.mark.parametrize("index,count", [(0, 12), (1, 12), (2, 9)])
def test_real_documents_compile_and_jaw_planes_match_task(index, count):
    setup = example_synthesis_request()
    document = candidate_document(setup, setup.candidates[index].id)
    shape, _ = compile_document(document, {})
    assert shape.is_valid and len(shape.solids()) == count
    bodies = named_solids(shape)
    suffix = "jaw" if index == 2 else "pad"
    left = bodies[occurrence_label("left_" + suffix)].bounding_box()
    right = bodies[occurrence_label("right_" + suffix)].bounding_box()
    assert pytest.approx(50, abs=1e-5) == right.min.X - left.max.X
    assert pytest.approx(0, abs=1e-5) == (right.min.X + left.max.X) / 2
    assert document.assembly
    assert all(p.production.material is None for p in document.assembly.parts)
    if index != 2:
        bought = [p for p in document.assembly.parts if p.production.role == "buy"]
        assert bought and all(p.production.procurement is None for p in bought)
        assert document.motion_rig is not None
    else:
        assert document.mechanism is not None


def test_solved_linkage_graph_matches_analytic_opening_and_rod_lengths():
    setup = example_synthesis_request()
    candidate = setup.candidates[2]
    assert isinstance(candidate, LinkageCandidate)
    document = candidate_document(setup, candidate.id)
    assert document.mechanism
    assert linkage_kinematics(candidate, math.radians(document.mechanism.driver.start))[
        0
    ] == pytest.approx(80)
    assert linkage_kinematics(candidate, math.radians(document.mechanism.driver.end))[
        0
    ] == pytest.approx(20)
    graph = resolve_mechanism(document.mechanism, document.values())
    seed = None
    for angle in [35, 55, 75, 95, 115]:
        solution = solve_mechanism(graph, angle, seed)
        assert solution["status"] == "solved"
        seed = solution["seed"]
        poses = solution["poses"]
        opening = poses["right_jaw"][0] - poses["left_jaw"][0] - 2 * candidate.jaw_offset_mm
        assert opening == pytest.approx(
            linkage_kinematics(candidate, math.radians(angle))[0], abs=1e-5
        )
        assert poses["right_jaw"][0] + poses["left_jaw"][0] == pytest.approx(0, abs=1e-5)
        for side in ("left", "right"):
            rod, jaw = poses[side + "_rod"], poses[side + "_jaw"]
            assert math.hypot(jaw[0] - rod[0], jaw[1] - rod[1]) == pytest.approx(
                candidate.rod_mm, abs=1e-5
            )


def test_unknown_efficiency_does_not_default_to_perfect_transmission():
    raw = example_synthesis_request().model_dump()
    raw["candidates"][0]["efficiency"] = None
    result = compare_mechanisms(SynthesisRequest.model_validate(raw))["candidates"][0]
    assert result["metrics"]["min_force_per_jaw_n"] is None
    assert result["checks"][1]["status"] == "unverified"
    assert all(
        row["force_per_jaw_low_n"] is None and row["output_power_high_w"] is None
        for row in result["curves"]
    )


def test_parameter_change_updates_motion_and_screening_without_changing_task(comparison):
    changed = example_synthesis_request().model_dump()
    changed["candidates"][0]["lead_mm"] = 8
    after = compare_mechanisms(SynthesisRequest.model_validate(changed))
    before, modified = comparison["candidates"][0], after["candidates"][0]
    assert after["task_hash"] == comparison["task_hash"]
    assert after["request_hash"] != comparison["request_hash"]
    assert before["document_hash"] != modified["document_hash"]
    assert modified["metrics"]["min_force_per_jaw_n"] == pytest.approx(
        before["metrics"]["min_force_per_jaw_n"] / 2
    )
    assert modified["metrics"]["close_time_s"] == pytest.approx(
        before["metrics"]["close_time_s"] / 2
    )
    assert before["status"] == "fail" and modified["status"] == "pass"
    assert modified["qualification_status"] == "unverified"


@pytest.mark.parametrize("maximum,angle_min", [(100, 5), (84, 0)])
def test_unreachable_or_singular_branch_cannot_realize_or_claim_force(maximum, angle_min):
    raw = example_synthesis_request().model_dump()
    raw["task"]["opening_max_mm"] = maximum
    raw["candidates"][2]["angle_min_deg"] = angle_min
    setup = SynthesisRequest.model_validate(raw)
    result = compare_mechanisms(setup)["candidates"][2]
    assert result["status"] == "fail" and result["document"] is None
    assert result["metrics"]["min_force_per_jaw_n"] is None
    with pytest.raises(ValueError):
        candidate_document(setup, setup.candidates[2].id)


def test_source_and_target_changes_invalidate_dependencies_without_confirming_sources(comparison):
    raw = example_synthesis_request().model_dump()
    raw["candidates"][0]["source_claim_ids"] = ["0" * 32]
    result = compare_mechanisms(SynthesisRequest.model_validate(raw))
    assert result["request_hash"] != comparison["request_hash"]
    assert result["candidates"][0]["dependencies"]["source_claim_ids"] == ["0" * 32]
    assert result["candidates"][0]["qualification_status"] == "unverified"
    raw["task"]["required_force_per_jaw_n"] = 1000
    assert (
        compare_mechanisms(SynthesisRequest.model_validate(raw))["task_hash"] != result["task_hash"]
    )


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("drive", "torque_nm", float("nan")),
        ("drive", "rpm", float("inf")),
        ("drive", "torque_nm", True),
        ("task", "opening_min_mm", -1),
        ("task", "opening_max_mm", 10),
        ("task", "required_force_per_jaw_n", 0),
    ],
)
def test_invalid_or_nonfinite_assumptions_rejected(section, key, value):
    raw = example_synthesis_request().model_dump()
    raw[section][key] = value
    with pytest.raises(ValidationError):
        SynthesisRequest.model_validate(raw)


def test_missing_targets_remain_unverified_and_no_winner_is_invented():
    raw = example_synthesis_request().model_dump()
    for key in ("required_force_per_jaw_n", "max_close_time_s", "max_width_mm"):
        raw["task"][key] = None
    result = compare_mechanisms(SynthesisRequest.model_validate(raw))
    assert all(c["status"] == "unverified" for c in result["candidates"])
    assert "winner" not in result and "rank" not in result
    assert all(
        c["document"]
        and c["document_hash"] == DesignDocument.model_validate(c["document"]).fingerprint()
        for c in result["candidates"]
    )
