"""Principal planes preserve local closure and map poses with proper rotations."""

import math
from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from forge_mind.workspace import mechanism
from forge_mind.workspace.document import DesignDocument
from forge_mind.workspace.mechanism_models import Mechanism, MechanismTemplate
from forge_mind.workspace.mechanism_templates import template_document

FRAMES = ("planar_xy_mm_deg", "planar_yz_mm_deg", "planar_zx_mm_deg")


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("slider_crank", "7df35cbfbb453383761b7e5a17a80a7e6dd3124839c2589d8a8a68fa8de0721b"),
        ("four_bar", "a43083d6339d34676a688596a25baa7ecca3efe711162315480f484aebe3b170"),
        ("gearbox", "f0fa0289a320e2be460e8aea3a0e610b141163ba80d0d7f780770d02b63743fc"),
    ],
)
def test_legacy_absent_and_explicit_xy_document_hashes_are_unchanged(kind, expected):
    document = template_document(MechanismTemplate(kind=kind))
    before = document.model_dump()
    assert document.fingerprint() == expected
    del before["mechanism"]["frame"]
    restored = DesignDocument.model_validate(before)
    assert restored.fingerprint() == expected
    assert restored.model_dump() == document.model_dump()


@pytest.mark.parametrize(
    "frame,columns,translation",
    [
        (FRAMES[0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [2, 3, 5]),
        (FRAMES[1], [[0, 1, 0], [0, 0, 1], [1, 0, 0]], [5, 2, 3]),
        (FRAMES[2], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], [3, 5, 2]),
    ],
)
def test_frame_and_pose_world_markers_use_right_handed_basis(frame, columns, translation):
    rotation = mechanism.frame_matrix(frame)
    expected = np.column_stack(columns)
    assert rotation[:3, :3] == pytest.approx(expected)
    assert rotation[:3, :3].T @ rotation[:3, :3] == pytest.approx(np.eye(3))
    assert np.linalg.det(rotation[:3, :3]) == pytest.approx(1)
    pose = mechanism.pose_matrix([2, 3, 90, 5], frame)
    assert pose[:3, 3] == pytest.approx(translation)
    assert pose[:3, 0] == pytest.approx(columns[1], abs=1e-15)
    assert pose[:3, 1] == pytest.approx(-np.array(columns[0]), abs=1e-15)
    assert pose[:3, 2] == pytest.approx(columns[2])
    assert pose[3] == pytest.approx([0, 0, 0, 1])
    rotation[0, 0] = 99
    assert mechanism.frame_matrix(frame)[:3, :3] == pytest.approx(expected)


@pytest.mark.parametrize("frame", FRAMES)
def test_noncommuting_rest_delta_moves_exported_marker_to_sample(frame):
    rest, sample = [11, -7, 30, 5], [-4, 9, 125, 5]
    before = deepcopy((rest, sample))
    local = np.array([2, -3, 4, 1])
    delta = mechanism.pose_delta_matrix(sample, rest, frame)
    assert delta @ mechanism.pose_matrix(rest, frame) @ local == pytest.approx(
        mechanism.pose_matrix(sample, frame) @ local
    )
    assert mechanism.pose_delta_matrix(rest, rest, frame) == pytest.approx(np.eye(4))
    assert (rest, sample) == before


@pytest.mark.parametrize("frame", ["planar_xz_mm_deg", "XY", "oblique", "", None, True])
def test_unknown_frames_are_refused_by_model_and_matrix_helper(frame):
    raw = template_document(MechanismTemplate()).mechanism.model_dump()
    with pytest.raises(ValidationError):
        Mechanism.model_validate({**raw, "frame": frame})
    with pytest.raises(ValueError, match="frame"):
        mechanism.frame_matrix(frame)


@pytest.mark.parametrize(
    "pose",
    [
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1, 2, float("nan"), 4],
        [float("inf"), 2, 3, 4],
        [1, 2, True, 4],
        [1, 2, "3", 4],
        [10**1000, 2, 3, 4],
        None,
    ],
)
def test_nonfinite_or_malformed_pose_cannot_become_a_world_transform(pose):
    with pytest.raises(ValueError, match="pose"):
        mechanism.pose_matrix(pose, FRAMES[0])


def test_finite_pose_delta_overflow_is_refused():
    with pytest.raises(ValueError, match="delta must remain finite"):
        mechanism.pose_delta_matrix([1e308, 0, 0, 0], [-1e308, 0, 0, 0])


@pytest.mark.parametrize("frame", FRAMES)
@pytest.mark.parametrize("kind", ["slider_crank", "four_bar"])
def test_frame_choice_does_not_change_local_closure_or_mobility(kind, frame):
    document = template_document(MechanismTemplate(kind=kind))
    document.mechanism.driver.samples = 5
    baseline = mechanism.resolve_mechanism(document.mechanism, document.values())
    framed = {**baseline, "frame": frame}
    assert mechanism.study_mechanism(framed) == mechanism.study_mechanism(baseline)


@pytest.mark.parametrize("frame", FRAMES)
def test_slider_world_travel_matches_independent_closed_form(frame):
    document = template_document(MechanismTemplate(kind="slider_crank"))
    document.mechanism.frame = frame
    graph = mechanism.resolve_mechanism(document.mechanism, document.values())
    for angle in (0, 45, 90, 180, 270, 360, 720):
        solved = mechanism.solve_mechanism(graph, angle)
        assert solved["status"] == "solved"
        pose = solved["poses"]["slider"]
        distance = 20 * math.cos(math.radians(angle)) + math.sqrt(
            65**2 - (20 * math.sin(math.radians(angle))) ** 2
        )
        expected = {
            FRAMES[0]: [distance, 0, pose[3]],
            FRAMES[1]: [pose[3], distance, 0],
            FRAMES[2]: [0, pose[3], distance],
        }[frame]
        assert mechanism.pose_matrix(pose, frame)[:3, 3] == pytest.approx(expected, abs=1e-6)
