"""Multiple contact islands must produce numerical failures, never lost witnesses."""

from types import SimpleNamespace

import build123d as b
import numpy as np
import pytest

from forge_mind.workspace.assembly_review import assembly_motion_review
from forge_mind.workspace.engineering import assembly_review, overlap
from forge_mind.workspace.engineering_models import EngineeringInputs
from forge_mind.workspace.geometry_results import boolean_volume
from forge_mind.workspace.mechanism_models import MechanismTemplate
from forge_mind.workspace.mechanism_review import mechanism_review
from forge_mind.workspace.mechanism_templates import template_document
from forge_mind.workspace.motion_check import review_poses
from forge_mind.workspace.product_tools import _volume


@pytest.fixture
def contacts():
    # A U-shaped solid cut by a plate gives two separate 2 x 2 x 1 contact islands.
    fork = (b.Box(6, 2, 4) - b.Pos(0, 0, 1) * b.Box(2, 4, 4)).solids()[0]
    plate = (b.Pos(0, 0, 1) * b.Box(6, 2, 1)).solids()[0]
    common = fork & plate
    assert isinstance(common, b.ShapeList)
    assert len(common) == 2
    return [fork, plate]


def test_static_report_keeps_all_contact_islands(contacts):
    assert overlap(*contacts) == pytest.approx(8)
    report, _ = assembly_review(contacts, EngineeringInputs(), "revision")
    assert report["pairs"][0]["status"] == "fail"
    assert report["pairs"][0]["overlap_mm3"] == pytest.approx(8)
    assert _volume(contacts[0] & contacts[1]) == pytest.approx(8)
    assert overlap(contacts[0], b.Pos(20, 0, 0) * contacts[1]) == 0


@pytest.mark.parametrize("optimized", [False, True])
def test_prescribed_motion_reports_disconnected_collision(contacts, optimized):
    report = review_poses(
        contacts,
        ["fork", "plate"],
        ["moving", None],
        [({"angle": 0.0}, [np.eye(4), np.eye(4)])],
        optimized=optimized,
    )
    assert report["status"] == "fail"
    assert report["errors"] == []
    assert report["completed_pairs"] == 1
    assert report["collisions"][0]["overlap_mm3"] == pytest.approx(8)


def test_solved_assembly_and_gear_review_keep_contact_islands(contacts, monkeypatch):
    pose = {
        "status": "solved",
        "angle_deg": 0,
        "residual_mm_equivalent": 0,
        "poses": {"a": [0, 0, 0, 0], "b": [0, 0, 0, 0]},
    }
    motion = {
        "current": pose,
        "bindings": [{"component": cid, "body_index": i} for i, cid in enumerate(["a", "b"])],
        "graph": {"components": [], "gears": [{"id": "pair", "a": "a", "b": "b"}]},
        "gear_layout": [],
        "status": "sampled",
        "requested_samples": 1,
        "range_deg": [0, 0],
        "method": "controlled contact fixture",
        "limitations": [],
    }
    shape = b.Compound(children=contacts)
    monkeypatch.setattr("forge_mind.workspace.assembly_review.contact_poses", lambda _: [])
    report = assembly_motion_review(
        shape, motion, {"occurrences": [{"body_index": i, "path": str(i)} for i in range(2)]}
    )
    assert report["status"] == "fail"
    assert report["errors"] == []
    assert report["collisions"][0]["overlap_mm3"] == pytest.approx(8)
    monkeypatch.setattr("forge_mind.workspace.mechanism_review.contact_poses", lambda _: [pose])
    document = template_document(MechanismTemplate(kind="gearbox"))
    gear_report = mechanism_review(shape, document, motion, EngineeringInputs())
    witness = gear_report["tooth_interference"][0]
    assert witness["status"] == "fail"
    assert witness["coverage"] == "discrete_only"
    assert witness["samples"][0]["overlap_mm3"] == pytest.approx(8)


@pytest.mark.parametrize("volume", [float("nan"), float("inf"), -1.0])
def test_invalid_member_cannot_masquerade_as_clearance(volume):
    with pytest.raises(ValueError, match="Invalid CAD Boolean volume"):
        boolean_volume([SimpleNamespace(volume=4), SimpleNamespace(volume=volume)])


def test_empty_and_single_boolean_results():
    assert boolean_volume(None) == boolean_volume([]) == 0
    assert boolean_volume(b.Box(1, 2, 3)) == pytest.approx(6)
    with pytest.raises(AttributeError):
        boolean_volume(object())
