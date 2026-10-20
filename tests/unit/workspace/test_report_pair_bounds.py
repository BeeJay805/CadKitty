"""Pose broadphase preserves collisions without measuring every distant pair."""

import json

import build123d as b
import pytest

from forge_mind.workspace.engineering import assembly_review
from forge_mind.workspace.engineering_models import EngineeringInputs


def test_separated_rotated_pair_reports_conservative_bound_without_exact_query(monkeypatch):
    a = b.Box(4, 5, 6).rotate(b.Axis.Z, 35)
    other = b.Pos(20, 30, 40) * b.Box(6, 5, 4).rotate(b.Axis.X, 21)
    exact = a.distance_to(other)

    def forbidden(*args, **kwargs):
        raise AssertionError("Distant pose pairs must not perform an exact distance query")

    monkeypatch.setattr(b.Shape, "distance_to", forbidden)
    report, _ = assembly_review([a, other], EngineeringInputs(), "revision")
    pair = report["pairs"][0]
    assert pair["status"] == "pass" and pair["overlap_mm3"] == 0
    assert pair["distance_mm"] is None
    assert 0 < pair["distance_lower_bound_mm"] <= exact
    assert "not evaluated" in pair["distance_method"]
    assert report["pair_evaluation"]["exact_distance_queries"] == 0
    assert json.loads(json.dumps(report))["pairs"][0]["distance_mm"] is None


@pytest.mark.parametrize("offset", [0, 1.5, 2, 2.0000005])
def test_overlapping_and_near_touching_bounds_keep_exact_distance(monkeypatch, offset):
    a, other = b.Box(2, 2, 2), b.Pos(offset, 0, 0) * b.Box(2, 2, 2)
    original = b.Shape.distance_to
    calls = []

    def tracked(self, shape):
        calls.append(True)
        return original(self, shape)

    monkeypatch.setattr(b.Shape, "distance_to", tracked)
    report, _ = assembly_review([a, other], EngineeringInputs(), "revision")
    pair = report["pairs"][0]
    assert len(calls) == 1
    assert pair["distance_mm"] == pytest.approx(max(0, offset - 2), abs=1e-7)
    assert pair["distance_lower_bound_mm"] is None
    assert pair["status"] == ("fail" if offset < 2 else "pass")


def test_overlapping_boxes_with_separated_solids_retain_exact_proximity():
    a, other = b.Sphere(1), b.Pos(1.5, 1.5, 0) * b.Sphere(1)
    report, _ = assembly_review([a, other], EngineeringInputs(), "revision")
    pair = report["pairs"][0]
    assert pair["status"] == "pass"
    assert pair["distance_mm"] == pytest.approx(2**0.5 * 1.5 - 2)
    assert pair["distance_lower_bound_mm"] is None


def test_service_path_tool_radius_still_uses_exact_clearance():
    inputs = EngineeringInputs.model_validate(
        {
            "assembly_revision": "a" * 32,
            "service_paths": [
                {
                    "id": "tool",
                    "body": 1,
                    "travel": [0, 0, 1],
                    "samples": 3,
                    "task": "Remove with tool envelope",
                    "tool_radius_mm": 2,
                }
            ],
        }
    )
    report, _ = assembly_review([b.Box(2, 2, 2), b.Pos(3, 0, 0) * b.Box(2, 2, 2)], inputs, "a" * 32)
    assert report["pose_status"] == "pass"
    assert report["pairs"][0]["distance_mm"] is None
    assert report["status"] == "fail"
    witness = report["service_paths"][0]["witness"]
    assert witness["overlap_mm3"] == 0 and witness["distance_mm"] == pytest.approx(1)
