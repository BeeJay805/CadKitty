"""Scheduled motion work can finish without a hidden exact-pair ceiling."""

from types import SimpleNamespace

import numpy as np
import pytest

from forge_mind.workspace import engineering, motion_check


@pytest.mark.parametrize("seconds", [None, 10.0])
def test_large_finite_motion_schedule_respects_only_explicit_time_budget(monkeypatch, seconds):
    clock = [0.0]

    class Shape:
        def __and__(self, other):
            clock[0] += 1.0
            return self

    monkeypatch.setattr(
        motion_check,
        "time",
        SimpleNamespace(perf_counter=lambda: clock[0], process_time=lambda: clock[0]),
    )
    monkeypatch.setattr(
        motion_check, "shape_bounds", lambda shape: np.array([[0, 0, 0], [1, 1, 1]])
    )
    monkeypatch.setattr(engineering, "transform_shape", lambda shape, transform: shape)
    monkeypatch.setattr(motion_check, "boolean_volume", lambda common: 0.0)
    names = [str(i) for i in range(128)]
    poses = [({"position": float(i)}, [np.eye(4)] * 128) for i in range(2)]
    report = motion_check.review_poses(
        [Shape() for _ in names], names, names, poses, seconds=seconds
    )
    assert report["total_pairs"] == 16256
    assert report["collisions"] == []
    # Completed sampling remains explicitly unverified for continuous motion.
    assert report["status"] == "unverified"
    if seconds is None:
        assert report["exact_intersections"] == report["completed_pairs"] == 16256
        assert report["completed_configurations"] == [{"position": 0.0}, {"position": 1.0}]
        assert report["coverage"] == "discrete_only"
        assert report["errors"] == []
    else:
        assert 0 < report["completed_pairs"] < report["total_pairs"]
        assert report["coverage"] == "incomplete"
        assert report["completed_configurations"] == []
        assert any("budget exhausted" in error for error in report["errors"])
