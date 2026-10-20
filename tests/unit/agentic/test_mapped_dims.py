"""Brief->feature POSITION mapping (ADR-0012 E1): the off_footprint_bore owning check.

The reconcile that produced this (2026-07-07) is pinned in the tests: a position defect exists
ONLY when the user stated the position (model-invented holes are a design DOF, not a defect) and
is CATCHABLE only on measurable bores (slots/blades stay fail-open). Value-only anchoring is noisy,
so a number qualifies only from a position phrase on a hole feature.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic.mapped_dims import (
    _measured_pcd,
    _pcd_candidates,
    mapped_position_mismatch,
    stated_bore_positions,
)

pytestmark = pytest.mark.slow


def _brief(features: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(features=[SimpleNamespace(**f) for f in features])


def test_parses_stated_bore_height_on_a_hole_feature() -> None:
    brief = _brief(
        [
            {
                "name": "bearing bore",
                "approx_dims_mm": "22 mm diameter, 30 mm deep, center 35 mm above the base",
                "standard": None,
            }
        ]
    )
    specs = stated_bore_positions(brief, "", frozenset({22.0, 35.0, 30.0}))
    assert len(specs) == 1
    assert specs[0].kind == "height" and specs[0].value_mm == 35.0
    assert specs[0].bore_d_mm == 22.0  # the "22 mm diameter", not the "30 mm deep"


def test_parses_pitch_and_pcd_only_when_user_anchored() -> None:
    brief = _brief(
        [
            {
                "name": "mount holes",
                "approx_dims_mm": "M3 clearance, 76.2 mm center-to-center",
                "standard": "M3",
            }
        ]
    )
    # 76.2 is user-stated -> a spec
    specs = stated_bore_positions(brief, "", frozenset({76.2}))
    assert [(s.kind, s.value_mm) for s in specs] == [("pitch", 76.2)]
    # 76.2 NOT in the stated set (model-chosen) -> nothing (the anchor invariant)
    assert stated_bore_positions(brief, "", frozenset({30.0})) == []
    # PCD phrasing
    pcd_brief = _brief(
        [
            {
                "name": "bolt holes",
                "approx_dims_mm": "6 holes on a 50 mm bolt circle",
                "standard": None,
            }
        ]
    )
    specs = stated_bore_positions(pcd_brief, "", frozenset({50.0, 6.0}))
    assert [(s.kind, s.value_mm) for s in specs] == [("pcd", 50.0)]


def test_non_bore_and_model_invented_positions_do_not_qualify() -> None:
    # a SLOT feature that states a pitch -> not a hole feature, no spec (sd_card class)
    slot = _brief(
        [{"name": "SD card slots", "approx_dims_mm": "slot-to-slot pitch 14 mm", "standard": None}]
    )
    assert stated_bore_positions(slot, "", frozenset({14.0})) == []
    # a hole feature whose pitch is NOT in the stated set -> model-invented, no spec
    inv = _brief(
        [{"name": "mounting holes", "approx_dims_mm": "4 holes, 42 mm pitch", "standard": None}]
    )
    assert stated_bore_positions(inv, "", frozenset({20.0, 60.0})) == []


def test_measured_pcd_of_a_bolt_circle() -> None:
    import numpy as np

    # four bores on a 50 mm PCD (radius 25) about the origin
    bores = [
        SimpleNamespace(axis_point=(25.0, 0.0, 0.0)),
        SimpleNamespace(axis_point=(-25.0, 0.0, 0.0)),
        SimpleNamespace(axis_point=(0.0, 25.0, 0.0)),
        SimpleNamespace(axis_point=(0.0, -25.0, 0.0)),
    ]
    assert _measured_pcd(bores, np) == pytest.approx(50.0)
    assert _measured_pcd(bores[:2], np) is None  # < 3 centers: not a circle


def test_mismatch_fails_open_without_a_spec_or_bores() -> None:
    brief = _brief([{"name": "body", "approx_dims_mm": "60 x 40 x 5 mm", "standard": None}])
    # no hole feature -> no spec -> fail open regardless of geometry
    assert mapped_position_mismatch(object(), None, brief, "", frozenset({60.0})) == (False, None)


def _ring(n: int, r: float, d: float, z: float = 0.0) -> list:
    import math

    return [
        SimpleNamespace(
            radius=d / 2.0,
            axis_point=(r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), z),
        )
        for i in range(n)
    ]


def test_pcd_candidates_exclude_the_central_bore() -> None:
    # The row-50 hard_pipe_flange false-FAIL geometry: 6x 5.5mm bolts at r=35 (BCD 70)
    # + one 32mm central through-bore. The old mean read 61; the ring must read 70.
    import numpy as np

    bores = [*_ring(6, 35.0, 5.5), SimpleNamespace(radius=16.0, axis_point=(0.0, 0.0, 0.0))]
    cands = _pcd_candidates(bores, np)
    assert cands == pytest.approx([70.0])


def test_pcd_candidates_drop_a_same_diameter_center_hole() -> None:
    import numpy as np

    bores = [*_ring(4, 35.0, 5.5), SimpleNamespace(radius=2.75, axis_point=(0.0, 0.0, 0.0))]
    assert _pcd_candidates(bores, np) == pytest.approx([70.0])


def test_pcd_candidates_yield_one_per_ring_for_multi_ring_parts() -> None:
    import numpy as np

    bores = _ring(4, 20.0, 4.4) + _ring(6, 35.0, 5.4)
    cands = sorted(_pcd_candidates(bores, np))
    assert cands == pytest.approx([40.0, 70.0])


def test_pcd_candidates_fail_open_below_three_ring_members() -> None:
    import numpy as np

    assert _pcd_candidates(_ring(2, 25.0, 5.0), np) == []


# ---- the L66 manifold false-FAIL (2026-07-21, two independent author reproductions) ----


def test_pcd_candidates_split_coplanar_rings_before_fitting() -> None:
    """8 M6 head-flange holes (z=0) + 4 M6 outlet holes (true BCD 48, z=174) share one
    diameter, and the pooled fit read 68 while both authors had CODED 48. Same-size hole
    groups on different planes are different patterns -- the 48 ring must survive."""
    import numpy as np

    head = [
        SimpleNamespace(radius=3.25, axis_point=(x, y, 0.0))
        for x in (-65.0, -31.0, 31.0, 65.0)
        for y in (-12.5, 12.5)
    ]
    outlet = _ring(4, 24.0, 6.5, z=174.0)
    cands = _pcd_candidates([*head, *outlet], np)
    assert any(abs(c - 48.0) <= 0.5 for c in cands)


def _junction_brief():
    return _brief(
        [
            {
                "name": "outlet-flange bolt holes",
                "approx_dims_mm": "M6 clearance holes, 4 equally spaced on a 48 mm bolt circle",
                "standard": "M6",
            }
        ]
    )


def _patch_stacks(monkeypatch, stacks) -> None:
    import forge_mind.validator.geometry as vg

    for s in stacks:
        s.concave = True
    monkeypatch.setattr(vg, "extract_cylinders", lambda part: stacks)
    monkeypatch.setattr(vg, "cluster_coaxial", lambda cyls: cyls)


_MESH = SimpleNamespace(bounds=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])


def test_pcd_offense_silent_when_no_recognizable_ring(monkeypatch) -> None:
    """Only the head flange's holes detected (the outlet ring missed): nearest fit ~68 is
    NOT the stated-48 ring misplaced -- silence, absence is the presence checks' job."""
    monkeypatch.delenv("FORGE_MAPPED_LEGACY", raising=False)
    head = [
        SimpleNamespace(radius=3.25, axis_point=(x, y, 0.0))
        for x in (-65.0, -31.0, 31.0, 65.0)
        for y in (-12.5, 12.5)
    ]
    _patch_stacks(monkeypatch, head)
    fired, _msg = mapped_position_mismatch(
        object(), _MESH, _junction_brief(), "", frozenset({48.0})
    )
    assert fired is False


def test_pcd_offense_fires_on_a_misplaced_recognizable_ring(monkeypatch) -> None:
    monkeypatch.delenv("FORGE_MAPPED_LEGACY", raising=False)
    _patch_stacks(monkeypatch, _ring(4, 27.0, 6.5))  # built 54 vs stated 48: off 6 <= 3x band
    fired, msg = mapped_position_mismatch(object(), _MESH, _junction_brief(), "", frozenset({48.0}))
    assert fired is True and msg is not None and "54.0" in msg


def test_mapped_legacy_env_restores_the_pooled_behavior(monkeypatch) -> None:
    """The instant-revert escape hatch: legacy mode keeps the pre-fix pooled fit and the
    unceilinged offense."""
    monkeypatch.setenv("FORGE_MAPPED_LEGACY", "1")
    head = [
        SimpleNamespace(radius=3.25, axis_point=(x, y, 0.0))
        for x in (-65.0, -31.0, 31.0, 65.0)
        for y in (-12.5, 12.5)
    ]
    _patch_stacks(monkeypatch, head + _ring(4, 24.0, 6.5, z=174.0))
    fired, msg = mapped_position_mismatch(object(), _MESH, _junction_brief(), "", frozenset({48.0}))
    assert fired is True and msg is not None  # the old pooled ~68 accusation returns
