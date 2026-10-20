"""The accept-side consistency gate: pinned to the EXACT L78 desk_grommet false-PASS.

The incident, verbatim from `benchmarks/2026-07-22_doctrine/D1/desk_grommet_s0` (results
row, accepting iteration i3): the critic scored 4 / resemblance 'weak' while its
prioritized fixes demanded the brief-declared outer ring, cover flap, hinge slot,
alignment recess, and rib texture -- on an artifact that was a SOLID DISC (33,915 mm^3 =
99.96% of the solid 60x12 cylinder). Every declared feature was unmeasurable (volumes
'60'/'52'/'10'/'2 x 10'/'5 x 3'/'4 x 1.5' parse to None), so no instrument could refute
the demands; the gate must refuse that accept. The clean contrast case is pipe_flange's
accepting critique from the SAME arm: empty fix list -> accept proceeds.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic.accept_consistency import accept_consistency_block

# The desk_grommet brief's feature roster, verbatim (name, approx_dims_mm, method).
_GROMMET_FEATURES = [
    SimpleNamespace(name="outer_ring_outer_diameter", approx_dims_mm="60", method="revolve"),
    SimpleNamespace(name="outer_ring_inner_diameter", approx_dims_mm="52", method="pocket"),
    SimpleNamespace(name="outer_ring_thickness", approx_dims_mm="10", method="box"),
    SimpleNamespace(name="cover_flap_outer_diameter", approx_dims_mm="52", method="revolve"),
    SimpleNamespace(name="cover_flap_thickness", approx_dims_mm="3", method="box"),
    SimpleNamespace(name="hinge_slot", approx_dims_mm="2 x 10", method="pocket"),
    SimpleNamespace(name="alignment_recess", approx_dims_mm="5 x 3", method="pocket"),
    SimpleNamespace(name="inner_rib_texture", approx_dims_mm="4 x 1.5", method="loft"),
]
_GROMMET_BRIEF = SimpleNamespace(features=_GROMMET_FEATURES)

# The accepting critique's fixes, verbatim from the incident row.
_GROMMET_FIXES = [
    "Add a distinct outer ring with an outer diameter of 60 mm and an inner diameter of "
    "52 mm to create the grommet's mounting interface.",
    "Create a separate cover flap with an outer diameter of 52 mm and thickness of 3 mm, "
    "ensuring it pivots on a hinge slot (2 x 10 mm) along the edge of the outer ring.",
    "Include an alignment recess (5 x 3 mm) in the outer ring to allow the cover flap tab "
    "to align properly when closed.",
    "Add vertical rib texture (4 x 1.5 mm) along the inner edge of the outer ring for grip.",
]


def _critique(fixes: list[str], resemblance: str = "weak") -> SimpleNamespace:
    return SimpleNamespace(prioritized_fixes=fixes, resemblance=resemblance)


@pytest.fixture(autouse=True)
def _flag(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_ACCEPT_CONSISTENCY", "1")


def test_the_l78_false_pass_is_refused() -> None:
    reason = accept_consistency_block(_critique(_GROMMET_FIXES), _GROMMET_BRIEF)
    assert reason is not None
    assert "cover_flap" in reason or "outer_ring" in reason


def test_a_single_named_unmeasured_demand_refuses() -> None:
    reason = accept_consistency_block(_critique([_GROMMET_FIXES[1]]), _GROMMET_BRIEF)
    assert reason is not None


def test_empty_fixes_accept_proceeds() -> None:
    # pipe_flange's accepting critique (same arm): no fixes -> nothing self-contradictory.
    assert accept_consistency_block(_critique([]), _GROMMET_BRIEF) is None


def test_measured_through_hole_demands_do_not_refuse() -> None:
    # The gen-6/L66 phantom class: add-demands naming hole/bore features. Those counts fold
    # into `passed` (features_missing) at any accept site -- measured, so never a block.
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="bolt_hole_1", approx_dims_mm="d6", method="bolt_holes")]
    )
    fix = "Add the four bolt holes on the 48 mm bolt circle as specified."
    assert accept_consistency_block(_critique([fix]), brief) is None


def test_measured_additive_volume_demands_do_not_refuse() -> None:
    # A declared boss whose volume PARSES participated in the measured expectation -- an
    # add-demand naming it is the phantom class, not a standing claim.
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="mounting_boss", approx_dims_mm="10 x 10 x 6", method="box")]
    )
    fix = "Add the mounting boss at the corner as the brief requires."
    assert accept_consistency_block(_critique([fix]), brief) is None


def test_non_add_fixes_never_refuse() -> None:
    # Adjustment/aesthetic phrasing is not an ADD/CREATE demand -- unrefutable text classes
    # stay out of scope (L72), and only demand-shaped fixes can contradict an approval.
    fix = "Adjust the overall proportions of the cover flap for a more balanced look."
    assert accept_consistency_block(_critique([fix]), _GROMMET_BRIEF) is None


def test_generic_dimension_words_do_not_name_features() -> None:
    # "diameter"/"thickness" alone must not name-match *_diameter/*_thickness features.
    fix = "Add a chamfer so the diameter transition prints cleanly."
    assert accept_consistency_block(_critique([fix]), _GROMMET_BRIEF) is None


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_ACCEPT_CONSISTENCY", raising=False)
    assert accept_consistency_block(_critique(_GROMMET_FIXES), _GROMMET_BRIEF) is None


def test_strong_resemblance_chamfer_polish_is_not_blocked() -> None:
    # The calibration's false-FAIL class, verbatim (adapter_plate, 4 arms): an Add-chamfer
    # polish fix on a STRONG-resemblance accept must never block -- conjunct (2).
    brief = SimpleNamespace(
        features=[SimpleNamespace(name="plate_body", approx_dims_mm="60 x 60", method="box")]
    )
    fix = "Add a 1mm chamfer to all edges of the plate for improved safety and reduced stress concentration."
    assert accept_consistency_block(_critique([fix], resemblance="strong"), brief) is None


def test_partial_resemblance_gen7_manifold_is_not_blocked() -> None:
    # The L76 first ship (founder-eyeballed REAL) accepted at resemblance 'partial' with
    # bore/collector demands in the surviving veto text -- must never block.
    brief = SimpleNamespace(
        features=[
            SimpleNamespace(name="merge_collector", approx_dims_mm="40", method="sweep"),
            SimpleNamespace(name="exhaust_port_bore_1", approx_dims_mm="d19", method="bore"),
        ]
    )
    fixes = [
        "Create through-bores for the exhaust ports at the head flange; they should be exactly 19 mm diameter.",
        "Refine the merge collector section to smoothly transition from four runner ends into one 40 mm diameter collector.",
    ]
    assert accept_consistency_block(_critique(fixes, resemblance="partial"), brief) is None


def test_blocks_regardless_of_looks_right() -> None:
    # The A1 escape (two-door law): the gate refused the i3 accept, the run capped at
    # iteration_cap, and the FINAL verdict read the kept-best's looks_right=True critique
    # -- the disc shipped anyway. The block must not care about looks_right, so the final
    # critic_approved conjunct (loop.py LoopResult assembly) refuses the same critique.
    crit = SimpleNamespace(
        prioritized_fixes=list(_GROMMET_FIXES), resemblance="weak", looks_right=True
    )
    assert accept_consistency_block(crit, _GROMMET_BRIEF) is not None


def test_missing_resemblance_fails_closed() -> None:
    # An accepting critique that never claimed fidelity cannot contradict its own
    # missing-feature findings -- treated as bottom-of-ordinal.
    crit = SimpleNamespace(prioritized_fixes=[_GROMMET_FIXES[0]], resemblance=None)
    assert accept_consistency_block(crit, _GROMMET_BRIEF) is not None
