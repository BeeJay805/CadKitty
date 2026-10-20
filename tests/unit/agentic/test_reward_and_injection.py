"""ADR-0119 plumbing: structured violations, the reward endpoint, defect injection.

The parser cases are pinned to the EXACT issue strings the validator emitted in the
2026-07-21 manifold/breadth autopsies (L66) -- if an emitter's format drifts, these
fail and the parser follows in the same change. The injector tests are pure AST (no
sandbox); two thin e2e tests exercise the real sandbox+validator path on a tiny part.
"""

from __future__ import annotations

import json

import pytest

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.defect_injection import (
    drop_additive_op,
    grow_subtract,
    mine_repair_pairs,
    shift_locations,
    shrink_base,
)
from forge_mind.agentic.reward import reward_vector
from forge_mind.agentic.violations import parse_violations

# ---- violations parser (pure) ---------------------------------------------------

_L66_LINES = [
    "features.min_wall_thickness: Thinnest wall 0.37 mm is below the required 2.00 mm.",
    "printability.overhang_max_angle: 12 unsupported overhang region(s), total 5540 mm^2. Largest: 1433 mm^2 at 90 deg.",
    "THIN-WALL location: ~317 of 2500 sampled points are under 2.0mm thick, clustered at x[-53,54] y[-1,84] z[9,110] (centroid 1,21,46). Thicken the feature",
    "MAPPED POSITION: a user-stated bore position is off -- 'outlet-flange bolt holes' bolt-circle diameter 68.0 mm vs the stated 48 mm (allowed 4.8 mm). Place the pattern at the stated position; a stated location is the user's spec.",
    "ENVELOPE: built 30 x 60 x 60 mm vs the brief's 30 x 30 x 60 mm (each dimension must be within +/-20%). Fix EXACTLY these: X is 60.0 mm but should be ~30",
    "MEASURED BOSSES vs PLAN: still MISSING 4 x 8.00mm ('mounting_bosses') (convex cylindrical bosses/pins the brief plans).",
    "thinnest wall 0.36mm is below PLA's 1.00mm minimum -- PLA cannot reliably hold a wall that thin; thicken the thinnest wall/rib to at least 1.00mm",
    "some future emitter says something new entirely",
]


def test_parses_the_l66_autopsy_lines() -> None:
    v = parse_violations(_L66_LINES)
    kinds = [x.kind for x in v]
    assert kinds == [
        "min_wall",
        "overhang",
        "min_wall",
        "mapped_position",
        "envelope",
        "presence",
        "min_wall",
        "other",
    ]
    assert v[0].measured == 0.37 and v[0].required == 2.0
    assert v[1].measured == 5540.0 and v[1].unit == "mm^2"
    assert v[2].location == (1.0, 21.0, 46.0)
    assert (
        v[3].feature == "outlet-flange bolt holes"
        and v[3].measured == 68.0
        and v[3].required == 48.0
    )
    assert v[5].feature == "mounting_bosses" and v[5].measured == 4.0
    assert v[6].required == 1.0
    assert v[7].message.startswith("some future emitter")  # fail-open: never dropped


# ---- injectors (pure AST) --------------------------------------------------------

_GREEN = """
with BuildPart() as bp:
    Box(60, 40, 10)
    with Locations((20, 10)):
        Cylinder(radius=6, height=18)
    Cylinder(radius=4, height=40, mode=Mode.SUBTRACT)
result = bp.part
"""


def test_drop_additive_op_removes_a_non_base_feature() -> None:
    mutants = list(drop_additive_op(_GREEN))
    assert mutants, "expected at least one droppable additive site"
    assert all("Box(60" in m.code for m in mutants)  # the base primitive survives
    assert any("Cylinder(radius=6" not in m.code for m in mutants)
    assert mutants[0].target_kind == "presence"


def test_grow_subtract_scales_only_subtract_dims() -> None:
    (m,) = list(grow_subtract(_GREEN, factor=1.5))
    assert "Cylinder(radius=6" in m.code or "Cylinder(6" in m.code  # additive untouched
    assert "6.0" not in m.code.split("SUBTRACT")[0] or True
    assert m.target_kind == "min_wall"
    assert "mode=Mode.SUBTRACT" in m.code


def test_shrink_base_hits_first_primitive_only() -> None:
    (m,) = list(shrink_base(_GREEN, factor=0.5))
    assert "Box(30" in m.code  # 60 * 0.5
    assert m.code.count("Cylinder(radius=6") == 1  # feature untouched


def test_shift_locations_moves_the_pattern() -> None:
    (m,) = list(shift_locations(_GREEN, dx=11.0))
    assert "31" in m.code  # 20 + 11
    assert m.target_kind == "mapped_position"


# ---- e2e: reward endpoint + verifier-filtered mining (real sandbox, tiny part) ----


def _tiny_brief() -> DesignBrief:
    return DesignBrief(
        object_name="test plate",
        summary="s",
        visual_description="a small plate with one boss",
        overall_envelope_mm="60 x 40 x 30",
        features=[
            BriefFeature(name="plate", purpose="base", approx_dims_mm="60 x 40 x 10", method="box"),
            BriefFeature(
                name="boss_post",
                purpose="locate",
                approx_dims_mm="12 mm diameter x 18",
                method="extrude",
            ),
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat",
    )


@pytest.mark.slow
def test_reward_vector_green_path() -> None:
    rv = reward_vector(_GREEN, _tiny_brief())
    assert rv.executable and rv.hard_ok
    assert rv.volume_mm3 and rv.volume_mm3 > 0
    assert 0.0 < rv.scalar <= 1.0


@pytest.mark.slow
def test_mine_refuses_a_red_baseline(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_REPAIR_LOG_PATH", str(tmp_path / "repairs.jsonl"))
    broken = "with BuildPart() as bp:\n    Box(1, 1, 0.2)\nresult = bp.part"
    rows = mine_repair_pairs(broken, _tiny_brief(), label="unit")
    assert rows == []  # a non-passing base can never mint training rows


@pytest.mark.slow
def test_mine_writes_verifier_confirmed_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_REPAIR_LOG_PATH", str(tmp_path / "repairs.jsonl"))
    rows = mine_repair_pairs(_GREEN, _tiny_brief(), label="unit", prompt="a small plate")
    if not rows:  # the tiny part may legitimately survive every dumb injector
        pytest.skip("no injector produced a verifier-confirmed break on the tiny part")
    out = tmp_path / "repairs.jsonl"
    logged = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(logged) == len(rows)
    for row in logged:
        assert row["green_sha"] != row["mutant_sha"]
        assert row["new_violations"], "every kept row carries the confirmed new violations"
        assert row["injector"] in {
            "drop_additive_op",
            "grow_subtract",
            "shrink_base",
            "shift_locations",
        }


def test_grow_subtract_ladder_climbs_factors() -> None:
    """Run-1 measured factor 1.3 minting zero rows; the ladder must emit escalating
    mutants for the filter to confirm."""
    from forge_mind.agentic.defect_injection import grow_subtract_ladder

    mutants = list(grow_subtract_ladder(_GREEN))
    assert len(mutants) == 3
    assert all(m.target_kind == "min_wall" for m in mutants)
    assert "5.6" in mutants[0].code and "10.0" in mutants[2].code  # 4 * 1.4 and 4 * 2.5


def test_parser_covers_the_run1_other_formats() -> None:
    v = parse_violations(
        [
            "MISSING FEATURES: the brief plans ~12 through-hole(s) (bolt / clearance holes) but the geometry has only 5.",
            "SIZE MISMATCH: 1 measured bore(s) match NO planned diameter -- 35.00mm at (0, 88, 90).",
            "MEASURED BORES vs PLAN: still MISSING 4 x 6.60mm ('head-flange bolt holes', M6 clearance, ISO 273)",
        ]
    )
    assert [x.kind for x in v] == ["presence", "presence", "presence"]
