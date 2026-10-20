"""Unit tests for the ADR-0083 defect-injection operators (pure code transforms).

Geometry-free: these pin the code-level behavior (site selection, splice fidelity,
precision guards). Whether an injected defect materializes in the MESH is the
confirmers' job and is covered by the audit run itself.
"""

from __future__ import annotations

import ast

from forge_mind.agentic.params import extract_params
from forge_mind.audit.mutations import (
    blind_hole_mutants,
    dropped_feature_mutants,
    envelope_mutants,
    off_footprint_mutants,
    stated_dim_mutants,
    subfloor_wall_mutants,
    undersized_clearance_bore_mutants,
)

FIXTURE = """from build123d import *

PARAMS = {"plate_len": 80.0, "plate_wid": 50.0, "wall_t": 3.0,
          "bolt_circle_r": 20.0, "n_bolts": 4, "shaft_bore_d": 8.0}

with BuildPart() as part:
    Box(PARAMS["plate_len"], PARAMS["plate_wid"], PARAMS["wall_t"])
    with Locations((0, 0, 0)):
        Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["wall_t"] + 2)
    with PolarLocations(PARAMS["bolt_circle_r"], PARAMS["n_bolts"]):
        Hole(radius=2.2, depth=PARAMS["wall_t"] + 2)
"""


def _params_of(code: str) -> dict[str, float | int]:
    spec = extract_params(code)
    assert spec is not None
    return spec.values


def test_stated_dim_moves_only_the_pinned_param() -> None:
    out = stated_dim_mutants(FIXTURE, frozenset({8.0}))
    assert len(out) == 2  # x1.12 and x0.85 of shaft_bore_d
    for m in out:
        assert m.defect_class == "off_brief_stated_dim"
        vals = _params_of(m.code)
        assert vals["shaft_bore_d"] != 8.0
        # everything else byte-preserved through the splice
        assert vals["plate_len"] == 80.0 and vals["n_bolts"] == 4
        ast.parse(m.code)


def test_stated_dim_no_stated_numbers_yields_nothing() -> None:
    assert stated_dim_mutants(FIXTURE, frozenset()) == []


def test_clearance_bore_mutant_stays_inside_assignment_window() -> None:
    out = undersized_clearance_bore_mutants(FIXTURE, [8.0])
    assert len(out) == 1
    assert out[0].defect_class == "undersized_clearance_bore"
    assert _params_of(out[0].code)["shaft_bore_d"] == 7.5


def test_subfloor_wall_thins_the_wall_param_below_floor() -> None:
    out = subfloor_wall_mutants(FIXTURE, 2.0)
    assert [m.meta["param"] for m in out] == ["wall_t"]
    vals = _params_of(out[0].code)
    assert vals["wall_t"] == 1.1  # 0.55 * declared 2.0
    assert out[0].defect_class == "sub_floor_wall"


def test_subfloor_wall_ignores_wall_scaled_but_unrelated_params() -> None:
    # plate_len carries no wall token; wall_t at 30mm is outside the plausible window
    code = FIXTURE.replace('"wall_t": 3.0', '"wall_t": 30.0')
    assert subfloor_wall_mutants(code, 2.0) == []


def test_dropped_feature_zeroes_matching_count_param() -> None:
    out = dropped_feature_mutants(FIXTURE, ["bolt holes"])
    zeroed = [m for m in out if "PARAMS" in m.description]
    assert len(zeroed) == 1
    assert _params_of(zeroed[0].code)["n_bolts"] == 0


def test_dropped_feature_deletes_the_labeled_block() -> None:
    out = dropped_feature_mutants(FIXTURE, ["bolt holes"])
    deleted = [m for m in out if "deleted" in m.description]
    assert len(deleted) == 1
    code = deleted[0].code
    ast.parse(code)
    assert "PolarLocations" not in code  # the bolt-circle block is gone
    assert "shaft_bore_d" in code  # the unrelated bore block survives


def test_dropped_feature_unknown_label_yields_nothing() -> None:
    assert dropped_feature_mutants(FIXTURE, ["cooling fins"]) == []


def test_deletion_is_safe_guard() -> None:
    from forge_mind.audit.mutations import _deletion_is_safe

    # a binding whose name is LOADed outside the deleted set -> unsafe (would NameError)
    t1 = ast.parse("a = f()\nb = a + 1\n")
    assert not _deletion_is_safe(t1, [t1.body[0]])
    # a pure side-effect statement binds nothing -> always safe
    t2 = ast.parse("g()\nh()\n")
    assert _deletion_is_safe(t2, [t2.body[0]])
    # the binding AND its only use are both deleted -> safe (nothing dangles)
    t3 = ast.parse("a = f()\nuse(a)\nother()\n")
    assert _deletion_is_safe(t3, [t3.body[0], t3.body[1]])


def test_blind_hole_halves_depth_on_the_labeled_line() -> None:
    out = blind_hole_mutants(FIXTURE, ["shaft bore"])
    assert len(out) == 1
    m = out[0]
    assert m.defect_class == "blind_where_through"
    ast.parse(m.code)
    assert '(PARAMS["wall_t"] + 2)*0.5' in m.code
    # the bolt-circle Hole line is untouched (label scoped the edit)
    assert m.code.count("*0.5") == 1


def test_blind_hole_with_hint_uses_shallow_depth_not_halving() -> None:
    # v2.1: halving over-specified cuts still pierces; WITH a hint the depth becomes the shallow
    # geometry-aware value (reliably blind), not (expr)*0.5
    out = blind_hole_mutants(FIXTURE, ["shaft bore"], depth_hint_mm=2.7)
    assert len(out) == 1
    m = out[0]
    ast.parse(m.code)
    assert "depth=2.700" in m.code
    assert "*0.5" not in m.code  # the hint path replaces, does not halve


def test_blind_hole_hint_preserves_extrude_sign() -> None:
    # v2.1: extrude amount is signed; the hint shrinks the magnitude but keeps the cut direction
    code = FIXTURE.replace(
        'Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["wall_t"] + 2)',
        "extrude(shaft_bore_profile, amount=-6.0, mode=Mode.SUBTRACT)",
    )
    out = blind_hole_mutants(code, ["shaft bore"], depth_hint_mm=2.7)
    assert len(out) == 1
    ast.parse(out[0].code)
    # sign-preserving form: (2.700) * (1.0 if (-6.0) >= 0 else -1.0) == -2.7 (same sign, smaller)
    assert "amount=(2.700) * (1.0 if (-6.0) >= 0 else -1.0)" in out[0].code
    ns: dict[str, object] = {}
    exec("v = (2.700) * (1.0 if (-6.0) >= 0 else -1.0)", ns)
    assert ns["v"] == -2.7  # negative kept, magnitude shrunk from 6.0


def test_blind_hole_bare_hole_gains_injected_depth() -> None:
    code = FIXTURE.replace(
        'Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["wall_t"] + 2)',
        'Hole(radius=PARAMS["shaft_bore_d"] / 2)',
    )
    out = blind_hole_mutants(code, ["shaft bore"], depth_hint_mm=1.35)
    assert len(out) == 1
    ast.parse(out[0].code)
    assert 'Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=1.350)' in out[0].code


def test_blind_hole_bare_hole_without_hint_is_skipped() -> None:
    code = FIXTURE.replace(
        'Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["wall_t"] + 2)',
        'Hole(radius=PARAMS["shaft_bore_d"] / 2)',
    )
    assert blind_hole_mutants(code, ["shaft bore"]) == []


def test_blind_hole_halves_extrude_amount() -> None:
    code = FIXTURE.replace(
        'Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["wall_t"] + 2)',
        "extrude(shaft_bore_profile, amount=-6.0, mode=Mode.SUBTRACT)",
    )
    out = blind_hole_mutants(code, ["shaft bore"])
    assert len(out) == 1
    assert "amount=(-6.0)*0.5" in out[0].code


def test_blind_hole_ambiguous_label_is_dropped() -> None:
    # "plate" lights up too many lines -> precision guard refuses to guess
    code = FIXTURE + "\n" + "\n".join(f"plate_ref_{i} = PARAMS['plate_len']" for i in range(5))
    assert blind_hole_mutants(code, ["plate"]) == []


def test_off_footprint_scales_position_param() -> None:
    out = off_footprint_mutants(FIXTURE)
    assert [m.meta["param"] for m in out] == ["bolt_circle_r"]
    assert _params_of(out[0].code)["bolt_circle_r"] == 26.0  # 20 * 1.3


def test_envelope_scales_the_dominant_dimension() -> None:
    out = envelope_mutants(FIXTURE)
    assert len(out) == 1
    m = out[0]
    assert m.meta["param"] == "plate_len"  # 80 is the largest envelope-token float
    assert _params_of(m.code)["plate_len"] == 112.0  # 80 * 1.4


def test_operators_fail_soft_on_paramless_code() -> None:
    code = "from build123d import *\nwith BuildPart() as p:\n    Box(10, 10, 10)\n"
    assert stated_dim_mutants(code, frozenset({10.0})) == []
    assert subfloor_wall_mutants(code, 2.0) == []
    assert off_footprint_mutants(code) == []
    assert envelope_mutants(code) == []


# ------------------------------------------------------- W4.3 site expansion ---


def test_subfloor_wall_matches_t_suffix_and_thk_and_plate() -> None:
    # `base_t` (suffix idiom), `lid_thk` (thk token), `plate_th`?? -- pin the three
    # W4.3 vocabularies: _t suffix, thk, plate. All wall-scaled for floor=2.0.
    code = FIXTURE.replace(
        '"wall_t": 3.0', '"base_t": 3.0, "lid_thk": 2.5, "plate_floor": 4.0'
    ).replace('PARAMS["wall_t"]', 'PARAMS["base_t"]')
    out = subfloor_wall_mutants(code, 2.0)
    hit = {m.meta["param"] for m in out}
    assert {"base_t", "lid_thk", "plate_floor"} <= hit
    assert all(m.meta["site"] == "wall-token" for m in out)


def test_subfloor_wall_thinnest_float_fallback_fires_without_wall_names() -> None:
    # No param reads as a wall; the thinnest in-scale float (s1=2.2) is the fallback
    # site; bore/radius params are excluded even though they are wall-scaled.
    code = """from build123d import *

PARAMS = {"body_len": 60.0, "s1": 2.2, "shaft_bore_d": 6.0, "corner_rad": 3.0}

with BuildPart() as part:
    Box(PARAMS["body_len"], 30.0, PARAMS["s1"])
    Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=PARAMS["s1"] + 2)
"""
    out = subfloor_wall_mutants(code, 2.0)
    assert [m.meta["param"] for m in out] == ["s1"]
    assert out[0].meta["site"] == "thinnest-float fallback"
    vals = _params_of(out[0].code)
    assert vals["s1"] == 1.1 and vals["shaft_bore_d"] == 6.0


def test_subfloor_wall_fallback_never_picks_hole_or_rounding_params() -> None:
    # ONLY hole/rounding params are in scale -> nothing to mutate, not a bad pick.
    code = """from build123d import *

PARAMS = {"body_len": 60.0, "shaft_bore_d": 6.0, "corner_rad": 3.0}

with BuildPart() as part:
    Box(PARAMS["body_len"], 30.0, 10.0)
    Hole(radius=PARAMS["shaft_bore_d"] / 2, depth=12.0)
"""
    assert subfloor_wall_mutants(code, 2.0) == []


def test_off_footprint_matches_expanded_position_tokens() -> None:
    code = FIXTURE.replace('"bolt_circle_r": 20.0', '"hole_dist": 20.0, "slot_sep": 15.0')
    out = off_footprint_mutants(code)
    hit = {m.meta["param"] for m in out}
    assert {"hole_dist", "slot_sep"} <= hit


def test_subfloor_wall_never_mutates_a_declaration_param() -> None:
    # Measured escape (2026-07-10 audit): mutating `wall_min` rewrites the SPEC the
    # validator enforces -- the part passes its own mutated contract. Declarations
    # (min/max/limit/tol) are never mutation sites; real floors/walls still are.
    code = FIXTURE.replace('"wall_t": 3.0', '"wall_min": 3.0, "tray_floor": 2.5')
    out = subfloor_wall_mutants(code, 2.0)
    hit = {m.meta["param"] for m in out}
    assert "wall_min" not in hit
    assert "tray_floor" in hit
