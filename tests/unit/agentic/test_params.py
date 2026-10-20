"""The parametric contract (ADR-0073 Stage 1): AST-extract PARAMS, splice overrides byte-exactly,
floats optimizable / ints frozen, fail-soft on anything non-literal.
"""

from __future__ import annotations

from forge_mind.agentic.params import default_bounds, extract_params, override_params

CODE = """PARAMS = {"journal_d": 55.0, "web_t": 18.5, "n_webs": 5, "note": "x"}
from math import pi
result = Cylinder(radius=PARAMS["journal_d"] / 2, height=PARAMS["web_t"])
"""


def test_extract_finds_values_and_kinds() -> None:
    spec = extract_params(CODE)
    assert spec is not None
    assert spec.values == {"journal_d": 55.0, "web_t": 18.5, "n_webs": 5}  # "note" ignored
    assert spec.float_names == ["journal_d", "web_t"]  # ints are frozen structure


def test_extract_handles_negative_and_multiline() -> None:
    code = 'x = 1\nPARAMS = {\n    "off": -12.5,\n    "count": 4,\n}\nresult = None\n'
    spec = extract_params(code)
    assert spec is not None and spec.values == {"off": -12.5, "count": 4}


def test_extract_fail_soft() -> None:
    assert extract_params("result = Box(1,1,1)") is None  # no PARAMS
    assert extract_params("PARAMS = make()") is None  # not a literal dict
    assert extract_params('PARAMS = {"a": 1 + 2}') is None  # computed value -> not optimizable
    assert extract_params("def f(:") is None  # syntax error
    assert extract_params('PARAMS = {"s": "only strings"}') is None  # no numeric entries


def test_override_splices_only_the_dict() -> None:
    spec = extract_params(CODE)
    assert spec is not None
    out = override_params(CODE, spec, {"journal_d": 60.0, "n_webs": 9.0, "junk": 1.0})
    spec2 = extract_params(out)
    assert spec2 is not None
    assert spec2.values["journal_d"] == 60.0  # float overridden
    assert spec2.values["web_t"] == 18.5  # unmentioned kept
    assert spec2.values["n_webs"] == 5  # int FROZEN even when named
    # everything outside the dict literal is byte-identical
    assert out[spec2.end :] == CODE[spec.end :]
    assert out[: spec.start] == CODE[: spec.start]


def test_override_roundtrip_is_runnable() -> None:
    spec = extract_params(CODE)
    assert spec is not None
    out = override_params(CODE, spec, {"web_t": 22.0})
    ns: dict = {"Cylinder": lambda radius, height: (radius, height)}
    exec(out, ns)
    assert ns["result"] == (27.5, 22.0)  # journal_d kept 55.0, web_t overridden


def test_default_bounds_spread_and_positivity() -> None:
    spec = extract_params(CODE)
    assert spec is not None
    b = default_bounds(spec, spread=0.4)
    assert set(b) == {"journal_d", "web_t"}
    lo, hi = b["journal_d"]
    assert abs(lo - 33.0) < 1e-9 and abs(hi - 77.0) < 1e-9
    neg = extract_params('PARAMS = {"off": -10.0}')
    assert neg is not None
    lo, hi = default_bounds(neg)["off"]
    assert lo < hi  # ordered even for negative authored values


# --- stated-dimension pinning (ADR-0080) ---------------------------------------------------------

INCIDENT_CODE = (
    'PARAMS = {"module": 2.0, "bore_d": 12.0, "boss_od": 22.0, "hole_d": 8.0, "web_t": 5.7}\n'
    "result = None\n"
)
INCIDENT_STATED = frozenset({76.0, 12.0, 22.0, 8.0})


def test_pinned_bounds_lock_the_timing_gear_incident() -> None:
    """ADR-0080 headline regression: the measured polish drifted bore 12->12.5, holes 8->6.76,
    boss 22->18.8. Stated-pinned bounds put every one of those values OUTSIDE the search box --
    the ES cannot reproduce the incident."""
    spec = extract_params(INCIDENT_CODE)
    assert spec is not None
    b = default_bounds(spec, stated=INCIDENT_STATED)
    assert not (b["bore_d"][0] <= 12.5 <= b["bore_d"][1])
    assert not (b["hole_d"][0] <= 6.76 <= b["hole_d"][1])
    assert not (b["boss_od"][0] <= 18.8 <= b["boss_od"][1])
    # the free dim keeps its full repair range
    lo, hi = b["web_t"]
    assert abs(lo - 5.7 * 0.6) < 1e-9 and abs(hi - 5.7 * 1.4) < 1e-9
    # DOCUMENTED derived-dim gap: module 2.0 matches no stated number (the brief states the TIP,
    # 76 = m*(z+2)), so 2.43 stays reachable by the pins -- the envelope term + the acceptance
    # audit own that channel, not the bounds.
    assert b["module"][0] <= 2.43 <= b["module"][1]


def test_pinned_bounds_radius_idiom_and_window() -> None:
    # radius idiom: a stated Ø12 bore authored as radius 6.0 is pinned.
    spec = extract_params('PARAMS = {"bore_r": 6.0}\nresult = None\n')
    assert spec is not None
    lo, hi = default_bounds(spec, stated=frozenset({12.0}))["bore_r"]
    assert lo > 6.0 * 0.97 and hi < 6.0 * 1.03
    # tolerance window: author wrote 76.2 for the stated 76 -- still pinned (1% window).
    spec2 = extract_params('PARAMS = {"tip_d": 76.2}\nresult = None\n')
    assert spec2 is not None
    lo2, hi2 = default_bounds(spec2, stated=frozenset({76.0}))["tip_d"]
    assert hi2 - lo2 < 76.2 * 0.05  # pinned, not the ±40% box
    # small stated values keep a usable minimum half-width (0.1mm), never a zero-width box.
    spec3 = extract_params('PARAMS = {"lip": 1.0}\nresult = None\n')
    assert spec3 is not None
    lo3, hi3 = default_bounds(spec3, stated=frozenset({1.0}))["lip"]
    assert abs(lo3 - 0.9) < 1e-9 and abs(hi3 - 1.1) < 1e-9


def test_pinned_bounds_are_always_inside_default_bounds() -> None:
    spec = extract_params(INCIDENT_CODE)
    assert spec is not None
    free = default_bounds(spec)
    pinned = default_bounds(spec, stated=INCIDENT_STATED)
    for name in spec.float_names:
        assert free[name][0] - 1e-9 <= pinned[name][0]
        assert pinned[name][1] <= free[name][1] + 1e-9


def test_no_stated_set_means_byte_identical_bounds() -> None:
    spec = extract_params(INCIDENT_CODE)
    assert spec is not None
    assert default_bounds(spec) == default_bounds(spec, stated=None)
    assert default_bounds(spec) == default_bounds(spec, stated=frozenset())


def test_lift_float_literals_roundtrip() -> None:
    """ADR-0117 follow-on: the lift synthesizes a PARAMS dict from call-arg float literals
    (the polish's delivery WITHOUT the author-prompt contract note arm PO measured out).
    Must be execution-equivalent and extractable by the existing machinery."""
    from forge_mind.agentic.params import extract_params, lift_float_literals

    code = "Cylinder(radius=20.0, height=14.0)\nfillet(edges, radius=0.5)\n"
    lifted = lift_float_literals(code)
    assert lifted is not None and lifted.startswith("PARAMS = {")
    spec = extract_params(lifted)
    assert spec is not None and spec.float_names == ["p1", "p2", "p3"]
    assert 'Cylinder(radius=PARAMS["p1"], height=PARAMS["p2"])' in lifted


def test_lift_respects_authored_params_and_bounds() -> None:
    from forge_mind.agentic.params import lift_float_literals

    # an authored PARAMS dict wins -- never double-lift
    assert lift_float_literals('PARAMS = {"a": 1.0}\nBox(PARAMS["a"], 2.0, 3.0)') is None
    # ints are frozen structure, never lifted
    assert lift_float_literals("Box(10, 20, 30)") is None
    # unparseable code fails soft
    assert lift_float_literals("def broken(:") is None
    # over the ambiguity cap fails soft
    many = "\n".join(f"Box({i}.5, 1, 1)" for i in range(30))
    assert lift_float_literals(many) is None


def test_lift_negative_floats() -> None:
    from forge_mind.agentic.params import extract_params, lift_float_literals

    lifted = lift_float_literals("extrude(amount=-5.5)")
    assert lifted is not None
    spec = extract_params(lifted)
    assert spec is not None and spec.values["p1"] == -5.5


def test_lift_int_dims_under_keyword_allowlist() -> None:
    """Measured 2026-07-21: local authors write INTEGER dims (0 float / 15 int call-arg
    literals in archived strong-reds). Ints lift ONLY under dimension keywords; counts and
    positional ints stay frozen."""
    from forge_mind.agentic.params import extract_params, lift_float_literals

    code = "Cylinder(radius=20, height=14)\nwith GridLocations(31, 31, 2, 2):\n    pass\n"
    lifted = lift_float_literals(code)
    assert lifted is not None
    spec = extract_params(lifted)
    assert spec is not None and spec.values == {"p1": 20.0, "p2": 14.0}
    assert "GridLocations(31, 31, 2, 2)" in lifted  # counts untouched
