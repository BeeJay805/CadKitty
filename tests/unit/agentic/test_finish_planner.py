"""The finish PLANNER pipeline: catalog generation (whole, from the registry), the fit-
unchanged ship gate, and the fail-closed planner call. The value guarantee is that finish
ships ONLY if it did not worsen the assembly fit."""

from __future__ import annotations

from types import SimpleNamespace

from forge_mind.agentic.detail import op_names
from forge_mind.agentic.finish_planner import (
    PartFinishContext,
    fit_not_worse,
    generate_catalog,
    plan_part_finish,
)


def test_catalog_lists_every_registered_op_with_params() -> None:
    cat = generate_catalog()
    for name in op_names():
        assert name in cat, f"{name} missing from catalog"
    # a couple of param names + a kinds annotation must show
    assert "surface_texture" in cat and "z_top" in cat
    assert "kinds:" in cat  # kinds parsed from docstrings


def test_fit_gate_ships_only_when_not_worse() -> None:
    clean = {"interference_mm3": 0.0, "seated": True}
    # unchanged -> ship
    assert fit_not_worse(clean, {"interference_mm3": 0.0, "seated": True})
    # tiny noise within tolerance -> ship
    assert fit_not_worse(clean, {"interference_mm3": 0.5, "seated": True})
    # new interference -> revert
    assert not fit_not_worse(clean, {"interference_mm3": 40.0, "seated": True})
    # unseated a mate -> revert
    assert not fit_not_worse(clean, {"interference_mm3": 0.0, "seated": False})
    # no finished fit (assembly failed) -> revert, fail-closed
    assert not fit_not_worse(clean, None)


def test_planner_call_failure_returns_empty() -> None:
    class _Boom:
        def generate_structured(self, **kw):
            raise RuntimeError("planner down")

    ctx = PartFinishContext(
        name="step",
        role="a ladder tread",
        bbox=((-20, -10, 0), (20, 10, 8)),
        mating_exclusions=["port 'pin' at (0,0,0)"],
    )
    assert plan_part_finish(ctx, client=_Boom()) == []


def test_planner_parses_directives_and_drops_malformed() -> None:
    directives = [
        SimpleNamespace(
            op="surface_texture",
            params_json='{"x0":-15,"x1":15,"y0":-8,"y1":8,"z_top":8,"kind":"diamond"}',
        ),
        SimpleNamespace(op="emboss_text", params_json="not json"),  # dropped
    ]

    class _Client:
        def generate_structured(self, **kw):
            return SimpleNamespace(directives=directives), SimpleNamespace(cost_usd=0.0)

    ctx = PartFinishContext(
        name="p", role="r", bbox=((-15, -8, 0), (15, 8, 8)), mating_exclusions=[]
    )
    ops = plan_part_finish(ctx, client=_Client())
    assert len(ops) == 1
    assert ops[0].op == "surface_texture" and ops[0].params["kind"] == "diamond"
