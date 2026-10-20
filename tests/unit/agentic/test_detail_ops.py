"""The designed-finish detail library (tons-of-breadth expansion): every op x kind must
produce a WATERTIGHT solid through the real kernel, and every op must FAIL CLOSED on bad
input (skipped, part kept). The LLM directs; this tested code makes the geometry."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from build123d import Box, Cylinder, export_stl

pytestmark = pytest.mark.slow

import trimesh  # noqa: E402

from forge_mind.agentic.detail import FinishOp, apply_finish, finish_enabled, op_names  # noqa: E402


def _watertight(part) -> bool:
    f = Path(tempfile.mktemp(suffix=".stl"))
    export_stl(part, str(f))
    try:
        return bool(trimesh.load(str(f), force="mesh").is_watertight)
    finally:
        f.unlink(missing_ok=True)


# (op, params, base-part factory) -- every op and every kind
_PLATE = lambda: Box(50, 30, 8)  # noqa: E731
_CYL = lambda: Cylinder(9, 12)  # noqa: E731
_FINBODY = lambda: Cylinder(19, 30)  # noqa: E731

_CASES = [
    (
        "surface_texture",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "diamond"},
        _PLATE,
    ),
    (
        "surface_texture",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "ribbed"},
        _PLATE,
    ),
    (
        "surface_texture",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "grid"},
        _PLATE,
    ),
    (
        "surface_texture",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "dimple", "pitch": 4},
        _PLATE,
    ),
    (
        "perforate",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "thickness": 8, "kind": "round"},
        _PLATE,
    ),
    (
        "perforate",
        {
            "x0": -20,
            "x1": 20,
            "y0": -12,
            "y1": 12,
            "z_top": 4,
            "thickness": 8,
            "kind": "hex",
            "size": 5,
        },
        _PLATE,
    ),
    (
        "perforate",
        {
            "x0": -20,
            "x1": 20,
            "y0": -12,
            "y1": 12,
            "z_top": 4,
            "thickness": 8,
            "kind": "slot",
            "size": 6,
            "cols": 2,
            "rows": 2,
        },
        _PLATE,
    ),
    ("ribs", {"x0": -20, "x1": 20, "y_center": 0, "z_base": -4, "z_top": 2, "count": 3}, _PLATE),
    (
        "lightening",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "pockets", "depth": 2},
        _PLATE,
    ),
    (
        "lightening",
        {
            "x0": -20,
            "x1": 20,
            "y0": -12,
            "y1": 12,
            "z_top": 4,
            "kind": "holes",
            "depth": 2,
            "thickness": 8,
        },
        _PLATE,
    ),
    (
        "lightening",
        {"x0": -20, "x1": 20, "y0": -12, "y1": 12, "z_top": 4, "kind": "honeycomb", "depth": 2},
        _PLATE,
    ),
    ("fastener_seat", {"x": 0, "y": 0, "z_top": 4, "thickness": 8, "kind": "countersink"}, _PLATE),
    ("fastener_seat", {"x": 0, "y": 0, "z_top": 4, "thickness": 8, "kind": "counterbore"}, _PLATE),
    ("edge_treatment", {"size": 1.0, "kind": "chamfer", "zmin": 3, "zmax": 6}, _PLATE),
    ("emboss_text", {"x": -15, "y": -3, "z_top": 4, "text": "FM-01", "size": 5}, _PLATE),
    ("direction_arrow", {"x": 0, "y": 0, "z_top": 4, "length": 12}, _PLATE),
    ("knurl", {"cx": 0, "cy": 0, "z0": -6, "z1": 6, "radius": 9, "kind": "straight"}, _CYL),
    ("knurl", {"cx": 0, "cy": 0, "z0": -6, "z1": 6, "radius": 9, "kind": "cross"}, _CYL),
    (
        "cooling_fins",
        {"cx": 0, "cy": 0, "z0": -12, "z1": 12, "bore_r": 15, "fin_r": 22, "count": 6},
        _FINBODY,
    ),
]


@pytest.mark.parametrize(
    "op,params,base", _CASES, ids=[f"{o}:{p.get('kind', '')}" for o, p, _ in _CASES]
)
def test_op_produces_watertight_solid(op, params, base) -> None:
    r = apply_finish(base(), [FinishOp(op, params)])
    assert op in r.applied, r.skipped
    assert _watertight(r.part)


def test_all_registered_ops_are_covered() -> None:
    covered = {op for op, _, _ in _CASES}
    assert covered == set(op_names()), f"untested ops: {set(op_names()) - covered}"


def test_flag_default_off() -> None:
    import os

    os.environ.pop("FORGE_FINISH", None)
    assert finish_enabled() is False


def test_bad_and_subnozzle_ops_fail_closed() -> None:
    plate = Box(40, 40, 6)
    plan = [
        FinishOp("nope", {}),
        FinishOp(
            "perforate",
            {
                "x0": -15,
                "x1": 15,
                "y0": -15,
                "y1": 15,
                "z_top": 3,
                "thickness": 6,
                "kind": "round",
                "size": 0.3,
                "cols": 2,
                "rows": 2,
            },
        ),  # sub-nozzle
        FinishOp(
            "surface_texture",
            {"x0": -15, "x1": 15, "y0": -15, "y1": 15, "z_top": 3, "kind": "martian"},
        ),
    ]
    r = apply_finish(plate, plan)
    assert r.applied == []
    assert len(r.skipped) == 3
    assert r.part.volume == plate.volume  # unchanged -- nothing broke the part
