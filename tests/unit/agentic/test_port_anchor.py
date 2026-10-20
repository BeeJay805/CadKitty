"""The port-anchor gate (row-55 lesson): a ported part whose built mesh is nowhere near
its declared port origin was authored in a LOCAL frame instead of the shared one (the i4
probe shipped conrod_3 and piston_4 at x=0 instead of stations 107/147 -- 5 of 15 parts).
Deterministic, pre-assembly, measured against the STL itself."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import trimesh

from forge_mind.agentic.functional import PartPlan, _port_anchor_callout
from forge_mind.assembly.joints import Port


def _stl_at(tmp_path: Path, center_x: float) -> Path:
    box = trimesh.creation.box(extents=(8.0, 18.0, 42.0))
    box.apply_translation((center_x, 0.0, 9.0))
    p = tmp_path / f"part_{center_x:.0f}.stl"
    box.export(str(p))
    return p


def _part(port_x: float) -> PartPlan:
    return PartPlan(
        name="conrod_3",
        prompt="rod card",
        ports=[
            Port(
                name="big_end",
                origin={"x": port_x, "y": 0.0, "z": -12.0},
                axis={"x": 1.0},
                normal={"z": 1.0},
            )
        ],
    )


def test_dropped_frame_part_is_called_out(tmp_path: Path) -> None:
    # built at the origin, port declared at station x=107 -> the row-55 defect verbatim
    lr = SimpleNamespace(final_stl=_stl_at(tmp_path, 0.0))
    callout = _port_anchor_callout(_part(107.0), lr)
    assert callout is not None
    assert "MEASURED PLACEMENT ERROR" in callout and "107.0" in callout


def test_anchored_part_passes(tmp_path: Path) -> None:
    lr = SimpleNamespace(final_stl=_stl_at(tmp_path, 107.0))
    assert _port_anchor_callout(_part(107.0), lr) is None


def test_portless_or_missing_stl_parts_are_skipped(tmp_path: Path) -> None:
    no_ports = PartPlan(name="base", prompt="base card")
    lr = SimpleNamespace(final_stl=_stl_at(tmp_path, 0.0))
    assert _port_anchor_callout(no_ports, lr) is None
    gone = SimpleNamespace(final_stl=tmp_path / "missing.stl")
    assert _port_anchor_callout(_part(107.0), gone) is None
