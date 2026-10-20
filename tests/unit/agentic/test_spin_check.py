"""balance_feedback -- the gated, fail-closed, advisory wiring of the balance check into the
agentic validator. Only a rotating-power part with a real shaft bore and a genuine imbalance
produces an advisory; everything else stays silent (no noise, no false claim)."""

from __future__ import annotations

from pathlib import Path

import trimesh
from build123d import Cylinder, Pos, export_stl

from forge_mind.agentic.brief import BriefFeature, DesignBrief
from forge_mind.agentic.spin_check import balance_feedback


def _brief(name: str) -> DesignBrief:
    return DesignBrief(
        object_name=name,
        summary=name,
        visual_description="x",
        overall_envelope_mm="80 x 80 x 12",
        features=[
            BriefFeature(name="bore", purpose="shaft", approx_dims_mm="10", method="revolve")
        ],
        min_wall_mm=2.0,
        print_orientation_note="flat down",
    )


def _part_and_mesh(part: object, tmp_path: Path) -> tuple[object, trimesh.Trimesh]:
    stl = tmp_path / "p.stl"
    export_stl(part, str(stl))  # type: ignore[arg-type]
    return part, trimesh.load(str(stl), force="mesh")


def _off_center_bore_disc() -> object:
    return Cylinder(radius=40, height=12) - (Pos(15, 0, 0) * Cylinder(radius=5, height=20))


def _centered_bore_disc() -> object:
    return Cylinder(radius=40, height=12) - Cylinder(radius=5, height=20)


def test_unbalanced_rotor_gets_a_balance_advisory(tmp_path: Path) -> None:
    part, mesh = _part_and_mesh(_off_center_bore_disc(), tmp_path)
    out = balance_feedback(_brief("flywheel"), part, mesh)
    assert len(out) == 1 and "UNBALANCED" in out[0]


def test_balanced_rotor_gets_no_advisory(tmp_path: Path) -> None:
    part, mesh = _part_and_mesh(_centered_bore_disc(), tmp_path)
    assert balance_feedback(_brief("flywheel"), part, mesh) == []


def test_non_rotating_part_is_never_balance_checked(tmp_path: Path) -> None:
    # A bracket does not spin -> gated out even with the same off-center bore (no noise).
    part, mesh = _part_and_mesh(_off_center_bore_disc(), tmp_path)
    assert balance_feedback(_brief("mounting bracket"), part, mesh) == []


def test_no_bore_fails_closed(tmp_path: Path) -> None:
    # A solid rotor with no shaft bore (journal-shaft style) -> no axis -> no claim.
    part, mesh = _part_and_mesh(Cylinder(radius=30, height=20), tmp_path)
    assert balance_feedback(_brief("flywheel"), part, mesh) == []
