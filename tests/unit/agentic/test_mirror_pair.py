"""Chirality guard for mirror-by-construction: reflection must refuse handed features.

The measured motivation is real geometry, not a hypothetical: 3 of 4 declared mirror pairs in the
shipped H3 humanoid are not mirrors (docs/research/2026-07-24-mirror-pair-defect-class.md). The
danger in fixing that by reflection is trading a visible defect for an invisible one -- a
mirrored right-hand thread is left-hand, and no standard screw enters it.
"""

from __future__ import annotations

import numpy as np
import pytest

from forge_mind.agentic.cheatsheet import VERIFIED_IDIOMS
from forge_mind.agentic.mirror_pair import (
    ACHIRAL_IDIOMS,
    CHIRAL_IDIOMS,
    chirality_blockers,
    decide_mirror,
    reflect_port,
    reflect_ports,
)
from forge_mind.assembly.joints import Port
from forge_mind.types import Vec3


def test_every_idiom_is_classified() -> None:
    """THE COMPLETENESS PIN. An allow-list is only fail-closed if it is exhaustive, and a human
    remembering to update it is not a mechanism. Adding an idiom to the engine's vocabulary
    without ruling on its chirality fails the build here."""
    assert frozenset(VERIFIED_IDIOMS) == CHIRAL_IDIOMS | ACHIRAL_IDIOMS
    assert not (CHIRAL_IDIOMS & ACHIRAL_IDIOMS), "an idiom cannot be both handed and achiral"


def test_a_threaded_part_refuses_to_mirror() -> None:
    code = (
        "from bd_warehouse.thread import IsoThread\n"
        "thread = IsoThread(major_diameter=8.0, pitch=1.25, length=20, external=True)\n"
    )
    d = decide_mirror(code=code)
    assert not d.may_mirror
    assert any("left-hand" in b for b in d.blockers), d.blockers


def test_a_helix_and_text_refuse_to_mirror() -> None:
    assert any(
        "winding" in b for b in chirality_blockers("path = Helix(pitch=3, height=20, radius=5)")
    )
    assert any("backwards" in b for b in chirality_blockers("sk = Text('L', font_size=6)"))
    assert any("backwards" in b for b in chirality_blockers("emboss_text(part, 'LEFT', 4)"))


def test_a_plain_shell_may_mirror() -> None:
    code = (
        "with BuildPart() as bp:\n"
        "    Box(60, 58, 120)\n"
        "    fillet(bp.edges().filter_by(Axis.Z), 3)\n"
        "    Hole(radius=7.15)\n"
    )
    d = decide_mirror(code=code, idioms_used={"fillet_chamfer", "through_holes", "shell"})
    assert d.may_mirror, d.blockers
    assert "congruent by construction" in d.reason


def test_unreadable_source_refuses_rather_than_assuming_achiral() -> None:
    for source in (None, "", "   "):
        d = decide_mirror(code=source)
        assert not d.may_mirror
        assert any("unavailable" in b for b in d.blockers), source


def test_an_unclassified_idiom_refuses() -> None:
    # A deny-list would mirror this; the allow-list refuses, which is the whole point.
    d = decide_mirror(code="Box(10, 10, 10)", idioms_used={"some_future_helical_gear_idiom"})
    assert not d.may_mirror
    assert any("unclassified" in b for b in d.blockers), d.blockers


def test_reflected_ports_mate_the_same_way_the_originals_do() -> None:
    """The chirality trap that does NOT bite, verified against the real solver rather than
    assumed. Reflection has det = -1, so the reflected triad is left-handed and
    (Ma) x (Mn) = -M(a x n). _frame_mate builds BOTH frames as [axis, normal, axis x normal] and
    solves for a proper rotation, so it consumes only the two declared directions -- the mate
    stays correct on naively reflected ports."""
    from forge_mind.assembly.solver import _frame_mate

    port = Port(
        name="hip_pivot",
        origin=Vec3(x=0.0, y=26.0, z=167.0),
        axis=Vec3(x=0.0, y=1.0, z=0.0),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
        diameter_mm=14.3,
    )
    m = reflect_port(port, mirror_axis="y")
    assert m.origin.as_tuple() == (0.0, -26.0, 167.0)
    assert m.axis.as_tuple() == (0.0, -1.0, 0.0)
    assert m.normal.as_tuple() == (1.0, 0.0, 0.0)  # no y component to flip
    assert m.diameter_mm == 14.3 and m.name == "hip_pivot"

    # The solver must produce a PROPER rotation (det +1) for the reflected port too: a mate that
    # secretly mirrored the mover would be a silent handedness bug.
    target_origin = np.array([0.0, 0.0, 0.0])
    rot = _frame_mate(
        target_origin,
        np.array([0.0, 1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        m.origin.as_tuple(),
        m.axis.as_tuple(),
        m.normal.as_tuple(),
    )
    assert np.linalg.det(rot[:3, :3]) == pytest.approx(1.0, abs=1e-9)


def test_reflecting_twice_is_the_identity() -> None:
    port = Port(
        name="p",
        origin=Vec3(x=3.0, y=-7.5, z=2.0),
        axis=Vec3(x=0.0, y=0.6, z=0.8),
        normal=Vec3(x=1.0, y=0.0, z=0.0),
    )
    back = reflect_port(reflect_port(port, mirror_axis="y"), mirror_axis="y")
    assert back.origin.as_tuple() == port.origin.as_tuple()
    assert back.axis.as_tuple() == port.axis.as_tuple()


def test_an_unknown_mirror_axis_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unknown mirror_axis"):
        reflect_ports([], mirror_axis="sagittal")


def test_a_handed_idiom_refuses_even_when_the_source_looks_clean() -> None:
    # The author may cite the idiom while writing the geometry some other way.
    d = decide_mirror(code="Box(10, 10, 10)", idioms_used={"external_iso_thread"})
    assert not d.may_mirror
    assert any("handed idiom" in b for b in d.blockers), d.blockers
