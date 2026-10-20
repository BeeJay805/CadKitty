"""main_bore_axis -- the mass-independent rotation axis for a bored rotor (the shaft bore),
which is the axis balance must be measured about. Fails closed when there is no bore."""

from __future__ import annotations

from build123d import Cylinder, Pos

from forge_mind.validator.geometry import main_bore_axis


def test_centered_bore_is_the_shaft_axis() -> None:
    part = Cylinder(radius=40, height=12) - Cylinder(radius=5, height=20)
    axis = main_bore_axis(part)
    assert axis is not None
    (px, py, _pz), direction = axis
    assert abs(px) < 0.5 and abs(py) < 0.5  # the bore sits at the centre
    assert abs(direction[2]) > 0.99  # along Z


def test_off_center_bore_axis_is_found_off_center() -> None:
    part = Cylinder(radius=40, height=12) - (Pos(15, 0, 0) * Cylinder(radius=5, height=20))
    axis = main_bore_axis(part)
    assert axis is not None
    (px, _py, _pz), _direction = axis
    assert abs(px - 15) < 0.5  # detected at the real (off-centre) bore, not the geometric centre


def test_no_bore_returns_none() -> None:
    # A solid cylinder has only a convex outer face -> no shaft bore -> fail closed.
    assert main_bore_axis(Cylinder(radius=20, height=30)) is None
