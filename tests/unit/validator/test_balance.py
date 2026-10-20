"""Rotational balance -- the first general functional-validation check. Verifies the
static (CoM-on-axis) and dynamic (axis is a principal axis) conditions against known
cases, and that it fails CLOSED on untrustworthy inputs."""

from __future__ import annotations

import numpy as np
import trimesh

from forge_mind.validator.balance import rotational_balance

_ORIGIN = (0.0, 0.0, 0.0)
_Z = (0.0, 0.0, 1.0)


def _cylinder() -> trimesh.Trimesh:
    # A right cylinder on Z, centered at the origin: a perfectly balanced rotor about Z.
    return trimesh.creation.cylinder(radius=10.0, height=40.0)


def test_centered_rotor_is_certified_balanced() -> None:
    report = rotational_balance(_cylinder(), _ORIGIN, _Z)
    assert report.balanced is True
    assert report.static_ratio < 1e-6
    assert report.dynamic_ratio < 1e-6


def test_static_imbalance_is_rejected() -> None:
    # Same cylinder, but asked to spin about an axis 8 mm off its CoM -> net force.
    report = rotational_balance(_cylinder(), (8.0, 0.0, 0.0), _Z)
    assert report.balanced is False
    assert report.static_ratio > 0.03


def test_dynamic_couple_imbalance_is_rejected() -> None:
    # Axis through the CoM but TILTED off the cylinder's principal axis -> a rocking
    # couple with the CoM still perfectly on the axis (pure dynamic imbalance).
    report = rotational_balance(_cylinder(), _ORIGIN, (0.35, 0.0, 1.0))
    assert report.balanced is False
    assert report.static_ratio < 0.03  # CoM stays on the axis...
    assert report.dynamic_ratio > 0.05  # ...but the couple rocks it


def test_balance_is_frame_general_not_z_only() -> None:
    # The same balanced cylinder, laid along X, is balanced about X.
    cyl = _cylinder().apply_transform(
        trimesh.transformations.rotation_matrix(np.pi / 2, [0.0, 1.0, 0.0])
    )
    assert rotational_balance(cyl, _ORIGIN, (1.0, 0.0, 0.0)).balanced is True


def test_fails_closed_on_a_non_watertight_mesh() -> None:
    # An open surface (one triangle) has no trustworthy mass -> UNVERIFIED, never PASS.
    surface = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
    assert rotational_balance(surface, _ORIGIN, _Z).balanced is None


def test_fails_closed_on_a_degenerate_axis() -> None:
    assert rotational_balance(_cylinder(), _ORIGIN, (0.0, 0.0, 0.0)).balanced is None
