"""Measured, validated kinematics (ADR-0086) — "no fake movement".

The viewer may only animate motion this package has measured and validated. ``axis``
discovers a single body's real spin axis (or refuses); ``pair`` certifies relative rotation
of a two-body pair by measured common axis + as-built interference + a full 360-degree swept
collision. Results are ANALYSIS dataclasses; ``physics.report.motion_project`` converts them
into the frozen frontend contract.
"""

from forge_mind.motion.axis import SpinAxis, spin_axis
from forge_mind.motion.mechanism import (
    Coupling,
    MechanismMotion,
    gear_pair_coupled,
    rolling_mechanism,
)
from forge_mind.motion.pair import PairMotion, pair_rotation

__all__ = [
    "Coupling",
    "MechanismMotion",
    "PairMotion",
    "SpinAxis",
    "gear_pair_coupled",
    "pair_rotation",
    "rolling_mechanism",
    "spin_axis",
]
