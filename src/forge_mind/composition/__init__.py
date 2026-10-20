"""Parametric composition engine: base body x interface subsets x placements -> a large
validated part space, every part using standards-backed hardware, nothing untrustworthy admitted."""

from forge_mind.composition.engine import (
    SHARDS,
    CompositionManifest,
    CompositionPart,
    CompositionReject,
    FeatureChoice,
    build_compositions,
    enumerate_recipes,
)

__all__ = [
    "SHARDS",
    "CompositionManifest",
    "CompositionPart",
    "CompositionReject",
    "FeatureChoice",
    "build_compositions",
    "enumerate_recipes",
]
