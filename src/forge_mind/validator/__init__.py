"""Validator: 4-tier check pipeline on a CADModel + Blueprint.

Tier 1: mesh (manifold, watertight, bbox).
Tier 2: features (B-rep introspection via topology_index).
Tier 3: printability (overhangs, bridges, supports).
Tier 4: interfaces (declared mating constraints).

Emits structured Fix hints for the Reviser.
"""
