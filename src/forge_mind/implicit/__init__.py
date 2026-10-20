"""Implicit / voxel (SDF) geometry lane (ADR-0026, Phase 7).

A third geometry path beside deterministic-B-rep and agentic-build123d, for parts B-rep can't express
-- gyroid/TPMS lattices and strut lattices. Deterministic (code derives the field, ADR-0009), meshed
to a watertight STL via scikit-image marching cubes. Mesh-only: no editable STEP for this geometry.

Submodules are imported lazily by callers so importing this package stays cheap (numpy only; skimage
loads inside the mesher).
"""

from __future__ import annotations

__all__: list[str] = []
