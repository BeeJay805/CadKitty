"""Assembly layer (ADR-0010 spike): multi-body PartDSL above the single-part IR.

The hinge spike's deterministic spine:

    AssemblyGraph (nodes=archetypes, edges=typed joints between ports)
      -> ConstraintSolver (frame-mate; owns every world transform)
      -> AssemblyProgram (N single-body PartPrograms + world transforms)
      -> compile_assembly (one body each, reusing the single-part compiler)
      -> validate_assembly (per-body mesh + coaxial + clearance + range-of-motion)

Nothing here is hinge-specific: a hinge EMERGES from generic ``knuckled_plate`` +
``rod`` archetypes wired by ``revolute`` / ``coaxial_insert`` joints. There is no
``hinge`` object-template anywhere in this package.
"""

from __future__ import annotations

__all__: list[str] = []
