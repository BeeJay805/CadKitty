"""Physics-computed structures (ADR-0097): the deterministic authors + the contracts around them.

The 2026-07-13 turn: for load-bearing geometry the engine COMPUTES the shape from physics +
simulation instead of asking an LLM to guess it. This package is that capability promoted from
`scripts/arm_search/` into the engine proper (Fable's consolidation ruling), typed and importable:

  - ``loadcase``    the canonical validated-FEA load cases (proposer-agnostic measurement)
  - ``objectives``  the objective vector + fail-closed scoring (mass, deflection, gates)
  - ``selection``   deterministic winner selection (survivors beat the reference, ranked)
  - ``sections``    the closed-form section author (beam theory + FEA correction + print gates)
  - ``topopt``      generative SIMP topology optimization (skeleton around components)
  - ``spec``        C1: the RequirementSpec every author consumes and every gate checks against
  - ``author``      C2: the Author contract (spec -> artifacts + provenance)

The law (ADR-0097 + Fable): authors PROPOSE, one validator DISPOSES, and routing is a
deterministic table over the typed spec -- never an LLM judgment call.
"""

from forge_mind.structures.author import AuthorResult
from forge_mind.structures.loadcase import (
    G_M_S2,
    SMOKE_CANTILEVER_5N,
    LoadCase,
    canonical_link_case,
    canonical_link_orient,
    probe_env,
    run_load_case,
)
from forge_mind.structures.objectives import (
    ALLOWED_DISPOSITIONS,
    ArmObjectives,
    PartObjectives,
    fea_deflection_within,
    score_mesh,
    score_stl,
    score_stl_dir,
)
from forge_mind.structures.selection import beats_reference, select_winner
from forge_mind.structures.spec import ComponentPlacement, RequirementSpec

__all__ = [
    "ALLOWED_DISPOSITIONS",
    "G_M_S2",
    "SMOKE_CANTILEVER_5N",
    "ArmObjectives",
    "AuthorResult",
    "ComponentPlacement",
    "LoadCase",
    "PartObjectives",
    "RequirementSpec",
    "beats_reference",
    "canonical_link_case",
    "canonical_link_orient",
    "fea_deflection_within",
    "probe_env",
    "run_load_case",
    "score_mesh",
    "score_stl",
    "score_stl_dir",
    "select_winner",
]
