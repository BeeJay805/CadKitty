"""The author router (ADR-0097 / Fable): a DETERMINISTIC table over the typed spec.

The constitution's rule promoted to routing: the LLM classifies and extracts (it builds the
``RequirementSpec``); THIS table decides which author runs -- never an LLM judgment call at
decision time. The founder's "some stuff doesn't need topology" is row one: no declared load ->
no physics solve, ever (a physics author on a guessed load case is the false-precision vector).

| spec facts                                   | author     |
|----------------------------------------------|------------|
| no load case declared                        | agentic    |
| load + components in the domain              | topopt     |
| load + a bare classical span (no components) | analytic   |

Fallbacks stay monotone in cost (analytic -> topopt -> agentic); an author that cannot meet the
bound falls through to the next, and EVERY author's output faces the same uniform gate.
"""

from __future__ import annotations

from forge_mind.structures.spec import RequirementSpec

AGENTIC = "agentic"
ANALYTIC = "analytic"
TOPOPT = "topopt"


def route(spec: RequirementSpec) -> str:
    """The deterministic routing decision. Total: every spec routes somewhere; the default is the
    agentic path (deepest validation coverage), never a physics author with guessed inputs."""
    if not spec.has_load:
        return AGENTIC  # form/appearance-driven: no load -> no physics solve, ever
    if spec.has_components:
        return TOPOPT  # grow the skeleton around the declared pockets/mounts
    return ANALYTIC  # a bare loaded span between interfaces: closed form wins


__all__ = ["AGENTIC", "ANALYTIC", "TOPOPT", "route"]
