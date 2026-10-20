"""C1: the RequirementSpec -- the lingua franca every structural author consumes and every gate
checks against (Fable's foundations ruling, 2026-07-13).

The spec is the SINGLE source of the load case for both authoring AND gating (generalizing the
ADR-0096 fair-comparison requirement: candidate and reference are always measured under the same
named case). The LLM's job is to EXTRACT a spec from intent; deterministic code routes on it and
authors from it. Fail-closed honesty: a spec with no load case routes to the agentic path -- a
physics author must never run on a guessed load (the false-precision vector Fable flagged).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forge_mind.structures.loadcase import LoadCase


@dataclass(frozen=True)
class ComponentPlacement:
    """One real component the structure must build AROUND (the founder's rule: the engine makes
    the skeleton and leaves a perfect pocket + mount; it never invents the component).

    ``kind`` names a hardware-registry entry (e.g. "mg996r_servo") whose datasheet footprint
    supplies the pocket/mount dimensions; ``at_mm`` is the component-body reference position in
    the domain plane; ``clearance_mm`` pads the pocket on every side.
    """

    kind: str
    at_mm: tuple[float, float]
    clearance_mm: float = 0.5


@dataclass(frozen=True)
class RequirementSpec:
    """What a structural part must DO, stated as data. Everything an author needs; nothing more.

    - ``envelope_mm``: the allowed (L, W, H) design volume (stated dims -> pinned, ADR-0080).
    - ``load``: the named load case, or None (no declared load -> not a physics-authored part).
    - ``deflection_bound_mm``: the stiffness requirement under ``load`` (None = no bound claimed).
    - ``ref_mass_g``: the reference mass to beat, when a reference exists (None = report-only).
    - ``interfaces``: free-form stated interface dims (bolt-hole d/spacing, spans) that pins
      protect; consumed by the section author's pads and the topopt keep_solid mounts.
    - ``components``: real parts to build around (pockets + mounts).
    """

    envelope_mm: tuple[float, float, float]
    material_id: str = "pla"
    load: LoadCase | None = None
    deflection_bound_mm: float | None = None
    ref_mass_g: float | None = None
    interfaces: dict[str, float] = field(default_factory=dict)
    components: list[ComponentPlacement] = field(default_factory=list)

    @property
    def has_load(self) -> bool:
        return self.load is not None

    @property
    def has_components(self) -> bool:
        return bool(self.components)


__all__ = ["ComponentPlacement", "RequirementSpec"]
