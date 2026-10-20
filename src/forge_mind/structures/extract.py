"""The LLM-side RequirementSpec extractor (Fable's next move; ADR-0098 follow-up).

The LLM's ONE structural job: parse intent into the typed spec -- it never chooses the author and
never invents geometry. The extraction is built fail-closed around the load-case single point of
failure Fable flagged (a wrong extracted load yields a confidently-optimal wrong part the gate
will PASS, because it gates against the same wrong spec):

  - ``has_declared_load`` is the honesty bit: True ONLY when the USER stated a load/payload.
    No load stated -> ``to_spec`` emits ``load=None`` -> the router sends it to the agentic path.
    The extractor NEVER invents a payload, bound, or direction.
  - Components are accepted only for REGISTRY-KNOWN kinds (``hardware.footprint_for``); unknown
    kinds are surfaced in ``dropped_components``, never guessed at.
  - A component without a stated position gets a NAMED deterministic placement policy ("tip" /
    "center" / "root"), applied by code -- the LLM supplies the hint, code supplies the numbers.
  - The finished spec's load case is echoed in human units (``describe_load_case``) so a human
    can spot '50 kg' where '0.5 kg' was meant.

Extraction itself costs one cheap structured call (same client stack as ``research_brief``;
subscription-only routing applies).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.llm.provider import LLMClientLike, make_client
from forge_mind.structures.hardware import footprint_for
from forge_mind.structures.loadcase import G_M_S2, LoadCase, describe_load_case
from forge_mind.structures.spec import ComponentPlacement, RequirementSpec

EXTRACT_MAX_TOKENS = 2000
_SONNET = "claude-sonnet-5"
# The default serviceability deflection ratio (span / N) used when a load is declared but no
# stiffness bound is stated (B2a). 250 is a standard structural-member serviceability limit;
# always LABELED bound_provenance='defaulted...' so it is never a hidden claim. Tunable.
_SERVICEABILITY_RATIO = 250.0


class ExtractedComponent(BaseModel):
    """One real component the user named. Position is optional -- never invented."""

    model_config = ConfigDict(extra="forbid")

    kind: str = Field(..., description="Hardware kind, e.g. 'mg996r_servo'. Use registry names.")
    at_mm: tuple[float, float] | None = Field(
        default=None,
        description="Stated (x, y) body-center position in mm, ONLY if the user gave one.",
    )
    placement_hint: Literal["tip", "center", "root"] = Field(
        default="tip", description="Where it belongs when no position was stated."
    )


class ExtractedRequirements(BaseModel):
    """What the LLM emits. Every load field is None unless the USER stated it."""

    model_config = ConfigDict(extra="forbid")

    part_name: str = Field(..., description="Short name for the part, e.g. 'drone arm'.")
    envelope_mm: tuple[float, float, float] | None = Field(
        default=None, description="Stated (L, W, H) in mm; None if not stated."
    )
    has_declared_load: bool = Field(
        ...,
        description="True ONLY if the user STATED a load/payload/force. Never infer one.",
    )
    payload_kg: float | None = Field(
        default=None, description="The STATED payload mass in kg; None unless stated."
    )
    deflection_bound_mm: float | None = Field(
        default=None, description="The STATED stiffness bound in mm; None unless stated."
    )
    material: str = Field(default="pla")
    components: list[ExtractedComponent] = Field(default_factory=list)
    interfaces: dict[str, float] = Field(
        default_factory=dict, description="Stated interface dims (hole d, spacing...), mm."
    )
    load_note: str = Field(
        default="", description="The load restated in the user's own words (audit trail)."
    )


_SYSTEM = """You extract STRUCTURAL REQUIREMENTS from a part request into a typed schema.

HARD RULES (fail-closed honesty -- a wrong load produces a confidently-wrong part):
1. has_declared_load is true ONLY if the request STATES a load, payload, force, or weight.
   "strong", "sturdy", "stiff" are NOT declared loads. Never invent payload_kg or a bound.
2. Copy numbers verbatim; convert g->kg only. Leave any unstated field null.
3. components: only real hardware the user names (servos, motors). Use registry kind names
   (e.g. 'mg996r_servo'). Include a position ONLY if the user gave one; otherwise set
   placement_hint (tip/center/root) from context.
4. load_note: quote the user's own load wording verbatim (or "" if none)."""


def extract_requirements(
    prompt: str, *, client: LLMClientLike | None = None, model: str | None = None
) -> ExtractedRequirements:
    """One cheap structured call: intent -> ExtractedRequirements (the LLM half)."""
    client = client or make_client()
    extracted, _usage = client.generate_structured(
        system=_SYSTEM,
        user=f"Part request:\n{prompt.strip()}\n\nExtract the requirements.",
        output_model=ExtractedRequirements,
        model=model or _SONNET,
        max_tokens=EXTRACT_MAX_TOKENS,
    )
    return extracted


def _place(
    comp: ExtractedComponent, envelope: tuple[float, float, float]
) -> ComponentPlacement | None:
    """Deterministic placement (code supplies numbers, never the LLM): a stated position is used
    verbatim; else the named policy places the body center in the L x H plane."""
    length, _depth, height = envelope
    if comp.at_mm is not None:
        at = comp.at_mm
    else:
        frac = {"tip": 0.8, "center": 0.5, "root": 0.2}[comp.placement_hint]
        at = (round(length * frac, 1), round(height / 2, 1))
    return ComponentPlacement(kind=comp.kind, at_mm=at, clearance_mm=0.5)


def to_spec(
    extracted: ExtractedRequirements,
    *,
    default_envelope_mm: tuple[float, float, float],
) -> tuple[RequirementSpec, dict[str, object]]:
    """The pure half: ExtractedRequirements -> (RequirementSpec, extraction_report).

    The report carries the honesty surface: the human-units load echo, the verbatim load_note,
    and any components DROPPED for being registry-unknown (surfaced, never guessed)."""
    envelope = extracted.envelope_mm or default_envelope_mm
    load: LoadCase | None = None
    if extracted.has_declared_load and extracted.payload_kg is not None:
        load = LoadCase(
            name=f"{extracted.part_name.replace(' ', '_')}_declared",
            force_n=extracted.payload_kg * G_M_S2,
        )
    # B2a -- the DENOMINATOR fix: a real user states a load ("holds 0.5 kg") but almost never a
    # deflection bound ("max 0.046 mm"), and the physics authors need one -- so a declared-load
    # request with no stated bound would fall through to the LLM, starving the metric. Default a
    # SERVICEABILITY bound from the span (span/250, the standard structural-member ratio) and
    # LABEL it (bound_provenance) exactly like the envelope default -- never a hidden number. A
    # stated bound is used verbatim; a precision requirement is always the user's to state.
    bound: float | None = None
    bound_provenance: str | None = None
    if load is not None:
        if extracted.deflection_bound_mm is not None:
            bound, bound_provenance = extracted.deflection_bound_mm, "stated"
        else:
            span = max(envelope)
            bound = round(span / _SERVICEABILITY_RATIO, 3)
            bound_provenance = f"defaulted: span/{_SERVICEABILITY_RATIO:.0f} serviceability limit"
    components: list[ComponentPlacement] = []
    dropped: list[str] = []
    for comp in extracted.components:
        try:
            footprint_for(comp.kind)
        except KeyError:
            dropped.append(comp.kind)
            continue
        placed = _place(comp, envelope)
        if placed is not None:
            components.append(placed)
    spec = RequirementSpec(
        envelope_mm=envelope,
        material_id=extracted.material.lower(),
        load=load,
        deflection_bound_mm=bound,
        interfaces=dict(extracted.interfaces),
        components=components,
    )
    report: dict[str, object] = {
        "part_name": extracted.part_name,
        "load_case_echo": None if load is None else describe_load_case(load),
        "load_note_verbatim": extracted.load_note,
        "deflection_bound_mm": bound,
        "bound_provenance": bound_provenance,  # stated | defaulted: span/N ... | None
        # Envelope PROVENANCE (T3): a defaulted design volume is honest only when LABELED --
        # the certificate must show whether the user stated the envelope or the engine assumed
        # one (the extractor never silently invents dims; it flags them).
        "envelope_mm": list(envelope),
        "envelope_provenance": "stated" if extracted.envelope_mm is not None else "defaulted",
        "dropped_components": dropped,
        "placed_components": [{"kind": c.kind, "at_mm": list(c.at_mm)} for c in components],
    }
    return spec, report


__all__ = [
    "ExtractedComponent",
    "ExtractedRequirements",
    "extract_requirements",
    "to_spec",
]
