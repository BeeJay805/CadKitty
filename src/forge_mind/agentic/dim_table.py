"""DimTable — parse the brief + prompt ONCE into typed, provenance-anchored quantities.

ADR-0088 slice 1 (council-unanimous): the measured pain is that every check re-parses the
same prose with its own regex (a depth once read as a diameter; provenance re-derived at
verdict time). This is the single extraction surface they converge on — a FLAT table of
typed ``Quantity`` values, NOT a graph and NOT a new LLM output format. The graph slices
(structured brief -> measured-graph binder) are pre-registered behind numeric gates in
ADR-0088; this slice changes no model behavior and, by construction, no verdict.

Each ``Quantity`` carries:
  * ``role`` — what KIND of number it is (diameter / length / angle / pitch / pcd / height /
    envelope_extent). A typed role makes "a depth parsed as a diameter" structurally
    impossible: the parse evidence sets the role, no consumer re-guesses it.
  * ``provenance`` — STATED (the user's own number, anchored to a prompt span), VERIFIED (a
    cited registry value), or DERIVED (the model's own brief number). Only STATED/VERIFIED
    are protected currency for the drift gate (the ADR-0080 anchor invariant, made a
    parse-time fact instead of a verdict-time re-derivation). A DERIVED number keeps only
    the coarse envelope gate — the row-41 lesson, now structural.
  * ``span`` — char offsets into the raw prompt for a STATED scalar (the anchor evidence);
    None when the value came from the brief/intake/registry (a different artifact).

FLAG-GATED (``FORGE_DIM_TABLE``): the validator builds the table and feeds the two E1 checks
from it only when the flag is set; unset = the current per-check parsing (byte-identical).
The queries (:meth:`stated_values`, :meth:`envelope`, :meth:`positions`) reproduce the exact
outputs of ``stated_numbers`` / ``parse_envelope_mm`` / ``stated_bore_positions`` so parity is
provable on the seeded-defect audit before the flag ever defaults on. Once it does, the
duplicated regexes in stated_dims/mapped_dims are deleted (this becomes their only home).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from forge_mind.agentic.brief import DesignBrief, parse_envelope_mm
from forge_mind.agentic.mapped_dims import (
    _DIA_TXT_RE as _POS_DIA_RE,
)

# Reuse the COMPILED patterns from the current owners so the table is parity-exact, not a
# re-implementation that could drift. This slice consolidates their USE; a follow-up moves
# their definitions here and deletes them there.
from forge_mind.agentic.mapped_dims import (
    _HEIGHT_RE,
    _PCD_RE,
    _PITCH_RE,
    PositionSpec,
    _is_hole_feature,
)
from forge_mind.agentic.provenance import parse_stated_dimensions
from forge_mind.agentic.spatial import _planned_hole_specs
from forge_mind.agentic.stated_dims import (
    _ANGLE_RE,
    _DIA_CONTEXT_RE,
    _DIM_CHAIN_RE,
    _MM_RE,
    _NUM_RE,
)

# Provenance tiers (aligned with provenance.Tier's concept; STATED == the user's number,
# VERIFIED == a cited registry value, DERIVED == the model's own reasoning).
STATED = "stated"
VERIFIED = "verified"
DERIVED = "derived"

# Roles: as specific as the parse evidence, never guessed.
DIAMETER = "diameter"
LENGTH = "length"
ANGLE = "angle"
PITCH = "pitch"
PCD = "pcd"
HEIGHT = "height"
ENVELOPE = "envelope_extent"

_VALUE_MAX_MM = 10_000.0

# Env flag: the value is irrelevant, only presence (same convention as the other levers).
_FLAG = "FORGE_DIM_TABLE"


def dim_table_enabled() -> bool:
    return bool(os.environ.get(_FLAG))


@dataclass(frozen=True)
class Quantity:
    """One parsed number with its kind and where it came from."""

    value_mm: float
    role: str
    provenance: str
    span: tuple[int, int] | None = None
    feature_label: str | None = None


@dataclass(frozen=True)
class DimTable:
    """Every quantity parsed from one (prompt, brief). Queries reproduce the legacy parsers."""

    quantities: tuple[Quantity, ...]
    _wall_mm: float
    _envelope: tuple[float, float, float] | None
    _positions: tuple[PositionSpec, ...]

    def stated_values(self) -> frozenset[float]:
        """The protected-number set == ``stated_dims.stated_numbers``: STATED/VERIFIED scalar
        values in range, excluding the wall floor (a floor, not a dimension)."""
        return frozenset(
            q.value_mm
            for q in self.quantities
            if q.provenance in (STATED, VERIFIED)
            and q.role != ENVELOPE
            and 0.0 < q.value_mm <= _VALUE_MAX_MM
            and abs(q.value_mm - self._wall_mm) > 1e-6
        )

    def envelope(self) -> tuple[float, float, float] | None:
        return self._envelope

    def positions(self) -> list[PositionSpec]:
        return list(self._positions)


def _scalar_quantities(prompt: str) -> list[Quantity]:
    """User-stated scalars from the prompt, span-anchored, mirroring stated_numbers' prompt
    parsing (same patterns -> same values, now typed)."""
    out: list[Quantity] = []
    for m in _MM_RE.finditer(prompt):
        out.append(Quantity(float(m.group(1)), LENGTH, STATED, m.span(1)))
    for m in _DIA_CONTEXT_RE.finditer(prompt):
        out.append(Quantity(float(m.group(1)), DIAMETER, STATED, m.span(1)))
    for m in _DIM_CHAIN_RE.finditer(prompt):
        for n in _NUM_RE.findall(m.group(1)):
            out.append(Quantity(float(n), LENGTH, STATED, m.span(1)))  # chain shares its span
    for m in _ANGLE_RE.finditer(prompt):
        out.append(Quantity(float(m.group(1)), ANGLE, STATED, m.span(1)))
    return out


def _context_quantities(extra_context: str | None) -> list[Quantity]:
    if not extra_context:
        return []
    return [
        Quantity(float(d.value), LENGTH, STATED, None)
        for d in parse_stated_dimensions(extra_context)
    ]


def _registry_quantities(brief: DesignBrief) -> list[Quantity]:
    """Registry-resolved (ISO) hole targets — VERIFIED provenance (a cited artifact)."""
    out: list[Quantity] = []
    try:
        for spec in _planned_hole_specs(brief):
            if spec.target_d_mm is not None and "ISO" in spec.source:
                out.append(Quantity(float(spec.target_d_mm), DIAMETER, VERIFIED, None))
    except Exception:
        pass  # an unparseable plan loses only the registry source, never the prompt's
    return out


def _position_specs(brief: DesignBrief, stated: frozenset[float]) -> list[PositionSpec]:
    """Reproduce ``mapped_dims.stated_bore_positions``: user-stated bore positions from hole
    features. Kept parity-exact (same patterns, same anchor test)."""
    specs: list[PositionSpec] = []

    def _anchored(v: float) -> bool:
        return any(abs(v - s) <= 0.05 for s in stated)

    for f in brief.features:
        if not _is_hole_feature(f.name, getattr(f, "standard", None)):
            continue
        dims = str(getattr(f, "approx_dims_mm", "") or "")
        dm = _POS_DIA_RE.search(dims)
        bore_d = float(next(g for g in dm.groups() if g is not None)) if dm else None
        for rx, kind in ((_PITCH_RE, PITCH), (_PCD_RE, PCD), (_HEIGHT_RE, HEIGHT)):
            m = rx.search(dims)
            if not m:
                continue
            val = float(next(g for g in m.groups() if g is not None))
            if _anchored(val):
                specs.append(PositionSpec(f.name, kind, val, bore_d))
    return specs


def build_dim_table(prompt: str, brief: DesignBrief, extra_context: str | None = None) -> DimTable:
    """Parse (prompt, brief) into the typed quantity table. Deterministic, no LLM."""
    quants = (
        _scalar_quantities(prompt or "")
        + _context_quantities(extra_context)
        + _registry_quantities(brief)
    )
    wall = float(brief.min_wall_mm) if brief.min_wall_mm else 0.0
    stated = frozenset(
        q.value_mm
        for q in quants
        if q.provenance in (STATED, VERIFIED)
        and 0.0 < q.value_mm <= _VALUE_MAX_MM
        and abs(q.value_mm - wall) > 1e-6
    )
    # SpecGraph slice 2 (ADR-0088): when the researcher emitted typed spec_dims (FORGE_SPEC_BRIEF,
    # emission gate 100% at n=11), ingest them as feature-scoped TYPED quantities -- the role is
    # authoritative, not inferred, and feature_id is the identity. Verdict-neutral by construction:
    # a spec_dim value is STATED only when it is already in the prompt set (so already in `stated`),
    # else DERIVED (excluded from protected currency) -- stated_values() cannot move.
    quants.extend(_spec_dim_quantities(brief, stated))
    env = parse_envelope_mm(brief.overall_envelope_mm)
    if env is not None:  # each extent is STATED iff the user's numbers contain it, else DERIVED
        for e in env:
            prov = STATED if any(abs(e - s) <= 0.01 for s in stated) else DERIVED
            quants.append(Quantity(float(e), ENVELOPE, prov, None))
    positions = _position_specs(brief, stated)
    return DimTable(tuple(quants), wall, env, tuple(positions))


def _spec_dim_quantities(brief: DesignBrief, stated: frozenset[float]) -> list[Quantity]:
    """Typed quantities from any feature's emitted spec_dims (SpecGraph slice 2). The role is
    the model's typed claim (authoritative, not inferred); provenance is STATED only when the
    value is in the user's prompt set, else DERIVED. feature_id carries identity."""
    out: list[Quantity] = []
    for f in brief.features:
        fid = getattr(f, "feature_id", None) or f.name
        for sd in getattr(f, "spec_dims", None) or []:
            val = float(sd.value_mm)
            prov = STATED if any(abs(val - s) <= 0.05 for s in stated) else DERIVED
            out.append(Quantity(val, sd.role, prov, None, fid))
    return out


__all__ = [
    "DERIVED",
    "STATED",
    "VERIFIED",
    "DimTable",
    "Quantity",
    "build_dim_table",
    "dim_table_enabled",
]
