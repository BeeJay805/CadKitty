"""Bill of Materials — what to buy to build a part, pulled ONLY from the standards registry.

A part declares its registry-backed hardware as ``PartProgram.hardware_features`` (bearing seats,
clearance holes, heat-set bosses, ...). Each feature maps to ONE buyable component: a clearance
hole / counterbore -> a screw, a heat-set boss -> an insert, a captive-nut pocket -> a hex nut, a
bearing seat -> a bearing, and so on. The BOM aggregates by component identity (4 M3 clearance
holes -> "4x M3 socket-head cap screw") and cites each item's standard straight from the registry
``source`` field -- the SAME registry the geometry was built from, so the citation cannot drift.

Discipline (the part to get right):
  - NEVER invent a part. The component label is fixed per category; the size/source/provisional
    flag come verbatim from the registry. A feature whose category has no buyable mapping, or whose
    key is not in the registry, is FLAGGED in ``unmapped`` -- never guessed into a line.
  - Length is NOT in the registry (it depends on the assembly stack-up), so a screw line carries no
    fabricated length. The make-instructions note this; the BOM does not invent one.
  - Purely additive + read-only over the program: it builds nothing and changes no geometry.

Scope: the BOM covers BOTH the standards ``hardware_features`` AND the face-mounted components
(a NEMA motor, an axial fan, a T-slot extrusion, a board mount) declared in ``program.interfaces``.
A motor/fan face contributes its 4 mounting screws; an extrusion its 2 T-nuts + 2 bolts; a board
mount its 4 screws -- each cited from the registry, with the thread read from the SAME interface
clearance facts the geometry used. A face interface with no buyable mapping is FLAGGED, not guessed.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.dsl.components.registry import (
    get_extrusion_component,
    known_extrusion_component_ids,
)
from forge_mind.dsl.interfaces.declaration import InterfaceDeclaration, InterfaceType
from forge_mind.dsl.interfaces.fan import FAN_BOLT_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m5 import M5_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.m6 import M6_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.nema17 import M3_CLEARANCE_RADIUS_MM
from forge_mind.dsl.interfaces.pi4 import M25_CLEARANCE_RADIUS_MM
from forge_mind.dsl.program import PartProgram
from forge_mind.dsl.standards import ALL_REGISTRIES, heat_set_insert, hex_nut

# A bolt/screw whose mate is OUTSIDE this part: the length depends on the assembly stack-up, so we
# refuse to guess it (no-false-PASS at the BOM level).
_LENGTH_DEPENDS = "length depends on what you're mounting to"

# Corner-bolt clearance RADIUS (mm) -> thread, from the SAME interface constants the geometry used
# (so the BOM thread can't drift from the built hole). Faces store a clearance radius, not a thread
# name; this reads it back. A radius not in this table is FLAGGED, never guessed.
_THREAD_BY_RADIUS: dict[float, str] = {
    M25_CLEARANCE_RADIUS_MM: "M2.5",
    M3_CLEARANCE_RADIUS_MM: "M3",
    FAN_BOLT_CLEARANCE_RADIUS_MM: "M3",  # axial-fan corners are M3 (== M3 radius; kept explicit)
    M5_CLEARANCE_RADIUS_MM: "M5",
    M6_CLEARANCE_RADIUS_MM: "M6",
}
_FASTENER_REG = "metric_fastener"  # the screw/bolt itself is an ISO 4762 SHCS in the standards reg


class _Buyable:
    """How one hardware-feature category becomes a purchasable line: the component label (from the
    key) + the registry that owns its cited dimension/provenance."""

    __slots__ = ("label", "reg_name")

    def __init__(self, label: Callable[[str], str], reg_name: str) -> None:
        self.label = label
        self.reg_name = reg_name


# Conformance category -> the buyable component. Covers every category the conformance gate knows
# (validator.conformance._CATEGORIES): the 7 the composable interpreter emits today plus the two
# conformance-only aliases (tapped_hole, bearing_od). The label is FIXED per category; only the
# size key varies. A clearance hole / counterbore / tapped hole all take a socket-head cap screw
# (the catalog's screw, which counterbores are sized for). A bearing seat and the bearing-OD check
# describe the SAME physical bearing.
_BUYABLE: dict[str, _Buyable] = {
    "clearance_hole": _Buyable(lambda k: f"{k} socket-head cap screw", "metric_fastener"),
    "counterbore": _Buyable(lambda k: f"{k} socket-head cap screw", "metric_fastener"),
    "tapped_hole": _Buyable(lambda k: f"{k} machine screw", "metric_fastener"),
    "heat_set_insert": _Buyable(lambda k: f"{k} heat-set threaded insert", "heat_set_insert"),
    "captive_nut_pocket": _Buyable(lambda k: f"{k} hex nut", "hex_nut"),
    "bearing_seat": _Buyable(lambda k: f"{k} deep-groove ball bearing", "bearing"),
    "bearing_od": _Buyable(lambda k: f"{k} deep-groove ball bearing", "bearing"),
    "dowel_hole": _Buyable(lambda k: f"Ø{k} mm dowel pin", "dowel_pin"),
    "magnet_pocket": _Buyable(lambda k: f"{k} disc magnet", "magnet"),
}

# The kind of role each category plays, so the line records what it IS without re-deriving it later.
_KIND: dict[str, str] = {
    "clearance_hole": "fastener",
    "counterbore": "fastener",
    "tapped_hole": "fastener",
    "heat_set_insert": "insert",
    "captive_nut_pocket": "nut",
    "bearing_seat": "bearing",
    "bearing_od": "bearing",
    "dowel_hole": "dowel",
    "magnet_pocket": "magnet",
}


class BomLine(BaseModel):
    """One purchasable line: a component, a quantity, and its cited standard."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    component: str = Field(..., description="What to buy, e.g. 'M3 socket-head cap screw'.")
    kind: str = Field(..., description="Role: fastener / insert / nut / bearing / dowel / magnet.")
    key: str = Field(..., description="The size key: 'M3', '608', '6x3'.")
    quantity: int = Field(..., ge=1, description="How many to buy.")
    source: str = Field(..., description="Cited standard / reference, verbatim from the registry.")
    provisional: bool = Field(
        default=False,
        description="True = the cited spec is NOT a published standard (verify before production).",
    )
    length_mm: float | None = Field(
        default=None,
        description="SUGGESTED fastener length (mm) where the in-part stack-up is known (a screw into "
        "a heat-set insert / a bolt into a captive nut). None when it depends on an external mate.",
    )
    length_note: str | None = Field(
        default=None,
        description="Honest length guidance: a deterministic suggestion, or that it depends on the "
        "external mate (never guessed). None for non-fastener lines (a bearing/magnet/dowel/T-nut).",
    )

    def render(self) -> str:
        """A one-line shopping entry, e.g. '4x M3 socket-head cap screw [PROVISIONAL]'."""
        tag = " [PROVISIONAL]" if self.provisional else ""
        note = f" ({self.length_note})" if self.length_note else ""
        return f"{self.quantity}x {self.component}{tag}{note}"


class BillOfMaterials(BaseModel):
    """The aggregated parts list for one part, plus any features that mapped to no buyable item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lines: list[BomLine] = Field(default_factory=list, description="Purchasable lines, aggregated.")
    unmapped: list[str] = Field(
        default_factory=list,
        description="Features with no registry-backed buyable component -- FLAGGED, not guessed.",
    )

    @property
    def total_items(self) -> int:
        """Total count of physical components across all lines."""
        return sum(line.quantity for line in self.lines)

    @property
    def has_provisional(self) -> bool:
        return any(line.provisional for line in self.lines)

    def summary(self) -> str:
        """A compact human summary: '4x M3 socket-head cap screw, 1x 608 deep-groove ball bearing'."""
        if not self.lines:
            return "no registry-backed hardware"
        return ", ".join(line.render() for line in self.lines)


def _thread_for_radius(radius: float) -> str | None:
    """Map a corner-bolt clearance radius back to its thread; None if not a known thread (flag it)."""
    for known, thread in _THREAD_BY_RADIUS.items():
        if abs(known - radius) < 1e-6:
            return thread
    return None


def _fastener_source(thread: str) -> str:
    """The cited standard for a screw/bolt of this thread (ISO 4762 SHCS, from the standards reg)."""
    return ALL_REGISTRIES[_FASTENER_REG].get(thread).source


def _extrusion_tnut_source(radius: float) -> str | None:
    """The cited extrusion-registry source for a T-nut whose bolt matches ``radius``; None if no
    extrusion profile uses that bolt size (flag it -- never invent a T-nut)."""
    for ext_id in known_extrusion_component_ids():
        ext = get_extrusion_component(ext_id)
        if abs(ext.slot.bolt_clearance_radius_mm - radius) < 1e-6:
            return ext.source
    return None


def _radius_param(decl: InterfaceDeclaration, key: str) -> float | None:
    """Read a numeric clearance-radius param off a declaration (legacy faces omit it -> None)."""
    v = decl.params.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def _face_lines(
    interfaces: list[InterfaceDeclaration],
    add: Callable[..., None],
    unmapped: list[str],
) -> None:
    """Bill the face-mounted components (motor / fan / extrusion / board mount) from the program's
    interface declarations. The thread is read from the declaration's clearance facts (legacy faces
    default to their documented size); each item is cited from the registry. PI4_PORTS are open
    cutouts (no buyable); an unrecognized face is FLAGGED."""
    for decl in interfaces:
        it = decl.interface_type
        if it is InterfaceType.NEMA17_FACE:
            # legacy NEMA 17 omits the param -> M3; a registry motor (NEMA 14/23) carries its radius.
            radius = _radius_param(decl, "bolt_clearance_radius")
            thread = "M3" if radius is None else _thread_for_radius(radius)
            if thread is None:
                unmapped.append(
                    f"{decl.name}: motor bolt clearance r={radius} is not a known thread"
                )
                continue
            add(
                component=f"{thread} socket-head cap screw",
                kind="fastener",
                key=thread,
                qty=4,
                source=_fastener_source(thread),
                provisional=False,
            )
        elif it is InterfaceType.FAN_FACE:
            # 40/60 mm axial fans use M3 corner screws (the registry fan facts).
            add(
                component="M3 socket-head cap screw",
                kind="fastener",
                key="M3",
                qty=4,
                source=_fastener_source("M3"),
                provisional=False,
            )
        elif it is InterfaceType.EXTRUSION_2020_SLOT:
            # legacy 2020 omits the param -> M5; a registry extrusion (3030) carries M6.
            radius = _radius_param(decl, "bolt_clearance_radius")
            r = M5_CLEARANCE_RADIUS_MM if radius is None else radius
            thread = _thread_for_radius(r)
            tnut_source = _extrusion_tnut_source(r)
            if thread is None or tnut_source is None:
                unmapped.append(f"{decl.name}: extrusion T-nut bolt r={r} is not a known profile")
                continue
            add(  # the T-nuts (cited from the extrusion component registry)
                component=f"{thread} T-slot nut",
                kind="tnut",
                key=thread,
                qty=2,
                source=tnut_source,
                provisional=False,
            )
            add(  # the bolts that thread into them (ISO 4762)
                component=f"{thread} socket-head cap screw",
                kind="fastener",
                key=thread,
                qty=2,
                source=_fastener_source(thread),
                provisional=False,
            )
        elif it is InterfaceType.PI4_MOUNT:
            radius = _radius_param(decl, "screw_radius")
            thread = _thread_for_radius(radius) if radius is not None else None
            if thread is None:
                unmapped.append(f"{decl.name}: board screw r={radius} is not a known thread")
                continue
            add(
                component=f"{thread} socket-head cap screw",
                kind="fastener",
                key=thread,
                qty=4,
                source=_fastener_source(thread),
                provisional=False,
            )
        elif it is InterfaceType.PI4_PORTS:
            continue  # an open port cutout, not a buyable component
        else:
            unmapped.append(f"{decl.name}: face interface {it.value} has no buyable mapping")


def _length_for(kind: str, key: str, min_wall: float | None) -> tuple[float | None, str | None]:
    """Suggested fastener length + an honest note for a BOM line.

    Deterministic where the part itself defines the engagement: a heat-set insert's mating screw =
    part wall + insert engagement; a captive nut's mating bolt = floor + nut thickness. Genuinely
    ambiguous (a plain screw / bolt into an unknown external mate) -> no number, say so. Non-fastener
    lines (bearing / magnet / dowel / T-nut) get nothing.
    """
    if kind == "insert":  # the insert IS the buyable; this is the screw that threads into it
        eng = heat_set_insert(key).length_mm
        if min_wall is None:
            return (
                None,
                f"mating {key} screw: part wall + {eng:g} mm engagement (set to your stack-up)",
            )
        length = float(math.ceil(min_wall + eng))
        return (
            length,
            f"mating {key} screw ~ {length:g} mm = {min_wall:g} mm wall + {eng:g} mm engagement (suggestion)",
        )
    if kind == "nut":  # the captive nut; this is the bolt that threads into it
        thick = hex_nut(key).thickness_mm
        if min_wall is None:
            return None, f"mating {key} bolt: floor + {thick:g} mm nut (set to your stack-up)"
        length = float(math.ceil(min_wall + thick))
        return (
            length,
            f"mating {key} bolt ~ {length:g} mm = {min_wall:g} mm floor + {thick:g} mm nut (suggestion)",
        )
    if kind == "fastener":  # a screw/bolt whose mate is external -> honestly unknown
        return None, _LENGTH_DEPENDS
    return None, None  # bearing / dowel / magnet / T-nut: length is not applicable


def build_bom(program: PartProgram) -> BillOfMaterials:
    """Build the Bill of Materials from a part's declared hardware + face interfaces.

    Aggregates by buyable component identity (a clearance hole + a motor face that both call for an
    M3 socket-head cap screw merge into one line, summed). Pulls each cited ``source`` + provisional
    flag from the registry; flags (never guesses) anything the registry cannot back.
    """
    # component label -> accumulated line data. Insertion order is made deterministic below.
    acc: dict[str, dict[str, object]] = {}
    unmapped: list[str] = []

    def add(
        *, component: str, kind: str, key: str, qty: int, source: str, provisional: bool
    ) -> None:
        entry = acc.get(component)
        if entry is None:
            acc[component] = {
                "component": component,
                "kind": kind,
                "key": key,
                "quantity": qty,
                "source": source,
                "provisional": provisional,
            }
        else:
            entry["quantity"] = int(entry["quantity"]) + qty  # type: ignore[call-overload]

    for feat in program.hardware_features:
        buyable = _BUYABLE.get(feat.category)
        if buyable is None:
            unmapped.append(f"{feat.category} {feat.key} (no buyable component in the registry)")
            continue
        registry = ALL_REGISTRIES.get(buyable.reg_name)
        if registry is None or not registry.has(feat.key):
            unmapped.append(f"{feat.category} {feat.key} (not in the {buyable.reg_name} registry)")
            continue
        spec = registry.get(feat.key)
        add(
            component=buyable.label(feat.key),
            kind=_KIND[feat.category],
            key=feat.key,
            qty=1,
            source=spec.source,
            provisional=spec.provisional,
        )

    _face_lines(program.interfaces, add, unmapped)

    # Order: fasteners, T-nuts, inserts, nuts, bearings, dowels, magnets; then by component label.
    kind_order = {
        "fastener": 0,
        "tnut": 1,
        "insert": 2,
        "nut": 3,
        "bearing": 4,
        "dowel": 5,
        "magnet": 6,
    }
    min_wall = program.constraints.min_wall_thickness
    lines = []
    for e in acc.values():
        kind, key = str(e["kind"]), str(e["key"])
        length_mm, length_note = _length_for(kind, key, min_wall)
        lines.append(
            BomLine(
                component=str(e["component"]),
                kind=kind,
                key=key,
                quantity=int(e["quantity"]),  # type: ignore[call-overload]
                source=str(e["source"]),
                provisional=bool(e["provisional"]),
                length_mm=length_mm,
                length_note=length_note,
            )
        )
    lines.sort(key=lambda line: (kind_order.get(line.kind, 99), line.component))
    return BillOfMaterials(lines=lines, unmapped=unmapped)


__all__ = ["BillOfMaterials", "BomLine", "build_bom"]
