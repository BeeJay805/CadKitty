"""Intent-fidelity gate — no-false-PASS at the INTENT level.

The composition LLM reads a Blueprint and authors a spec; the deterministic interpreter then builds
a part that the geometry validator happily PASSES. But "geometrically valid" is not "what was asked":
a strut routed to ``computed_body`` cannot host the heat-set inserts the user requested, so they are
silently DROPPED; or the LLM mis-authors a bearing seat as a clearance hole, a silent MISMAP. Either
way the part validates -- a SILENT WRONG PART.

This gate compares the Blueprint's ``required_interfaces`` (the structured contract, made faithful by
the Step-1 vocabulary fix) against what the PartProgram ACTUALLY built (face ``InterfaceDeclaration``s
+ registry ``HardwareFeature``s). If a requested interface KIND is missing from the built part it was
either dropped or mismapped to a different type -> REFUSE, rather than ship the wrong part.

Conservative by design (a false refusal is acceptable; a silent wrong part is forbidden):
  - comparison is at the KIND level (a bearing seat vs a clearance hole), the granularity the task
    cares about ("dropped or mismapped to a different TYPE"); component-id and QUANTITY fidelity
    (608 vs 625; four holes vs two) are deliberately OUT of scope here -- a coarser kind match keeps
    known-good compositions passing while still catching every drop / type-mismap;
  - face/board refs are coalesced to a coarse kind (pi4_mount + pi4_ports -> "pi4") so a tray that
    legitimately omits the conditional ports interface is not false-refused.
"""

from __future__ import annotations

from forge_mind.blueprint.schema import Blueprint
from forge_mind.dsl.program import PartProgram

# Every vocabulary's spelling -> ONE canonical feature kind, so a requested kind and a built kind are
# compared on equal terms regardless of which enum named them. Covers RequestedInterface (blueprint),
# InterfaceType (program face declarations), and HardwareFeature.category (program hardware).
_CANONICAL: dict[str, str] = {
    # Face / board (RequestedInterface + declaration.InterfaceType spellings).
    "nema17_face": "motor",
    "fan_face": "fan",
    "extrusion_2020": "extrusion",
    "extrusion_2020_slot": "extrusion",
    "m5_tnut": "extrusion",
    "pi4_mount": "pi4",
    "pi4_ports": "pi4",
    # Hardware (RequestedInterface + HardwareFeature.category spellings).
    "clearance_hole": "clearance_hole",
    "counterbore": "counterbore",
    "heat_set_insert": "heat_set",
    "heat_set_boss": "heat_set",
    "captive_nut_pocket": "captive_nut",
    "bearing_seat": "bearing",
    "bearing_od": "bearing",
    "dowel_hole": "dowel",
    "magnet_pocket": "magnet",
    "tapped_hole": "tapped_hole",
}

# Human-readable label per canonical kind, for the refusal message.
_LABEL: dict[str, str] = {
    "motor": "a NEMA 17 motor face",
    "fan": "a fan mount",
    "extrusion": "an extrusion mount",
    "pi4": "a Raspberry Pi mount",
    "clearance_hole": "clearance hole(s)",
    "counterbore": "counterbored hole(s)",
    "heat_set": "heat-set insert(s)",
    "captive_nut": "captive nut pocket(s)",
    "bearing": "a bearing seat",
    "dowel": "dowel hole(s)",
    "magnet": "magnet pocket(s)",
    "tapped_hole": "tapped hole(s)",
}


def _canon(name: str) -> str:
    """Canonical kind for a vocabulary spelling. Falls back to the raw value (NOT a silent drop): an
    unmapped requested kind then only matches an identically-spelled built kind -- conservative."""
    return _CANONICAL.get(name, name)


def _label(kind: str) -> str:
    return _LABEL.get(kind, kind)


def requested_kinds(blueprint: Blueprint) -> set[str]:
    """The set of interface KINDS the Blueprint says the part must present."""
    return {_canon(i.interface_type.value) for i in blueprint.required_interfaces}


def built_kinds(program: PartProgram) -> set[str]:
    """The set of interface KINDS the PartProgram actually built: face declarations + hardware."""
    kinds = {_canon(d.interface_type.value) for d in program.interfaces}
    kinds |= {_canon(f.category) for f in program.hardware_features}
    return kinds


def check_fidelity(blueprint: Blueprint, program: PartProgram) -> str | None:
    """Refusal message if the built part DROPPED or MISMAPPED a requested interface, else None.

    The single no-false-PASS-at-the-intent-level check: every kind the Blueprint required must appear
    in what the part built. A missing kind means it was dropped (a path that can't host it) or
    mismapped to a different type -- refuse rather than ship a silently-wrong part.
    """
    missing = requested_kinds(blueprint) - built_kinds(program)
    if not missing:
        return None
    wanted = ", ".join(sorted(_label(k) for k in missing))
    built = built_kinds(program)
    built_desc = (
        ", ".join(sorted(_label(k) for k in built)) if built else "a bare body with no interfaces"
    )
    return (
        f"You asked for {wanted}, but I couldn't place that on the part I generated "
        f"(built: {built_desc}). I won't ship a part that's missing a feature you requested."
    )


__all__ = ["built_kinds", "check_fidelity", "requested_kinds"]
