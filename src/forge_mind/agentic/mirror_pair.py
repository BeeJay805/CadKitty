"""Chirality guard for mirror-by-construction (the L/R defect class, measured 2026-07-24).

Three of the four declared mirror pairs in the shipped H3 humanoid are not mirrors: forearm off
by 2.58 mm / 7.43% volume, shin_foot by 25.98 mm / 9.80% (the two legs are different LENGTHS,
120 vs 130 mm), thigh outwardly symmetric to 0.32 mm but 2.49% different in volume. Only the
shoulder pin -- a plain cylinder -- passes. Every mechanism the engine has for this is prose
aimed at the model (``functional.py`` doctrine, ``mech_arch.mirrored_pairs``) or a name-only
sibling check, so the right part is independently re-authored and merely HOPED to match. Full
measurement: ``docs/research/2026-07-24-mirror-pair-defect-class.md``.

The fix is to REFLECT the built left solid instead of authoring the right one, which makes
congruence exact by construction rather than a threshold. Reflecting a mesh is arithmetic. The
engineering content -- and this module -- is knowing when reflection is WRONG.

REFLECTION FLIPS HANDEDNESS. A mirrored right-hand ISO thread becomes a LEFT-HAND thread that no
standard screw will enter; a mirrored helix coils the other way; mirrored text prints backwards.
Each of those survives every visual review and the assembled render, so a mirror lever without
this guard trades a symmetry defect for a worse, less visible one.

WHY THIS IS AN ALLOW-LIST, NOT A DENY-LIST. A deny-list that forgets a handed idiom mirrors a
part it should not -- a false PASS in the very guard meant to prevent one, which is the failure
shape this whole program exists to kill. So a part is mirrored ONLY when every idiom it uses is
positively classified achiral; anything unrecognised refuses. The classification covers the
engine's whole published vocabulary and ``test_every_idiom_is_classified`` fails the build if an
idiom is ever added without a chirality ruling, so completeness is mechanically enforced instead
of remembered.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from forge_mind.agentic.cheatsheet import VERIFIED_IDIOMS
from forge_mind.assembly.joints import Port
from forge_mind.types import Vec3

# Handed: reflection changes what the feature IS or MEANS, not merely where it sits.
CHIRAL_IDIOMS: frozenset[str] = frozenset(
    {
        "external_iso_thread",  # right-hand thread -> left-hand; no standard screw fits
        "internal_iso_thread",  # same, and the mating part is unchanged -> unassemblable
        "helix_coil",  # a coil's winding direction reverses
        "sweep",  # is_frenet sweeps ride a path that may itself be a helix
        "catalog_screw",  # purchased fastener geometry is right-hand by standard
        "grip_loop",  # the detail lane's marking/label surface (text reads backwards)
    }
)

# Achiral: reflection yields the same feature in the mirrored position, which is exactly what a
# left/right pair wants. Straight-tooth spur gears are mirror-symmetric about a plane through the
# axis (helical gears would NOT be, and are not in this vocabulary).
ACHIRAL_IDIOMS: frozenset[str] = frozenset(VERIFIED_IDIOMS) - CHIRAL_IDIOMS

# Handed CONSTRUCTS in authored source, independent of which idiom the author cited: the model
# writes plain build123d, so scanning for the idiom name alone would miss a hand-rolled thread.
_CHIRAL_CONSTRUCTS: tuple[tuple[str, str], ...] = (
    (r"\bIsoThread\b", "IsoThread (a mirrored right-hand thread becomes left-hand)"),
    (r"\bHelix\b", "Helix (a mirrored coil reverses its winding direction)"),
    (r"\bText\b", "Text (mirrored text prints backwards)"),
    (r"\bemboss_text\b", "emboss_text (mirrored text prints backwards)"),
    (r"\bdirection_arrow\b", "direction_arrow (a mirrored arrow points the wrong way)"),
    (r"\bis_frenet\s*=\s*True", "a Frenet sweep (its path may be handed)"),
)


@dataclass(frozen=True)
class MirrorDecision:
    """Whether the right part may be produced by reflecting the left, and why."""

    may_mirror: bool
    reason: str
    blockers: tuple[str, ...] = ()


def chirality_blockers(code: str | None) -> list[str]:
    """Handed constructs found in the authored source that make reflection invalid.

    ``None`` or blank source yields a blocker rather than an empty list: source we cannot read is
    source we cannot clear.
    """
    if code is None or not code.strip():
        return ["the authored source is unavailable, so its chirality cannot be established"]
    return [why for pattern, why in _CHIRAL_CONSTRUCTS if re.search(pattern, code)]


def decide_mirror(
    *, code: str | None, idioms_used: frozenset[str] | set[str] | None = None
) -> MirrorDecision:
    """May the right part be produced by reflecting the built left part?

    Refuses -- never silently mirrors -- when the source is unreadable, when it contains a handed
    construct, or when it cites an idiom that is not positively classified achiral. A refusal is
    not a failure: it means this pair falls back to authoring the right part, which is exactly
    today's behaviour.
    """
    blockers = chirality_blockers(code)
    unknown = sorted(set(idioms_used or set()) - ACHIRAL_IDIOMS - CHIRAL_IDIOMS)
    if unknown:
        blockers.append(
            f"unclassified idiom(s) {unknown} -- chirality unknown, so reflection is not cleared"
        )
    handed = sorted((set(idioms_used or set())) & CHIRAL_IDIOMS)
    if handed:
        blockers.append(f"handed idiom(s) {handed}")
    if blockers:
        return MirrorDecision(
            False,
            "reflection would change what this part IS, not just where it sits; author the "
            "right part instead",
            tuple(blockers),
        )
    return MirrorDecision(
        True, "no handed feature found; reflection makes the pair congruent by construction"
    )


_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


def reflect_port(port: Port, *, mirror_axis: str) -> Port:
    """The same port on the reflected part: origin, axis and normal each mirrored.

    ``mirror_axis`` is a PARAMETER, not a constant. ``mech_arch.mirrored_pairs`` names which
    pairs must mirror but never states the plane ("sagittal" is prose), and freezing the global
    frame convention is held pending founder ratification -- so the caller states the plane, the
    same contract ``validator.symmetry.mirror_congruence`` already uses.

    Origins are points and axis/normal are directions, but a reflection through a coordinate
    plane is linear, so all three transform by negating the same component.

    ON HANDEDNESS, because reflection has det = -1 and this is where a mirror lever would
    normally break: the reflected triad is left-handed, so ``(Ma) x (Mn) = -M(a x b)``. That does
    NOT corrupt placement here -- ``assembly.solver._frame_mate`` builds BOTH frames as
    ``[axis, normal, axis x normal]`` and solves for a proper rotation, so it consumes only the
    two declared directions and derives the third consistently on each side. The mate therefore
    lands correctly on naively reflected ports. This was verified against the solver rather than
    assumed; a consumer that ever reads a DECLARED third axis would need the sign handled.
    """
    idx = _AXIS_INDEX[mirror_axis.lower()]

    def flip(v: Vec3) -> Vec3:
        xyz = list(v.as_tuple())
        xyz[idx] = -xyz[idx]
        return Vec3(x=xyz[0], y=xyz[1], z=xyz[2])

    return port.model_copy(
        update={
            "origin": flip(port.origin),
            "axis": flip(port.axis),
            "normal": flip(port.normal),
        }
    )


def reflect_ports(ports: list[Port], *, mirror_axis: str) -> list[Port]:
    """Every port of a part, reflected. Raises on an unknown axis rather than guessing one."""
    if mirror_axis.lower() not in _AXIS_INDEX:
        raise ValueError(f"unknown mirror_axis {mirror_axis!r} (use x|y|z)")
    return [reflect_port(p, mirror_axis=mirror_axis) for p in ports]


__all__ = [
    "ACHIRAL_IDIOMS",
    "CHIRAL_IDIOMS",
    "MirrorDecision",
    "chirality_blockers",
    "decide_mirror",
    "reflect_port",
    "reflect_ports",
]
