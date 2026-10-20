"""Composition engine — deterministically enumerate base x interface-subset x placement and keep
only the parts that pass EVERY check clean.

The breadth multiplier: the composable plate now hosts the full standards-backed hardware
vocabulary, so a base body composes with subsets of {clearance, counterbore, heat-set boss,
captive nut, bearing seat, dowel, magnet} at registry sizes, plus the legacy face mounts. Every
dimension comes from ``dsl.standards`` (the LLM/grammar picks categoricals only).

Keep gate (a part enters ONLY if all clean -- no-false-PASS):
  derive (incl. feature-collision / min-wall-between-features) -> compile -> the library builder's
  robust MULTI-SEED 4-tier validation (deterministic) -> the A4 hardware conformance check.

Material is a recorded axis (PLA/PETG/ABS mass per geometry); geometry is enumerated once.
``SHARDS`` partitions the space so it can be enumerated in parallel and merged.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.layout.composable_hardware import hardware_feature_of
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import (
    BaseArchetype,
    ComposablePlateSpec,
    Face,
    InterfacePlacement,
    InterfaceRef,
)
from forge_mind.layout.interpreter import LayoutError
from forge_mind.library.builder import _validate_clean
from forge_mind.types import Vec3
from forge_mind.validator.conformance import HardwareFeature, check_conformance
from forge_mind.validator.metrics import density_for
from forge_mind.validator.report import CheckStatus

_ENVELOPE = Vec3(
    x=300.0, y=200.0, z=120.0
)  # generous; the interpreter derives the tight plate within it
_MIN_WALL_MM = 4.0
_MATERIALS = ("PLA", "PETG", "ABS")  # recorded mass variants per geometry (same STL)

_R = InterfaceRef
# Hardware vocabulary: (ref, key, fit) tuples. Fits matter geometrically where they change a hole.
_CLEARANCE = [
    (_R.CLEARANCE_HOLE, k, f)
    for k in ("M2", "M2.5", "M3", "M4", "M5", "M6", "M8")
    for f in ("close", "normal")
]
_COUNTERBORE = [(_R.COUNTERBORE, k, None) for k in ("M3", "M4", "M5", "M6", "M8")]
_DOWEL = [(_R.DOWEL_HOLE, k, f) for k in ("2", "3", "4", "5", "6", "8") for f in ("press", "slip")]
_HEATSET = [(_R.HEAT_SET_BOSS, k, None) for k in ("M2", "M2.5", "M3", "M4", "M5")]
_BEARING = [
    (_R.BEARING_SEAT, k, f)
    for k in ("623", "625", "626", "608", "688", "6800", "6900", "6000")
    for f in ("press", "slip")
]
_MAGNET = [(_R.MAGNET_POCKET, k, None) for k in ("5x2", "6x3", "8x3", "10x3")]
_NUT = [(_R.CAPTIVE_NUT_POCKET, k, None) for k in ("M3", "M4", "M5", "M6", "M8")]

# Curated shortlists for pairs (keep the pair space meaningful + bounded).
# fan_60mm (not fan_40mm): a 40 mm fan's airflow bore eats its bolt land at a 4 mm wall (the
# ADR-0011 bound) -> always infeasible here; the 60 mm fan composes cleanly at min_wall 4.
_FACES = [
    (_R.NEMA17_FACE, None, None),
    (_R.FAN_FACE, "fan_60mm", None),
    (_R.EXTRUSION_2020, None, None),
]
_HW_SHORTLIST = [
    (_R.CLEARANCE_HOLE, "M3", "normal"),
    (_R.HEAT_SET_BOSS, "M3", None),
    (_R.DOWEL_HOLE, "4", "press"),
    (_R.BEARING_SEAT, "608", "press"),
    (_R.CAPTIVE_NUT_POCKET, "M4", None),
    (_R.MAGNET_POCKET, "6x3", None),
    (_R.COUNTERBORE, "M5", None),
    (_R.BEARING_SEAT, "625", "press"),
]

# Every hardware-hosting base (the walled bases + the round plate) reuses the SAME hardware
# vocabulary as the flat plate (their floor IS a flat/circular plate), so each is a catalog
# multiplier: <name>_singles over every hardware single + <name>_pairs over the hardware-shortlist
# pairs. Shard name -> base archetype value (data, not code -- a new base adds two entries here).
_WALLED_SINGLE_SHARDS: dict[str, str] = {
    "box_singles": "open_box",
    "bracket_singles": "angle_bracket",
    "uchannel_singles": "u_channel",
    "tray_singles": "tray",
    "partial_singles": "partial_enclosure",
    "round_singles": "round_plate",
}
_WALLED_PAIR_SHARDS: dict[str, str] = {
    "box_pairs": "open_box",
    "bracket_pairs": "angle_bracket",
    "uchannel_pairs": "u_channel",
    "tray_pairs": "tray",
    "partial_pairs": "partial_enclosure",
    "round_pairs": "round_plate",
}

SHARDS = ("flat_singles", "flat_pairs", "lbracket", *_WALLED_SINGLE_SHARDS, *_WALLED_PAIR_SHARDS)


@dataclass(frozen=True)
class FeatureChoice:
    ref: str
    key: str | None = None
    fit: str | None = None

    def slug(self) -> str:
        return "+".join(p for p in (self.ref, self.key, self.fit) if p)


@dataclass(frozen=True)
class _Recipe:
    base: str
    features: tuple[FeatureChoice, ...]

    @property
    def cid(self) -> str:
        return f"{self.base}__" + "__".join(f.slug() for f in self.features)

    def placements(self) -> list[InterfacePlacement]:
        return [
            InterfacePlacement(
                ref=InterfaceRef(f.ref), face=Face.PLATE_TOP, component_id=f.key, fit=f.fit
            )
            for f in self.features
        ]


def _t(triple: tuple[InterfaceRef, str | None, str | None]) -> FeatureChoice:
    return FeatureChoice(ref=triple[0].value, key=triple[1], fit=triple[2])


def enumerate_recipes(shard: str) -> Iterator[_Recipe]:
    """Deterministic, order-stable enumeration for a shard."""
    if shard == "flat_singles":
        for triple in itertools.chain(
            _CLEARANCE, _COUNTERBORE, _DOWEL, _HEATSET, _BEARING, _MAGNET, _NUT
        ):
            yield _Recipe("flat_plate", (_t(triple),))
    elif shard == "flat_pairs":
        for face in _FACES:  # a face mount + an extra standards feature
            for hw in _HW_SHORTLIST:
                yield _Recipe("flat_plate", (_t(face), _t(hw)))
        for a, b in itertools.combinations(_HW_SHORTLIST, 2):  # two standards features
            yield _Recipe("flat_plate", (_t(a), _t(b)))
    elif shard == "lbracket":
        # The l_bracket base composes exactly its proven NEMA 17 + 2020 combo.
        yield _Recipe(
            "l_bracket_bend",
            (_t((_R.NEMA17_FACE, None, None)), _t((_R.EXTRUSION_2020, None, None))),
        )
    elif shard in _WALLED_SINGLE_SHARDS:
        # Any walled base hosts the SAME hardware vocabulary as the flat plate (its floor IS a flat
        # plate), so each base is a catalog multiplier over the hardware singles.
        base = _WALLED_SINGLE_SHARDS[shard]
        for triple in itertools.chain(
            _CLEARANCE, _COUNTERBORE, _DOWEL, _HEATSET, _BEARING, _MAGNET, _NUT
        ):
            yield _Recipe(base, (_t(triple),))
    elif shard in _WALLED_PAIR_SHARDS:
        base = _WALLED_PAIR_SHARDS[shard]
        for a, b in itertools.combinations(_HW_SHORTLIST, 2):  # two standards features on the floor
            yield _Recipe(base, (_t(a), _t(b)))
    else:
        raise ValueError(f"unknown shard {shard!r}; known: {SHARDS}")


@dataclass(frozen=True)
class CompositionPart:
    id: str
    base: str
    features: list[dict[str, str | None]]
    dims_mm: dict[str, float]
    material_mass_g: dict[str, float]
    tiers_passed: list[str]
    conformance: list[str]  # "<category>:<key>=pass" per hardware feature
    stl_path: str


@dataclass(frozen=True)
class CompositionReject:
    id: str
    base: str
    features: list[dict[str, str | None]]
    stage: str  # derive | compile | <tier> | conformance
    detail: str


@dataclass
class CompositionManifest:
    parts: list[CompositionPart] = field(default_factory=list)
    rejects: list[CompositionReject] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "total_parts": len(self.parts),
            "total_rejects": len(self.rejects),
            "parts": [asdict(p) for p in self.parts],
            "rejects": [asdict(r) for r in self.rejects],
        }


def _features_dict(recipe: _Recipe) -> list[dict[str, str | None]]:
    return [{"ref": f.ref, "key": f.key, "fit": f.fit} for f in recipe.features]


def _conformance(recipe: _Recipe, placements: list[InterfacePlacement]) -> tuple[bool, list[str]]:
    """Run the A4 conformance check on every hardware placement; (all_clean, summary)."""
    from forge_mind.layout.composable_hardware import HARDWARE_REFS

    results: list[str] = []
    clean = True
    for p in placements:
        if p.ref not in HARDWARE_REFS:
            continue
        cat, key, fit, nominal = hardware_feature_of(p)
        check = check_conformance(
            HardwareFeature(category=cat, key=key, fit=fit, nominal_mm=nominal)
        )
        results.append(f"{cat}:{key}={check.status.value}")
        if check.status != CheckStatus.PASS:
            clean = False
    return clean, results


def _attempt(
    recipe: _Recipe, stl_dir: Path
) -> tuple[CompositionPart | None, CompositionReject | None]:
    fdict = _features_dict(recipe)
    base = BaseArchetype(recipe.base)
    placements = recipe.placements()
    spec = ComposablePlateSpec(
        name=recipe.cid[:120],
        envelope_mm=_ENVELOPE,
        material="PLA",
        min_wall_mm=_MIN_WALL_MM,
        base=base,
        interfaces=placements,
    )
    try:
        program = interpret_composable_plate(spec)  # collision + no-spec rejected here
    except LayoutError as exc:
        return None, CompositionReject(recipe.cid, recipe.base, fdict, "derive", str(exc)[:200])
    out = stl_dir / recipe.base / recipe.cid[:120]
    try:
        model = compile_program(program, out)
    except Exception as exc:
        return None, CompositionReject(
            recipe.cid, recipe.base, fdict, "compile", f"{type(exc).__name__}: {exc}"[:200]
        )
    report, failing = _validate_clean(model, program)  # robust multi-seed 4-tier
    if report is None:
        bad = next(c for c in failing.checks if c.status != CheckStatus.PASS)
        return None, CompositionReject(
            recipe.cid, recipe.base, fdict, bad.tier.value, f"{bad.id}={bad.status.value}"
        )
    conform_ok, conform = _conformance(recipe, placements)
    if not conform_ok:
        return None, CompositionReject(
            recipe.cid, recipe.base, fdict, "conformance", ";".join(conform)
        )

    bb = model.bounding_box
    dims = {
        "x": round(bb.max.x - bb.min.x, 3),
        "y": round(bb.max.y - bb.min.y, 3),
        "z": round(bb.max.z - bb.min.z, 3),
    }
    vol = float(model.part.volume)
    masses = {m: round(vol / 1000.0 * density_for(m), 3) for m in _MATERIALS}
    part = CompositionPart(
        id=recipe.cid,
        base=recipe.base,
        features=fdict,
        dims_mm=dims,
        material_mass_g=masses,
        tiers_passed=sorted(t.value for t in report.tiers),
        conformance=conform,
        stl_path=model.stl_path.relative_to(stl_dir).as_posix(),
    )
    return part, None


def build_compositions(stl_dir: Path, shards: tuple[str, ...] = SHARDS) -> CompositionManifest:
    manifest = CompositionManifest()
    for shard in shards:
        for recipe in enumerate_recipes(shard):
            part, reject = _attempt(recipe, stl_dir)
            if part is not None:
                manifest.parts.append(part)
            elif reject is not None:
                manifest.rejects.append(reject)
    return manifest


__all__ = [
    "SHARDS",
    "CompositionManifest",
    "CompositionPart",
    "CompositionReject",
    "FeatureChoice",
    "build_compositions",
    "enumerate_recipes",
]
