"""Deterministic autonomous library builder.

Generalizes the 102-family adapter catalog loop (ADR-0016/0017) across EVERY enumerable part
family. For each family it enumerates the categorical space, derives dimensions deterministically
(the family interpreter), compiles, validates ALL FOUR TIERS, keeps only the clean passes, and
records the rejects with the tier that caught them. Offline + deterministic ($0, no LLM): the
SAME categorical menu + code-derives-geometry thesis, run exhaustively.

NO-FALSE-PASS is the whole point: a part enters the manifest ONLY if ``report.passed`` -- which is
``overall == PASS``, i.e. every check PASSED (a WARN makes overall WARN, so a WARN is NOT a pass).
The keep gate is never relaxed; a part that would need a relaxed check is excluded and logged.

Families covered (PART families validated by the 4-tier part validator):
  - composable_plate : flat-plate adapters/mounts/fans (the 102 set) + the l_bracket combo,
    enumerated by motor x extrusion x fan component id (the registry multiplier).
  - computed_body    : faceted_strut (x cross_section) + lattice (x density), x a size class.
  - motor_arm        : solid (x cross_section) + lattice (x density), x reach class.
  - electronics_tray : board x ventilation (ports tied to the board, lid none).
ASSEMBLY families (hinge/slide/linkage/actuated arm) are EXCLUDED: they are multi-body mechanisms
validated by ``validate_assembly`` (a moving joint + fits), not the 4-tier part validator -- a
different validation contract, out of scope for a single-part library.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Literal

from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.compiler.compile import CADModel, compile_program
from forge_mind.dsl.computed_ops import FacetedStrutOp, LatticeOp
from forge_mind.dsl.program import PartProgram
from forge_mind.layout.arm_interpreter import ArmSpec
from forge_mind.layout.computed_interpreter import ComputedBodySpec
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import LayoutSpec, interpret_layout
from forge_mind.layout.pi4_spec import ElectronicsTraySpec, LidPolicy, PortPolicy, VentilationPolicy
from forge_mind.scaling.enumerate import FamilyShape, build_spec, enumerate_shapes
from forge_mind.types import Vec3
from forge_mind.validator import feature_checks
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.validator.run import validate

# The min_wall tier samples the surface with the global np.random RNG, so a single validate() is
# stochastic and can flip a part whose wall sits exactly at min_wall. For a DETERMINISTIC,
# NO-FALSE-PASS library we validate under K fixed seeds (driving feature_checks.MIN_WALL_SAMPLE_SEED)
# and keep ONLY if EVERY seeding passes clean: a part that does not robustly pass is rejected (a safe
# false-FAIL), never admitted on a lucky sample. K independent coverages also widen min_wall sampling
# K-fold. The validator and the part geometry are untouched (byte-equivalence intact); we only drive
# the sampler seed and demand robustness.
_VALIDATION_SEEDS = (0, 1, 2)

_MIN_WALL_MM = 4.0
_MAX_OVERHANG_DEG = 45.0  # the spec default, passed explicitly (geometry-identical to omitting it)
_MATERIAL = "PLA"
_PROBE_ENVELOPE = Vec3(
    x=400.0, y=400.0, z=40.0
)  # generous; the composable interpreter derives within it
_TRAY_ENVELOPE = Vec3(x=110.0, y=80.0, z=30.0)
_TRAY_MIN_WALL_MM = (
    2.0  # the M2.5/M3 standoff land is ~2.1-2.4 mm; a thicker wall would false-FAIL it
)
# Coarse size class -> envelope for the computed-body family (categorical -> derived, like reach).
_COMPUTED_SIZES: dict[str, Vec3] = {
    "compact": Vec3(x=80.0, y=24.0, z=24.0),
    "standard": Vec3(x=120.0, y=30.0, z=28.0),
    "large": Vec3(x=160.0, y=40.0, z=32.0),
}

# Typed categorical axes (Literal tuples so iteration preserves the spec field types under mypy).
_CROSS: tuple[Literal["hexagonal", "rectangular"], ...] = ("hexagonal", "rectangular")
_REACH: tuple[Literal["compact", "standard", "long"], ...] = ("compact", "standard", "long")
_DENSITY: tuple[Literal["light", "medium", "dense"], ...] = ("light", "medium", "dense")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:80]


@dataclass(frozen=True)
class Candidate:
    """One enumerated categorical point: its stable id, the choices, and a thunk that builds its
    LayoutSpec (deferred so a derive-time LayoutError is caught uniformly by the build loop)."""

    family: str
    cid: str
    choices: dict[str, str]
    make: Callable[[], LayoutSpec]


# --- spec factories (named, so partial() binds them with clean typing) ----------------------
def _composable_spec(shape: FamilyShape) -> LayoutSpec:
    return build_spec(
        shape, name=_slug(shape.key), envelope_mm=_PROBE_ENVELOPE, min_wall_mm=_MIN_WALL_MM
    )


def _faceted_spec(env: Vec3, cross_section: Literal["hexagonal", "rectangular"]) -> LayoutSpec:
    return ComputedBodySpec(
        name="cb",
        envelope_mm=env,
        min_wall_mm=_MIN_WALL_MM,
        max_overhang_deg=_MAX_OVERHANG_DEG,
        op=FacetedStrutOp(cross_section=cross_section),
    )


def _lattice_body_spec(env: Vec3, density: Literal["light", "medium", "dense"]) -> LayoutSpec:
    return ComputedBodySpec(
        name="cb",
        envelope_mm=env,
        min_wall_mm=_MIN_WALL_MM,
        max_overhang_deg=_MAX_OVERHANG_DEG,
        op=LatticeOp(cell="grid", density=density),
    )


def _solid_arm_spec(
    cross_section: Literal["hexagonal", "rectangular"],
    reach: Literal["compact", "standard", "long"],
) -> LayoutSpec:
    return ArmSpec(
        name="arm",
        min_wall_mm=_MIN_WALL_MM,
        max_overhang_deg=_MAX_OVERHANG_DEG,
        body="solid",
        cross_section=cross_section,
        reach=reach,
    )


def _lattice_arm_spec(
    density: Literal["light", "medium", "dense"], reach: Literal["compact", "standard", "long"]
) -> LayoutSpec:
    return ArmSpec(
        name="arm",
        min_wall_mm=_MIN_WALL_MM,
        max_overhang_deg=_MAX_OVERHANG_DEG,
        body="lattice",
        density=density,
        reach=reach,
    )


def _tray_spec(board: str, ports: PortPolicy, vent: VentilationPolicy) -> LayoutSpec:
    return ElectronicsTraySpec(
        name="tray",
        envelope_mm=_TRAY_ENVELOPE,
        material=_MATERIAL,
        min_wall_mm=_TRAY_MIN_WALL_MM,
        clearance_mm=2.0,
        board_component_id=board,
        ports=ports,
        ventilation=vent,
        lid=LidPolicy.NONE,
    )


# --- per-family enumerators -----------------------------------------------------------------
def _enum_composable() -> Iterator[Candidate]:
    for shape in (
        enumerate_shapes()
    ):  # 137 shapes: flat_plate singles+pairs (motor x ext x fan) + the l_bracket combo
        choices = {
            "base": shape.base.value,
            "placements": "+".join(p.label() for p in shape.placements),
        }
        yield Candidate("composable_plate", shape.key, choices, partial(_composable_spec, shape))


def _enum_computed() -> Iterator[Candidate]:
    for cross_section in _CROSS:
        for size, env in _COMPUTED_SIZES.items():
            yield Candidate(
                "computed_body",
                f"faceted-{cross_section}-{size}",
                {"op": "faceted_strut", "cross_section": cross_section, "size": size},
                partial(_faceted_spec, env, cross_section),
            )
    for density in _DENSITY:
        for size, env in _COMPUTED_SIZES.items():
            yield Candidate(
                "computed_body",
                f"lattice-{density}-{size}",
                {"op": "lattice", "cell": "grid", "density": density, "size": size},
                partial(_lattice_body_spec, env, density),
            )


def _enum_arm() -> Iterator[Candidate]:
    for cross_section in _CROSS:
        for reach in _REACH:
            yield Candidate(
                "motor_arm",
                f"solid-{cross_section}-{reach}",
                {"body": "solid", "cross_section": cross_section, "reach": reach},
                partial(_solid_arm_spec, cross_section, reach),
            )
    for density in _DENSITY:
        for reach in _REACH:
            yield Candidate(
                "motor_arm",
                f"lattice-{density}-{reach}",
                {"body": "lattice", "density": density, "reach": reach},
                partial(_lattice_arm_spec, density, reach),
            )


def _enum_tray() -> Iterator[Candidate]:
    for board in ("raspberry_pi_4", "generic_rect_pcb"):
        ports = PortPolicy.STANDARD if board == "raspberry_pi_4" else PortPolicy.NONE
        for vent in (VentilationPolicy.NONE, VentilationPolicy.SLOTS, VentilationPolicy.GRID):
            yield Candidate(
                "electronics_tray",
                f"{board}-{vent.value}",
                {"board": board, "ports": ports.value, "ventilation": vent.value, "lid": "none"},
                partial(_tray_spec, board, ports, vent),
            )


FAMILIES: dict[str, Callable[[], Iterator[Candidate]]] = {
    "composable_plate": _enum_composable,
    "computed_body": _enum_computed,
    "motor_arm": _enum_arm,
    "electronics_tray": _enum_tray,
}


# --- manifest records -----------------------------------------------------------------------
@dataclass(frozen=True)
class LibraryPart:
    id: str
    family: str
    choices: dict[str, str]
    dims_mm: dict[str, float]
    mass_g: float
    tiers_passed: list[str]
    stl_path: str


@dataclass(frozen=True)
class LibraryReject:
    id: str
    family: str
    choices: dict[str, str]
    tier: str  # derive | compile | mesh | features | printability | interfaces
    detail: str


@dataclass(frozen=True)
class FamilyStat:
    family: str
    attempted: int
    passed: int
    rejected: int
    wilson95_lb: float


@dataclass
class LibraryManifest:
    parts: list[LibraryPart] = field(default_factory=list)
    rejects: list[LibraryReject] = field(default_factory=list)
    families: list[FamilyStat] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "families": [asdict(s) for s in self.families],
            "parts": [asdict(p) for p in self.parts],
        }

    def rejects_to_dict(self) -> dict[str, object]:
        return {"total": len(self.rejects), "rejects": [asdict(r) for r in self.rejects]}


def _validate_clean(
    model: CADModel, program: PartProgram
) -> tuple[ValidationReport | None, ValidationReport]:
    """Robust, DETERMINISTIC keep gate. Validate under each fixed min_wall-sampler seed; return
    (report, report) iff EVERY seeding is ``report.passed`` (overall == PASS, so a WARN fails it too),
    else (None, the failing report). The sampler seed is saved/restored so the build stays a pure
    function (feature_checks itself saves/restores the global RNG state around each sample)."""
    saved_seed = feature_checks.MIN_WALL_SAMPLE_SEED
    try:
        last: ValidationReport | None = None
        for seed in _VALIDATION_SEEDS:
            feature_checks.MIN_WALL_SAMPLE_SEED = seed
            last = validate(model, program, material=_MATERIAL)
            if not last.passed:
                return None, last
        assert last is not None
        return last, last
    finally:
        feature_checks.MIN_WALL_SAMPLE_SEED = saved_seed


def _attempt(cand: Candidate, stl_dir: Path) -> tuple[LibraryPart | None, LibraryReject | None]:
    """Derive -> compile -> validate one candidate. Returns (part, None) iff it passed all four
    tiers clean under EVERY validation seed (no-false-PASS), else (None, reject) tagged with the
    tier that caught it."""
    pid = f"{cand.family}::{cand.cid}"
    try:
        program = interpret_layout(
            cand.make()
        )  # build_spec + interpret; either may raise LayoutError
    except LayoutError as exc:
        return None, LibraryReject(pid, cand.family, cand.choices, "derive", str(exc)[:200])
    out = stl_dir / cand.family / _slug(cand.cid)
    try:
        model = compile_program(program, out)
    except Exception as exc:  # a disconnected/multi-body result can raise on STL export
        return None, LibraryReject(
            pid, cand.family, cand.choices, "compile", f"{type(exc).__name__}: {exc}"[:200]
        )
    report, failing = _validate_clean(model, program)
    if report is None:  # a FAIL or WARN under some seed -> never admitted (no false-PASS)
        bad = next(c for c in failing.checks if c.status != CheckStatus.PASS)
        return None, LibraryReject(
            pid, cand.family, cand.choices, bad.tier.value, f"{bad.id}={bad.status.value}"
        )
    bb = model.bounding_box
    dims = {
        "x": round(bb.max.x - bb.min.x, 3),
        "y": round(bb.max.y - bb.min.y, 3),
        "z": round(bb.max.z - bb.min.z, 3),
    }
    mass = round(report.metrics.mass_g, 3) if report.metrics else 0.0
    tiers = sorted(t.value for t in report.tiers)
    stl_rel = model.stl_path.relative_to(stl_dir).as_posix()
    return LibraryPart(pid, cand.family, cand.choices, dims, mass, tiers, stl_rel), None


def build_library(stl_dir: Path, families: list[str] | None = None) -> LibraryManifest:
    """Run the full deterministic loop across families, writing each kept part's STL under
    ``stl_dir/<family>/<cid>/`` and returning the indexed manifest + reject log + per-family
    Wilson-LB. Order-stable + re-runnable: same input -> byte-identical manifest."""
    names = families if families is not None else list(FAMILIES)
    manifest = LibraryManifest()
    for fam in names:
        attempted = passed = 0
        for cand in FAMILIES[fam]():
            attempted += 1
            part, reject = _attempt(cand, stl_dir)
            if part is not None:
                manifest.parts.append(part)
                passed += 1
            elif reject is not None:
                manifest.rejects.append(reject)
        manifest.families.append(
            FamilyStat(
                fam,
                attempted,
                passed,
                attempted - passed,
                round(wilson_lower_bound(passed, attempted), 4) if attempted else 0.0,
            )
        )
    return manifest


__all__ = [
    "FAMILIES",
    "Candidate",
    "FamilyStat",
    "LibraryManifest",
    "LibraryPart",
    "LibraryReject",
    "build_library",
]
