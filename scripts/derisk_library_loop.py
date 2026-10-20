"""Library-builder Step-0 de-risk (deterministic, $0): does the catalog/multiplier loop that
made the 102 adapter families GENERALIZE across other part families?

Takes three families BEYOND the flat-plate adapter -- the L-bracket motor mount (a different
base of the composable_plate family), the motor arm (a different spec + interpreter), and the
computed body / lattice (a third spec + interpreter) -- and runs the SAME loop on each:

    enumerate the categorical axes -> derive dims deterministically (interpret) -> compile ->
    validate ALL FOUR TIERS -> keep iff every check PASSES (report.passed == overall PASS, which
    is strict: a WARN fails it -> no false-PASS) -> else log the reject + the tier that caught it.

Zero LLM calls (pure interpret/compile/validate). If a family can't be enumerated cleanly it is
reported + excluded, never forced.

Run: uv run python scripts/derisk_library_loop.py
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.compiler.compile import CompileError, compile_program
from forge_mind.dsl.computed_ops import FacetedStrutOp, LatticeOp
from forge_mind.dsl.program import PartProgram
from forge_mind.layout.arm_interpreter import ArmSpec, interpret_arm
from forge_mind.layout.composable_interpreter import interpret_composable_plate
from forge_mind.layout.composable_spec import BaseArchetype
from forge_mind.layout.computed_interpreter import ComputedBodySpec, interpret_computed_body
from forge_mind.layout.interpreter import LayoutError
from forge_mind.scaling.enumerate import build_spec, enumerate_shapes
from forge_mind.types import Vec3
from forge_mind.validator.report import CheckStatus
from forge_mind.validator.run import validate

WORK = Path(tempfile.mkdtemp(prefix="library_derisk_"))
_COMPUTED_SIZES = {
    "compact": Vec3(x=80.0, y=24.0, z=24.0),
    "standard": Vec3(x=120.0, y=30.0, z=28.0),
    "large": Vec3(x=160.0, y=40.0, z=32.0),
}

# A "candidate" = (categorical choices, a thunk that derives its PartProgram or raises LayoutError).
Candidate = tuple[dict[str, str], Callable[[], PartProgram]]


def lbracket_candidates() -> Iterator[Candidate]:
    for shape in enumerate_shapes():
        if shape.base is BaseArchetype.L_BRACKET_BEND:
            yield (
                {"key": shape.key},
                lambda s=shape: interpret_composable_plate(  # type: ignore[misc]
                    build_spec(s, name="lb", envelope_mm=Vec3(x=400.0, y=400.0, z=40.0))
                ),
            )


def arm_candidates() -> Iterator[Candidate]:
    for cs in ("hexagonal", "rectangular"):
        for reach in ("compact", "standard", "long"):
            yield (
                {"body": "solid", "cross_section": cs, "reach": reach},
                lambda cs=cs, reach=reach: interpret_arm(  # type: ignore[misc]
                    ArmSpec(
                        name="arm", min_wall_mm=4.0, body="solid", cross_section=cs, reach=reach
                    )
                ),
            )
    for density in ("light", "medium", "dense"):
        for reach in ("compact", "standard", "long"):
            yield (
                {"body": "lattice", "density": density, "reach": reach},
                lambda density=density, reach=reach: interpret_arm(  # type: ignore[misc]
                    ArmSpec(
                        name="arm", min_wall_mm=4.0, body="lattice", density=density, reach=reach
                    )
                ),
            )


def computed_candidates() -> Iterator[Candidate]:
    for cs in ("hexagonal", "rectangular"):
        for size, env in _COMPUTED_SIZES.items():
            yield (
                {"op": "faceted_strut", "cross_section": cs, "size": size},
                lambda env=env, cs=cs: interpret_computed_body(  # type: ignore[misc]
                    ComputedBodySpec(
                        name="cb",
                        envelope_mm=env,
                        min_wall_mm=4.0,
                        op=FacetedStrutOp(cross_section=cs),
                    )
                ),
            )
    for density in ("light", "medium", "dense"):
        for size, env in _COMPUTED_SIZES.items():
            yield (
                {"op": "lattice", "density": density, "size": size},
                lambda env=env, density=density: interpret_computed_body(  # type: ignore[misc]
                    ComputedBodySpec(
                        name="cb",
                        envelope_mm=env,
                        min_wall_mm=4.0,
                        op=LatticeOp(cell="grid", density=density),
                    )
                ),
            )


@dataclass
class Tally:
    attempted: int = 0
    passed: int = 0
    rejects: list[tuple[str, str]] | None = None  # (tier, detail)

    def __post_init__(self) -> None:
        self.rejects = self.rejects or []


def attempt(
    family: str, idx: int, choices: dict[str, str], make: Callable[[], PartProgram], tally: Tally
) -> None:
    tally.attempted += 1
    label = ",".join(f"{k}={v}" for k, v in choices.items())
    try:
        program = make()
    except LayoutError as e:
        tally.rejects.append(("derive", f"{label}: {e}"))  # type: ignore[union-attr]
        return
    try:
        model = compile_program(program, WORK / f"{family}_{idx}")
    except (CompileError, Exception) as e:
        tally.rejects.append(("compile", f"{label}: {type(e).__name__}"))  # type: ignore[union-attr]
        return
    report = validate(model, program, material="PLA")
    if (
        report.passed
    ):  # overall == PASS -> every check PASSED (a WARN would fail this): no false-PASS
        tally.passed += 1
        return
    bad = next(c for c in report.checks if c.status != CheckStatus.PASS)
    tally.rejects.append((bad.tier.value, f"{label}: {bad.id}={bad.status.value}"))  # type: ignore[union-attr]


def run_family(family: str, candidates: Iterator[Candidate]) -> Tally:
    tally = Tally()
    for idx, (choices, make) in enumerate(candidates):
        attempt(family, idx, choices, make, tally)
    return tally


def main() -> None:
    families = {
        "composable_plate/l_bracket": lbracket_candidates(),
        "motor_arm": arm_candidates(),
        "computed_body": computed_candidates(),
    }
    print("=== Library loop generalization de-risk (offline, $0) ===")
    all_ok = True
    for family, cands in families.items():
        t = run_family(family, cands)
        lb = wilson_lower_bound(t.passed, t.attempted) if t.attempted else 0.0
        print(
            f"\n[{family}] enumerated={t.attempted}  passed(all 4 tiers)={t.passed}  "
            f"rejected={len(t.rejects or [])}  Wilson95LB={lb * 100:.1f}%"
        )
        for tier, detail in t.rejects or []:
            print(f"    REJECT @{tier}: {detail}")
        all_ok = all_ok and t.passed > 0
    print(
        f"\n{'GENERALIZES — every family enumerated + produced validated parts, $0' if all_ok else '*** a family produced no passes'}"
    )
    print("  (zero LLM calls: pure interpret -> compile -> validate)")


if __name__ == "__main__":
    main()
