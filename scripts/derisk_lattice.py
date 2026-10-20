"""LATTICE Step-0 de-risk (deterministic, $0): does a straight-strut lattice VALIDATE, and is
it LIGHTER than a solid of the same envelope?

The premise to prove BEFORE building the op (ADR-0019 deferred lattice "until the SDF min_wall
because the conservative min_wall over-rejects thin/curved geometry"). The escape: a
VERTICAL-WALLED cellular grid. Members are straight vertical wall-boxes unioned into one body;
the cells are vertical through-channels. Why this is validator-ready TODAY with the conservative
checks:

  - overhang: EVERY member face is vertical (the walls extrude full height along the print Z),
    so there are NO downward-facing faces above the bed. Overhang = 0 deg. (A naive 3D strut
    grid FAILS here -- its interior horizontal members have flat downward undersides that are
    not concave bridgeable holes. The spike's flat grid only passed because it lay ON the bed.)
  - min_wall: walls are axis-aligned boxes meeting at 90 deg (obtuse, no acute slivers), so the
    ray-cast reads the true wall thickness t exactly. We DERIVE t >= min_wall, so it is safe by
    construction.
  - single_body + watertight: the unioned walls form one connected watertight slab with
    vertical through-cells (a waffle / nut-plate), exactly like the spike's lattice boolean.
  - mass: wall volume << envelope volume -> LIGHTER than solid (the whole point of a lattice).

This hand-authors the SAME Box-list geometry _derive_lattice will emit in Step 1 (reusing the
existing Box primitive + the compiler's union -- no new primitive earned, ADR-0006), so it
de-risks the exact production geometry. HARD STOP if a reasonable-density lattice can't hold
its struts >= min_wall.

Run: uv run python scripts/derisk_lattice.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from forge_mind.compiler.compile import compile_program
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.types import Vec3
from forge_mind.validator.metrics import compute_physics
from forge_mind.validator.run import validate

# density -> target cell interior size as a multiple of the wall thickness (the mass lever).
_DENSITY_CELL_FACTOR = {"light": 6.0, "medium": 4.0, "dense": 2.5}
_MIN_CELL_FACTOR = 1.0  # a real opening must be at least this * t wide, else infeasible


def derive_counts(span: float, t: float, density: str) -> int:
    """Number of cells along a lateral span for a wall thickness t and coarse density."""
    target_cell = _DENSITY_CELL_FACTOR[density] * t
    n = max(1, round((span - t) / (t + target_cell)))
    # Shrink until the resulting cell interior is a real opening (>= _MIN_CELL_FACTOR * t).
    while n > 1 and (span - t * (n + 1)) / n < _MIN_CELL_FACTOR * t:
        n -= 1
    return n


def lattice_walls(env: Vec3, t: float, nx: int, ny: int) -> list[Box]:
    """A grid of straight vertical wall-members spanning the X-Y envelope, full height along Z.

    (nx+1) walls perpendicular to X + (ny+1) walls perpendicular to Y, outer walls flush with
    the envelope. Unioned by the compiler -> one waffle slab with nx*ny vertical through-cells.
    """
    walls: list[Box] = []
    for i in range(nx + 1):
        cx = -env.x / 2 + t / 2 + i * (env.x - t) / nx
        walls.append(
            Box(
                op_id=f"wx_{i}",
                mode=OpMode.ADD,
                position=Vec3(x=cx, y=0.0, z=0.0),
                width=t,
                depth=env.y,
                height=env.z,
            )
        )
    for j in range(ny + 1):
        cy = -env.y / 2 + t / 2 + j * (env.y - t) / ny
        walls.append(
            Box(
                op_id=f"wy_{j}",
                mode=OpMode.ADD,
                position=Vec3(x=0.0, y=cy, z=0.0),
                width=env.x,
                depth=t,
                height=env.z,
            )
        )
    return walls


def lattice_program(env: Vec3, min_wall: float, density: str) -> PartProgram:
    t = min_wall
    nx, ny = derive_counts(env.x, t, density), derive_counts(env.y, t, density)
    return PartProgram(
        metadata=Metadata(
            name=f"lattice_{density}", units="mm", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)
        ),
        operations=lattice_walls(env, t, nx, ny),
        constraints=Constraints(min_wall_thickness=min_wall, max_overhang_deg=45.0, max_bbox=env),
    )


def solid_mass(env: Vec3, material: str = "PLA") -> float:
    prog = PartProgram(
        metadata=Metadata(name="solid"),
        operations=[Box(op_id="b", width=env.x, depth=env.y, height=env.z)],
    )
    model = compile_program(prog, Path(tempfile.mkdtemp(prefix="solid_")))
    return compute_physics(model.part, material).mass_g


def probe(env: Vec3, min_wall: float, density: str) -> bool:
    prog = lattice_program(env, min_wall, density)
    nx = derive_counts(env.x, min_wall, density)
    ny = derive_counts(env.y, min_wall, density)
    model = compile_program(prog, Path(tempfile.mkdtemp(prefix="lat_")))
    report = validate(model, prog, material="PLA")
    ck = {c.id: c for c in report.checks}
    key = [
        "mesh.single_body",
        "mesh.watertight",
        "features.min_wall_thickness",
        "printability.overhang_max_angle",
    ]
    lat_mass = report.metrics.mass_g if report.metrics else float("nan")
    sol_mass = solid_mass(env)
    lighter = lat_mass < sol_mass

    print(
        f"\n=== lattice {density}  env=({env.x:.0f},{env.y:.0f},{env.z:.0f})  "
        f"min_wall={min_wall}  cells={nx}x{ny}  walls={len(prog.operations)} ==="
    )
    for cid in key:
        c = ck.get(cid)
        print(f"  {c.status.value if c else '-':<5} {cid:<34} {c.actual if c else ''}")
    print(
        f"  mass: lattice={lat_mass:.1f}g  solid={sol_mass:.1f}g  "
        f"saved={100 * (1 - lat_mass / sol_mass):.0f}%  {'LIGHTER' if lighter else '*** NOT lighter ***'}"
    )
    all_pass = all(ck.get(c) and ck[c].status.value == "pass" for c in key)
    return all_pass and lighter


def main() -> None:
    cases = [
        (Vec3(x=60.0, y=60.0, z=20.0), 4.0),
        (Vec3(x=80.0, y=40.0, z=25.0), 3.0),
        (Vec3(x=120.0, y=30.0, z=20.0), 4.0),
    ]
    results: list[bool] = []
    for env, mw in cases:
        for density in ("light", "medium", "dense"):
            results.append(probe(env, mw, density))

    print("\n=== infeasible guard (envelope too small for a cell at min_wall) ===")
    tiny = Vec3(x=10.0, y=10.0, z=10.0)
    nx = derive_counts(tiny.x, 4.0, "light")
    cell = (tiny.x - 4.0 * (nx + 1)) / nx
    print(
        f"  env={tiny.x:.0f} min_wall=4 -> cells={nx}, cell_interior={cell:.1f}mm "
        f"({'INFEASIBLE (reject)' if cell < 4.0 else 'feasible'})"
    )

    ok = all(results)
    print(
        f"\n{'ALL PASS — lattice is validator-ready + lighter' if ok else '*** SOME FAILED — see above (HARD STOP)'}"
    )
    print(
        f"  {sum(results)}/{len(results)} (env x density) cases pass all 4 checks AND are lighter"
    )


if __name__ == "__main__":
    main()
