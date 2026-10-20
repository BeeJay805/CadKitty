"""Deterministic involute gear helpers + a rule-based gear-train validator.

The agentic author can model a gear *housing* freely, but the TEETH are where freehand CAD
fails: an LLM-approximated tooth looks plausible and does not mesh. So gears are generated here,
deterministically, from the standard parameters -- module, tooth count, pressure angle -- exactly
the way ``helpers.bolt_holes`` removes bolt-circle trig from the model's hands.

Two layers, both pure of any LLM:

  GEOMETRY (needs build123d): ``spur_gear`` (external, involute) and ``ring_gear`` (internal).
    Each tooth is a real involute flank profile; the gear body is built on a solid hub/annulus
    backbone the teeth fuse into, so the result is watertight + manifold by construction.

  VALIDATION (pure math, no kernel): ``validate_gear_pair`` / ``validate_planetary`` /
    ``gear_sanity`` enforce the meshing rules a wrong gear set silently violates -- equal module,
    correct centre distance, the planetary tooth relation (ring = sun + 2*planet), bore + tooth
    sanity. These unit-test in microseconds.

  MESHING (kinematic, needs the kernel): ``clock_deg`` + ``mesh_clock_deg`` / ``planet_clock_deg``
    phase the teeth so gears actually engage, and ``validate_gear_mesh_kinematic`` /
    ``validate_planetary_kinematic`` PROVE it by rotating the meshing gears and measuring tooth
    interference -- a correctly-clocked pair / planetary set turns at ~0 interference; a mis-clock
    binds. ``planetary_gearset`` clocks every contact so the whole set meshes.

The tooth geometry is selectable per ``backend``: 'native' (our generator) or 'bd_warehouse' (the
standard-parts library's involute SpurGear, with proper trochoidal root fillets). Both share the
same tooth-phase convention, so clocking + the kinematic checks are backend-agnostic.

LIMITATIONS (honest): the NATIVE tooth ROOT is a straight radial relief, not a swept trochoidal
fillet (fine for printed/most-machined gears; the bd_warehouse backend has trochoidal roots). The
``ring_gear`` is native-only and its internal flank uses the conjugate external-tooth profile
(a standard simplification) -- bd_warehouse has no internal gear.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, cos, degrees, pi, radians, sin, tan
from typing import Any

from forge_mind.materials import Material

# ----------------------------------------------------------------------------
# Spec + derived gear geometry (pure)
# ----------------------------------------------------------------------------


# Below this tooth count a standard 20-deg involute UNDERCUTS at the root (the cutter gouges
# the flank), so the generated tooth is weaker / not a textbook-clean involute. We still BUILD it
# (it meshes), but flag the spec PROVISIONAL so a caller never treats it as fully-valid geometry.
_UNDERCUT_MIN_TEETH = 17


@dataclass(frozen=True)
class GearSpec:
    """A standard spur/ring gear by its defining parameters. All derived radii are properties."""

    module: float  # mm of pitch diameter per tooth (THE meshing parameter)
    teeth: int
    pressure_angle: float = 20.0  # degrees; 20 is the modern standard
    thickness: float = 6.0  # face width (mm)
    bore: float = 0.0  # central shaft-bore diameter (mm); 0 = solid

    @property
    def provisional(self) -> bool:
        """True when the involute UNDERCUTS (teeth < 17 at <=20 deg): it builds and meshes, but
        the flank is not a fully-valid involute -- a caller must not treat it as certified geometry."""
        return self.teeth < _UNDERCUT_MIN_TEETH and self.pressure_angle <= 20.0 + 1e-9

    @property
    def pitch_radius(self) -> float:
        return self.module * self.teeth / 2.0

    @property
    def base_radius(self) -> float:
        return self.pitch_radius * cos(radians(self.pressure_angle))

    @property
    def outer_radius(self) -> float:  # addendum = 1 module
        return self.pitch_radius + self.module

    @property
    def root_radius(self) -> float:  # dedendum = 1.25 module
        return self.pitch_radius - 1.25 * self.module

    @property
    def circular_pitch(self) -> float:
        return pi * self.module

    @property
    def tooth_thickness_at_pitch(self) -> float:
        return pi * self.module / 2.0


def _inv(alpha: float) -> float:
    """The involute function inv(a) = tan(a) - a (a in radians)."""
    return tan(alpha) - alpha


def _rotate(p: tuple[float, float], ang: float) -> tuple[float, float]:
    c, s = cos(ang), sin(ang)
    x, y = p
    return (x * c - y * s, x * s + y * c)


def _involute_tooth_points(
    module: float,
    teeth: int,
    pressure_angle: float,
    r_inner: float,
    r_outer: float,
    *,
    root_overlap: float = 0.4,
    n: int = 14,
) -> list[tuple[float, float]]:
    """Closed polygon of ONE involute tooth, centred on +X, spanning ``r_inner``..``r_outer``.

    Both flanks are true involutes of the base circle; the angular tooth thickness at the pitch
    circle is the standard pi*module/2. The root chord is pushed ``root_overlap`` mm inward so the
    tooth fuses into (or, for an internal cutter, reaches past) the backbone. Used by both the
    external gear (as solid teeth) and the ring gear (as gap cutters).
    """
    alpha = radians(pressure_angle)
    r_p = module * teeth / 2.0
    r_b = r_p * cos(alpha)
    half = pi / (2.0 * teeth)  # half tooth-thickness angle at the pitch circle
    inv_p = _inv(alpha)

    def gamma(r: float) -> float:
        rr = max(r, r_b)  # the involute is undefined below the base circle -> clamp
        a_r = acos(min(1.0, r_b / rr))
        return half + inv_p - _inv(a_r)

    flank_start = max(r_b, r_inner)
    radii = [flank_start + (r_outer - flank_start) * k / (n - 1) for k in range(n)]
    right = [(r * cos(-gamma(r)), r * sin(-gamma(r))) for r in radii]  # lower flank, in->out
    left = [(r * cos(gamma(r)), r * sin(gamma(r))) for r in radii]  # upper flank, in->out

    g0 = gamma(flank_start)
    r_chord = max(0.5, r_inner - root_overlap)
    root_r = (r_chord * cos(-g0), r_chord * sin(-g0))
    root_l = (r_chord * cos(g0), r_chord * sin(g0))

    # CCW: root(-) -> right flank up -> across the tip -> left flank down -> root(+) -> (auto-close)
    return [root_r, *right, *reversed(left), root_l]


# ----------------------------------------------------------------------------
# Geometry builders (need build123d; importable in the sandbox)
# ----------------------------------------------------------------------------

_MIN_TEETH = 6  # below this an involute gear self-intersects at the centre

# Gear-tooth geometry backends. "native" = our deterministic involute generator (below).
# "bd_warehouse" = the bd_warehouse standard-parts library's SpurGear (ISO involute with proper
# trochoidal root fillets). Both share the SAME tooth-phase convention (a tooth centred on +X at
# clock 0 -- verified: the alignment offset is 0 deg), so clocking + meshing are backend-agnostic.
GearBackend = str  # "native" | "bd_warehouse"


def _bd_spur_gear(
    module: float,
    teeth: int,
    thickness: float,
    pressure_angle: float,
    bore: float,
    clock_deg: float,
) -> Any:
    """A spur gear from the bd_warehouse library, normalised to OUR conventions: Z-CENTERED
    (the library's native frame, same as our spur_gear), an optional central bore, and
    ``clock_deg`` applied as the tooth phase. Returns a watertight build123d solid that meshes
    interchangeably with a native gear."""
    from bd_warehouse.gear import SpurGear
    from build123d import Cylinder

    gear = SpurGear(
        module=module,
        tooth_count=teeth,
        pressure_angle=pressure_angle,
        thickness=thickness,
        rotation=(0.0, 0.0, clock_deg),  # same phase convention as native (tooth on +X at 0)
    )
    if bore > 0.0:
        gear = gear - Cylinder(radius=bore / 2.0, height=thickness + 2.0)  # centred bore through z
    return gear  # already centred on z=0 -- the shared author-facing frame


def spur_gear(
    module: float,
    teeth: int,
    thickness: float,
    *,
    pressure_angle: float = 20.0,
    bore: float = 0.0,
    clock_deg: float = 0.0,
    backend: GearBackend = "native",
) -> Any:
    """An external involute spur gear as a single watertight build123d solid.

    Z-CENTERED at the origin, axis +Z (spans -thickness/2 .. +thickness/2), the same frame as
    build123d's ``Cylinder`` -- so ``disc + spur_gear(...)`` and boss/rim compositions line up
    without translation. (Measured on the timing-gear run: the old 0..thickness frame silently
    sank a hub boss 4 mm into the web because the author composed against the documented
    centered frame. The DSL compiler op restores its own [0, thickness] contract explicitly.)

    Teeth fuse onto a solid hub disk at the root circle, so the body is one closed solid
    regardless of minor tooth-profile error. ``bore`` (diameter) cuts a central shaft hole.
    ``clock_deg`` rotates the whole tooth ring about the gear axis -- the tooth-PHASE that makes a
    mating gear actually MESH (a tooth on +X at clock 0; see ``mesh_clock_deg``).

    ``backend``: 'native' (our involute generator, default) or 'bd_warehouse' (the standard-parts
    library's SpurGear, with trochoidal root fillets). Both share the same phase convention, so
    clocking and the kinematic mesh check work for either.
    """
    if backend not in ("native", "bd_warehouse"):
        raise ValueError(f"unknown gear backend {backend!r}; use 'native' or 'bd_warehouse'")
    if teeth < _MIN_TEETH:
        raise ValueError(f"teeth must be >= {_MIN_TEETH} for a valid involute gear (got {teeth})")
    if module <= 0 or thickness <= 0:
        raise ValueError("module and thickness must be > 0")
    if backend == "bd_warehouse":
        return _bd_spur_gear(module, teeth, thickness, pressure_angle, bore, clock_deg)

    from build123d import BuildPart, BuildSketch, Circle, Mode, Polygon, Pos, extrude

    spec = GearSpec(module, teeth, pressure_angle, thickness, bore)
    tooth = _involute_tooth_points(
        module, teeth, pressure_angle, spec.root_radius, spec.outer_radius
    )
    phase = radians(clock_deg)
    with BuildPart() as gp:
        with BuildSketch():
            Circle(spec.root_radius)  # hub backbone
            for k in range(teeth):
                pts = [_rotate(p, 2.0 * pi * k / teeth + phase) for p in tooth]
                Polygon(*pts, align=None)  # fuse this tooth at its absolute angular position
        extrude(amount=thickness)
        if bore > 0:
            if bore / 2.0 >= spec.root_radius:
                raise ValueError("bore is >= the root circle; it would erase the gear body")
            with BuildSketch():
                Circle(bore / 2.0)
            extrude(amount=thickness, mode=Mode.SUBTRACT)
    return Pos(0.0, 0.0, -thickness / 2.0) * gp.part  # centered: the author-facing frame


def ring_gear(
    module: float,
    teeth: int,
    thickness: float,
    *,
    pressure_angle: float = 20.0,
    rim: float = 5.0,
    clock_deg: float = 0.0,
) -> Any:
    """An internal (ring) gear: teeth point INWARD from a solid rim. One watertight solid.
    Z-CENTERED at the origin, axis +Z -- the same frame as ``spur_gear`` (a planetary set's
    sun/planets/ring must share one frame).

    Built as a solid annulus (outer rim down to the tooth-tip circle) with ``teeth`` involute
    GAP profiles subtracted, leaving the internal teeth between them. ``rim`` is the wall
    thickness of material beyond the tooth roots. ``clock_deg`` rotates the tooth ring (the phase
    that meshes the planets in a planetary set).
    """
    if teeth < _MIN_TEETH:
        raise ValueError(f"teeth must be >= {_MIN_TEETH} for a valid ring gear (got {teeth})")
    if module <= 0 or thickness <= 0 or rim <= 0:
        raise ValueError("module, thickness, and rim must be > 0")

    from build123d import BuildPart, BuildSketch, Circle, Mode, Polygon, Pos, extrude

    r_p = module * teeth / 2.0
    tip_radius = r_p - module  # internal teeth tips (innermost reach)
    root_radius = r_p + 1.25 * module  # tooth gaps bottom out here, toward the rim
    outer_radius = root_radius + rim
    gap = _involute_tooth_points(
        module, teeth, pressure_angle, tip_radius, root_radius, root_overlap=0.6
    )
    phase = radians(clock_deg)
    with BuildPart() as gp:
        with BuildSketch():
            Circle(outer_radius)
            Circle(tip_radius, mode=Mode.SUBTRACT)  # bore to the tooth-tip circle -> annulus
            for k in range(teeth):
                pts = [_rotate(p, 2.0 * pi * k / teeth + phase) for p in gap]
                Polygon(*pts, align=None, mode=Mode.SUBTRACT)  # carve a tooth gap
        extrude(amount=thickness)
    return Pos(0.0, 0.0, -thickness / 2.0) * gp.part  # centered: the author-facing frame


# ----------------------------------------------------------------------------
# Gear-train validation (pure)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class GearIssue:
    """One finding from a gear validator. ``severity`` is 'error' (won't work) or 'warn' (risky)."""

    severity: str
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


def center_distance(g1: GearSpec, g2: GearSpec) -> float:
    """Theoretical centre distance for an EXTERNAL mesh: the sum of pitch radii."""
    return g1.pitch_radius + g2.pitch_radius


def mesh_clock_deg(teeth: int) -> float:
    """Tooth-phase (deg) that lets this gear MESH a reference gear sitting on its -X side.

    The reference gear is built tooth-on-+X (clock 0); a meshing partner placed at +centre-distance
    must present a tooth GAP toward it (a gap centred on the -X axis). Depends only on the tooth
    count. Apply as ``spur_gear(..., clock_deg=mesh_clock_deg(teeth))``.
    """
    pitch = 2.0 * pi / teeth
    k = round(pi / pitch - 0.5)
    return degrees(pi - pitch * (k + 0.5))


@dataclass(frozen=True)
class GearMeshResult:
    """Outcome of the kinematic mesh check: do the teeth turn together without binding?"""

    meshes: bool
    max_overlap_mm3: float  # peak tooth interference over the sampled rotation (0 = clean)
    samples: int
    note: str


def validate_gear_mesh_kinematic(
    g1: GearSpec,
    g2: GearSpec,
    *,
    clock1_deg: float = 0.0,
    clock2_deg: float | None = None,
    samples: int = 9,
    backend: GearBackend = "native",
) -> GearMeshResult:
    """Rotate a clocked external gear PAIR through a tooth cycle and prove it turns without jamming.

    Builds both gears (g2 clocked to mesh, by default ``mesh_clock_deg(g2.teeth)``), places g2 at the
    exact centre distance, then drives g1 by an angle theta and the conjugate gear g2 by
    ``-theta * teeth1/teeth2`` (the gear-ratio constraint) across ``samples`` poses, measuring the
    boolean tooth-interference volume at each. A correctly-clocked pair interleaves collision-free
    (peak interference ~0); a mis-clocked pair or a wrong centre distance BINDS (large interference).
    The pass threshold scales with one tooth's volume, so it is size-independent.
    """
    if abs(g1.module - g2.module) > 1e-9:
        return GearMeshResult(False, 0.0, 0, "modules differ -- the gears cannot mesh at all")

    import os
    import tempfile

    import numpy as np
    import trimesh
    from build123d import export_stl

    def _load(spec: GearSpec, clock: float) -> Any:
        part = spur_gear(
            spec.module,
            spec.teeth,
            spec.thickness,
            pressure_angle=spec.pressure_angle,
            bore=spec.bore,
            clock_deg=clock,
            backend=backend,
        )
        path = os.path.join(tempfile.mkdtemp(), "g.stl")
        export_stl(part, path)
        return trimesh.load(path, force="mesh")

    def _rotz(px: float, py: float, deg: float) -> Any:
        a = np.radians(deg)
        c, s = np.cos(a), np.sin(a)
        r = np.eye(4)
        r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
        t1, t2 = np.eye(4), np.eye(4)
        t1[:3, 3] = [-px, -py, 0.0]
        t2[:3, 3] = [px, py, 0.0]
        return t2 @ r @ t1

    c2 = mesh_clock_deg(g2.teeth) if clock2_deg is None else clock2_deg
    cd = center_distance(g1, g2)
    a_mesh = _load(g1, clock1_deg)
    b_mesh = _load(g2, c2)
    b_mesh.apply_translation([cd, 0.0, 0.0])

    pitch_a = 360.0 / g1.teeth
    ratio = g1.teeth / g2.teeth
    max_ov = 0.0
    for i in range(samples):
        theta = pitch_a * i / max(1, samples - 1)
        a = a_mesh.copy()
        a.apply_transform(_rotz(0.0, 0.0, theta))
        b = b_mesh.copy()
        b.apply_transform(_rotz(cd, 0.0, -theta * ratio))
        try:
            ov = abs(float(trimesh.boolean.intersection([a, b]).volume))
        except Exception as exc:
            # FAIL CLOSED (incident 2026-06-11): a missing/erroring boolean backend
            # used to record ZERO interference -- a mis-clocked pair then read as
            # meshing wherever manifold3d was absent (CI caught it; Fly would have
            # shipped it). A check that cannot measure must never certify.
            return GearMeshResult(
                False,
                0.0,
                i,
                f"tooth-interference boolean unavailable ({type(exc).__name__}) -- "
                "cannot certify the mesh (fail closed)",
            )
        max_ov = max(max_ov, ov)

    tooth_vol = g1.tooth_thickness_at_pitch * (2.25 * g1.module) * g1.thickness
    tol = 0.2 * tooth_vol  # well above numeric noise (~0), far below a mis-clock jam (~1 tooth)
    meshes = max_ov <= tol
    note = (
        f"meshes & turns: peak tooth interference {max_ov:.1f} mm^3 over {samples} poses "
        f"(< {tol:.1f} mm^3 budget)"
        if meshes
        else f"BINDS: peak tooth interference {max_ov:.1f} mm^3 exceeds the {tol:.1f} mm^3 budget -- "
        "mis-clocked or wrong centre distance"
    )
    return GearMeshResult(meshes, round(max_ov, 1), samples, note)


def validate_gear_pair(g1: GearSpec, g2: GearSpec) -> list[GearIssue]:
    """Meshing rules for two external gears: equal module + equal pressure angle (else they jam)."""
    issues: list[GearIssue] = []
    if abs(g1.module - g2.module) > 1e-9:
        issues.append(
            GearIssue(
                "error",
                f"modules differ ({g1.module} vs {g2.module}) -- gears only mesh at EQUAL module; "
                "the teeth will not engage.",
            )
        )
    if abs(g1.pressure_angle - g2.pressure_angle) > 1e-9:
        issues.append(
            GearIssue(
                "error",
                f"pressure angles differ ({g1.pressure_angle} vs {g2.pressure_angle} deg) -- "
                "meshing gears must share a pressure angle.",
            )
        )
    return issues


def validate_planetary(
    sun_teeth: int, planet_teeth: int, ring_teeth: int, n_planets: int
) -> list[GearIssue]:
    """The two hard planetary constraints, plus a neighbour-collision sanity check.

    - meshing: ring_teeth must equal sun_teeth + 2*planet_teeth (a planet meshes the sun on one
      side and the ring on the other -- this is the only ring count that closes both meshes).
    - equal spacing: (sun_teeth + ring_teeth) must be divisible by n_planets, or the planets do
      not land on valid tooth phases at equal angles.
    """
    issues: list[GearIssue] = []
    required_ring = sun_teeth + 2 * planet_teeth
    if ring_teeth != required_ring:
        issues.append(
            GearIssue(
                "error",
                f"ring teeth = {ring_teeth} but must equal sun + 2*planet = "
                f"{sun_teeth} + 2*{planet_teeth} = {required_ring} for the planets to mesh both "
                "the sun and the ring.",
            )
        )
    if n_planets >= 2 and (sun_teeth + ring_teeth) % n_planets != 0:
        issues.append(
            GearIssue(
                "error",
                f"(sun + ring) = {sun_teeth + ring_teeth} is not divisible by {n_planets} planets "
                "-- equal angular spacing is impossible at valid tooth phases; change the planet "
                "count or tooth numbers.",
            )
        )
    if n_planets >= 2 and (sun_teeth + planet_teeth) * sin(pi / n_planets) < planet_teeth + 2:
        # Neighbours collide when the chord between adjacent planet centres is under the planet OD:
        # 2*carrier*sin(pi/n) < OD, with carrier = m*(sun+planet)/2 and OD = m*(planet+2); m cancels.
        issues.append(
            GearIssue("warn", "planets are large relative to the sun; check neighbour clearance.")
        )
    return issues


def gear_sanity(spec: GearSpec, material: Material | None = None) -> list[GearIssue]:
    """Single-gear sanity: tooth count, bore vs body, undercut, and material producibility."""
    issues: list[GearIssue] = []
    if spec.teeth < _MIN_TEETH:
        issues.append(GearIssue("error", f"teeth = {spec.teeth} < {_MIN_TEETH}: not a valid gear."))
    if spec.bore > 0 and spec.bore / 2.0 >= spec.root_radius:
        issues.append(
            GearIssue(
                "error",
                f"bore radius {spec.bore / 2.0:.2f}mm >= root radius {spec.root_radius:.2f}mm -- "
                "the bore erases the gear body; shrink the bore or add teeth.",
            )
        )
    if spec.provisional:
        issues.append(
            GearIssue(
                "warn",
                f"teeth = {spec.teeth} < {_UNDERCUT_MIN_TEETH} at {spec.pressure_angle:.0f} deg: "
                "the involute will undercut near the root (weaker teeth, PROVISIONAL geometry); "
                "raise the tooth count or pressure angle.",
            )
        )
    if material is not None:
        tt = spec.tooth_thickness_at_pitch
        if tt < material.min_feature_mm:
            issues.append(
                GearIssue(
                    "error",
                    f"tooth thickness {tt:.2f}mm at the pitch line is below {material.name}'s "
                    f"{material.min_feature_mm:.2f}mm minimum feature -- increase the module.",
                )
            )
        if spec.thickness < material.min_wall_mm:
            issues.append(
                GearIssue(
                    "warn",
                    f"face width {spec.thickness:.2f}mm is below {material.name}'s "
                    f"{material.min_wall_mm:.2f}mm minimum wall -- the gear may be fragile.",
                )
            )
    return issues


# ----------------------------------------------------------------------------
# Planetary assembly (geometry; needs build123d)
# ----------------------------------------------------------------------------


def planet_carrier_radius(sun_teeth: int, planet_teeth: int, module: float) -> float:
    """Orbit radius of the planet centres: the sun<->planet centre distance."""
    return module * (sun_teeth + planet_teeth) / 2.0


def planet_clock_deg(sun_teeth: int, planet_teeth: int, carrier_angle_deg: float) -> float:
    """Tooth-phase (deg) for a planet at ``carrier_angle_deg`` so it MESHES the sun.

    A planet orbiting a fixed sun rolls on it: its tooth phase is the base external-mesh clock plus
    the orbit angle times the roll ratio (1 + sun/planet). Proven to mesh the sun at 0 interference.
    """
    return mesh_clock_deg(planet_teeth) + carrier_angle_deg * (1.0 + sun_teeth / planet_teeth)


def planetary_gearset(
    module: float,
    sun_teeth: int,
    planet_teeth: int,
    n_planets: int,
    thickness: float,
    *,
    pressure_angle: float = 20.0,
    bore: float = 0.0,
    rim: float = 5.0,
    backend: GearBackend = "native",
) -> Any:
    """A labelled build123d Compound of a MESHING planetary set: sun + N planets + ring.

    Validates the train first and RAISES on any error (no silently-broken gear set). Every gear is
    tooth-phase CLOCKED so the set actually meshes: the sun is the reference, each planet is clocked
    by ``planet_clock_deg`` (rolling on the sun) and placed at the carrier radius, and the ring is
    clocked by ``-mesh_clock_deg(ring_teeth)``. All sun-planet and ring-planet contacts mesh at zero
    interference (verifiable with ``validate_planetary_kinematic``). ``backend`` selects the spur
    (sun + planet) tooth geometry; the ring is always native (bd_warehouse has no internal gear).
    """
    ring_teeth = sun_teeth + 2 * planet_teeth
    errs = [
        i for i in validate_planetary(sun_teeth, planet_teeth, ring_teeth, n_planets) if i.is_error
    ]
    if errs:
        raise ValueError("; ".join(i.message for i in errs))

    from build123d import Compound, Pos

    children: list[Any] = []
    sun = spur_gear(
        module, sun_teeth, thickness, pressure_angle=pressure_angle, bore=bore, backend=backend
    )
    sun.label = "sun"  # the reference: a tooth on +X (clock 0)
    children.append(sun)

    cr = planet_carrier_radius(sun_teeth, planet_teeth, module)
    for k in range(n_planets):
        phi_deg = 360.0 * k / n_planets
        planet = spur_gear(
            module,
            planet_teeth,
            thickness,
            pressure_angle=pressure_angle,
            bore=bore,
            clock_deg=planet_clock_deg(sun_teeth, planet_teeth, phi_deg),
            backend=backend,
        )
        ang = radians(phi_deg)
        placed = Pos(cr * cos(ang), cr * sin(ang), 0.0) * planet
        placed.label = f"planet_{k + 1}"
        children.append(placed)

    ring = ring_gear(
        module,
        ring_teeth,
        thickness,
        pressure_angle=pressure_angle,
        rim=rim,
        clock_deg=-mesh_clock_deg(ring_teeth),
    )
    ring.label = "ring"
    children.append(ring)
    return Compound(children=children)


def validate_planetary_kinematic(
    module: float,
    sun_teeth: int,
    planet_teeth: int,
    n_planets: int,
    thickness: float,
    *,
    pressure_angle: float = 20.0,
    backend: GearBackend = "native",
) -> GearMeshResult:
    """Confirm a clocked planetary set MESHES: every sun-planet and ring-planet contact is checked
    for tooth interference at the assembled pose. A correctly-clocked set interferes ~0 mm^3; a
    bad clock binds. Builds the gears, so it is slow (used by tests / the CLI, not the hot loop)."""
    import os
    import tempfile

    import trimesh
    from build123d import export_stl

    ring_teeth = sun_teeth + 2 * planet_teeth
    cr = planet_carrier_radius(sun_teeth, planet_teeth, module)

    def _load(part: Any) -> Any:
        path = os.path.join(tempfile.mkdtemp(), "g.stl")
        export_stl(part, path)
        return trimesh.load(path, force="mesh")

    sun = _load(
        spur_gear(module, sun_teeth, thickness, pressure_angle=pressure_angle, backend=backend)
    )
    ring = _load(
        ring_gear(
            module,
            ring_teeth,
            thickness,
            pressure_angle=pressure_angle,
            clock_deg=-mesh_clock_deg(ring_teeth),
        )
    )
    max_ov = 0.0
    for k in range(n_planets):
        phi = 360.0 * k / n_planets
        pm = _load(
            spur_gear(
                module,
                planet_teeth,
                thickness,
                pressure_angle=pressure_angle,
                clock_deg=planet_clock_deg(sun_teeth, planet_teeth, phi),
                backend=backend,
            )
        )
        pm.apply_translation([cr * cos(radians(phi)), cr * sin(radians(phi)), 0.0])
        for other in (sun, ring):
            try:
                ov = abs(float(trimesh.boolean.intersection([pm, other]).volume))
            except Exception as exc:
                # FAIL CLOSED -- same incident-fix as the pair check above: an
                # unmeasurable interference must never read as a clean mesh.
                return GearMeshResult(
                    False,
                    0.0,
                    k,
                    f"tooth-interference boolean unavailable ({type(exc).__name__}) -- "
                    "cannot certify the planetary set (fail closed)",
                )
            max_ov = max(max_ov, ov)

    tooth_vol = (pi * module / 2.0) * (2.25 * module) * thickness
    tol = 0.2 * tooth_vol
    meshes = max_ov <= tol
    note = (
        f"all {n_planets} planets mesh the sun + ring: peak interference {max_ov:.1f} mm^3 "
        f"(< {tol:.1f} mm^3 budget)"
        if meshes
        else f"planetary BINDS: peak interference {max_ov:.1f} mm^3 exceeds the {tol:.1f} mm^3 budget"
    )
    return GearMeshResult(meshes, round(max_ov, 1), n_planets * 2, note)


def toothed_rim(
    rim_d: float,
    teeth: int,
    width: float,
    *,
    engagement: float = 4.0,
    pressure_angle: float = 20.0,
) -> Any:
    """External tooth BAND sized to fuse onto an existing disc rim of diameter ``rim_d`` -- the
    flywheel starter-ring / ratchet-wheel / timing-pulley idiom (ADR-0078: the hard-5 flywheel
    dropped its 120-tooth ring because composing teeth onto a rim had no reachable idiom). The
    tooth ROOT circle lands on the rim (module = rim_d/(teeth-2.5)) so every tooth stands proud,
    and the band's bore is ``rim_d - 2*engagement`` so a union FUSES instead of merely touching.
    Centered at the origin, axis +Z. Use: ``result = disc + toothed_rim(disc_od, 120, 12)``."""
    if teeth < 8:
        raise ValueError("teeth must be >= 8")
    if rim_d <= 0 or width <= 0 or engagement <= 0:
        raise ValueError("rim_d, width, engagement must be > 0")
    module = rim_d / (teeth - 2.5)
    bore = max(rim_d - 2.0 * engagement, rim_d * 0.5)
    return spur_gear(module, teeth, width, pressure_angle=pressure_angle, bore=bore)


__all__ = [
    "GearIssue",
    "GearMeshResult",
    "GearSpec",
    "center_distance",
    "gear_sanity",
    "mesh_clock_deg",
    "planet_carrier_radius",
    "planet_clock_deg",
    "planetary_gearset",
    "ring_gear",
    "spur_gear",
    "toothed_rim",
    "validate_gear_mesh_kinematic",
    "validate_gear_pair",
    "validate_planetary",
    "validate_planetary_kinematic",
]
