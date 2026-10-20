"""MotionProof projector (ADR-0086): measured kinematics -> the frozen contract.

``project_motion(mesh)`` runs the motion analysis (``forge_mind.motion``) over the uploaded
mesh's solid bodies and returns ``(MotionReport | None, findings)``:

  * 1 body  -> T1 spin: a measured axis earns a ``spin`` claim (with the balance evidence);
    no measurable axis -> no claim, one honest "nothing animates" finding.
  * 2 bodies -> T2 pair rotation: the full-turn swept certification (free / arc_limited /
    locked / refused), always with the reason. An arc-limited pair that measures as an
    external gear mesh may EARN the coupled upgrade (T3 ``gear_pair_coupled``).
  * 3+ bodies -> T3 rolling-element mechanism (``rolling_mechanism``): a bearing pattern is
    certified over its COUPLED trajectory at the measured epicyclic ratio; unrecognized
    patterns refuse with the reason.

Never raises and never gates: any internal failure projects a ``not_run`` finding with the
reason. The viewer animates ONLY what a claim states (the no-fake-movement law).
"""

from __future__ import annotations

from typing import Literal

import trimesh

from forge_mind.motion import PairMotion, SpinAxis, pair_rotation, spin_axis
from forge_mind.motion.mechanism import (
    Coupling,
    MechanismMotion,
    gear_pair_coupled,
    rolling_mechanism,
)
from forge_mind.physics.report.schema import (
    Confidence,
    Finding,
    Measured,
    MotionAxis,
    MotionBody,
    MotionClaim,
    MotionCoupling,
    MotionReport,
    PhysicsLocator,
    PhysicsStatus,
)
from forge_mind.types import Vec3

_SUITE = "motion"
_METHOD = "seeded interior-sample swept collision + exact boolean interference (ADR-0086)"


def _finding(
    fid: str,
    label: str,
    status: PhysicsStatus,
    explanation: str,
    *,
    measured: Measured | None = None,
    locator: PhysicsLocator | None = None,
    method: str = _METHOD,
    reason_not_run: str | None = None,
) -> Finding:
    return Finding(
        id=fid,
        suite=_SUITE,
        label=label,
        status=status,
        severity="advisory",  # motion never blocks: it reports what really moves
        measured=measured,
        limit=None,
        locator=locator,
        explanation=explanation,
        fix=None,
        confidence=Confidence(tier="computed", method=method),
        reason_not_run=reason_not_run,
    )


def _vec3(t: tuple[float, float, float]) -> Vec3:
    return Vec3(x=float(t[0]), y=float(t[1]), z=float(t[2]))


def _axis_locator(origin: tuple[float, float, float]) -> PhysicsLocator:
    return PhysicsLocator(point=_vec3(origin))


def _balance_finding(axis: SpinAxis) -> Finding:
    b = axis.balance
    if b.balanced is None:
        return _finding(
            "motion.balance",
            "Rotor balance",
            "not_run",
            "Balance could not be certified.",
            reason_not_run=b.note,
        )
    if b.balanced:
        return _finding(
            "motion.balance",
            "Rotor balance",
            "pass",
            f"Geometrically balanced about the spin axis ({b.note}).",
            measured=Measured(value=round(b.static_ratio, 4), unit="static ratio"),
        )
    return _finding(
        "motion.balance",
        "Rotor balance",
        "warn",
        f"Unbalanced about the spin axis — it will wobble under speed ({b.note}). "
        "It still physically rotates; balance is a quality finding, not a motion gate.",
        measured=Measured(value=round(b.static_ratio, 4), unit="static ratio"),
    )


def _body(
    ix: int, mesh: trimesh.Trimesh, role: Literal["part", "rotor", "stator", "roller"]
) -> MotionBody:
    vol: float | None
    try:
        vol = float(abs(mesh.volume))
    except Exception:
        vol = None
    return MotionBody(id=f"body_{ix}", role=role, volume_mm3=vol)


def _project_spin(
    mesh: trimesh.Trimesh, brep_part: object | None
) -> tuple[MotionReport | None, list[Finding]]:
    axis = spin_axis(mesh, brep_part)
    if axis is None:
        return None, [
            _finding(
                "motion.none",
                "Motion",
                "info",
                "No measured spin axis on this body — nothing animates. A part with no "
                "rotor signature (bracket, plate, box) has no validated motion to show.",
            )
        ]
    axis_f = _finding(
        "motion.axis",
        "Spin axis",
        "pass",
        f"Measured spin axis ({axis.method}): {axis.evidence}.",
        locator=_axis_locator(axis.origin),
    )
    claim = MotionClaim(
        id="claim_spin_0",
        kind="spin",
        status="free",
        axis=MotionAxis(origin=_vec3(axis.origin), direction=_vec3(axis.direction)),
        moving=["body_0"],
        static=[],
        evidence=[axis_f, _balance_finding(axis)],
        explanation=f"Functional spin about the part's own measured axis ({axis.method}).",
    )
    report = MotionReport(bodies=[_body(0, mesh, "part")], claims=[claim], method=_METHOD)
    return report, list(claim.evidence)


_PAIR_STATUS: dict[str, PhysicsStatus] = {
    "free": "pass",
    "arc_limited": "warn",
    "locked": "info",  # an honest measured fact (press fits are often intended), not a defect
    "refused": "not_run",
}


def _coupling(c: Coupling) -> MotionCoupling:
    # mechanism.Coupling -> the contract DTO (body index -> the GLB node id)
    return MotionCoupling(
        body=f"body_{c.body_index}",
        ratio=float(c.ratio),
        axis=MotionAxis(origin=_vec3(c.axis_origin), direction=_vec3(c.axis_direction)),
    )


def _project_mechanism(
    bodies: list[trimesh.Trimesh], mm: MechanismMotion, claim_id: str
) -> tuple[MotionReport | None, list[Finding]]:
    """A certified coupled mechanism -> one coupled_rotation claim (driver + ratio'd
    followers). Refused mechanisms return no claim — the finding carries the reason."""
    status = _PAIR_STATUS[mm.status]
    label = "Bearing motion" if mm.kind == "rolling" else "Gear mesh"
    measured: Measured | None = None
    if mm.interference_volume_mm3 is not None:
        measured = Measured(value=round(mm.interference_volume_mm3, 2), unit="mm³ overlap")
    finding = _finding(
        f"motion.{mm.kind}",
        label,
        status,
        (f"{mm.explanation}" + (f" [{mm.evidence}]" if mm.evidence else "")),
        measured=measured,
        locator=_axis_locator(mm.axis_origin) if mm.axis_origin is not None else None,
        reason_not_run=mm.explanation if mm.status == "refused" else None,
    )
    if mm.status == "refused":
        return None, [finding]
    moving = [f"body_{mm.driver_index}", *(f"body_{c.body_index}" for c in mm.couplings)]
    roles: dict[int, Literal["part", "rotor", "stator", "roller"]] = {mm.driver_index: "rotor"}
    for s in mm.static_indices:
        roles[s] = "stator"
    for c in mm.couplings:
        roles[c.body_index] = "roller" if mm.kind == "rolling" else "rotor"
    claim = MotionClaim(
        id=claim_id,
        kind="coupled_rotation",
        status=mm.status,
        axis=(
            MotionAxis(origin=_vec3(mm.axis_origin), direction=_vec3(mm.axis_direction))
            if mm.axis_origin is not None and mm.axis_direction is not None
            else None
        ),
        moving=moving,
        static=[f"body_{s}" for s in mm.static_indices],
        allowed_arc_deg=mm.allowed_arc_deg,
        blocked_at_deg=mm.blocked_at_deg or None,
        couplings=[_coupling(c) for c in mm.couplings],
        evidence=[finding],
        explanation=mm.explanation,
    )
    report = MotionReport(
        bodies=[_body(i, b, roles.get(i, "part")) for i, b in enumerate(bodies)],
        claims=[claim],
        method=_METHOD,
    )
    return report, [finding]


def _project_pair(bodies: list[trimesh.Trimesh]) -> tuple[MotionReport | None, list[Finding]]:
    pm: PairMotion = pair_rotation(bodies)
    if pm.status == "arc_limited":
        # a meshed gear pair is arc-limited BY CONSTRUCTION when swept uncoupled; try to
        # EARN the coupled claim (measured tooth ratio + full coupled sweep). Anything
        # short of certification keeps the honest uncoupled verdict below.
        mm = gear_pair_coupled(bodies, pm)
        if mm is not None:
            return _project_mechanism(bodies, mm, "claim_gear_0")
    status = _PAIR_STATUS[pm.status]
    measured: Measured | None = None
    if pm.clearance_mm is not None:
        measured = Measured(value=round(pm.clearance_mm, 3), unit="mm min clearance")
    elif pm.interference_volume_mm3 is not None:
        measured = Measured(value=round(pm.interference_volume_mm3, 2), unit="mm³ overlap")
    finding = _finding(
        "motion.pair",
        "Relative rotation",
        status,
        pm.explanation,
        measured=measured,
        locator=_axis_locator(pm.axis_origin) if pm.axis_origin is not None else None,
        reason_not_run=pm.explanation if pm.status == "refused" else None,
    )
    if pm.status == "refused":
        # no claim: the viewer must not animate an uncertified pair
        return None, [finding]
    roles: list[Literal["rotor", "stator"]] = ["rotor", "stator"]
    order = [pm.rotor_index, pm.stator_index]
    claim = MotionClaim(
        id="claim_pair_0",
        kind="pair_rotation",
        status=pm.status,
        axis=(
            MotionAxis(origin=_vec3(pm.axis_origin), direction=_vec3(pm.axis_direction))
            if pm.axis_origin is not None and pm.axis_direction is not None
            else None
        ),
        moving=[f"body_{pm.rotor_index}"],
        static=[f"body_{pm.stator_index}"],
        allowed_arc_deg=pm.allowed_arc_deg,
        blocked_at_deg=pm.blocked_at_deg or None,
        evidence=[finding],
        explanation=pm.explanation,
    )
    report = MotionReport(
        bodies=[_body(ix, bodies[ix], roles[i]) for i, ix in enumerate(order)],
        claims=[claim],
        method=_METHOD,
    )
    return report, [finding]


def project_motion(
    mesh: trimesh.Trimesh, brep_part: object | None = None
) -> tuple[MotionReport | None, list[Finding]]:
    """Analyze + project motion for an uploaded mesh. Never raises; fail-soft to not_run."""
    try:
        # Weld duplicate seam vertices BEFORE body-splitting: connectivity on an unwelded
        # mesh reads one solid as several shells and over-counts bodies (boolean outputs
        # and some STL exporters leave seams). Copy first — the caller's mesh is not ours.
        work = mesh.copy()
        work.merge_vertices()
        bodies = [b for b in work.split(only_watertight=False) if len(b.faces) > 0]
        if len(bodies) <= 1:
            return _project_spin(bodies[0] if bodies else work, brep_part)
        if len(bodies) == 2:
            return _project_pair(bodies)
        # 3+ bodies: the measured rolling-element mechanism (ADR-0086 T3). Unrecognized
        # patterns refuse with the reason — nothing animates unguessed.
        return _project_mechanism(bodies, rolling_mechanism(bodies), "claim_rolling_0")
    except Exception as exc:  # never break the upload; never fake a claim
        return None, [
            _finding(
                "motion.error",
                "Motion",
                "not_run",
                "Motion analysis could not run.",
                reason_not_run=f"{type(exc).__name__}: {exc}",
            )
        ]


__all__ = ["project_motion"]
