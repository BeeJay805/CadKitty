"""Executable alternative designs and scoped two-jaw virtual-work screening.

Formula references describe mechanics, not the suitability of any selected hardware.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
from scipy.optimize import brentq

from forge_mind.workspace.document import DesignDocument, digest
from forge_mind.workspace.synthesis_models import Candidate, LinkageCandidate, SynthesisRequest

REFERENCES = [
    {
        "title": "MIT Introduction to Robotics, statics and virtual work",
        "url": "https://ocw.mit.edu/courses/2-12-introduction-to-robotics-fall-2005/5c160fb678fa75191c399a373c6ce648_chapter6.pdf",
        "scope": "Generalized forces and the kinematic Jacobian; applied here to both opposing jaws",
    },
    {
        "title": "Thomson lead screw catalog, torque and lead",
        "url": "https://www.thomsonlinear.com/downloads/screws/Lead_Screws_cten.pdf",
        "scope": "Rotary-to-linear torque/lead/efficiency relationship; not a rating for modeled envelopes",
    },
]
SCOPE = (
    "Bounded opposing-jaw architecture screening under equal quasistatic jaw loads and an explicitly declared simultaneous input torque/speed. "
    "F_jaw * abs(dgap/dtheta) = efficiency * input_torque (N mm); both jaw work contributions are included. "
    "Force and speed are operating-point predictions, not verified motor capability, contact, strength, holding force or product qualification. "
    "No weighted winner or automatic requirement approval."
)


def linkage_kinematics(candidate: LinkageCandidate, theta: float) -> tuple[float, float]:
    """Opening and signed opening derivative, mm and mm/radian, on the outer branch."""
    r, length = candidate.crank_mm, candidate.rod_mm
    sine, cosine = math.sin(theta), math.cos(theta)
    q = math.sqrt(length * length - r * r * sine * sine)
    return 2 * (r * cosine + q - candidate.jaw_offset_mm), -2 * (
        r * sine + r * r * sine * cosine / q
    )


def linkage_jacobian_bound(candidate: LinkageCandidate, start: float, end: float) -> float:
    """Upper envelope on |dg/dtheta| over an ordered outer-branch interval.

    Bound each factor on 128 intervals. Negative cosine requires the largest
    denominator for an upper ratio bound. The small outward margin addresses
    floating-point evaluation; this is not a formal directed-rounding certificate.
    """
    if not 0 <= start <= end <= math.pi:
        raise ValueError("Jacobian bound requires an ordered outer-branch angle interval")
    radius, length = candidate.crank_mm, candidate.rod_mm
    bounds = []
    for a, b in pairwise(np.linspace(start, end, 129)):
        sine_min = min(math.sin(a), math.sin(b))
        sine_max = 1.0 if a <= math.pi / 2 <= b else max(math.sin(a), math.sin(b))
        cosine_max = math.cos(a)
        q_min = math.sqrt(length**2 - radius**2 * sine_max**2)
        q_max = math.sqrt(length**2 - radius**2 * sine_min**2)
        ratio_max = radius * cosine_max / (q_min if cosine_max >= 0 else q_max)
        bounds.append(2 * radius * sine_max * (1 + ratio_max))
    return max(bounds) * (1 + 1e-12) + 1e-12


def _state(candidate: Candidate, opening: float) -> tuple[float, float]:
    if candidate.family == "screw":
        # One input shaft drives opposite-handed threads, each carrying one jaw.
        jacobian = candidate.lead_mm / math.pi
        return opening / jacobian, jacobian
    if candidate.family == "rack":
        jacobian = candidate.module_mm * candidate.pinion_teeth
        return opening / jacobian, jacobian
    lo, hi = math.radians(candidate.angle_min_deg), math.radians(candidate.angle_max_deg)
    maximum, minimum = linkage_kinematics(candidate, lo)[0], linkage_kinematics(candidate, hi)[0]
    if not minimum - 1e-9 <= opening <= maximum + 1e-9:
        raise ValueError(
            f"Opening {opening:g} mm is outside the declared linkage branch [{minimum:.6g}, {maximum:.6g}] mm"
        )
    if abs(opening - maximum) <= 1e-9:
        theta = lo
    elif abs(opening - minimum) <= 1e-9:
        theta = hi
    else:
        theta = float(
            brentq(lambda t: linkage_kinematics(candidate, t)[0] - opening, lo, hi, xtol=1e-13)
        )
    return theta, abs(linkage_kinematics(candidate, theta)[1])


def candidate_document(request: SynthesisRequest, candidate_id: str) -> DesignDocument:
    """Rebuild the exact parameter-bound proposal, never a cached illustrative card."""
    from forge_mind.workspace.synthesis_geometry import realize

    candidate = next((c for c in request.candidates if c.id == candidate_id), None)
    if candidate is None:
        raise ValueError("Unknown synthesis candidate")
    # Do not silently clamp an unreachable task to a convenient demonstration range.
    angles = []
    for opening in (request.task.opening_min_mm, request.task.opening_max_mm):
        theta, jacobian = _state(candidate, opening)
        angles.append(math.degrees(theta))
        if jacobian < 1e-6:
            raise ValueError(
                "Requested task includes a dead-center singularity; no finite-force realization"
            )
    opening = (request.task.opening_min_mm + request.task.opening_max_mm) / 2
    angle = math.degrees(_state(candidate, opening)[0])
    return realize(request, candidate, opening, angle, (angles[0], angles[1]))


def _compare(request: SynthesisRequest, candidate: Candidate) -> dict[str, Any]:
    omega = request.drive.rpm * 2 * math.pi / 60
    torque = request.drive.torque_nm
    task = request.task
    efficiency = candidate.efficiency
    samples = []
    errors = []
    for opening in np.linspace(task.opening_min_mm, task.opening_max_mm, request.samples):
        row: dict[str, Any] = {
            "opening_mm": float(opening),
            "status": "unverified",
            "input_angle_deg": None,
            "jaw_speed_mm_s": None,
            "closing_speed_mm_s": None,
            "force_per_jaw_low_n": None,
            "force_per_jaw_high_n": None,
            "required_torque_low_nm": None,
            "required_torque_high_nm": None,
            "output_power_low_w": None,
            "output_power_high_w": None,
        }
        try:
            theta, jacobian = _state(candidate, float(opening))
            row.update(
                input_angle_deg=math.degrees(theta),
                gap_jacobian_mm_per_rad=jacobian,
                closing_speed_mm_s=jacobian * omega,
                jaw_speed_mm_s=jacobian * omega / 2,
            )
            if jacobian < 1e-6:
                raise ValueError("Dead-center singularity: no finite force prediction")
            if efficiency:
                row.update(
                    status="computed",
                    force_per_jaw_low_n=1000 * torque * efficiency.low / jacobian,
                    force_per_jaw_high_n=1000 * torque * efficiency.high / jacobian,
                    output_power_low_w=torque * omega * efficiency.low,
                    output_power_high_w=torque * omega * efficiency.high,
                )
                if task.required_force_per_jaw_n is not None:
                    row.update(
                        required_torque_low_nm=task.required_force_per_jaw_n
                        * jacobian
                        / (1000 * efficiency.high),
                        required_torque_high_nm=task.required_force_per_jaw_n
                        * jacobian
                        / (1000 * efficiency.low),
                    )
            else:
                row["reason"] = (
                    "Transmission efficiency unspecified; force and output power are unknown"
                )
        except ValueError as exc:
            row.update(status="fail", reason=str(exc))
            errors.append(str(exc))
        samples.append(row)
    # Derivative extrema of slider-crank transmission can be inside the task interval.
    # Curves are sampled; numerical required-force comparisons use a bounded search
    # over every interval rather than treating sample maxima as global certificates.
    from scipy.optimize import minimize_scalar

    max_jacobian = None
    jacobian_bound = None
    extremum_angle = None
    turns: float | None = None
    close_time: float | None = None
    if not errors:
        states = [_state(candidate, x)[0] for x in (task.opening_min_mm, task.opening_max_mm)]
        if candidate.family == "linkage":
            jacobian_bound = linkage_jacobian_bound(candidate, min(states), max(states))
            angles = np.linspace(min(states), max(states), 65)
            maxima = [(abs(linkage_kinematics(candidate, float(a))[1]), float(a)) for a in angles]
            for a, b in pairwise(angles):
                peak = minimize_scalar(
                    lambda t: -abs(linkage_kinematics(candidate, t)[1]),
                    bounds=(a, b),
                    method="bounded",
                )
                if not peak.success:
                    errors.append("Linkage Jacobian extremum search failed")
                    break
                maxima.append((-float(peak.fun), float(peak.x)))
            max_jacobian, extremum_angle = max(maxima)
        else:
            max_jacobian = _state(candidate, task.opening_min_mm)[1]
            jacobian_bound = max_jacobian
        turns = abs(states[1] - states[0]) / (2 * math.pi)
        close_time = turns * 60 / request.drive.rpm
    else:
        turns = close_time = None
    document = None
    try:
        document = candidate_document(request, candidate.id)
    except ValueError as exc:
        errors.append(str(exc))
    estimated_minimum_force = (
        1000 * torque * efficiency.low / max_jacobian
        if max_jacobian and efficiency and not errors
        else None
    )
    minimum_force = (
        1000 * torque * efficiency.low / jacobian_bound
        if jacobian_bound and efficiency and not errors
        else None
    )
    force_witness = None
    if isinstance(candidate, LinkageCandidate) and extremum_angle is not None and not errors:
        force_witness = {
            "input_angle_deg": math.degrees(extremum_angle),
            "opening_mm": linkage_kinematics(candidate, extremum_angle)[0],
            "force_per_jaw_low_n": estimated_minimum_force,
            "scope": "Evaluated pose witness; not a certified global extremum",
        }
    width = document.variables["base_length"] if document else None
    checks: list[dict[str, Any]] = []
    for identifier, measured, limit, violation, missing in (
        (
            "force_per_jaw",
            minimum_force,
            task.required_force_per_jaw_n,
            estimated_minimum_force is not None
            and task.required_force_per_jaw_n is not None
            and estimated_minimum_force < task.required_force_per_jaw_n,
            "Force target or transmission efficiency is unspecified",
        ),
        (
            "close_time",
            close_time,
            task.max_close_time_s,
            close_time is not None
            and task.max_close_time_s is not None
            and close_time > task.max_close_time_s,
            "Closing-time target or reachable motion is unavailable",
        ),
        (
            "width",
            width,
            task.max_width_mm,
            width is not None and task.max_width_mm is not None and width > task.max_width_mm,
            "Width target or realization is unavailable",
        ),
    ):
        checks.append(
            {
                "id": identifier,
                "measured": measured,
                "limit": limit,
                "status": "fail"
                if violation
                else "unverified"
                if measured is None or limit is None or errors
                else "pass",
                "reason": "Exceeds the declared task limit"
                if violation
                else missing
                if measured is None or limit is None
                else "Conditional analytical screening only; physical requirement remains unverified",
            }
        )
        if identifier == "force_per_jaw" and candidate.family == "linkage":
            checks[-1]["method"] = "interval_factor_upper_jacobian_bound"
            if (
                checks[-1]["status"] == "pass"
                and minimum_force is not None
                and limit is not None
                and minimum_force < limit
            ):
                checks[-1].update(
                    status="unverified",
                    reason="Conservative force bound is below target without an evaluated violating pose; refine the bound or change the design",
                )
            elif checks[-1]["status"] == "pass":
                checks[-1]["reason"] = (
                    "Interval factor bounds cover the declared ideal kinematic branch with a floating-point outward margin; physical force remains unverified"
                )
            elif checks[-1]["status"] == "fail":
                checks[-1]["reason"] = (
                    "An evaluated ideal-kinematic pose falls below the force target"
                )
    checks.insert(
        0,
        {
            "id": "travel",
            "status": "fail" if errors else "pass",
            "measured": [task.opening_min_mm, task.opening_max_mm],
            "limit": None,
            "reason": "; ".join(dict.fromkeys(errors))
            if errors
            else "Task endpoints and declared branch are reachable in the ideal kinematics",
        },
    )
    limitations = [
        SCOPE,
        "Motion/force models neglect inertia, backlash, elastic deformation and unequal jaw loading. Input torque and speed must be jointly available.",
        "Width is the generated base design span; actual STEP extents, interference and manufacturing remain separate build checks.",
        "Editing the realized document invalidates this comparison; rerun synthesis with the changed parameters.",
        "Material/process, motor ratings, bearing reactions, fasteners, retention and physical validation are unresolved.",
    ]
    if candidate.family == "screw":
        limitations.append(
            "Opposite-handed screw and nut purchase envelopes have no thread geometry. Coupling is prescribed; engagement, friction model, backdrivability and self-locking are unverified."
        )
    elif candidate.family == "rack":
        limitations.append(
            "Native pinion and toothless purchase rack envelopes use prescribed motion. Tooth engagement, rack-to-carriage attachment and tooth strength are unverified."
        )
    else:
        limitations += [
            "Two mirrored slider-cranks use the existing planar closure graph; ideal synchronizer gearing, joint pins and axial retention are not physically realized.",
            "Force PASS uses an interval-factor upper Jacobian envelope with a 1e-12 relative/absolute outward margin. Numerical extrema are estimates and can supply failure witnesses, not PASS coverage. Floating-point arithmetic is not a formal directed-rounding proof. No toggle locking claim.",
        ]
    return {
        "id": candidate.id,
        "name": candidate.name,
        "family": candidate.family,
        "status": "fail"
        if any(c["status"] == "fail" for c in checks)
        else "pass"
        if all(c["status"] == "pass" for c in checks)
        else "unverified",
        "qualification_status": "unverified",
        "parameters": candidate.model_dump(),
        "document": document.model_dump() if document else None,
        "document_hash": document.fingerprint() if document else None,
        "curves": samples,
        "metrics": {
            "close_time_s": close_time,
            "input_turns": turns,
            "min_force_per_jaw_n": minimum_force,
            "estimated_min_force_per_jaw_n": estimated_minimum_force,
            "max_jacobian_bound_mm_per_rad": jacobian_bound,
            "max_jacobian_numerical_mm_per_rad": max_jacobian,
            "force_witness": force_witness,
            "width_mm": width,
            "input_power_w": torque * omega,
        },
        "checks": checks,
        "limitations": limitations,
        "unresolved": list(
            dict.fromkeys(
                [
                    *errors,
                    *(["Efficiency is unknown"] if efficiency is None else []),
                    "Source references require independent confirmation and applicability review",
                    "Physical assembly qualification remains unverified",
                ]
            )
        ),
        "dependencies": {
            "task_hash": digest(request.task.model_dump()),
            "drive_hash": digest(request.drive.model_dump()),
            "candidate_hash": digest(candidate.model_dump()),
            "source_claim_ids": candidate.source_claim_ids,
            "document_hash": document.fingerprint() if document else None,
        },
        "motion_method": "planar_loop_closure"
        if candidate.family == "linkage"
        else "prescribed_rotary_linear_coupling",
    }


def compare_mechanisms(request: SynthesisRequest) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_hash": digest(request.task.model_dump()),
        "request_hash": digest(request.model_dump()),
        "request": request.model_dump(),
        "candidates": [_compare(request, c) for c in request.candidates],
        "scope": SCOPE,
        "references": REFERENCES,
        "qualification_status": "unverified",
    }
