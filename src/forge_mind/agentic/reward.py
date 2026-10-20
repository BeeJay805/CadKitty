"""The reward endpoint (ADR-0119): ONE callable contract from (code, brief) to a
measured reward vector, so search harnesses, defect-injection mining, and any future
RL trainer consume the SAME instrument the pipeline gates on -- never a parallel,
drift-prone reimplementation.

Composes the existing machinery verbatim: ``run_build123d_code`` (sandbox) then
``validate_authored_part`` (the fail-closed validator), then ``parse_violations``.
The scalar is a DOCUMENTED convenience for ranking/shaping -- the vector is the
product; anything that needs nuance reads the fields.

Scalar (v1, monotone, in [0, 1]):
    not executable ................ 0.0
    executable .................... 0.3
    + hard_ok ..................... +0.2
    + gate cleanliness ............ +0.5 * (1 - min(1, gating_issue_count / 4))
    (so a fully passing part scores exactly 1.0)
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.agentic.validate import validate_authored_part
from forge_mind.agentic.violations import Violation, parse_violations
from forge_mind.fabrication import FabricationMode

if TYPE_CHECKING:
    from forge_mind.agentic.brief import DesignBrief
    from forge_mind.materials import Material


@dataclass(frozen=True)
class RewardVector:
    """Measured outcome of one (code, brief) evaluation. ``scalar`` is derived."""

    executable: bool
    hard_ok: bool
    passed: bool
    violations: list[Violation] = field(default_factory=list)
    volume_mm3: float | None = None
    error_type: str | None = None
    error_message: str | None = None
    scalar: float = 0.0


def _scalar(executable: bool, hard_ok: bool, passed: bool, n_issues: int) -> float:
    if not executable:
        return 0.0
    score = 0.3 + (0.2 if hard_ok else 0.0)
    if passed:
        return 1.0
    return score + 0.5 * (1.0 - min(1.0, n_issues / 4.0))


def reward_vector(
    code: str,
    brief: DesignBrief,
    *,
    job_dir: Path | None = None,
    material: Material | None = None,
    prompt: str | None = None,
    timeout_s: float = 120.0,
) -> RewardVector:
    """Execute ``code`` in the sandbox, validate against ``brief``, return the vector.

    Never raises on authored-code failure: a crash comes back as executable=False with
    the sandbox's error fields. ``job_dir`` defaults to a throwaway temp dir (artifacts
    are not the point here; pass a real dir to keep them)."""
    if job_dir is None:
        job_dir = Path(tempfile.mkdtemp(prefix="fm_reward_"))
    res = run_build123d_code(
        code, job_dir, part_name="reward_probe", limits=SandboxLimits(timeout_s=timeout_s)
    )
    if not res.ok or res.stl_path is None or res.step_path is None:
        return RewardVector(
            executable=False,
            hard_ok=False,
            passed=False,
            error_type=str(res.error_type) if res.error_type else None,
            error_message=res.error_message,
            scalar=0.0,
        )
    val = validate_authored_part(
        res.stl_path,
        res.step_path,
        brief,
        fabrication_mode=FabricationMode.PRINT_3D,
        material=material,
        author_code=code,
        prompt=prompt,
    )
    violations = parse_violations(val.issues)
    return RewardVector(
        executable=True,
        hard_ok=val.hard_ok,
        passed=val.passed,
        violations=violations,
        volume_mm3=val.volume_mm3,
        scalar=_scalar(True, val.hard_ok, val.passed, len(violations)),
    )


__all__ = ["RewardVector", "reward_vector"]
