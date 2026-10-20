"""ValidationReport schema — the typed output of the validator.

Granularity is deliberate: the report carries one ``Check`` per assertion (not a
single boolean), grouped into tiers, plus structured ``Fix`` hints that the
Phase 7 Reviser consumes directly. Everything is JSON-serializable so a report
can be saved per benchmark trial and diffed across runs.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.types import Vec3


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    # ADVISORY-FIRST evidence (ADR-0035 B1): a measured observation that gates NOTHING by
    # construction -- ``aggregate_status`` ignores it (overall/passed cannot flip), and
    # ``passes_criteria`` does not block on it. New measured checks (declared-vs-measured
    # conformance, concavity, scale) ship as INFO and promote to FAIL only after the
    # pre-registered criterion in ADR-0035 is met. Distinct from WARN, which DOES make a
    # report non-PASS.
    INFO = "info"


class Evidence(StrEnum):
    """HOW a non-PASS status was reached — so a consumer that must fail OPEN on absence of
    proof can tell a proven violation from a fail-closed one.

    This codebase fails CLOSED everywhere (the constitution's rule), which is right for
    gating: an unmeasurable overlap and a check that never ran both block a clean ship. But
    the non-shipping strict tier (ADR-0126) has the opposite law — "an unrun check must never
    manufacture a blocker" — and the two are indistinguishable from ``status == FAIL`` alone.
    That is what this field records.

    ``None`` (the default, and every check that has not opted in) means UNCLASSIFIED, and the
    strict tier treats it as not-proven. So the tri-state fails open by construction: a check
    must positively claim ``MEASURED`` to reach a non-shipping disposition.
    """

    MEASURED = "measured"  # a measured quantity violating a DECLARED requirement
    ASSUMED = "assumed"  # rests on a substituted default, or a quantity that could not be measured


class Tier(StrEnum):
    MESH = "mesh"
    FEATURES = "features"
    PRINTABILITY = "printability"
    INTERFACES = "interfaces"
    ASSEMBLY = "assembly"


class Locator(BaseModel):
    """WHERE on the part a check's finding sits, in COORDINATES — so the viewer can point at it.

    Coordinate-based on purpose: raw mesh face/vertex indices would NOT survive the STL->GLB export
    boundary (the validator measures the Z-up STL; the viewer loads a separately-tessellated Y-up
    GLB), so a "highlight face N" contract would silently point at the wrong place. A point and/or a
    bounding box in a STATED frame does survive — the viewer converts the frame and draws a
    marker/box. Optional on every Check: present ONLY when the check genuinely computed a locus
    (never fabricated for a check that doesn't know where the problem is)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    frame: str = Field(
        default="stl_z_up",
        description="Coordinate frame of the points: the validator's mesh frame (Z-up mm). The "
        "viewer converts to its own Y-up space.",
    )
    point: Vec3 | None = Field(
        default=None, description="A single locus (e.g. the thinnest-wall sample)."
    )
    bbox_min: Vec3 | None = Field(
        default=None, description="Lower corner of the finding's region, if it is a region."
    )
    bbox_max: Vec3 | None = Field(default=None, description="Upper corner of the region.")
    lens: str | None = Field(
        default=None,
        description="The viewer inspection lens that visualizes this finding "
        "(e.g. 'overhang', 'thickness'), if one applies.",
    )


class Check(BaseModel):
    """A single validator assertion and its result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., description="Stable dotted id, e.g. 'mesh.watertight'.")
    tier: Tier
    status: CheckStatus
    message: str
    expected: str | None = Field(default=None, description="Expected value, formatted.")
    actual: str | None = Field(default=None, description="Measured value, formatted.")
    tolerance: str | None = None
    op_ref: str | None = Field(default=None, description="Related op_id, if any.")
    locator: Locator | None = Field(
        default=None,
        description="Where the finding is, coordinate-based; present only when the check computed it.",
    )
    evidence: Evidence | None = Field(
        default=None,
        description="How a non-PASS status was reached (measured violation vs fail-closed); "
        "None = unclassified, which the strict non-shipping tier treats as not-proven.",
    )

    @property
    def ok(self) -> bool:
        return self.status != CheckStatus.FAIL


class Fix(BaseModel):
    """A structured repair hint for the Reviser. Not prose — a target + change."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    op_id: str | None = Field(default=None, description="Operation to modify (None if structural).")
    field: str | None = Field(default=None, description="Field of that op to change.")
    reason: str
    suggested_change: dict[str, float | int | str | bool] = Field(default_factory=dict)


class TierResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tier: Tier
    status: CheckStatus
    checks: list[Check]

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.FAIL)


class PhysicsMetrics(BaseModel):
    """Measured physical quantities of the compiled body (ADR-0019) — REPORTED, not gated.
    The first "Einstein tool": ``mass_g`` is the quantity a future optimize loop minimizes.

    HONEST CAVEAT: ``mass_g`` = volume x density assumes 100% SOLID infill, so it is a
    COMPARATIVE estimate (good for "variant A vs B"), not a scale-accurate print mass.
    ``center_of_mass`` assumes uniform density.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    volume_mm3: float
    mass_g: float
    center_of_mass: Vec3
    material: str
    density_g_cm3: float


class StepProof(BaseModel):
    """STEP export integrity proof (ISO 10303-49 spirit) — REPORTED, never gated. Re-imports the
    exported STEP and confirms a downstream CAD tool (SolidWorks / Fusion) reads back the SAME solid:
    volume, surface area, and centroid match the source body within tolerance. A mismatch means the
    STEP write or the geometry is lossy — surfaced as a proof, NOT a refusal (the STL the user prints
    is untouched, and the B-rep lanes are exactly where ADR-0024's "STEP + proof" promise is real)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    matches: bool
    volume_mm3: float
    area_mm2: float
    centroid: Vec3
    volume_rel_err: float
    area_rel_err: float
    centroid_dist_mm: float
    # Named-B-rep fidelity (ADR-0037 rung 2), all additive + best-effort (None = could not
    # inspect, which never demotes). A volume-matching TESSELLATED lump can no longer read as
    # a faithful STEP: `matches` is demoted only when a field below AFFIRMATIVELY detects a
    # problem (no new false-FAIL risk from an unreadable file).
    schema_id: str | None = None  # "AP214" / "AP242" / first schema token
    is_brep: bool | None = None  # analytic B-rep entities present (MANIFOLD_SOLID_BREP + faces)
    tessellated: bool | None = None  # tessellated/triangulated entities present in the file
    product_name: str | None = None  # what an engineer's CAD tree shows
    named: bool | None = None  # product carries the PART's name, not an OCC default
    solid_count: int | None = None  # re-imported topology
    face_count: int | None = None
    topology_matches: bool | None = None  # source vs re-import: solid + face counts equal


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: CheckStatus
    tiers: dict[Tier, TierResult]
    checks: list[Check]
    suggested_fixes: list[Fix] = Field(default_factory=list)
    metrics: PhysicsMetrics | None = Field(
        default=None, description="Measured physics (mass/COM); reported, never a gate."
    )
    step_proof: StepProof | None = Field(
        default=None, description="STEP roundtrip integrity proof; reported, never a gate."
    )

    # ------------------------------------------------------------------
    # Query helpers (used by tests and the future harness)
    # ------------------------------------------------------------------

    def failed_check_ids(self) -> list[str]:
        return [c.id for c in self.checks if c.status == CheckStatus.FAIL]

    def check(self, check_id: str) -> Check | None:
        for c in self.checks:
            if c.id == check_id:
                return c
        return None

    @property
    def passed(self) -> bool:
        return self.overall == CheckStatus.PASS

    def warned_check_ids(self) -> list[str]:
        return [c.id for c in self.checks if c.status == CheckStatus.WARN]

    def info_check_ids(self) -> list[str]:
        """Advisory measured-evidence checks (ADR-0035 B1) -- never gate, always recorded."""
        return [c.id for c in self.checks if c.status == CheckStatus.INFO]

    @staticmethod
    def _matches(check_id: str, pattern: str) -> bool:
        """A check id matches a required/accepted pattern if it is equal OR ends with
        ``.<pattern>``. The dotted-suffix form lets a bench require a check by KIND
        (``bore_grips_shaft``) without pinning the part id (``interfaces.arm.\
bore_grips_shaft`` vs ``...output_arm...``) -- the LLM names parts freely, so
        id-exact requirements would be brittle. An exact id still works unchanged."""
        return check_id == pattern or check_id.endswith("." + pattern)

    def passes_criteria(
        self,
        *,
        required_checks: list[str] | None = None,
        accepted_warns: list[str] | None = None,
    ) -> bool:
        """Honest pass with explicit allowances. Stricter than ``passed`` in one way and
        more lenient in exactly one controlled way:

        - ANY FAIL -> not passing.
        - Every entry in ``required_checks`` must match a present-AND-PASS check (exact
          id or dotted-suffix kind -- see ``_matches``).
        - A WARN blocks UNLESS it matches an ``accepted_warns`` entry -- an explicitly
          declared known-deferred warning (e.g. an un-sweepable hardware-anchored motion
          check). An *unaccepted* WARN still blocks: warnings never pass silently.

        With both args empty, a report with any WARN still fails (same as ``passed``),
        so existing callers are unaffected; only a bench that opts in changes behaviour.
        """
        if self.failed_check_ids():
            return False
        accepted = accepted_warns or []
        for c in self.checks:
            if c.status == CheckStatus.WARN and not any(self._matches(c.id, a) for a in accepted):
                return False
        passing = [c.id for c in self.checks if c.status == CheckStatus.PASS]
        return all(any(self._matches(cid, rc) for cid in passing) for rc in (required_checks or []))


def aggregate_status(checks: list[Check]) -> CheckStatus:
    """FAIL if any check failed; else WARN if any warned; else PASS. INFO checks are advisory
    evidence and never move the aggregate (ADR-0035 B1)."""
    if any(c.status == CheckStatus.FAIL for c in checks):
        return CheckStatus.FAIL
    if any(c.status == CheckStatus.WARN for c in checks):
        return CheckStatus.WARN
    return CheckStatus.PASS


def build_report(
    checks: list[Check],
    fixes: list[Fix],
    metrics: PhysicsMetrics | None = None,
    step_proof: StepProof | None = None,
) -> ValidationReport:
    """Group checks by tier and assemble the top-level report."""
    tiers: dict[Tier, TierResult] = {}
    for tier in Tier:
        tier_checks = [c for c in checks if c.tier == tier]
        if not tier_checks:
            continue
        tiers[tier] = TierResult(
            tier=tier,
            status=aggregate_status(tier_checks),
            checks=tier_checks,
        )
    return ValidationReport(
        overall=aggregate_status(checks),
        tiers=tiers,
        checks=checks,
        suggested_fixes=fixes,
        metrics=metrics,
        step_proof=step_proof,
    )


__all__ = [
    "Check",
    "CheckStatus",
    "Evidence",
    "Fix",
    "Locator",
    "PhysicsMetrics",
    "StepProof",
    "Tier",
    "TierResult",
    "ValidationReport",
    "aggregate_status",
    "build_report",
]
