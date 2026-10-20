"""Failure taxonomy + root-cause classifier for end-to-end trials.

The classifier assigns ONE primary label per failed trial by earliest root cause:
it walks the pipeline from intent forward and stops at the first stage that
explains the failure. The exact failed check_id is recorded separately, so the
coarse label never loses precision.

Three labels are MANUAL-ONLY (the auto-classifier never assigns them because they
need ground-truth judgement): ``validator_false_positive``,
``validator_false_negative``, ``valid_but_mechanically_ugly``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from forge_mind.blueprint.grader import grade_blueprint
from forge_mind.intent.grader import grade_intent
from forge_mind.pipeline import PipelineResult
from forge_mind.validator.report import Check, ValidationReport


class FailureLabel(StrEnum):
    PASSED = "passed"
    INTENT_MISSING_REQUIREMENT = "intent_missing_requirement"
    BLUEPRINT_MISSING_REQUIREMENT = "blueprint_missing_requirement"
    BLUEPRINT_BAD_GEOMETRY_CONSTRAINT = "blueprint_bad_geometry_constraint"
    DSL_SCHEMA_INVALID = "dsl_schema_invalid"
    DSL_COMPILE_FAILED = "dsl_compile_failed"
    TOPOLOGY_WRONG = "topology_wrong"  # disconnected / non-manifold solid
    WALL_TOO_THIN = "wall_too_thin"  # a wall/land below the minimum thickness
    OVERHANG_TOO_STEEP = "overhang_too_steep"  # an unsupported overhang past the limit
    INTERFACE_FAILED = "interface_failed"
    ENVELOPE_FAILED = "envelope_failed"
    PRINTABILITY_FAILED = "printability_failed"  # other printability checks
    # Manual-only (need ground-truth judgement; never auto-assigned):
    VALIDATOR_FALSE_POSITIVE = "validator_false_positive"
    VALIDATOR_FALSE_NEGATIVE = "validator_false_negative"
    VALID_BUT_MECHANICALLY_UGLY = "valid_but_mechanically_ugly"
    UNKNOWN = "unknown"


# Labels the auto-classifier may produce.
AUTO_LABELS = frozenset(
    {
        FailureLabel.PASSED,
        FailureLabel.INTENT_MISSING_REQUIREMENT,
        FailureLabel.BLUEPRINT_MISSING_REQUIREMENT,
        FailureLabel.BLUEPRINT_BAD_GEOMETRY_CONSTRAINT,
        FailureLabel.DSL_SCHEMA_INVALID,
        FailureLabel.DSL_COMPILE_FAILED,
        FailureLabel.TOPOLOGY_WRONG,
        FailureLabel.WALL_TOO_THIN,
        FailureLabel.OVERHANG_TOO_STEEP,
        FailureLabel.INTERFACE_FAILED,
        FailureLabel.ENVELOPE_FAILED,
        FailureLabel.PRINTABILITY_FAILED,
        FailureLabel.UNKNOWN,
    }
)


@dataclass(frozen=True)
class Expectations:
    """Ground-truth requirements used to root-cause intent/blueprint failures."""

    required_interface_types: set[str]
    envelope_cap_mm: tuple[float, float, float]
    min_wall_mm: float
    material: str | None = None


def label_for_check(check_id: str) -> FailureLabel:
    """Map a single validator check_id to its failure label.

    The two dominant validate-stage failures get dedicated labels
    (``wall_too_thin``, ``overhang_too_steep``). ``topology_wrong`` is reserved
    for genuinely disconnected / non-manifold solids (the non-bbox ``mesh.*``
    checks). Specific check_ids are matched before the prefix catch-alls.
    """
    if check_id == "features.min_wall_thickness":
        return FailureLabel.WALL_TOO_THIN
    if check_id == "printability.overhang_max_angle":
        return FailureLabel.OVERHANG_TOO_STEEP
    if check_id.startswith("interfaces."):
        return FailureLabel.INTERFACE_FAILED
    if check_id == "mesh.bbox_within_max":
        return FailureLabel.ENVELOPE_FAILED
    if check_id.startswith("printability."):
        return FailureLabel.PRINTABILITY_FAILED
    if check_id.startswith("mesh.") or check_id.startswith("features."):
        return FailureLabel.TOPOLOGY_WRONG
    return FailureLabel.UNKNOWN


# Labels that were derived from a validator check (so they change when the
# check -> label mapping changes). Upstream root-cause labels are preserved.
_VALIDATE_DERIVED_LABELS = frozenset(
    {
        FailureLabel.TOPOLOGY_WRONG.value,
        FailureLabel.WALL_TOO_THIN.value,
        FailureLabel.OVERHANG_TOO_STEEP.value,
        FailureLabel.INTERFACE_FAILED.value,
        FailureLabel.ENVELOPE_FAILED.value,
        FailureLabel.PRINTABILITY_FAILED.value,
    }
)


def remap_validate_label(old_label: str, primary_check_id: str | None) -> FailureLabel:
    """Re-derive a recorded label under the current taxonomy WITHOUT re-running
    the pipeline.

    Only labels that came from a validator check are remapped (from the recorded
    ``primary_check_id``); upstream root-cause labels (intent / blueprint / dsl /
    compile / passed / unknown) are preserved exactly. Lets us re-render an old
    dataset accurately when the check -> label mapping changes.
    """
    if old_label in _VALIDATE_DERIVED_LABELS and primary_check_id:
        return label_for_check(primary_check_id)
    try:
        return FailureLabel(old_label)
    except ValueError:
        return FailureLabel.UNKNOWN


# Priority for choosing the PRIMARY failed check when several fail at once.
# Most fundamental first: a malformed solid explains downstream symptoms.
def _check_priority(check: Check) -> int:
    cid = check.id
    if cid.startswith("mesh.") and cid != "mesh.bbox_within_max":
        return 0  # topology of the solid
    if cid.startswith("features."):
        return 1  # thin wall / min feature
    if cid.startswith("interfaces."):
        return 2
    if cid == "mesh.bbox_within_max":
        return 3  # envelope
    if cid.startswith("printability."):
        return 4
    return 5


def primary_failed_check(report: ValidationReport) -> Check | None:
    """The single most fundamental failed check (by priority), or None."""
    failed = [c for c in report.checks if c.status.value == "fail"]
    if not failed:
        return None
    return min(failed, key=_check_priority)


def classify(result: PipelineResult, expectations: Expectations) -> FailureLabel:
    """Assign the primary failure label by earliest root cause."""
    if result.passed:
        return FailureLabel.PASSED

    # 1. Intent dropped a required interface?
    if result.intent is not None:
        ig = grade_intent(
            result.intent,
            required_interface_types=expectations.required_interface_types,
        )
        if any("missing interfaces" in f for f in ig.failures):
            return FailureLabel.INTENT_MISSING_REQUIREMENT

    # 2. Blueprint dropped a requirement or declared a bad constraint?
    if result.blueprint is not None:
        bg = grade_blueprint(
            result.blueprint,
            required_interface_types=expectations.required_interface_types,
            envelope_cap_mm=expectations.envelope_cap_mm,
            expected_min_wall_mm=expectations.min_wall_mm,
        )
        if any("missing required interfaces" in f for f in bg.failures):
            return FailureLabel.BLUEPRINT_MISSING_REQUIREMENT
        # Declared-constraint problems (envelope, wall, orientation, features).
        if bg.failures:
            return FailureLabel.BLUEPRINT_BAD_GEOMETRY_CONSTRAINT

    # 3. DSL generation / compile.
    if result.failed_stage == "dsl":
        return FailureLabel.DSL_SCHEMA_INVALID
    if result.failed_stage == "compile":
        return FailureLabel.DSL_COMPILE_FAILED

    # 4. Validation failure -> label by the primary failed check.
    if result.failed_stage == "validate" and result.report is not None:
        check = primary_failed_check(result.report)
        if check is not None:
            return label_for_check(check.id)

    return FailureLabel.UNKNOWN


__all__ = [
    "AUTO_LABELS",
    "Expectations",
    "FailureLabel",
    "classify",
    "label_for_check",
    "primary_failed_check",
    "remap_validate_label",
]
