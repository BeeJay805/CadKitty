"""/patch orchestration — parametric edit-survival for standard-lane parts (ADR-0037).

``patch_part`` replays a stored LayoutSpec (the parametric model that rode the
/generate response) with ONE edited field, through the SAME deterministic
interpreter + compiler + strict 4-tier validator the standard lane uses. The
contract is edit-survival with no-false-PASS:

  - only the targeted spec field changes (``layout/patch.py`` guarantees it);
  - a new version ships ONLY on a strict validation PASS;
  - an unmappable or infeasible edit REFUSES with the reason (the LayoutError /
    failed checks surfaced verbatim) — it never silently re-cuts the part and
    never ships a wrong number.

Returns the same ``GenerateResult`` shape as /generate, so the HTTP layer and
the frontend treat a patched part exactly like a generated one (new artifacts,
new validation view, new spec riding the payload for the NEXT edit).
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.api.core import (
    GLB_NAME,
    STL_NAME,
    GenerateResult,
    StandardLaneOutcome,
    _bridge_kit,
    _confidence_view,
    _report_view,
    _stl_to_glb,
)
from forge_mind.api.schema import (
    Files,
    GenerateError,
    GenerateOk,
    GenerateRefused,
    PatchView,
)
from forge_mind.compiler.compile import compile_program
from forge_mind.kit import build_kit
from forge_mind.layout.interpreter import LayoutError
from forge_mind.layout.layoutspec import interpret_layout
from forge_mind.layout.patch import (
    SpecEdit,
    SpecPatchError,
    apply_spec_edit,
    apply_spec_structural_edit,
    editable_fields,
    map_edit,
    validate_spec_payload,
)
from forge_mind.llm.client import LLMError
from forge_mind.llm.provider import LLMClientLike
from forge_mind.validator.run import validate

_LOG = logging.getLogger(__name__)


def _refused(reason: str, lane: StandardLaneOutcome) -> GenerateResult:
    return GenerateResult(status="refused", refused=GenerateRefused(reason=reason), lane=lane)


def _error(stage: str, message: str, lane: StandardLaneOutcome) -> GenerateResult:
    return GenerateResult(
        status="error", error=GenerateError(stage=stage, message=message), lane=lane
    )


def patch_part(
    spec_data: dict[str, Any],
    edit: str,
    job_dir: Path,
    *,
    client: LLMClientLike | None = None,
) -> GenerateResult:
    """Apply one parametric edit to a stored spec and rebuild deterministically.

    Never raises: every outcome is a ``GenerateResult`` (ok / refused / error) with a
    ``StandardLaneOutcome`` attached for telemetry (``gen_path="standard:patch"``).
    """
    start = time.monotonic()
    job_dir.mkdir(parents=True, exist_ok=True)

    def _lane(outcome: str, *, error: str | None = None, cost: float = 0.0) -> StandardLaneOutcome:
        return StandardLaneOutcome(
            attempted=True,
            outcome=outcome,
            gen_path="standard:patch",
            error=error,
            duration_s=time.monotonic() - start,
            cost_usd=cost,
        )

    try:
        spec = validate_spec_payload(spec_data)
    except SpecPatchError as exc:
        return _error("spec", str(exc), _lane("failed_stage", error=str(exc)))

    cost = 0.0
    try:
        edit_op, usage = map_edit(spec, edit, client=client)
        cost = usage.cost_usd
    except LLMError as exc:
        return _error(
            "patch_map",
            f"could not interpret the edit: {exc}",
            _lane("failed_stage", error=str(exc)),
        )

    if not edit_op.mappable:
        reason = edit_op.reason or (
            "That edit doesn't map to a single parameter of this part. Try naming one "
            "dimension or option and the value you want."
        )
        return _refused(reason, _lane("refused", cost=cost))

    return _rebuild_from_edit(spec, edit_op, edit, job_dir, start, cost)


def _patch_lane(
    outcome: str, start: float, cost: float, *, error: str | None = None
) -> StandardLaneOutcome:
    """The patch lane's telemetry record (gen_path=standard:patch), shared by every /patch flow."""
    return StandardLaneOutcome(
        attempted=True,
        outcome=outcome,
        gen_path="standard:patch",
        error=error,
        duration_s=time.monotonic() - start,
        cost_usd=cost,
    )


def _rebuild_from_edit(
    spec: Any,
    edit_op: SpecEdit,
    edit_label: str,
    job_dir: Path,
    start: float,
    cost: float,
) -> GenerateResult:
    """Apply ONE already-mapped PARAMETER edit, then re-derive. Shared by the prose path
    (``patch_part``, LLM-mapped) and the structured field path (``patch_part_structured``, no LLM).
    Structural add/remove (``patch_part_structural``) applies its edit then calls ``_finish_rebuild``
    directly; all three share the same no-false-PASS spine."""
    try:
        new_spec, old_value, new_value = apply_spec_edit(spec, edit_op)
    except SpecPatchError as exc:
        return _refused(
            f"Can't apply that edit: {exc}", _patch_lane("refused", start, cost, error=str(exc))
        )
    return _finish_rebuild(
        new_spec,
        field_path=edit_op.field_path or "",
        op=edit_op.op or "set",
        old_value=old_value,
        new_value=new_value,
        edit_label=edit_label,
        job_dir=job_dir,
        start=start,
        cost=cost,
    )


def _finish_rebuild(
    new_spec: Any,
    *,
    field_path: str,
    op: str,
    old_value: Any,
    new_value: Any,
    edit_label: str,
    job_dir: Path,
    start: float,
    cost: float,
) -> GenerateResult:
    """Re-derive a part from an ALREADY-APPLIED spec edit and ship iff the strict gates PASS: interpret
    -> compile -> validate -> ship. The shared tail of every /patch flow (parameter + structural), so
    no-false-PASS is enforced in ONE place. An infeasible edit (LayoutError) or a gates-failing rebuild
    REFUSES and ships nothing; ``field_path``/``op``/old/new describe the change for the PatchView."""
    try:
        program = interpret_layout(new_spec)
    except LayoutError as exc:
        # InfeasibleRequestError included: the edit is honestly impossible at these
        # constraints. The stored part is untouched; the reason is the refusal.
        return _refused(
            f"Can't apply that edit: {exc}", _patch_lane("refused", start, cost, error=str(exc))
        )

    try:
        model = compile_program(program, job_dir)
    except Exception as exc:
        return _error(
            "compile",
            f"{type(exc).__name__}: {exc}",
            _patch_lane("failed_stage", start, cost, error=f"compile: {exc}"),
        )

    material = str(getattr(new_spec, "material", "PLA"))
    try:
        report = validate(model, program, material=material)
    except Exception as exc:
        return _error(
            "validate",
            f"{type(exc).__name__}: {exc}",
            _patch_lane("failed_stage", start, cost, error=f"validate: {exc}"),
        )
    if not report.passed:
        # No-false-PASS: an edit that breaks the strict gates never ships. Surface WHICH
        # checks failed so the user can adjust the edit.
        failed = (
            ", ".join(c.id for c in report.checks if c.status.value == "fail")
            or report.overall.value
        )
        return _refused(
            f"That edit makes the part fail validation ({failed}). The part was left unchanged.",
            _patch_lane("validation_failed", start, cost, error=failed),
        )

    stl_src = Path(model.stl_path)
    if not stl_src.exists():
        return _error(
            "artifacts", "STL missing after rebuild", _patch_lane("missing_artifacts", start, cost)
        )
    stl_dst = job_dir / STL_NAME
    if stl_src.resolve() != stl_dst.resolve():
        shutil.copy2(stl_src, stl_dst)
    if not _stl_to_glb(stl_dst, job_dir / GLB_NAME):
        return _error(
            "artifacts", "GLB conversion failed", _patch_lane("missing_artifacts", start, cost)
        )
    step_src = Path(model.step_path) if model.step_path else None
    if (
        step_src is not None
        and step_src.exists()
        and step_src.resolve() != (job_dir / "part.step").resolve()
    ):
        shutil.copy2(step_src, job_dir / "part.step")

    spec_dump = new_spec.model_dump(mode="json")
    (job_dir / "part.layoutspec.json").write_text(json.dumps(spec_dump, indent=2), encoding="utf-8")

    part_name = program.metadata.name
    try:
        kit = build_kit(program, material=material, report=report)
    except Exception:
        kit = None

    lane = _patch_lane("shipped", start, cost)
    _LOG.info(
        "patch SHIPPED: %s %s -> %r (cost=$%.4f in %.1fs)",
        field_path,
        op,
        new_value,
        cost,
        lane.duration_s or 0.0,
    )
    return GenerateResult(
        status="ok",
        ok=GenerateOk(
            prompt=edit_label,
            gen_path="standard:patch",
            cost_usd=round(cost, 4),
            validation=_report_view(report),
            kit=_bridge_kit(kit, part_name),
            files=Files(glb=GLB_NAME, stl=STL_NAME),
            fabrication_mode="print_3d",
            spec=spec_dump,
            patch=PatchView(
                field_path=field_path,
                op=op,
                old_value=_scalar(old_value),
                new_value=_scalar(new_value),
            ),
            # A patched part re-earned the deterministic lane's evidence (ADR-0041).
            confidence=_confidence_view(
                standard_lane_label(len(program.interfaces) + len(program.hardware_features))
            ),
        ),
        lane=lane,
    )


def patch_part_structured(
    spec_data: dict[str, Any],
    field_path: str,
    op: Literal["set", "increase", "decrease"],
    value: float | bool | str,
    job_dir: Path,
) -> GenerateResult:
    """Apply a STRUCTURED parametric edit (``field_path`` + ``op`` + ``value``) deterministically -- NO
    LLM. The slider/dropdown path: the UI picks the catalog field + value directly (from the
    ``/blueprint-fields`` catalog), so there is no edit-mapping call -- it is free and instant. Same
    deterministic rebuild + strict gates as ``patch_part``, and the same no-false-PASS refusals
    (unknown field, out-of-range, infeasible, or failing validation -> refuse, ship nothing)."""
    start = time.monotonic()
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec = validate_spec_payload(spec_data)
    except SpecPatchError as exc:
        return _error(
            "spec",
            str(exc),
            StandardLaneOutcome(
                attempted=True,
                outcome="failed_stage",
                gen_path="standard:patch",
                error=str(exc),
                duration_s=time.monotonic() - start,
                cost_usd=0.0,
            ),
        )
    edit_op = SpecEdit(mappable=True, field_path=field_path, op=op, value=value)
    return _rebuild_from_edit(spec, edit_op, f"{field_path} {op} {value}", job_dir, start, cost=0.0)


def patch_part_structural(
    spec_data: dict[str, Any],
    op: Literal["add", "remove"],
    job_dir: Path,
    *,
    index: int | None = None,
    placement: dict[str, Any] | None = None,
) -> GenerateResult:
    """Apply a STRUCTURAL edit -- ADD or REMOVE an interface/feature on the composable plate --
    deterministically, with NO LLM (ADR-0060 slice C; the structural edit ADR-0037 deferred). The
    blueprint's add/remove-feature path: the UI picks the interface to add (``ref`` + ``face``, with
    optional placement fields) or the ``index`` to remove, and the SAME deterministic rebuild + strict
    gates run as ``patch_part``. Same no-false-PASS refusals: unknown ref/face, min/max-length, a bad
    index, an infeasible add, or a failing rebuild -> refuse, ship nothing."""
    start = time.monotonic()
    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        spec = validate_spec_payload(spec_data)
    except SpecPatchError as exc:
        return _error("spec", str(exc), _patch_lane("failed_stage", start, 0.0, error=str(exc)))
    try:
        new_spec, field_path, old_value, new_value = apply_spec_structural_edit(
            spec, op, index=index, placement=placement
        )
    except SpecPatchError as exc:
        return _refused(
            f"Can't apply that edit: {exc}", _patch_lane("refused", start, 0.0, error=str(exc))
        )
    return _finish_rebuild(
        new_spec,
        field_path=field_path,
        op=op,
        old_value=old_value,
        new_value=new_value,
        edit_label=f"{op} interface ({field_path})",
        job_dir=job_dir,
        start=start,
        cost=0.0,
    )


def blueprint_fields(spec_data: dict[str, Any]) -> list[dict[str, Any]]:
    """The editable-field catalog for a stored LayoutSpec, serialized so a UI can render direct
    controls (number sliders, enum dropdowns, boolean toggles) for the structured blueprint.

    The READ side of the editable blueprint: the same closed catalog the deterministic patch path
    edits (``layout/patch.editable_fields``), exposed for the frontend. Raises
    :class:`SpecPatchError` if the spec is not a usable parametric model (never returns a fake set).
    """
    spec = validate_spec_payload(spec_data)
    return [
        {
            "path": f.path,
            "kind": f.kind,
            "current": f.current,
            "description": f.description,
            "choices": list(f.choices) if f.choices else None,
        }
        for f in editable_fields(spec)
    ]


def _scalar(value: Any) -> float | bool | str | None:
    """Project a spec leaf into the PatchView value union (defensive; leaves are scalars)."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        return float(value)
    return str(value)


__all__ = [
    "blueprint_fields",
    "patch_part",
    "patch_part_structural",
    "patch_part_structured",
]
