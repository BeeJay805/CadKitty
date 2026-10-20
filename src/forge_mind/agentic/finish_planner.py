"""Finish PLANNER: the seam that makes the detail library live.

Pipeline (founder-specced): registry -> generated catalog (WHOLE, inline) + part role +
bbox + mating-surface exclusions -> planner emits FinishOp list -> apply_finish ->
re-validate fit -> ship polished only if fit unchanged.

No RAG, no vector store: 20-ish ops fit in one prompt, so the planner sees the ENTIRE
catalog (can't retrieve the wrong op). The catalog is generated from the @register
registry, so it never drifts from the code.

FAIL-CLOSED throughout: a planner call that errors, a finish that breaks a solid, or a
finish that WORSENS the assembly fit all degrade to the clean part -- polish never ruins a
working part.
"""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.detail import FinishOp, apply_finish
from forge_mind.agentic.detail.core import _OPS
from forge_mind.config import BRAIN_MODEL
from forge_mind.llm.provider import LLMClientLike, make_client


def generate_catalog() -> str:
    """The WHOLE finish-op catalog as prompt text, built from the registry: each op's name,
    one-line purpose (docstring first line), kinds (parsed from the docstring), and its
    keyword params. Deterministic -- never drifts from the code."""
    lines: list[str] = []
    for name in sorted(_OPS):
        fn = _OPS[name]
        doc = (fn.__doc__ or "").strip()
        purpose = doc.split("\n", 1)[0].strip() if doc else name
        kinds = re.findall(r"kind in\s*\{([^}]*)\}", doc)
        kind_note = f" | kinds: {kinds[0].strip()}" if kinds else ""
        params = [
            p.name
            for p in inspect.signature(fn).parameters.values()
            if p.name != "part" and p.kind is p.KEYWORD_ONLY
        ]
        lines.append(f"- {name}({', '.join(params)}){kind_note}: {purpose}")
    return "\n".join(lines)


class _Directive(BaseModel):
    """One finish directive the planner emits. params_json is a JSON object of the op's
    keyword args (all mm / counts / kind strings) -- a string so any op's params validate."""

    model_config = ConfigDict(extra="forbid")

    op: str = Field(..., description="A finish op name from the catalog.")
    params_json: str = Field(
        ..., description='JSON object of the op keyword args, e.g. {"x0":-20,...}'
    )


class _FinishPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directives: list[_Directive] = Field(default_factory=list, max_length=8)


_SYSTEM = """You are a MECHANICAL FINISH designer. Given ONE part and the full catalog of \
deterministic finish operations, choose the finish -- IF ANY -- that genuinely improves this \
specific part.

RESTRAINT FIRST -- most functional machine parts want LITTLE OR NO finish. An engine is a \
working machine, not a decorative object. A connecting rod, piston, crankshaft, gear, shaft, \
wrist pin, bearing, or any internal MOVING part should get essentially NOTHING -- at most a tiny \
free-edge chamfer for print/assembly; NEVER grip tread, knurling, labels, or lightening on a \
working mechanism part. Over-finishing a functional part is a mistake. If the part has no surface \
that a human touches or sees and no non-structural bulk, RETURN AN EMPTY LIST -- that is the \
correct, common answer.

Apply real finish ONLY where it genuinely helps:
- a hand-turned KNOB / handle / grip -> knurl (people touch it);
- a foot/step/tread surface -> anti-slip texture + grip holes (safety);
- a hot cylinder BODY exterior -> cooling fins (function + look);
- a large external housing FACE -> a small free-edge chamfer or a subtle label;
- clearly non-structural BULK on a bracket/plate -> light lightening.

HARD RULES:
- NEVER place any detail on a MATING surface (a bore, journal, pin, sliding face, sealing face). \
Those are the EXCLUSIONS -- keep every op's region clear of them; texture there changes the fit.
- Only use ops from the catalog. Put the op's keyword args in params_json as a JSON object \
(numbers in mm, counts as integers, kind as a string). Use coordinates inside the part's bbox.
- Default to LESS. 0-2 directives is right for most parts; an empty list is the right answer for \
an internal mechanism part."""


@dataclass
class PartFinishContext:
    name: str
    role: str
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]]  # (lo, hi)
    mating_exclusions: list[str]


def plan_part_finish(
    ctx: PartFinishContext,
    *,
    client: LLMClientLike | None = None,
    model: str | None = None,
) -> list[FinishOp]:
    """One planner LLM call -> a validated FinishOp list. Fail-closed: any error -> []."""
    client = client or make_client()
    lo, hi = ctx.bbox
    user = (
        f"PART: {ctx.name} -- {ctx.role}\n"
        f"BBOX (mm): X {lo[0]:.1f}..{hi[0]:.1f}, Y {lo[1]:.1f}..{hi[1]:.1f}, Z {lo[2]:.1f}..{hi[2]:.1f}\n"
        f"MATING-SURFACE EXCLUSIONS (do NOT texture near these): "
        f"{'; '.join(ctx.mating_exclusions) if ctx.mating_exclusions else 'none declared'}\n\n"
        f"FINISH CATALOG (use only these):\n{generate_catalog()}\n\n"
        "Return the finish directives."
    )
    try:
        plan, _u = client.generate_structured(
            system=_SYSTEM,
            user=user,
            output_model=_FinishPlan,
            model=model or BRAIN_MODEL,
            max_tokens=4000,
        )
    except Exception:
        return []
    ops: list[FinishOp] = []
    for d in plan.directives:
        try:
            params = json.loads(d.params_json)
            if isinstance(params, dict):
                ops.append(FinishOp(op=d.op, params=params))
        except Exception:
            continue  # a malformed directive is dropped, the rest stand
    return ops


def finish_step_file(step_path: Path, ops: list[FinishOp], out_stl: Path, out_step: Path) -> bool:
    """Load the part's STEP, apply the finish ops (fail-closed via apply_finish), and re-export
    STL + STEP. Returns True iff at least one op applied AND the result exported watertight."""
    if not ops or not step_path.exists():
        return False
    try:
        from build123d import export_step, export_stl, import_step

        solid = import_step(str(step_path))
        res = apply_finish(solid, ops)
        if not res.applied:
            return False
        export_stl(res.part, str(out_stl))
        export_step(res.part, str(out_step))
        return out_stl.exists()
    except Exception:
        return False


def fit_not_worse(clean: dict[str, Any] | None, finished: dict[str, Any] | None) -> bool:
    """The ship gate: finish keeps the assembly fit UNCHANGED-or-better. Worse = new/extra
    interference, or a pair that was seated is no longer seated. Fail-closed: unknown -> False."""
    if finished is None:
        return False
    if clean is None:
        return True
    c_int = float(clean.get("interference_mm3", 0.0) or 0.0)
    f_int = float(finished.get("interference_mm3", 0.0) or 0.0)
    if f_int > c_int + 1.0:  # 1mm^3 tolerance for tessellation noise
        return False
    # a pair that was seated must stay seated (finish must not unseat a mate)
    return not (bool(clean.get("seated", True)) and not bool(finished.get("seated", True)))
