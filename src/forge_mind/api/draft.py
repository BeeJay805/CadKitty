"""The blueprint PREGEN: a fast, cheap DRAFT of the part BEFORE the validated build.

Founder direction (2026-07-02): the blueprint step shows real, mutable geometry before a
credit is spent. A draft is a SIMPLE-PROMPT author pass (small model, minimal code,
PARAMS-first) executed in the SAME sandbox as real builds -- honest geometry, honestly
labeled DRAFT (no validator gate ran, no critic, no repair loop). The PARAMS dict comes
back with the mesh, and because every draft job dir carries its ``program.py``, the
job-scoped parametric endpoints (/jobs/{job}/params + /jobs/{job}/rebuild) mutate a draft
and a finished part through one mechanism (ADR-0073 extract/override machinery).

Deliberately NOT here: validation, judging, repair, credits. A draft that fails to build
returns a plain error -- it never pretends.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from forge_mind.agentic.author import clean_code
from forge_mind.agentic.params import ParamSpec, extract_params, override_params
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.config import FABLE, HAIKU, OPUS, SONNET
from forge_mind.llm.provider import LLMClientLike, make_client

PROGRAM_NAME = "program.py"  # the job dir's script -- same name the agentic loop persists

# The SIMPLE draft prompt (founder-requested). It asks for the least code that produces an
# honest massing of the request, with every editable dimension in PARAMS so the blueprint's
# fields map 1:1 onto the script. Kept short on purpose: the draft's job is speed + shape,
# not detail -- the full author + validator pipeline owns the real part.
DRAFT_SYSTEM_PROMPT = """\
You write a short build123d Python script that DRAFTS a part for a quick preview.

Rules:
- First statement: PARAMS = {...} with every dimension a user might edit, as plain
  numeric literals in mm. Descriptive snake_case names ending in _mm (angles: _deg).
  Counts and other structure are ints; dimensions are floats.
- Read every dimension from PARAMS (e.g. PARAMS["plate_width_mm"]); never repeat a
  number inline.
- Use only build123d. Bind the finished single solid to a variable named `result`.
- Keep it minimal: primitive solids, holes, and booleans that match the request. No
  fillets, chamfers, threads, or decoration unless the request names them.
- No imports besides build123d, no file I/O, no prints, under 40 lines.
- This is a rough draft for a preview, not the final validated part: prefer simple,
  correct massing over detail.
"""

DRAFT_MAX_TOKENS = 1200
_DRAFT_TIMEOUT_S = 60.0  # drafts are minimal scripts; a stuck build fails fast


class DraftCode(BaseModel):
    """The draft author's output: just the script."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., description="The complete build123d draft script.")


class DraftError(Exception):
    """A draft could not be authored or built; message is user-safe."""


def draft_model() -> str:
    """The draft author model: ``FORGE_DRAFT_MODEL`` (alias or full id), default HAIKU --
    the draft is a speed/cost play; the full pipeline keeps the frontier author."""
    raw = os.environ.get("FORGE_DRAFT_MODEL", "").strip().lower()
    aliases = {"opus": OPUS, "fable": FABLE, "sonnet": SONNET, "haiku": HAIKU}
    return aliases.get(raw, raw or HAIKU)


@dataclass(frozen=True)
class DraftBuild:
    """A built draft: the script (as written to the job dir) and its PARAMS."""

    code: str
    params: dict[str, float | int]


def author_draft(
    prompt: str, *, client: LLMClientLike | None = None, model: str | None = None
) -> tuple[str, ParamSpec]:
    """One simple-prompt author pass -> (code, params). Fails CLOSED on a script without a
    usable PARAMS contract -- a draft whose dimensions can't be edited is not a blueprint."""
    llm = client or make_client()
    out, _usage = llm.generate_structured(
        system=DRAFT_SYSTEM_PROMPT,
        user=prompt.strip(),
        output_model=DraftCode,
        model=model or draft_model(),
        max_tokens=DRAFT_MAX_TOKENS,
    )
    code = clean_code(out.code)
    spec = extract_params(code)
    if spec is None:
        raise DraftError("The draft came back without an editable PARAMS contract.")
    return code, spec


def build_into_job(
    code: str, job_dir: Path, *, glb_writer: Callable[[Path, Path], bool]
) -> DraftBuild:
    """Execute ``code`` in the sandbox and leave a complete, parametric job dir behind:
    part.stl + part.step (sandbox), part.glb (``glb_writer``), and program.py (the script,
    so /jobs/{job}/params + /jobs/{job}/rebuild work on this job).

    ``glb_writer(stl: Path, glb: Path) -> bool`` is injected (api.core._stl_to_glb in
    production) so this module stays import-light and the tests stay mesh-free."""
    result = run_build123d_code(
        code, job_dir, part_name="part", limits=SandboxLimits(timeout_s=_DRAFT_TIMEOUT_S)
    )
    if not result.ok:
        raise DraftError(f"The draft failed to build ({result.error_type or 'build error'}).")
    stl = job_dir / "part.stl"
    if not stl.is_file():
        raise DraftError("The draft built but produced no mesh.")
    if not glb_writer(stl, job_dir / "part.glb"):
        raise DraftError("The draft mesh could not be converted for the viewer.")
    (job_dir / PROGRAM_NAME).write_text(code, encoding="utf-8")
    spec = extract_params(code)
    if spec is None:  # unreachable for author_draft output; guards hand-written callers
        raise DraftError("The draft script lost its PARAMS contract.")
    return DraftBuild(code=code, params=dict(spec.values))


def job_params(job_dir: Path) -> dict[str, float | int] | None:
    """The PARAMS of a job's stored script, or None when the job has no parametric contract
    (no program.py, or a script without a literal PARAMS dict). None = feature unavailable,
    never an error -- the UI simply doesn't offer the panel."""
    path = job_dir / PROGRAM_NAME
    if not path.is_file():
        return None
    spec = extract_params(path.read_text(encoding="utf-8"))
    return dict(spec.values) if spec else None


def rebuild_job(
    job_dir: Path, overrides: dict[str, float], *, glb_writer: Callable[[Path, Path], bool]
) -> DraftBuild:
    """Re-execute a job's stored script with PARAMS ``overrides`` (the parametric edit).

    Fail-closed on unknown or non-float names: a silent ignore would tell the user their
    edit applied when it did not. Structure (int params) is frozen by contract (ADR-0073)."""
    path = job_dir / PROGRAM_NAME
    if not path.is_file():
        raise DraftError("This part has no stored parametric script.")
    code = path.read_text(encoding="utf-8")
    spec = extract_params(code)
    if spec is None:
        raise DraftError("This part's script has no editable PARAMS contract.")
    editable = set(spec.float_names)
    unknown = sorted(set(overrides) - editable)
    if unknown:
        raise DraftError(
            f"Unknown or non-editable parameter(s): {', '.join(unknown)}. "
            f"Editable: {', '.join(sorted(editable))}."
        )
    new_code = override_params(code, spec, {k: float(v) for k, v in overrides.items()})
    return build_into_job(new_code, job_dir, glb_writer=glb_writer)
