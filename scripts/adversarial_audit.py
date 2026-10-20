"""Adversarial induced-distribution audit (ADR-0012 section 7, Engine Month E2/W3.3).

The seeded-mutant suite measures a STATIC defect distribution; the optimizing author moves
off it. This audit measures the distribution the REAL loop induces: take every gate-PASSED
part from a frontier_v1 run and have an INDEPENDENT judge -- a different model than the
author, with the original prompt AND the compiled brief in hand -- score brief compliance
per part. The published number (judged-noncompliant / gate-passed) is month 2's calibration
baseline: it predicts whether the physical loop's labels start clean. NO threshold this
month, by design -- the number is reported, not gated on.

Independence: the frontier author arm is OPUS (row-24 parity economy); the judge here is
SONNET. Founder spot-check remains part of the protocol (the per-part JSON + renders are
written for exactly that).

Fail-closed accounting: a gate-PASSED part whose render or brief is missing is counted
UNJUDGEABLE and reported loudly -- excluded from the denominator, never silently compliant.

Args: results_json [judge_model]
  results_json  benchmarks/results/passk_opus_<stamp>/results.json from a frontier_v1 run
  judge_model   sonnet|opus|fable (default sonnet; refuses to match the run's author arm)
Run:  $env:FORGE_LLM_PROVIDER='claude_code'
      uv run --no-sync python -u scripts/adversarial_audit.py benchmarks/results/passk_opus_<stamp>/results.json
Writes adversarial_audit.json next to the input results.json.
"""

from __future__ import annotations

import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.agentic.brief import DesignBrief  # noqa: E402
from forge_mind.config import FABLE, OPUS, SONNET  # noqa: E402
from forge_mind.llm.provider import make_client  # noqa: E402

_MODELS = {"sonnet": SONNET, "opus": OPUS, "fable": FABLE}
_JUDGE_MAX_TOKENS = 1200


class ComplianceVerdict(BaseModel):
    """Independent brief-compliance verdict on one gate-PASSED part."""

    model_config = ConfigDict(extra="forbid")

    complies: bool = Field(
        ...,
        description=(
            "True ONLY if the part satisfies every requirement stated in the prompt that "
            "the image can verify: object identity, named features present, proportions "
            "plausible against the stated dimensions."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(...)
    violations: list[str] = Field(
        default_factory=list,
        description="Prompt requirements the part visibly fails. Empty iff complies.",
    )
    unverifiable: list[str] = Field(
        default_factory=list,
        description="Prompt requirements the render cannot verify (exact dims, tolerances).",
    )
    reason: str = Field(..., description="One concise paragraph naming the deciding evidence.")


_SYSTEM = """You are an INDEPENDENT auditor for an AI CAD pipeline, reviewing parts its \
validator already gate-PASSED. Your job is to catch what the validator missed, so be \
skeptical: you are the last line before these parts are used to calibrate a physical \
print loop, and a wrongly-blessed part poisons that calibration.

You are shown ONE image: a six-panel render (isometric, front, right, rear, top, bottom) \
of a single 3D part, plus the ORIGINAL USER PROMPT and the pipeline's compiled brief.

DECIDE per the schema. Rules:
- Judge against the PROMPT first; the brief is context (it may itself have dropped a
  requirement -- a part matching the brief but not the prompt does NOT comply).
- complies=true ONLY if every visually-verifiable prompt requirement is met: correct
  object, every named feature present (count holes and slots panel by panel),
  proportions consistent with the stated dimensions.
- List requirements the image genuinely cannot verify under `unverifiable` (exact
  dimensions, thread specs) -- do NOT fail a part for those, and do NOT silently
  assume them met either.
- A missing or miscounted named feature is always a violation, never unverifiable."""


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.959964
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def _brief_context(job_dir: Path) -> str:
    """The compiled brief as judge context; absent brief.json degrades to a stated fact."""
    bp = job_dir / "brief.json"
    if not bp.exists():
        return "(no compiled brief was persisted for this part)"
    try:
        brief = DesignBrief.model_validate_json(bp.read_text(encoding="utf-8"))
    except Exception:
        return "(brief.json present but unparseable)"
    lines = [f"object: {brief.object_name}", f"looks like: {brief.visual_description}"]
    lines.extend(f"feature: {f.name} -- {f.purpose}" for f in brief.features)
    return "\n".join(lines)


def _judge_one(
    client: Any, model_id: str, prompt: str, job_dir: Path, render: Path
) -> ComplianceVerdict:
    user = (
        "ORIGINAL USER PROMPT:\n"
        f"{prompt}\n\n"
        "PIPELINE-COMPILED BRIEF (context only -- the prompt is authoritative):\n"
        f"{_brief_context(job_dir)}\n\n"
        "Here is the six-panel render of the part the validator gate-PASSED."
    )
    verdict, _usage = client.generate_structured_with_image(
        system=_SYSTEM,
        user=user,
        image_png=render.read_bytes(),
        output_model=ComplianceVerdict,
        model=model_id,
        max_tokens=_JUDGE_MAX_TOKENS,
    )
    return verdict


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    results_path = Path(sys.argv[1])
    judge_arm = (sys.argv[2] if len(sys.argv) > 2 else "sonnet").strip().lower()
    judge_id = _MODELS.get(judge_arm)
    if judge_id is None:
        raise SystemExit(f"unknown judge {judge_arm!r}; use sonnet|opus|fable")

    data = json.loads(results_path.read_text(encoding="utf-8"))
    author_id = str(data.get("arm_model", ""))
    if judge_id == author_id:
        raise SystemExit(
            f"judge model {judge_arm!r} == the run's author arm -- ADR-0012 section 7 requires an "
            "INDEPENDENT judge; pick a different model"
        )
    run_dir = results_path.parent
    passed = [r for r in data.get("results", []) if r.get("passed")]
    print(f"[adversarial] {len(passed)} gate-PASSED parts from {run_dir.name}", flush=True)
    if not passed:
        raise SystemExit("no gate-PASSED parts in this run -- nothing to audit")

    client = make_client()
    records: list[dict[str, Any]] = []
    out_path = run_dir / "adversarial_audit.json"

    def _save() -> None:
        judged = [r for r in records if r["status"] == "judged"]
        bad = [r for r in judged if not r["complies"]]
        unjudgeable = [r for r in records if r["status"] == "unjudgeable"]
        n, k = len(judged), len(bad)
        out_path.write_text(
            json.dumps(
                {
                    "source_results": str(results_path),
                    "benchmark": data.get("benchmark"),
                    "author_arm": author_id,
                    "judge_arm": judge_id,
                    "stamp": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                    "headline": {
                        "gate_passed": len(passed),
                        "judged": n,
                        "judged_noncompliant": k,
                        "induced_false_pass": round(k / n, 4) if n else None,
                        "wilson95": [round(x, 3) for x in _wilson(k, n)],
                        "unjudgeable": len(unjudgeable),
                    },
                    "note": (
                        "NO threshold this month by design (ADR-0012 section 7): this number is "
                        "month 2's physical-loop calibration baseline. Unjudgeable parts are "
                        "excluded from the denominator and must be resolved, never assumed good."
                    ),
                    "records": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    for ix, r in enumerate(passed, 1):
        label, variant = str(r["label"]), int(r["variant"])
        job_dir = run_dir / f"{label}_k{variant}"
        render = Path(r["render_src"]) if r.get("render_src") else job_dir / "part.png"
        if not render.exists():
            records.append(
                {"label": label, "variant": variant, "status": "unjudgeable", "why": "no render"}
            )
            _save()
            print(f"[{ix}/{len(passed)}] {label}: UNJUDGEABLE (no render)", flush=True)
            continue
        try:
            v = _judge_one(client, judge_id, str(r["prompt"]), job_dir, render)
        except Exception as exc:  # a judge failure is recorded, never a silent pass
            records.append(
                {
                    "label": label,
                    "variant": variant,
                    "status": "unjudgeable",
                    "why": f"judge error: {type(exc).__name__}: {exc}",
                }
            )
            _save()
            print(f"[{ix}/{len(passed)}] {label}: UNJUDGEABLE (judge error)", flush=True)
            continue
        records.append(
            {
                "label": label,
                "variant": variant,
                "status": "judged",
                "complies": v.complies,
                "confidence": v.confidence,
                "violations": v.violations,
                "unverifiable": v.unverifiable,
                "reason": v.reason,
            }
        )
        _save()
        tag = "COMPLIES" if v.complies else f"VIOLATES ({'; '.join(v.violations[:2])})"
        print(f"[{ix}/{len(passed)}] {label}: {tag}", flush=True)

    h = json.loads(out_path.read_text(encoding="utf-8"))["headline"]
    print(json.dumps(h, indent=2), flush=True)
    print(f"[done] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
