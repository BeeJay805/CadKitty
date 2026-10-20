"""Run ONE prompt end-to-end through the GENERAL agentic engine and print the receipts.

The build-session proof runner: no deterministic lane (FORGE_ASSEMBLY_LANE stays unset),
no cache, the lever flags come from the environment. Usage:

    uv run python scripts/run_one_prompt.py "<prompt>" <out_dir> [author_arm] [orchestrator_arm]

author_arm: sonnet|opus|fable (default opus = the production default).
orchestrator_arm: sonnet|opus|fable (default sonnet = the production default) -- the
  BRIEF + DECOMPOSE model, i.e. the orchestration role that researches the object and
  writes the per-part prompts/joints. The fable-as-orchestrator A/B varies THIS arm
  while holding the author constant.
$0 subscription: set FORGE_LLM_PROVIDER=claude_code first.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from forge_mind.agentic.result import produce_part
from forge_mind.config import FABLE, OPUS, SONNET
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.claude_code_client import subscription_client_if_selected

_MODELS = {"sonnet": SONNET, "opus": OPUS, "fable": FABLE}


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    prompt = sys.argv[1]
    out_dir = Path(sys.argv[2])
    arm = sys.argv[3] if len(sys.argv) > 3 else "opus"
    orch = sys.argv[4] if len(sys.argv) > 4 else "sonnet"
    author_model = _MODELS[arm]
    brief_model = _MODELS[orch]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    result = produce_part(
        prompt,
        out_dir,
        n_drafts=1,
        max_iterations=8,
        budget_usd=1.0,  # inert on subscription (cost records ~0); bounds API runs
        fabrication_mode=FabricationMode.GENERAL,
        author_model=author_model,
        brief_model=brief_model,
        client=subscription_client_if_selected(),
    )
    wall_s = time.monotonic() - t0

    loop = result.loop
    summary: dict[str, Any] = {
        "prompt": prompt,
        "author_arm": arm,
        "orchestrator_arm": orch,
        "disposition": result.disposition,
        "geometry_passed": loop.geometry_passed,
        "stopped_reason": loop.stopped_reason,
        "iterations": len(loop.iterations),
        "wall_s": round(wall_s, 1),
        "cost_usd": round(result.total_cost_usd, 4),
        "artifacts": result.artifacts,
        "iteration_log": [
            {
                "i": it.iteration,
                "phase": it.phase,
                "built": it.built,
                "score": it.score,
                "passed": it.validation_passed,
                "err": (it.sandbox_error or "")[:160],
                "issues": [s[:160] for s in it.issues[:4]],
            }
            for it in loop.iterations
        ],
    }
    (out_dir / "oneshot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "iteration_log"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
