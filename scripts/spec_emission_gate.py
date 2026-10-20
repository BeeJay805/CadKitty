"""SpecGraph slice 2 ENTRY GATE (ADR-0088): is the structured brief emission trustworthy?

The council's pre-registered gate: before any VERDICT consumes the model's structured fields,
the model must emit them FAITHFULLY -- schema-valid AND agreeing with its own prose. The skeptic's
kill line: >=95% agreement -> proceed; <90% -> the structured emission is a 'second liar', do not
wire it. This harness measures it CHEAPLY (brief calls only, no builds) across part families.

Agreement per brief = (spec_dims whose value appears in the same feature's approx_dims_mm) / (all
spec_dims), plus feature_id uniqueness. A brief with zero spec_dims is an emission MISS (the model
was asked and declined), counted against coverage separately from agreement.

Subscription ($0): set FORGE_LLM_PROVIDER=claude_code. Directional at small n; the row logs n.

Run:
  $env:FORGE_LLM_PROVIDER='claude_code'; $env:FORGE_SPEC_BRIEF='1'
  uv run --no-sync python -u scripts/spec_emission_gate.py [per_family]
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from forge_mind.agentic.brief import (  # noqa: E402
    SpecBrief,
    feature_spec_agreement,
    research_brief,
)
from forge_mind.config import BRAIN_MODEL  # noqa: E402
from forge_mind.llm.provider import make_client  # noqa: E402

# 5 families x parametric prompts (disjoint from the frozen benches; these are throwaway probes).
_FAMILIES: dict[str, list[str]] = {
    "bracket": [
        "an L-bracket 60 x 40 x 4 mm with two M4 mounting holes 30 mm apart",
        "a flat mounting plate 100 x 50 x 5 mm, four M5 holes 76 mm x 40 mm pattern",
        "a shelf bracket 80 x 80 x 6 mm with a 45-degree gusset and three 5 mm holes",
    ],
    "enclosure": [
        "a project box 90 x 60 x 40 mm, 2 mm walls, four M3 corner bosses",
        "a DIN-rail enclosure 100 x 70 x 45 mm with a vented lid",
    ],
    "pulley": [
        "an idler pulley 40 mm outer dia, 8 mm bore, 30 mm wide with two clip grooves",
        "a timing pulley 30 mm dia, 5 mm bore, 20 teeth, 12 mm wide",
    ],
    "duct": [
        "a round-to-round reducer 60 mm to 30 mm, 50 mm long, 2 mm wall",
        "a rectangular fan duct 80 x 80 to a 60 mm round outlet, 70 mm tall",
    ],
    "standoff": [
        "a hex standoff M4, 20 mm long, 7 mm across flats",
        "a flanged bushing 8 mm bore, 12 mm flange, 20 mm long",
    ],
}


def main() -> int:
    per_family = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = all listed
    if not os.environ.get("FORGE_SPEC_BRIEF"):
        os.environ["FORGE_SPEC_BRIEF"] = "1"  # the gate measures the ON path by definition
    client = make_client()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = REPO / "benchmarks" / "results" / f"spec_gate_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for family, prompts in _FAMILIES.items():
        for prompt in prompts[: per_family or len(prompts)]:
            row: dict = {"family": family, "prompt": prompt[:70]}
            try:
                brief, _ = research_brief(prompt, client=client, model=BRAIN_MODEL)
                is_spec = isinstance(brief, SpecBrief)
                agreed = total = 0
                for f in brief.features:
                    a, t = feature_spec_agreement(f)
                    agreed += a
                    total += t
                ids = [f.feature_id for f in brief.features if getattr(f, "feature_id", None)]
                row.update(
                    schema_ok=is_spec,
                    n_features=len(brief.features),
                    n_spec_dims=total,
                    agreed=agreed,
                    agreement=(agreed / total) if total else None,
                    ids_present=len(ids),
                    ids_unique=len(set(ids)) == len(ids),
                    covered=total > 0,
                )
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            (out / "gate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            ag = row.get("agreement")
            print(
                f"[{family}] dims={row.get('n_spec_dims')} "
                f"agree={f'{ag:.0%}' if ag is not None else 'n/a'} "
                f"ids_uniq={row.get('ids_unique')} {row.get('error', '')}",
                flush=True,
            )

    scored = [r for r in rows if r.get("agreement") is not None]
    micro_a = sum(r["agreed"] for r in scored)
    micro_t = sum(r["n_spec_dims"] for r in scored)
    covered = sum(1 for r in rows if r.get("covered"))
    print("\n=== EMISSION GATE ===", flush=True)
    print(f"briefs: {len(rows)} | covered (emitted spec_dims): {covered}/{len(rows)}", flush=True)
    print(
        f"micro agreement: {micro_a}/{micro_t} = {(micro_a / micro_t) if micro_t else 0:.1%} "
        f"(gate >=95% proceed, <90% kill)",
        flush=True,
    )
    print(
        f"feature_id unique on all covered briefs: "
        f"{all(r.get('ids_unique') for r in rows if r.get('covered'))}",
        flush=True,
    )
    print(f"results: {out / 'gate.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
