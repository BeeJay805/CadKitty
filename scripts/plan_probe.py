"""Plan-stage probe: the cheap confidence instrument for expensive arms (founder rule
2026-07-23: don't spend the full-gauntlet arm until the evidence says the outcome
will be engineered).

One probe = brief -> mech architecture -> decompose (2-3 subscription calls, minutes,
no GPU, no authoring) with the SAME env flags the arm will carry, then every
deterministic plan gate the run would apply, plus probe-only observations (annex
stations, mirrored pairs, per-card form language). Scorecard printed and saved; the
arm launches only after N consecutive fully-green probes.

Usage:
  python scripts/plan_probe.py --out <dir> [--prompt-file <txt>] [--height 350]
Env: set the exact arm flag roster before calling (FORGE_* flags are read live)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_DEFAULT_PROMPT = (
    "A desktop humanoid robot, 350 mm tall, styled like a modern consumer humanoid "
    "(smooth tapered limb shells, curved chest and back plates, a rounded helmet-like "
    "head with a visor recess, cylindrical joint modules with domed caps at shoulders, "
    "elbows, hips, and knees). Human proportions. MG996R-class hobby servos (bought "
    "parts) drive the joints. Printed parts are elegant lofted shells and modules — NO "
    "rectangular boxes. PLA, printable with minimal support."
)

# The FORGE_FIDELITY_DOCTRINE annex stations for a 350 mm figure; scaled by --height.
_ANNEX_STATIONS = {"knee": 105.0, "hip": 180.0, "shoulder": 290.0}
_BAND = 0.10  # +/-10%


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--height", type=float, default=350.0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else _DEFAULT_PROMPT
    )

    from forge_mind.agentic.brief import research_brief
    from forge_mind.agentic.functional import (
        _enabled_overlays,
        _plan_dimension_issues,
        _plan_engineering_issues,
        _plan_form_issues,
        _plan_mirror_issues,
        _plan_topology_issues,
        decompose_functional,
    )
    from forge_mind.agentic.mech_arch import (
        arch_context_block,
        hardware_decls_from,
        plan_mech_architecture,
    )
    from forge_mind.llm.provider import make_client
    from forge_mind.structures.servo_bridge import placed_hardware_for

    client = make_client()
    brief_model = os.environ.get("FORGE_PROBE_BRIEF_MODEL", "fable")

    brief, _u = research_brief(prompt, client=client, model=brief_model)
    (out / "brief.json").write_text(brief.model_dump_json(indent=2), encoding="utf-8")

    arch, _u = plan_mech_architecture(
        prompt,
        client=client,
        model=brief_model,
        brief_text=brief.visual_description,
        doctrine=_enabled_overlays(),
    )
    (out / "mech_arch.json").write_text(arch.model_dump_json(indent=2), encoding="utf-8")
    hardware = hardware_decls_from(arch)

    plan, _u = decompose_functional(
        prompt,
        client=client,
        model=brief_model,
        brief=brief,
        extra_callout=arch_context_block(arch),
    )
    (out / "plan.json").write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    hw_ports: dict[str, set[str]] = {}
    for decl in hardware:
        try:
            hw_ports[decl.hw_id] = {p.name for p in placed_hardware_for(decl).ports}
        except Exception:
            hw_ports[decl.hw_id] = set()

    gates: dict[str, list[str]] = {
        "topology": _plan_topology_issues(plan, hw_ports),
        "dimensions": _plan_dimension_issues(plan),
        "engineering": _plan_engineering_issues(plan, hardware),
    }

    # Probe observations. Two flavors: (1) plan-level defect checks shared with the arm's
    # vocabulary (mirror-sibling completeness, box-as-final-form) -- pinned helpers in
    # functional.py; (2) the arch-level annex-station drift check (typed arch, probe-only).
    obs: list[str] = _plan_mirror_issues(plan) + _plan_form_issues(plan)
    scale = args.height / 350.0
    for j in arch.joints:
        for key, z350 in _ANNEX_STATIONS.items():
            if key in j.name.lower():
                want = z350 * scale
                got = float(j.station_mm[2])
                if abs(got - want) > _BAND * want:
                    obs.append(
                        f"station: {j.name} z={got:.0f} vs annex {want:.0f} "
                        f"(+/-{_BAND:.0%}) -- proportion drift"
                    )

    n_issues = sum(len(v) for v in gates.values())
    scorecard = {
        "prompt_sha_head": prompt[:80],
        "parts": len(plan.parts),
        "joints_plan": len(plan.joints),
        "joints_arch": len(arch.joints),
        "actuators": len(arch.actuators),
        "gates": gates,
        "gate_issues_total": n_issues,
        "observations": obs,
        "green": n_issues == 0 and not obs,
        "env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("FORGE_")},
    }
    (out / "scorecard.json").write_text(json.dumps(scorecard, indent=1), encoding="utf-8")

    print(f"PROBE {'GREEN' if scorecard['green'] else 'RED'}: {out}")
    print(
        f"  parts={len(plan.parts)} arch_joints={len(arch.joints)} "
        f"actuators={len(arch.actuators)} gate_issues={n_issues} obs={len(obs)}"
    )
    for family, issues in gates.items():
        for line in issues:
            print(f"  GATE[{family}]: {line}")
    for line in obs:
        print(f"  OBS: {line}")
    return 0 if scorecard["green"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
