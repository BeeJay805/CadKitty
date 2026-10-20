"""Parallel-run MATRIX: agentic breadth x param-fit ARMS, resumable + fail-open, with a failure
taxonomy -- the substrate for "what breaks / what excels" + the council's unmeasured question
(param-fit's breadth effect).

Reuses bench_agentic v2's ``_part_record`` (the deployed ``produce_part`` + the HONEST gate) across
the frozen 12-part diverse set, run under each param-fit arm:

  blanket      FORGE_PARAM_OPT=1                             (the LIVE default -- note + polish always)
  conditional  FORGE_PARAM_OPT=1 FORGE_PARAM_OPT_MODE=cond.  (ADR-0108 -- only reference-rich parts)
  off          FORGE_PARAM_OPT=0                             (no note, no polish)

Robustness (the recurring `claude -p` 401 WILL hit a multi-hour run):
  * APPEND-ONLY JSONL -- a kill never loses a completed cell.
  * RESUMABLE -- a re-run skips cells that already have a REAL record; a 401/infra cell is retried.
  * FAIL-OPEN -- a 401/exception is recorded as ``infra_fail`` and EXCLUDED from the pass-rate (it is
    not a part failure), so one auth blip never poisons the arm's number.

WINDOWS SPAWN GUARD (row-83): FEA runs in spawned subprocesses; ALL execution stays under
``if __name__ == "__main__":`` or every child re-runs the matrix.

Run (unbuffered, subscription):
    FORGE_LLM_PROVIDER=claude_code python -u scripts/bench_matrix.py [trials] [cap_usd] [arms]
    arms default = blanket,conditional,off   (comma-list to subset; e.g. "blanket,off")
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

# Part CLASS for the "what breaks by class" rollup (the frozen labels ARE the class).
_CLASS = {
    "right_angle_bracket": "bracket",
    "nema17_plate": "plate",
    "fan_shroud": "shroud",
    "d_shaft_knob": "knob",
    "wall_hook": "hook",
    "bearing_block": "housing",
    "spur_gear": "gear",
    "enclosure_lid": "enclosure",
    "cable_clip": "clip",
    "standoff": "threaded",
    "hinge_leaf": "mechanism",
    "adapter_plate": "plate",
}

ARMS: dict[str, dict[str, str]] = {
    "blanket": {"FORGE_PARAM_OPT": "1", "FORGE_PARAM_OPT_MODE": ""},
    "conditional": {"FORGE_PARAM_OPT": "1", "FORGE_PARAM_OPT_MODE": "conditional"},
    "off": {"FORGE_PARAM_OPT": "0", "FORGE_PARAM_OPT_MODE": ""},
}

# FORGE_MATRIX_PROMPTS=extended adds 12 harder/diverse parts to the frozen 12 (local is $0, so run
# tons). Chosen to stress PLANNING (mechanisms/assemblies) AND BUILDING (threads/curves) separately.
_EXTENDED: list[tuple[str, str, str]] = [
    ("bottle_cap", "a threaded bottle cap, 30 mm diameter, M28x2 thread, knurled grip", "threaded"),
    (
        "herringbone_gear",
        "a herringbone gear, module 1.5, 30 teeth, 8 mm bore with a keyway",
        "gear",
    ),
    (
        "snapfit_base",
        "a snap-fit enclosure base, 60 x 40 x 25 mm, four internal M3 screw bosses",
        "enclosure",
    ),
    ("vbelt_pulley", "a V-belt pulley, 50 mm OD, 10 mm bore, one M4 set screw", "mechanism"),
    (
        "pipe_flange",
        "a pipe flange, 60 mm OD, four M6 bolt holes on a 45 mm circle, 25 mm bore",
        "flange",
    ),
    (
        "strain_relief",
        "a cable strain relief for an 8 mm cable, screw-mounted with two M3 holes",
        "clip",
    ),
    ("spacer_stack", "a spacer, 10 mm ID, 20 mm OD, 15 mm tall, chamfered ends", "spacer"),
    (
        "servo_horn",
        "a servo horn adapter from a 25-tooth spline to a 4 mm D-shaft, two arms",
        "mechanism",
    ),
    (
        "fan_duct",
        "a 40 mm fan cooling duct that necks down to a 20 mm round outlet with two tabs",
        "shroud",
    ),
    ("tslot_nut", "a drop-in T-slot nut for 2020 aluminium extrusion, M5 thread", "threaded"),
    (
        "vented_panel",
        "a 80 x 60 mm vented panel, 3 mm thick, hex ventilation pattern, four corner holes",
        "plate",
    ),
    (
        "desk_grommet",
        "a two-part desk grommet for a 60 mm hole with a removable cover flap",
        "enclosure",
    ),
]
_CLASS.update({label: klass for label, _prompt, klass in _EXTENDED})


def _apply_arm(env: dict[str, str]) -> None:
    for k, v in env.items():
        if v:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


def _is_auth_fail(msg: str) -> bool:
    m = msg.lower()
    return "401" in m or "authenticate" in m or "invalid authentication" in m


def _done_keys(out: Path) -> set[str]:
    """Keys with a REAL (non-infra) record -> skip on resume; infra_fail cells are retried."""
    done: set[str] = set()
    if not out.exists():
        return done
    for line in out.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("key") and not r.get("infra_fail"):
            done.add(str(r["key"]))
    return done


def _rollup(rows: list[dict[str, Any]]) -> None:
    real = [r for r in rows if not r.get("infra_fail")]
    infra = [r for r in rows if r.get("infra_fail")]
    print(f"\n==== MATRIX ROLLUP: {len(real)} real cells, {len(infra)} infra/401 (excluded) ====")
    print("  BUILD = wrote working code (hard_ok) | FEATURES = all declared features present |")
    print("  ACCEPTED = judge intent-match | PASS = full deterministic gate")
    for arm in ARMS:
        a = [r for r in real if r["arm"] == arm]
        if not a:
            continue
        n = len(a)
        built = sum(1 for r in a if r.get("hard_ok"))
        feat = sum(1 for r in a if r.get("features_missing") is False)
        acc = sum(1 for r in a if r.get("disposition") == "accepted")
        p = sum(1 for r in a if r.get("passed"))
        lat = [float(r["seconds"]) for r in a if r.get("seconds")]
        mean_lat = f"{sum(lat) / len(lat):.0f}s" if lat else "-"
        print(
            f"  {arm:11} BUILD {built}/{n} ({100 * built // n}%) | FEATURES {feat}/{n} "
            f"({100 * feat // n}%) | ACCEPTED {acc}/{n} ({100 * acc // n}%) | PASS {p}/{n} "
            f"({100 * p // n}%)  mean {mean_lat}"
        )
        checks: Counter[str] = Counter()
        for r in a:
            for c in r.get("failed_checks") or []:
                checks[c] += 1
        print(
            f"       top-fails: {', '.join(f'{c} x{v}' for c, v in checks.most_common(5)) or 'none'}"
        )
    # per-class BUILD vs PASS (does it break at planning/build or at the gate, and on which classes)
    print("  -- by class (BUILD / PASS, all arms) --")
    for k in sorted({r["klass"] for r in real}):
        a = [r for r in real if r["klass"] == k]
        b = sum(1 for r in a if r.get("hard_ok"))
        p = sum(1 for r in a if r.get("passed"))
        print(f"     {k:11} build {b}/{len(a)}  pass {p}/{len(a)}")


def main() -> int:
    from bench_agentic import _MODELS, PROMPTS, _part_record  # reuse the v2 record

    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cap_usd = float(sys.argv[2]) if len(sys.argv) > 2 else 1e9
    arms = sys.argv[3].split(",") if len(sys.argv) > 3 else list(ARMS)
    arms = [a for a in arms if a in ARMS]
    # Author model: Opus by default (Fable is out; the founder asked for Opus as the brain). Under
    # FORGE_LLM_PROVIDER=ollama the OllamaClient ignores this and always uses the local model.
    model_id = _MODELS.get(os.environ.get("FORGE_BENCH_MODEL", "opus"), _MODELS["opus"])
    prompts = list(PROMPTS)
    if os.environ.get("FORGE_MATRIX_PROMPTS") == "extended":
        prompts += [(lbl, txt) for lbl, txt, _k in _EXTENDED]
    provider = os.environ.get("FORGE_LLM_PROVIDER", "api")
    print(
        f"matrix: provider={provider} model={model_id} arms={arms} trials={trials} "
        f"parts={len(prompts)}",
        flush=True,
    )

    out = Path("benchmarks/results/matrix") / os.environ.get("FORGE_MATRIX_OUT", "matrix.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_keys(out)
    rows: list[dict[str, Any]] = []
    if out.exists():
        rows = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.strip()]
    total = 0.0
    workdir = out.parent / "work"

    for arm in arms:
        _apply_arm(ARMS[arm])
        for label, prompt in prompts:
            for t in range(trials):
                key = f"{arm}|{label}|{t}"
                if key in done:
                    continue
                if total >= cap_usd:
                    print(f"-- spend cap ${cap_usd:.2f} reached --", flush=True)
                    _rollup(rows)
                    return 0
                t0 = time.perf_counter()
                try:
                    rec = _part_record(label, prompt, model_id, workdir / key.replace("|", "_"))
                    row: dict[str, Any] = {
                        "key": key,
                        "arm": arm,
                        "label": label,
                        "klass": _CLASS.get(label, "other"),
                        "trial": t,
                        "seconds": round(time.perf_counter() - t0, 1),
                        "passed": bool(rec["passed"]),
                        "hard_ok": rec.get("hard_ok"),  # BUILD: wrote working build123d code
                        "disposition": rec.get("disposition"),  # PLAN: judge intent-match
                        "iterations": rec.get("iterations"),
                        "failed_checks": rec.get("failed_checks") or [],
                        "features_missing": rec.get("features_missing"),
                        "stopped_reason": rec.get("stopped_reason"),
                        "cost_usd": float(rec.get("cost_usd") or 0.0),
                        "infra_fail": False,
                    }
                    # A brief_failed / author_failed is an intermittent LLM/CLI blip (auth 401,
                    # rate-limit) that produce_part SWALLOWS -- not a real part-quality failure. Tag
                    # it infra so it is EXCLUDED from the pass-rate AND retried on the next resume
                    # (else a transient blip is frozen as a permanent "fail" the resume skips).
                    if rec.get("stopped_reason") in ("author_failed", "brief_failed"):
                        row["infra_fail"] = True
                        row["auth"] = True
                        row["error"] = f"swallowed blip: {rec.get('stopped_reason')}"
                    total += row["cost_usd"]
                except Exception as e:  # fail-open: 401/infra never poisons the arm's number
                    row = {
                        "key": key,
                        "arm": arm,
                        "label": label,
                        "klass": _CLASS.get(label, "other"),
                        "trial": t,
                        "seconds": round(time.perf_counter() - t0, 1),
                        "passed": False,
                        "infra_fail": True,
                        "auth": _is_auth_fail(str(e)),
                        "error": str(e)[:160],
                    }
                rows.append(row)
                with out.open("a", encoding="utf-8") as fh:  # append-only: kill-safe
                    fh.write(json.dumps(row) + "\n")
                tag = "INFRA" if row["infra_fail"] else ("PASS" if row["passed"] else "FAIL")
                print(
                    f"[{arm:11} {label:18} t{t}] {tag:5} {row['seconds']:5.0f}s "
                    f"{'auth401' if row.get('auth') else ''} "
                    f"{'' if row['infra_fail'] else (row.get('failed_checks') or '')}",
                    flush=True,
                )

    _rollup(rows)
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
