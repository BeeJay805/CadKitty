"""Live agentic-path accuracy benchmark v2 (PAID -- runs the real author loop).

v2 fixes the v1 disaster (an 86-minute run that lost ALL data when killed):
  * flush-per-part progress (also run with `python -u`) -- live, not buffered;
  * the results JSON is REWRITTEN after EVERY part, so a kill never loses what already ran;
  * records disposition + failed_checks + stopped_reason + iteration count per part (the "why");
  * a model knob (sonnet|opus) to A/B the author, plus a hard TOTAL-spend cap.

Calls ``agentic.result.produce_part`` directly (the engine the deployed /generate runs) with the
chosen author model and the loop's new dynamic early-stop. PASS = the honest gate (validation.passed,
or geometry_passed when there is no per-part report). The benchmark config is the CANDIDATE live setup
(1 draft, early-stop, a low safety iteration cap) -- NOT the old 20-iteration grind we are leaving.

Args: N [model] [cap_usd] [trials].  `trials` repeats EACH prompt that many times -- a stochastic
loop needs repeats, so the per-prompt pass-rate (in the JSON's ``per_prompt``) is the honest
reliability number; one shot is not a claim. Total runs = N * trials, bounded by the $ cap.

Run (unbuffered):
    uv run --no-sync python -u scripts/bench_agentic.py 12 sonnet 4      # 12 prompts x1, Sonnet, $4 cap
    uv run --no-sync python -u scripts/bench_agentic.py 7 sonnet 12 5    # 7 prompts x5 trials, $12 cap
    uv run --no-sync python -u scripts/bench_agentic.py 12 opus 8        # Opus author, $8 cap

Needs ANTHROPIC_API_KEY. Writes/updates benchmarks/results/agentic_<model>_<stamp>.json live.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from forge_mind.agentic.result import produce_part
from forge_mind.assembly.bench import wilson_lower_bound
from forge_mind.config import FABLE, OPUS, SONNET
from forge_mind.fabrication import FabricationMode
from forge_mind.llm.claude_code_client import subscription_client_if_selected

# FORGE_LLM_PROVIDER=claude_code routes EVERY engine LLM call (brief/author/critic/
# judge) through headless Claude Code -- subscription-billed, $0 API spend. Results
# are tagged with the provider and are for iteration, NOT production swap decisions
# (see claude_code_client docstring for the fidelity boundaries).
_PROVIDER_CLIENT = subscription_client_if_selected()

_MAX_ITERATIONS = 8  # safety cap; the dynamic early-stop usually halts well before this
_N_DRAFTS = 1
_PART_BUDGET_USD = 1.0  # per-part hard ceiling (the total cap is the argv arg)
_MODELS = {"sonnet": SONNET, "opus": OPUS, "fable": FABLE}

PROMPTS: list[tuple[str, str]] = [
    (
        "right_angle_bracket",
        "a right-angle bracket 40 mm x 40 mm with two M4 clearance holes per leg",
    ),
    (
        "nema17_plate",
        "a NEMA-17 motor mount plate 5 mm thick with the 31 mm bolt circle and a bore",
    ),
    ("fan_shroud", "a 60 mm fan shroud that bolts to a 2020 aluminum extrusion with M5 hardware"),
    ("d_shaft_knob", "a 25 mm knob with a 6 mm D-shaft bore and a knurled rim"),
    (
        "wall_hook",
        "a wall hook that clips over a 12 mm rail with two M3 screw holes in the back plate",
    ),
    ("bearing_block", "a pillow block that seats a 608 bearing, bolted down with two M5 screws"),
    ("spur_gear", "a module-2 spur gear with 24 teeth, 6 mm thick, with a 5 mm bore"),
    ("enclosure_lid", "a 70 mm x 50 mm enclosure lid 3 mm wall with four M3 corner standoffs"),
    ("cable_clip", "a snap cable clip for an 8 mm cable, screwed down with one M3 screw"),
    ("standoff", "an M3 standoff 20 mm tall, threaded both ends"),
    ("hinge_leaf", "a hinge leaf with three knuckles for a 4 mm pin, 30 mm wide"),
    ("adapter_plate", "an adapter plate from a NEMA-17 face to four M5 holes on a 50 mm square"),
]


def _part_record(label: str, prompt: str, model_id: str, out_dir: Path) -> dict[str, object]:
    result = produce_part(
        prompt,
        out_dir,
        n_drafts=_N_DRAFTS,
        max_iterations=_MAX_ITERATIONS,
        budget_usd=_PART_BUDGET_USD,
        # GENERAL is the deployed default (ADR-0031): measure the real product. Overhang is not
        # checked and min-wall is relaxed, so the bench no longer reports FDM-printability noise.
        fabrication_mode=FabricationMode.GENERAL,
        author_model=model_id,
        client=_PROVIDER_CLIENT,
    )
    loop = result.loop
    val = loop.validation
    passed = bool(val.passed) if val is not None else bool(loop.geometry_passed)
    failed = val.report.failed_check_ids() if (val is not None and val.report is not None) else []
    # The dropped-holes gate (`features_missing`) is NOT a report check, so `failed_checks` never
    # shows it -- a part can FAIL while listing only advisory checks. Surface it directly (+ the
    # "MISSING FEATURES" detail and body_count) so a holes failure is visible, not inferred.
    features_missing = bool(val.features_missing) if val is not None else None
    missing_detail = next(
        (i for i in (val.issues if val is not None else []) if i.startswith("MISSING FEATURES")),
        "",
    )
    return {
        "label": label,
        "prompt": prompt,
        "passed": passed,
        "disposition": result.disposition,
        "stopped_reason": loop.stopped_reason,
        "iterations": len(loop.iterations),
        "failed_checks": failed,
        "features_missing": features_missing,
        "missing_detail": missing_detail,
        "body_count": (val.body_count if val is not None else None),
        "hard_ok": (bool(val.hard_ok) if val is not None else None),
        # The backstop's transparency flag: "the author built it" vs "the repair saved it".
        "repaired": bool(getattr(loop, "repaired_features", False)),
        "cost_usd": round(result.total_cost_usd, 4),
        # The best candidate's render lives in the temp dir (deleted at run end), so the caller copies
        # it to a persistent gallery while it still exists. None on a no-build / render failure.
        "render_src": str(loop.final_render) if loop.final_render else None,
    }


def _dump(
    out_path: Path,
    model_key: str,
    records: list[dict[str, object]],
    total_cost: float,
    trials: int,
) -> None:
    passed = sum(1 for r in records if r["passed"])
    n_run = len(records)
    # Per-prompt reliability view: each prompt is run `trials` times, so its pass-rate across trials is
    # the honest reliability number (a stochastic system needs repeats -- one shot is not a claim).
    by_label: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_label.setdefault(str(r["label"]), []).append(r)
    per_prompt: dict[str, dict[str, object]] = {}
    for label, rs in by_label.items():
        k = len(rs)
        p = sum(1 for r in rs if r["passed"])
        per_prompt[label] = {
            "trials": k,
            "passed": p,
            "pass_rate": round(p / k, 3) if k else 0.0,
            "holes_ok": sum(1 for r in rs if r.get("features_missing") is False),
        }
    out_path.write_text(
        json.dumps(
            {
                "model": model_key,
                "provider": ("claude_code" if _PROVIDER_CLIENT is not None else "api"),
                "trials_per_prompt": trials,
                "n": n_run,
                "passed": passed,
                "pass_rate": (passed / n_run if n_run else 0.0),
                "wilson95_lb": (wilson_lower_bound(passed, n_run) if n_run else 0.0),
                "total_cost_usd": round(total_cost, 2),
                "max_iterations": _MAX_ITERATIONS,
                "per_prompt": per_prompt,
                "results": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _montage(records: list[dict[str, object]], out_png: Path, trials: int) -> Path | None:
    """Tile the persisted per-run renders into ONE contact sheet (rows = prompt, cols = trial), each
    cell titled with its verdict, so the whole reliability run reads at a glance. Best-effort: returns
    None if matplotlib is unavailable or no renders were saved (the individual PNGs still exist)."""
    if not any(isinstance(r.get("render"), str) for r in records):
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels: list[str] = []
    by_label: dict[str, list[dict[str, object]]] = {}
    for r in records:
        lbl = str(r["label"])
        if lbl not in by_label:
            by_label[lbl] = []
            labels.append(lbl)
        by_label[lbl].append(r)
    ncols = max(trials, max((len(v) for v in by_label.values()), default=1))
    nrows = max(len(labels), 1)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.6, nrows * 2.5), squeeze=False)
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")
    for ri, lbl in enumerate(labels):
        for ci, r in enumerate(by_label[lbl][:ncols]):
            ax = axes[ri][ci]
            fm = r.get("features_missing")
            holes = "holes MISSING" if fm else ("holes ok" if fm is False else "")
            verdict = "PASS" if r["passed"] else "FAIL"
            color = "#1a7f37" if r["passed"] else "#cf222e"
            src = r.get("render")
            if isinstance(src, str) and Path(src).exists():
                with contextlib.suppress(Exception):
                    ax.imshow(mpimg.imread(src))
            trial_no = int(r.get("trial", 0)) + 1
            ax.set_title(f"{lbl} t{trial_no}: {verdict}  {holes}", fontsize=8, color=color)
    fig.suptitle(
        f"agentic bench -- {len(labels)} prompts x {trials} trials (rows=prompt, cols=trial)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    try:
        fig.savefig(out_png, dpi=90)
    except Exception:
        plt.close(fig)
        return None
    plt.close(fig)
    return out_png


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(PROMPTS)
    model_key = sys.argv[2] if len(sys.argv) > 2 else "sonnet"
    cap_usd = float(sys.argv[3]) if len(sys.argv) > 3 else 1e9
    trials = (
        int(sys.argv[4]) if len(sys.argv) > 4 else 1
    )  # repeats PER PROMPT (stochastic -> repeat)
    model_id = _MODELS.get(model_key, SONNET)
    prompts = PROMPTS[:n]

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path("benchmarks/results") / f"agentic_{model_key}_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # PERSIST renders: each part's render lives in the temp dir (deleted at run end), so copy the best
    # candidate's render here while it exists -- a stats run with no visuals is a gap. Named with the
    # outcome so the gallery + contact sheet read as the reliability story (PASS/FAIL, holes ok/MISSING).
    render_dir = out_path.parent / f"renders_{model_key}_{stamp}"
    render_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    total_cost = 0.0
    stopped = False
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, (label, prompt) in enumerate(prompts):
            if stopped:
                break
            for t in range(trials):
                if total_cost >= cap_usd:
                    print(
                        f"-- spend cap ${cap_usd:.2f} reached; stop at prompt {i}/{len(prompts)} "
                        f"trial {t}/{trials} --",
                        flush=True,
                    )
                    stopped = True
                    break
                rec = _part_record(label, prompt, model_id, root / f"p{i}_t{t}")
                rec["trial"] = t
                total_cost += float(rec["cost_usd"])
                records.append(rec)
                why = "" if rec["passed"] else (rec["failed_checks"] or rec["disposition"])
                fm = rec["features_missing"]
                holes = "holes:MISSING" if fm else ("holes:ok" if fm is False else "holes:?")
                # Copy the render out of the temp dir into the persistent gallery (best-effort).
                src = rec.get("render_src")
                if isinstance(src, str) and Path(src).exists():
                    hole_tag = "holesMISSING" if fm else ("holesok" if fm is False else "holesNA")
                    verdict = "PASS" if rec["passed"] else "FAIL"
                    dst = render_dir / f"{i + 1:02d}_{label}_t{t + 1}_{verdict}_{hole_tag}.png"
                    with contextlib.suppress(OSError):
                        shutil.copy2(src, dst)
                        rec["render"] = str(dst)
                tag = f"{i + 1:2}/{len(prompts)}" + (f" t{t + 1}/{trials}" if trials > 1 else "")
                print(
                    f"[{tag}] {label:18} {'PASS' if rec['passed'] else 'FAIL':4} "
                    f"${rec['cost_usd']:5.2f} it={rec['iterations']} {rec['stopped_reason']:11} "
                    f"{holes} bodies={rec['body_count']} {why}",
                    flush=True,
                )
                _dump(
                    out_path, model_key, records, total_cost, trials
                )  # incremental: survives a kill

    passed = sum(1 for r in records if r["passed"])
    n_run = len(records)
    rate = passed / n_run if n_run else 0.0
    lb = wilson_lower_bound(passed, n_run) if n_run else 0.0
    print(
        f"\n=== AGENTIC accuracy ({model_key}): {passed}/{n_run} = {rate * 100:.0f}%  "
        f"Wilson95 LB {lb * 100:.0f}%   total ${total_cost:.2f} ===",
        flush=True,
    )
    print(f"wrote {out_path}", flush=True)
    sheet = _montage(records, render_dir / "_contact_sheet.png", trials)
    if sheet is not None:
        print(f"contact sheet: {sheet}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
