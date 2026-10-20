"""Local generation MONITOR (ADR-0112 companion): type a prompt, watch each iteration build in real
time, see per-iteration speed, then the final part image + validator verdict.

Runs the REAL agentic engine (`produce_part` -- the same path `/generate` uses), so this is where the
CAD-Coder first-pass shows up: pick the engine in the sidebar. The engine runs in a background thread
writing per-iteration renders to a work dir; the page polls that dir and streams the renders + a live
clock as they appear, then shows the final render, the PASS/FAIL per tier, and a per-iteration timing
table when the run finishes.

Additive + optional (the `ui` extra). Launch:
    # local model first-pass (needs `ollama serve` + the model):
    FORGE_GENERATOR=cadcoder uv run --extra ui streamlit run src/forge_mind/ui/monitor_app.py
    # default engine (subscription/API, no local model needed):
    uv run --extra ui streamlit run src/forge_mind/ui/monitor_app.py
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from forge_mind.agentic.result import PartResult, produce_part

# Every run's metrics are appended here, in the repo, so nothing is lost between sessions.
_METRICS_FILE = Path(__file__).resolve().parents[3] / "benchmarks" / "runs" / "monitor_runs.jsonl"


def _save_run_metrics(result: PartResult, prompt: str, engine: str) -> Path:
    """Append one JSON record (timing, per-iteration speed, validator verdict, cost) to the repo's
    metrics file. Best-effort: a logging failure must never break the UI."""
    loop = result.loop
    val = loop.validation
    report = getattr(val, "report", None) if val is not None else None
    record: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "prompt": prompt,
        "engine": engine,
        "disposition": result.disposition,
        "validator_passed": bool(report.passed) if report is not None else None,
        "failed_checks": report.failed_check_ids() if report is not None else [],
        "wall_s": round(loop.total_duration_s, 2),
        "cost_usd": round(result.total_cost_usd, 4),
        "iterations": [
            {
                "iteration": it.iteration,
                "phase": it.phase,
                "built": it.built,
                "seconds": round(it.duration_s, 2),
                "score": it.score,
                "validation_passed": it.validation_passed,
            }
            for it in loop.iterations
        ],
        "stopped_reason": loop.stopped_reason,
    }
    _METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _METRICS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    return _METRICS_FILE


_PLACEHOLDER = "a 40x20x10 mm mounting bracket with two 5 mm holes"


def _run(
    prompt: str,
    out_dir: Path,
    material: str,
    max_iters: int,
    events: queue.Queue[Any],
    holder: dict[str, Any],
) -> None:
    """Worker thread: run the engine, push phase events, stash the result. Never touches `st.*`."""
    try:
        holder["result"] = produce_part(
            prompt,
            out_dir,
            material=material,
            max_iterations=max_iters,
            on_phase=lambda key, status: events.put(("phase", key, status)),
        )
    except Exception as exc:  # a crash must surface in the UI, not vanish in the thread
        holder["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        events.put(("done", None, None))


def _iter_renders(out_dir: Path) -> list[Path]:
    """Per-iteration renders in build order (iter_01, iter_02, ...), as they land on disk."""
    return sorted(out_dir.glob("**/iter_*/render.png"), key=lambda p: str(p.parent))


def _validation_block(result: PartResult) -> None:
    val = result.loop.validation
    report = getattr(val, "report", None) if val is not None else None
    if report is None:
        st.info("No validator report (the part did not reach validation).")
        return
    if report.passed:
        st.success("PASS — every validator tier is green.")
    else:
        st.error(f"FAIL — failing checks: {', '.join(report.failed_check_ids()) or 'see tiers'}")
    tiers = list(report.tiers.items())
    if tiers:
        cols = st.columns(len(tiers))
        for col, (tier, tr) in zip(cols, tiers, strict=False):
            icon = "✅" if tr.n_failed == 0 else "❌"
            col.metric(str(tier), icon, f"{tr.n_failed} failed")


def _final_block(result: PartResult, out_dir: Path, prompt: str) -> None:
    loop = result.loop
    st.subheader("Result")
    badge = {
        "shipped": "✅ shipped",
        "shipped_with_caveats": "🟡 shipped with caveats",
        "rejected": "❌ rejected",
        "no_build": "⛔ no build",
    }.get(result.disposition, result.disposition)
    st.markdown(f"**Disposition:** {badge}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wall time", f"{loop.total_duration_s:.1f} s")
    c2.metric("Cost", f"${result.total_cost_usd:.4f}")
    c3.metric("Iterations", str(len(loop.iterations)))
    avg = sum(it.duration_s for it in loop.iterations) / max(1, len(loop.iterations))
    c4.metric("Avg / iter", f"{avg:.1f} s")

    final = loop.final_render
    if final is not None and Path(final).exists():
        st.image(str(final), caption=f"final — {prompt[:70]}", use_container_width=True)
    else:
        renders = _iter_renders(out_dir)
        if renders:
            st.image(str(renders[-1]), caption="last built iteration", use_container_width=True)
        else:
            st.write("(no render produced)")

    if loop.final_stl is not None and Path(loop.final_stl).exists():
        st.download_button(
            "Download STL",
            Path(loop.final_stl).read_bytes(),
            file_name=Path(loop.final_stl).name,
            mime="model/stl",
        )

    _validation_block(result)

    st.subheader("Per-iteration speed")
    st.table(
        [
            {
                "iter": it.iteration,
                "phase": it.phase,
                "built": "yes" if it.built else "no",
                "seconds": round(it.duration_s, 1),
                "score": it.score if it.score is not None else "—",
            }
            for it in loop.iterations
        ]
        or [{"iter": "—", "phase": "no iterations", "built": "—", "seconds": "—", "score": "—"}]
    )


def main() -> None:
    st.set_page_config(page_title="Forge-Mind Monitor", page_icon="🛠️", layout="wide")
    st.title("🛠️ Forge-Mind — live part monitor")

    with st.sidebar:
        st.header("Engine")
        engine = st.radio(
            "Generator",
            ["Local-first (CAD-Coder + local roster)", "Cloud author"],
            help="Local-first (DEFAULT): CAD-Coder codes, qwen3-coder repairs, Qwen2.5-VL sees, Fable "
            "plans (API). Cloud author: the build123d author on the subscription/API.",
        )
        material = st.selectbox("Material", ["PLA", "PETG", "ABS", "TPU"], index=0)
        max_iters = st.slider("Max iterations", 1, 8, 4)

    prompt = st.text_input("Describe the part", placeholder=_PLACEHOLDER)
    if not st.button("Generate", type="primary", disabled=not prompt.strip()):
        st.caption("Enter a prompt and press Generate. Each run is a real engine call.")
        return

    # Select the coder for THIS process. Local-first is the default (the machine roster); "Cloud
    # author" opts OUT by forcing the build123d author path.
    os.environ["FORGE_GENERATOR"] = "author" if engine.startswith("Cloud") else "cadcoder"

    out_dir = Path(tempfile.mkdtemp(prefix="fm_monitor_"))
    events: queue.Queue[Any] = queue.Queue()
    holder: dict[str, Any] = {}
    worker = threading.Thread(
        target=_run,
        args=(prompt.strip(), out_dir, material, max_iters, events, holder),
        daemon=True,
    )

    phase_box = st.empty()
    clock_box = st.empty()
    gallery = st.empty()
    start = time.monotonic()
    phase = "starting"
    seen = 0

    worker.start()
    done = False
    while not done:
        # Drain phase events without blocking.
        try:
            while True:
                kind, key, status = events.get_nowait()
                if kind == "done":
                    done = True
                elif kind == "phase" and status == "active":
                    phase = key
        except queue.Empty:
            pass

        elapsed = time.monotonic() - start
        phase_box.markdown(f"**Phase:** `{phase}`  ·  engine: `{engine}`")
        clock_box.markdown(f"⏱️ {elapsed:.1f} s elapsed")
        renders = _iter_renders(out_dir)
        if len(renders) != seen:
            seen = len(renders)
        if renders:
            with gallery.container():
                st.caption(f"iterations built so far: {len(renders)}")
                cols = st.columns(min(4, len(renders)))
                for i, r in enumerate(renders[-4:]):
                    cols[i].image(
                        str(r), caption=f"iter {len(renders) - len(renders[-4:]) + i + 1}"
                    )
        if not done:
            time.sleep(1.0)

    worker.join(timeout=5.0)
    phase_box.empty()
    clock_box.empty()
    gallery.empty()

    if "error" in holder:
        st.error(f"Engine crashed: {holder['error']}")
        return
    result = holder.get("result")
    if result is None:
        st.error("No result produced.")
        return
    _final_block(result, out_dir, prompt.strip())

    # Persist every run's metrics to a repo file (best-effort -- never break the UI over logging).
    try:
        path = _save_run_metrics(result, prompt.strip(), engine)
        st.caption(f"metrics appended to `{path.relative_to(Path(__file__).resolve().parents[3])}`")
    except Exception as exc:
        st.caption(f"(metrics not saved: {type(exc).__name__})")


if __name__ == "__main__":
    main()
