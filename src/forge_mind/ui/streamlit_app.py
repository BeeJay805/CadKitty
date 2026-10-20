"""Minimal local UI for Forge-Mind: type a request -> validated part + buildable kit.

A single-file Streamlit app. ADDITIVE: it only calls the existing ``run_pipeline`` (the SAME
path ``forge-mind run`` uses, ``use_layoutspec=True``), so its behavior matches the CLI. The
engine, validator, and kit code are untouched.

Launch (one command):

    uv run --extra ui streamlit run src/forge_mind/ui/streamlit_app.py

Each "Generate" makes ONE real, paid Anthropic API call (the pipeline's LLM stages) -- expected
for a usable tool. The API key is read from the environment / ``.env`` by the pipeline; it is
never read, shown, or logged by this UI.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from forge_mind.pipeline import PipelineResult, run_pipeline
from forge_mind.validator.report import CheckStatus, ValidationReport
from forge_mind.viz.render import render_part_to_png

OUT_DIR = Path("benchmarks/results/_ui")
_PLACEHOLDER = "a flat plate with a 608 bearing seat and four M3 clearance holes"
# INFO = advisory measured evidence (ADR-0035 B1): shown as a neutral note, never as a failure.
_ICON = {
    CheckStatus.PASS: "✅",
    CheckStatus.WARN: "⚠️",
    CheckStatus.FAIL: "❌",
    CheckStatus.INFO: "📐",  # measured evidence (a ruler, and not in the RUF001 confusable set)
}


def _run(prompt: str) -> PipelineResult:
    """Run the real pipeline. ``run_pipeline`` already captures stage failures into the result
    (it never raises), so the UI gets an inspectable object for every outcome."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return run_pipeline(prompt, OUT_DIR, use_layoutspec=True)


def _show_tiers(report: ValidationReport) -> None:
    """One status chip per validator tier, then the plain-language non-PASS checks."""
    tiers = list(report.tiers.items())
    if tiers:
        for col, (tier, tr) in zip(st.columns(len(tiers)), tiers, strict=True):
            col.metric(tier.value, f"{_ICON[tr.status]} {tr.status.value.upper()}")
    for check in report.checks:
        if check.status not in (CheckStatus.PASS, CheckStatus.INFO):
            st.write(f"{_ICON[check.status]} **{check.id}** -- {check.message}")


def _show_kit(result: PipelineResult) -> None:
    """The buildable kit: BOM table + advisory make-it steps + STL download."""
    kit = result.kit
    if kit is None:
        return

    st.subheader("Bill of materials (what to buy)")
    if kit.bom.lines:
        st.table(
            [
                {
                    "qty": line.quantity,
                    "component": line.component,
                    "length": line.length_note or "",
                    "standard": line.source,
                    "provisional": "yes" if line.provisional else "",
                }
                for line in kit.bom.lines
            ]
        )
    else:
        st.write("No registry-backed hardware.")
    for flagged in kit.bom.unmapped:
        st.warning(f"unmapped (not in the registry): {flagged}")

    st.subheader("Make it (how to fabricate, advisory)")
    st.caption(
        f"material {kit.instructions.material} · layer {kit.instructions.layer_height_mm:g} mm"
    )
    st.write(kit.instructions.wall_note)
    for i, step in enumerate(kit.instructions.steps, 1):
        st.write(f"{i}. {step}")
    if kit.instructions.insert_install_note:
        st.info(kit.instructions.insert_install_note)


def _show_part(result: PipelineResult, prompt: str) -> None:
    """Render the part image and offer the STL download. Best-effort: a render hiccup must not
    take down the page."""
    if result.model is None:
        return
    png = OUT_DIR / "ui_part.png"
    try:
        render_part_to_png(result.model.stl_path, png, title=prompt[:60], caption=prompt)
        st.image(str(png))
    except Exception:  # rendering is cosmetic; never crash the UI over it
        st.write("(render unavailable)")
    stl = Path(result.model.stl_path)
    if stl.exists():
        st.download_button("Download STL", stl.read_bytes(), file_name=stl.name, mime="model/stl")


def _show_result(result: PipelineResult, prompt: str) -> None:
    """Map any pipeline outcome to a clean view: refusal, hard error, or a part + kit."""
    # 1) A clean human refusal (out-of-domain / infeasible): no part, just the message.
    if result.refusal:
        st.warning(f"Refused: {result.refusal}")
        return
    # 2) A hard failure before any part existed (intent/blueprint/compile/validate crash).
    if result.report is None:
        reason = result.error or "no part was produced"
        st.error(f"Could not build a part (failed at {result.failed_stage or 'unknown'}): {reason}")
        return
    # 3) A part exists -- PASS, or built-but-failed-a-check (still honest, still shown).
    if result.passed:
        st.success("PASS -- all validator tiers green")
    else:
        st.error("FAIL -- the part built but did not pass every check (see below)")
    _show_tiers(result.report)
    _show_part(result, prompt)
    _show_kit(result)


def main() -> None:
    st.set_page_config(page_title="Forge-Mind", page_icon="\U0001f6e0️", layout="centered")
    st.title("\U0001f6e0️ Forge-Mind")
    st.caption("Type a mechanical part request. Each run makes a real (paid) API call.")

    prompt = st.text_input("Describe the part", placeholder=_PLACEHOLDER)
    if not st.button("Generate", type="primary", disabled=not prompt.strip()):
        return

    with st.status("Running the pipeline (real API call)…", expanded=False) as status:
        try:
            result = _run(prompt.strip())
        except Exception as exc:  # defensive: show the reason, never a stack trace
            status.update(label="Error", state="error")
            st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
            return
        status.update(
            label=f"Done · path {result.gen_path} · ${result.total_cost_usd:.4f}",
            state="complete",
        )

    _show_result(result, prompt.strip())


main()
