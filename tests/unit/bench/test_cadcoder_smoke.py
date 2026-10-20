"""CAD-Coder smoke harness (ADR-0112): scored-trial + table logic, driven by a STUB generator so
the suite runs $0 and offline. The real exec bridge and validator DO run (a stub only replaces the
model call), so this also guards that a passed-in CadQuery string scores end-to-end.
"""

from __future__ import annotations

from pathlib import Path

from forge_mind.bench.cadcoder_smoke import PROMPTS, Prompt, TrialResult, render_table, run_trial

_BOX = "import cadquery as cq\nresult = cq.Workplane('XY').box(40, 20, 10)\n"


def test_run_trial_scores_a_valid_box(tmp_path: Path) -> None:
    prompt = Prompt("stub_block", "simple", "a block")
    res = run_trial("stub", prompt, lambda _p: _BOX, tmp_path)
    assert res.executed is True
    assert res.error_type is None
    assert res.arm == "stub" and res.prompt_id == "stub_block"
    assert res.wall_s >= 0.0


def test_run_trial_records_generation_failure_without_raising(tmp_path: Path) -> None:
    def _boom(_p: str) -> str:
        raise RuntimeError("model down")

    res = run_trial("stub", Prompt("x", "simple", "p"), _boom, tmp_path)
    assert res.executed is False
    assert res.error_type is not None and res.error_type.startswith("generation:")


def test_run_trial_records_exec_failure_without_raising(tmp_path: Path) -> None:
    res = run_trial("stub", Prompt("x", "simple", "p"), lambda _p: "result = cq.oops()", tmp_path)
    assert res.executed is False
    assert res.error_type == "exec"


def test_render_table_has_a_row_per_result_and_a_summary() -> None:
    results = [
        TrialResult(arm="cadcoder", prompt_id="a", tier="simple", executed=True, validated=True),
        TrialResult(
            arm="frontier",
            prompt_id="a",
            tier="simple",
            executed=True,
            validated=False,
            failing_tiers=["interfaces"],
        ),
    ]
    md = render_table(results)
    assert "validator-verdict pass@1" in md
    assert "| cadcoder | a | simple |" in md
    assert "interfaces" in md
    assert "## Summary" in md
    # summary counts both arms.
    assert "| cadcoder | 1/1 | 1/1 |" in md
    assert "| frontier | 1/1 | 0/1 |" in md


def test_prompt_set_is_stratified_single_part() -> None:
    tiers = {p.tier for p in PROMPTS}
    assert tiers == {"simple", "multi_feature", "hard"}
    assert len({p.id for p in PROMPTS}) == len(PROMPTS)  # unique ids
