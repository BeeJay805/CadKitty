"""The designed-finish REVALIDATION gate (2026-07-25 review, production precondition).

The finish stage's only ship gate was assembly-fit; the per-part record still described the
CLEAN body, so a finish op that broke a part-level property would have ridden into the report
under a stale per-part PASS. `_finished_part_not_worse` closes that: same gates, same brief,
fail-closed, and NOT-WORSE semantics (a best-at-cap part may stay imperfect -- finish is never
reverted for defects it did not cause).
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from forge_mind.agentic.functional import _finished_part_not_worse
from forge_mind.agentic.loop import LoopResult


def _lr(brief=True, clean_passed=True) -> LoopResult:
    fields = dict.fromkeys(LoopResult.__dataclass_fields__)
    fields.update(
        brief=SimpleNamespace(object_name="x") if brief else None,
        validation=SimpleNamespace(passed=clean_passed) if clean_passed is not None else None,
        stopped_reason="accepted",
    )
    return LoopResult(**fields)


def _patch_validation(monkeypatch, *, hard_ok: bool, passed: bool):
    import forge_mind.agentic.validate as validate_mod

    def fake(stl, step, brief, **kw):
        return SimpleNamespace(hard_ok=hard_ok, passed=passed)

    monkeypatch.setattr(validate_mod, "validate_authored_part", fake)


def test_no_brief_fails_closed(tmp_path) -> None:
    ok, why, _v = _finished_part_not_worse(
        _lr(brief=False), tmp_path / "a.stl", tmp_path / "a.step"
    )
    assert not ok and "no brief" in why


def test_revalidation_crash_fails_closed(tmp_path, monkeypatch) -> None:
    import forge_mind.agentic.validate as validate_mod

    def boom(*a, **k):
        raise RuntimeError("kernel died")

    monkeypatch.setattr(validate_mod, "validate_authored_part", boom)
    ok, why, _v = _finished_part_not_worse(_lr(), tmp_path / "a.stl", tmp_path / "a.step")
    assert not ok and "RuntimeError" in why


def test_finished_part_that_lost_watertightness_reverts(tmp_path, monkeypatch) -> None:
    _patch_validation(monkeypatch, hard_ok=False, passed=False)
    ok, why, _v = _finished_part_not_worse(_lr(), tmp_path / "a.stl", tmp_path / "a.step")
    assert not ok and "watertight" in why


def test_clean_pass_must_stay_a_pass(tmp_path, monkeypatch) -> None:
    _patch_validation(monkeypatch, hard_ok=True, passed=False)
    ok, why, _v = _finished_part_not_worse(
        _lr(clean_passed=True), tmp_path / "a.stl", tmp_path / "a.step"
    )
    assert not ok and "does not" in why


def test_best_at_cap_part_may_stay_imperfect(tmp_path, monkeypatch) -> None:
    """Finish must not be punished for defects it did not cause: a clean part that never fully
    passed (best-at-cap) only owes hard_ok after finishing."""
    _patch_validation(monkeypatch, hard_ok=True, passed=False)
    ok, why, _v = _finished_part_not_worse(
        _lr(clean_passed=False), tmp_path / "a.stl", tmp_path / "a.step"
    )
    assert ok and why == "not-worse"


def test_fully_passing_finish_is_accepted(tmp_path, monkeypatch) -> None:
    _patch_validation(monkeypatch, hard_ok=True, passed=True)
    ok, why, _v = _finished_part_not_worse(_lr(), tmp_path / "a.stl", tmp_path / "a.step")
    assert ok and why == "not-worse"


def test_missing_validation_record_counts_as_not_passed(tmp_path, monkeypatch) -> None:
    """A clean part with NO validation record (checkpoint-resumed) is treated as not-passed,
    so its finish only owes hard_ok -- the resumption-safe read, same as the runner's."""
    _patch_validation(monkeypatch, hard_ok=True, passed=False)
    lr = replace(_lr(), validation=None)
    ok, _why, _v = _finished_part_not_worse(lr, tmp_path / "a.stl", tmp_path / "a.step")
    assert ok


# --- the SINGLE-PART wiring (ADR-0129 fix #7) ----------------------------------------------
# FORGE_FINISH existed only in produce_functional_part, so the flag was a measured no-op on
# every single-part run (S4: armed, zero finish lines). These drive the new
# _maybe_apply_single_part_finish through its gate/apply/revert semantics with the planner and
# kernel ops monkeypatched -- the real ops are covered by the multi-part path's tests.


def _accepted_result(tmp_path):
    from forge_mind.agentic.loop import LoopResult

    stl = tmp_path / "final.stl"
    step = tmp_path / "final.step"
    stl.write_bytes(b"solid x endsolid x")
    step.write_text("STEP;", encoding="utf-8")
    fields = dict.fromkeys(LoopResult.__dataclass_fields__)
    fields.update(
        brief=SimpleNamespace(object_name="widget", summary="a widget"),
        validation=SimpleNamespace(passed=True),
        final_stl=stl,
        final_step=step,
        stopped_reason="accepted",
    )
    return LoopResult(**fields)


def test_single_part_finish_is_a_noop_when_the_flag_is_off(tmp_path, monkeypatch) -> None:
    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.delenv("FORGE_FINISH", raising=False)
    res = _accepted_result(tmp_path)
    assert _maybe_apply_single_part_finish(res, tmp_path, None, None) is res


def test_single_part_finish_swaps_files_when_not_worse(tmp_path, monkeypatch) -> None:
    import trimesh

    import forge_mind.agentic.finish_planner as fp
    import forge_mind.agentic.functional as fn
    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.setenv("FORGE_FINISH", "1")
    res = _accepted_result(tmp_path)
    # a real tiny mesh so the bbox load works
    trimesh.creation.box(extents=(5, 5, 5)).export(str(res.final_stl))

    monkeypatch.setattr(
        fp,
        "plan_part_finish",
        lambda ctx, *, client, model: [SimpleNamespace(op="edge_treatment", params={})],
    )

    def fake_apply(step_in, ops, out_stl, out_step):
        out_stl.write_bytes(b"finished")
        out_step.write_text("STEP;finished", encoding="utf-8")
        return True

    monkeypatch.setattr(fp, "finish_step_file", fake_apply)
    fin_v = SimpleNamespace(passed=True)
    monkeypatch.setattr(fn, "_finished_part_not_worse", lambda lr, s, p: (True, "not-worse", fin_v))
    import forge_mind.agentic.loop as loop_mod

    monkeypatch.setattr(
        loop_mod, "render_part_to_png", lambda stl, png, title=None: png.write_bytes(b"png")
    )

    out = _maybe_apply_single_part_finish(res, tmp_path, object(), None)
    assert out is not res
    assert out.final_stl.name == "part.stl" and out.final_stl.parent.name == "finish"
    assert out.final_stl.read_bytes() == b"finished"
    # review-caught invariants: the judge sees the FINISHED render, the record carries the
    # FINISHED validation, and the applied ops are auditable.
    assert out.final_render is not None and out.final_render.name == "render.png"
    assert out.validation is fin_v and out.geometry_passed is True
    assert (out.final_stl.parent / "finish_ops.json").exists()


def test_single_part_finish_reverts_when_revalidation_fails(tmp_path, monkeypatch) -> None:
    import trimesh

    import forge_mind.agentic.finish_planner as fp
    import forge_mind.agentic.functional as fn
    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.setenv("FORGE_FINISH", "1")
    res = _accepted_result(tmp_path)
    trimesh.creation.box(extents=(5, 5, 5)).export(str(res.final_stl))
    monkeypatch.setattr(
        fp,
        "plan_part_finish",
        lambda ctx, *, client, model: [SimpleNamespace(op="cooling_fins", params={})],
    )

    def fake_apply(step_in, ops, out_stl, out_step):
        out_stl.write_bytes(b"finished")
        out_step.write_text("STEP;finished", encoding="utf-8")
        return True

    monkeypatch.setattr(fp, "finish_step_file", fake_apply)
    monkeypatch.setattr(
        fn,
        "_finished_part_not_worse",
        lambda lr, s, p: (False, "finished part is no longer one watertight solid", None),
    )

    out = _maybe_apply_single_part_finish(res, tmp_path, object(), None)
    assert out is res, "revalidation failure must keep the clean part"


def test_single_part_finish_reverts_when_the_finished_render_fails(tmp_path, monkeypatch) -> None:
    """A part we cannot show honestly is not shipped polished: the acceptance judge reads
    final_render, so a finish whose render fails must revert rather than inherit the clean
    part's image (review-caught)."""
    import trimesh

    import forge_mind.agentic.finish_planner as fp
    import forge_mind.agentic.functional as fn
    import forge_mind.agentic.loop as loop_mod
    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.setenv("FORGE_FINISH", "1")
    res = _accepted_result(tmp_path)
    trimesh.creation.box(extents=(5, 5, 5)).export(str(res.final_stl))
    monkeypatch.setattr(
        fp,
        "plan_part_finish",
        lambda ctx, *, client, model: [SimpleNamespace(op="edge_treatment", params={})],
    )

    def fake_apply(step_in, ops, out_stl, out_step):
        out_stl.write_bytes(b"finished")
        out_step.write_text("STEP;finished", encoding="utf-8")
        return True

    monkeypatch.setattr(fp, "finish_step_file", fake_apply)
    monkeypatch.setattr(fn, "_finished_part_not_worse", lambda lr, s, p: (True, "not-worse", None))

    def boom_render(stl, png, title=None):
        raise RuntimeError("render died")

    monkeypatch.setattr(loop_mod, "render_part_to_png", boom_render)
    assert _maybe_apply_single_part_finish(res, tmp_path, object(), None) is res


def test_single_part_finish_skips_a_blown_budget(tmp_path, monkeypatch) -> None:
    """A blown dollar cap must not fund one more planner call (review-caught)."""
    from dataclasses import replace as _replace

    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.setenv("FORGE_FINISH", "1")
    res = _replace(_accepted_result(tmp_path), stopped_reason="budget")
    assert _maybe_apply_single_part_finish(res, tmp_path, None, None) is res


def test_single_part_finish_excludes_brief_mating_features(tmp_path, monkeypatch) -> None:
    """The planner receives the brief's bore/thread/journal features as CONCRETE exclusions --
    'none declared' left edge_treatment free to chamfer a bearing bore (review-caught)."""
    import trimesh

    import forge_mind.agentic.finish_planner as fp
    from forge_mind.agentic.loop import _maybe_apply_single_part_finish

    monkeypatch.setenv("FORGE_FINISH", "1")
    res = _accepted_result(tmp_path)
    res.brief.features = [
        SimpleNamespace(name="shaft bore", approx_dims_mm="diameter 12 mm through"),
        SimpleNamespace(name="back plate", approx_dims_mm="OD 160 mm x 5 mm"),
    ]
    trimesh.creation.box(extents=(5, 5, 5)).export(str(res.final_stl))
    seen: dict = {}

    def capture(ctx, *, client, model):
        seen["exclusions"] = list(ctx.mating_exclusions)
        return []

    monkeypatch.setattr(fp, "plan_part_finish", capture)
    _maybe_apply_single_part_finish(res, tmp_path, object(), None)
    assert any("shaft bore" in e for e in seen["exclusions"])
    assert not any("back plate" in e for e in seen["exclusions"])
