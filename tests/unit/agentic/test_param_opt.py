"""Parameter fitting (ADR-0073 Stage 3): the ES converges on an analytic function and is
deterministic; the fit loss prefers the closer FORM and hard-rejects invalid meshes; and the full
polish loop -- with an injectable in-process builder, no sandbox -- moves a box's authored
proportions toward the reference's and reports improvement honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from forge_mind.agentic.param_opt import (
    fit_loss,
    fit_target,
    max_param_evals,
    minimize_es,
    polish_params,
)
from forge_mind.agentic.params import (
    extract_params,
    param_opt_enabled,
    param_opt_note_enabled,
    param_opt_polish_enabled,
)


def test_flag_and_evals_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_PARAM_OPT", raising=False)
    assert param_opt_enabled() is False
    monkeypatch.setenv("FORGE_PARAM_OPT", "1")
    assert param_opt_enabled() is True
    monkeypatch.delenv("FORGE_PARAM_OPT_EVALS", raising=False)
    assert max_param_evals() == 48
    monkeypatch.setenv("FORGE_PARAM_OPT_EVALS", "12")
    assert max_param_evals() == 12
    monkeypatch.setenv("FORGE_PARAM_OPT_EVALS", "junk")
    assert max_param_evals() == 48


def test_split_gates_backward_compat_and_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR-0106: the note (Stage 1) and polish (Stage 3) gates split from one flag. The master
    FORGE_PARAM_OPT still drives BOTH (backward compatible); the per-stage overrides let the note
    be enabled WITHOUT the polish -- the isolation arm the council needs to measure whether the
    author-note alone degrades a reference-less baseline."""
    for e in ("FORGE_PARAM_OPT", "FORGE_PARAM_OPT_NOTE", "FORGE_PARAM_OPT_POLISH"):
        monkeypatch.delenv(e, raising=False)
    # default off -> both off
    assert param_opt_note_enabled() is False and param_opt_polish_enabled() is False
    # master on -> both on (byte-identical to the old single-flag behaviour)
    monkeypatch.setenv("FORGE_PARAM_OPT", "1")
    assert param_opt_note_enabled() is True and param_opt_polish_enabled() is True
    # per-stage override wins over the master: note ON, polish OFF (arm B)
    monkeypatch.setenv("FORGE_PARAM_OPT_POLISH", "0")
    assert param_opt_note_enabled() is True and param_opt_polish_enabled() is False
    # and the reverse: polish ON, note OFF, master unset
    monkeypatch.delenv("FORGE_PARAM_OPT", raising=False)
    monkeypatch.setenv("FORGE_PARAM_OPT_NOTE", "0")
    monkeypatch.setenv("FORGE_PARAM_OPT_POLISH", "1")
    assert param_opt_note_enabled() is False and param_opt_polish_enabled() is True
    assert param_opt_enabled() is False  # master stays the deployed default


def test_conditional_mode_gates_on_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADR-0108 narrowing: FORGE_PARAM_OPT_MODE=conditional fires the engine ONLY for a
    reference-rich part (has_reference), leaving the reference-less majority off -> the ~150 s tax is
    paid only where a gain is plausible. Opt-in: unset (blanket) ignores has_reference, byte-
    identical to before, so the live setting is untouched until the founder flips it."""
    for e in (
        "FORGE_PARAM_OPT",
        "FORGE_PARAM_OPT_NOTE",
        "FORGE_PARAM_OPT_POLISH",
        "FORGE_PARAM_OPT_MODE",
    ):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("FORGE_PARAM_OPT", "1")
    # blanket (default mode): has_reference is IGNORED -> both fire regardless (backward compatible)
    assert param_opt_note_enabled(has_reference=False) is True
    assert param_opt_polish_enabled(has_reference=False) is True
    # conditional: fires only WITH a reference
    monkeypatch.setenv("FORGE_PARAM_OPT_MODE", "conditional")
    assert param_opt_note_enabled(has_reference=True) is True
    assert param_opt_note_enabled(has_reference=False) is False
    assert param_opt_polish_enabled(has_reference=True) is True
    assert param_opt_polish_enabled(has_reference=False) is False
    # conditional does not turn anything ON: master OFF stays off even with a reference
    monkeypatch.delenv("FORGE_PARAM_OPT", raising=False)
    assert param_opt_note_enabled(has_reference=True) is False


def test_es_converges_on_quadratic_and_is_deterministic() -> None:
    target = np.array([12.0, 3.0, 40.0])
    bounds = [(0.0, 20.0), (0.0, 20.0), (0.0, 100.0)]

    def batch(xs: list[np.ndarray]) -> list[float]:
        return [float(np.sum((x - target) ** 2)) for x in xs]

    x1, loss1, evals1 = minimize_es([5.0, 15.0, 80.0], bounds, batch, max_evals=120, seed=0)
    x2, loss2, _ = minimize_es([5.0, 15.0, 80.0], bounds, batch, max_evals=120, seed=0)
    # measured: ~4.1 at 120 evals from a far corner of a 20x20x100 box (1.9 at 200) -- assert with
    # margin; the box-proportions e2e test below covers the real use case
    assert loss1 < 6.0
    assert np.allclose(x1, x2) and loss1 == loss2  # deterministic given the seed
    assert evals1 <= 120
    for v, (lo, hi) in zip(x1, bounds, strict=True):
        assert lo - 1e-9 <= v <= hi + 1e-9  # bounds respected


def test_fit_loss_prefers_the_closer_form(tmp_path: Path) -> None:
    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[40, 10, 10]).export(str(ref))
    target = fit_target(ref)
    assert target is not None
    close = tmp_path / "close.stl"
    trimesh.creation.box(extents=[36, 11, 11]).export(str(close))  # nearly the same proportions
    far = tmp_path / "far.stl"
    trimesh.creation.box(extents=[12, 10, 10]).export(str(far))  # stubby
    assert fit_loss(close, target) < fit_loss(far, target)


def test_fit_loss_hard_rejects_invalid(tmp_path: Path) -> None:
    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[20, 10, 10]).export(str(ref))
    target = fit_target(ref)
    assert target is not None
    # two disconnected bodies -> not a single solid -> +inf (can never win the search)
    a = trimesh.creation.box(extents=[10, 10, 10])
    b = trimesh.creation.box(extents=[10, 10, 10])
    b.apply_translation([50, 0, 0])
    two = tmp_path / "two.stl"
    trimesh.util.concatenate([a, b]).export(str(two))
    assert fit_loss(two, target) == float("inf")
    assert fit_loss(tmp_path / "missing.stl", target) == float("inf")


CODE = 'PARAMS = {"l": 15.0, "w": 10.0, "h": 10.0}\nresult = Box(PARAMS["l"], PARAMS["w"], PARAMS["h"])\n'


def _box_builder(code: str, out_dir: Path) -> tuple[Path | None, Path | None]:
    """In-process stand-in for the sandbox: build the box the candidate's PARAMS describe."""
    spec = extract_params(code)
    if spec is None:
        return None, None
    out_dir.mkdir(parents=True, exist_ok=True)
    stl = out_dir / "part.stl"
    dims = [float(spec.values[k]) for k in ("l", "w", "h")]
    trimesh.creation.box(extents=dims).export(str(stl))
    return stl, stl  # step unused by the search loss


def test_polish_moves_proportions_toward_the_reference(tmp_path: Path) -> None:
    """The end-to-end claim: authored 15x10x10 vs a 40x10x10 reference -- the optimizer must
    lengthen the box (raise l/w) and report a real loss drop, all with zero LLM involvement."""
    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[40, 10, 10]).export(str(ref))
    out = polish_params(CODE, ref, tmp_path / "opt", max_evals=40, build_fn=_box_builder)
    assert out is not None and out.improved
    assert out.best_loss < out.baseline_loss * 0.7  # a real drop, not epsilon noise
    spec = extract_params(out.best_code)
    assert spec is not None
    ratio0 = 15.0 / 10.0
    ratio = float(spec.values["l"]) / float(spec.values["w"])
    assert ratio > ratio0 + 0.4  # the box measurably lengthened toward the reference's 4.0


def test_polish_not_applicable_without_params(tmp_path: Path) -> None:
    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[20, 10, 10]).export(str(ref))
    assert polish_params("result = Box(1,1,1)", ref, tmp_path / "o", build_fn=_box_builder) is None


def test_polish_holds_a_stated_pinned_param(tmp_path: Path) -> None:
    """ADR-0080: the same fit that lengthens a free box (the test above) CANNOT spend a stated
    dimension -- 'l' value-matches the stated 15, so every candidate the ES proposes (and the
    winner) stays inside the ±2% pin, and the pin is itemized in the result."""
    ref = tmp_path / "ref.stl"
    trimesh.creation.box(extents=[40, 10, 10]).export(str(ref))
    out = polish_params(
        CODE,
        ref,
        tmp_path / "opt",
        stated=frozenset({15.0}),
        max_evals=40,
        build_fn=_box_builder,
    )
    assert out is not None
    spec = extract_params(out.best_code)
    assert spec is not None
    half = max(0.02 * 15.0, 0.1)
    assert 15.0 - half - 1e-9 <= float(spec.values["l"]) <= 15.0 + half + 1e-9
    assert "l" in out.pinned
    authored, winner = out.pinned["l"]
    assert authored == 15.0 and abs(winner - 15.0) <= half + 1e-9
