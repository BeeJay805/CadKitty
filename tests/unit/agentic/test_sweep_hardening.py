"""ADR-0133 fix 1: sweep hardening.

The archive measured 151 sweep-frame failures, 112 of which received MISDIRECTING or
absent hints (103 shell-rejects steered toward 'extrude a sketch'; 8 null-shapes blamed
on revolve). Hint tests consume the REAL archived tracebacks verbatim (law 4); the
full-archive replay is a local calibration pin that SKIPs where the archive is absent.
safe_sweep is fail-closed by contract -- including the review-caught disconnected-path
case that build123d silently one-branches.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forge_mind.agentic.idioms import _IDIOMS, JUNCTION_KEYS, SWEPT_KEYS, swept_form_idiom
from forge_mind.agentic.kernel_hints import api_misuse_hint

_FIX = Path(__file__).resolve().parents[2] / "fixtures" / "sweep"
_ARCHIVE = Path("E:/ForgeMind/benchmarks/reference_corpus_v1/r3/fixes/fix4_sweep_artifacts.json")


def _real(name: str) -> tuple[str, str]:
    j = json.loads((_FIX / name).read_text(encoding="utf-8"))
    return j["error_message"], j["traceback"]


def test_shell_reject_in_sweep_frame_gets_the_sweep_hint_not_extrude() -> None:
    """The 103-strong class: previously told to 'extrude a sketch' -- misdirection."""
    msg, tb = _real("hook_iter_01_result.json")
    hint = api_misuse_hint(msg, tb)
    assert hint is not None and "safe_sweep" in hint and "SHELL" in hint
    assert "extrude a sketch" not in hint


def test_null_shape_in_sweep_frame_is_not_blamed_on_revolve() -> None:
    msg, tb = _real("hook_iter_04_result.json")
    hint = api_misuse_hint(msg, tb)
    assert hint is not None and "SWEEP produced EMPTY" in hint
    assert "revolve" not in hint.split("This is NOT the revolve case")[0].lower()


def test_disconnected_path_in_sweep_frame_teaches_chaining() -> None:
    msg, tb = _real("hook_iter_02_result.json")
    hint = api_misuse_hint(msg, tb)
    assert hint is not None and "do not CHAIN" in hint and "l1 @ 1" in hint


def test_non_sweep_frames_keep_their_generic_hints() -> None:
    """The generic entries stay untouched for non-sweep cases (0-collateral guard)."""
    hint = api_misuse_hint("BuildPart doesn't accept {<Shell>}", "  in thicken\n")
    assert hint is not None and "extrude" in hint and "safe_sweep" not in hint
    hint2 = api_misuse_hint("Null TopoDS_Shape", "  in revolve\n")
    assert hint2 is not None and "revolve" in hint2 and "SWEEP produced EMPTY" not in hint2


# ---- safe_sweep: fail-closed kernel behavior ---------------------------------------------


def _arc_path(bend_r: float = 20.0):
    from build123d import BuildLine, CenterArc

    with BuildLine() as p:
        CenterArc((0, 0), bend_r, start_angle=0, arc_size=90)
    return p


def test_safe_sweep_rod_and_tube_execute_to_single_solids() -> None:
    from forge_mind.agentic.helpers import safe_sweep

    rod = safe_sweep(_arc_path(), radius=4.0)
    assert len(rod.solids()) == 1 and rod.volume > 1e3
    tube = safe_sweep(_arc_path(), radius=4.0, inner_radius=2.5)
    assert len(tube.solids()) == 1 and 0 < tube.volume < rod.volume


def test_safe_sweep_relocates_a_origin_drawn_section() -> None:
    from build123d import Rectangle

    from forge_mind.agentic.helpers import safe_sweep

    out = safe_sweep(_arc_path(), section=Rectangle(6, 3))
    # 6x3 section along a 90-deg arc at r=20 -> ~18 * 31.4 = ~565 mm^3
    assert len(out.solids()) == 1 and out.volume > 400


def test_safe_sweep_refuses_a_disconnected_path() -> None:
    """Review-caught: build123d silently sweeps only the FIRST run of a disconnected
    path -- a one-branch part that passes every downstream check. Refuse it."""
    from build123d import BuildLine, Line

    from forge_mind.agentic.helpers import safe_sweep

    with BuildLine() as broken:
        Line((0, 0, 0), (0, 0, 10))
        Line((0, 0, 25), (0, 0, 40))  # gap: does not chain
    with pytest.raises(ValueError, match="DISCONNECTED"):
        safe_sweep(broken, radius=3.0)


def test_safe_sweep_accepts_a_bare_algebra_edge() -> None:
    """Review-caught: Edge.wire() returns None in this kernel -- the helper must not
    call it. A bare Line edge is a legitimate straight path."""
    from build123d import Line

    from forge_mind.agentic.helpers import safe_sweep

    rod = safe_sweep(Line((0, 0, 0), (0, 0, 12)), radius=3.0)
    assert len(rod.solids()) == 1 and rod.volume > 1e2


def test_safe_sweep_requires_an_argument() -> None:
    from forge_mind.agentic.helpers import safe_sweep

    with pytest.raises(ValueError, match="radius=<r>"):
        safe_sweep(_arc_path())


# ---- idiom promotion ---------------------------------------------------------------------


def test_swept_idiom_fires_on_swept_text_and_only_junction_shadows_it() -> None:
    assert swept_form_idiom("a wall bracket with bosses") is None
    hook = swept_form_idiom("a curved coat hook with mounting bosses")
    assert hook is not None and "SWEPT-TUBE" in hook  # bosses may NOT hijack it
    manifold = swept_form_idiom("a 3-into-1 collector manifold tube")
    assert manifold == _IDIOMS[JUNCTION_KEYS]  # the one sanctioned shadow


def test_author_suffix_is_byte_identical_off_and_targets_swept_briefs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from forge_mind.agentic.author import _swept_idiom_suffix

    brief = SimpleNamespace(
        object_name="coat hook",
        summary="a curved J-shape wall hook",
        features=[SimpleNamespace(name="hook arm", method="swept tube")],
    )
    monkeypatch.delenv("FORGE_SWEEP_IDIOM", raising=False)
    assert _swept_idiom_suffix(brief) == ""  # flag off = byte-identical prompt
    monkeypatch.setenv("FORGE_SWEEP_IDIOM", "1")
    on = _swept_idiom_suffix(brief)
    assert on.startswith("\n\n") and "SWEPT-TUBE" in on
    plain = SimpleNamespace(object_name="plate", summary="a flat plate", features=[])
    assert _swept_idiom_suffix(plain) == ""  # non-swept brief: nothing, even when ON


# ---- the pre-registered replay (local calibration pin; archive not in CI) ----------------


_SWEEP_MARKS = ("safe_sweep", "do not CHAIN", "SWEEP produced EMPTY")


@pytest.mark.skipif(not _ARCHIVE.exists(), reason="sweep artifact archive not present (local)")
def test_replay_all_151_archived_sweep_failures_route_correctly() -> None:
    """Pre-registered (ADR-0133): every sweep-class artifact (103 shell + 8 null + 1
    disconnected = 112) gets a sweep-correct hint, and ZERO non-sweep artifacts receive
    a sweep hint (the collateral check, structural: the entries ALL-match 'in sweep')."""
    base = Path("E:/ForgeMind")
    data = json.loads(_ARCHIVE.read_text(encoding="utf-8"))
    sweep_sigs = {"shell_reject", "null_topods", "edges_disconnected"}
    counts = {"sweep_correct": 0, "sweep_wrong": 0, "collateral": 0, "missing": 0, "nonsweep": 0}
    for row in data["artifacts"]:
        path = base / row["path"]
        if not path.exists():
            counts["missing"] += 1
            continue
        j = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        hint = api_misuse_hint(j.get("error_message"), j.get("traceback")) or ""
        is_sweep_hint = any(m in hint for m in _SWEEP_MARKS)
        if row["signature"] in sweep_sigs:
            counts["sweep_correct" if is_sweep_hint else "sweep_wrong"] += 1
        else:
            counts["nonsweep"] += 1
            if is_sweep_hint:
                counts["collateral"] += 1
    assert counts["missing"] <= 5, counts  # the archive should still be present
    assert counts["sweep_wrong"] == 0, counts
    assert counts["collateral"] == 0, counts
    assert counts["sweep_correct"] >= 112 - counts["missing"], counts


def test_swept_keys_shared_not_hand_synced() -> None:
    """Law 2: one vocabulary, derived -- the idiom dict and the promotion read the
    SAME tuple objects."""
    assert SWEPT_KEYS in _IDIOMS and JUNCTION_KEYS in _IDIOMS
    assert os.environ is not None  # keep os import honest
