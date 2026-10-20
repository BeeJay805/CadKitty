"""ADR-0133 fix 3: FORGE_EDGE_FINISH -- silent edge-finish degradation made visible.

Study 0 measured the class: 5/7 archived fillet-calling shipped parts had ZERO toroidal
faces (safe_fillet silently no-oped) and every gate passed them. The fix records every
safe_fillet/safe_chamfer outcome at the failure point, carries it through the sandbox
result ONLY when flagged, and teaches wall-scaled fillet-first sizing. Never gates.
"""

from __future__ import annotations

import pytest

from forge_mind.agentic.helpers import (
    drain_edge_finish_events,
    safe_chamfer,
    safe_fillet,
)


def _thin_box():
    from build123d import Box

    return Box(30, 30, 2)  # 2 mm wall: a 20 mm fillet cannot succeed


def test_success_records_realized_equals_requested() -> None:
    from build123d import Box

    drain_edge_finish_events()
    part = Box(20, 20, 20)
    out = safe_fillet(part, part.edges(), 1.0)
    events = drain_edge_finish_events()
    assert out.volume < part.volume  # the fillet genuinely happened
    assert events == [{"op": "fillet", "requested": 1.0, "realized": 1.0, "edges": 12}]


def test_oversized_fillet_records_backoff_or_skip_and_never_raises() -> None:
    drain_edge_finish_events()
    part = _thin_box()
    out = safe_fillet(part, part.edges(), 20.0)
    events = drain_edge_finish_events()
    assert out is not None  # NEVER raises -- the graceful contract stands
    assert len(events) == 1 and events[0]["op"] == "fillet"
    realized = events[0]["realized"]
    assert realized is None or realized < 20.0  # the degradation is now VISIBLE
    if realized is None:
        assert out.volume == part.volume  # skipped entirely = unchanged part


def test_chamfer_records_symmetrically() -> None:
    from build123d import Box

    drain_edge_finish_events()
    part = Box(20, 20, 20)
    safe_chamfer(part, part.edges(), 1.0)
    events = drain_edge_finish_events()
    assert events[0]["op"] == "chamfer" and events[0]["realized"] == 1.0


def test_drain_clears_no_cross_part_leakage() -> None:
    from build123d import Box

    drain_edge_finish_events()
    part = Box(10, 10, 10)
    safe_fillet(part, part.edges(), 0.5)
    assert len(drain_edge_finish_events()) == 1
    assert drain_edge_finish_events() == []  # drained = gone (warm-worker isolation)


_SNIPPET = (
    "from build123d import Box\n"
    "part = Box(30, 30, 2)\n"
    "result = safe_fillet(part, part.edges(), 20.0)\n"
)


def test_sandbox_payload_carries_events_only_when_flagged(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real sandbox round-trip: flag off -> result has NO edge_finish (byte-identical
    artifact); flag on -> the skip event rides SandboxResult.edge_finish."""
    from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code

    monkeypatch.delenv("FORGE_EDGE_FINISH", raising=False)
    off = run_build123d_code(
        _SNIPPET, tmp_path / "off", part_name="ef_off", limits=SandboxLimits(timeout_s=180)
    )
    assert off.ok and off.edge_finish is None
    monkeypatch.setenv("FORGE_EDGE_FINISH", "1")
    on = run_build123d_code(
        _SNIPPET, tmp_path / "on", part_name="ef_on", limits=SandboxLimits(timeout_s=180)
    )
    assert on.ok and on.edge_finish is not None and len(on.edge_finish) == 1
    ev = on.edge_finish[0]
    assert ev["op"] == "fillet" and (ev["realized"] is None or ev["realized"] < 20.0)


def test_exact_half_backoff_counts_as_degraded() -> None:
    """Review-caught twice: the boundary must exercise the PRODUCTION predicate (a
    literal-comparison test would survive a revert to <). One failed kernel attempt
    succeeds at EXACTLY 50% -- the most common degradation -- and must be reported."""
    from forge_mind.agentic.loop import _edge_finish_degraded

    events = [
        {"op": "fillet", "requested": 2.0, "realized": 1.0, "edges": 4},  # exact half
        {"op": "fillet", "requested": 2.0, "realized": 2.0, "edges": 4},  # clean
        {"op": "chamfer", "requested": 1.0, "realized": None, "edges": 0},  # skip
        {"op": "fillet", "requested": 2.0, "realized": 1.2, "edges": 4},  # mild backoff
    ]
    degraded = _edge_finish_degraded(events)
    assert [e["realized"] for e in degraded] == [1.0, None]  # half + skip; not clean/mild


def test_cache_key_partitions_by_the_new_prompt_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review-caught: a cache hit returns before authoring, so a prompt-shaping flag
    flip must produce a DIFFERENT key -- else off-mode artifacts get served."""
    from forge_mind.agentic.build_cache import compute_key
    from forge_mind.fabrication import FabricationMode

    def key() -> str:
        return compute_key(
            "a bracket",
            material=None,
            n_drafts=1,
            max_iterations=2,
            fabrication_mode=FabricationMode.PRINT_3D,
            author_model=None,
            critic_model=None,
            brief_model=None,
        )

    monkeypatch.delenv("FORGE_EDGE_FINISH", raising=False)
    monkeypatch.delenv("FORGE_SWEEP_IDIOM", raising=False)
    base = key()
    monkeypatch.setenv("FORGE_EDGE_FINISH", "1")
    with_edge = key()
    assert with_edge != base
    monkeypatch.delenv("FORGE_EDGE_FINISH", raising=False)
    monkeypatch.setenv("FORGE_SWEEP_IDIOM", "1")
    assert key() != base and key() != with_edge


def test_prompts_gain_the_addendum_only_when_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    from forge_mind.agentic import author as au

    monkeypatch.delenv("FORGE_EDGE_FINISH", raising=False)
    monkeypatch.delenv("FORGE_ONECALL_FEATURES", raising=False)
    assert au._system_initial() == au._SYSTEM_INITIAL  # byte-identical off
    monkeypatch.setenv("FORGE_EDGE_FINISH", "1")
    on = au._system_initial()
    assert on.startswith(au._SYSTEM_INITIAL) and "fillet-first" in on
    assert "1/3 of the smallest adjacent wall" in on
    # composes with the one-call flag (both dark levers stack cleanly)
    monkeypatch.setenv("FORGE_ONECALL_FEATURES", "1")
    both = au._system_revise()
    assert both.startswith(au._SYSTEM_REVISE_ONECALL) and "fillet-first" in both
