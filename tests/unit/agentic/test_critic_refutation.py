"""The measured-refutation override: pinned to the EXACT L72 veto it was built for.

Gen-5's critic fix, verbatim: "Thicken the runner tubes and collector tube walls to at
least 2.5 mm" -- on walls measured AT 2.5 in a validator-green, audit-clean part. The
override must accept that case and NOTHING wider (fail-closed on unverifiable fixes).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forge_mind.agentic.critic_refutation import OVERRIDE_MARKER, refutation_override

_GREEN = SimpleNamespace(passed=True)
_RED = SimpleNamespace(passed=False)

_L72_WALL_FIX = (
    "Thicken the runner tubes and collector tube walls to at least 2.5 mm, as currently "
    "they appear too thin"
)
_L71_WALL_FIX = "Thicken all walls to meet the minimum required thickness of 2.5 mm"
_FEATURE_FIX = "Add four curved runner tubes extending from each exhaust port, sweeping upwards"


def _critique(fixes: list[str], looks_right: bool = False) -> SimpleNamespace:
    return SimpleNamespace(looks_right=looks_right, prioritized_fixes=fixes)


@pytest.fixture(autouse=True)
def _flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGE_CRITIC_REFUTATION_OVERRIDE", "1")
    monkeypatch.setenv("FORGE_INTERFACE_AUDIT", "1")


def test_the_l72_veto_is_overridden() -> None:
    assert refutation_override(_GREEN, _critique([_L72_WALL_FIX, _L71_WALL_FIX])) is True


def test_feature_requests_are_never_refutable() -> None:
    assert refutation_override(_GREEN, _critique([_FEATURE_FIX])) is False


def test_one_unrefutable_fix_blocks_the_override() -> None:
    assert refutation_override(_GREEN, _critique([_L72_WALL_FIX, _FEATURE_FIX])) is False


def test_red_validation_never_overrides() -> None:
    assert refutation_override(_RED, _critique([_L72_WALL_FIX])) is False


def test_empty_fixes_never_override() -> None:
    assert refutation_override(_GREEN, _critique([])) is False


def test_flag_off_is_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_CRITIC_REFUTATION_OVERRIDE", raising=False)
    assert refutation_override(_GREEN, _critique([_L72_WALL_FIX])) is False


def test_dim_demands_need_the_interface_audit_on(monkeypatch: pytest.MonkeyPatch) -> None:
    fix = "Adjust the runner tubes to match the specified dimensions (24 mm OD)"
    assert refutation_override(_GREEN, _critique([fix])) is True
    monkeypatch.delenv("FORGE_INTERFACE_AUDIT", raising=False)
    assert refutation_override(_GREEN, _critique([fix])) is False


def test_approved_critique_needs_no_override() -> None:
    assert refutation_override(_GREEN, _critique([_L72_WALL_FIX], looks_right=True)) is False


def test_marker_is_a_stable_string() -> None:
    assert "MEASURED REFUTATION" in OVERRIDE_MARKER


_BRIEF = SimpleNamespace(
    features=[
        SimpleNamespace(name="head-flange bolt holes"),
        SimpleNamespace(name="outlet flange"),
        SimpleNamespace(name="runner tubes"),
    ]
)


def test_gen6_add_demand_for_measured_present_feature_is_refuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gen-6 veto: 'Add eight M6 clearance holes...' on a part whose hole-count audit
    gated green. Requires FEATURE_PRESENCE on and a brief-declared feature named."""
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    fix = "Add eight M6 clearance holes around the exhaust ports on the head flange, spaced as described"
    assert refutation_override(_GREEN, _critique([fix]), _BRIEF) is True


def test_blind_negative_feature_is_never_refuted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A green presence aggregate says nothing about slots, which that audit excludes."""
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    brief = SimpleNamespace(features=[SimpleNamespace(name="cable slot")])
    fix = "Add the declared cable slot through the grommet body"
    assert refutation_override(_GREEN, _critique([fix]), brief) is False


def test_shared_token_does_not_refute_the_wrong_feature(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    brief = SimpleNamespace(
        features=[
            SimpleNamespace(name="head flange bolt holes"),
            SimpleNamespace(name="outlet flange"),
        ]
    )
    fix = "Add the head flange bolt holes around the ports"
    assert refutation_override(_GREEN, _critique([fix]), brief) is True


def test_add_demand_for_undeclared_flourish_is_not_refuted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    fix = "Add decorative chamfered accent ribs along the crown"
    assert refutation_override(_GREEN, _critique([fix]), _BRIEF) is False


def test_aesthetic_redesign_demands_stay_unrefutable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    fix = "Redesign the runner tubes to sweep in smooth arcs that converge organically"
    assert refutation_override(_GREEN, _critique([fix]), _BRIEF) is False
