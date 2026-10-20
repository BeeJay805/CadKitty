"""The /generate credit gate resolves the user's entitlement and enforces quality + balance BEFORE
any spend. Displayed == enforced: it reads the SAME `entitlements` row the web app renders, so a plan
can never show one thing and meter another. No network here -- the entitlement read is stubbed."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from forge_mind.server import app as server_app


def _ent(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "plan_id": "free",
        "is_unlimited": False,
        "is_complimentary": False,
        "credits_balance": 10,
        "allowed_qualities": ["standard"],
        "per_quality_cost": {"standard": 1, "studio": 5},
        "upgrade_to": "pro",
    }
    base.update(over)
    return base


def test_metered_standard_charges_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "_get_entitlement", lambda _t: _ent())
    skip, cost = server_app._resolve_credit_gate("tok", "standard")
    assert skip is False
    assert cost == 1


def test_unlimited_bypasses_metering(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_app,
        "_get_entitlement",
        lambda _t: _ent(
            plan_id="pro",
            is_unlimited=True,
            is_complimentary=True,
            allowed_qualities=["standard", "studio"],
            credits_balance=0,
        ),
    )
    # Even at 0 credits, a complimentary/unlimited account builds and is NOT charged.
    skip, cost = server_app._resolve_credit_gate("tok", "studio")
    assert skip is True
    assert cost == 5


def test_quality_not_allowed_returns_structured_paywall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "_get_entitlement", lambda _t: _ent())  # free: standard only
    with pytest.raises(HTTPException) as exc:
        server_app._resolve_credit_gate("tok", "studio")
    assert exc.value.status_code == 402
    detail = exc.value.detail
    assert detail["error_code"] == "quality_not_allowed"
    assert detail["current_plan_id"] == "free"
    assert detail["upgrade_to_plan_id"] == "pro"  # the exact plan that unlocks studio


def test_insufficient_credits_returns_structured_paywall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server_app,
        "_get_entitlement",
        lambda _t: _ent(
            plan_id="pro",
            allowed_qualities=["standard", "studio"],
            credits_balance=2,
            upgrade_to="max",
        ),
    )
    with pytest.raises(HTTPException) as exc:
        server_app._resolve_credit_gate("tok", "studio")  # costs 5, balance 2
    assert exc.value.status_code == 402
    detail = exc.value.detail
    assert detail["error_code"] == "insufficient_credits"
    assert detail["credits_required"] == 5
    assert detail["upgrade_to_plan_id"] == "max"


def test_read_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "_get_entitlement", lambda _t: None)
    with pytest.raises(HTTPException) as exc:
        server_app._resolve_credit_gate("tok", "standard")
    assert exc.value.status_code == 503
