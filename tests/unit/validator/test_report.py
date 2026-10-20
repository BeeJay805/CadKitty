"""Unit tests for report aggregation."""

from __future__ import annotations

from forge_mind.validator.report import (
    Check,
    CheckStatus,
    Tier,
    aggregate_status,
    build_report,
)


def _c(id: str, tier: Tier, status: CheckStatus) -> Check:
    return Check(id=id, tier=tier, status=status, message="")


def test_aggregate_all_pass() -> None:
    checks = [_c("a", Tier.MESH, CheckStatus.PASS), _c("b", Tier.MESH, CheckStatus.PASS)]
    assert aggregate_status(checks) == CheckStatus.PASS


def test_aggregate_any_fail_is_fail() -> None:
    checks = [_c("a", Tier.MESH, CheckStatus.PASS), _c("b", Tier.MESH, CheckStatus.FAIL)]
    assert aggregate_status(checks) == CheckStatus.FAIL


def test_aggregate_warn_without_fail() -> None:
    checks = [_c("a", Tier.MESH, CheckStatus.PASS), _c("b", Tier.MESH, CheckStatus.WARN)]
    assert aggregate_status(checks) == CheckStatus.WARN


def test_build_report_groups_by_tier() -> None:
    checks = [
        _c("mesh.x", Tier.MESH, CheckStatus.PASS),
        _c("feat.y", Tier.FEATURES, CheckStatus.FAIL),
        _c("iface.z", Tier.INTERFACES, CheckStatus.PASS),
    ]
    report = build_report(checks, [])
    assert report.overall == CheckStatus.FAIL
    assert set(report.tiers.keys()) == {Tier.MESH, Tier.FEATURES, Tier.INTERFACES}
    assert report.tiers[Tier.FEATURES].status == CheckStatus.FAIL
    assert report.failed_check_ids() == ["feat.y"]
    assert report.check("feat.y") is not None
    assert report.check("nonexistent") is None


def test_report_is_json_round_trip_safe() -> None:
    checks = [_c("mesh.x", Tier.MESH, CheckStatus.PASS)]
    report = build_report(checks, [])
    reloaded = type(report).model_validate_json(report.model_dump_json())
    assert reloaded.overall == report.overall
    assert reloaded.checks[0].id == "mesh.x"
