"""ADR-0109: the DEFAULT (agentic) path now emits a MANUFACTURABILITY certificate -- the
trustworthy verdict on EVERY part, not only the narrow physics lane. The full validator's per-check
verdicts map to the same legible CertificateView, scoped honestly (geometry, NOT load).

The mapper reads a small slice of PartResult, so it is mocked with SimpleNamespace + real
CheckStatus values -- this isolates the mapping + the honesty caveats.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import forge_mind.api.core as core
from forge_mind.fabrication import FabricationMode
from forge_mind.validator.report import CheckStatus


def _result(statuses: list[CheckStatus], *, passed: bool, features_missing: bool = False) -> Any:
    checks = [SimpleNamespace(id=f"c{i}.{s.value}", status=s) for i, s in enumerate(statuses)]
    val = SimpleNamespace(
        report=SimpleNamespace(checks=checks),
        passed=passed,
        hard_ok=passed,
        features_missing=features_missing,
    )
    brief = SimpleNamespace(
        object_name="a right-angle bracket",
        overall_envelope_mm="40 x 40 x 10",
        features=[1, 2, 3],
        material="PLA",
    )
    return SimpleNamespace(
        loop=SimpleNamespace(validation=val, brief=brief), disposition="accepted"
    )


def test_maps_every_verdict_and_scopes_to_manufacturability() -> None:
    result = _result(
        [CheckStatus.PASS, CheckStatus.FAIL, CheckStatus.WARN, CheckStatus.INFO], passed=False
    )
    cert = core._agentic_certificate(result, FabricationMode.GENERAL, "a bracket")
    assert cert is not None
    assert set(cert.checks.values()) == {"pass", "fail", "unproven", "n/a"}  # PASS/FAIL/WARN/INFO
    assert cert.claimable is False  # a FAIL is present -> not claimable
    assert cert.provenance == "agentic"
    assert cert.spec_echo["object"] == "a right-angle bracket"
    # the load-scope honesty is MANDATORY -- a geometry cert must never read as a load claim
    assert any("not load capacity" in c.lower() for c in cert.caveats)


def test_all_pass_is_claimable() -> None:
    result = _result([CheckStatus.PASS, CheckStatus.PASS, CheckStatus.INFO], passed=True)
    cert = core._agentic_certificate(result, FabricationMode.GENERAL, "x")
    assert cert is not None
    assert cert.claimable is True
    assert all(v in ("pass", "n/a") for v in cert.checks.values())


def test_missing_features_surfaces_a_caveat() -> None:
    result = _result([CheckStatus.FAIL], passed=False, features_missing=True)
    cert = core._agentic_certificate(result, FabricationMode.GENERAL, "x")
    assert cert is not None
    assert any("missing" in c.lower() for c in cert.caveats)


def test_no_report_yields_no_certificate() -> None:
    # nothing deterministic to certify -> None (never an empty/guessed certificate)
    no_report = SimpleNamespace(loop=SimpleNamespace(validation=None, brief=None), disposition="")
    assert core._agentic_certificate(no_report, FabricationMode.GENERAL, "x") is None
