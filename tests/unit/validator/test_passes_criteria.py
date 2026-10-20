"""ValidationReport.passes_criteria — honest scoring with explicit warn allowances.

The scoring fix for the actuated-assembly P2a gap: a statically-valid actuated joint
WARNs on the un-sweepable range_of_motion check, which must NOT be a phantom failure
(report.passed=False, empty failed list) -- but warnings must not pass silently
anywhere else. This pins the full matrix deterministically (no LLM, no geometry).
"""

from __future__ import annotations

from forge_mind.validator.report import Check, CheckStatus, Tier, build_report


def _check(cid: str, status: CheckStatus) -> Check:
    return Check(id=cid, tier=Tier.ASSEMBLY, status=status, message=cid)


def _report(*pairs: tuple[str, CheckStatus]):  # type: ignore[no-untyped-def]
    return build_report([_check(cid, st) for cid, st in pairs], [])


PASS, WARN, FAIL = CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL


def test_all_pass_passes() -> None:
    r = _report(("assembly.hardware_coaxial", PASS), ("assembly.body.hub.single_body", PASS))
    assert r.passes_criteria()
    assert r.passed  # consistent with the plain property when nothing warns


def test_any_fail_blocks_even_with_accepted_warn() -> None:
    r = _report(("assembly.pin_clearance", FAIL), ("assembly.range_of_motion", WARN))
    assert not r.passes_criteria(accepted_warns=["assembly.range_of_motion"])
    # The fail is still reported (never masked).
    assert r.failed_check_ids() == ["assembly.pin_clearance"]


def test_unaccepted_warn_blocks() -> None:
    # The whole point: a WARN does NOT pass unless explicitly accepted.
    r = _report(("assembly.hardware_coaxial", PASS), ("assembly.range_of_motion", WARN))
    assert not r.passes_criteria()  # no allowance -> blocked
    assert not r.passes_criteria(
        accepted_warns=["assembly.some_other_check"]
    )  # wrong id -> blocked


def test_accepted_warn_passes() -> None:
    # A static-actuated joint: hardware_coaxial PASS, range_of_motion WARN (deferred).
    r = _report(("assembly.hardware_coaxial", PASS), ("assembly.range_of_motion", WARN))
    assert r.passes_criteria(accepted_warns=["assembly.range_of_motion"])
    # ...but report.passed is still False (overall is WARN) -- so ONLY a bench that
    # opts in via passes_criteria sees the pass; nothing changes globally.
    assert not r.passed


def test_required_check_must_be_present_and_pass() -> None:
    r = _report(("assembly.range_of_motion", WARN))  # hardware_coaxial absent
    # Even with the warn accepted, a required check that is absent blocks.
    assert not r.passes_criteria(
        required_checks=["assembly.hardware_coaxial"],
        accepted_warns=["assembly.range_of_motion"],
    )
    # Present + passing required check + accepted warn -> passes.
    r2 = _report(("assembly.hardware_coaxial", PASS), ("assembly.range_of_motion", WARN))
    assert r2.passes_criteria(
        required_checks=["assembly.hardware_coaxial"],
        accepted_warns=["assembly.range_of_motion"],
    )


def test_required_check_that_only_warns_does_not_count_as_present() -> None:
    # A required check must PASS, not merely exist -- a warn on a REQUIRED id blocks
    # even if that same id is (contradictorily) in accepted_warns.
    r = _report(("assembly.hardware_coaxial", WARN))
    assert not r.passes_criteria(
        required_checks=["assembly.hardware_coaxial"],
        accepted_warns=["assembly.hardware_coaxial"],
    )


def test_warned_check_ids() -> None:
    r = _report(("a", PASS), ("b", WARN), ("c", WARN), ("d", FAIL))
    assert r.warned_check_ids() == ["b", "c"]
