"""Refusal-honesty (serve-but-stamp): a part that BUILT but FAILED validation is delivered for
inspection, clearly STAMPED not-for-manufacture; a passing part carries no stamp. Guards the gap where
the API returned status:ok + a download on validation.passed=false."""

from __future__ import annotations

from typing import Any

from forge_mind.server.app import _apply_manufacture_stamp


def test_failed_validation_is_stamped() -> None:
    payload: dict[str, Any] = {
        "validation": {"passed": False},
        "files": {"stl": "files/x/part.stl"},
    }
    _apply_manufacture_stamp(payload)
    assert payload["not_for_manufacture"] is True
    assert "not for manufacture" in payload["manufacture_warning"].lower()
    # serve-but-stamp: the file URL is still present (the user can inspect it).
    assert payload["files"]["stl"]


def test_passed_validation_is_not_stamped() -> None:
    payload: dict[str, Any] = {"validation": {"passed": True}, "files": {"stl": "x"}}
    _apply_manufacture_stamp(payload)
    assert payload["not_for_manufacture"] is False
    assert payload["manufacture_warning"] is None


def test_missing_validation_defaults_manufacturable() -> None:
    payload: dict[str, Any] = {"files": {}}
    _apply_manufacture_stamp(payload)
    assert payload["not_for_manufacture"] is False
    assert payload["manufacture_warning"] is None


def test_judge_rejected_is_stamped_even_when_validation_passed() -> None:
    """ADR-0035 B8: a part the acceptance judge REJECTED used to serve as a clean ok -- the
    verdict never reached the payload. Now the disposition is projected and stamped."""
    payload: dict[str, Any] = {
        "validation": {"passed": True},
        "disposition": "rejected",
        "files": {"stl": "files/x/part.stl"},
    }
    _apply_manufacture_stamp(payload)
    assert payload["not_for_manufacture"] is True
    assert "acceptance review" in payload["manufacture_warning"]
    assert payload["files"]["stl"]  # serve-but-stamp: still inspectable


def test_shipped_disposition_is_not_stamped() -> None:
    payload: dict[str, Any] = {"validation": {"passed": True}, "disposition": "shipped"}
    _apply_manufacture_stamp(payload)
    assert payload["not_for_manufacture"] is False
    assert payload["manufacture_warning"] is None
