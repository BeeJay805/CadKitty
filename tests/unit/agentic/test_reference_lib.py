"""Sketchfab reference retrieval (Step 2): the relevance guard (name must contain the query's
head noun -> refuse a whole-engine model for 'crankshaft') and fail-closed behavior. Network is
mocked, so this runs offline in CI."""

from __future__ import annotations

from pathlib import Path

import pytest

from forge_mind.agentic import reference_lib
from forge_mind.agentic.reference_lib import _head_noun, fetch_reference_mesh, search_reference


def test_head_noun_is_the_last_word() -> None:
    assert _head_noun("spur gear") == "gear"
    assert _head_noun("crankshaft") == "crankshaft"
    assert _head_noun("4-cylinder engine crankshaft") == "crankshaft"


def test_relevance_guard_skips_whole_assemblies(monkeypatch: pytest.MonkeyPatch) -> None:
    # A 'crankshaft' search returns engines FIRST; the guard skips them for the named match.
    results = {
        "results": [
            {
                "name": "Supercharged V8 Engine",
                "isDownloadable": True,
                "uid": "engine",
                "faceCount": 90000,
            },
            {"name": "2022 Yamaha R1", "isDownloadable": True, "uid": "bike", "faceCount": 90000},
            {
                "name": "Crankshaft (4cyl)",
                "isDownloadable": True,
                "uid": "good",
                "faceCount": 50000,
            },
        ]
    }
    monkeypatch.setattr(reference_lib, "_get_json", lambda url, token, timeout=30.0: results)
    assert search_reference("crankshaft", token="t") == "good"


def test_too_heavy_named_match_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    # A relevant, downloadable model too heavy to render (no decimation backend) is refused.
    results = {
        "results": [
            {"name": "Crankshaft", "isDownloadable": True, "uid": "heavy", "faceCount": 5_000_000}
        ]
    }
    monkeypatch.setattr(reference_lib, "_get_json", lambda url, token, timeout=30.0: results)
    assert search_reference("crankshaft", token="t") is None


def test_no_relevant_match_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    results = {
        "results": [
            {"name": "V8 Engine", "isDownloadable": True, "uid": "a"},
            {"name": "Motorcycle", "isDownloadable": True, "uid": "b"},
        ]
    }
    monkeypatch.setattr(reference_lib, "_get_json", lambda url, token, timeout=30.0: results)
    assert search_reference("crankshaft", token="t") is None


def test_non_downloadable_named_match_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    results = {"results": [{"name": "Crankshaft", "isDownloadable": False, "uid": "nd"}]}
    monkeypatch.setattr(reference_lib, "_get_json", lambda url, token, timeout=30.0: results)
    assert search_reference("crankshaft", token="t") is None


def test_search_falls_back_to_the_head_noun(monkeypatch: pytest.MonkeyPatch) -> None:
    # A verbose brief name ('automotive inline-4 crankshaft') finds nothing named; the bare
    # head noun ('crankshaft') does. The two-tier search must fall back to it.
    def fake(url: str, token: str, timeout: float = 30.0) -> dict:
        if "q=crankshaft&" in url:  # the head-noun fallback query
            return {
                "results": [
                    {
                        "name": "Crankshaft",
                        "isDownloadable": True,
                        "uid": "noun",
                        "faceCount": 40000,
                    }
                ]
            }
        return {"results": [{"name": "Automotive Engine", "isDownloadable": True, "uid": "x"}]}

    monkeypatch.setattr(reference_lib, "_get_json", fake)
    assert search_reference("automotive inline-4 crankshaft", token="t") == "noun"


def test_fetch_fails_closed_without_a_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(reference_lib, "sketchfab_token", lambda: None)
    assert fetch_reference_mesh("crankshaft", tmp_path, token=None) is None
