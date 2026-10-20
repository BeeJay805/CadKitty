"""ADR-0035 B8 + ADR-0036 at the API boundary: the judge's disposition is projected into
GenerateOk, and the typed brief context threads generate_part -> produce_part."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from forge_mind.api import core


def _fake_part_result(disposition: str) -> Any:
    return SimpleNamespace(
        loop=SimpleNamespace(
            final_stl=Path("x.stl"),
            brief=None,
            validation=None,
            geometry_passed=True,
            stopped_reason="done",
        ),
        verdict=None,
        disposition=disposition,
        total_cost_usd=0.5,
        artifacts={},
        provenance=[],  # part of the PartResult contract since ADR-0040
    )


def test_generate_part_projects_the_disposition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A judge-REJECTED part used to come back as a clean ok with no trace of the verdict
    (api/core served disposition='rejected' as status='ok'). The disposition now rides the
    payload so the server stamps it and the UI can label it."""
    monkeypatch.setattr(core, "produce_part", lambda *a, **k: _fake_part_result("rejected"))
    monkeypatch.setattr(core, "_finalize_files", lambda r, d: True)
    out = core.generate_part("a strange artifact", tmp_path)
    assert out.status == "ok" and out.ok is not None
    assert out.ok.disposition == "rejected"


def test_generate_part_threads_brief_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}

    def _fake_produce(prompt: str, job_dir: Path, **kw: Any) -> Any:
        captured["extra_context"] = kw.get("extra_context")
        return _fake_part_result("shipped")

    monkeypatch.setattr(core, "produce_part", _fake_produce)
    monkeypatch.setattr(core, "_finalize_files", lambda r, d: True)
    out = core.generate_part("a strange artifact", tmp_path, brief_context="STATED FACTS\n- x")
    assert out.status == "ok"
    assert captured["extra_context"] == "STATED FACTS\n- x"


def test_standard_lane_results_carry_no_disposition() -> None:
    """The deterministic lanes run no acceptance judge -- their strict gates ARE the verdict, so
    the field stays None there (the schema default)."""
    from forge_mind.api.schema import GenerateOk

    assert GenerateOk.model_fields["disposition"].default is None
