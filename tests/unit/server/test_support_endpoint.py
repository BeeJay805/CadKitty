"""POST /invention/support (ADR-0050): the read-only support copilot endpoint. Flag-gated behind
FORGE_INVENTION (404 when off), grounded answers cite their KB section, a load/safety question is
REFUSED, an ungrounded question is "unknown", and an injected model phrases ONLY from the grounding.
Money-free: the live answerer is patched out so the deterministic grounding path runs."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from forge_mind.agentic.confidence import standard_lane_label
from forge_mind.dsl.primitives import Box, OpMode
from forge_mind.dsl.program import Constraints, Metadata, PartProgram
from forge_mind.invention import build_invention_package
from forge_mind.invention.support import build_support_kb
from forge_mind.kit import build_kit
from forge_mind.server import app as server_app
from forge_mind.types import Vec3

_UP_Z = Vec3(x=0.0, y=0.0, z=1.0)


def _kb_json() -> dict[str, Any]:
    program = PartProgram(
        metadata=Metadata(name="cable-clip", print_orientation=_UP_Z),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    package = build_invention_package(
        part_name="cable-clip",
        disposition="shipped",
        kit=build_kit(program),
        confidence=standard_lane_label(1),
        problem_statement="route two cables",
    )
    return build_support_kb(package).model_dump(mode="json")


def _post(question: str) -> Any:
    return TestClient(server_app.app).post(
        "/invention/support", json={"kb": _kb_json(), "question": question}
    )


@pytest.fixture()
def enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "invention_enabled", lambda: True)
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    # No live model in tests -> deterministic grounded answers, zero spend.
    monkeypatch.setattr(server_app, "_make_kb_answerer", lambda: None)


def test_support_404_when_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "invention_enabled", lambda: False)
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))
    assert _post("how do I assemble it").status_code == 404


def test_support_answers_grounded_question(enabled: None) -> None:
    r = _post("how do I assemble it")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert "assembly" in body["citations"]
    assert body["answer"] is not None


def test_support_refuses_safety_question(enabled: None) -> None:
    r = _post("can it safely hold 5 kg?")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "refused"
    assert body["citations"] == ["safety_warnings"]


def test_support_unknown_when_ungrounded(enabled: None) -> None:
    r = _post("what is the weather today")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "unknown"
    assert body["answer"] is None


def test_support_empty_question_is_400(enabled: None) -> None:
    assert _post("   ").status_code == 400


def test_support_uses_injected_model_for_grounded_phrasing(
    enabled: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Stub:
        def answer(self, query: str, snippets: list[Any]) -> str:
            return "PHRASED FROM GROUNDING"

    monkeypatch.setattr(server_app, "_make_kb_answerer", lambda: _Stub())
    r = _post("how do I assemble it")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "answered"
    assert body["answer"] == "PHRASED FROM GROUNDING"
