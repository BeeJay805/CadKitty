"""The /lab/* endpoints (ADR-0087 S6b): 404-dark behind FORGE_PROBLEM_MODE, SPEC prompts routed to
/generate BYTE-UNTOUCHED (P12/P17), and the full intake -> answers -> spec -> confirm loop over
injected fakes (no model, no network, no spend before confirm)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh
from fastapi import FastAPI
from fastapi.testclient import TestClient

from forge_mind.intake.session import InMemorySessionStore
from forge_mind.invention.decompose import ProposedNode, ProposedTree
from forge_mind.invention.lifecycle import GeneratedPart
from forge_mind.invention.research import ResearchProposal
from forge_mind.kit import Kit, build_kit
from forge_mind.llm.cost import LLMUsage
from forge_mind.server import app as server_app
from forge_mind.server.lab import create_lab_router

# ---------------------------------------------------------------------------------------------
# the DEPLOYED app: flag gating + byte-untouched spec routing
# ---------------------------------------------------------------------------------------------


@pytest.fixture()
def authed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server_app, "_require_user", lambda auth: ("tok", "user-1"))


def test_all_lab_endpoints_404_when_flag_off(authed: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_PROBLEM_MODE", raising=False)
    c = TestClient(server_app.app)
    assert c.post("/lab/intake", json={"goal": "x"}).status_code == 404
    assert c.post("/lab/answers", json={"session_id": "s", "answers": []}).status_code == 404
    assert c.get("/lab/spec/s").status_code == 404
    assert c.post("/lab/confirm", json={"session_id": "s"}).status_code == 404


def test_spec_prompt_routes_to_generate_byte_untouched(
    authed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P12/P17 on the DEPLOYED app: a plain SPEC prompt comes straight back for today's /generate
    path — byte-identical, no session, no model call (classification is deterministic code)."""
    monkeypatch.setenv("FORGE_PROBLEM_MODE", "1")
    prompt = "a 40 x 40 x 4 mm plate with four M3 holes at the corners"
    r = TestClient(server_app.app).post("/lab/intake", json={"goal": prompt})
    assert r.status_code == 200
    body = r.json()
    assert body == {"route": "spec", "prompt": prompt}  # byte-untouched, nothing else ran


def test_problem_without_context_is_rejected_with_the_exact_ask(
    authed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FORGE_PROBLEM_MODE", "1")
    r = TestClient(server_app.app).post("/lab/intake", json={"goal": "I need one more SSD"})
    assert r.status_code == 422
    assert "context required" in r.json()["detail"]


# ---------------------------------------------------------------------------------------------
# the full loop over injected fakes (a bare app with the same router factory the server mounts)
# ---------------------------------------------------------------------------------------------


def _tree() -> ProposedTree:
    n = ProposedNode
    return ProposedTree(
        nodes=[
            n(node_id="r", parent_id=None, statement="mount an SSD"),
            n(
                node_id="t",
                parent_id="r",
                statement="match the SSD thickness",
                # "other": no playbook -> no base gather; the endpoint test pins the HTTP flow
                # with a MINIMAL form (the gather richness is test_problem_intake's job)
                domain="other",
                closure_kind="fact",
                closure_detail="payload_thickness",
            ),
        ]
    )


class _Client:
    def generate_structured(
        self, *, output_model: type, model: str, **_: Any
    ) -> tuple[Any, LLMUsage]:
        usage = LLMUsage(model=model, input_tokens=1, output_tokens=1)
        if output_model is ProposedTree:
            return _tree(), usage
        if output_model is ResearchProposal:
            return ResearchProposal(), usage
        raise AssertionError(output_model)


def _kit() -> Kit:
    from forge_mind.dsl.primitives import Box, OpMode
    from forge_mind.dsl.program import Constraints, Metadata, PartProgram
    from forge_mind.types import Vec3

    program = PartProgram(
        metadata=Metadata(name="ssd-bracket", print_orientation=Vec3(x=0.0, y=0.0, z=1.0)),
        operations=[Box(op_id="plate", mode=OpMode.ADD, width=40.0, depth=40.0, height=4.0)],
        hardware_features=[],
        constraints=Constraints(min_wall_thickness=4.0),
    )
    return build_kit(program)


class _Produce:
    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, prompt: str, extra_context: str, material: str = "PLA") -> GeneratedPart:
        self.calls.append((prompt, extra_context, material))
        stl = self.tmp / "part.stl"
        trimesh.creation.box(extents=(90.0, 60.0, 8.0)).export(str(stl))
        return GeneratedPart(
            part_name="ssd bracket",
            disposition="shipped",
            kit=_kit(),
            artifacts={"stl": str(stl)},
        )


def _lab_app(tmp: Path) -> tuple[TestClient, _Produce]:
    produce = _Produce(tmp)
    app = FastAPI()
    app.include_router(
        create_lab_router(
            store=InMemorySessionStore(),
            produce=produce,
            client_factory=lambda: _Client(),
            model="m",
            provider_factory=lambda: None,  # research honestly BLOCKED -> the fact is ASKED
            enabled=lambda: True,
        )
    )
    return TestClient(app), produce


def test_full_loop_intake_answers_spec_confirm(tmp_path: Path) -> None:
    client, produce = _lab_app(tmp_path)
    # 1) intake: the single pass returns the COMPLETE typed form
    r = client.post(
        "/lab/intake", json={"goal": "I need to mount one more SSD", "context": "my desk rig"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "problem" and body["stage"] == "awaiting_answers"
    sid = body["session_id"]
    fields = body["form"]["fields"]
    assert [f["id"] for f in fields] == ["payload_thickness", "material"]  # +the base fact
    assert produce.calls == []  # NO SPEND at intake
    # 2) the spec card is readable while answering
    r = client.get(f"/lab/spec/{sid}")
    assert r.status_code == 200 and r.json()["stage"] == "awaiting_answers"
    # 3) answers: one deterministic round -> spec_ready (still NO spend)
    r = client.post(
        "/lab/answers",
        json={
            "session_id": sid,
            "answers": [
                {"id": "payload_thickness", "value": 9.5, "unit": "mm"},
                {"id": "material", "choice": "PETG"},
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "spec_ready" and body["form"] is None
    assert body["material"] == "PETG"
    assert any(f["name"] == "payload_thickness" and f["source"] == "user" for f in body["facts"])
    assert body["card"]  # the confirm card exists
    assert produce.calls == []  # an unconfirmed spec structurally cannot spend (L3)
    # 4) confirm: the ONLY spend path builds the EXACT spec and returns the ledger
    r = client.post("/lab/confirm", json={"session_id": sid})
    assert r.status_code == 200
    body = r.json()
    assert body["stage"] == "built"
    assert body["build"]["disposition"] == "shipped"
    assert len(produce.calls) == 1
    prompt, ctx, material = produce.calls[0]
    assert "payload_thickness: 9.5 mm" in prompt
    assert "(source: user)" in ctx
    assert material == "PETG"  # the user's pick reached the engine adapter


def test_wrong_stage_and_bad_answers_are_409_and_422(tmp_path: Path) -> None:
    client, _produce = _lab_app(tmp_path)
    sid = client.post(
        "/lab/intake", json={"goal": "I need to mount one more SSD", "context": "my desk rig"}
    ).json()["session_id"]
    # confirming before the spec is ready is a 409 (the gate, not a build)
    assert client.post("/lab/confirm", json={"session_id": sid}).status_code == 409
    # an incomplete answer set is a 422 and mutates nothing
    r = client.post("/lab/answers", json={"session_id": sid, "answers": []})
    assert r.status_code == 422 and "unanswered" in r.json()["detail"]
    # unknown sessions are 404
    assert client.get("/lab/spec/nope").status_code == 404
    assert (
        client.post("/lab/answers", json={"session_id": "nope", "answers": []}).status_code == 404
    )


def test_answers_after_spec_ready_are_409(tmp_path: Path) -> None:
    client, _produce = _lab_app(tmp_path)
    sid = client.post(
        "/lab/intake", json={"goal": "I need to mount one more SSD", "context": "my desk rig"}
    ).json()["session_id"]
    ok = client.post(
        "/lab/answers",
        json={
            "session_id": sid,
            "answers": [
                {"id": "payload_thickness", "value": 7.0},
                {"id": "material", "choice": "PLA"},
            ],
        },
    )
    assert ok.json()["stage"] == "spec_ready"
    again = client.post(
        "/lab/answers",
        json={"session_id": sid, "answers": [{"id": "payload_thickness", "value": 9.5}]},
    )
    assert again.status_code == 409


# ---------------------------------------------------------------------------------------------
# S9 metering: the paywall gates BEFORE any engine spend; only a built confirm is charged
# ---------------------------------------------------------------------------------------------


class _Billing:
    """Spy seams mirroring the /generate discipline."""

    def __init__(self, *, skip: bool = False, paywall: bool = False) -> None:
        self.skip = skip
        self.paywall = paywall
        self.gate_calls: list[str | None] = []
        self.charges: list[tuple[str, int, str]] = []

    def gate(self, authorization: str | None) -> tuple[str, bool, int]:
        self.gate_calls.append(authorization)
        if self.paywall:
            from fastapi import HTTPException

            raise HTTPException(status_code=402, detail="insufficient_credits")
        return "tok-1", self.skip, 1

    def charge(self, token: str, cost: int, session_id: str) -> None:
        self.charges.append((token, cost, session_id))


def _metered_app(tmp: Path, billing: _Billing) -> tuple[TestClient, _Produce]:
    produce = _Produce(tmp)
    app = FastAPI()
    app.include_router(
        create_lab_router(
            store=InMemorySessionStore(),
            produce=produce,
            client_factory=lambda: _Client(),
            model="m",
            provider_factory=lambda: None,
            enabled=lambda: True,
            billing_gate=billing.gate,
            billing_charge=billing.charge,
        )
    )
    return TestClient(app), produce


def _to_spec_ready(client: TestClient) -> str:
    sid = client.post(
        "/lab/intake", json={"goal": "I need to mount one more SSD", "context": "my desk rig"}
    ).json()["session_id"]
    r = client.post(
        "/lab/answers",
        json={
            "session_id": sid,
            "answers": [
                {"id": "payload_thickness", "value": 9.5, "unit": "mm"},
                {"id": "material", "choice": "PLA"},
            ],
        },
    )
    assert r.json()["stage"] == "spec_ready"
    return sid


def test_paywall_blocks_confirm_before_any_engine_spend(tmp_path: Path) -> None:
    billing = _Billing(paywall=True)
    client, produce = _metered_app(tmp_path, billing)
    sid = _to_spec_ready(client)
    r = client.post("/lab/confirm", json={"session_id": sid})
    assert r.status_code == 402
    assert produce.calls == []  # the 402 fired BEFORE the engine ran — no spend on a paywall
    assert billing.charges == []


def test_successful_confirm_charges_exactly_once(tmp_path: Path) -> None:
    billing = _Billing()
    client, produce = _metered_app(tmp_path, billing)
    sid = _to_spec_ready(client)
    r = client.post("/lab/confirm", json={"session_id": sid})
    assert r.status_code == 200 and r.json()["stage"] == "built"
    assert len(produce.calls) == 1
    assert billing.charges == [("tok-1", 1, sid)]  # only the real build is charged


def test_unlimited_account_builds_without_a_charge(tmp_path: Path) -> None:
    billing = _Billing(skip=True)
    client, produce = _metered_app(tmp_path, billing)
    sid = _to_spec_ready(client)
    r = client.post("/lab/confirm", json={"session_id": sid})
    assert r.status_code == 200 and len(produce.calls) == 1
    assert billing.charges == []  # complimentary/unlimited: metering bypassed, build allowed


def test_no_billing_seams_means_no_metering(tmp_path: Path) -> None:
    client, produce = _lab_app(tmp_path)  # the un-metered app from the flow tests
    sid = _to_spec_ready(client)
    assert client.post("/lab/confirm", json={"session_id": sid}).status_code == 200
    assert len(produce.calls) == 1
