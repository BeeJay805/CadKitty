"""API wiring of the physics route (T3 / ADR-0102): flag-gated on FORGE_PHYSICS_ROUTE, ships
only a CLAIMABLE physics-authored part WITH the certificate, and is byte-inert when the flag is
unset. Money-free: the extractor + dispatch loop are mocked at their module boundary and the
agentic engine is stubbed to prove exactly which path ran."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import trimesh

import forge_mind.api.core as core
import forge_mind.structures.dispatch as dispatch_mod
import forge_mind.structures.extract as extract_mod
from forge_mind.api.core import render_certificate_html
from forge_mind.api.schema import CertificateView
from forge_mind.structures.author import AuthorResult
from forge_mind.structures.extract import ExtractedRequirements


class _MustNotReachAgenticError(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_train_log_pollution(monkeypatch: pytest.MonkeyPatch):
    """The physics lane logs a T6 selection row; keep test runs out of the real
    ~/.forge-train/parts.jsonl dataset (the tap is fail-open, so disabling it is invisible)."""
    monkeypatch.setenv("FORGE_TRAIN_LOG", "0")


@pytest.fixture()
def no_agentic(monkeypatch: pytest.MonkeyPatch):
    def _boom(*a: object, **k: object) -> object:
        raise _MustNotReachAgenticError("agentic engine must not run in this test")

    monkeypatch.setattr(core, "produce_part", _boom)


def _extracted(*, load: bool, envelope: bool = True) -> ExtractedRequirements:
    return ExtractedRequirements(
        part_name="test link",
        envelope_mm=(130.7, 24.0, 64.4) if envelope else None,
        has_declared_load=load,
        payload_kg=0.5 if load else None,
        deflection_bound_mm=0.0459 if load else None,
        material="pla",
        load_note="holds 0.5 kg at the tip" if load else "",
    )


def _claimable_outcome(work_dir: Path) -> dict[str, Any]:
    stl = work_dir / "attempt_analytic" / "part.stl"
    stl.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(20.0, 10.0, 5.0)).export(str(stl))
    gate = {
        "checks": {"artifact": "pass", "watertight": "pass", "deflection_bound": "pass"},
        "measured": {"mass_g": 75.6, "deflection_mm": 0.0434, "min_wall_mm": 1.2},
        "claimable": True,
    }
    final = AuthorResult(
        provenance="analytic_isection", stl=stl, step=None, self_reported={"tf_mm": 2.4}
    )
    return {
        "chosen": "analytic",
        "attempts": [
            {
                "rung": "analytic",
                "provenance": "analytic_isection",
                "gate": gate,
                "self_reported": final.self_reported,
            }
        ],
        "final": final,
    }


def test_flag_off_is_byte_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.delenv("FORGE_PHYSICS_ROUTE", raising=False)

    def _no_extract(*a: object, **k: object) -> object:
        raise AssertionError("the extractor must not run when the flag is off")

    monkeypatch.setattr(extract_mod, "extract_requirements", _no_extract)
    res = core.generate_part("a bracket that holds 5 kg with max 0.1 mm deflection", tmp_path)
    # extract never ran; the prompt fell straight to the (stubbed) agentic engine.
    assert res.status == "error" and "MustNotReachAgentic" in (res.error.message or "")


def test_flag_on_stated_load_ships_with_certificate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_ROUTE", "1")
    monkeypatch.setattr(extract_mod, "extract_requirements", lambda *a, **k: _extracted(load=True))
    authored: list[Path] = []

    def _fake_author(spec: Any, work_dir: Path, **kwargs: Any) -> dict[str, Any]:
        authored.append(work_dir)
        assert "agentic" in kwargs.get("authors", {})  # the LLM floor is injected as a refusal
        return _claimable_outcome(work_dir)

    monkeypatch.setattr(dispatch_mod, "author_part", _fake_author)
    res = core.generate_part("a link 130.7 x 24 x 64.4 mm that holds 0.5 kg", tmp_path)
    assert res.status == "ok" and res.ok is not None
    ok = res.ok
    assert ok.gen_path == "physics:analytic_isection"
    assert ok.disposition == "shipped"
    assert ok.validation.passed
    assert ok.cost_usd == 0.0
    # THE CERTIFICATE rides the response: per-check verdicts + echo + provenance + measured
    cert = ok.certificate
    assert cert is not None and cert.claimable
    assert cert.provenance == "analytic_isection"
    assert cert.checks["deflection_bound"] == "pass"
    assert cert.measured["deflection_mm"] == 0.0434
    assert cert.spec_echo["envelope_provenance"] == "stated"
    assert cert.attempts == [
        {"rung": "analytic", "provenance": "analytic_isection", "claimable": True}
    ]
    # servable artifacts + the one-page certificate exist
    assert (tmp_path / "part.stl").exists() and (tmp_path / "part.glb").exists()
    html = (tmp_path / "certificate.html").read_text(encoding="utf-8")
    assert "CLAIMABLE" in html and "analytic_isection" in html
    assert authored and authored[0] == tmp_path / "physics"


def test_flag_on_no_load_routes_agentic_through_production(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    """The honesty bit end-to-end: "sturdy" is NOT a load -> the physics lane declines and the
    PRODUCTION agentic path runs (proven by reaching the stubbed engine)."""
    monkeypatch.setenv("FORGE_PHYSICS_ROUTE", "1")
    monkeypatch.setattr(extract_mod, "extract_requirements", lambda *a, **k: _extracted(load=False))

    def _no_author(*a: object, **k: object) -> object:
        raise AssertionError("author_part must not run for an agentic-routed spec")

    monkeypatch.setattr(dispatch_mod, "author_part", _no_author)
    res = core.generate_part("a sturdy decorative stand", tmp_path)
    assert res.status == "error" and "MustNotReachAgentic" in (res.error.message or "")


def test_flag_on_not_claimable_falls_through_to_agentic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_agentic: None
) -> None:
    monkeypatch.setenv("FORGE_PHYSICS_ROUTE", "1")
    monkeypatch.setattr(extract_mod, "extract_requirements", lambda *a, **k: _extracted(load=True))

    def _refusing(spec: Any, work_dir: Path, **kwargs: Any) -> dict[str, Any]:
        gate = {"checks": {"artifact": "fail"}, "measured": {}, "claimable": False}
        final = AuthorResult(provenance="agentic", stl=None, step=None, self_reported={})
        return {
            "chosen": "analytic",
            "attempts": [
                {"rung": "agentic", "provenance": "agentic", "gate": gate, "self_reported": {}}
            ],
            "final": final,
        }

    monkeypatch.setattr(dispatch_mod, "author_part", _refusing)
    res = core.generate_part("a link that holds 0.5 kg", tmp_path)
    # no claimable physics part -> strictly-no-worse fall-through to the (stubbed) agentic path
    assert res.status == "error" and "MustNotReachAgentic" in (res.error.message or "")
    assert (tmp_path / "certificate.html").exists() is False


def test_declined_physics_yields_a_visible_refusal_certificate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """B3 -- the visible refusal (half the moat): a load WAS declared but no author could prove
    it. _generate_physics returns (None, refusal_cert) so the claimable:false certificate rides
    the agentic fall-through -- attempted, could not prove, here is exactly which check failed."""
    monkeypatch.setenv("FORGE_PHYSICS_ROUTE", "1")
    monkeypatch.setattr(extract_mod, "extract_requirements", lambda *a, **k: _extracted(load=True))

    def _refusing(spec: Any, work_dir: Path, **kwargs: Any) -> dict[str, Any]:
        gate = {
            "checks": {"artifact": "pass", "deflection_bound": "fail"},
            "measured": {"deflection_mm": 0.9},
            "claimable": False,
        }
        final = AuthorResult(provenance="analytic_isection", stl=None, step=None, self_reported={})
        return {
            "chosen": "analytic",
            "attempts": [
                {
                    "rung": "analytic",
                    "provenance": "analytic_isection",
                    "gate": gate,
                    "self_reported": {},
                }
            ],
            "final": final,
        }

    monkeypatch.setattr(dispatch_mod, "author_part", _refusing)
    result, refusal = core._generate_physics(
        "a link 130.7 x 24 x 64.4 mm that holds 5 kg", tmp_path, lambda k, s: None
    )
    assert result is None  # not claimable -> falls through to the agentic engine
    assert refusal is not None  # ...BUT the refusal is now VISIBLE, not silently dropped
    assert refusal.claimable is False
    assert refusal.checks["deflection_bound"] == "fail"
    assert refusal.provenance == "analytic_isection"


def test_certificate_html_escapes_user_text() -> None:
    cert = CertificateView(
        spec_echo={"part_name": "<script>alert(1)</script>", "envelope_provenance": "defaulted"},
        checks={"artifact": "pass", "slice": "unproven"},
        provenance="topopt2d_extrude",
        measured={"mass_g": 200.1},
        caveats=["envelope was not stated; the engine used a labeled default design volume"],
        claimable=False,
        attempts=[{"rung": "topopt", "provenance": "topopt2d_extrude", "claimable": False}],
    )
    html = render_certificate_html(cert, "make me a <b>thing</b>")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<b>thing</b>" not in html and "&lt;b&gt;" in html
    assert "NOT CLAIMABLE" in html and "UNPROVEN" in html and "topopt2d_extrude" in html
