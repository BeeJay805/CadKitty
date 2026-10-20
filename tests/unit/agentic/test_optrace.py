"""FORGE_OP_TRACE (ADR-0119): the per-op measurement transcript.

Transformer tests are pure (no kernel). One slow test runs the real sandbox with the
flag and checks the transcript lands on the SandboxResult; flag off = field absent
(byte-identical path).
"""

from __future__ import annotations

import pytest

from forge_mind.agentic._sandbox_runner import _instrument_for_optrace, _make_op_probe
from forge_mind.agentic.sandbox import run_build123d_code

_CODE = """
with BuildPart() as bp:
    Box(60, 40, 10)
    with Locations((20, 10)):
        Cylinder(radius=6, height=18)
    Cylinder(radius=4, height=40, mode=Mode.SUBTRACT)
result = bp.part
"""


def test_transformer_probes_each_buildpart_statement() -> None:
    out = _instrument_for_optrace(_CODE)
    assert out is not None
    # one probe per top-level statement in the BuildPart body (3), none elsewhere
    assert out.count("__fm_op_probe(") == 3
    assert "__fm_op_probe(3, bp)" in out  # carries the source line + the builder name
    assert "result = bp.part" in out.replace("\n", " ")


def test_transformer_leaves_non_buildpart_code_unprobed() -> None:
    out = _instrument_for_optrace("x = 1\nwith open('f') as fh:\n    y = 2\n")
    assert out is not None and "__fm_op_probe" not in out


def test_transformer_fails_soft_on_bad_syntax() -> None:
    assert _instrument_for_optrace("with BuildPart( as bp:\n  oops") is None


def test_probe_records_errors_without_raising() -> None:
    trace: list[dict[str, object]] = []
    probe = _make_op_probe(trace)

    class _Boom:
        @property
        def part(self):
            raise RuntimeError("no part yet")

    probe(7, _Boom())
    assert trace == [{"line": 7, "err": "RuntimeError"}]


def test_probe_caps_the_trace() -> None:
    trace: list[dict[str, object]] = [{"line": i} for i in range(64)]
    _make_op_probe(trace)(99, object())
    assert len(trace) == 64  # capped: a pathological script cannot balloon result.json


@pytest.mark.slow
def test_sandbox_carries_the_transcript_under_the_flag(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("FORGE_OP_TRACE", "1")
    res = run_build123d_code(_CODE, tmp_path / "on", part_name="p")
    assert res.ok and res.optrace
    assert [e["line"] for e in res.optrace] == [3, 4, 6]
    assert res.optrace[0]["solids"] == 1 and res.optrace[0]["volume"] > 0

    monkeypatch.delenv("FORGE_OP_TRACE", raising=False)
    res_off = run_build123d_code(_CODE, tmp_path / "off", part_name="p")
    assert res_off.ok and res_off.optrace is None  # flag off = byte-identical path
