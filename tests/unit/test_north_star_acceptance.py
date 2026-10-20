"""The north-star loop's tamper-proof acceptance checker (safety-critical): a weakened
spec must NOT be able to declare PASS. This test guards the guardrail -- if it ever goes
green after a spec edit, the loop can cheat and the moat is gone."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_ACC = Path(__file__).resolve().parents[2] / "scripts" / "north_star_loop" / "acceptance.py"
_spec = importlib.util.spec_from_file_location("ns_acceptance", _ACC)
assert _spec and _spec.loader
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)


def test_spec_hash_changes_when_a_condition_is_weakened() -> None:
    strong = [{"path": "motion.joints_validated", "op": "gte", "value": 13, "desc": "x"}]
    weak = [{"path": "motion.joints_validated", "op": "gte", "value": 1, "desc": "x"}]
    assert acc._spec_hash(strong) != acc._spec_hash(weak)
    # order-independent, whitespace-independent (the canonical form is hashed)
    reordered = [{"desc": "x", "value": 13, "op": "gte", "path": "motion.joints_validated"}]
    assert acc._spec_hash(strong) == acc._spec_hash(reordered)


def test_eval_reads_dotted_paths_and_ops() -> None:
    report = {"motion": {"status": "pass", "joints": 13, "art": True}, "disposition": "shipped"}
    assert acc._eval({"path": "motion.status", "op": "eq", "value": "pass"}, report)[0]
    assert acc._eval({"path": "motion.joints", "op": "gte", "value": 13}, report)[0]
    assert not acc._eval({"path": "motion.joints", "op": "gte", "value": 14}, report)[0]
    assert acc._eval({"path": "motion.art", "op": "true"}, report)[0]
    assert acc._eval({"path": "disposition", "op": "in", "value": ["shipped", "x"]}, report)[0]
    assert not acc._eval({"path": "missing.key", "op": "present"}, report)[0]


def test_registry_specs_are_all_locked() -> None:
    """Every seeded north star must have a pinned hash that still matches -- a shipped
    registry with an unlocked or drifted spec is a hole in the guardrail."""
    reg = {ns["name"]: ns for ns in json.loads(acc.REGISTRY.read_text(encoding="utf-8"))}
    lock = json.loads(acc.LOCK.read_text(encoding="utf-8"))
    for name, ns in reg.items():
        assert name in lock, f"{name} is not locked"
        assert lock[name] == acc._spec_hash(ns["accept"]), f"{name} spec drifted from its lock"


def test_check_refuses_a_tampered_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The core guarantee: after a spec condition is weakened, `check` returns non-zero
    (REFUSED) even if the run would satisfy the weakened conditions."""
    reg = [
        {
            "name": "toy",
            "accept": [{"path": "disposition", "op": "eq", "value": "shipped", "desc": "d"}],
        }
    ]
    registry = tmp_path / "north_stars.json"
    lock = tmp_path / "north_stars.lock.json"
    registry.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(acc, "REGISTRY", registry)
    monkeypatch.setattr(acc, "LOCK", lock)
    assert acc.cmd_lock(None) == 0  # human blesses the strong spec

    run = tmp_path / "run"
    run.mkdir()
    (run / "functional_report.json").write_text(
        json.dumps({"disposition": "shipped_with_caveats"}), encoding="utf-8"
    )
    # now the loop weakens 'shipped' -> accept caveats, to make a caveats run 'pass'
    reg[0]["accept"][0]["value"] = "shipped_with_caveats"
    registry.write_text(json.dumps(reg), encoding="utf-8")
    assert acc.cmd_check("toy", str(run)) == 2  # REFUSED: spec hash != lock
    assert acc.cmd_verify("toy") == 1


def test_baselines_lists_the_protected_passing_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Anti-regression: the loop must protect every north star marked passing. baselines
    lists exactly those (a frontier-only registry protects nothing yet)."""
    reg = [
        {"name": "base1", "status": "passing-baseline", "accept": []},
        {"name": "front1", "status": "frontier-stretch", "accept": []},
        {"name": "base2", "status": "passing", "accept": []},
    ]
    registry = tmp_path / "north_stars.json"
    registry.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(acc, "REGISTRY", registry)
    acc.cmd_baselines()
    out = capsys.readouterr().out.split()
    assert set(out) == {"base1", "base2"} and "front1" not in out


def test_metrics_quantifies_efficiency_without_touching_correctness(tmp_path: Path) -> None:
    """Efficiency is report-tier: metrics extracts effort/cost, and is deliberately
    separate from the frozen acceptance spec (a speed number can never relax a gate)."""
    run = tmp_path / "run"
    run.mkdir()
    (run / "functional_report.json").write_text(
        json.dumps(
            {
                "disposition": "shipped_with_caveats",
                "total_cost_usd": 0.0,
                "parts": [
                    {"name": "a", "iterations": 2, "stopped_reason": "accepted"},
                    {"name": "b", "iterations": 3, "stopped_reason": "iteration_cap"},
                ],
            }
        ),
        encoding="utf-8",
    )
    assert acc.cmd_metrics(str(run)) == 0


def test_smell_flags_benchmark_special_casing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generalization guard: a fix that names the north star in engine SOURCE is
    special-casing the instance, not the class. smell must exit 3 (STOP)."""
    reg = [{"name": "planetary", "accept": [], "signature_tokens": ["planetary", "epicyclic"]}]
    registry = tmp_path / "north_stars.json"
    registry.write_text(json.dumps(reg), encoding="utf-8")
    monkeypatch.setattr(acc, "REGISTRY", registry)

    class _Run:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    # a diff that ADDS a special-case branch keyed on the benchmark name
    narrow = '+++ b/src/forge_mind/x.py\n+    if "planetary" in prompt:  # special-case\n+        return hack()\n'
    monkeypatch.setattr(acc.subprocess, "run", lambda *a, **k: _Run(narrow))
    assert acc.cmd_smell("HEAD", "planetary") == 3

    general = "+++ b/src/forge_mind/x.py\n+def four_bar_systems(joints):  # general driver\n+    return detect(joints)\n"
    monkeypatch.setattr(acc.subprocess, "run", lambda *a, **k: _Run(general))
    assert acc.cmd_smell("HEAD", "planetary") == 0


def test_guard_diff_spec_paths_hard_stop_and_validator_paths_ratchet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The founder's 2026-07-12 grant, pinned: the spec/checker stays a HARD STOP
    (exit 3), while validator-tier paths downgrade to the strictness ratchet (exit 4,
    reported not silent). A clean tree is exit 0."""

    class _Run:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    def _fake(files: list[str], porcelain: list[str]):
        calls = iter([_Run("\n".join(files)), _Run("\n".join(porcelain))])
        return lambda *a, **k: next(calls)

    monkeypatch.setattr(
        acc.subprocess, "run", _fake(["scripts/north_star_loop/north_stars.json"], [])
    )
    assert acc.cmd_guard_diff("HEAD") == 3  # spec path: hard stop, no exceptions

    monkeypatch.setattr(
        acc.subprocess, "run", _fake(["src/forge_mind/validator/assembly_checks.py"], [])
    )
    assert acc.cmd_guard_diff("HEAD") == 4  # validator tier: ratchet applies

    monkeypatch.setattr(acc.subprocess, "run", _fake(["src/forge_mind/agentic/functional.py"], []))
    assert acc.cmd_guard_diff("HEAD") == 0  # ordinary engine path: clean
