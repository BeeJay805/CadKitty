"""The 4090-flywheel training tap: validator-labeled (prompt -> code) rows, fail-open.

The dataset is the flywheel's intake -- but logging must NEVER break a run: disabled,
code-less, or erroring inputs all degrade to 'no row', silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge_mind.train_log import load_training_rows, log_part, log_selection


def _lr(code: str | None = "from build123d import *", reason: str = "accepted") -> SimpleNamespace:
    return SimpleNamespace(
        prompt="a bracket",
        brief=SimpleNamespace(object_name="bracket", material="PLA"),
        final_code=code,
        validation=SimpleNamespace(hard_ok=True),
        geometry_passed=True,
        critic_approved=True,
        stopped_reason=reason,
        iterations=[object(), object()],
    )


def test_logs_a_validator_labeled_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    monkeypatch.delenv("FORGE_TRAIN_LOG", raising=False)
    assert log_part(_lr(), material=SimpleNamespace(name="PLA")) is True
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["prompt"] == "a bracket"
    assert row["code"].startswith("from build123d")
    assert row["accepted"] is True and row["hard_ok"] is True
    assert row["material"] == "PLA" and row["iterations"] == 2
    assert len(row["code_sha"]) == 16


def test_appends_not_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    log_part(_lr())
    log_part(_lr(reason="iteration_cap"))
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[1]["accepted"] is False  # non-accepted results still logged, labeled


def test_disabled_and_codeless_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    monkeypatch.setenv("FORGE_TRAIN_LOG", "0")
    assert log_part(_lr()) is False
    monkeypatch.setenv("FORGE_TRAIN_LOG", "1")
    assert log_part(_lr(code=None)) is False  # author_failed: nothing to learn from
    assert not out.exists()


def test_never_raises_on_garbage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(tmp_path / "nested" / "parts.jsonl"))
    assert log_part(object()) is False  # no attrs at all -> False, no exception
    assert log_part(None) is False


# ---- T6 flywheel labels (provenance / spec / verdicts / claimable), additive ----------------


def test_log_part_carries_flywheel_labels_additively(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The agentic row gains provenance (default 'agentic') + the optional spec/checks/claimable
    fields WITHOUT changing any existing key -- old readers keep working."""
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    assert log_part(_lr(), material=SimpleNamespace(name="PLA")) is True
    row = json.loads(out.read_text(encoding="utf-8").strip())
    # every legacy key still present + unchanged (additive contract)
    for key in ("prompt", "code", "code_sha", "accepted", "hard_ok", "iterations", "material"):
        assert key in row
    # the new labels: agentic default, spec/checks/claimable None on the agentic path
    assert row["provenance"] == "agentic"
    assert row["spec"] is None and row["checks"] is None and row["claimable"] is None


def test_log_selection_row_for_a_physics_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A physics-authored part logs a SELECTION row: no build123d code, but the method label +
    spec + gate verdicts + claimable the near-term flywheel learns. Same schema as log_part."""
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    spec = {"envelope_mm": [130.7, 24.0, 64.4], "load": {"force_n": 4.9}}
    checks = {"deflection_bound": "pass", "min_wall": "pass", "slice": "pass"}
    assert (
        log_selection(
            "a link that holds 0.5 kg",
            provenance="analytic_isection",
            spec=spec,
            checks=checks,
            claimable=True,
            material="pla",
            object_name="structural link",
        )
        is True
    )
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert row["provenance"] == "analytic_isection"
    assert row["code"] is None and row["code_sha"] is None  # a selection row, not code training
    assert row["spec"] == spec and row["checks"] == checks and row["claimable"] is True
    assert row["accepted"] is True  # claimable -> accepted, for the same reader filter
    # the schema matches log_part's keys so ONE reader handles both row kinds
    assert set(row) >= {"ts", "source", "provenance", "prompt", "spec", "checks", "claimable"}


def test_log_selection_respects_the_disable_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    monkeypatch.setenv("FORGE_TRAIN_LOG", "0")
    assert log_selection("x", provenance="topopt2d_extrude", claimable=False) is False
    assert not out.exists()


def test_label_env_stamps_the_coder_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Teacher/student separation is a field read (2026-07-21 additive keys). Benches export
    the ollama model env unconditionally, so a cloud-authored row must stamp the AUTHOR
    model, never the exported ollama one."""
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    monkeypatch.delenv("FORGE_TRAIN_LOG", raising=False)
    # local roster: coder=ollama -> the ollama model is the author
    monkeypatch.setenv("FORGE_PROVIDER_CODER", "ollama")
    monkeypatch.setenv("FORGE_OLLAMA_MODEL", "qwen3-coder:30b")
    monkeypatch.setenv("FORGE_AUTHOR_MODEL", "opus")  # present but NOT the coder's model
    log_part(_lr())
    # teacher arm: coder=claude_code -> the author-model env is the author
    monkeypatch.setenv("FORGE_PROVIDER_CODER", "claude_code")
    log_part(_lr())
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["label_env"]["coder_provider"] == "ollama"
    assert rows[0]["label_env"]["coder_model"] == "qwen3-coder:30b"
    assert rows[1]["label_env"]["coder_provider"] == "claude_code"
    assert rows[1]["label_env"]["coder_model"] == "opus"
    assert rows[1]["label_env"]["critic_refutation_override"] is False


def test_hardened_reader_excludes_legacy_and_weak_validator_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "parts.jsonl"
    monkeypatch.setenv("FORGE_TRAIN_LOG_PATH", str(out))
    monkeypatch.setenv("FORGE_FEATURE_PRESENCE", "1")
    assert log_part(_lr()) is True
    stamped = json.loads(out.read_text(encoding="utf-8").strip())
    legacy = {k: v for k, v in stamped.items() if k != "label_env"}
    weak = {**stamped, "label_env": {**stamped["label_env"], "feature_presence": False}}
    out.write_text(
        "\n".join(json.dumps(row) for row in (legacy, weak, stamped)) + "\n",
        encoding="utf-8",
    )
    assert load_training_rows(out) == [stamped]
    assert len(load_training_rows(out, hardened_only=False)) == 3
