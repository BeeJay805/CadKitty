"""Unit tests for the generation-record types (no I/O)."""

from __future__ import annotations

from forge_mind.learning import GenerationRecord, TokenUsage


def test_to_row_drops_none_and_flattens_usage() -> None:
    rec = GenerationRecord(
        user_id="u1",
        prompt="a hook",
        engine="agentic",
        status="ok",
        usage=TokenUsage(input_tokens=10, output_tokens=20, cost_usd=0.5),
    )
    row = rec.to_row()
    assert row["user_id"] == "u1"
    assert row["engine"] == "agentic"
    assert row["cost_usd"] == 0.5
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 20
    assert row["provider"] == "anthropic"
    # Unset optionals must NOT appear in the insert payload.
    assert "validation" not in row
    assert "files" not in row
    assert "error_type" not in row


def test_from_row_round_trips_key_fields() -> None:
    row = {
        "user_id": "u1",
        "prompt": "a hook",
        "engine": "agentic",
        "status": "ok",
        "cost_usd": 0.5,
        "input_tokens": 10,
        "output_tokens": 20,
        "model": "claude-opus-4-8",
        "validation": {"passed": True},
        "files": {"glb": "x.glb", "stl": "x.stl"},
    }
    rec = GenerationRecord.from_row(row)
    assert rec.user_id == "u1"
    assert rec.usage.cost_usd == 0.5
    assert rec.usage.input_tokens == 10
    assert rec.model == "claude-opus-4-8"
    assert rec.validation == {"passed": True}
    assert rec.files == {"glb": "x.glb", "stl": "x.stl"}
    # The model field must round-trip back into an insert payload (protected-namespace safe).
    assert rec.to_row()["model"] == "claude-opus-4-8"
