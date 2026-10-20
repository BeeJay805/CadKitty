"""The claude_code provider runs its child in a NEUTRAL cwd: without it, every engine
LLM call inherits the Forge-Mind checkout and silently loads the repo's CLAUDE.md /
hooks / MCP config as session context -- measured to nondeterministically poison
structured output (the v2b/v2c decompose deaths) and to tax every call's latency."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from forge_mind.llm import claude_code_client as ccc


def test_run_uses_a_neutral_empty_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd: list[str], **kwargs: object) -> object:
        captured["cwd"] = kwargs.get("cwd")

        class _P:
            returncode = 0
            stdout = json.dumps(
                {
                    "result": "{}",
                    "is_error": False,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            stderr = b""

        return _P()

    monkeypatch.setattr(ccc.shutil, "which", lambda _n: "claude")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    client = ccc.ClaudeCodeClient()
    client._run("hi", "claude-haiku-4-5", None)

    cwd = captured["cwd"]
    assert cwd is not None, "the child must NOT inherit the engine's cwd"
    cwd_path = Path(str(cwd))
    assert cwd_path.exists()
    # The neutral dir carries no project context the CLI would load as a session.
    assert not (cwd_path / "CLAUDE.md").exists()
    assert not (cwd_path / ".claude").exists()
    # And it is not the repo we are running from.
    assert Path.cwd() != cwd_path
    assert not (cwd_path / "pyproject.toml").exists()


def test_neutral_cwd_is_created_and_stable() -> None:
    a = ccc._neutral_cwd()
    b = ccc._neutral_cwd()
    assert a == b and a.exists() and a.is_dir()
    assert "forge_mind_claude_code_neutral" in str(a)
    assert os.path.basename(str(a)) == "forge_mind_claude_code_neutral"
