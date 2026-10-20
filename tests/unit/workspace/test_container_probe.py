from __future__ import annotations

import pytest

from forge_mind.workspace import container_probe


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_non_linux_probe_refuses_before_request_access(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(container_probe.sys, "platform", platform)
    with pytest.raises(RuntimeError, match="requires Linux"):
        container_probe.main(tmp_path / "missing-request.json")
