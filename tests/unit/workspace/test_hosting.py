from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from types import SimpleNamespace

import pytest

from forge_mind.workspace.hosting import (
    HostConfig,
    HostingError,
    HostLock,
    qualification_receipt,
    readiness,
)
from forge_mind.workspace.hosting_identity import canonical_principal


def config(tmp_path):
    root = tmp_path / "store"
    root.mkdir()
    with sqlite3.connect(root / "workspace.sqlite3") as con:
        con.execute("CREATE TABLE jobs (id TEXT, state TEXT, worker_host TEXT)")
    return HostConfig(
        root,
        tmp_path / "spool",
        tmp_path / "runner.sock",
        "sha256:" + "a" * 64,
        "private1",
        "b" * 64,
        ("https://cadkitty.com",),
        "https://auth.example/auth/v1",
        1,
        "c" * 32,
        "test-model",
    )


def test_host_lock_excludes_service_and_maintenance(tmp_path):
    with (
        HostLock(tmp_path),
        pytest.raises(HostingError, match="already serving"),
        HostLock(tmp_path),
    ):
        pass
    with HostLock(tmp_path):
        pass


def test_canonical_principal_is_shared_and_issuer_bound():
    principal = canonical_principal("https://auth.example/auth/v1", "customer-1")
    assert principal == canonical_principal("https://auth.example/auth/v1", "customer-1")
    assert principal != canonical_principal("https://other.example/auth/v1", "customer-1")
    assert principal != canonical_principal("https://auth.example/auth/v1", "customer-2")
    with pytest.raises(HostingError):
        canonical_principal("http://auth.example/auth/v1", "customer-1")


def test_readiness_requires_actual_qualification_and_bound_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_NATIVE_STAGED_PLANNING", "1")
    monkeypatch.setenv("FORGE_CODEX_MODEL", "test-model")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_SECONDS", "1800")
    monkeypatch.setenv("FORGE_CUSTOMER_HOSTED_MAX_ACTIONS", "90")
    policy = config(tmp_path)
    executor = SimpleNamespace(
        typed_container_image=policy.image,
        typed_container_verified=True,
        customer_container_verified=True,
        runner_verified=True,
    )
    assert not readiness(
        policy, current_engine=policy.engine, executor=executor, worker_alive=True
    )["ready"]
    result = readiness(
        policy,
        current_engine=policy.engine,
        executor=executor,
        worker_alive=True,
        provider_qualified=True,
        recovery_qualified=True,
    )
    assert result["ready"]
    assert "spool" not in json.dumps(result)
    assert not readiness(
        policy,
        current_engine="d" * 64,
        executor=executor,
        worker_alive=True,
        provider_qualified=True,
        recovery_qualified=True,
    )["ready"]
    executor.runner_verified = False
    assert not readiness(
        policy,
        current_engine=policy.engine,
        executor=executor,
        worker_alive=True,
        provider_qualified=True,
        recovery_qualified=True,
    )["ready"]


def test_receipts_require_provider_and_restore_runs_at_this_release(tmp_path):
    policy = config(tmp_path)
    assert qualification_receipt(policy) == {"provider": False, "recovery": False}
    receipt = {
        "version": 1,
        "release": policy.release,
        "engine": policy.engine,
        "image": policy.image,
        "provider": {
            "model": policy.provider_model,
            "request_sha256": "a" * 64,
            "response_sha256": "b" * 64,
        },
        "restore": {
            "complete": True,
            "snapshot_manifest_sha256": "c" * 64,
            "restored_inventory_sha256": "d" * 64,
        },
    }
    (policy.root / ".host-qualification.json").write_text(json.dumps(receipt))
    assert qualification_receipt(policy) == {"provider": True, "recovery": True}
    assert qualification_receipt(replace(policy, release="changed")) == {
        "provider": False,
        "recovery": False,
    }
    assert qualification_receipt(replace(policy, provider_model="another"))["provider"] is False


def test_config_cannot_promote_local_or_mutable_worker(tmp_path):
    env = {
        "FORGE_WORKSPACE_DIR": str(tmp_path / "store"),
        "FORGE_RUNNER_SPOOL": str(tmp_path / "spool"),
        "FORGE_RUNNER_SOCKET": str(tmp_path / "runner.sock"),
        "FORGE_WORKSPACE_IMAGE": "sha256:" + "a" * 64,
        "FORGE_HOST_ENGINE": "b" * 64,
        "FORGE_HOST_RELEASE": "private1",
        "FORGE_WORKSPACE_ORIGINS": "https://cadkitty.com",
        "SUPABASE_URL": "https://auth.example",
        "SUPABASE_ANON_KEY": "not-a-real-key",
        "FORGE_RUNNER_FENCE": "c" * 32,
        "FORGE_HOST_PROVIDER_MODEL": "test-model",
        "FORGE_CODEX_MODEL": "test-model",
        "FORGE_NATIVE_STAGED_PLANNING": "1",
    }
    assert HostConfig.from_env(env).root == tmp_path / "store"
    for key, value in (
        ("FORGE_WORKSPACE_IMAGE", "mutable:latest"),
        ("FORGE_WORKSPACE_ORIGINS", "http://localhost:3000"),
        ("FORGE_RUNNER_SPOOL", str(tmp_path / "store/spool")),
    ):
        with pytest.raises(HostingError):
            HostConfig.from_env({**env, key: value})
