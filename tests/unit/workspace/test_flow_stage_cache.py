"""Owned stage caching never reuses unsealed, stale, or altered CFD evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from tests.unit.workspace.test_flow_solver import history, settings

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace import flow_stage_cache as cache_module
from forge_mind.workspace.flow_solver import convergence_evidence, solver_config
from forge_mind.workspace.flow_stage_cache import FlowStageCache


def test_atomic_durable_write_retries_transient_windows_sharing_lock(tmp_path, monkeypatch):
    path = tmp_path / "usage.json"
    real_replace = cache_module.os.replace
    attempts = []
    monkeypatch.setattr(cache_module.time, "sleep", lambda seconds: None)

    def sharing_lock(source, target):
        attempts.append(target)
        if len(attempts) < 3:
            raise PermissionError("Windows sharing violation")
        real_replace(source, target)

    monkeypatch.setattr(cache_module.os, "replace", sharing_lock)
    cache_module._seal(path, {"attempts": {}})
    assert len(attempts) == 3
    assert cache_module._unseal(path) == {"attempts": {}}
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.skipif(cache_module.os.name != "nt", reason="Windows denies replacing open readers")
def test_real_windows_progress_reader_cannot_stop_work_and_next_update_recovers(stage_cache):
    cache, _, _, _, _ = stage_cache
    cache.update_progress(phase="meshing")
    with cache.progress_path.open("rb") as held:
        previous = held.read()
        cache.update_progress(phase="solving")
        assert cache.progress_path.read_bytes() == previous
    cache.update_progress(phase="postprocessing")
    assert json.loads(cache.progress_path.read_text())["phase"] == "postprocessing"
    assert not list(cache.root.glob("*.tmp"))


@pytest.mark.skipif(cache_module.os.name != "nt", reason="Windows denies replacing open readers")
def test_real_windows_usage_reader_keeps_accounting_fail_closed(stage_cache):
    cache, _, _, _, _ = stage_cache
    before = cache.usage_path.read_bytes()
    with cache.usage_path.open("rb"), pytest.raises(PermissionError):
        cache._save_usage()
    assert cache.usage_path.read_bytes() == before
    cache._save_usage()
    assert not list(cache.root.glob("*.tmp"))


def test_persistent_diagnostic_write_locks_do_not_interrupt_stage(stage_cache, monkeypatch):
    cache, calls, _, _, _ = stage_cache
    real_replace = cache_module.os.replace
    locked = set()
    monkeypatch.setattr(cache_module.time, "sleep", lambda seconds: None)

    def sharing_lock(source, target):
        if Path(target).name in {"progress.json", "runtime.json"}:
            locked.add(Path(target).name)
            raise PermissionError("Windows persistent reader lock")
        real_replace(source, target)

    monkeypatch.setattr(cache_module.os, "replace", sharing_lock)
    result, directory, reused = cache.run(settings(), "locked diagnostics")
    assert result["eligible"] is True and not reused and len(calls) == 1
    assert locked == {"progress.json", "runtime.json"}
    assert (directory.parent / "receipt.json").is_file()
    assert (
        cache_module._unseal(cache.usage_path)["attempts"][directory.parent.name]["state"]
        == "complete"
    )
    assert not list(cache.root.rglob("*.tmp"))


@pytest.mark.parametrize("failure", ["locked", "malformed"])
def test_diagnostic_read_failure_preserves_phase_and_checkpoint_continues(
    stage_cache, monkeypatch, failure
):
    cache, _, clock, source, _ = stage_cache
    cache.progress["phase"] = "solving"
    cache._active = ("active", 0.0, 0.0)
    cache.usage["attempts"]["active"] = {"seconds": 0.0, "state": "running", "key": "key"}
    clock[0] = 2.0
    real_json = cache_module._json

    def bad_diagnostic(path):
        if path.name in {"runtime.json", "progress.json"}:
            if failure == "locked":
                raise PermissionError("Windows reader lock")
            raise ValueError("Truncated diagnostic JSON")
        return real_json(path)

    monkeypatch.setattr(cache_module, "_json", bad_diagnostic)
    assert cache._checkpoint() is False
    assert cache.progress["phase"] == "solving"
    resumed = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    assert resumed.progress == {}


@pytest.mark.parametrize("target", ["usage.json", "receipt.json", "child_complete.json"])
def test_persistent_durable_write_lock_fails_closed(stage_cache, monkeypatch, target):
    cache, _, _, _, _ = stage_cache
    real_replace, real_link = cache_module.os.replace, cache_module.os.link
    failures = []
    monkeypatch.setattr(cache_module.time, "sleep", lambda seconds: None)

    def guarded(operation, source, destination):
        if Path(destination).name == target:
            failures.append(destination)
            raise PermissionError("Windows persistent evidence lock")
        operation(source, destination)

    monkeypatch.setattr(
        cache_module.os, "replace", lambda source, dest: guarded(real_replace, source, dest)
    )
    monkeypatch.setattr(
        cache_module.os, "link", lambda source, dest: guarded(real_link, source, dest)
    )
    with pytest.raises(PermissionError, match="persistent evidence lock"):
        cache.run(settings(), "locked evidence")
    assert len(failures) >= 4
    assert not list(cache.root.glob("*/receipt.json"))
    assert all(record["state"] != "complete" for record in cache.usage["attempts"].values())


@pytest.fixture
def stage_cache(tmp_path, monkeypatch):
    source = tmp_path / "source.step"
    source.write_text("owned STEP")
    solver = tmp_path / "solver.exe"
    solver.write_text("owned solver")
    monkeypatch.setattr(cache_module, "solver_executable", lambda: solver)
    monkeypatch.setattr(cache_module, "engine_fingerprint", lambda: "engine")
    clock = [0.0]
    monkeypatch.setattr(cache_module.time, "monotonic", lambda: clock[0])
    calls = []

    def evaluate(source, directory, stage_settings, cancelled, progress):
        directory.mkdir()
        for name, value in {
            "source.step": source.read_text(),
            "settings.json": stage_settings.model_dump_json(),
            "run.cfg": solver_config(stage_settings),
            "mesh.su2": "owned mesh",
            "surface_field.json": "{}",
        }.items():
            (directory / name).write_text(value)
        with (directory / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history()[0]))
            writer.writeheader()
            writer.writerows(history())
        measured = convergence_evidence(history(), stage_settings)
        artifacts = {
            p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in directory.iterdir()
        }
        result = {
            "status": "complete",
            "eligible": True,
            "settings": stage_settings.model_dump(mode="json"),
            "metrics": {key: measured[key] for key in ("drag_n", "drag_coefficient")},
            "convergence": measured,
            "wall": {"eligible": True},
            "provenance": {
                "source_step_sha256": file_sha256(source),
                "solver_sha256": file_sha256(solver),
                "settings_sha256": artifacts["settings.json"]["sha256"],
                "config_sha256": artifacts["run.cfg"]["sha256"],
                "initial_config_sha256": artifacts["run.cfg"]["sha256"],
                "mesh_sha256": artifacts["mesh.su2"]["sha256"],
            },
            "artifacts": artifacts,
            "elapsed_seconds": 3.0,
            "solver_replays": [],
        }
        clock[0] += 3
        progress({"stage": "solving"})
        (directory / "flow_result.json").write_text(json.dumps(result))
        return result

    def process(directory, payload, timeout, cancelled):
        calls.append((directory, payload, timeout))
        assert not cancelled()
        result = cache_module.run_flow_stage(directory, payload)
        assert not cancelled()
        return result

    monkeypatch.setattr("forge_mind.workspace.flow_solver.evaluate_flow", evaluate)
    monkeypatch.setattr(cache_module, "process_revision", process)
    cache = FlowStageCache(
        source, tmp_path / "candidate" / "flow-cache", {"protocol": "fixed"}, lambda: False
    )
    return cache, calls, clock, source, solver


def test_exact_stage_reuses_verified_artifacts_across_restart(stage_cache):
    cache, calls, _, source, _ = stage_cache
    original, directory, reused = cache.run(settings(), "coarse")
    assert reused is False and directory.name == "solve"
    assert (directory.parent / "planned.json").is_file()
    assert (directory.parent / "receipt.json").is_file()
    again = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    replay, replay_directory, reused = again.run(settings(), "baseline reused")
    assert reused is True and replay == original and replay_directory == directory
    assert len(calls) == 1
    progress = json.loads(again.progress_path.read_text())
    assert progress["completed_stages"] == 1 and progress["reused_stages"] == 1
    assert progress["stage_directory"] == f"flow-cache/{directory.parent.name}/solve"
    assert progress["elapsed_seconds"] == pytest.approx(3)


@pytest.mark.parametrize("change", ["settings", "identity", "source", "solver", "engine"])
def test_changed_effective_inputs_never_reuse_stage(stage_cache, monkeypatch, change):
    cache, calls, _, source, solver = stage_cache
    cache.run(settings(), "initial")
    effective = settings()
    identity = {"protocol": "fixed"}
    if change == "settings":
        effective.mesh.domain_padding_factor *= 1.5
    elif change == "identity":
        identity = {"protocol": "another"}
    elif change == "source":
        source.write_text("changed STEP")
    elif change == "solver":
        solver.write_text("changed solver")
    else:
        monkeypatch.setattr(cache_module, "engine_fingerprint", lambda: "new engine")
    replacement = FlowStageCache(source, cache.root, identity, lambda: False)
    _, _, reused = replacement.run(effective, "new inputs")
    assert reused is False and len(calls) == 2


@pytest.mark.parametrize(
    "target",
    ["mesh.su2", "flow_result.json", "receipt.json", "child_complete.json", "planned.json"],
)
def test_tampered_artifact_or_receipt_fails_closed(stage_cache, target):
    cache, _, _, _, _ = stage_cache
    _, directory, _ = cache.run(settings(), "initial")
    path = (
        directory / target
        if target in {"mesh.su2", "flow_result.json"}
        else directory.parent / target
    )
    path.write_text("tampered")
    with pytest.raises(ValueError):
        cache.run(settings(), "tampered replay")


@pytest.mark.parametrize("target", ["metrics", "convergence", "wall", "config", "source"])
def test_self_consistent_receipts_cannot_bless_false_evidence(stage_cache, target):
    cache, _, _, _, _ = stage_cache
    result, directory, _ = cache.run(settings(), "initial")
    if target == "metrics":
        result["metrics"]["drag_n"] = 1.0
    elif target == "convergence":
        result["convergence"]["rows"] = 999
    elif target == "wall":
        result["wall"]["eligible"] = False
    else:
        name = "run.cfg" if target == "config" else "source.step"
        (directory / name).write_text("changed frozen input")
        result["artifacts"][name] = {
            "sha256": file_sha256(directory / name),
            "bytes": (directory / name).stat().st_size,
        }
        key = "config_sha256" if target == "config" else "source_step_sha256"
        result["provenance"][key] = result["artifacts"][name]["sha256"]
        if target == "config":
            result["provenance"]["initial_config_sha256"] = result["provenance"][key]
    native = directory / "flow_result.json"
    native.write_text(json.dumps(result))
    child_path = directory.parent / "child_complete.json"
    child = cache_module._unseal(child_path)
    child["result_sha256"] = file_sha256(native)
    cache_module._seal(child_path, child)
    (directory.parent / "receipt.json").unlink()
    with pytest.raises(ValueError):
        cache.run(settings(), "adopt fabricated result")
    assert not (directory.parent / "receipt.json").exists()


@pytest.mark.parametrize("corruption", [None, "physics", "history", "minimum", "policy"])
def test_cold_replay_requires_exact_config_and_recomputed_early_stop(stage_cache, corruption):
    cache, _, _, _, _ = stage_cache
    result, directory, _ = cache.run(settings(), "initial")
    effective = cache_module.FlowSettings.model_validate(result["settings"])
    prefix = "solver-replay-001-"
    config = directory / (prefix + "run.cfg")
    config.write_text(solver_config(effective))
    rows = history()
    rows[1]["rms[P]"] = -1.0  # Only the final instantaneous residual passes.
    archived_history = directory / (prefix + "history.csv")
    with archived_history.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archived = {p.name: file_sha256(p) for p in (config, archived_history)}
    for p in (config, archived_history):
        result["artifacts"][p.name] = {"sha256": file_sha256(p), "bytes": p.stat().st_size}
    result["solver_replays"] = [
        {
            "convergence": convergence_evidence(rows, effective),
            "artifacts": archived,
            "next_minimum_iteration": 10,
        }
    ]
    result["replay_policy"] = (
        "cold replay; delay native checks by one force window; fixed iteration ceiling v1"
    )
    result["convergence"] = convergence_evidence(history(12), effective)
    final_config = solver_config(effective).replace("CONV_STARTITER= 5\n", "CONV_STARTITER= 10\n")
    if corruption == "physics":
        final_config += "CFL_NUMBER= 999\n"
    elif corruption == "history":
        result["solver_replays"][0]["convergence"]["force_gate"] = False
    elif corruption == "minimum":
        result["solver_replays"][0]["next_minimum_iteration"] = 100
    elif corruption == "policy":
        result["replay_policy"] = "arbitrary restart"
    (directory / "run.cfg").write_text(final_config)
    if corruption:
        with pytest.raises(ValueError):
            cache_module._verify_solver_config(directory, effective, result)
    else:
        cache_module._verify_solver_config(directory, effective, result)


def test_complete_child_result_is_adopted_after_parent_receipt_crash(stage_cache, monkeypatch):
    cache, calls, _, source, _ = stage_cache
    real_receipt = cache._receipt
    monkeypatch.setattr(
        cache, "_receipt", lambda *a: (_ for _ in ()).throw(RuntimeError("parent crash"))
    )
    with pytest.raises(RuntimeError, match="parent crash"):
        cache.run(settings(), "interrupted parent")
    stage_directory = calls[0][0]
    assert (stage_directory / "child_complete.json").is_file()
    assert not (stage_directory / "receipt.json").exists()
    monkeypatch.setattr(cache, "_receipt", real_receipt)
    restarted = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    result, directory, reused = restarted.run(settings(), "adopt")
    assert result["eligible"] and reused and directory.parent == stage_directory
    assert len(calls) == 1 and (stage_directory / "receipt.json").exists()


def test_unsealed_native_result_is_not_adopted(stage_cache):
    cache, calls, _, source, _ = stage_cache
    _, directory, _ = cache.run(settings(), "old")
    (directory.parent / "child_complete.json").unlink()
    (directory.parent / "receipt.json").unlink()
    resumed = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    _, replacement, reused = resumed.run(settings(), "retry")
    assert not reused and replacement != directory and len(calls) == 2
    assert (directory / "flow_result.json").is_file()


def test_cancelled_attempt_has_new_directory_and_cumulative_budget(stage_cache, monkeypatch):
    cache, calls, clock, source, _ = stage_cache
    real_process = cache_module.process_revision

    def interrupted(directory, payload, timeout, cancelled):
        calls.append((directory, payload, timeout))
        clock[0] += 4
        cancelled()
        raise ValueError("stopped during meshing")

    monkeypatch.setattr(cache_module, "process_revision", interrupted)
    cache.max_total_seconds = 5
    with pytest.raises(ValueError, match="stopped"):
        cache.run(settings(), "first")
    assert cache.elapsed_seconds == pytest.approx(4)
    clock[0] += 10000  # Downtime must not count against the saved work budget.
    resumed = FlowStageCache(
        source, cache.root, {"protocol": "fixed"}, lambda: False, max_total_seconds=5
    )
    assert resumed.elapsed_seconds == pytest.approx(4)
    observed = []

    def limited(directory, payload, timeout, cancelled):
        observed.append(timeout)
        clock[0] += 2
        assert cancelled()
        raise ValueError("cumulative budget exhausted")

    monkeypatch.setattr(cache_module, "process_revision", limited)
    with pytest.raises(ValueError, match="cumulative wall-time budget"):
        resumed.run(settings(), "resume")
    assert observed == [1]
    assert len(list(cache.root.glob("*/planned.json"))) == 2
    monkeypatch.setattr(cache_module, "process_revision", real_process)
    final = FlowStageCache(
        source, cache.root, {"protocol": "fixed"}, lambda: False, max_total_seconds=5
    )
    with pytest.raises(ValueError, match="cumulative wall-time budget"):
        final.run(settings(), "cannot reset budget")
    assert len(list(cache.root.glob("*/planned.json"))) == 2


def test_checkpoint_does_not_hash_artifacts_and_accepts_keyword_progress(stage_cache, monkeypatch):
    cache, _, clock, _, _ = stage_cache

    def process(directory, payload, timeout, cancelled):
        with monkeypatch.context() as checking:
            checking.setattr(
                cache_module, "file_sha256", lambda *a: pytest.fail("Hash during polling")
            )
            clock[0] += 2
            assert not cancelled()
            cache.update_progress(mesh_level=2, domain_level=1, sensitivity={"eligible": False})
        raise ValueError("end probe")

    monkeypatch.setattr(cache_module, "process_revision", process)
    with pytest.raises(ValueError, match="end probe"):
        cache.run(settings(), "poll")
    progress = json.loads(cache.progress_path.read_text())
    assert progress["mesh_level"] == 2 and progress["domain_level"] == 1
    assert progress["sensitivity"]["eligible"] is False


def test_missing_usage_ledger_cannot_reset_spent_budget(stage_cache):
    cache, _, _, source, _ = stage_cache
    cache.run(settings(), "used")
    cache.usage_path.unlink()
    with pytest.raises(ValueError, match="accounting is missing"):
        FlowStageCache(
            source, cache.root, {"protocol": "fixed"}, lambda: False, max_total_seconds=4
        )


def test_orphan_child_seal_recovers_running_usage_without_charging_downtime(stage_cache):
    cache, _, clock, source, _ = stage_cache
    _, directory, _ = cache.run(settings(), "child finished")
    (directory.parent / "receipt.json").unlink()
    cache.usage["attempts"][directory.parent.name].update(state="running", seconds=1)
    cache._save_usage()
    clock[0] += 20000
    resumed = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    _, _, reused = resumed.run(settings(), "adopt sealed child")
    assert reused and 3 <= resumed.elapsed_seconds < 5


def test_crash_adoption_uses_sealed_timing_when_runtime_diagnostic_is_missing(stage_cache):
    cache, _, clock, source, _ = stage_cache
    _, directory, _ = cache.run(settings(), "child finished")
    (directory.parent / "receipt.json").unlink()
    (directory.parent / "runtime.json").unlink()
    cache.usage["attempts"][directory.parent.name].update(state="running", seconds=1)
    cache._save_usage()
    clock[0] += 20000
    resumed = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    _, _, reused = resumed.run(settings(), "adopt sealed child without telemetry")
    assert reused and 3 <= resumed.elapsed_seconds < 5


def test_child_alive_before_runtime_snapshot_blocks_duplicate_dispatch(stage_cache, monkeypatch):
    from types import SimpleNamespace

    cache, calls, _, source, _ = stage_cache

    def killed_parent(directory, payload, timeout, cancelled):
        calls.append((directory, payload, timeout))
        (directory / "processing.json").write_text(json.dumps(payload))
        raise ValueError("parent interrupted during child imports")

    monkeypatch.setattr(cache_module, "process_revision", killed_parent)
    with pytest.raises(ValueError, match="parent interrupted"):
        cache.run(settings(), "interrupted")
    directory = calls[-1][0]
    monkeypatch.setattr(
        cache_module.psutil,
        "process_iter",
        lambda *a: [
            SimpleNamespace(
                info={
                    "cmdline": [
                        "python",
                        "-m",
                        "forge_mind.workspace.processing",
                        str(directory / "processing.json"),
                    ]
                }
            )
        ],
    )
    resumed = FlowStageCache(source, cache.root, {"protocol": "fixed"}, lambda: False)
    with pytest.raises(ValueError, match="still running"):
        resumed.run(settings(), "must not overlap")
    assert len(calls) == 1


def test_cancelled_before_dispatch_creates_no_stage(stage_cache):
    cache, calls, _, _, _ = stage_cache
    cache.cancelled = lambda: True
    with pytest.raises(ValueError, match="cancelled"):
        cache.run(settings(), "cancelled")
    assert not calls and not list(cache.root.glob("*/planned.json"))


def test_processing_dispatch_uses_owned_stage_entrypoint(tmp_path, monkeypatch):
    from forge_mind.workspace import processing

    monkeypatch.chdir(tmp_path)
    payload = {"operation": "study_flow_stage", "stage_key": "dispatch-proof"}
    request = tmp_path / "request.json"
    request.write_text(json.dumps(payload))
    observed = []

    def stage(directory, actual):
        observed.append((directory.resolve(), actual))
        return {"eligible": False, "status": "unverified"}

    monkeypatch.setattr(cache_module, "run_flow_stage", stage)
    processing.main(request)
    assert observed == [(tmp_path, payload)]
    assert json.loads((tmp_path / "processed.json").read_text())["eligible"] is False
