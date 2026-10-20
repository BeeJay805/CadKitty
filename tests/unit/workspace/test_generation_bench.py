"""A pilot score must not turn exportability or missing checks into task success."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import workspace_generation_bench as benchmark
from scripts.workspace_generation_bench import parse_args, score

from forge_mind.workspace.models import Measurement

SYNTHETIC_GIT_HEAD = "a" * 40
SYNTHETIC_GIT_STATUS = "?? synthetic-unit-provenance-fixture\n"


@pytest.fixture(autouse=True)
def isolate_benchmark_control_boundaries(monkeypatch):
    """Control Git metadata only; real trials must still read actual Git state."""
    expected_root = Path(benchmark.__file__).resolve().parents[1]

    def git_output(command, *, cwd, text):
        assert Path(cwd).resolve() == expected_root
        assert text is True
        if command == ["git", "rev-parse", "HEAD"]:
            return SYNTHETIC_GIT_HEAD + "\n"
        if command == ["git", "status", "--porcelain"]:
            return SYNTHETIC_GIT_STATUS
        raise AssertionError(f"Unexpected provenance command: {command!r}")

    # Replace this runner's module reference, not shared subprocess functions.
    # Dependencies such as platform still need their normal subprocess calls.
    monkeypatch.setattr(benchmark, "subprocess", SimpleNamespace(check_output=git_output))
    # main() writes these directly. Track both keys so teardown restores the
    # incoming environment, including absent keys, after every control.
    monkeypatch.setenv("FORGE_CODEX_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("FORGE_FORBID_API_SPEND", "1")


def test_missing_and_duplicate_measurements_cannot_complete_numeric_score():
    revision = {
        "measurement": {
            "geometry_valid": True,
            "findings": [
                {"requirement_id": "x", "status": "pass"},
                {"requirement_id": "x", "status": "pass"},
            ],
        }
    }
    result = score(revision, {"x", "y"})
    assert not result["numeric_complete"]
    assert result["numeric_passed"] == 0
    assert result["full_task_success"] is None


def test_geometric_completion_remains_separate_from_full_task_and_physical_claims():
    revision = {
        "measurement": {
            "geometry_valid": True,
            "findings": [
                {"requirement_id": "x", "status": "pass"},
                {"requirement_id": "intent", "status": "unverified"},
            ],
        }
    }
    result = score(revision, {"x"})
    assert result["numeric_complete"]
    assert result["full_task_success"] is None
    assert result["physical_readiness"] == "unverified"
    assert result["findings"][1]["status"] == "unverified"


def test_cli_refuses_to_overwrite_retained_evidence(tmp_path):
    (tmp_path / "results.json").write_text("retained", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--output", str(tmp_path), "--corpus", "cases.json"])
    assert (tmp_path / "results.json").read_text(encoding="utf-8") == "retained"


def test_bound_score_rejects_changed_artifacts_or_contract_and_engine(tmp_path):
    hashes = {}
    for name in ("program.py", "part.step", "part.stl"):
        content = f"Integrity fixture {name}".encode()
        (tmp_path / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    revision = {
        "id": "a" * 32,
        "measurement": Measurement(
            revision_id="a" * 32,
            contract_hash="contract",
            engine_fingerprint="engine",
            geometry_valid=True,
            artifact_hashes=hashes,
            findings=[
                {
                    "requirement_id": "x",
                    "status": "pass",
                    "measured": 40,
                    "expected": 40,
                    "method": "fixture",
                    "reason": "integrity control",
                }
            ],
        ).model_dump(),
    }
    bindings = {"directory": tmp_path, "contract_hash": "contract", "engine_hash": "engine"}
    assert score(revision, {"x"}, **bindings)["evidence_verified"]
    assert score(revision, {"x"}, **bindings)["numeric_complete"]
    assert not score(revision, {"x"}, **{**bindings, "contract_hash": "new"})["numeric_complete"]
    assert not score(revision, {"x"}, **{**bindings, "engine_hash": "new"})["numeric_complete"]
    (tmp_path / "part.step").write_text("tampered")
    result = score(revision, {"x"}, **bindings)
    assert result["evidence_verified"] is False
    assert not result["numeric_complete"]
    assert "hashes" in result["evidence_error"]


def corpus_file(tmp_path, **fields):
    corpus = tmp_path / "corpus-source.json"
    corpus.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "controlled",
                        "prompt": "40 by 30 by 8 mm block",
                        "extents": [40, 30, 8],
                        "solids": 1,
                        **fields,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return corpus


def test_harness_exception_retains_session_error_row_and_runtime_provenance(tmp_path, monkeypatch):
    from scripts.workspace_generation_bench import main

    from forge_mind.workspace.agent import WorkspaceService

    output, corpus = tmp_path / "run", corpus_file(tmp_path)

    def fail(self, job):
        raise RuntimeError("Controlled harness failure after session creation")

    monkeypatch.setattr(WorkspaceService, "run_job", fail)
    with pytest.raises(RuntimeError, match="Controlled harness failure"):
        main(["--output", str(output), "--corpus", str(corpus)])
    snapshot = json.loads((output / "controlled-session.json").read_text())
    row = json.loads((output / "results.json").read_text())[0]
    provenance = json.loads((output / "provenance.json").read_text())
    assert snapshot["prompt"] == "40 by 30 by 8 mm block"
    assert row["state"] == "harness_error"
    assert row["harness_error"]["type"] == "RuntimeError"
    assert row["builds"] == row["model_calls"] == 0
    assert provenance["python"]["executable"]
    assert provenance["git_head"] == SYNTHETIC_GIT_HEAD
    assert provenance["git_status"] == SYNTHETIC_GIT_STATUS
    assert len(provenance["runner"]["sha256"]) == 64
    assert len(provenance["contract_helper"]["sha256"]) == 64
    assert (output / "corpus.json").read_bytes() == corpus.read_bytes()


def test_call_error_and_cli_image_metadata_are_persisted_even_when_harness_aborts(
    tmp_path, monkeypatch
):
    from scripts.workspace_generation_bench import main

    from forge_mind.llm.codex_client import CodexClient
    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.models import Action

    output, corpus = tmp_path / "run", corpus_file(tmp_path)
    image = {"index": 1, "label": "Controlled input", "sha256": "a" * 64, "bytes": 16}

    def failed_call(self, *args, **kwargs):
        self._last_input_images = (image,)
        self._last_call_diagnostics = {"call_id": "controlled", "outcome": "nonzero_exit"}
        raise RuntimeError("Controlled provider failure")

    def fail(self, job):
        client = self.client_factory(timeout_s=10)
        client.generate_structured(
            system="control",
            user=json.dumps({"image_observation": {"revision_id": "revision"}}),
            output_model=Action,
            model="ignored",
            max_tokens=10,
        )

    monkeypatch.setattr(CodexClient, "_generate", failed_call)
    monkeypatch.setattr(WorkspaceService, "run_job", fail)
    with pytest.raises(RuntimeError, match="Controlled provider failure"):
        main(["--output", str(output), "--corpus", str(corpus)])
    record = json.loads((output / "controlled-calls.json").read_text())[0]
    assert record["input_images"] == [image]
    assert record["call_diagnostics"] == {"call_id": "controlled", "outcome": "nonzero_exit"}
    assert record["prepared_image_observation"] == {"revision_id": "revision"}
    assert "Controlled provider failure" in record["error"]
    assert json.loads((output / "results.json").read_text())[0]["model_calls"] == 1


def test_supplied_seed_is_separate_from_model_builds_and_cannot_be_valid(tmp_path, monkeypatch):
    from scripts.workspace_generation_bench import main

    from forge_mind.workspace.agent import WorkspaceService

    output, corpus = (
        tmp_path / "run",
        corpus_file(tmp_path, repair_seed_code="supplied invalid fixture"),
    )

    def seed(self, sid, code, contract, *args):
        revision = {
            "id": "a" * 32,
            "code": code,
            "parent_id": None,
            "measurement": Measurement(
                revision_id="a" * 32,
                contract_hash=contract.fingerprint(),
                engine_fingerprint="fixture",
                geometry_valid=False,
                artifact_hashes={},
                errors=["Controlled seed failure"],
            ).model_dump(),
        }
        self.store.add_revision(sid, revision)
        return revision

    def finish(self, job):
        self.store.finish(job["id"], job["session"], "needs_review")

    monkeypatch.setattr(WorkspaceService, "build", seed)
    monkeypatch.setattr(WorkspaceService, "run_job", finish)
    main(["--output", str(output), "--corpus", str(corpus)])
    row = json.loads((output / "results.json").read_text())[0]
    assert row["trial_kind"] == "controlled_repair"
    assert row["builds"] == 0 and row["total_revisions"] == 1
    assert row["first"]["revision_id"] is None
    assert row["supplied_seed"]["origin"].startswith("Supplied")
    assert json.loads((output / "controlled-supplied-seed.json").read_text())["errors"] == [
        "Controlled seed failure"
    ]

    def valid_seed(self, sid, code, contract, *args):
        revision = seed(self, sid, code, contract, *args)
        revision["measurement"]["geometry_valid"] = True
        return revision

    monkeypatch.setattr(WorkspaceService, "build", valid_seed)
    invalid_output = tmp_path / "rejected-valid-seed"
    with pytest.raises(ValueError, match="must fail actual geometry"):
        main(["--output", str(invalid_output), "--corpus", str(corpus)])
    row = json.loads((invalid_output / "results.json").read_text())[0]
    assert row["state"] == "harness_error"


def test_snapshot_scoring_exception_still_retains_explicit_error_row(tmp_path, monkeypatch):
    from scripts.workspace_generation_bench import main

    from forge_mind.workspace.agent import WorkspaceService
    from forge_mind.workspace.store import Store

    output, corpus = tmp_path / "run", corpus_file(tmp_path)

    def malformed(self, job):
        self.store.add_revision(job["session"], {"id": "a" * 32, "measurement": {}})
        self.store.finish(job["id"], job["session"], "needs_review")

    original = Store.get

    def malformed_snapshot(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        result["revisions"] = [{}]
        result["selected_revision"] = "not-present"
        return result

    monkeypatch.setattr(WorkspaceService, "run_job", malformed)
    monkeypatch.setattr(Store, "get", malformed_snapshot)
    with pytest.raises(KeyError):
        main(["--output", str(output), "--corpus", str(corpus)])
    row = json.loads((output / "results.json").read_text())[0]
    assert row["state"] == "harness_error"
    assert row["scoring_error"].startswith("KeyError")
    assert (output / "controlled-session.json").is_file()
