"""CLI presentation controls with no CAD compiler or native validator execution."""

from __future__ import annotations

import importlib.metadata
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

import forge_mind
from forge_mind.cli.main import app
from forge_mind.validator.report import Check, CheckStatus, Tier, ValidationReport


def _module(monkeypatch: pytest.MonkeyPatch, name: str, **values: Any) -> None:
    module = ModuleType(name)
    module.__dict__.update(values)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("command", ["validate", "layout", "assemble"])
@pytest.mark.parametrize("overall", [CheckStatus.PASS, CheckStatus.WARN, CheckStatus.FAIL])
def test_reports_show_info_and_preserve_result_and_exit(
    command: str, overall: CheckStatus, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = [CheckStatus.PASS, CheckStatus.INFO]
    if overall in {CheckStatus.WARN, CheckStatus.FAIL}:
        statuses.append(CheckStatus.WARN)
    if overall == CheckStatus.FAIL:
        statuses.append(CheckStatus.FAIL)
    report = ValidationReport(
        overall=overall,
        tiers={},
        checks=[
            Check(id=f"test.{s.value}", tier=Tier.MESH, status=s, message=f"{s.value} detail")
            for s in statuses
        ],
    )
    before = report.model_dump_json()
    model = SimpleNamespace(stl_path=Path("part.stl"), step_path=Path("part.step"))
    program = SimpleNamespace(model_validate=lambda value: value)
    _module(monkeypatch, "forge_mind.dsl.program", PartProgram=program)
    _module(monkeypatch, "forge_mind.compiler.compile", compile_program=lambda *_: model)
    _module(monkeypatch, "forge_mind.validator.run", validate=lambda *_: report)
    _module(monkeypatch, "forge_mind.layout.layoutspec", LayoutSpec=dict[str, Any])
    _module(
        monkeypatch,
        "forge_mind.layout.pipeline",
        run_layout_pipeline=lambda *_: SimpleNamespace(
            error=None, model=model, report=report, passed=report.passed
        ),
    )
    _module(
        monkeypatch,
        "forge_mind.assembly.joints",
        AssemblyGraph=SimpleNamespace(model_validate_json=lambda value: value),
    )
    _module(monkeypatch, "forge_mind.assembly.solver", solve=lambda graph: graph)
    _module(
        monkeypatch,
        "forge_mind.compiler.compile_assembly",
        compile_assembly=lambda *_: SimpleNamespace(bodies=[SimpleNamespace(cad=model)]),
    )
    _module(
        monkeypatch, "forge_mind.validator.assembly_checks", validate_assembly=lambda *_: report
    )
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")

    result = CliRunner().invoke(app, [command, str(source), "--out-dir", str(tmp_path / "out")])

    assert result.exit_code == (0 if report.passed else 1), result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"OVERALL: {overall.value.upper()}" in result.output
    assert "STL:" in result.output and "STEP:" in result.output
    for status in statuses:
        assert f"[{status.value.upper()}] test.{status.value}" in result.output
        if status != CheckStatus.PASS:
            assert f"-> {status.value} detail" in result.output
    assert "-> pass detail" not in result.output
    assert report.model_dump_json() == before
    assert source.read_text(encoding="utf-8") == "{}"
    assert not (tmp_path / "out").exists()


def test_cli_version_matches_installed_distribution() -> None:
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"forge-mind {importlib.metadata.version('forge-mind')}"


def test_package_version_reads_distribution_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.metadata, "version", lambda name: "12.34.56" if name == "forge-mind" else None
    )
    namespace = runpy.run_path(str(Path(forge_mind.__file__)))
    assert namespace["__version__"] == "12.34.56"


def test_package_version_without_installed_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    namespace = runpy.run_path(str(Path(forge_mind.__file__)))
    assert namespace["__version__"] == "0+unknown"
