"""Numerical failure cannot become an eligible aerodynamic candidate."""

import csv
import json
import math
import sys
import time
from threading import Timer

import pytest
from pydantic import ValidationError

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace import flow_solver
from forge_mind.workspace.flow_models import FlowSettings
from forge_mind.workspace.flow_solver import (
    FlowStoppedError,
    _run_owned,
    convergence_evidence,
    evaluate_flow,
    read_history,
    solver_config,
    surface_field,
)


def settings(**updates):
    return FlowSettings.model_validate(
        {
            "density_kg_m3": 1.2,
            "dynamic_viscosity_pa_s": 0.1,
            "velocity_m_s": [3, 4, 0],
            "reference_area_m2": 2,
            "reference_length_m": 1,
            "numerics": {"force_window": 5},
            **updates,
        }
    )


def history(count=6):
    return [
        {
            "Inner_Iter": float(i),
            "rms[P]": -8.0,
            "rms[U]": -7.0,
            "rms[V]": -9.0,
            "rms[W]": -10.0,
            "CFx": 0.6,
            "CFy": 0.8,
            "CFz": 0.0,
            "RefForce": 30.0,
        }
        for i in range(count)
    ]


@pytest.mark.parametrize(
    "updates",
    [
        {"velocity_m_s": [0, 0, 0]},
        {"density_kg_m3": float("nan")},
        {"dynamic_viscosity_pa_s": 0},
        {"reference_area_m2": -1},
        {"numerics": {"iterations": 20, "force_window": 30}},
        {"solver_path": "arbitrary"},
    ],
)
def test_undefined_physics_or_command_inputs_are_rejected(updates):
    with pytest.raises(ValidationError):
        settings(**updates)


def test_force_projection_uses_frozen_vector_and_reference_area():
    result = convergence_evidence(history(), settings())
    assert result["eligible"]
    assert result["drag_coefficient"] == pytest.approx(1)
    assert result["drag_n"] == pytest.approx(30)
    rows = history()
    rows[-1]["RefForce"] = 15
    with pytest.raises(ValueError, match="normalization"):
        convergence_evidence(rows, settings())


@pytest.mark.parametrize("key,value", [("rms[P]", -3), ("rms[U]", -3), ("CFx", 1.6)])
def test_stable_exit_is_not_numerical_convergence(key, value):
    rows = history()
    rows[-1][key] = value
    assert not convergence_evidence(rows, settings())["eligible"]


def test_short_history_missing_columns_and_duplicate_iterations_fail_closed():
    assert not convergence_evidence(history(4), settings())["eligible"]
    rows = history()
    del rows[0]["rms[W]"]
    with pytest.raises(ValueError, match="missing"):
        convergence_evidence(rows, settings())
    rows = history()
    rows[-1]["Inner_Iter"] = 0
    with pytest.raises(ValueError, match="increasing"):
        convergence_evidence(rows, settings())


def test_su2_csv_whitespace_and_nonfinite_values(tmp_path):
    path = tmp_path / "history.csv"
    path.write_text('"Inner_Iter",     "rms[P]"     ,"CFx"\n0,-8,1\n')
    assert read_history(path) == [{"Inner_Iter": 0, "rms[P]": -8, "CFx": 1}]
    path.write_text('"Inner_Iter","CFx"\n0,nan\n')
    with pytest.raises(ValueError, match="nonfinite"):
        read_history(path)


def test_viscous_configs_do_not_substitute_euler_or_wallfunctions():
    config = solver_config(settings())
    assert "SOLVER= INC_NAVIER_STOKES\n" in config
    assert "INC_NONDIM= DIMENSIONAL\n" in config
    assert "MARKER_HEATFLUX= (body, 0.0)\n" in config
    config = solver_config(settings(regime="sa", mesh={"wall_layers": 12}))
    assert "SOLVER= INC_RANS\n" in config
    assert "KIND_TURB_MODEL= SA\n" in config
    assert "RMS_NU_TILDE" in config
    assert "FORCE_X, FORCE_Y, FORCE_Z" in config
    assert "CONV_CAUCHY_ELEMS= 5\n" in config


def test_native_stop_hints_cannot_bypass_full_residual_window():
    rows = history()
    rows[-2]["rms[U]"] = -3
    # Native instantaneous convergence can pass at the final iteration, while
    # this earlier member of the required window still invalidates the result.
    assert not convergence_evidence(rows, settings())["eligible"]


@pytest.mark.parametrize("key", ["rms[P]", "CFx", "Inner_Iter"])
def test_nonfinite_direct_convergence_input_is_rejected(key):
    rows = history()
    rows[-1][key] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        convergence_evidence(rows, settings())


def test_cancelled_evaluation_has_no_score_or_solver_launch(tmp_path):
    result = evaluate_flow(tmp_path / "missing.step", tmp_path / "run", settings(), lambda: True)
    assert result["status"] == "cancelled"
    assert not result["eligible"]
    assert result["metrics"] == {}
    assert json.loads((tmp_path / "run" / "flow_result.json").read_text())["status"] == "cancelled"


def test_old_artifacts_cannot_be_reused_as_new_solver_output(tmp_path):
    (tmp_path / "history.csv").write_text("old results")
    with pytest.raises(ValueError, match="fresh empty"):
        evaluate_flow(tmp_path / "missing.step", tmp_path, settings(), lambda: False)


def test_explicit_resource_stop_terminates_owned_worker(tmp_path):
    started = time.monotonic()
    with pytest.raises(FlowStoppedError, match="wall-time"):
        _run_owned(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            tmp_path,
            "solver.log",
            settings(resources={"max_seconds": 0.15}),
            lambda: False,
            started,
            None,
        )
    assert time.monotonic() - started < 10


def test_surface_uses_real_mesh_values_and_gates_sa_yplus(tmp_path):
    pv = pytest.importorskip("pyvista")
    np = pytest.importorskip("numpy")
    surface = pv.Sphere(radius=1, theta_resolution=8, phi_resolution=8).cast_to_unstructured_grid()
    n = surface.n_points
    surface["Pressure"] = np.arange(n, dtype=float)
    surface["Velocity"] = np.zeros((n, 3))
    surface["Skin_Friction_Coefficient"] = np.ones((n, 3))
    surface["Y_Plus"] = np.ones(n) * 10
    surface.save(tmp_path / "surface.vtu")
    mesh = {
        "body_bounds_m": [[-1, -1, -1], [1, 1, 1]],
        "body_length_m": 2,
        "mesh_sha256": "meshidentity",
    }
    result = surface_field(tmp_path, settings(regime="sa", mesh={"wall_layers": 12}), mesh)
    assert not result["eligible"]
    field = json.loads((tmp_path / "surface_field.json").read_text())
    assert field["mesh_sha256"] == "meshidentity"
    assert max(field["pressure_pa"]) == n - 1
    assert math.isclose(result["y_plus_max"], 10)


def run_fake_native(tmp_path, monkeypatch, attempts):
    source = tmp_path / "source.step"
    source.write_text("test CAD")
    executable = tmp_path / "solver.exe"
    executable.write_text("test executable")
    monkeypatch.setattr(flow_solver, "solver_executable", lambda: executable)
    starts, configs = [], []

    def owned(command, directory, log_name, profile, cancelled, started, progress):
        starts.append(started)
        if log_name == "mesh.log":
            (directory / "mesh.su2").write_text("test mesh")
            (directory / "mesh_receipt.json").write_text(
                json.dumps(
                    {
                        "source_step_sha256": file_sha256(source),
                        "mesh_sha256": file_sha256(directory / "mesh.su2"),
                    }
                )
            )
            return
        configs.append((directory / "run.cfg").read_text())
        rows = attempts[len(configs) - 1]
        if isinstance(rows, Exception):
            raise rows
        with (directory / "history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (directory / "surface.vtu").write_text(f"field-{len(configs)}")
        (directory / "flow.vtu").write_text(f"volume-{len(configs)}")
        (directory / "solver.log").write_text("native success")

    def field(directory, profile, mesh):
        (directory / "surface_field.json").write_text((directory / "surface.vtu").read_text())
        return {"eligible": True}

    monkeypatch.setattr(flow_solver, "_run_owned", owned)
    monkeypatch.setattr(flow_solver, "surface_field", field)
    result = evaluate_flow(
        source,
        tmp_path / "run",
        settings(
            numerics={
                "iterations": 20,
                "force_window": 5,
            }
        ),
        lambda: False,
    )
    return result, configs, starts


def early_history():
    rows = history()
    rows[-2]["rms[U]"] = -3
    return rows


def test_native_early_stop_replays_with_same_deadline_and_preserved_evidence(tmp_path, monkeypatch):
    result, configs, starts = run_fake_native(tmp_path, monkeypatch, [early_history(), history(11)])
    assert result["eligible"]
    assert len(configs) == 2
    assert configs[1] == configs[0].replace("CONV_STARTITER= 5\n", "CONV_STARTITER= 10\n")
    assert len(set(starts)) == 1
    directory = tmp_path / "run"
    assert (directory / "solver-replay-001-surface.vtu").read_text() == "field-1"
    assert (directory / "surface_field.json").read_text() == "field-2"
    saved = read_history(directory / "solver-replay-001-history.csv")
    assert saved[-2]["rms[U]"] == -3
    assert result["provenance"]["initial_config_sha256"] == file_sha256(
        directory / "solver-replay-001-run.cfg"
    )
    assert result["provenance"]["config_sha256"] == file_sha256(directory / "run.cfg")
    assert result["solver_replays"][0]["next_minimum_iteration"] == 10


@pytest.mark.parametrize("failure", ["residual", "force", "ceiling", "timeout", "process"])
def test_true_failure_or_iteration_ceiling_does_not_replay(tmp_path, monkeypatch, failure):
    rows = early_history()
    if failure == "residual":
        rows[-1]["rms[U]"] = -3
    elif failure == "force":
        rows[-1]["CFx"] = 2
    elif failure == "ceiling":
        rows = history(20)
        rows[-2]["rms[U]"] = -3
    elif failure == "timeout":
        rows = FlowStoppedError("wall-time budget")
    elif failure == "process":
        rows = RuntimeError("native failure")
    result, configs, _ = run_fake_native(tmp_path, monkeypatch, [rows])
    assert not result["eligible"]
    assert len(configs) == 1
    assert result["solver_replays"] == []


def test_replay_timeout_cannot_adopt_previous_surface_or_metrics(tmp_path, monkeypatch):
    result, configs, starts = run_fake_native(
        tmp_path,
        monkeypatch,
        [
            early_history(),
            FlowStoppedError("wall-time budget"),
        ],
    )
    assert not result["eligible"]
    assert len(configs) == 2
    assert len(set(starts)) == 1
    assert result["metrics"] == {}
    assert not (tmp_path / "run" / "surface.vtu").exists()
    assert not (tmp_path / "run" / "surface_field.json").exists()
    assert (tmp_path / "run" / "solver-replay-001-surface.vtu").is_file()


def test_replay_without_iteration_progress_stops(tmp_path, monkeypatch):
    result, configs, _ = run_fake_native(tmp_path, monkeypatch, [early_history(), early_history()])
    assert not result["eligible"]
    assert len(configs) == 2


@pytest.mark.skipif(sys.platform != "win32", reason="Windows delete-sharing regression")
@pytest.mark.parametrize("release_reader", [True, False])
def test_archive_handles_real_windows_reader_without_adopting_stale_output(
    tmp_path, release_reader
):
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    source = tmp_path / "history.csv"
    archived = tmp_path / "solver-replay-001-history.csv"
    source.write_text("previous diagnostic history")
    # GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, OPEN_EXISTING. Deliberately
    # omit FILE_SHARE_DELETE, matching a progress reader that blocks rename.
    handle = kernel.CreateFileW(str(source), 0x80000000, 3, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    timer = None
    try:
        with pytest.raises(PermissionError):
            source.rename(archived)
        if release_reader:
            timer = Timer(0.2, kernel.CloseHandle, args=(handle,))
            timer.start()
            flow_solver._archive_solver_output(
                source, archived, lambda: False, settings(), time.monotonic()
            )
            assert not source.exists()
            assert archived.read_text() == "previous diagnostic history"
        else:
            with pytest.raises(PermissionError):
                flow_solver._archive_solver_output(
                    source, archived, lambda: False, settings(), time.monotonic()
                )
            assert source.is_file()
            assert not archived.exists()
    finally:
        if timer is not None:
            timer.join()
        else:
            kernel.CloseHandle(handle)


def test_archival_honors_original_deadline_without_moving_output(tmp_path):
    source = tmp_path / "history.csv"
    source.write_text("diagnostic")
    with pytest.raises(FlowStoppedError, match="wall-time budget"):
        flow_solver._archive_solver_output(
            source,
            tmp_path / "archive.csv",
            lambda: False,
            settings(resources={"max_seconds": 0.1}),
            time.monotonic() - 1,
        )
    assert source.is_file()
