"""Controlled SU2 execution and fail-closed numerical evidence for CAD flow studies."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forge_mind.validation_provenance import file_sha256
from forge_mind.workspace.flow_models import FLOW_SCOPE, FlowSettings


def solver_executable() -> Path:
    """The operator controls installation; study requests never contain executable paths."""
    configured = os.environ.get("CADKITTY_SU2_EXE")
    found = configured or shutil.which("SU2_CFD") or shutil.which("SU2_CFD.exe")
    if not found or not Path(found).is_file():
        raise ValueError(
            "SU2_CFD is unavailable; install SU2 and set operator CADKITTY_SU2_EXE or PATH"
        )
    return Path(found).resolve()


def solver_config(settings: FlowSettings) -> str:
    s, n = settings, settings.numerics
    fields = ["RMS_PRESSURE", "RMS_VELOCITY-X", "RMS_VELOCITY-Y", "RMS_VELOCITY-Z"]
    if s.regime == "sa":
        fields.append("RMS_NU_TILDE")
    # SU2 ANDs residual and coefficient criteria. Monitor each force component
    # so arbitrary freestream directions cannot hide transverse force drift.
    fields.extend(("FORCE_X", "FORCE_Y", "FORCE_Z"))
    values: dict[str, Any] = {
        "SOLVER": "INC_RANS" if s.regime == "sa" else "INC_NAVIER_STOKES",
        "KIND_TURB_MODEL": "SA" if s.regime == "sa" else "NONE",
        "INC_DENSITY_MODEL": "CONSTANT",
        "INC_DENSITY_INIT": s.density_kg_m3,
        "INC_VELOCITY_INIT": f"({', '.join(str(v) for v in s.velocity_m_s)})",
        "INC_NONDIM": "DIMENSIONAL",
        "INC_ENERGY_EQUATION": "NO",
        "VISCOSITY_MODEL": "CONSTANT_VISCOSITY",
        "MU_CONSTANT": s.dynamic_viscosity_pa_s,
        "REF_AREA": s.reference_area_m2,
        "REF_LENGTH": s.reference_length_m,
        "MARKER_HEATFLUX": "(body, 0.0)",
        "MARKER_FAR": "(farfield)",
        "MARKER_PLOTTING": "(body)",
        "MARKER_MONITORING": "(body)",
        "NUM_METHOD_GRAD": "WEIGHTED_LEAST_SQUARES",
        "CONV_NUM_METHOD_FLOW": "FDS",
        "MUSCL_FLOW": "YES",
        "SLOPE_LIMITER_FLOW": "VENKATAKRISHNAN" if s.regime == "sa" else "NONE",
        "TIME_DISCRE_FLOW": "EULER_IMPLICIT",
        "CFL_NUMBER": n.cfl,
        "BETA_FACTOR": n.beta_factor,
        "CFL_ADAPT": "NO",
        "LINEAR_SOLVER": "FGMRES",
        "LINEAR_SOLVER_PREC": "ILU",
        "LINEAR_SOLVER_ERROR": 1e-12,
        "LINEAR_SOLVER_ITER": n.linear_solver_iterations,
        "ITER": n.iterations,
        # Native early completion writes final fields. These conservative stop
        # hints do not replace the independent full-window eligibility gates.
        "CONV_FIELD": f"({', '.join(fields)})",
        "CONV_RESIDUAL_MINVAL": n.residual_log10_max - 0.5,
        "CONV_CAUCHY_ELEMS": n.force_window,
        "CONV_CAUCHY_EPS": n.force_relative_tolerance / (10 * n.force_window),
        "CONV_STARTITER": n.force_window,
        "MESH_FILENAME": "mesh.su2",
        "MESH_FORMAT": "SU2",
        "TABULAR_FORMAT": "CSV",
        "CONV_FILENAME": "history",
        "VOLUME_FILENAME": "flow",
        "SURFACE_FILENAME": "surface",
        "OUTPUT_FILES": "(PARAVIEW, SURFACE_PARAVIEW)",
        "OUTPUT_WRT_FREQ": n.iterations,
        "HISTORY_OUTPUT": "(ITER, RMS_RES, AERO_COEFF)",
        "SCREEN_OUTPUT": "(INNER_ITER, RMS_PRESSURE, RMS_VELOCITY-X, DRAG)",
        "SCREEN_WRT_FREQ_INNER": 100,
    }
    if s.regime == "sa":
        values.update(
            SA_OPTIONS="NONE",
            CONV_NUM_METHOD_TURB="SCALAR_UPWIND",
            MUSCL_TURB="NO",
            FREESTREAM_NU_FACTOR=s.sa_freestream_nu_factor,
        )
    return "\n".join(f"{key}= {value}" for key, value in values.items()) + "\n"


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = []
        for row in csv.DictReader(stream, skipinitialspace=True):
            values = {str(k).strip().strip('"'): float(str(v).strip()) for k, v in row.items()}
            if not values or any(not math.isfinite(v) for v in values.values()):
                raise ValueError("Solver history contains missing or nonfinite values")
            rows.append(values)
    if not rows:
        raise ValueError("Solver produced no convergence history")
    return rows


def convergence_evidence(rows: list[dict[str, float]], settings: FlowSettings) -> dict[str, Any]:
    n = settings.numerics
    residuals = ["rms[P]", "rms[U]", "rms[V]", "rms[W]"]
    if settings.regime == "sa":
        residuals.append("rms[nu]")
    required = {*residuals, "CFx", "CFy", "CFz", "RefForce", "Inner_Iter"}
    if not rows or any(required - row.keys() for row in rows):
        raise ValueError("Solver history is missing required residual, force or iteration columns")
    if any(not math.isfinite(row[k]) for row in rows for k in required):
        raise ValueError("Solver history contains nonfinite convergence values")
    if any(
        row["Inner_Iter"] < 0 or not float(row["Inner_Iter"]).is_integer() for row in rows
    ) or any(rows[i]["Inner_Iter"] + 1 != rows[i + 1]["Inner_Iter"] for i in range(len(rows) - 1)):
        raise ValueError("Solver history iterations are not consecutively increasing integers")
    speed = math.hypot(*settings.velocity_m_s)
    unit = [v / speed for v in settings.velocity_m_s]
    expected_force = 0.5 * settings.density_kg_m3 * speed**2 * settings.reference_area_m2
    if any(not math.isclose(r["RefForce"], expected_force, rel_tol=1e-7) for r in rows):
        raise ValueError("Solver force normalization disagrees with frozen density, speed and area")
    coefficients = [
        sum(r[k] * u for k, u in zip(("CFx", "CFy", "CFz"), unit, strict=True)) for r in rows
    ]
    tail = coefficients[-n.force_window :]
    mean = sum(tail) / len(tail)
    spread = (max(tail) - min(tail)) / max(abs(mean), 1e-12)
    worst = {k: max(r[k] for r in rows[-n.force_window :]) for k in residuals}
    residual_ok = all(v <= n.residual_log10_max for v in worst.values())
    force_ok = len(tail) == n.force_window and spread <= n.force_relative_tolerance and mean > 0
    return {
        "eligible": bool(residual_ok and force_ok),
        "residual_gate": residual_ok,
        "force_gate": force_ok,
        "rows": len(rows),
        "last_iteration": rows[-1]["Inner_Iter"],
        "force_window": n.force_window,
        "force_relative_spread": spread,
        "force_relative_tolerance": n.force_relative_tolerance,
        "worst_window_residuals_log10": worst,
        "residual_log10_max": n.residual_log10_max,
        "drag_coefficient": coefficients[-1],
        "drag_n": coefficients[-1] * expected_force,
        "force_normalization_n": expected_force,
    }


def surface_field(directory: Path, settings: FlowSettings, mesh: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import pyvista as pv

    path = directory / "surface.vtu"
    surface = pv.read(path).extract_surface(algorithm="dataset_surface").triangulate()
    points = np.asarray(surface.points, dtype=float)
    faces = np.asarray(surface.faces).reshape(-1, 4)
    if len(points) < 4 or not np.all(faces[:, 0] == 3):
        raise ValueError("Solver body field has no complete triangular boundary")
    required = ("Pressure", "Velocity", "Skin_Friction_Coefficient", "Y_Plus")
    if any(k not in surface.point_data for k in required):
        raise ValueError("Solver body field is missing pressure, velocity, wall shear or y-plus")
    arrays = {k: np.asarray(surface.point_data[k], dtype=float) for k in required}
    for key, shape in {
        "Pressure": (len(points),),
        "Y_Plus": (len(points),),
        "Velocity": (len(points), 3),
        "Skin_Friction_Coefficient": (len(points), 3),
    }.items():
        if arrays[key].shape != shape:
            raise ValueError("Solver surface scalar/vector association is invalid")
    if any(len(a) != len(points) or not np.isfinite(a).all() for a in [points, *arrays.values()]):
        raise ValueError("Solver surface field has nonfinite or unbound values")
    bounds = np.asarray(mesh["body_bounds_m"])
    tolerance = max(2e-6, mesh["body_length_m"] * 1e-5)
    if np.any(points < bounds[0] - tolerance) or np.any(points > bounds[1] + tolerance):
        raise ValueError("Solver field coordinates are not bound to selected CAD surface")
    yplus = arrays["Y_Plus"]
    wall_gate = bool(
        np.all(yplus >= 0) and (settings.regime == "laminar" or 0 < float(yplus.max()) <= 1.0)
    )
    field = {
        "schema_version": 1,
        "frame": "cad_xyz_m",
        "association": "solver_boundary_nodes",
        "positions_m": points.tolist(),
        "triangles": faces[:, 1:].tolist(),
        "pressure_pa": arrays["Pressure"].tolist(),
        "velocity_m_s": arrays["Velocity"].tolist(),
        "skin_friction_coefficient": arrays["Skin_Friction_Coefficient"].tolist(),
        "y_plus": yplus.tolist(),
        "mesh_sha256": mesh["mesh_sha256"],
        "solver_surface_sha256": file_sha256(path),
        "note": "Velocity at the no-slip body wall is zero; fluid velocity is in the full flow.vtu volume artifact.",
    }
    (directory / "surface_field.json").write_text(
        json.dumps(field, allow_nan=False), encoding="utf-8"
    )
    return {
        "eligible": wall_gate,
        "policy": settings.wall_policy,
        "y_plus_max": float(yplus.max()),
        "y_plus_p95": float(np.quantile(yplus, 0.95)),
        "points": len(points),
        "triangles": len(faces),
    }


class FlowStoppedError(RuntimeError):
    pass


def _archive_solver_output(
    source: Path,
    destination: Path,
    cancelled: Callable[[], bool],
    settings: FlowSettings,
    started: float,
) -> None:
    """Brief Windows progress readers can hold a file without delete sharing."""
    retry_until = time.monotonic() + 1.0
    while True:
        if cancelled():
            raise FlowStoppedError("Flow evaluation cancelled during solver archival")
        if (
            settings.resources.max_seconds is not None
            and time.monotonic() - started >= settings.resources.max_seconds
        ):
            raise FlowStoppedError("Flow wall-time budget exhausted during solver archival")
        try:
            source.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or time.monotonic() >= retry_until:
                raise
            time.sleep(0.07)


def _kill_tree(process: subprocess.Popen[bytes]) -> None:
    import psutil

    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in reversed(children):
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
        parent.kill()
        psutil.wait_procs([*children, parent], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=5)


def _run_owned(
    command: list[str],
    directory: Path,
    log: str,
    settings: FlowSettings,
    cancelled: Callable[[], bool],
    started: float,
    progress: Callable[[dict[str, Any]], None] | None,
) -> None:
    import psutil

    with (directory / log).open("wb") as stream:
        options: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NO_WINDOW}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        env = {**os.environ, "OMP_NUM_THREADS": str(settings.resources.threads)}
        with subprocess.Popen(
            command, cwd=directory, stdout=stream, stderr=subprocess.STDOUT, env=env, **options
        ) as process:
            last_progress = 0.0
            try:
                while process.poll() is None:
                    now = time.monotonic()
                    if cancelled():
                        raise FlowStoppedError("Flow evaluation cancelled")
                    if (
                        settings.resources.max_seconds is not None
                        and now - started > settings.resources.max_seconds
                    ):
                        raise FlowStoppedError(
                            "Flow evaluation exhausted the explicit wall-time resource policy"
                        )
                    if settings.resources.max_memory_mb is not None:
                        try:
                            parent = psutil.Process(process.pid)
                            memory = sum(
                                p.memory_info().rss
                                for p in [parent, *parent.children(recursive=True)]
                            )
                            if memory > settings.resources.max_memory_mb * 1024**2:
                                raise FlowStoppedError(
                                    "Flow evaluation exhausted the explicit memory resource policy"
                                )
                        except psutil.NoSuchProcess:
                            pass
                    if progress and now - last_progress >= 2:
                        progress(
                            {
                                "stage": "meshing" if log == "mesh.log" else "solving",
                                "elapsed_seconds": now - started,
                            }
                        )
                        last_progress = now
                    time.sleep(0.1)
            except BaseException:
                _kill_tree(process)
                raise
            if process.returncode:
                tail = (directory / log).read_text(encoding="utf-8", errors="replace")[-3000:]
                raise ValueError(f"Flow worker exited {process.returncode}: {tail}")


def evaluate_flow(
    step_path: Path,
    directory: Path,
    settings: FlowSettings,
    cancelled: Callable[[], bool],
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Evaluate in a fresh dedicated directory; incomplete calculations are never eligible."""
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Flow evaluation needs a fresh empty evidence directory")
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "unverified",
        "eligible": False,
        "metrics": {},
        "settings": settings.model_dump(mode="json"),
        "scope": FLOW_SCOPE,
        "errors": [],
        "provenance": {},
        "artifacts": {},
        "timings_seconds": {},
    }
    try:
        if cancelled():
            raise FlowStoppedError("Flow evaluation cancelled")
        executable = solver_executable()
        source_hash = file_sha256(step_path)
        shutil.copyfile(step_path, directory / "source.step")
        if file_sha256(directory / "source.step") != source_hash:
            raise ValueError("Source STEP changed during evidence capture")
        (directory / "settings.json").write_text(settings.model_dump_json(), encoding="utf-8")
        (directory / "run.cfg").write_text(solver_config(settings), encoding="utf-8")
        result["provenance"] = {
            "source_step_sha256": source_hash,
            "solver_sha256": file_sha256(executable),
            "solver_binary": executable.name,
            "config_sha256": file_sha256(directory / "run.cfg"),
            "settings_sha256": file_sha256(directory / "settings.json"),
        }
        result["provenance"]["initial_config_sha256"] = result["provenance"]["config_sha256"]
        result["solver_replays"] = []
        result["replay_policy"] = (
            "cold replay; delay native checks by one force window; fixed iteration ceiling v1"
        )
        stage_started = time.monotonic()
        _run_owned(
            [sys.executable, "-m", "forge_mind.workspace.flow_meshing", str(directory)],
            directory,
            "mesh.log",
            settings,
            cancelled,
            started,
            progress,
        )
        result["timings_seconds"]["meshing"] = time.monotonic() - stage_started
        mesh = json.loads((directory / "mesh_receipt.json").read_text(encoding="utf-8"))
        if mesh["source_step_sha256"] != source_hash or mesh["mesh_sha256"] != file_sha256(
            directory / "mesh.su2"
        ):
            raise ValueError("Fluid mesh receipt does not match source STEP and solver mesh")
        result["mesh"] = mesh
        result["provenance"]["mesh_sha256"] = mesh["mesh_sha256"]
        stage_started = time.monotonic()
        protected = {p.name for p in directory.iterdir()}
        minimum_iteration = settings.numerics.force_window
        previous_iteration = -1
        while True:
            _run_owned(
                [str(executable), "run.cfg", "-t", str(settings.resources.threads)],
                directory,
                "solver.log",
                settings,
                cancelled,
                started,
                progress,
            )
            if (
                file_sha256(directory / "mesh.su2") != mesh["mesh_sha256"]
                or file_sha256(directory / "run.cfg") != result["provenance"]["config_sha256"]
            ):
                raise ValueError("Solver inputs changed during execution")
            rows = read_history(directory / "history.csv")
            convergence = convergence_evidence(rows, settings)
            last_iteration = int(convergence["last_iteration"])
            # Only repair an early native exit whose instantaneous residuals
            # passed but whose full residual window did not. Never replay a
            # genuinely unconverged final state, force failure or exhausted run.
            if (
                convergence["residual_gate"]
                or not convergence["force_gate"]
                or last_iteration >= settings.numerics.iterations - 1
                or last_iteration <= previous_iteration
                or any(
                    rows[-1][key] > settings.numerics.residual_log10_max
                    for key in convergence["worst_window_residuals_log10"]
                )
            ):
                break
            next_minimum = min(
                settings.numerics.iterations - 1,
                last_iteration + settings.numerics.force_window,
            )
            if next_minimum <= minimum_iteration:
                break
            number = len(result["solver_replays"]) + 1
            saved = {}
            for path in list(directory.iterdir()):
                if (
                    path.is_file()
                    and not path.name.startswith("solver-replay-")
                    and (path.name == "run.cfg" or path.name not in protected)
                ):
                    archived = directory / f"solver-replay-{number:03d}-{path.name}"
                    # Move old output away: a failed replay cannot adopt its fields.
                    if path.name == "run.cfg":
                        shutil.copyfile(path, archived)
                    else:
                        _archive_solver_output(path, archived, cancelled, settings, started)
                    saved[archived.name] = file_sha256(archived)
            result["solver_replays"].append(
                {
                    "convergence": convergence,
                    "artifacts": saved,
                    "next_minimum_iteration": next_minimum,
                }
            )
            config = (directory / "run.cfg").read_text(encoding="utf-8")
            config = config.replace(
                f"CONV_STARTITER= {minimum_iteration}\n", f"CONV_STARTITER= {next_minimum}\n"
            )
            (directory / "run.cfg").write_text(config, encoding="utf-8")
            result["provenance"]["config_sha256"] = file_sha256(directory / "run.cfg")
            previous_iteration, minimum_iteration = last_iteration, next_minimum
            if progress is not None:
                progress({"phase": "solver replay", "minimum_iteration": minimum_iteration})
        result["timings_seconds"]["solving"] = time.monotonic() - stage_started
        stage_started = time.monotonic()
        if cancelled():
            raise FlowStoppedError("Flow evaluation cancelled")
        if (
            file_sha256(directory / "mesh.su2") != mesh["mesh_sha256"]
            or file_sha256(directory / "run.cfg") != result["provenance"]["config_sha256"]
        ):
            raise ValueError("Solver inputs changed during execution")
        wall = surface_field(directory, settings, mesh)
        result["timings_seconds"]["postprocessing"] = time.monotonic() - stage_started
        result.update(convergence=convergence, wall=wall)
        result["metrics"] = {key: convergence[key] for key in ("drag_n", "drag_coefficient")}
        result["eligible"] = bool(convergence["eligible"] and wall["eligible"])
        result["status"] = "complete" if result["eligible"] else "unverified"
        if not convergence["eligible"]:
            result["errors"].append(
                "Residual and drag-history convergence gates were not both satisfied"
            )
        if not wall["eligible"]:
            result["errors"].append(
                "Resolved no-slip turbulence policy requires finite nonnegative y-plus <= 1 everywhere"
            )
    except FlowStoppedError as exc:
        result["status"] = "cancelled" if cancelled() else "unverified"
        result["errors"].append(str(exc))
    except Exception as exc:
        result["status"] = "failed"
        result["errors"].append(f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = time.monotonic() - started
    result["artifacts"] = {
        p.name: {"sha256": file_sha256(p), "bytes": p.stat().st_size}
        for p in sorted(directory.iterdir())
        if p.is_file()
    }
    (directory / "flow_result.json").write_text(
        json.dumps(result, allow_nan=False), encoding="utf-8"
    )
    return result
