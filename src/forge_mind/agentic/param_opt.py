"""Parameter fitting (ADR-0073 Stage 3): the optimizer turns the knobs the LLM can't.

The engine's measured failure is PARAMETER error (wrong proportions/offsets on a structurally
recognizable part), and LLMs are demonstrably the wrong tool for continuous spatial numbers. So:
the author proposes the structure with an explicit ``PARAMS`` dict (``params.py``); THIS module
fits the float parameters against a deterministic loss with a derivative-free (1+lambda)
evolution strategy -- the kernel is the simulator, candidates build in parallel sandboxes, and
the inner loop costs ZERO LLM calls.

Loss (scale-invariant on purpose -- the user's stated dimensions stay sovereign, only SHAPE is
fitted): axial-profile distance to the reference (``profile_match``) + the concavity delta
(``shape_signature``). No-Goodhart guards, layered:

  * during SEARCH a candidate that is not a single watertight solid scores ``+inf`` (it can never
    win), and
  * the CALLER (the loop's polish hook) re-runs the FULL validator on the winner and accepts it
    only if the validity tier did not degrade -- final acceptance authority never moves here.

OPT-IN via ``FORGE_PARAM_OPT=1`` (``params.py:param_opt_enabled``); eval budget via
``FORGE_PARAM_OPT_EVALS`` (default 48; ~2-4 min wall clock at 4 parallel builds).
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from forge_mind.agentic.params import (
    default_bounds,
    extract_params,
    override_params,
    pinned_float_names,
)
from forge_mind.agentic.profile_match import AxialProfile, axial_profile, profile_distance
from forge_mind.agentic.sandbox import SandboxLimits, run_build123d_code
from forge_mind.agentic.shape_signature import signature_of
from forge_mind.agentic.validator_loss import ValidatorTarget, validator_loss

_DEFAULT_EVALS = 48

# code, out_dir -> (stl_path, step_path), (None, None) on a failed build. Injectable for tests.
BuildFn = Callable[[str, Path], tuple[Path | None, Path | None]]


@dataclass(frozen=True)
class MassTarget:
    """The ONE deliberately-absolute optimizer objective (the superhuman-arm mass lever).

    The other loss terms are scale-invariant so a user's stated dimensions stay sovereign; this
    one instead drives ABSOLUTE printed mass toward ``target_mass_g``. It is safe against the
    Goodhart "erase material" failure two ways: it is ONE-SIDED (only mass ABOVE target is
    penalized -- there is no reward for eroding below target), and the stated-dim pins (ADR-0080)
    fix the interface so the freed mass can only come from non-stated structural material. Off
    unless a caller passes it; the live pipeline never does."""

    target_mass_g: float
    density_g_cm3: float
    weight: float = 1.0


def max_param_evals() -> int:
    raw = os.environ.get("FORGE_PARAM_OPT_EVALS")
    if raw:
        try:
            return max(int(raw), 2)
        except ValueError:
            pass
    return _DEFAULT_EVALS


def _build_workers() -> int:
    """Parallel candidate builds (engine research idea 8): each sandbox is capped at 2 GB, so
    the worker count must respect the HOST's memory -- production's 4 GB VM sets
    FORGE_PARAM_OPT_WORKERS=2 in fly.toml; dev boxes keep the default 4."""
    raw = os.environ.get("FORGE_PARAM_OPT_WORKERS")
    if raw:
        try:
            return max(int(raw), 1)
        except ValueError:
            pass
    return 4


_POOL: Any = None


def _warm_pool() -> Any:
    """Lazy module singleton, sized to the build workers; reused across polishes so the kernel
    import is paid once per process lifetime, not once per part."""
    global _POOL
    if _POOL is None:
        from forge_mind.agentic.warm_pool import WarmPool

        _POOL = WarmPool(_build_workers())
    return _POOL


# ---- the deterministic loss --------------------------------------------------------------------


@dataclass(frozen=True)
class FitTarget:
    """The reference, reduced once to the two signals the loss fits: profile + concavity."""

    profile: AxialProfile
    concavity: float | None


def fit_target(reference_stl: Path) -> FitTarget | None:
    """Compute the reference's target signals ONCE (reused across every candidate eval).

    GUARD (ADR-0075, from the measured turbo-wheel incident): a DEGENERATE reference -- near-zero
    extents (the caught case rendered as 0 x 0 x 0 mm) -- must never become a fit target; the
    optimizer would faithfully drag a good build toward garbage. Fail-closed to None (the
    validator-loss terms still fit; only the reference profile is dropped)."""
    try:
        mesh: Any = trimesh.load(str(reference_stl), force="mesh")
        extents = np.asarray(mesh.bounding_box.extents, dtype=np.float64)
        if not np.all(np.isfinite(extents)) or float(extents.max()) < 1.0:
            return None
    except Exception:
        return None
    profile = axial_profile(reference_stl)
    if profile is None:
        return None
    sig = signature_of(reference_stl)
    return FitTarget(profile=profile, concavity=(sig.concavity if sig else None))


def _mass_term(mesh: Any, mt: MassTarget) -> float:
    """One-sided mass hinge, normalized by the target: ``weight * max(0, (mass-target)/target)``.
    Zero at/below target (no erode-past-target incentive). Mass = trimesh volume x density, the
    same formula as ``materials.mass_grams``. A non-positive volume (inverted/degenerate winding)
    is REJECTED (+inf), never scored as a free "zero mass" -- mirrors score_arm's positive-volume
    guard so a degenerate part can never win the mass race."""
    if mt.target_mass_g <= 0.0:
        return 0.0
    volume_mm3 = float(mesh.volume)
    if volume_mm3 <= 0.0:
        return float("inf")
    mass_g = (volume_mm3 / 1000.0) * mt.density_g_cm3
    return mt.weight * max(0.0, (mass_g - mt.target_mass_g) / mt.target_mass_g)


def fit_loss(
    stl_path: Path,
    target: FitTarget | None,
    vtarget: ValidatorTarget | None = None,
    *,
    mass_target: MassTarget | None = None,
) -> float:
    """Form loss of one built candidate: reference terms (profile + concavity, when a reference
    exists) + validator-derived terms (wall/envelope/balance, ADR-0074 -- no reference needed) +
    an optional absolute ``mass_target`` hinge (the superhuman-arm lever). ``+inf`` for anything
    that is not a single watertight solid (the search-time no-Goodhart guard). At least one
    target must be given."""
    try:
        mesh: Any = trimesh.load(str(stl_path), force="mesh")
        if not bool(mesh.is_watertight) or int(mesh.body_count) != 1:
            return float("inf")
    except Exception:
        return float("inf")
    loss = 0.0
    if mass_target is not None:
        loss += _mass_term(mesh, mass_target)
    if target is not None:
        profile = axial_profile(stl_path)
        if profile is None:
            return float("inf")
        d = profile_distance(profile, target.profile)
        if d is None:
            return float("inf")
        loss += d
        if target.concavity is not None:
            sig = signature_of(stl_path)
            if sig is not None and sig.concavity is not None:
                loss += abs(sig.concavity - target.concavity)
    if vtarget is not None:
        loss += validator_loss(stl_path, vtarget)
    return loss


# ---- the search strategy (pure -- unit-tested on analytic functions) ----------------------------


def minimize_es(
    x0: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    batch_loss: Callable[[list[np.ndarray]], list[float]],
    *,
    lam: int = 6,
    max_evals: int = _DEFAULT_EVALS,
    seed: int = 0,
    sigma0: float = 0.25,
    shrink: float = 0.6,
    sigma_min: float = 0.02,
) -> tuple[np.ndarray, float, int]:
    """(1+lambda) evolution strategy in the normalized unit box: perturb the incumbent, keep the
    best, shrink the step on a stagnant generation. Deterministic (seeded), derivative-free,
    bounded. Returns ``(best_x, best_loss, n_evals)``; the first eval is x0 itself."""
    lo = np.asarray([b[0] for b in bounds], dtype=np.float64)
    hi = np.asarray([b[1] for b in bounds], dtype=np.float64)
    span = np.where(hi > lo, hi - lo, 1.0)

    def to_x(z: np.ndarray) -> np.ndarray:
        return np.asarray(lo + z * span, dtype=np.float64)

    z_best = (np.clip(np.asarray(x0, dtype=np.float64), lo, hi) - lo) / span
    best_loss = float(batch_loss([to_x(z_best)])[0])
    evals = 1
    rng = np.random.default_rng(seed)
    sigma = sigma0
    while evals < max_evals and sigma >= sigma_min:
        k = min(lam, max_evals - evals)
        cands = [
            np.clip(z_best + rng.normal(0.0, sigma, size=z_best.size), 0.0, 1.0) for _ in range(k)
        ]
        losses = batch_loss([to_x(c) for c in cands])
        evals += k
        i = int(np.argmin(losses))
        if float(losses[i]) < best_loss:
            best_loss = float(losses[i])
            z_best = cands[i]
            sigma = min(sigma * 1.3, sigma0)  # 1/5-rule flavor: grow on success, never past sigma0
        else:
            sigma *= shrink
    return to_x(z_best), best_loss, evals


# ---- the glue: optimize one authored script against one reference -------------------------------


@dataclass
class PolishResult:
    """Outcome of one parameter-fit run (LLM cost: zero)."""

    improved: bool
    baseline_loss: float
    best_loss: float
    evals: int
    best_code: str
    best_stl: Path | None
    best_step: Path | None
    # ADR-0080: stated-pinned params, name -> (authored, winner) -- itemized evidence that a
    # protected dimension stayed inside its pin, and the drift log for deferred v2 channels.
    pinned: dict[str, tuple[float, float]] = field(default_factory=dict)


def polish_params(
    code: str,
    reference_stl: Path | None,
    work_dir: Path,
    *,
    vtarget: ValidatorTarget | None = None,
    stated: frozenset[float] | None = None,
    part_name: str = "part",
    sandbox_limits: SandboxLimits | None = None,
    max_evals: int | None = None,
    build_fn: BuildFn | None = None,
    mass_target: MassTarget | None = None,
) -> PolishResult | None:
    """Fit the script's float PARAMS against the reference and/or the validator-derived targets
    (ADR-0074), and optionally an absolute ``mass_target`` (the superhuman-arm mass lever). None
    when not applicable (no PARAMS dict / no float params / no usable target at all) -- fail-soft,
    the caller keeps its candidate. ``stated`` (ADR-0080) pins value-matched params to +/-2% so
    stated dimensions are never the currency the fit spends."""
    spec = extract_params(code)
    if spec is None or not spec.float_names:
        return None
    target = fit_target(reference_stl) if reference_stl is not None else None
    if vtarget is not None and not vtarget.applicable:
        vtarget = None
    if target is None and vtarget is None and mass_target is None:
        return None  # nothing measurable to fit against

    limits = sandbox_limits or SandboxLimits(timeout_s=240)

    # Warm workers (engine research idea 1, default ON): the fit re-runs the SAME script that
    # already survived the one-shot sandbox, so per-eval cold isolation only buys the measured
    # 7.3 s import tax back. Fail-closed to the SLOW path: any warm harness failure retries the
    # candidate in the cold sandbox, so the pool can cost time, never correctness.
    pool = _warm_pool() if os.environ.get("FORGE_PARAM_OPT_WARM", "1") != "0" else None

    def default_build(c: str, out: Path) -> tuple[Path | None, Path | None]:
        if pool is not None:
            res = pool.run(c, out, part_name=part_name, limits=limits)
            if res.error_type == "harness":
                res = run_build123d_code(c, out, part_name=part_name, limits=limits)
        else:
            res = run_build123d_code(c, out, part_name=part_name, limits=limits)
        if not res.ok:
            return None, None
        return res.stl_path, res.step_path

    build = build_fn or default_build
    names = spec.float_names
    all_bounds = default_bounds(spec, stated=stated)
    bounds = [all_bounds[n] for n in names]
    x0 = [float(spec.values[n]) for n in names]

    state: dict[str, Any] = {
        "n": 0,
        "baseline": None,
        "best_loss": float("inf"),
        "best_code": code,
        "best_stl": None,
        "best_step": None,
    }

    def batch_loss(xs: list[np.ndarray]) -> list[float]:
        jobs: list[tuple[str, Path]] = []
        for x in xs:
            theta = {n: float(v) for n, v in zip(names, x, strict=True)}
            state["n"] += 1
            jobs.append((override_params(code, spec, theta), work_dir / f"eval_{state['n']:03d}"))
        with ThreadPoolExecutor(max_workers=min(len(jobs), _build_workers())) as ex:
            paths = list(ex.map(lambda j: build(j[0], j[1]), jobs))
        losses: list[float] = []
        for (cand_code, _), (stl, step) in zip(jobs, paths, strict=True):
            loss = (
                fit_loss(stl, target, vtarget, mass_target=mass_target)
                if stl is not None
                else float("inf")
            )
            losses.append(loss)
            if state["baseline"] is None:
                state["baseline"] = loss  # the very first eval IS x0 (minimize_es contract)
            if loss < state["best_loss"]:
                state.update(best_loss=loss, best_code=cand_code, best_stl=stl, best_step=step)
        return losses

    _, _es_loss, evals = minimize_es(
        x0, bounds, batch_loss, max_evals=max_evals or max_param_evals()
    )
    baseline = float(state["baseline"]) if state["baseline"] is not None else float("inf")
    improved = (
        state["best_stl"] is not None
        and np.isfinite(state["best_loss"])
        and state["best_loss"] < baseline - 1e-6
    )
    pinned: dict[str, tuple[float, float]] = {}
    best_spec = extract_params(str(state["best_code"]))
    for n in pinned_float_names(spec, stated):
        authored = float(spec.values[n])
        winner = authored
        if best_spec is not None and isinstance(best_spec.values.get(n), (int, float)):
            winner = float(best_spec.values[n])
        pinned[n] = (authored, winner)
    return PolishResult(
        improved=bool(improved),
        baseline_loss=baseline,
        best_loss=float(state["best_loss"]),
        evals=evals,
        best_code=str(state["best_code"]),
        best_stl=state["best_stl"],
        best_step=state["best_step"],
        pinned=pinned,
    )


__all__ = [
    "BuildFn",
    "FitTarget",
    "PolishResult",
    "fit_loss",
    "fit_target",
    "max_param_evals",
    "minimize_es",
    "polish_params",
]
