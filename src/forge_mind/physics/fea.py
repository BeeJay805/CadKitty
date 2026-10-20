"""ILLUSTRATIVE linear-elastic stress (ADR-0058) — a RELATIVE stress heatmap, NEVER a gate, never "safe".

For a 3D-printed part you can MEASURE where stress concentrates but you cannot truthfully CERTIFY
strength — layer adhesion is too orientation-dependent (ADR-0039). So this is reported-only: a
linear-elastic FEM solve under a CANONICAL generic load, surfaced as a relative heatmap with a
mandatory disclaimer. It NO-RESULTs (returns ``None``) — never a misleading number — when the geometry
can't be tetrahedralized, the solve is unconstrained/non-finite, or the optional FEA deps are absent.

Stack: pytetwild (tetrahedralize the surface, robust to messy foreign geometry) + scikit-fem (linear
elasticity). Both are LAZY-imported so importing this module never requires the heavy deps; their
absence degrades to NO-RESULT, not a crash (FEA is a reported feature, not a gate).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as _queue
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# A canonical material (PLA-ish). The field is RELATIVE — the modulus only sets a scale the heatmap
# normalizes away — so the exact value is not a claim.
DEFAULT_E_MPA = 3500.0
DEFAULT_NU = 0.35

DISCLAIMER = (
    "Relative stress under a generic load (base fixed, pushed from the top). It shows WHERE stress "
    "concentrates, not real-world values, and is NOT a safety rating — a printed part's real strength "
    "depends on print orientation, layer adhesion, and infill."
)

# pytetwild (fTetWild) writes scratch files to the CWD, and a linear solve is CPU-heavy; serialize FEA
# per process so concurrent uploads can't race on those files or thrash the CPU.
_FEA_LOCK = threading.Lock()


@dataclass(frozen=True)
class StressField:
    """Per-vertex relative von Mises (ADR-0058). ``von_mises`` is aligned to the INPUT mesh's vertex
    order so a caller can colour the GLB by vertex index. REPORTED-only; never a verdict."""

    von_mises: tuple[float, ...]  # per input-mesh vertex, under the canonical unit load
    von_mises_max: float
    von_mises_p95: (
        float  # 95th percentile — a robust scale for the heatmap (dodges the clamp singularity)
    )
    max_location_mm: tuple[float, float, float]
    load_case: str
    material_e_mpa: float
    note: str


def solve_elasticity(
    mesh_tet: Any,
    e_mpa: float,
    nu: float,
    fixed_pred: Callable[[Any], Any],
    load_facet_pred: Callable[[Any], Any],
    load_dir: tuple[float, float, float],
    *,
    element: str = "P1",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Linear-elastic solve on a scikit-fem ``MeshTet``. Returns ``(nodal_displacement (n_nodes, 3) in
    mm, von_mises_per_node (n_nodes,) in the load's stress units)``, or ``None`` if the system is
    unconstrained / the solve is non-finite. The load is a TRACTION (force/area) on the load facets, so
    with E in MPa and the traction in MPa the von Mises and displacement come out in MPa / mm. Pure
    scikit-fem (no mesher) so it is unit-testable against analytic cases (the FEA validation harness).

    ``element`` selects the displacement order: "P1" (linear tets — fast, but shear-locks in bending; the
    illustrative heatmap uses this) or "P2" (quadratic tets — no locking, validated decision-grade for
    bending; the load-case verdict uses this). Displacement is returned at the mesh's CORNER nodes either
    way (``basis.nodal_dofs`` covers the corner vertices; P2's edge dofs are internal to the solve)."""
    from skfem import (
        Basis,
        ElementTetP1,
        ElementTetP2,
        ElementVector,
        FacetBasis,
        LinearForm,
        condense,
        solve,
    )
    from skfem.models.elasticity import lame_parameters, linear_elasticity

    e = ElementVector(ElementTetP2() if element == "P2" else ElementTetP1())
    basis = Basis(mesh_tet, e)
    lam, mu = lame_parameters(e_mpa, nu)
    k = linear_elasticity(lam, mu).assemble(basis)

    fixed = basis.get_dofs(facets=_boundary_facets(mesh_tet, fixed_pred))
    facets = _boundary_facets(mesh_tet, load_facet_pred)
    if len(fixed) == 0 or len(facets) == 0:
        return None
    fb = FacetBasis(mesh_tet, e, facets=facets)
    ld = np.asarray(load_dir, dtype=float)

    @LinearForm  # type: ignore[untyped-decorator]
    def load(v: Any, w: Any) -> Any:
        return ld[0] * v[0] + ld[1] * v[1] + ld[2] * v[2]

    rhs = load.assemble(fb)
    u = solve(*condense(k, rhs, D=fixed))
    if not np.all(np.isfinite(u)):
        return None

    # Per-element stress: P1 displacement → constant strain per tet. grad is (3, 3, nelems, nqp).
    grad = np.asarray(basis.interpolate(u).grad).mean(axis=-1)  # (3, 3, nelems)
    eps = 0.5 * (grad + grad.transpose(1, 0, 2))
    tr = eps[0, 0] + eps[1, 1] + eps[2, 2]
    sig = 2.0 * mu * eps
    for i in range(3):
        sig[i, i] = sig[i, i] + lam * tr
    sxx, syy, szz = sig[0, 0], sig[1, 1], sig[2, 2]
    sxy, syz, szx = sig[0, 1], sig[1, 2], sig[2, 0]
    vm_e = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy**2 + syz**2 + szx**2)
    )
    if not np.all(np.isfinite(vm_e)):
        return None

    tets = np.asarray(mesh_tet.t)  # (4, nelems)
    n_node = int(mesh_tet.p.shape[1])
    vm_node = np.zeros(n_node)
    count = np.zeros(n_node)
    for c in range(4):
        np.add.at(vm_node, tets[c], vm_e)
        np.add.at(count, tets[c], 1)
    vm_node = vm_node / np.maximum(count, 1.0)

    # Nodal displacement (n_nodes, 3): basis.nodal_dofs is (3, n_nodes) for the P1 vector element.
    disp = np.asarray(u)[basis.nodal_dofs].T
    return disp, vm_node


def _solve_von_mises(
    mesh_tet: Any,
    e_mpa: float,
    nu: float,
    fixed_pred: Callable[[Any], Any],
    load_facet_pred: Callable[[Any], Any],
    load_dir: tuple[float, float, float],
) -> np.ndarray | None:
    """Per-tet-node von Mises only (the relative-heatmap path). Thin wrapper over ``solve_elasticity``."""
    res = solve_elasticity(mesh_tet, e_mpa, nu, fixed_pred, load_facet_pred, load_dir)
    return None if res is None else res[1]


def _boundary_facets(mesh_tet: Any, pred: Callable[[Any], Any]) -> Any:
    """Require all three vertices on the declared surface, and exclude interior facets."""
    indices = np.asarray(mesh_tet.boundary_facets())
    triangles = np.asarray(mesh_tet.facets)[:, indices]
    points = np.asarray(mesh_tet.p)[:, triangles]
    matches = np.asarray(pred(points.reshape(3, -1))).reshape(3, -1).all(axis=0)
    return indices[matches]


def _facet_area(mesh_tet: Any, pred: Callable[[Any], Any]) -> float:
    """Total surface area (mm^2) of the triangular facets satisfying ``pred`` — used to turn a total
    FORCE into the uniform traction (force/area) the solve applies. Pure geometry."""
    fi = _boundary_facets(mesh_tet, pred)
    facets = np.asarray(mesh_tet.facets)[:, fi]  # (3, n) triangle vertex indices
    pts = np.asarray(mesh_tet.p)  # (3, n_nodes)
    v0, v1, v2 = pts[:, facets[0]].T, pts[:, facets[1]].T, pts[:, facets[2]].T
    return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())


def load_response_on_tetmesh(
    mesh_tet: Any,
    load_n: float,
    fixed_pred: Callable[[Any], Any],
    load_facet_pred: Callable[[Any], Any],
    load_dir_unit: tuple[float, float, float],
    e_mpa: float,
    nu: float,
) -> dict[str, Any] | None:
    """Absolute response to a TOTAL force ``load_n`` (N) spread over the load facets, in unit direction
    ``load_dir_unit``, with ``fixed_pred`` held. Quadratic (P2) tets — the harness-validated decision-grade
    path. Returns ``{max_deflection_mm, deflection_at_mm, max_von_mises_mpa, stress_at_mm}`` (absolute), or
    ``None`` if the load facets have no area or the solve is non-finite. Pure scikit-fem (no mesher) so it
    is unit-testable against an analytic cantilever."""
    area = _facet_area(mesh_tet, load_facet_pred)
    if area <= 0.0 or load_n <= 0.0:
        return None
    traction = (
        load_n / area
    )  # uniform traction so the integral over the face equals the total force
    ld = (load_dir_unit[0] * traction, load_dir_unit[1] * traction, load_dir_unit[2] * traction)
    res = solve_elasticity(mesh_tet, e_mpa, nu, fixed_pred, load_facet_pred, ld, element="P2")
    if res is None:
        return None
    disp, vm_node = res
    dmag = np.linalg.norm(disp, axis=1)
    pts = np.asarray(mesh_tet.p)  # (3, n_nodes)
    i_d = int(np.argmax(dmag))
    i_s = int(np.argmax(vm_node))
    if not (np.isfinite(dmag[i_d]) and np.isfinite(vm_node[i_s])):
        return None
    return {
        "max_deflection_mm": float(dmag[i_d]),
        "deflection_at_mm": (
            round(float(pts[0, i_d]), 2),
            round(float(pts[1, i_d]), 2),
            round(float(pts[2, i_d]), 2),
        ),
        "max_von_mises_mpa": float(vm_node[i_s]),
        "stress_at_mm": (
            round(float(pts[0, i_s]), 2),
            round(float(pts[1, i_s]), 2),
            round(float(pts[2, i_s]), 2),
        ),
    }


# The crash-prone native solve runs in an ISOLATED subprocess (below) so a tetrahedralizer OOM,
# segfault, or hang can only kill that child, never the request worker. A hung solve is terminated.
_FEA_TIMEOUT_DEFAULT_S = 30.0


def _fea_timeout_s() -> float:
    """Subprocess FEA budget (seconds). 30 s for the LIVE SERVER — an upload must not hold a request
    worker longer. Offline batch callers (the arm-search harness) raise it via FORGE_FEA_TIMEOUT_S:
    arm-scale parts (~35k tets) legitimately solve in 2-3 min, far past 30 s, and there is no request
    worker to protect. Read per call so a caller can set the env without a reimport; default unchanged.
    Fails SOFT to the default on an empty/malformed value -- this module's contract is NO-RESULT, never
    a crash, so a blank ``FORGE_FEA_TIMEOUT_S=`` must not raise into the request worker (matches the
    ``max_param_evals`` / ``_build_workers`` env-parse guards)."""
    raw = os.getenv("FORGE_FEA_TIMEOUT_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return _FEA_TIMEOUT_DEFAULT_S


def _fea_compute(
    verts: np.ndarray,
    faces: np.ndarray,
    e_mpa: float,
    nu: float,
    edge_length_fac: float,
) -> dict[str, Any] | None:
    """The heavy, crash-prone FEA: tetrahedralize → linear solve → map to input vertices. Returns a
    plain dict of the StressField fields (picklable across the process boundary) or None on a non-crash
    failure. MUST run inside the isolated subprocess (``_fea_entry``): pytetwild/fTetWild can OOM or
    segfault, which would take down the caller if run in-process. Uses a throwaway CWD (fTetWild litters
    the CWD with scratch files)."""
    try:
        import pytetwild
        from scipy.spatial import cKDTree
        from skfem import MeshTet
    except Exception:
        return None  # FEA deps absent → degrade silently (reported feature, not a gate)

    prev_cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="fea_")
    try:
        os.chdir(scratch)
        if len(verts) < 4 or len(faces) < 4:
            return None

        # pytetwild ships no type exports; the call is validated by the de-risk + the box test.
        res = pytetwild.tetrahedralize(verts, faces, edge_length_fac=edge_length_fac)
        tpts = np.asarray(res[0], dtype=float)
        ttet = np.asarray(res[1], dtype=np.int64)
        if len(tpts) < 4 or len(ttet) < 1:
            return None

        zmin, zmax = float(tpts[:, 2].min()), float(tpts[:, 2].max())
        span = zmax - zmin
        if span <= 1e-9:
            return None
        tol = max(span * 0.02, 1e-6)  # generic illustrative heatmap end regions

        vm_node = _solve_von_mises(
            MeshTet(tpts.T, ttet.T),
            e_mpa,
            nu,
            fixed_pred=lambda x: x[2] < zmin + tol,
            load_facet_pred=lambda x: x[2] > zmax - tol,
            load_dir=(0.0, 0.0, -1.0),
        )
        if vm_node is None:
            return None

        # Map tet-node stress → INPUT-mesh vertices (nearest tet node) so the caller can colour the GLB
        # by vertex index (the tet mesh remeshes the surface, so vertices differ).
        _, idx = cKDTree(tpts).query(verts, k=1)
        vm_surface = np.asarray(vm_node)[idx]

        vmax = float(vm_surface.max())
        if not np.isfinite(vmax) or vmax <= 0.0:
            return None
        argmax = int(np.argmax(vm_surface))
        return {
            "von_mises": tuple(round(float(v), 4) for v in vm_surface),
            "von_mises_max": round(vmax, 4),
            "von_mises_p95": round(float(np.percentile(vm_surface, 95)), 4),
            "max_location_mm": (
                round(float(verts[argmax][0]), 2),
                round(float(verts[argmax][1]), 2),
                round(float(verts[argmax][2]), 2),
            ),
            "load_case": "base fixed, unit downward push on the top face",
            "material_e_mpa": e_mpa,
            "note": DISCLAIMER,
        }
    except Exception:
        return None  # any mesher/solver failure → NO RESULT, never a misleading number
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def _fea_entry(
    verts: np.ndarray,
    faces: np.ndarray,
    e_mpa: float,
    nu: float,
    edge_length_fac: float,
    q: Any,
) -> None:
    """Subprocess entrypoint: compute and hand the result back over the queue. A native crash here kills
    only this child; the parent sees the dead process and degrades to NO-RESULT."""
    try:
        q.put(_fea_compute(verts, faces, e_mpa, nu, edge_length_fac))
    except Exception:
        q.put(None)


def _fea_run(target: Callable[..., None], args: tuple[Any, ...]) -> dict[str, Any] | None:
    """Run a crash-prone FEA ``target`` in an ISOLATED spawned subprocess (fresh interpreter — shares no
    FastAPI/uvicorn state) with a timeout + ``_FEA_LOCK`` serialization. The target receives ``*args`` plus
    a result queue as its last arg and must put its dict result (or None) on it. A native OOM / segfault /
    hang in the child returns None here, never taking down the caller."""
    ctx = mp.get_context("spawn")
    q: Any = ctx.Queue()
    proc = ctx.Process(target=target, args=(*args, q), daemon=True)
    payload: Any = None
    with _FEA_LOCK:
        proc.start()
        deadline = time.monotonic() + _fea_timeout_s()
        while time.monotonic() < deadline:
            try:
                payload = q.get(timeout=0.5)
                break  # got the result (or the child's own None)
            except _queue.Empty:
                if not proc.is_alive():
                    break  # child died (OOM/segfault) before producing a result
        if proc.is_alive():
            proc.terminate()  # timed out → stop the hung solve
        proc.join(timeout=5)
    return payload if isinstance(payload, dict) else None


def estimate_stress(
    mesh: Any,
    *,
    e_mpa: float = DEFAULT_E_MPA,
    nu: float = DEFAULT_NU,
    edge_length_fac: float = 0.05,
) -> StressField | None:
    """Relative von Mises under a canonical load (fix the lowest face, push down on the highest face)
    for a Z-up ``trimesh.Trimesh``. NO-RESULT (``None``) on any failure — never a fake number.

    The crash-prone native solve runs in an ISOLATED spawned subprocess: a tetrahedralizer OOM,
    segfault, or hang kills only that child and this returns None (the verdict still ships; the
    illustrative lens is simply omitted), instead of taking down the request worker."""
    # Kill switch (incident mitigation): FORGE_FEA_STRESS=0 disables this lens entirely. With the
    # subprocess isolation below, a crash can no longer reach the verdict, so it defaults back ON.
    if os.getenv("FORGE_FEA_STRESS", "1") == "0":
        return None
    try:
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    except Exception:
        return None
    if len(verts) < 4 or len(faces) < 4:
        return None

    payload = _fea_run(_fea_entry, (verts, faces, e_mpa, nu, edge_length_fac))
    if payload is None:
        return None
    try:
        return StressField(**payload)
    except Exception:
        return None


# ── Decision-grade real-load response (slice 8) ───────────────────────────────────────────────────

LOAD_DISCLAIMER = (
    "Isotropic linear-elastic FEA (quadratic tets, validated vs closed-form to a few percent). A 3D-printed "
    "part is markedly WEAKER across its layer lines, so treat these as values for a SOLID/isotropic part: "
    "compare the peak stress to your material's tested yield with an FDM derate, and verify a load-bearing "
    "part with a real build. Not a safety guarantee."
)


@dataclass(frozen=True)
class LoadResponse:
    """Absolute response to a stated load (ADR-0064): max deflection + peak von Mises, each located, under
    a quadratic-element solve. Decision-grade for a solid/isotropic material (the harness validates the
    solver); the FDM derate caveat rides in ``note``. REPORTED — a conservative caller may FAIL on
    over-stress but must keep a PASS advisory (no certifiable FDM yield)."""

    max_deflection_mm: float
    deflection_at_mm: tuple[float, float, float]
    max_von_mises_mpa: float
    stress_at_mm: tuple[float, float, float]
    load_n: float
    load_case: str
    material_e_mpa: float
    note: str
    boundary_tolerance_mm: float | None = None
    loaded_area_mm2: float | None = None
    fixed_area_mm2: float | None = None


def _unit(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5
    return (0.0, 0.0, -1.0) if n < 1e-12 else (v[0] / n, v[1] / n, v[2] / n)


def _load_response_compute(
    verts: np.ndarray,
    faces: np.ndarray,
    load_n: float,
    load_dir: tuple[float, float, float],
    e_mpa: float,
    nu: float,
    edge_length_fac: float,
    load_patch: tuple[tuple[float, float], float] | None = None,
    planar_faces: bool = False,
) -> dict[str, Any] | None:
    """Tetrahedralize the foreign surface, then solve the absolute load response (base fixed, the stated
    force spread over the top face -- or a ``load_patch`` sub-region of it -- in ``load_dir``). MUST run
    in the isolated subprocess (pytetwild can OOM/segfault). Returns the LoadResponse fields or None."""
    try:
        import pytetwild
        from skfem import MeshTet
    except Exception:
        return None
    prev_cwd = os.getcwd()
    scratch = tempfile.mkdtemp(prefix="fea_load_")
    try:
        os.chdir(scratch)
        if len(verts) < 4 or len(faces) < 4:
            return None
        res = pytetwild.tetrahedralize(
            verts,
            faces,
            edge_length_fac=edge_length_fac,
            epsilon=1e-4 if planar_faces else 0.001,
        )
        tpts = np.asarray(res[0], dtype=float)
        ttet = np.asarray(res[1], dtype=np.int64)
        if len(tpts) < 4 or len(ttet) < 1:
            return None
        zmin, zmax = float(verts[:, 2].min()), float(verts[:, 2].max())
        span = zmax - zmin
        if span <= 1e-9:
            return None
        # fTetWild has a declared surface envelope. Accept only exterior facets
        # entirely on the original extreme planes within twice that envelope.
        tol = max(float(np.linalg.norm(np.ptp(verts, axis=0))) * 2e-4, 1e-7)
        triangles = verts[faces]
        areas = (
            np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
                axis=1,
            )
            / 2
        )
        original_top = float(areas[np.all(abs(triangles[:, :, 2] - zmax) < 1e-8, axis=1)].sum())
        original_base = float(areas[np.all(abs(triangles[:, :, 2] - zmin) < 1e-8, axis=1)].sum())
        if planar_faces and min(original_top, original_base) <= 0:
            return None  # curved/sloped extremes are not the declared planar load case
        if not planar_faces:
            tol = max(span * 0.02, 1e-6)
        mesh_tet = MeshTet(tpts.T, ttet.T)
        full_top_area = _facet_area(mesh_tet, lambda x: abs(x[2] - zmax) <= tol)
        base_area = _facet_area(mesh_tet, lambda x: abs(x[2] - zmin) <= tol)
        if planar_faces and (
            abs(full_top_area / original_top - 1) > 0.005
            or abs(base_area / original_base - 1) > 0.005
        ):
            return None  # meshing lost the specified face; never silently load a smaller patch
        # The loaded region: the whole top face, OR a PATCH (facets within radius r of (px, py) on
        # the top face). x is (dim, n_facet_midpoints) in the skfem convention -> vectorized preds.
        _patch = None if load_patch is None else (load_patch[0], float(load_patch[1]) ** 2)

        def load_pred(x: Any) -> Any:
            on_top = abs(x[2] - zmax) <= tol
            if _patch is None:
                return on_top
            (px, py), r2 = _patch
            return on_top & ((x[0] - px) ** 2 + (x[1] - py) ** 2 <= r2)

        out = load_response_on_tetmesh(
            mesh_tet,
            load_n,
            fixed_pred=lambda x: abs(x[2] - zmin) <= tol,
            load_facet_pred=load_pred,
            load_dir_unit=_unit(load_dir),
            e_mpa=e_mpa,
            nu=nu,
        )
        if out is None:
            return None
        ud = _unit(load_dir)
        out["load_n"] = round(float(load_n), 3)
        where = (
            "spread over the planar top face"
            if planar_faces
            else "spread over exterior facets in the top 2% height region"
            if load_patch is None
            else f"on an exterior end-region patch (r={load_patch[1]:.1f}) at {load_patch[0]}"
        )
        out["load_case"] = (
            f"{load_n:.1f} N {where}, {'planar base' if planar_faces else 'bottom 2% exterior region'} fixed, direction ({ud[0]:.2f}, {ud[1]:.2f}, {ud[2]:.2f})"
        )
        out["material_e_mpa"] = e_mpa
        out["boundary_tolerance_mm"] = tol
        out["loaded_area_mm2"] = _facet_area(mesh_tet, load_pred)
        out["fixed_area_mm2"] = base_area
        out["note"] = LOAD_DISCLAIMER
        return out
    except Exception:
        return None
    finally:
        os.chdir(prev_cwd)
        shutil.rmtree(scratch, ignore_errors=True)


def _load_response_entry(
    verts: np.ndarray,
    faces: np.ndarray,
    load_n: float,
    load_dir: tuple[float, float, float],
    e_mpa: float,
    nu: float,
    edge_length_fac: float,
    load_patch: tuple[tuple[float, float], float] | None,
    planar_faces: bool,
    q: Any,
) -> None:
    try:
        q.put(
            _load_response_compute(
                verts, faces, load_n, load_dir, e_mpa, nu, edge_length_fac, load_patch, planar_faces
            )
        )
    except Exception:
        q.put(None)


def estimate_load_response(
    mesh: Any,
    load_n: float,
    *,
    load_dir: tuple[float, float, float] = (0.0, 0.0, -1.0),
    e_mpa: float = DEFAULT_E_MPA,
    nu: float = DEFAULT_NU,
    edge_length_fac: float = 0.05,
    load_patch: tuple[tuple[float, float], float] | None = None,
    planar_faces: bool = False,
) -> LoadResponse | None:
    """Absolute deflection + peak stress for a Z-up ``trimesh.Trimesh`` under a stated total force (N).
    GATED on ``fea_is_validated()`` — decision-grade numbers run ONLY when the solver is proven against
    closed-form cases (else None: no proof -> no number, fail closed). Same crash-isolated subprocess as
    ``estimate_stress``; NO-RESULT on any failure.

    ``planar_faces=True`` requires the original extreme planar caps and checks remeshed cap area.
    The legacy default uses explicitly reported exterior end regions within 2% of total Z height.
    ``load_patch`` restricts the selected load facets to a circle. Empty regions yield None."""
    if os.getenv("FORGE_FEA_STRESS", "1") == "0":
        return None
    # Lazy import avoids a circular import (fea_validation imports this module).
    from forge_mind.physics.fea_validation import fea_is_validated

    if not fea_is_validated():
        return None  # the solver is not proven on this box -> never emit a decision-grade number
    try:
        verts = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int32)
    except Exception:
        return None
    if len(verts) < 4 or len(faces) < 4 or load_n <= 0.0:
        return None
    payload = _fea_run(
        _load_response_entry,
        (verts, faces, load_n, load_dir, e_mpa, nu, edge_length_fac, load_patch, planar_faces),
    )
    if payload is None:
        return None
    try:
        return LoadResponse(**payload)
    except Exception:
        return None


__all__ = [
    "DEFAULT_E_MPA",
    "DEFAULT_NU",
    "DISCLAIMER",
    "LOAD_DISCLAIMER",
    "LoadResponse",
    "StressField",
    "estimate_load_response",
    "estimate_stress",
    "load_response_on_tetmesh",
    "solve_elasticity",
]
