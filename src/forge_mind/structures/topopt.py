"""Generative structural design by 2-D SIMP topology optimization (promoted from scripts/;
ADR-0097, founder direction 2026-07-13).

Given a rectangular design domain, WHERE it is anchored and WHERE loads act, decide per-element
whether material should exist, to make the STIFFEST structure at a fixed mass budget -- computed
fresh from the load case every time (the "generative drone frame" method; nothing looked up).
It rediscovers trusses/I-beams on its own; that self-rediscovery is the proof it works.

The founder's component rule: the engine builds the SKELETON, never the components. A component's
mounting face enters as KEEP-SOLID (always material, carrying the interface) and its body +
clearance as KEEP-VOID (a clean pocket the real part drops into); SIMP decides only the space
between. Density-based SIMP (Sigmund 2001 / Andreassen 2011) with an Optimality-Criteria update
and a cone density filter (kills checkerboarding). Pure numpy/scipy 2-D quad FEA -- independent of
the tet FEA that VERIFIES the extruded 3-D part downstream (validation stays in the loop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def _element_stiffness(nu: float = 0.3) -> np.ndarray:
    """8x8 stiffness of a unit bilinear quad (plane stress, E=1) -- the canonical SIMP element."""
    k = np.array(
        [
            0.5 - nu / 6,
            0.125 + nu / 8,
            -0.25 - nu / 12,
            -0.125 + 3 * nu / 8,
            -0.25 + nu / 12,
            -0.125 - nu / 8,
            nu / 6,
            0.125 - 3 * nu / 8,
        ]
    )
    idx = np.array(
        [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 0, 7, 6, 5, 4, 3, 2],
            [2, 7, 0, 5, 6, 3, 4, 1],
            [3, 6, 5, 0, 7, 2, 1, 4],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 4, 3, 2, 1, 0, 7, 6],
            [6, 3, 4, 1, 2, 7, 0, 5],
            [7, 2, 1, 4, 3, 6, 5, 0],
        ]
    )
    result: np.ndarray = (k[idx]) / (1 - nu**2)
    return result


def _density_filter(nelx: int, nely: int, rmin: float) -> tuple[Any, np.ndarray]:
    """Cone filter weights H (element-to-element within rmin) + row sums (mesh independence)."""
    nel = nelx * nely
    rows, cols, vals = [], [], []
    r = int(np.ceil(rmin)) - 1
    for i in range(nelx):
        for j in range(nely):
            e1 = i * nely + j
            for ii in range(max(i - r, 0), min(i + r + 1, nelx)):
                for jj in range(max(j - r, 0), min(j + r + 1, nely)):
                    w = rmin - float(np.hypot(i - ii, j - jj))
                    if w > 0:
                        rows.append(e1)
                        cols.append(ii * nely + jj)
                        vals.append(w)
    h = coo_matrix((vals, (rows, cols)), shape=(nel, nel)).tocsr()
    return h, np.asarray(h.sum(1)).ravel()


@dataclass
class Domain:
    """The design problem: a nelx x nely grid, fixed node DOFs (supports), point loads, and the
    component regions (keep_solid mounts / keep_void pockets, element indices e = ix*nely + iy)."""

    nelx: int
    nely: int
    volfrac: float = 0.4
    penal: float = 3.0
    rmin: float = 1.5
    fixed_dofs: list[int] = field(default_factory=list)
    load_dofs: list[int] = field(default_factory=list)
    load_vals: list[float] = field(default_factory=list)
    keep_solid: list[int] = field(default_factory=list)
    keep_void: list[int] = field(default_factory=list)

    def node(self, ix: int, iy: int) -> int:
        return ix * (self.nely + 1) + iy

    def dof(self, ix: int, iy: int, comp: int) -> int:
        return 2 * self.node(ix, iy) + comp  # comp 0=x, 1=y

    def elements_in_box(self, ix0: int, iy0: int, ix1: int, iy1: int) -> list[int]:
        """Element indices whose cell lies in [ix0,ix1) x [iy0,iy1) -- a component footprint."""
        return [
            ix * self.nely + iy
            for ix in range(max(ix0, 0), min(ix1, self.nelx))
            for iy in range(max(iy0, 0), min(iy1, self.nely))
        ]


def optimize(dom: Domain, *, max_iter: int = 60, tol: float = 0.01) -> dict[str, Any]:
    """Run SIMP. Returns the density field (nely x nelx, 1=solid) + compliance history."""
    nelx, nely = dom.nelx, dom.nely
    nel = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    emin, e0, p = 1e-9, 1.0, dom.penal
    ke = _element_stiffness()

    edof = np.zeros((nel, 8), dtype=int)
    for i in range(nelx):
        for j in range(nely):
            e = i * nely + j
            n1 = (nely + 1) * i + j
            n2 = (nely + 1) * (i + 1) + j
            edof[e] = [
                2 * n1,
                2 * n1 + 1,
                2 * n2,
                2 * n2 + 1,
                2 * n2 + 2,
                2 * n2 + 3,
                2 * n1 + 2,
                2 * n1 + 3,
            ]
    ik = np.kron(edof, np.ones((8, 1))).flatten()
    jk = np.kron(edof, np.ones((1, 8))).flatten()

    h, hs = _density_filter(nelx, nely, dom.rmin)
    force = np.zeros(ndof)
    for d, v in zip(dom.load_dofs, dom.load_vals, strict=True):
        force[d] += v
    free = np.setdiff1d(np.arange(ndof), np.asarray(dom.fixed_dofs, dtype=int))
    solid_mask = np.zeros(nel, dtype=bool)
    solid_mask[np.asarray(dom.keep_solid, dtype=int)] = True
    void_mask = np.zeros(nel, dtype=bool)
    void_mask[np.asarray(dom.keep_void, dtype=int)] = True

    x = np.full(nel, dom.volfrac)
    x[solid_mask] = 1.0
    x[void_mask] = 0.0
    hist: list[float] = []
    xphys = x.copy()
    for _ in range(max_iter):
        sk = (ke.flatten()[:, None] * (emin + xphys**p * (e0 - emin))).flatten(order="F")
        k = coo_matrix((sk, (ik, jk)), shape=(ndof, ndof)).tocsc()
        u = np.zeros(ndof)
        u[free] = spsolve(k[free, :][:, free], force[free])
        ue = u[edof]
        ce = np.einsum("ij,jk,ik->i", ue, ke, ue)
        compliance = float((emin + xphys**p * (e0 - emin) * ce).sum())
        hist.append(compliance)
        dc = -p * xphys ** (p - 1) * (e0 - emin) * ce
        dv = np.ones(nel)
        dc = np.asarray(h @ (dc / hs))
        dv = np.asarray(h @ (dv / hs))
        l1, l2, move = 0.0, 1e9, 0.2
        xnew = x
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(
                x * np.sqrt(np.maximum(-dc, 0) / (dv * lmid + 1e-30)),
                np.maximum(x - move, 0.0),
                np.minimum(x + move, 1.0),
            )
            xphys = np.asarray(h @ xnew) / hs
            xphys[solid_mask] = 1.0  # passive regions override the filter (hard pocket / mount)
            xphys[void_mask] = 0.0
            (l1, l2) = (lmid, l2) if xphys.sum() > dom.volfrac * nel else (l1, lmid)
        xnew[solid_mask] = 1.0
        xnew[void_mask] = 0.0
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        if len(hist) > 2 and change < tol:
            break
    return {"density": xphys.reshape((nelx, nely)).T, "compliance": hist, "iters": len(hist)}


def cantilever(nelx: int = 60, nely: int = 30, volfrac: float = 0.4, rmin: float = 2.0) -> Domain:
    """Classic validation problem: left edge fully fixed, unit downward load at mid-right."""
    dom = Domain(nelx=nelx, nely=nely, volfrac=volfrac, rmin=rmin)
    dom.fixed_dofs = [dom.dof(0, iy, c) for iy in range(nely + 1) for c in (0, 1)]
    dom.load_dofs = [dom.dof(nelx, nely // 2, 1)]
    dom.load_vals = [-1.0]
    return dom


def as_ascii(density: np.ndarray, thresh: float = 0.5) -> str:
    """Quick visual: solid elements as '#', intermediate '.', void as space."""
    chars = np.where(density >= thresh, "#", np.where(density >= 0.2, ".", " "))
    return "\n".join("".join(row) for row in chars)


__all__ = ["Domain", "as_ascii", "cantilever", "optimize"]
