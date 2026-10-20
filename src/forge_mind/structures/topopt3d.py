"""FULL 3-D SIMP topology optimization (the founder's "3-D problem"; ADR-0098 stated limit).

The 2-D module (``topopt``) is honest for links/frames/plates -- extruded, uniform along one
axis. THIS solves the general case: a nelx x nely x nelz voxel domain, 8-node hexahedral FEA,
material free to arrange in all three dimensions (out-of-plane load paths, 3-D pockets). Same
contracts as 2-D: keep_solid mounts / keep_void pockets (the founder's component rule),
Optimality-Criteria update, a density filter for mesh-independence, and the REAL tet FEA verifies
whatever the mesher extracts downstream -- the optimizer proposes, the validated solver disposes.

Implementation notes (deliberate):
- The 24x24 hex stiffness is built NUMERICALLY (trilinear shape functions, 2x2x2 Gauss) rather
  than transcribed from the top3d paper -- correct by construction and unit-testable (symmetry,
  6 rigid-body zero-modes).
- The density filter is a normalized Gaussian (scipy.ndimage) applied to x, dc and dv -- the
  standard teaching-code form of the density-filter chain rule; downstream FEA verification is
  the backstop for any filter-induced bias.
- Cost honesty (Fable's warning made a number): the default 48x20x14 grid is ~46k dofs,
  ~1-3 s/solve, minutes per optimize() -- fine offline, NOT a live-server path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def hex_stiffness(nu: float = 0.3) -> np.ndarray:
    """24x24 stiffness of a unit trilinear hexahedron (E=1), 2x2x2 Gauss quadrature."""
    e, v = 1.0, nu
    lam = e * v / ((1 + v) * (1 - 2 * v))
    mu = e / (2 * (1 + v))
    c = np.zeros((6, 6))
    c[:3, :3] = lam
    c[np.arange(3), np.arange(3)] = lam + 2 * mu
    c[3:, 3:] = np.eye(3) * mu
    # node local coords in {-1,1}^3, ordered (ix, iy, iz) bits
    nodes = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], dtype=float)
    gp = np.array([-1.0, 1.0]) / np.sqrt(3.0)
    ke = np.zeros((24, 24))
    for gx in gp:
        for gy in gp:
            for gz in gp:
                # dN/dxi for the 8 trilinear shape functions at this Gauss point
                dn = np.zeros((8, 3))
                for a in range(8):
                    xa, ya, za = nodes[a]
                    dn[a, 0] = xa * (1 + ya * gy) * (1 + za * gz) / 8.0
                    dn[a, 1] = (1 + xa * gx) * ya * (1 + za * gz) / 8.0
                    dn[a, 2] = (1 + xa * gx) * (1 + ya * gy) * za / 8.0
                jac = dn.T @ nodes / 2.0  # d(xi)/d(x) for the unit cube [0,1]^3 element
                dndx = dn @ np.linalg.inv(jac).T
                b = np.zeros((6, 24))
                for a in range(8):
                    bx, by, bz = dndx[a]
                    b[0, 3 * a] = bx
                    b[1, 3 * a + 1] = by
                    b[2, 3 * a + 2] = bz
                    b[3, 3 * a] = by
                    b[3, 3 * a + 1] = bx
                    b[4, 3 * a + 1] = bz
                    b[4, 3 * a + 2] = by
                    b[5, 3 * a] = bz
                    b[5, 3 * a + 2] = bx
                ke += b.T @ c @ b * float(np.linalg.det(jac))
    return (ke + ke.T) / 2.0  # enforce exact symmetry


@dataclass
class Domain3D:
    """The 3-D design problem. Element (ix,iy,iz), e = ix*nely*nelz + iy*nelz + iz; node
    n = ix*(nely+1)*(nelz+1) + iy*(nelz+1) + iz; dof = 3*n + comp (0=x,1=y,2=z)."""

    nelx: int
    nely: int
    nelz: int
    volfrac: float = 0.3
    penal: float = 3.0
    rmin: float = 1.5  # Gaussian filter sigma, in cells
    fixed_dofs: list[int] = field(default_factory=list)
    load_dofs: list[int] = field(default_factory=list)
    load_vals: list[float] = field(default_factory=list)
    keep_solid: list[int] = field(default_factory=list)
    keep_void: list[int] = field(default_factory=list)

    def node(self, ix: int, iy: int, iz: int) -> int:
        return ix * (self.nely + 1) * (self.nelz + 1) + iy * (self.nelz + 1) + iz

    def dof(self, ix: int, iy: int, iz: int, comp: int) -> int:
        return 3 * self.node(ix, iy, iz) + comp

    def element(self, ix: int, iy: int, iz: int) -> int:
        return ix * self.nely * self.nelz + iy * self.nelz + iz

    def elements_in_box(
        self, ix0: int, iy0: int, iz0: int, ix1: int, iy1: int, iz1: int
    ) -> list[int]:
        return [
            self.element(ix, iy, iz)
            for ix in range(max(ix0, 0), min(ix1, self.nelx))
            for iy in range(max(iy0, 0), min(iy1, self.nely))
            for iz in range(max(iz0, 0), min(iz1, self.nelz))
        ]


def _edof_table(dom: Domain3D) -> np.ndarray:
    """(nel, 24) global dof indices per element (nodes in the hex_stiffness bit order)."""
    nel = dom.nelx * dom.nely * dom.nelz
    edof = np.zeros((nel, 24), dtype=int)
    for ix in range(dom.nelx):
        for iy in range(dom.nely):
            for iz in range(dom.nelz):
                e = dom.element(ix, iy, iz)
                cols = []
                for bx in (0, 1):
                    for by in (0, 1):
                        for bz in (0, 1):
                            n = dom.node(ix + bx, iy + by, iz + bz)
                            cols += [3 * n, 3 * n + 1, 3 * n + 2]
                edof[e] = cols
    return edof


def _filt(a3: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian filter (constant padding) -- edges keep unit total weight."""
    num = gaussian_filter(a3, sigma, mode="constant", cval=0.0)
    den = gaussian_filter(np.ones_like(a3), sigma, mode="constant", cval=0.0)
    result: np.ndarray = num / np.maximum(den, 1e-12)
    return result


def optimize3d(dom: Domain3D, *, max_iter: int = 40, tol: float = 0.01) -> dict[str, Any]:
    """Run 3-D SIMP. Returns density (nelx, nely, nelz) + compliance history."""
    nel = dom.nelx * dom.nely * dom.nelz
    ndof = 3 * (dom.nelx + 1) * (dom.nely + 1) * (dom.nelz + 1)
    emin, e0, p = 1e-9, 1.0, dom.penal
    ke = hex_stiffness()
    edof = _edof_table(dom)
    ik = np.repeat(edof, 24, axis=1).flatten()
    jk = np.tile(edof, (1, 24)).flatten()

    force = np.zeros(ndof)
    for d, v in zip(dom.load_dofs, dom.load_vals, strict=True):
        force[d] += v
    free = np.setdiff1d(np.arange(ndof), np.asarray(dom.fixed_dofs, dtype=int))
    shape3 = (dom.nelx, dom.nely, dom.nelz)
    solid = np.zeros(nel, dtype=bool)
    solid[np.asarray(dom.keep_solid, dtype=int)] = True
    void = np.zeros(nel, dtype=bool)
    void[np.asarray(dom.keep_void, dtype=int)] = True

    x = np.full(nel, dom.volfrac)
    x[solid], x[void] = 1.0, 0.0
    xphys = x.copy()
    hist: list[float] = []
    for _ in range(max_iter):
        sk = (ke.flatten()[:, None] * (emin + xphys**p * (e0 - emin))).flatten(order="F")
        k = coo_matrix((sk, (ik, jk)), shape=(ndof, ndof)).tocsc()
        u = np.zeros(ndof)
        u[free] = spsolve(k[free, :][:, free], force[free])
        ue = u[edof]
        ce = np.einsum("ij,jk,ik->i", ue, ke, ue)
        hist.append(float((emin + xphys**p * (e0 - emin) * ce).sum()))
        dc = -p * xphys ** (p - 1) * (e0 - emin) * ce
        dc = _filt(dc.reshape(shape3), dom.rmin).flatten()
        dv = _filt(np.ones(shape3), dom.rmin).flatten()
        l1, l2, move = 0.0, 1e9, 0.2
        xnew = x
        while (l2 - l1) / (l1 + l2 + 1e-30) > 1e-3:
            lmid = 0.5 * (l1 + l2)
            xnew = np.clip(
                x * np.sqrt(np.maximum(-dc, 0) / (dv * lmid + 1e-30)),
                np.maximum(x - move, 0.0),
                np.minimum(x + move, 1.0),
            )
            xphys = _filt(xnew.reshape(shape3), dom.rmin).flatten()
            xphys[solid], xphys[void] = 1.0, 0.0
            (l1, l2) = (lmid, l2) if xphys.sum() > dom.volfrac * nel else (l1, lmid)
        xnew[solid], xnew[void] = 1.0, 0.0
        change = float(np.max(np.abs(xnew - x)))
        x = xnew
        if len(hist) > 2 and change < tol:
            break
    return {"density": xphys.reshape(shape3), "compliance": hist, "iters": len(hist)}


def cantilever3d(
    nelx: int = 32, nely: int = 12, nelz: int = 10, volfrac: float = 0.3, rmin: float = 1.4
) -> Domain3D:
    """Validation problem: the x=0 face fully fixed, a -y point load at the far-face center."""
    dom = Domain3D(nelx=nelx, nely=nely, nelz=nelz, volfrac=volfrac, rmin=rmin)
    dom.fixed_dofs = [
        dom.dof(0, iy, iz, c) for iy in range(nely + 1) for iz in range(nelz + 1) for c in (0, 1, 2)
    ]
    dom.load_dofs = [dom.dof(nelx, nely // 2, nelz // 2, 1)]
    dom.load_vals = [-1.0]
    return dom


__all__ = ["Domain3D", "cantilever3d", "hex_stiffness", "optimize3d"]
