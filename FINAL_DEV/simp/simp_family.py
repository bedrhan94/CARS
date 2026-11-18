"""Supplementary application family SIMP3D: stiffness sequences of a 3D SIMP compliance-minimization run.

Governing problem (Bendsoe 1989; Sigmund 2001; Andreassen et al. 2011; Liu & Tovar 2014): minimize the compliance
f^T u subject to K(x) u = f, mean(x) = v_f, 0 <= x_e <= 1, with the modified SIMP interpolation
E_e(x) = E_min + x_e^p (E_0 - E_min), E_0 = 1, p = 3; sensitivity filter of radius r_min (in element widths); optimality-
criteria (OC) update with move limit 0.2 and damping 0.5, bisection on the Lagrange multiplier.
Discretization: the CARS trilinear hexahedral elasticity mesh on the unit cube (N^3 elements, clamped at x = 0), the
frozen element factor Z_ref_elas_h1 (nu = 0.3) and the frozen float assembly; load: the uniform downward body load of
the E3 families (design-independent). Coefficients: c_e = E_e / N (the E3 scaling).
This is an independently specified optimization process acting on the CARS discretization: the coefficient-change
pattern comes from the optimizer, not from the CARS generator; the discretization and assembly are CARS's own.

  generate(N, vf, iters, emin, rmin)  -> densities x_k (k = 1..iters) and compliance history (1 thread, deterministic)
  SimpFamily(path, size, inst, window) -> a Family3D-compatible object over a frozen window of 21 iterations
"""
import os
import json
import hashlib
from pathlib import Path
import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
import sys                                                           # noqa: E402
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import runner6 as R6                                                 # noqa: E402  (path set-up and helpers only)
from cars import _reference_m0 as v                                  # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402
from fem3d_ref import assemble_float                                 # noqa: E402

SIZES = {'S2': 12, 'S3': 16}


def _filter_matrix(mesh, N, rmin):
    """Sensitivity-filter weights H_ij = max(0, rmin - dist(i, j)) on element centres (distances in element widths)."""
    c = mesh.centers * N
    rows, cols, vals = [], [], []
    r = int(np.ceil(rmin))
    idx = {tuple(np.floor(p).astype(int)): e for e, p in enumerate(c)}
    for e, p in enumerate(c):
        i0, j0, k0 = np.floor(p).astype(int)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                for dk in range(-r, r + 1):
                    f = idx.get((i0 + di, j0 + dj, k0 + dk))
                    if f is None:
                        continue
                    w = rmin - np.sqrt(di * di + dj * dj + dk * dk)
                    if w > 0:
                        rows.append(e); cols.append(f); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(len(c), len(c)))
    return H, np.asarray(H.sum(axis=1)).ravel()


def generate(N, vf, iters, emin, rmin=1.5, p=3.0, move=0.2):
    import pypardiso
    fam = Family3D('E3-TS', {12: 'S2', 16: 'S3'}[N], seed=0, split='dev')
    mesh, Z, Kref = fam.mesh, fam.Z, fam.Kref
    b = fam.get(0)[1]                                            # the E3 body load (design-independent)
    ne = mesh.centers.shape[0]
    H, Hs = _filter_matrix(mesh, N, rmin)
    x = np.full(ne, vf); hist = []; xs = []
    dofs = mesh.dofs
    for k in range(1, iters + 1):
        E = emin + x ** p * (1.0 - emin)
        A = v.canonical(assemble_float(mesh, Z, E / N, Kref))
        u = pypardiso.spsolve(A.tocsr(), b)
        ue = np.where(dofs >= 0, u[np.maximum(dofs, 0)], 0.0)     # element displacement vectors (0 on fixed dofs)
        ce = np.einsum('ei,ij,ej->e', ue, Kref, ue) / N
        comp = float(np.dot(b, u))
        dc = -p * x ** (p - 1) * (1.0 - emin) * ce
        dc = np.asarray(H @ (x * dc)) / Hs / np.maximum(1e-3, x)   # Sigmund (2001) sensitivity filter
        xs.append(x.copy()); hist.append(dict(k=k, compliance=comp, volume=float(x.mean()), u_norm=float(np.linalg.norm(u))))
        l1, l2 = 0.0, 1e9
        while (l2 - l1) / (l1 + l2) > 1e-6:                        # OC update, bisection on the multiplier
            lm = 0.5 * (l1 + l2)
            xn = np.maximum(0.0, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * np.sqrt(-dc / lm)))))
            if xn.mean() > vf:
                l1 = lm
            else:
                l2 = lm
        hist[-1]['change'] = float(np.abs(xn - x).max())
        x = xn
    return xs, hist


class SimpFamily(Family3D):
    """Family3D-compatible view of a frozen SIMP window: t = 0..20 <-> iterations k0 .. k0 + 20."""

    def __init__(self, frozen_dir, size_label, inst, window):
        super().__init__('E3-TS', size_label, seed=0, split='dev')
        meta = json.loads((Path(frozen_dir) / 'simp_manifest.json').read_text())
        w = meta['instances'][str(inst)]['windows'][window]
        self.id = 'SIMP3D'; self.label = 'P'; self.change = 'simp'; self.split = 'simp-' + window
        self.seed = int(str(inst).split('_')[0]); self.inst_key = str(inst); self.size_label = size_label
        self._c = [np.array([float.fromhex(h) for h in row], dtype=np.float64) for row in
                   json.loads((Path(frozen_dir) / w['coeff_hex_file']).read_text())]
        self.T = len(self._c) - 1
        self.window = window; self.k0 = w['k0']

    def coeff(self, t):
        return self._c[t].copy()
