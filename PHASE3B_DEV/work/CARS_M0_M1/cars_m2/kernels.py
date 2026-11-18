"""Phase 3B preconditioner kernels (new file; the pilot's runner.py is left unchanged).

K1  exact operator P^T L^{-T} D^{-1} L^{-1} P via scipy spsolve_triangular (M1 / pilot kernel).
K3  the SAME exact operator, applied with SuperLU's own triangular kernels: the unit-lower L~ is itself
    factored by splu(NATURAL, diag_pivot_thresh=0). The factorization is accepted only if it is the
    identity factorization  L~ = L' * I  with identity permutations, L' == L~ bitwise and U' == I bitwise.
    Then lu_L.solve(.) and lu_L.solve(., trans='T') apply L~^{-1} and L~^{-T}; with all d > 0 the operator
    is symmetric positive definite in exact arithmetic (identical mathematical object to K1).
K2  (1/2)[(LU)^{-1} + (LU)^{-T}] is NOT provided: its positive definiteness is not proven (Phase 3B
    NUMERICAL_VALIDITY_AND_K2.md).
Preconditioners never enter a certificate.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from cars import _reference_m0 as v


def _perm(perm):
    n = len(perm)
    return sp.csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))


def k1(factor):
    return v.ldlt_preconditioner(factor), {'kernel': 'K1'}


def k3(factor):
    L, d, perm = factor['L'], factor['d'], factor['perm']
    if not np.all(d > 0):
        raise v.CertFail('PRECONDITIONER_NOT_SPD')
    n = L.shape[0]
    Lc = sp.csc_matrix(L)
    luL = sla.splu(Lc, permc_spec='NATURAL', diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
    ar = np.arange(n)
    if not (np.array_equal(luL.perm_r, ar) and np.array_equal(luL.perm_c, ar)):
        raise v.CertFail('K3_STRUCTURE', 'non-identity permutation')
    Lp = v.canonical(luL.L); Up = v.canonical(luL.U)
    if abs(Lp - v.canonical(L)).nnz != 0 or abs(Up - sp.eye(n, format='csr')).nnz != 0:
        raise v.CertFail('K3_STRUCTURE', 'factor of L is not the identity factorization')
    P = _perm(perm); PT = P.T.tocsr()
    dd = np.asarray(d, dtype=np.float64)

    def apply(r):
        y = luL.solve(P @ np.asarray(r, dtype=np.float64).ravel())
        z = luL.solve(y / dd, trans='T')
        return PT @ z
    return sla.LinearOperator((n, n), matvec=apply, dtype=np.float64), {'kernel': 'K3'}


def make(kernel, factor):
    """Return (operator, info). K3 falls back to K1 (logged) if its identity checks fail."""
    if kernel == 'K1':
        return k1(factor)
    if kernel == 'K3':
        try:
            return k3(factor)
        except v.CertFail as e:
            op, info = k1(factor)
            info['fallback_from_K3'] = e.reason
            return op, info
    raise ValueError(kernel)
