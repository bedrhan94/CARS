"""
CARS minimal REFERENCE implementation of the validated numerical core (Phase 2A).

Purpose: an executable, independently checkable statement of the M0 formulas.
NOT production code, NOT optimised. Every function documents the inequality it relies on.

Arithmetic assumptions (checked by `arith_check`, see handoff section 2):
  A1 IEEE-754 binary64, round-to-nearest-even for + - * / sqrt in NumPy ufuncs and SciPy sparse kernels.
  A2 No flush-to-zero / denormals-are-zero.
  A3 Library reductions (sparse matvec, sparse matmul, np.sum) are sums of individually rounded
     products/terms in SOME order, possibly with FMA (bounds below hold for any order and with FMA).
  A4 Any inf/nan in a bound => failure (fail closed).
Rounding control: RN followed by np.nextafter (Lemma V1), never the FPU control word.
"""
import hashlib
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

U = 2.0 ** -53          # unit roundoff
ETA = 2.0 ** -1074      # smallest positive subnormal
INF = np.inf


class CertFail(Exception):
    """Raised with a reason code; callers must map it to UNVERIFIED (fail closed)."""
    def __init__(self, reason, detail=""):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


# ----------------------------------------------------------------- V1: directed rounding emulation
def up(x):
    """Lemma V1: if y = RN(z) for a real z, then nextafter(y, +inf) >= z. Valid incl. subnormals."""
    return np.nextafter(x, INF)


def dn(x):
    return np.nextafter(x, -INF)


def arith_check():
    """A1/A2 smoke test (necessary, not sufficient)."""
    a, b = np.float64(2.0 ** -1060), np.float64(2.0 ** -10)
    sub = np.float64(ETA)
    ok = (a * b == 2.0 ** -1070)                                         # no FTZ (scalar)
    ok &= bool(np.all(np.full(16, a) * np.full(16, b) == 2.0 ** -1070))   # no FTZ (SIMD path)
    ok &= bool(sub > 0) and bool(np.all(np.full(16, sub) * 1.0 > 0))     # no DAZ
    ok &= (np.float64(1.0) + 2.0 ** -53 == 1.0)                          # ties-to-even
    ok &= (np.float64(1.0) + 3 * 2.0 ** -53 == 1.0 + 2.0 ** -51)
    return bool(ok)


def finite_or_fail(x, what):
    if not np.all(np.isfinite(x)):
        raise CertFail("NONFINITE", what)
    return x


def gamma_hi(m):
    """Upper bound of gamma_m = m u / (1 - m u). m*U and 1 - m*U are exact for integer m < 2^52."""
    m = np.asarray(m, dtype=np.float64)
    if np.any(m * U >= 0.5):
        raise CertFail("RESOURCE_LIMIT", "m*u >= 1/2")
    return up((m * U) / (1.0 - m * U))


def two_sum(a, b):
    """Knuth TwoSum: a + b = s + e exactly in RN binary64 (absent overflow)."""
    s = a + b
    bb = s - a
    e = (a - (s - bb)) + (b - bb)
    finite_or_fail(s, "two_sum")
    return s, e


def sum_nonneg_hi(v):
    """Upper bound of the exact sum of nonnegative floats v (any summation order):
    S <= fl(S) / (1 - gamma_{m-1})   (no underflow term needed: additions are exact in the subnormal range)."""
    v = np.asarray(v, dtype=np.float64).ravel()
    if v.size == 0:
        return 0.0
    if np.any(v < 0):
        raise ValueError("sum_nonneg_hi needs v >= 0")
    s = float(np.sum(v))
    den = dn(1.0 - gamma_hi(max(v.size - 1, 1)))
    return float(finite_or_fail(up(s / den), "sum"))


def absmatvec_hi(A, x):
    """Upper bound, row-wise, of the exact (|A| |x|)_i for canonical CSR A.
    Computed y = fl(|A||x|) with m_i products per row satisfies y_i >= (1 - gamma_{m_i}) a_i - m_i*eta/2,
    hence a_i <= (y_i + m_i*eta) / (1 - gamma_{m_i})."""
    m = np.diff(A.indptr).astype(np.float64)
    y = abs(A) @ np.abs(x)
    num = up(y + up(m * ETA))
    den = dn(1.0 - gamma_hi(np.maximum(m, 1.0)))
    return finite_or_fail(up(num / den), "absmatvec")


def canonical(A):
    A = sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    A.sort_indices()
    return A


def matrix_sha256(A):
    A = canonical(A)
    h = hashlib.sha256()
    h.update(np.int64(A.shape[0]).tobytes())
    for arr in (A.indptr.astype("<i8"), A.indices.astype("<i8"), A.data.astype("<f8")):
        h.update(arr.tobytes())
    return h.hexdigest()


def vec_sha256(x):
    return hashlib.sha256(np.asarray(x, dtype="<f8").tobytes()).hexdigest()


# ----------------------------------------------------------------- P1: validated residual bound
def residual_bound(A, b, x):
    """R_hi >= ||b - A x||_2 (exact reals).  r~ = fl(b - fl(A x));
    |r_i - r~_i| <= gamma_{m_i+1} (|b_i| + (|A||x|)_i) + (m_i+1) eta    (Lemma V3)."""
    A = canonical(A)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    finite_or_fail(A.data, "A"); finite_or_fail(b, "b"); finite_or_fail(x, "x")
    r = b - A @ x
    m = np.diff(A.indptr).astype(np.float64)
    a_hi = absmatvec_hi(A, x)
    delta = up(up(gamma_hi(m + 1.0) * up(np.abs(b) + a_hi)) + up((m + 1.0) * ETA))
    v = up(np.abs(r) + delta)
    return float(finite_or_fail(up(np.sqrt(sum_nonneg_hi(up(v * v)))), "R_hi"))


# ----------------------------------------------------------------- P9: Gershgorin / Collatz-Wielandt
def gershgorin_lower(A, w=None):
    """lambda_min(A) >= min_i ( a_ii - sum_{j!=i} |a_ij| w_j / w_i )   for symmetric A, any w > 0
    (Gershgorin applied to D^{-1} A D, D = diag(w)). For symmetric Z-matrices this is Collatz-Wielandt."""
    A = canonical(A)
    n = A.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    if not (np.all(w > 0) and np.all(np.isfinite(w))):
        raise CertFail("TRANSFER_DATA_INVALID", "w must be positive")
    d = A.diagonal()
    off = canonical(A - sp.diags(d))      # exact: a_ij - 0 for i != j, a_ii - a_ii = 0 on diagonal
    off.eliminate_zeros()
    t_hi = absmatvec_hi(off, w)
    q = up(t_hi / w)
    return float(np.min(dn(d - q)))


# ----------------------------------------------------------------- P3: norm bounds
def norm2_sym_hi(Rbar, iters=30):
    """||R||_2 <= rho(Rbar) <= max_i (Rbar w)_i / w_i, for ANY w > 0, PROVIDED R is SYMMETRIC and
    |R| <= Rbar entrywise (Rbar may be nonsymmetric).  Do NOT use for nonsymmetric R (Lemma V5)."""
    Rbar = canonical(Rbar)
    if Rbar.nnz == 0:
        return 0.0
    if np.any(Rbar.data < 0):
        raise ValueError("Rbar must be nonnegative")
    n = Rbar.shape[0]
    w = np.ones(n)
    for _ in range(iters):            # heuristic choice of w; validity does not depend on it
        w2 = Rbar @ w
        if w2.max() <= 0:
            break
        w = w2 / w2.max()
    w = np.maximum(w, 2.0 ** -40)     # strictly positive, representable
    t_hi = absmatvec_hi(Rbar, w)
    return float(finite_or_fail(np.max(up(t_hi / w)), "norm2_sym"))


def norm2_general_hi(Rbar):
    """For arbitrary (possibly nonsymmetric) R with |R| <= Rbar:  ||R||_2 <= sqrt(||Rbar||_1 ||Rbar||_inf)."""
    Rbar = canonical(Rbar)
    if Rbar.nnz == 0:
        return 0.0
    rows = max(sum_nonneg_hi(Rbar.getrow(i).data) for i in range(Rbar.shape[0]))
    C = Rbar.tocsc()
    cols = max(sum_nonneg_hi(C.getcol(j).data) for j in range(C.shape[1]))
    return float(up(np.sqrt(up(rows * cols))))


# ----------------------------------------------------------------- P5: inertia certificate
def anchor_inertia(A, sigma, k_max=0):
    """Returns dict with nu (=#negative pivots) and rho such that  lambda_{nu+1}(A) >= rho  (exact reals).
    Construction: SuperLU of M~ = A - sigma I (diagonal formed by TwoSum, low parts kept),
    candidate L D L^T with L = lu.L (must be unit lower triangular), D = diag(lu.U),
    validated residual R = P(A - sigma I)P^T - L D L^T, delta >= ||R||_2, Sylvester + Weyl.
    SuperLU's U is NOT exactly D L^T; only the candidate L D L^T is used, never U."""
    A = canonical(A)
    if abs(A - A.T).nnz != 0:
        raise CertFail("NOT_SYMMETRIC")
    sigma = float(sigma)
    n = A.shape[0]
    d0 = A.diagonal()
    s_hi, s_lo = two_sum(d0, np.full(n, -sigma))
    Mt = A.tolil()
    Mt.setdiag(s_hi)
    Mt = canonical(Mt)
    try:
        lu = spla.splu(Mt.tocsc(), permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                       options=dict(SymmetricMode=True))
    except RuntimeError as e:
        raise CertFail("ANCHOR_FACTORIZATION_FAILED", str(e))
    if not np.array_equal(lu.perm_r, lu.perm_c):
        raise CertFail("PERM_NOT_SYMMETRIC")
    L = canonical(lu.L)
    if sp.triu(L, 1).nnz != 0 or not np.all(L.diagonal() == 1.0):
        raise CertFail("FACTOR_STRUCTURE")
    d = lu.U.diagonal().astype(np.float64)
    finite_or_fail(L.data, "L"); finite_or_fail(d, "D")
    if np.any(d == 0):
        raise CertFail("ZERO_PIVOT")
    nu = int(np.sum(d < 0))
    if nu > k_max:
        raise CertFail("ANCHOR_INERTIA_MISMATCH", f"nu={nu} > k_max={k_max}")
    ar = np.arange(n)
    P = sp.csr_matrix((np.ones(n), (lu.perm_r, ar)), shape=(n, n))   # Pr[perm_r[i], i] = 1
    B = canonical(P @ Mt @ P.T)                                      # exact (permutation)
    lo_p = P @ s_lo                                                  # exact
    C = canonical(L @ sp.diags(d) @ L.T)                             # fl(L D L^T)
    absL = abs(L)
    absC = canonical(absL @ sp.diags(np.abs(d)) @ absL.T)            # fl(|L||D||L^T|)
    pat = (absL > 0).astype(np.float64)
    patC = canonical(pat @ pat.T)                                    # structural pattern (exact counts)
    patC.data[:] = 1.0
    c = float(np.max(np.diff(L.indptr)))                             # max terms per entry
    g = gamma_hi(c + 1.0)                                            # (L_ik*d_k)*L_jk: c+1 roundings/term
    den = dn(1.0 - g)
    tiny = up(2.0 * c * ETA)                                         # underflow allowance per entry
    # Every step below is ONE RN operation on .data followed by up() (Lemma V1).
    T = canonical(absC + patC * tiny); T.data = up(T.data)           # fl(|L||D||L^T|) + 2c*eta
    T.data = up(T.data / den)                                        # >= exact |L||D||L^T|
    EC = T.copy(); EC.data = up(EC.data * g)                         # >= gamma_{c+1}|L||D||L^T|
    EC = canonical(EC + patC * tiny); EC.data = up(EC.data)          # >= |C~ - L D L^T|
    Rt = canonical(B - C)                                            # fl(B - C~): one subtraction/entry
    Rbar = abs(Rt); Rbar.data = up(Rbar.data / dn(1.0 - U))         # >= |B - C~|
    Rbar = canonical(Rbar + EC); Rbar.data = up(Rbar.data)
    Rbar = canonical(Rbar + sp.diags(np.abs(lo_p))); Rbar.data = up(Rbar.data)   # + TwoSum low parts
    delta = norm2_sym_hi(Rbar)                                       # R is symmetric => Lemma V5 applies
    rho = float(dn(sigma - delta))
    finite_or_fail(rho, "rho")
    return dict(nu=nu, rho=rho, delta=delta, sigma=sigma, n=n,
                maxL=float(abs(L).max()), pivot_ratio=float(np.abs(d).max() / np.abs(d).min()),
                factor=dict(L=L, d=d, perm=lu.perm_r.copy()))


# ----------------------------------------------------------------- P4: anchored one-sided Weyl transfer
def weyl_transfer_lower(As, At, lam_s_lo, w=None):
    """lambda_min(A_t) >= lam_s_lo + l,  l <= lambda_min(A_t - A_s) by scaled Gershgorin.
    Entries of E = A_t - A_s are represented EXACTLY as s + e (TwoSum).  Requires lam_s_lo <= lambda_min(A_s).
    Anchored use only (E against the anchor, never chained)."""
    As, At = canonical(As), canonical(At)
    if As.shape != At.shape:
        raise CertFail("TRANSFER_DATA_INVALID", "shape")
    n = As.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, dtype=np.float64)
    if not (np.all(w > 0) and np.all(np.isfinite(w))):
        raise CertFail("TRANSFER_DATA_INVALID", "w must be positive")
    pat = canonical((As != 0).astype(np.float64) + (At != 0).astype(np.float64))
    rows, cols = pat.nonzero()
    a = np.asarray(As[rows, cols]).ravel(); b = np.asarray(At[rows, cols]).ravel()
    s, e = two_sum(b, -a)                                   # b - a = s + e exactly
    diag = rows == cols
    d_lo = np.zeros(n)
    d_lo[rows[diag]] = dn(s[diag] + e[diag])                # one RN op, then down
    off = ~diag
    mag = up(np.abs(s[off]) + np.abs(e[off]))               # >= |e_ij| exactly
    Mbar = sp.csr_matrix((mag, (rows[off], cols[off])), shape=(n, n))
    q = up(absmatvec_hi(Mbar, w) / w)
    ell = float(np.min(dn(d_lo - q)))
    return float(finite_or_fail(dn(float(lam_s_lo) + ell), "weyl"))


# ----------------------------------------------------------------- safe SPD preconditioner from an anchor
def ldlt_preconditioner(factor):
    """M^{-1} r = P^T L^{-T} D^{-1} L^{-1} P r  built from the STORED L (unit lower) and D = diag(U).
    As an exact operator this is (P^T L D L^T P)^{-1}: symmetric, and positive definite iff all d > 0.
    Never use SuperLU's lu.solve as a CG preconditioner: (LU)^{-1} is not symmetric because U != D L^T.
    The preconditioner affects only convergence, never certificate validity."""
    L, d, perm = factor["L"], factor["d"], factor["perm"]
    if not np.all(d > 0):
        raise CertFail("PRECONDITIONER_NOT_SPD")
    n = L.shape[0]
    P = sp.csr_matrix((np.ones(n), (perm, np.arange(n))), shape=(n, n))
    Lc, Ut = L.tocsr(), L.T.tocsr()

    def apply(r):
        y = spla.spsolve_triangular(Lc, P @ np.asarray(r).ravel(), lower=True, unit_diagonal=True)
        z = spla.spsolve_triangular(Ut, y / d, lower=False, unit_diagonal=True)
        return P.T @ z
    return spla.LinearOperator((n, n), matvec=apply, dtype=np.float64)


# ----------------------------------------------------------------- certificate status (fail closed)
def certificate(A, b, x, lam_lo, lam_valid, target):
    """Status rules (handoff section 6):
       CERTIFIED  iff lam_valid and lam_lo > 0 and R_hi finite; eps = up(R_hi / lam_lo);
                  target_met = (eps <= target)   (exact float comparison; True is rigorous).
       otherwise UNVERIFIED with reason; target_met = None."""
    rec = dict(schema="cars-cert/0.2", matrix_sha256=matrix_sha256(A), rhs_sha256=vec_sha256(b),
               x_sha256=vec_sha256(x), target=float(target).hex())
    try:
        if not lam_valid:
            raise CertFail("LAMBDA_NOT_VALIDATED")
        if not (np.isfinite(lam_lo) and lam_lo > 0):
            raise CertFail("LAMBDA_NONPOSITIVE")
        R_hi = residual_bound(A, b, x)
        eps = float(up(R_hi / lam_lo))
        finite_or_fail(eps, "eps")
        rec.update(status="CERTIFIED", reason=None, R_hi=R_hi.hex(), lambda_lo=float(lam_lo).hex(),
                   eps=eps.hex(), target_met=bool(eps <= float(target)))
    except CertFail as e:
        rec.update(status="UNVERIFIED", reason=e.reason, target_met=None)
    return rec


def check_record_invariants(rec):
    """Consistency checks an independent verifier performs on a record (without recomputing bounds)."""
    if rec["status"] == "CERTIFIED":
        lam = float.fromhex(rec["lambda_lo"]); R = float.fromhex(rec["R_hi"]); eps = float.fromhex(rec["eps"])
        if not (lam > 0 and R >= 0 and np.isfinite(eps)):
            return False
        if eps != float(up(R / lam)):
            return False
        if rec["target_met"] != (eps <= float.fromhex(rec["target"])):
            return False
        return rec["reason"] is None
    return rec["target_met"] is None and rec["reason"] is not None
