"""Phase 10A: rigorous accurate residual bound R_acc >= ||b - A x||_2 (exact reals) for binary64 A (CSR), b, x.

Derivation (all quantities per row i; N = K + 2 terms where K = max stored entries per row):
  1. TwoProduct (Dekker / Veltkamp splitting, 134217729 = 2^27 + 1):  a_ij x_j = p_ij + e_ij EXACTLY, provided
       (P1) all inputs finite and |a|, |x| < 2^995 (no overflow in the splitting), and
       (P2) for every product: p_ij == 0 only if a_ij == 0 or x_j == 0, and otherwise |p_ij| > 2^-969
            (then e_a + e_b >= e_min + p - 1, so every partial product and the error term are exactly representable).
  2. S_e = fl(sum_j e_ij) in any order:  |S_e - sum_j e_ij| <= gamma_{K-1} sum_j |e_ij|   (additions are exact in the
     subnormal range, so no underflow term).
  3. Sum2 (Ogita, Rump, Oishi 2005, Alg. 4.4 / Prop. 4.5) over the N terms (b_i, -p_i1, ..., -p_iK, -S_e):
       |res_i - s_i| <= u |s_i| + gamma_{N-1}^2 T_i,   s_i = b_i - sum_j p_ij - S_e,   T_i = sum |terms|.
     Hence |s_i| <= (|res_i| + gamma_{N-1}^2 T_i) / (1 - u).
  4. |r_i| <= |s_i| + gamma_{K-1} sum_j |e_ij|,  r_i = b_i - sum_j a_ij x_j exactly.
  5. v_i := up( up(|res_i| + up(g2 T_i)) / dn(1-u) ) + up(gK * Eabs_i) + N*eta  (the eta term is a harmless safety
     allowance); R_acc = up( sqrt( sum_nonneg_hi( up(v*v) ) ) ) >= ||r||_2.
Every bound operation is round-to-nearest followed by nextafter towards +inf (Lemma V1 of M0), so it is >= the exact
value of the expression it encloses. If (P1) or (P2) fails, the function falls back to the FROZEN bound
(cars._reference_m0.residual_bound), which is always valid; the fallback is recorded.
"""
import numpy as np
import scipy.sparse as sp
from cars import _reference_m0 as v

U = 2.0 ** -53
ETA = 2.0 ** -1074
SPLITTER = 134217729.0
TP_MIN = 2.0 ** -969
MAXABS = 2.0 ** 995


def two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def split(a):
    c = SPLITTER * a
    h = c - (c - a)
    return h, a - h


class EftLayout:
    """ELL view of a fixed CSR pattern (built once per pattern; values are re-gathered for every matrix)."""

    def __init__(self, A):
        A = sp.csr_matrix(A)
        self.n = A.shape[0]
        self.indptr = A.indptr.copy(); self.indices = A.indices.copy()
        cnt = np.diff(A.indptr)
        self.K = int(cnt.max()) if cnt.size else 0
        self.mask = np.arange(self.K)[None, :] < cnt[:, None]
        self.idx = np.zeros((self.n, self.K), np.int64)
        self.idx[self.mask] = A.indices

    def matches(self, A):
        return (A.shape[0] == self.n and np.array_equal(A.indptr, self.indptr)
                and np.array_equal(A.indices, self.indices))


def eft_residual_bound(A, b, x, layout=None, mutant=None):
    """Returns (R, info). info['kind'] in {'eft', 'frozen-fallback'}. `mutant` is for negative controls ONLY
    (deliberately wrong variants used to test the power of the exact checks); never used by a method."""
    A = v.canonical(A)
    b = np.asarray(b, dtype=np.float64); x = np.asarray(x, dtype=np.float64)
    reason = None
    if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
        reason = 'NONFINITE_INPUT'
    elif (A.data.size and np.max(np.abs(A.data)) >= MAXABS) or (x.size and np.max(np.abs(x)) >= MAXABS):
        reason = 'SPLIT_OVERFLOW_RISK'
    if reason is None:
        if layout is None or not layout.matches(A):
            layout = EftLayout(A)
        n, K = layout.n, layout.K
        VAL = np.zeros((n, K)); VAL[layout.mask] = A.data
        X = x[layout.idx] * layout.mask                 # padded slots: value 0 * x = 0 (masked)
        P = VAL * X
        ah, al = split(VAL); bh, bl = split(X)
        E = al * bl - (((P - ah * bh) - al * bh) - ah * bl)
        nz_in = (VAL != 0) & (X != 0)
        bad = (nz_in & (P == 0)) | ((P != 0) & (np.abs(P) <= TP_MIN))
        if bad.any():
            reason = 'TWOPRODUCT_UNDERFLOW'
        elif not (np.all(np.isfinite(P)) and np.all(np.isfinite(E))):
            reason = 'NONFINITE_PRODUCT'
    if reason is not None:
        if mutant is not None:
            raise ValueError('mutant on fallback path')
        return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': reason}
    if mutant == 'float_residual':
        r = b - A @ x
        return float(np.linalg.norm(r)), {'kind': 'mutant'}
    if mutant == 'no_product_errors':
        E = np.zeros_like(E)
    Se = E.sum(axis=1)
    Eabs = np.abs(E).sum(axis=1)
    s = b.copy(); sig = np.zeros(n)
    for k in range(K):
        s, t = two_sum(s, -P[:, k]); sig = sig + t
    s, t = two_sum(s, -Se); sig = sig + t
    res = s + sig
    if not np.all(np.isfinite(res)):
        return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': 'NONFINITE_SUM'}
    N = K + 2
    gN = v.gamma_hi(float(max(N - 1, 1))); g2 = v.up(gN * gN)
    gK = v.gamma_hi(float(max(K - 1, 1)))
    # T_i >= |b_i| + sum_j |p_ij| + |S_e|  (sum of |P| in any order: exact <= fl / (1 - gamma_{K-1}))
    Pabs = v.up(np.abs(P).sum(axis=1) / v.dn(1.0 - gK))
    T = v.up(v.up(np.abs(b) + Pabs) + np.abs(Se))
    Eabs_hi = v.up(Eabs / v.dn(1.0 - gK))
    term_g2 = v.up(g2 * T) if mutant != 'no_gamma2' else np.zeros(n)
    num = v.up(np.abs(res) + term_g2)
    vi = v.up(num / v.dn(1.0 - U)) if mutant != 'no_div_1mu' else num
    if mutant != 'no_Se_bound':
        vi = v.up(vi + v.up(gK * Eabs_hi))
    vi = v.up(vi + N * ETA)
    R = float(v.up(np.sqrt(v.sum_nonneg_hi(v.up(vi * vi)))))
    if not np.isfinite(R):
        return float(v.residual_bound(A, b, x)), {'kind': 'frozen-fallback', 'reason': 'NONFINITE_BOUND'}
    return R, {'kind': 'eft' if mutant is None else 'mutant', 'N': N}


def certificate_eft(A, b, x, lam_lo, target, layout=None):
    """Certificate record with the EFT residual bound (same status rules as the frozen certificate)."""
    rec = dict(schema='cars-cert/0.4-eft', matrix_sha256=v.matrix_sha256(A), rhs_sha256=v.vec_sha256(b),
               x_sha256=v.vec_sha256(x), target=float(target).hex())
    try:
        if not (np.isfinite(lam_lo) and lam_lo > 0):
            raise v.CertFail('LAMBDA_NONPOSITIVE')
        R, info = eft_residual_bound(A, b, x, layout)
        eps = float(v.up(R / lam_lo))
        v.finite_or_fail(eps, 'eps')
        rec.update(status='CERTIFIED', reason=None, R_hi=float(R).hex(), lambda_lo=float(lam_lo).hex(), eps=eps.hex(),
                   target_met=bool(eps <= float(target)), residual_kind=info['kind'],
                   residual_fallback_reason=info.get('reason'))
    except v.CertFail as e:
        rec.update(status='UNVERIFIED', reason=e.reason, target_met=None)
    return rec
