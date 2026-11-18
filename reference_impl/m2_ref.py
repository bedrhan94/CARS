"""
CARS Phase 3 -- minimal executable reference of the M2 certificate formulas.  NOT production code.

  M2-A  gray-box structural transfer: min-theta and split-theta with Weyl remainder, validated assembly discrepancy
  M2-B  shift-reserve Schur-complement transfer (SRI): two-shift anchor, validated forward solves,
        enclosure of G_JJ, square-root-free robust test (c G + G E G >= 0)
  M2-C  rank-one LDL^T modification (Bennett form) with a running a-posteriori residual bound  [optional arm]

Normative statements: research/15_m2_mathematical_spec.md, research/16_m2_numerical_validity.md,
research/18_downdating_reassessment.md.  M0 primitives are imported unchanged from cars_ref.py.
Every function returns a bound in the stated direction or raises CertFail (fail closed).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cars_ref import (U, ETA, CertFail, up, dn, gamma_hi, two_sum, sum_nonneg_hi,  # noqa: E402
                      canonical, residual_bound, norm2_sym_hi, anchor_inertia, finite_or_fail)

GUARD_MIN = 2.0 ** -1021      # products/quotients with |result| below this (and nonzero operands) -> fail closed
RANGE_LO, RANGE_HI = 2.0 ** -500, 2.0 ** 500


def norm_hi(x):
    """Upper bound of ||x||_2 for a float array."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0:
        return 0.0
    return float(up(np.sqrt(sum_nonneg_hi(up(x * x)))))


def _guard_prod(res, *ops):
    """Underflow guard for a product/quotient: operands nonzero but |result| < 2^-1021 -> fail."""
    nz = np.ones(np.shape(res), bool)
    for o in ops:
        nz &= (np.asarray(o) != 0)
    if np.any(nz & (np.abs(res) < GUARD_MIN)):
        raise CertFail("UNDERFLOW_GUARD")
    finite_or_fail(res, "guard")


def _sym_majorant_norm(M):
    """||X||_2 <= norm2_sym_hi(M) for a SYMMETRIC X with |X| <= M entrywise (M dense or sparse)."""
    return norm2_sym_hi(sp.csr_matrix(np.asarray(M) if not sp.issparse(M) else M))


# ===================================================================== M2-A
def assembly_discrepancy(mesh, Z, c, Ahat=None):
    """zeta >= || Ahat - A_exact ||_2, A_exact = sum_e c_e Z Z^T (free dofs), Ahat = fem3d_ref.assemble_float.
    Own-assembly mode (Ahat None): a priori bound gamma_{r+1+m} S + tiny with S = sum_e c_e |Z||Z|^T.
    External mode (Ahat given): adds the validated |Ahat - Atilde| (Atilde = own assembly)."""
    from fem3d_ref import assemble_float
    c = np.asarray(c, dtype=np.float64)
    if np.any(~np.isfinite(c)) or np.any(c < 0):
        raise CertFail("TRANSFER_DATA_INVALID", "coefficients must be finite and >= 0")
    r = Z.shape[1]
    m_max = 8.0                                               # max elements sharing a node pair (hexahedra)
    g = gamma_hi(r + 1.0 + m_max)
    Sabs = assemble_float(mesh, np.abs(Z), c)                 # fl(sum_e c_e |Z||Z|^T): nonneg, same op order
    tiny = up(m_max * up(up(float(c.max(initial=0.0)) * r) + 1.0) * ETA)
    S_hi = Sabs.copy(); S_hi.data = up(up(S_hi.data + tiny) / dn(1.0 - g))
    Psi = S_hi.copy(); Psi.data = up(up(g * Psi.data) + tiny)
    if Ahat is not None:
        At = assemble_float(mesh, Z, c)
        D = canonical(canonical(Ahat) - At)
        D.data = up(np.abs(D.data) / dn(1.0 - U))
        Psi = canonical(Psi + D); Psi.data = up(Psi.data)
    return _sym_majorant_norm(Psi)


def assembly_discrepancy_rowsum(mesh, Z, c):
    """Fast a-priori variant (own-assembly mode ONLY; Ahat must come from fem3d_ref.assemble_float):
    zeta <= rho(Psi) <= max_i (Psi 1)_i,  Psi = gamma_{r+1+m} S + tiny,  (S 1)_i <= sum_{e ∋ i} c_e (|Z||Z|^T 1)_{a(e,i)}.
    Cost O(ne * dofs_per_element); no assembly of a majorant."""
    c = np.asarray(c, dtype=np.float64)
    if np.any(~np.isfinite(c)) or np.any(c < 0):
        raise CertFail("TRANSFER_DATA_INVALID", "coefficients must be finite and >= 0")
    m_loc, r = Z.shape
    absZ = np.abs(Z)
    s = absZ.T @ np.ones(m_loc)                                     # r nonneg sums of m_loc terms
    rho_loc = absZ @ s                                              # m_loc nonneg sums of r terms
    g_loc = gamma_hi(float(m_loc + r))
    rho_hi = up(up(rho_loc + up((m_loc + r) * ETA)) / dn(1.0 - g_loc))   # >= exact (|Z||Z|^T 1)_a
    vals = (c[:, None] * rho_hi[None, :]).ravel()
    rows = mesh.dofs.ravel()
    keep = rows >= 0
    R = np.bincount(rows[keep], weights=vals[keep], minlength=mesh.n)  # <= 8 nonneg terms per dof
    m_max = 8.0
    R_hi = up(up(R + up(m_max * ETA)) / dn(1.0 - gamma_hi(m_max + 1.0)))
    g = gamma_hi(r + 1.0 + m_max)
    tiny = up(m_max * up(up(float(c.max(initial=0.0)) * r) + 1.0) * ETA)
    row_nnz_max = 27.0 * mesh.k                                     # hexahedral meshes: <= 27 nodes per row
    return float(up(up(g * float(R_hi.max(initial=0.0))) + up(row_nnz_max * tiny)))


def theta_lo(c_s, c_t):
    """theta <= min_e c_t/c_s exactly (requires c_s > 0, c_t >= 0)."""
    c_s, c_t = np.asarray(c_s, float), np.asarray(c_t, float)
    if np.any(c_s <= 0) or np.any(c_t < 0) or not (np.isfinite(c_s).all() and np.isfinite(c_t).all()):
        raise CertFail("TRANSFER_DATA_INVALID", "need c_s > 0, c_t >= 0")
    q = c_t / c_s
    q = np.where(c_t == 0, 0.0, dn(q))
    return float(max(0.0, q.min()))


def m2a_transfer(mesh, Z, c_s, c_t, lam_s_lo, zeta_s, zeta_t, theta0=None):
    """lambda_1(Ahat_t) >= theta0*(lam_s_lo - zeta_s) - wmax - zeta_t   (split form; plain min-theta: theta0=None).
    lam_s_lo <= lambda_1(Ahat_s) (any certificate), zeta_* >= ||Ahat_* - A_*_exact||_2.
    Split: D = {e: c_t < up(theta0*c_s)}; W = sum_{e in D} (theta0 c_s - c_t)^+ K_e  (PSD); wmax >= ||W||_2."""
    from fem3d_ref import assemble_float
    c_s, c_t = np.asarray(c_s, float), np.asarray(c_t, float)
    base = dn(float(lam_s_lo) - float(zeta_s))
    if theta0 is None:
        th = theta_lo(c_s, c_t)
        wmax = 0.0
    else:
        th = float(theta0)
        if not (th >= 0 and np.isfinite(th)):
            raise CertFail("TRANSFER_DATA_INVALID", "theta0")
        prod_hi = up(th * c_s)
        inD = c_t < prod_hi
        kappa = np.where(inD, up(prod_hi - c_t), 0.0)        # >= (theta0 c_s - c_t)^+ exactly
        if np.any(inD):
            r = Z.shape[1]
            Wabs = assemble_float(mesh, np.abs(Z), kappa)      # fl(sum kappa_e |Z||Z|^T), nonneg
            g = gamma_hi(r + 1.0 + 8.0)
            tiny = up(8.0 * up(up(float(kappa.max()) * r) + 1.0) * ETA)
            Wabs.data = up(up(Wabs.data + tiny) / dn(1.0 - g))   # >= exact |W| majorant
            wmax = _sym_majorant_norm(Wabs)
        else:
            wmax = 0.0
    lam = dn(dn(dn(th * base) - wmax) - float(zeta_t))
    return float(finite_or_fail(lam, "m2a")), dict(theta=th, wmax=float(wmax))


# ===================================================================== M2-B
def sri_anchor(As, sigma_hi, sigma_lo):
    """Two-shift anchor. Returns factor of As - sigma_lo I (kept), delta_lo, lam_s_lo = rho_hi, and
    ell <= lambda_min(N), N = P^T L D L^T P.  Raises SRI_NO_MARGIN unless ell > 0."""
    hi = anchor_inertia(As, sigma_hi, 0)
    lo = anchor_inertia(As, sigma_lo, 0)
    if hi["nu"] != 0 or lo["nu"] != 0:
        raise CertFail("ANCHOR_INERTIA_MISMATCH")
    ell = dn(dn(hi["rho"] - lo["sigma"]) - lo["delta"])
    if not ell > 0:
        raise CertFail("SRI_NO_MARGIN", f"ell={ell}")
    return dict(factor=lo["factor"], sigma=lo["sigma"], delta=lo["delta"], lam_s_lo=hi["rho"],
                ell=float(ell), delta_hi=hi["delta"])


def exact_difference(As, At):
    """E = At - As exactly as (Ehi + Elo) on the union pattern (TwoSum). Returns sorted J and dense blocks."""
    As, At = canonical(As), canonical(At)
    pat = canonical((As != 0).astype(np.float64) + (At != 0).astype(np.float64))
    rows, cols = pat.nonzero()
    a = np.asarray(As[rows, cols]).ravel(); b = np.asarray(At[rows, cols]).ravel()
    s, e = two_sum(b, -a)
    nz = (s != 0) | (e != 0)
    rows, cols, s, e = rows[nz], cols[nz], s[nz], e[nz]
    J = np.unique(np.concatenate([rows, cols]))
    pos = {j: i for i, j in enumerate(J)}
    k = len(J)
    Ehi = np.zeros((k, k)); Elo = np.zeros((k, k))
    for i, j, x, y in zip(rows, cols, s, e):
        Ehi[pos[i], pos[j]] = x; Elo[pos[i], pos[j]] = y
    return J, Ehi, Elo


def sri_columns(anchor, J):
    """Ytilde ~ L^{-1} U_J (permuted coords) and Wf >= ||U_J - L Ytilde||_2 (validated, Frobenius)."""
    L, d, perm = anchor["factor"]["L"], anchor["factor"]["d"], anchor["factor"]["perm"]
    Lc = canonical(L)
    n, k = L.shape[0], len(J)
    Uj = np.zeros((n, k))
    Uj[perm[np.asarray(J)], np.arange(k)] = 1.0
    Y = spla.spsolve_triangular(Lc, Uj, lower=True, unit_diagonal=True)
    Y = np.asarray(Y, dtype=np.float64).reshape(n, k)
    finite_or_fail(Y, "Y")
    w2 = 0.0
    for c in range(k):
        R = residual_bound(Lc, Uj[:, c], Y[:, c])
        w2 = up(w2 + up(R * R))
    return Y, float(up(np.sqrt(w2)))


def sri_gram(Y, d, Wf, ell):
    """Ghat (k x k float, exactly symmetric) and DeltaG >= ||G - Ghat||_2 where G = U_J^T N^{-1} U_J (exact)."""
    rows = np.nonzero(np.any(Y != 0, axis=1))[0]
    Yr, dr = Y[rows], d[rows]
    m = float(max(len(rows), 1))
    V = Yr / dr[:, None]
    _guard_prod(V, Yr, np.broadcast_to(dr[:, None], Yr.shape))
    G = V.T @ Yr
    G = np.triu(G) + np.triu(G, 1).T
    M = np.abs(V).T @ np.abs(Yr)                                   # fl of nonneg dot products
    g = gamma_hi(m + 1.0)
    S_hi = up(up(M + up(m * ETA)) / dn(1.0 - g))                  # >= sum_i |y_ia||y_ib|/d_i
    Ebar = up(up(g * S_hi) + up(m * ETA))                          # >= |Ghat - Y^T D^-1 Y|
    gG = _sym_majorant_norm(up(np.abs(G) + Ebar))                 # >= ||Y^T D^-1 Y||_2
    omega = up(Wf / dn(np.sqrt(ell)))
    DeltaG = up(_sym_majorant_norm(Ebar) + up(up(2.0 * up(up(np.sqrt(gG)) * omega)) + up(omega * omega)))
    return G, float(DeltaG)


def sri_test(G, DeltaG, Ehi, Elo, c):
    """Certify Q(c) = c*G + G E G >= 0 for every G with ||G - Ghat|| <= DeltaG, E = Ehi + Elo exactly.
    Returns margin_lo (> 0 means certified). c is a float in (0, 1]."""
    k = G.shape[0]
    for X in (G, Ehi):
        x = np.abs(X[X != 0])
        if x.size and (x.min() < RANGE_LO or x.max() > RANGE_HI):
            raise CertFail("RANGE_GUARD")
    T1 = Ehi @ G
    Q = c * G + G @ T1
    Q = np.triu(Q) + np.triu(Q, 1).T
    S = c * np.abs(G) + np.abs(G) @ (np.abs(Ehi) @ np.abs(G))
    g = gamma_hi(2.0 * k + 2.0)
    tinyQ = up((2.0 * k + 2.0) * ETA)                                 # underflow allowance (cancellation in T1)
    S_hi = up(up(S + tinyQ) / dn(1.0 - g))
    errQ = _sym_majorant_norm(up(up(g * S_hi) + tinyQ))              # >= ||Qhat - Q0||, Q0 = cGh + Gh Ehi Gh
    gh = _sym_majorant_norm(np.abs(G))
    eh = _sym_majorant_norm(np.abs(Ehi))
    el = _sym_majorant_norm(np.abs(Elo)) if np.any(Elo) else 0.0
    gplus = up(gh + DeltaG)
    xi = up(up(up(abs(c) * DeltaG) + up(up(2.0 * eh) * up(gh * DeltaG))) +
            up(up(eh * up(DeltaG * DeltaG)) + up(el * up(gplus * gplus))))
    lam_est = float(np.linalg.eigvalsh(Q)[0])
    if not lam_est > 0:
        return -1.0
    try:
        aq = anchor_inertia(sp.csr_matrix(Q), 0.9 * lam_est, 0)
    except CertFail:
        return -1.0
    return float(dn(dn(aq["rho"] - errQ) - xi))


def sri_certify(anchor, As, At, c_try=None):
    """Full M2-B step: lambda_1(At) >= sigma - delta + f*ell with f = dn(1 - c), c from a heuristic search.
    Returns (lam_lo, info). Raises CertFail('SRI_TEST_FAILED') if not even c = 1 (f = 0) certifies."""
    J, Ehi, Elo = exact_difference(As, At)
    base = dn(anchor["sigma"] - anchor["delta"])
    if len(J) == 0:
        return float(dn(base + dn(anchor["ell"]))), dict(J=0, f=1.0)
    Y, Wf = sri_columns(anchor, J)
    G, DeltaG = sri_gram(Y, anchor["factor"]["d"], Wf, anchor["ell"])
    # heuristic estimate of f* = 1 + min(0, lambda_min(E G)); never part of the proof
    w = np.linalg.eigvals((Ehi + Elo) @ G).real
    fstar = 1.0 + min(0.0, float(w.min()))
    trials = ([c_try] if c_try is not None else []) + [1.0 - 0.98 * fstar, 1.0 - 0.5 * fstar, 1.0]
    for c in trials:
        if not (0 < c <= 1):
            continue
        if sri_test(G, DeltaG, Ehi, Elo, c) > 0:
            f = max(0.0, float(dn(1.0 - c)))
            lam = dn(base + dn(f * anchor["ell"]))
            return float(lam), dict(J=len(J), f=f, c=c, DeltaG=DeltaG, Wf=Wf, fstar_est=fstar)
    raise CertFail("SRI_TEST_FAILED", f"fstar_est={fstar:.3e}")


# ===================================================================== M2-C (optional arm)
def rank1_modify(L, d, alpha, w):
    """Bennett-form rank-one modification of a DENSE unit-lower L and d > 0:
         L D L^T + alpha w w^T  ->  L' D' L'^T  + rho,   ||rho||_2 <= B  (returned, running bound).
    Operation sequence is normative (research/18 section 4). Fails closed on d' <= 0 or underflow."""
    L = np.array(L, dtype=np.float64, copy=True); d = np.array(d, dtype=np.float64, copy=True)
    w = np.array(w, dtype=np.float64, copy=True); a = float(alpha)
    n = d.size
    g1, g2, g4 = gamma_hi(1.0), gamma_hi(2.0), gamma_hi(4.0)
    B = 0.0
    for j in range(n):
        p = w[j]
        if p == 0.0:
            continue
        dj = d[j]
        l = L[j + 1:, j].copy(); wt = w[j + 1:].copy()
        t1 = p * p; _guard_prod(t1, p, p)
        t2 = a * t1; _guard_prod(t2, a, t1)
        dp = dj + t2
        if not (np.isfinite(dp) and dp > 0):
            raise CertFail("UPDOWN_NOT_PD", f"j={j}")
        q = a * p; _guard_prod(q, a, p)
        beta = q / dp; _guard_prod(beta, q, dp)
        gg = a * dj; _guard_prod(gg, a, dj)
        ap = gg / dp; _guard_prod(ap, gg, dp)
        s = p * l; _guard_prod(s, p, l)
        v = wt - s
        mm = beta * v; _guard_prod(mm, beta, v)
        lp = l + mm
        # ---- running bound for this step (all quantities upward) ----
        aa, pp = abs(a), abs(p)
        ap2 = up(aa * up(pp * pp))                                       # >= |a| p^2
        c1 = up(up(g2 * ap2) + up(g1 * dp))
        c2 = up(g2 * up(aa * pp))
        c3 = up(up(up(up(aa * c1) + up(g4 * up(ap2 * aa))) + up(g2 * up(aa * dj))) / dn(dp))
        nl = norm_hi(np.concatenate([[1.0], l]))
        nv = norm_hi(v)
        nlp = norm_hi(np.concatenate([[1.0], lp]))
        fw = up(g1 * up(up(pp * nl) + nv))
        z = up(nv + up(pp * nl))
        fl_ = up(up(U * up(abs(beta) * nv)) + up(g1 * nlp))
        y = up(nlp + fl_)
        bj = up(up(c1 * up(nl * nl)) + up(up(2.0 * c2) * up(nl * nv)))
        bj = up(bj + up(c3 * up(nv * nv)))
        bj = up(bj + up(aa * up(up(2.0 * up(z * fw)) + up(fw * fw))))
        bj = up(bj + up(dp * up(up(2.0 * up(y * fl_)) + up(fl_ * fl_))))
        B = up(B + bj)
        L[j + 1:, j] = lp; d[j] = dp; w[j + 1:] = v; w[j] = 0.0; a = ap
    finite_or_fail(B, "B")
    return L, d, float(B)


def symmetric_to_rank1(Ehi, Elo, J):
    """Exact representation of a symmetric E (= Ehi + Elo on J x J) as sum_k alpha_k w_k w_k^T with
    w_k in {e_i, e_i + e_j, e_i - e_j}. Returns list of (alpha, i, j, sign); updates (alpha>0) first."""
    terms = []
    k = len(J)
    for X in (Ehi, Elo):
        for a in range(k):
            if X[a, a] != 0:
                terms.append((float(X[a, a]), J[a], J[a], 0))
            for b in range(a + 1, k):
                x = X[a, b]
                if x != 0:
                    h = x / 2.0
                    if abs(h) < GUARD_MIN:
                        raise CertFail("UNDERFLOW_GUARD")
                    terms.append((h, J[a], J[b], +1))
                    terms.append((-h, J[a], J[b], -1))
    terms.sort(key=lambda t: t[0] < 0)          # all updates before downdates (stable sort)
    return terms
