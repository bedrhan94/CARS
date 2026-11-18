"""Phase 5, task 5: INDEPENDENT ARITHMETIC checks of CARS certificates (development data only).
Nothing here uses the M0 floating-point primitives (up/dn, gamma bounds, two_sum, norm2_sym_hi) or any Cholesky
library. Arithmetic used:
  * exact rationals (python-flint fmpq, Python Fraction/int) -- RIGOROUS (exact);
  * Arb midpoint-radius ball arithmetic (python-flint arb_mat) -- RIGOROUS (ball enclosures);
  * LAPACK double precision -- NUMERICAL ONLY (tightness context, never a proof).
Checks per target matrix A_t (family, size, dev seed, t), over all repeat-0 certified records of the pilot:
  L  lambda:   lambda_min(A_t) > lam_max := max claimed lambda_lo.
               S0: exact LDL^T of A_t - lam_max I in fmpq (Sylvester's criterion).
               S1/S2: congruence B = V^T (A_t - lam_max I) V with V = LAPACK eigenvectors, evaluated in Arb;
               V nonsingular (||V^T V - I||_inf < 1 in balls) and B strictly diagonally dominant with positive
               diagonal (balls) => B PD => A_t - lam_max I PD (Sylvester's law of inertia).
  E  final error: S0: exact x* = A^{-1} b (fmpq), ||x^ - x*||_2^2 <= eps^2 exactly.
               S1/S2: exact residual r = b - A x^ (integers), ||r||_2^2 <= (eps * lambda_lo)^2 exactly; with L this
               proves ||x^ - x*|| <= ||r|| / lambda_min <= ||r|| / lambda_lo <= eps.
  Z  M2-A ingredients (exact): theta <= min_e c_t/c_s; zeta_t >= ||A^_t - sum_e c_e Z Z^T||_inf (>= ||.||_2, the
               difference is exactly symmetric); same for zeta_s; and the composed bound
               lambda_t^lo <= theta (lambda_s^lo - zeta_s) - zeta_t in exact arithmetic.
  F  M0 factor residual (FRZ anchors): R = P(A - sigma I)P^T - L D L^T; S0 exact, S1 Arb balls;
               rho <= sigma - ||R||_inf (upper) and all d > 0.
Usage: python independent5.py <plan_dir> <family> <size> <seed> <t,t,...> [--cpu k]"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import psutil
from flint import fmpq, fmpq_mat, arb_mat, arb, ctx

P5 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P5.parent / 'PHASE3B_DEV' / 'work' / 'CARS_M0_M1'))
from cars_m2.families import Family3D                       # frozen generator only (problem data)


def Q(x):
    return Fraction(float(x))


def fq(x):
    n, d = float(x).as_integer_ratio(); return fmpq(n, d)


# ------------------------------------------------------------------ L: lambda checks
def exact_pd(A, lam):
    """Exact Gaussian elimination without pivoting on the symmetric rational matrix A - lam I (dense, fmpq)."""
    n = A.shape[0]; M = A.toarray(); l = fq(lam)
    W = [[fq(M[i, j]) for j in range(n)] for i in range(n)]
    for i in range(n):
        W[i][i] -= l
    for k in range(n):
        p = W[k][k]
        if not p > 0:
            return False, k
        rowk = W[k]
        for i in range(k + 1, n):
            f = W[i][k]
            if f != 0:
                f = f / p; Wi = W[i]
                for j in range(k + 1, n):
                    if rowk[j] != 0:
                        Wi[j] -= f * rowk[j]
    return True, n


def arb_congruence_pd(A, lam, prec=64, block=512):
    """Rigorous (ball) proof that A - lam I is PD, via a LAPACK eigenvector congruence (see module doc).
    Column-blocked: B = V^T A V - lam V^T V and G = V^T V - I are formed block by block in Arb; per-row sums of
    |B_ij| (j != i) and |G_ij| are accumulated as balls. Inputs A, V, lam are exact doubles."""
    ctx.prec = prec
    n = A.shape[0]
    w, V = np.linalg.eigh(A.toarray())
    Aa = arb_mat(A.toarray().tolist())
    VaT = arb_mat(np.ascontiguousarray(V.T).tolist())
    la = arb(float(lam))
    zero = arb(0)
    boff = [zero] * n; goff = [zero] * n; diag_B = [None] * n; diag_G = [None] * n
    for j0 in range(0, n, block):
        j1 = min(n, j0 + block); bs = j1 - j0
        VJ = arb_mat(np.ascontiguousarray(V[:, j0:j1]).tolist())
        GJ = VaT * VJ
        BJ = VaT * (Aa * VJ) - GJ * la
        for i in range(n):
            for jj in range(bs):
                j = j0 + jj
                bij = BJ[i, jj]; gij = GJ[i, jj]
                if i == j:
                    diag_B[i] = bij; diag_G[i] = gij
                else:
                    boff[i] = boff[i] + abs(bij); goff[i] = goff[i] + abs(gij)
        del VJ, GJ, BJ
    ok = True; min_marg = None; max_g = zero
    for i in range(n):
        marg = diag_B[i] - boff[i]
        g = abs(diag_G[i] - 1) + goff[i]
        if not (marg > 0) or not (g < 1):
            ok = False
        if min_marg is None or float(marg.mid()) < float(min_marg.mid()):
            min_marg = marg
        if g > max_g:
            max_g = g
    return dict(proved=bool(ok), min_gershgorin_margin=str(min_marg), max_rowsum_VtV_minus_I=str(max_g),
                lapack_lambda1=float(w[0]), prec=prec, block=block)


# ------------------------------------------------------------------ E: final error
def exact_residual_sq(A, b, x):
    """||b - A x||_2^2 exactly (Python integers after scaling every double to a common dyadic exponent)."""
    A = sp.csr_matrix(A)
    tot = Fraction(0)
    xs = [Fraction(float(v)) for v in x]
    for i in range(A.shape[0]):
        s = Fraction(float(b[i]))
        for p in range(A.indptr[i], A.indptr[i + 1]):
            s -= Fraction(float(A.data[p])) * xs[A.indices[p]]
        tot += s * s
    return tot


def exact_solution(A, b):
    n = A.shape[0]; M = A.toarray()
    Am = fmpq_mat(n, n, [fq(M[i, j]) for i in range(n) for j in range(n)])
    bm = fmpq_mat(n, 1, [fq(v) for v in b])
    xs = Am.solve(bm)
    return [Fraction(int(xs[i, 0].p), int(xs[i, 0].q)) for i in range(n)]


# ------------------------------------------------------------------ Z: M2-A ingredients
def exact_model_discrepancy_inf(fam, c, Ahat):
    """max_i sum_j |(A^ - sum_e c_e Z Z^T)_ij| exactly (free dofs only)."""
    Z = fam.Z; m_loc, r = Z.shape
    Zq = [[Q(Z[a, k]) for k in range(r)] for a in range(m_loc)]
    K = [[sum(Zq[a][k] * Zq[b][k] for k in range(r)) for b in range(m_loc)] for a in range(m_loc)]
    D = {}
    dofs = fam.mesh.dofs
    for e in range(dofs.shape[0]):
        ce = Q(c[e]); de = dofs[e]
        for a in range(m_loc):
            i = int(de[a])
            if i < 0:
                continue
            Ka = K[a]
            for bb in range(m_loc):
                j = int(de[bb])
                if j < 0:
                    continue
                D[(i, j)] = D.get((i, j), 0) - ce * Ka[bb]
    Ah = sp.coo_matrix(Ahat)
    for i, j, v in zip(Ah.row, Ah.col, Ah.data):
        D[(int(i), int(j))] = D.get((int(i), int(j)), 0) + Q(v)
    rows = {}
    for (i, j), v in D.items():
        rows[i] = rows.get(i, 0) + abs(v)
    sym = all(D.get((j, i), 0) == v for (i, j), v in D.items())
    return max(rows.values()) if rows else Fraction(0), sym


# ------------------------------------------------------------------ F: M0 factor residual
def factor_residual_inf(A, npz, exact):
    f = np.load(npz)
    L = sp.csr_matrix((f['data'], f['indices'], f['indptr']), shape=tuple(f['shape'])); d = f['d']; perm = f['perm']
    sigma = float(f['sigma']); n = A.shape[0]
    Ad = A.toarray(); Pm = np.zeros((n, n)); Pm[perm, np.arange(n)] = 1.0
    if exact:
        Lq = L.tocoo(); C = {}
        cols = {}
        for i, j, v in zip(Lq.row, Lq.col, Lq.data):
            cols.setdefault(int(j), []).append((int(i), Q(v)))
        for k, lst in cols.items():
            dk = Q(d[k])
            for (i, vi) in lst:
                for (j, vj) in lst:
                    C[(i, j)] = C.get((i, j), 0) + vi * dk * vj
        R = {}
        Aco = sp.coo_matrix(A)
        sq = Q(sigma)
        for i, j, v in zip(Aco.row, Aco.col, Aco.data):
            a, b = int(perm[i]), int(perm[j])
            R[(a, b)] = R.get((a, b), 0) + Q(v) - (sq if i == j else 0)
        for i in range(n):
            if (int(perm[i]), int(perm[i])) not in R:
                R[(int(perm[i]), int(perm[i]))] = -sq
        for key, v in C.items():
            R[key] = R.get(key, 0) - v
        rows = {}
        for (i, j), v in R.items():
            rows[i] = rows.get(i, 0) + abs(v)
        return float(max(rows.values())), Fraction(max(rows.values())), sigma, bool(np.all(d > 0)), 'exact'
    ctx.prec = 200
    Ld = arb_mat(L.toarray().tolist()); Dd = arb_mat([[float(d[i]) if i == j else 0.0 for j in range(n)] for i in range(n)])
    C = Ld * Dd * Ld.transpose()
    Ba = arb_mat((Pm @ Ad @ Pm.T).tolist())
    s = arb(sigma); mx = arb(0)
    for i in range(n):
        row = arb(0)
        for j in range(n):
            rij = Ba[i, j] - (s if i == j else 0) - C[i, j]
            row += abs(rij)
        if row > mx:
            mx = row
    return float(mx.upper()), None, sigma, bool(np.all(d > 0)), 'arb'


# ------------------------------------------------------------------ driver
def collect(plan_dir, fam_id, size, seed, t):
    cell = Path(plan_dir) / f'{fam_id}_{size}_s{seed}'
    out = []
    for eng_dir in sorted(d for d in cell.iterdir() if d.is_dir()):
        for f in sorted(eng_dir.glob('*_r0.jsonl')):
            recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            cfg = '|'.join((recs[0].get('engine', 'FRZ') if eng_dir.name != 'FRZ' else 'FRZ', recs[0]['method'],
                            recs[0].get('track', 'T1'), recs[0].get('binding', 'frozen')))
            byt = {r['t']: r for r in recs}
            r = byt.get(t)
            if r and r['status'] == 'CERTIFIED':
                out.append((cfg, r, eng_dir, byt))
    return out


def main():
    plan_dir, fam_id, size, seed, ts = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), [int(x) for x in sys.argv[5].split(',')]
    if '--cpu' in sys.argv:
        psutil.Process().cpu_affinity([int(sys.argv[sys.argv.index('--cpu') + 1])])
    if seed not in (0, 1, 2):
        raise SystemExit('dev seeds only')
    fam = Family3D(fam_id, size, seed=seed, split='dev')
    results = []
    for t in ts:
        t0 = time.perf_counter()
        A, b, meta = fam.get(t); n = A.shape[0]
        recs = collect(plan_dir, fam_id, size, seed, t)
        lam_max = max(float.fromhex(r['lambda_lo']) for _, r, _, _ in recs)
        res = dict(family=fam_id, size=size, seed=seed, t=t, n=n, records=len(recs), lam_max=lam_max.hex())
        # L
        if n <= 200:
            ok, k = exact_pd(A, lam_max); res['L'] = dict(method='exact-rational LDLT (fmpq)', proved=ok, rigour='rigorous-exact')
        else:
            res['L'] = dict(method='Arb congruence with LAPACK eigenvectors', rigour='rigorous-ball', **arb_congruence_pd(A, lam_max))
        res['L']['lapack_lambda1_over_lam_max'] = (float(np.linalg.eigvalsh(A.toarray())[0]) if n <= 200 else res['L']['lapack_lambda1']) / lam_max
        # E
        xstar = exact_solution(A, b) if n <= 200 else None
        E = []
        for cfg, r, eng_dir, byt in recs:
            x = np.load(eng_dir / r['solution_file'], allow_pickle=False); eps = Q(float.fromhex(r['eps']))
            lo = Q(float.fromhex(r['lambda_lo']))
            if xstar is not None:
                err2 = sum((Q(x[i]) - xstar[i]) ** 2 for i in range(n))
                E.append(dict(config=cfg, check='exact ||x^-x*||^2 <= eps^2', ok=bool(err2 <= eps * eps),
                              ratio_err_over_eps=float((err2 ** 0.5) / eps) if eps > 0 else None))
            else:
                r2 = exact_residual_sq(A, b, x)
                E.append(dict(config=cfg, check='exact ||r||^2 <= (eps*lambda_lo)^2 (+ L)',
                              ok=bool(r2 <= (eps * lo) ** 2 and lo <= Q(lam_max) and res['L']['proved'])))
        res['E'] = dict(rigour='rigorous-exact' + ('' if xstar is not None else ' (+L)'), all_ok=all(e['ok'] for e in E), checks=E)
        # Z
        Zc = []; zcache = {}
        zt_inf, sym_t = exact_model_discrepancy_inf(fam, meta['c'], A)
        for cfg, r, eng_dir, byt in recs:
            ch = r['lambda_chain'][0]
            if ch.get('type') != 'm2a':
                continue
            s = int(ch['anchor_t']); As, _, ms = fam.get(s)
            if s not in zcache:
                zcache[s] = exact_model_discrepancy_inf(fam, ms['c'], As)
            zs_inf, sym_s = zcache[s]
            theta = Q(float.fromhex(ch['theta'])); zeta_s = Q(float.fromhex(ch['zeta_s'])); zeta_t = Q(float.fromhex(ch['zeta_t']))
            ratios = [Q(ct) / Q(cs) for ct, cs in zip(meta['c'], ms['c'])]
            lam_s = Q(float.fromhex(byt[s]['lambda_lo'])) if byt.get(s) and byt[s]['status'] == 'CERTIFIED' else None
            comp = theta * (lam_s - zeta_s) - zeta_t if lam_s is not None else None
            Zc.append(dict(config=cfg, anchor_t=s, theta_ok=bool(theta <= min(ratios)), zeta_t_ok=bool(zt_inf <= zeta_t and sym_t),
                           zeta_s_ok=bool(zs_inf <= zeta_s and sym_s), composition_ok=(bool(Q(float.fromhex(r['lambda_lo'])) <= comp) if comp is not None else None),
                           zeta_t_exact_over_bound=float(zt_inf / zeta_t) if zeta_t > 0 else None))
        res['Z'] = dict(rigour='rigorous-exact', checks=Zc, all_ok=all(z['theta_ok'] and z['zeta_t_ok'] and z['zeta_s_ok'] and z['composition_ok'] is not False for z in Zc))
        # F (FRZ C-per-1F anchor at this t)
        Fc = []
        for cfg, r, eng_dir, byt in recs:
            if cfg == 'FRZ|C-per-1F|T1|frozen' and n <= 2500:
                npz = eng_dir / r['lambda_chain'][0]['factor_file']
                rinf, rq, sigma, dpos, how = factor_residual_inf(A, npz, exact=(n <= 200))
                rho = Q(float.fromhex(r['lambda_lo']))
                ok = dpos and (rho <= Q(sigma) - (rq if rq is not None else Q(rinf)))
                Fc.append(dict(config=cfg, how=how, R_inf_upper=rinf, sigma=sigma, rho=float(rho), d_positive=dpos, ok=bool(ok)))
        res['F'] = dict(rigour='rigorous-exact (S0) / rigorous-ball (S1)', checks=Fc, all_ok=all(f['ok'] for f in Fc))
        res['wall_s'] = time.perf_counter() - t0
        results.append(res)
        print(json.dumps({k: (v if k not in ('E', 'Z', 'F') else {'all_ok': v['all_ok'], 'n': len(v['checks'])}) for k, v in res.items()}), flush=True)
    outp = P5 / 'arith' / f'indep_{fam_id}_{size}_s{seed}_t{"-".join(map(str, ts))}.json'
    outp.write_text(json.dumps(results, indent=1, default=str))


if __name__ == '__main__':
    main()
