"""Phase 10A: exact-arithmetic validation of the EFT residual bound, with negative controls (mutants).

For every case the exact residual ||b - A x||_2^2 is computed in rational arithmetic (python-flint fmpq; every binary64
converted exactly) and compared with R^2 (exact). A case VIOLATES a bound if R^2 < exact.
  real bound   must never be violated (hard stop otherwise)
  mutants      deliberately wrong variants; the suite must catch them, demonstrating its power:
               float_residual (plain ||fl(b - Ax)||), no_product_errors (drop TwoProduct errors), no_gamma2 (drop the
               Sum2 error term), no_div_1mu, no_Se_bound
Generators: G1 consistent rhs b = fl(A x); G2 near-solution x = solve(A, b); G3 multi-level cancellation rows
(condition up to ~2^200); G4 underflow (TwoProduct precondition fails -> frozen fallback); G5 mixed magnitudes;
G6 three-scale rows (M, m, tiny, -M, -m) in random order, where Sum2 itself can return 0 for a nonzero sum.
Usage: python eft_exact_tests.py -> eft_exact_tests.json   (hard-stop flag in the output if any real violation)"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, math, time
from fractions import Fraction
from pathlib import Path
import numpy as np
import scipy.sparse as sp
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CARS / 'PHASE8A_DEV' / 'g0'))
sys.path.insert(0, str(CARS / 'PHASE3B_DEV' / 'work' / 'CARS_M0_M1'))
from cars import _reference_m0 as v       # noqa: E402
import eft_residual as EFT                # noqa: E402
from exact_tools import Q, fq             # noqa: E402

MUTANTS = ('float_residual', 'no_product_errors', 'no_gamma2', 'no_div_1mu', 'no_Se_bound')


def exact_r2(A, b, x):
    A = sp.csr_matrix(A); s = Q(0)
    xq = [fq(t) for t in x.tolist()]
    for i in range(A.shape[0]):
        z = fq(b[i])
        for p in range(A.indptr[i], A.indptr[i + 1]):
            z -= fq(A.data[p]) * xq[A.indices[p]]
        s += z * z
    return s


def rand_spd(rng, n, dens):
    B = sp.random(n, n, density=dens, random_state=int(rng.integers(1 << 30)), data_rvs=lambda k: rng.standard_normal(k))
    A = (B @ B.T + sp.identity(n) * float(rng.uniform(0.01, 2.0))).tocsr()
    A = sp.csr_matrix(np.triu(A.toarray()) + np.triu(A.toarray(), 1).T)
    A.eliminate_zeros(); A.sort_indices()
    return A


def gen(kind, rng):
    n = int(rng.integers(3, 40))
    if kind == 'G1':
        A = rand_spd(rng, n, 0.3); x = rng.standard_normal(n) * 10.0 ** rng.uniform(-3, 3)
        b = A @ x
    elif kind == 'G2':
        A = rand_spd(rng, n, 0.3); b = rng.standard_normal(n)
        x = np.linalg.solve(A.toarray(), b)
    elif kind == 'G3':
        n = int(rng.integers(6, 14)); D = np.zeros((n, n)); x = np.zeros(n)
        x[:] = rng.standard_normal(n) * 2.0 ** rng.integers(-20, 20, n)
        x[x == 0] = 1.0
        b = np.zeros(n)
        for i in range(n):
            free = list(range(n)); rng.shuffle(free)
            lev = free[:4]; rest = free[4:]
            for j in rest:
                D[i, j] = float(rng.standard_normal() * 2.0 ** int(rng.integers(0, 120)))
            P = sum((fq(D[i, j]) * fq(x[j]) for j in rest), Q(0))
            for j in lev:                                   # four cancellation levels
                a = -P / fq(x[j])
                D[i, j] = float(Fraction(int(a.p), int(a.q)))
                P += fq(D[i, j]) * fq(x[j])
            b[i] = 0.0                                      # r_i = -P: tiny remainder, condition ~ 2^200
        A = sp.csr_matrix(D); A.sort_indices()
    elif kind == 'G4':
        A = rand_spd(rng, n, 0.3) * 2.0 ** -600; x = rng.standard_normal(n) * 2.0 ** -400; b = rng.standard_normal(n) * 2.0 ** -1000
        A = sp.csr_matrix(A)
    elif kind == 'G6':                                      # three-scale rows: Sum2 can lose the smallest scale
        n = int(rng.integers(3, 10)); m = 5; D = np.zeros((n, n * m)); x = np.ones(n * m); b = np.zeros(n)
        for i in range(n):
            M = 2.0 ** int(rng.integers(60, 200)); m1 = float(rng.standard_normal()); tiny = float(rng.standard_normal()) * 2.0 ** -int(rng.integers(60, 200))
            vals = [M, m1, tiny, -M, -m1]; order = rng.permutation(m)
            for k, o in enumerate(order):
                D[i, i * m + k] = vals[o]
        A = sp.csr_matrix(D); A.sort_indices()
    else:  # G5 mixed magnitudes per row
        A = rand_spd(rng, n, 0.4).toarray()
        sc = 2.0 ** rng.integers(-300, 300, n)
        A = sp.csr_matrix(A * sc[:, None]); x = rng.standard_normal(n) * 2.0 ** rng.integers(-100, 100, n)
        b = A @ x + rng.standard_normal(n) * 2.0 ** -200
    return sp.csr_matrix(A), np.asarray(b, float), np.asarray(x, float)


def main():
    t0 = time.perf_counter()
    rng = np.random.default_rng(20261114)
    per = {'G1': 400, 'G2': 400, 'G3': 400, 'G4': 100, 'G5': 300, 'G6': 400}
    out = {'per_kind': {}, 'violations': [], 'hard_stop': False}
    for kind, cnt in per.items():
        st = dict(cases=0, eft=0, fallback=0, violations=0, mutant_caught={m: 0 for m in MUTANTS},
                  ratio_eft_min=None, ratio_eft_max=None, ratio_frozen_median=None)
        ratios = []; frz = []
        for c in range(cnt):
            A, b, x = gen(kind, rng)
            if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
                continue
            st['cases'] += 1
            ex = exact_r2(A, b, x)
            try:
                R, info = EFT.eft_residual_bound(A, b, x)
            except v.CertFail as e:
                st.setdefault('certfail', {}); st['certfail'][e.reason] = st['certfail'].get(e.reason, 0) + 1
                continue
            st['eft' if info['kind'] == 'eft' else 'fallback'] += 1
            if fq(R) ** 2 < ex:
                st['violations'] += 1; out['hard_stop'] = True
                out['violations'].append(dict(kind=kind, case=c, R=R, exact_sq=float(Fraction(int(ex.p), int(ex.q)))))
            if ex > 0:
                exr = math.sqrt(float(Fraction(int(ex.p), int(ex.q))))
                if exr > 0:
                    ratios.append(R / exr)
                    try:
                        frz.append(float(v.residual_bound(A, b, x)) / exr)
                    except v.CertFail:
                        pass
            if info['kind'] == 'eft':
                for m in MUTANTS:
                    Rm, _ = EFT.eft_residual_bound(A, b, x, mutant=m)
                    if fq(Rm) ** 2 < ex:
                        st['mutant_caught'][m] += 1
        if ratios:
            st['ratio_eft_min'] = float(min(ratios)); st['ratio_eft_max'] = float(max(ratios))
            st['ratio_eft_median'] = float(np.median(ratios))
        if frz:
            st['ratio_frozen_median'] = float(np.median(frz)); st['ratio_frozen_max'] = float(max(frz))
        out['per_kind'][kind] = st
        print(kind, json.dumps(st), flush=True)
    out['total_cases'] = sum(s['cases'] for s in out['per_kind'].values())
    out['total_violations'] = sum(s['violations'] for s in out['per_kind'].values())
    out['mutants_caught_total'] = {m: sum(s['mutant_caught'][m] for s in out['per_kind'].values()) for m in MUTANTS}
    out['seconds'] = time.perf_counter() - t0
    (HERE / 'eft_exact_tests.json').write_text(json.dumps(out, indent=1))
    print('TOTAL', out['total_cases'], 'violations', out['total_violations'], 'mutants', out['mutants_caught_total'],
          'hard_stop', out['hard_stop'], 'seconds', round(out['seconds'], 1))


if __name__ == '__main__':
    main()
