"""Revision 3: equivalence of the compiled shift kernels (compiled_full.py) with the frozen runner6 functions
(is_symmetric, diag_positions, the shifted copy, and the whole rebuilt rump_anchor_once).
Matrices: every matrix of the held-out S2 and S3 sequences (both families, seeds 200-204, t = 0..20; already consumed)
and of the development S0/S1 sequences (seeds 0-2). For each matrix the unchanged matrix and five perturbations are
tested: a one-ulp change of one strictly upper entry (asymmetric value), an explicit zero added in one triangle only
(asymmetric pattern), one diagonal entry removed, a NaN on the diagonal, and a symmetric pair with -0.0 / 0.0 values.
Both is_symmetric and diag_positions must return identical results (booleans; position arrays bit for bit).
Writes test_shift.json; exit 1 on any difference."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
import run_cell10b as RC                                             # noqa: E402
N = RC.N; R6 = RC.R6; v = RC.v
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import compiled_full as CF                                           # noqa: E402

CF.install(N, R6, N.EFT, v)
FS, FD = R6.is_symmetric, R6.diag_positions
out = dict(matrices=0, cases=0, identical=0, failures=[], timing_ms={'frozen': 0.0, 'compiled': 0.0})


def canon(M):
    M = sp.csr_matrix(M); M.sort_indices()
    M.has_canonical_format = True
    return M


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.dtype == b.dtype and np.array_equal(a, b)


def check(A, label):
    out['cases'] += 1
    t0 = time.perf_counter(); s1, d1 = FS(A), FD(A); t1 = time.perf_counter()
    s2, d2 = CF.is_symmetric_c(A), CF.diag_positions_c(A); t2 = time.perf_counter()
    out['timing_ms']['frozen'] += 1000 * (t1 - t0); out['timing_ms']['compiled'] += 1000 * (t2 - t1)
    ok = (s1 == s2) and same(d1, d2)
    out['identical'] += ok
    if not ok:
        out['failures'].append(dict(case=label, frozen=[s1, None if d1 is None else int(d1.size)], compiled=[s2, None if d2 is None else int(d2.size)]))


def check_shifted(A, label):
    """shifted_copy_c against the frozen expression At = A.copy(); At.data[pos] = v.dn(At.data[pos] - s_hi)."""
    pos = FD(A)
    if pos is None:
        return
    s_hi = float(10.0 ** rng.uniform(-8, -1)) * float(np.abs(A.data).max())
    At1 = A.copy(); At1.data[pos] = v.dn(At1.data[pos] - s_hi)
    At2 = CF.shifted_copy_c(A, pos, s_hi)
    out['shift_cases'] = out.get('shift_cases', 0) + 1
    ok = (np.array_equal(At1.indptr, At2.indptr) and np.array_equal(At1.indices, At2.indices)
          and np.array_equal(At1.data.view(np.uint64), At2.data.view(np.uint64)))
    out['shift_identical'] = out.get('shift_identical', 0) + ok
    if not ok:
        out['failures'].append(dict(case=label + '|shifted_copy'))


def check_anchor(A, label):
    """Frozen rump_anchor_once against the compiled one: same returned fields and the same factor (bitwise solve)."""
    tm = R6.Timer()
    lam, _ = N._estimate(A, None, None)
    r = rng.standard_normal(A.shape[0])
    a1 = CF._FROZEN['rump_anchor_once'](A, 0.5 * lam, 'RPS', tm); x1 = a1['fac'].solve(r)
    a2 = CF._FROZEN['anchor_c'](A, 0.5 * lam, 'RPS', tm); x2 = a2['fac'].solve(r)
    keys = ('kind', 'engine', 'lam_lo', 'rho', 'sigma', 'c', 's_hi', 'n', 'extra')
    ok = all(a1[k] == a2[k] for k in keys) and np.array_equal(x1.view(np.uint64), x2.view(np.uint64))
    out['anchor_cases'] = out.get('anchor_cases', 0) + 1
    out['anchor_identical'] = out.get('anchor_identical', 0) + ok
    if not ok:
        out['failures'].append(dict(case=label + '|anchor'))


K.check_freeze()
rng = np.random.default_rng(20261116)
cells = [(f, s, k, 'heldout') for s in ('S2', 'S3') for f in K.FAMILIES for k in K.HELDOUT_SEEDS] + \
        [(f, s, k, 'dev') for s in ('S0', 'S1') for f in K.FAMILIES for k in K.DEV_SEEDS]
for fam_id, size, seed, split in cells:
    fam = Family3D(fam_id, size, seed=seed, split=split)
    for t in range(fam.T + 1):
        A, _, _ = fam.get(t); A = R6.ensure_canonical(A); out['matrices'] += 1
        lab = f'{fam_id}|{size}|{seed}|{t}'
        check(A, lab)
        check_shifted(A, lab)
        if split == 'heldout' and seed == 200 and t in (0, 10, 20):
            check_anchor(A, lab)
        n = A.shape[0]
        rows = np.repeat(np.arange(n), np.diff(A.indptr))
        up = np.flatnonzero(A.indices > rows)
        A1 = A.copy(); q = int(rng.choice(up)); A1.data[q] = np.nextafter(A1.data[q], np.inf); check(canon(A1), lab + '|ulp')
        L = A.tolil(); i = int(rng.integers(n)); j = (i + n // 3) % n
        if L[i, j] == 0 and i != j:
            C = sp.coo_matrix(A); r = np.append(C.row, i); c = np.append(C.col, j); d = np.append(C.data, 0.0)
            A2 = sp.csr_matrix((d, (r, c)), shape=A.shape); A2.sort_indices(); check(A2, lab + '|zero1')
        C = sp.coo_matrix(A); m = ~((C.row == C.col) & (C.row == int(rng.integers(n))))
        A3 = sp.csr_matrix((C.data[m], (C.row[m], C.col[m])), shape=A.shape); A3.sort_indices(); check(A3, lab + '|nodiag')
        A4 = A.copy(); dp = FD(A4); A4.data[dp[int(rng.integers(n))]] = np.nan; check(A4, lab + '|nan')
        A5 = A.copy(); q = int(rng.choice(up)); r_, c_ = rows[q], A5.indices[q]
        A5.data[q] = 0.0
        qq = A5.indptr[c_] + int(np.searchsorted(A5.indices[A5.indptr[c_]:A5.indptr[c_ + 1]], r_)); A5.data[qq] = -0.0
        check(A5, lab + '|signedzero')
    print(fam_id, size, seed, out['cases'], out['identical'], flush=True)
out['ALL_IDENTICAL'] = not out['failures']
(HERE / 'test_shift.json').write_text(json.dumps(out, indent=1))
print('ALL_IDENTICAL', out['ALL_IDENTICAL'], out['matrices'], out['cases'], out.get('shift_identical'), out.get('shift_cases'), out.get('anchor_identical'), out.get('anchor_cases'), {k: round(x, 1) for k, x in out['timing_ms'].items()})
sys.exit(0 if out['ALL_IDENTICAL'] else 1)
