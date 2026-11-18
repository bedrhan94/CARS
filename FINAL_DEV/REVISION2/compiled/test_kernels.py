"""Revision 2: bitwise equivalence of the compiled kernels (compiled_kernels.py) with the frozen implementations.

EFT residual bound (frozen PHASE10A_DEV/eft/eft_residual.eft_residual_bound vs compiled):
  (a) every record of the EFT arms (B1, B, B9, D, D2) of the Phase 10B ST repeat-0 cells at S2 and S3, all five
      held-out seeds (already consumed): the stored solution vector is re-read and both implementations must return the
      same R (bit for bit) and the same info, equal to the R_hi stored in the record;
  (b) synthetic stress cases: random sparse matrices with uneven rows (including empty rows), wide exponent ranges,
      and inputs that trigger every fallback branch (non-finite input, split-overflow risk, TwoProduct underflow,
      non-finite products, non-finite sums), compared on (R, info) or on the raised exception.
Binding (frozen FastAssembler vs CompiledAssembler), per family / size / seed and per step t:
  data(c) bit for bit; matches(A, c) True for both; on perturbed inputs (a one-ulp change of one stored value, an
  extra stored entry, an explicit zero, a one-ulp change of one coefficient, the previous step's coefficients) both
  must return the same answer, and an acceptance is correct only if the frozen full assembly of the coefficients is
  bitwise equal to the matrix (a one-ulp coefficient change can be absorbed by rounding, or concern an element without
  free degrees of freedom); coefficient vectors with exact zeros (zero-dropping path) assembled with the frozen
  assembler and accepted by both.
Writes test_kernels.json; exit code 1 on any difference."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
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
N = RC.N; R6 = RC.R6; v = RC.v; EFT = N.EFT
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import compiled_kernels as CK                                        # noqa: E402

CK.install(N, R6, EFT, v)
FROZEN = EFT.eft_residual_bound
RUNS = CARS / 'FINAL_DEV' / 'phase10b' / 'runs'
EFT_ARMS = ('B1', 'B', 'B9', 'D', 'D2')
out = {'eft_records': {}, 'eft_synthetic': {}, 'binding': {}, 'failures': []}


def call(f, A, b, x):
    try:
        R, info = f(A, b, x)
        return ('ok', float(R).hex(), json.dumps(info, sort_keys=True))
    except Exception as e:                                            # noqa: BLE001
        return ('raise', type(e).__name__, getattr(e, 'reason', str(e)))


def fail(kind, **kw):
    out['failures'].append(dict(kind=kind, **kw))


# ------------------------------------------------------------------ (a) stored records
t0 = time.time()
K.check_freeze()
for size in ('S3', 'S2'):
    for fam_id in K.FAMILIES:
        for seed in K.HELDOUT_SEEDS:
            cell = RUNS / f'ST_{size}' / f'{fam_id}_{size}_s{seed}_r0_ST'
            fam = Family3D(fam_id, size, seed=seed, split='heldout')
            mats = {}
            n_eq = n_rec = 0
            for arm in EFT_ARMS:
                f = next((cell / arm).glob('*.jsonl'))
                for line in f.read_text().splitlines():
                    r = json.loads(line)
                    if r['status'] != 'CERTIFIED':
                        continue
                    t = int(r['t'])
                    if t not in mats:
                        A, b, _ = fam.get(t); mats[t] = (v.canonical(A), b)
                    A, b = mats[t]
                    x = np.load(cell / arm / 'artifacts' / f'{f.stem}_x_t{t}.npy')
                    a1 = call(FROZEN, A, b, x); a2 = call(CK.eft_residual_bound, A, b, x)
                    n_rec += 1
                    same = a1 == a2 and (r.get('residual_kind') != 'eft' or a1[1] == float.fromhex(r['R_hi']).hex())
                    n_eq += same
                    if not same:
                        fail('eft_record', cell=cell.name, arm=arm, t=t, frozen=a1, compiled=a2, stored=r.get('R_hi'))
            out['eft_records'][f'{fam_id}|{size}|{seed}'] = dict(records=n_rec, identical=n_eq)
            print('EFT', fam_id, size, seed, n_rec, n_eq, f'{time.time() - t0:.0f}s', flush=True)

# ------------------------------------------------------------------ (b) synthetic stress cases
rng = np.random.default_rng(20261116)


def rand_csr(n, maxrow, lo, hi, empty_frac=0.05):
    rows = []; cols = []; vals = []
    for i in range(n):
        k = 0 if rng.random() < empty_frac else int(rng.integers(1, maxrow + 1))
        c = rng.choice(n, size=min(k, n), replace=False)
        rows += [i] * c.size; cols += list(c)
        vals += list(rng.choice([-1, 1], c.size) * 10.0 ** rng.uniform(lo, hi, c.size))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


cases = []
for rep in range(150):
    n = int(rng.integers(1, 400)); A = rand_csr(n, int(rng.integers(1, 40)), -8, 8)
    x = rng.standard_normal(n) * 10.0 ** rng.uniform(-6, 6, n); b = A @ x + rng.standard_normal(n) * 1e-10
    cases.append(('generic', A, b, x))
    xr = x * (1 + 1e-15 * rng.standard_normal(n)); cases.append(('near-solution', A, A @ xr, x))
for rep in range(30):
    n = int(rng.integers(5, 200)); A = rand_csr(n, 20, -150, 150); x = rng.standard_normal(n) * 10.0 ** rng.uniform(-150, 150, n)
    cases.append(('wide-exponent', A, A @ x, x))
for rep in range(20):
    n = 50; A = rand_csr(n, 10, -1, 1); x = rng.standard_normal(n); b = A @ x
    xn = x.copy(); xn[3] = np.nan; cases.append(('nonfinite-input', A, b, xn))
    xb = x.copy(); xb[4] = 2.0 ** 996; cases.append(('split-overflow', A, b, xb))
    xt = x.copy(); xt[:] = 1e-300; At = A.copy(); At.data[:] = 1e-300; cases.append(('tp-underflow', At, b, xt))
    Ah = A.copy(); Ah.data[:] = 1e299; xh = np.full(n, 1e299); cases.append(('nonfinite-product', Ah, b, xh))
    As = A.copy(); As.data[:] = 1e154; xs = np.full(n, 1.7e154); cases.append(('nonfinite-sum', As, np.zeros(n), xs))
    Ab = A.copy(); Ab.data[:] = 1e150; xq = np.full(n, 1e150); cases.append(('large-bound', Ab, np.zeros(n), xq))
    cases.append(('zero-x', A, b, np.zeros(n)))
    Az = A.copy(); Az.data[::3] = 0.0; cases.append(('explicit-zeros', Az, b, x))
    cases.append(('subnormal-b', A, np.full(n, 5e-324), np.zeros(n)))
    Ao = A.copy(); Ao.data[:] = 1e-300 * np.sign(Ao.data); xo = x.copy(); xo[4] = 2.0 ** 996
    cases.append(('split-overflow-finite', Ao, Ao @ xo, xo))
    Am = A.copy(); Am.data[::2] = 1e-200; xm = rng.standard_normal(n); xm[::3] = 1e-200
    cases.append(('partial-underflow', Am, Am @ xm, xm))
    bn = b.copy(); bn[0] = np.inf; cases.append(('nonfinite-b', A, bn, x))
for rep in range(40):
    n = int(rng.integers(5, 300)); A = rand_csr(n, 30, -60, 60); x = rng.standard_normal(n) * 10.0 ** rng.uniform(-60, 60, n)
    cases.append(('moderate-wide-exponent', A, A @ x, x))
for name, A, b, x in cases:
    a1 = call(FROZEN, A, b, x); a2 = call(CK.eft_residual_bound, A, b, x)
    d = out['eft_synthetic'].setdefault(name, dict(cases=0, identical=0, outcomes=set()))
    d['cases'] += 1; d['identical'] += (a1 == a2); d['outcomes'].add(a1[0] + ':' + (a1[2] if a1[0] == 'raise' else json.loads(a1[2]).get('reason', 'eft') or 'eft'))
    if a1 != a2:
        fail('eft_synthetic', case=name, frozen=a1, compiled=a2)
for d in out['eft_synthetic'].values():
    d['outcomes'] = sorted(d['outcomes'])
print('SYNTH', json.dumps(out['eft_synthetic']), flush=True)

# ------------------------------------------------------------------ binding
FA = CK.FA_FROZEN; CA = CK.CompiledAssembler
bits = lambda a: np.asarray(a, np.float64).view(np.uint64)
cells = [(f, s, k, 'heldout') for s in ('S2', 'S3') for f in K.FAMILIES for k in K.HELDOUT_SEEDS] + \
        [(f, s, k, 'dev') for s in ('S0', 'S1') for f in K.FAMILIES for k in K.DEV_SEEDS]
for fam_id, size, seed, split in cells:
    fam = Family3D(fam_id, size, seed=seed, split=split)
    A0, _, m0 = fam.get(0); n = A0.shape[0]
    fa = FA(fam.mesh, fam.Kref, n); ca = CA(fam.mesh, fam.Kref, n)
    st = dict(steps=0, data_identical=0, true_accepted=0, negatives=0, negatives_rejected=0, zero_coeff=0, zero_coeff_ok=0)
    prev_c = None
    for t in range(fam.T + 1):
        A, b, meta = fam.get(t); A = v.canonical(A); c = np.asarray(meta['c'], np.float64)
        st['steps'] += 1
        st['data_identical'] += np.array_equal(bits(fa.data(c)), bits(ca.data(c)))
        m1, m2 = fa.matches(A, c), ca.matches(A, c)
        st['true_accepted'] += (m1 and m2)
        if not (m1 and m2):
            fail('binding_true', cell=f'{fam_id}|{size}|{seed}', t=t, frozen=m1, compiled=m2)
        negs = []
        A1 = A.copy(); j = int(rng.integers(A1.nnz)); A1.data[j] = np.nextafter(A1.data[j], np.inf); negs.append(('ulp', A1, c))
        c1 = c.copy(); e = int(rng.integers(c1.size)); c1[e] = np.nextafter(c1[e], -np.inf); negs.append(('coeff', A, c1))
        if prev_c is not None and not np.array_equal(prev_c, c):
            negs.append(('other-step', A, prev_c))
        A2 = A.tolil(); i = int(rng.integers(n)); jj = (i + n // 2) % n
        if A2[i, jj] == 0:
            A2[i, jj] = 1e-300; A2[jj, i] = 1e-300; negs.append(('extra-entry', sp.csr_matrix(A2), c))
        A3 = A.copy(); A3.data[0] = 0.0; negs.append(('explicit-zero', A3, c))
        for nm, An, cn in negs:
            r1, r2 = fa.matches(An, cn), ca.matches(An, cn)
            st['negatives'] += 1; st['negatives_rejected'] += (not r1 and not r2)
            if r1 != r2:
                fail('binding_negative', cell=f'{fam_id}|{size}|{seed}', t=t, case=nm, frozen=r1, compiled=r2)
            elif r1:                                                  # accepted by both: correct only if A IS the assembly
                own = v.canonical(N.assemble_float(fam.mesh, fam.Z, cn, fam.Kref))
                correct = v.matrix_sha256(own) == v.matrix_sha256(An)
                st['accepted_correctly'] = st.get('accepted_correctly', 0) + correct
                st.setdefault('accepted_cases', []).append(nm)
                if not correct:
                    fail('binding_false_accept', cell=f'{fam_id}|{size}|{seed}', t=t, case=nm)
        prev_c = c
        if t % 5 == 0:                                                # zero-dropping path
            cz = c.copy(); cz[rng.random(cz.size) < 0.15] = 0.0
            Az = v.canonical(N.assemble_float(fam.mesh, fam.Z, cz, fam.Kref))
            st['zero_coeff'] += 1
            ok = (fa.matches(Az, cz) and ca.matches(Az, cz)
                  and np.array_equal(bits(fa.data(cz)), bits(ca.data(cz))))
            st['zero_coeff_ok'] += ok
            if not ok:
                fail('binding_zero_coeff', cell=f'{fam_id}|{size}|{seed}', t=t)
    out['binding'][f'{fam_id}|{size}|{seed}'] = st
    print('BIND', fam_id, size, seed, st, flush=True)

out['seconds'] = time.time() - t0
out['ALL_IDENTICAL'] = not out['failures']
out['environment'] = dict(python=sys.version.split()[0], numpy=np.__version__, numba=CK.nb.__version__)
(HERE / 'test_kernels.json').write_text(json.dumps(out, indent=1))
print('ALL_IDENTICAL', out['ALL_IDENTICAL'], 'failures', len(out['failures']), out['failures'][:3])
sys.exit(0 if out['ALL_IDENTICAL'] else 1)
