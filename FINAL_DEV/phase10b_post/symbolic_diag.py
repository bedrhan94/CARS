"""POST-HOC EXPLORATORY diagnostic (not pre-registered; no decision role): how often does the per-system baseline lose
its symbolic reuse, and what does a symbolic analysis cost?

(a) Exact re-analysis counts. The frozen PARDISO session (runner6.PardisoSession) keeps the union U of all analysed
    sparsity patterns and re-analyses (phase 11: METIS ordering + symbolic factorization) whenever a matrix has a stored
    entry outside U. The stored pattern of the assembled matrices is value-dependent: SciPy's sparse addition in the
    exact symmetrization drops entries that cancel to exactly zero. Re-analyses are counted here by replaying the union
    rule on the frozen held-out matrices (patterns only), for a factorization at every step (per-system arms) and for
    the anchor steps recorded by the reuse arm (ST, repeat 0).
(b) Timing of phase 11 (analysis) and phase 22 (numerical factorization) separately, single thread, CPU 2, on held-out
    matrices of each family and size (median of 5 calls).
Usage: python symbolic_diag.py -> symbolic_diag.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import runner6 as R6                                                 # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

PRIMARY = {'E3-TS': ('B9', 'D'), 'D3-TS': ('B', 'D2')}


def pattern(A):
    A = R6.ensure_canonical(A)
    return set(zip(np.repeat(np.arange(A.shape[0]), np.diff(A.indptr)).tolist(), A.indices.tolist())), A


def anchor_steps(fam, size, seed, arm):
    d = CARS / 'FINAL_DEV/phase10b/runs' / f'ST_{size}' / f'{fam}_{size}_s{seed}_r0_ST' / arm
    recs = [json.loads(l) for l in next(d.glob('*.jsonl')).read_text().splitlines() if l.strip()]
    return [r['t'] for r in recs if r['factor_count'] > 0]


def timing(A, reps=5):
    """Phase 11 (analysis) and phase 22 (numerical factorization) of the SPD matrix A in a fresh PARDISO session."""
    S = R6.PardisoSession()
    t11, t22 = [], []
    for _ in range(reps):
        S._reset(); S.full_indptr = None
        t0 = time.perf_counter(); S._analyse(A); t11.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); S.factor(A); t22.append(time.perf_counter() - t0)
    S._reset()
    return float(np.median(t11)), float(np.median(t22))


def main():
    K.check_freeze()
    p = psutil.Process(); p.cpu_affinity([2]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    out = {'counts': {}, 'timing': {}, 'note': 'post-hoc exploratory; not part of the registered analysis'}
    for fam in K.FAMILIES:
        for size in K.SIZES:
            for seed in K.HELDOUT_SEEDS:
                F = Family3D(fam, size, seed=seed, split='heldout')
                pats = []; mats = []
                for t in range(F.T + 1):
                    P, A = pattern(F.get(t)[0]); pats.append(P)
                    if seed == 200 and t in (0, 10):
                        mats.append(A)
                U = set(); every = 0
                for P in pats:
                    if not P <= U:
                        every += 1; U |= P
                Pq = PRIMARY[fam][1]; steps = anchor_steps(fam, size, seed, Pq)
                U = set(); anc = 0
                for t in steps:
                    if not pats[t] <= U:
                        anc += 1; U |= pats[t]
                distinct = len({frozenset(P) for P in pats})
                out['counts'][f'{fam}|{size}|{seed}'] = dict(per_system_analyses=every, reuse_anchor_steps=len(steps),
                                                            reuse_analyses=anc, distinct_patterns=distinct,
                                                            nnz_range=[min(len(P) for P in pats), max(len(P) for P in pats)])
                print(fam, size, seed, out['counts'][f'{fam}|{size}|{seed}'], flush=True)
                if seed == 200:
                    t11, t22 = timing(mats[0])
                    out['timing'][f'{fam}|{size}'] = dict(n=int(A.shape[0]), analysis_s=t11, numeric_s=t22)
                    print('timing', fam, size, out['timing'][f'{fam}|{size}'], flush=True)
    (HERE / 'symbolic_diag.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
