"""Revision 3, exploratory (untimed, after the cells of FULLC_PROTOCOL.md): interpreted overhead of the conjugate
gradient loop. For E3-TS and D3-TS at S3 (seed 200, already consumed), an anchor factorization of A_0 (sigma = 0.9 l~)
preconditions frozen pcg_rtol solves of A_t x = b_t, t = 1..5, started from zero with rtol = 1e-12. Each solve runs
twice: (1) unchanged, giving its total time T and the iteration count k; (2) with timing wrappers around the matrix-
vector product and the preconditioner solve, giving their times Tm and Tp. The overhead per iteration is
(T - Tm - Tp) / k (the wrappers' own cost is inside Tm and Tp, so this slightly underestimates the overhead). It is
multiplied by the mean PCG iteration counts of the arms in analysis_fullc.json to estimate the interpreted CG cost per
sequence on each side. An estimate, not a timed comparison. Writes cg_overhead.json."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
import psutil
import scipy.sparse.linalg as sla
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
import run_cell10b as RC                                             # noqa: E402
N = RC.N; R6 = RC.R6
from cars_m2.families import Family3D                                # noqa: E402


def main():
    try:
        psutil.Process().cpu_affinity([2]); psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception:
        pass
    K.check_freeze()
    out = {}
    for fam_id in ('E3-TS', 'D3-TS'):
        fam = Family3D(fam_id, 'S3', seed=200, split='heldout')
        A0, _, _ = fam.get(0); A0 = R6.ensure_canonical(A0)
        tm = R6.Timer(); lam, _ = N._estimate(A0, None, None)
        anc = R6.rump_anchor_once(A0, 0.9 * lam, 'RPS', tm); M, _ = R6.precond_from_anchor(anc, tm)
        per_it = []; rows = []
        for t in range(1, 6):
            A, b, _ = fam.get(t); A = R6.ensure_canonical(A)
            N.pcg_rtol(A, b, M, 1e-12, None)                         # warm
            t0 = time.perf_counter(); x, si = N.pcg_rtol(A, b, M, 1e-12, None); T = time.perf_counter() - t0
            acc = {'m': 0.0, 'p': 0.0}

            def mv(z):
                s = time.perf_counter(); y = A @ z; acc['m'] += time.perf_counter() - s; return y

            def pv(z):
                s = time.perf_counter(); y = M.matvec(z); acc['p'] += time.perf_counter() - s; return y
            Aw = sla.LinearOperator(A.shape, matvec=mv, dtype=np.float64); Mw = sla.LinearOperator(A.shape, matvec=pv, dtype=np.float64)
            x2, si2 = N.pcg_rtol(Aw, b, Mw, 1e-12, None)
            k = si['iters']
            ov = (T - acc['m'] - acc['p']) / max(1, k)
            per_it.append(ov); rows.append(dict(t=t, iters=k, total_s=T, matvec_s=acc['m'], precond_s=acc['p'], overhead_per_iter_s=ov,
                                                 same_iterate=bool(np.array_equal(x.view(np.uint64), x2.view(np.uint64)))))
        out[fam_id] = dict(solves=rows, overhead_per_iter_s=float(np.median(per_it)))
        print(fam_id, round(1e6 * out[fam_id]['overhead_per_iter_s'], 1), 'us/iteration', flush=True)
    fc = HERE / 'analysis_fullc.json'
    if fc.exists():
        a = json.loads(fc.read_text())
        est = {}
        for fam_id, (P, Q) in (('E3-TS', ('B9f', 'Df')), ('D3-TS', ('Bf', 'D2f'))):
            r = a['configs'][f'{fam_id}|S3']['arms']; ov = out[fam_id]['overhead_per_iter_s']
            est[fam_id] = dict(per_system_arm=P, reuse_arm=Q, iters_per_sequence=[r[P]['pcg_iters_mean'], r[Q]['pcg_iters_mean']],
                               overhead_s=[ov * r[P]['pcg_iters_mean'], ov * r[Q]['pcg_iters_mean']],
                               cost_s=[r[P]['cost_mean'], r[Q]['cost_mean']])
        out['estimate_per_sequence'] = est
        print(json.dumps(est, indent=1))
    (HERE / 'cg_overhead.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
