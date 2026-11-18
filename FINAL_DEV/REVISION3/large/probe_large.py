"""Revision 3: untimed feasibility probe for the large-size experiment (development seed 0 only; no held-out data).
E3-TS geometry with N = 32 (size label S5 of the frozen generator, n about 1.05e5): time for generating and assembling
one matrix, the cold eigenvalue estimate, one Rump anchor (shift + PARDISO analysis and factorization), one PCG solve
and the accurate residual bound; peak memory. Writes probe_large.json. Used only to set the budget of the protocol."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import json, time
from pathlib import Path
import numpy as np
import psutil
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import run_cell10b as RC                                             # noqa: E402
N = RC.N; R6 = RC.R6; v = RC.v
from cars_m2.families import Family3D                                # noqa: E402

p = psutil.Process()
try:
    p.cpu_affinity([2])
except Exception:
    pass
out = {}
t0 = time.perf_counter(); fam = Family3D('E3-TS', 'S5', seed=0, split='dev'); out['family_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); A, b, meta = fam.get(0); A = R6.ensure_canonical(A); out['assembly_s'] = time.perf_counter() - t0
out['n'] = int(A.shape[0]); out['nnz'] = int(A.nnz)
tm = R6.Timer()
t0 = time.perf_counter(); lam, ev = N._estimate(A, None, None); out['cold_estimate_s'] = time.perf_counter() - t0; out['lambda_est'] = lam
t0 = time.perf_counter(); anc = R6.rump_anchor_once(A, 0.9 * lam, 'RPS', tm); out['anchor_s'] = time.perf_counter() - t0
out['anchor_components'] = {k: v_ for k, v_ in tm.t.items() if v_}
out['nnz_factor'] = anc['extra']['nnz_factor']
op, _ = R6.precond_from_anchor(anc, tm)
t0 = time.perf_counter(); lam2, ev2 = N._estimate(A, op, ev); out['warm_estimate_s'] = time.perf_counter() - t0
A1, b1, _ = fam.get(1); A1 = R6.ensure_canonical(A1)
t0 = time.perf_counter(); anc1 = R6.rump_anchor_once(A1, 0.9 * lam, 'RPS', tm); out['anchor2_s'] = time.perf_counter() - t0
op1, _ = R6.precond_from_anchor(anc1, tm)
t0 = time.perf_counter(); x, si = N.pcg_rtol(A1, b1, op1, 1e-12, None); out['pcg_s'] = time.perf_counter() - t0; out['pcg_iters'] = si['iters']
t0 = time.perf_counter(); R, info = N.EFT.eft_residual_bound(A1, b1, x); out['eft_s'] = time.perf_counter() - t0
t0 = time.perf_counter(); Rs = v.residual_bound(A1, b1, x); out['std_residual_s'] = time.perf_counter() - t0
out['peak_wset_GiB'] = getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30
(HERE / 'probe_large.json').write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
