"""Revision 2, post hoc (referee comment on the EFT tightness claim): absolute values for the frozen exact EFT test suite.

Re-generates the 2 000 cases of PHASE10A_DEV/eft/eft_exact_tests.py with the same generator and the same seed (the
frozen module is imported, not modified, and its own JSON is not rewritten) and records, per generator, the ratio of
the bound R to the exact residual ||r||_2 and the ratio of R to the data scale ||  |b| + |A||x|  ||_2 (exact rational
values converted to float). Writes eft_exact_abs.json."""
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
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'PHASE10A_DEV' / 'eft'))
import eft_exact_tests as T                                          # noqa: E402  frozen generators (main() not called)

ff = lambda q: float(Fraction(int(q.p), int(q.q)))


def scale(A, b, x):
    A = sp.csr_matrix(A)
    s = np.abs(b) + abs(A) @ np.abs(x)
    return float(np.linalg.norm(s))


def main():
    t0 = time.time()
    rng = np.random.default_rng(20261114)                            # same seed and order as the frozen suite
    per = {'G1': 400, 'G2': 400, 'G3': 400, 'G4': 100, 'G5': 300, 'G6': 400}
    out = {}
    for kind, cnt in per.items():
        rows = []
        for c in range(cnt):
            A, b, x = T.gen(kind, rng)
            if not (np.all(np.isfinite(A.data)) and np.all(np.isfinite(b)) and np.all(np.isfinite(x))):
                continue
            ex = T.exact_r2(A, b, x)
            try:
                R, info = T.EFT.eft_residual_bound(A, b, x)
            except T.v.CertFail:
                continue
            exr = math.sqrt(ff(ex)) if ex > 0 else 0.0
            sc = scale(A, b, x)
            rows.append(dict(kind=info['kind'], R=R, exact=exr, ratio=(R / exr if exr > 0 else None),
                             rel_scale=(R / sc if sc > 0 else None), violation=bool(T.fq(R) ** 2 < ex)))
        e = [r for r in rows if r['kind'] == 'eft']
        rat = [r['ratio'] for r in e if r['ratio'] is not None]
        rel = [r['rel_scale'] for r in e if r['rel_scale'] is not None]
        out[kind] = dict(cases=len(rows), eft=len(e), violations=sum(r['violation'] for r in rows),
                         ratio_median=float(np.median(rat)) if rat else None, ratio_max=float(max(rat)) if rat else None,
                         rel_scale_median=float(np.median(rel)) if rel else None, rel_scale_max=float(max(rel)) if rel else None,
                         R_median=float(np.median([r['R'] for r in e])) if e else None,
                         exact_median=float(np.median([r['exact'] for r in e])) if e else None,
                         exact_min=float(min(r['exact'] for r in e)) if e else None)
        print(kind, out[kind], flush=True)
    frozen = json.loads((CARS / 'PHASE10A_DEV' / 'eft' / 'eft_exact_tests.json').read_text())
    same = all(out[k]['cases'] == frozen['per_kind'][k]['cases'] and
               (frozen['per_kind'][k]['ratio_eft_max'] is None or
                abs(out[k]['ratio_max'] / frozen['per_kind'][k]['ratio_eft_max'] - 1) < 1e-12) for k in per)
    res = dict(per_kind=out, reproduces_frozen_suite=bool(same), seconds=time.time() - t0)
    (HERE / 'eft_exact_abs.json').write_text(json.dumps(res, indent=1))
    print('reproduces frozen suite:', same)


if __name__ == '__main__':
    main()
