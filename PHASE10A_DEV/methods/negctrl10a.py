"""Phase 10A negative controls (development data, S0, exact arithmetic decides):
  NC-INCONSISTENT  the matrix handed to the methods differs from the element model by one ulp in one symmetric pair of
                   entries from t = 5 on. M2-A arms (C, D, D2) must fail closed (no CERTIFIED record for a perturbed
                   matrix via the transfer); per-system arms may certify the GIVEN matrix, and every such certificate is
                   checked exactly (exact rational inertia of A - lambda_lo I and exact residual) against the perturbed
                   matrix actually solved.
(The PD-loss family SAFE-S and the cut-plane family SAFE-P are run through run_cell10a.py and replay10a.py with exact
checks; see run_plan10a.py.)
Usage: python negctrl10a.py <out_dir>"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner10a as N                                                # noqa: E402
R6 = N.R6
from cars_m2.families import Family3D                                # noqa: E402
from cars import _reference_m0 as v                                  # noqa: E402
from cars.verify import exact_inertia, exact_residual2               # noqa: E402


class Perturbed:
    """Wraps a family; from t >= t0 one off-diagonal symmetric pair is moved by one ulp (no longer the element model)."""

    def __init__(self, fam, t0=5):
        self._f = fam; self.t0 = t0

    def get(self, t):
        A, b, meta = self._f.get(t)
        A = R6.ensure_canonical(A).copy()
        if t >= self.t0:
            i = 3; row = A.indices[A.indptr[i]:A.indptr[i + 1]]
            j = int(row[row != i][0])
            for (p, q) in ((i, j), (j, i)):
                k = A.indptr[p] + int(np.flatnonzero(A.indices[A.indptr[p]:A.indptr[p + 1]] == q)[0])
                A.data[k] = np.nextafter(A.data[k], np.inf)
            meta = dict(meta); meta['matrix_sha256'] = v.matrix_sha256(A)
        return A, b, meta

    def __getattr__(self, name):
        return getattr(self._f, name)


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    fam = Perturbed(Family3D('E3-TS', 'S0', seed=0, split='dev'))
    res = {}
    arms = [('A', 'RPS|P1F'), ('B1', 'RPS-EFT'), ('B', 'RPS-OPT'), ('B9', 'RPS-OPT-S09'), ('C', 'RPS|M2A'), ('D', 'M2A-EFT'),
            ('D2', 'M2A-EFT-S05')]
    for label, cfg in arms:
        if cfg.startswith('RPS|'):
            path, rows = R6.run_sequence_5(fam, cfg.split('|')[1], 'RPS', out / label, track='T1', binding='lean',
                                           target=1e-8, write_artifacts=True)
        else:
            path, rows = N.run_sequence_10a(fam, cfg, out / label, target=1e-8, write_artifacts=True)
        checks = []
        for r in rows:
            A, b, _ = fam.get(int(r['t']))
            x = np.load(out / label / r['solution_file'])
            if r['status'] == 'CERTIFIED':
                lo = float.fromhex(r['lambda_lo']); R = float.fromhex(r['R_hi'])
                inert = exact_inertia(A.toarray(), lo)
                ok_l = inert is not None and inert[0] == 0
                ok_r = Fraction(R) ** 2 >= exact_residual2(A, b, x)
                via = (r['lambda_chain'][0].get('type') if r.get('lambda_chain') else None)
                checks.append(dict(t=r['t'], status='CERTIFIED', via=via, exact_lambda_ok=bool(ok_l), exact_residual_ok=bool(ok_r)))
            else:
                checks.append(dict(t=r['t'], status=r['status'], reason=r.get('reason')))
        perturbed_certified_by_transfer = [c for c in checks if c['status'] == 'CERTIFIED' and c['t'] >= 5 and c['via'] == 'm2a']
        res[label] = dict(config=cfg, certified=sum(c['status'] == 'CERTIFIED' for c in checks),
                          unverified=[(c['t'], c.get('reason')) for c in checks if c['status'] != 'CERTIFIED'][:6],
                          exact_all_ok=all(c.get('exact_lambda_ok', True) and c.get('exact_residual_ok', True) for c in checks),
                          perturbed_steps_certified_via_transfer=len(perturbed_certified_by_transfer))
        print(label, json.dumps(res[label]), flush=True)
    res['pass'] = all(r['exact_all_ok'] for r in res.values()) and all(
        res[k]['perturbed_steps_certified_via_transfer'] == 0 for k in ('C', 'D', 'D2'))
    (out / 'negctrl10a.json').write_text(json.dumps(res, indent=1))
    print('NEGCTRL_PASS', res['pass'])


if __name__ == '__main__':
    main()
