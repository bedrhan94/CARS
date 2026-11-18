"""Phase 10A independent arithmetic (no M0/EFT/Cholesky code): samples of the Phase 10A records.
  L  lambda: for selected matrices, the LARGEST lambda_lo claimed by any arm is proved with an Arb ball congruence
     (PHASE5_DEV/arith/independent5.arb_congruence_pd; S1 n = 1 445-1 944, S2 one matrix per family)
  E  error / target: for sampled records of every NEW arm, the exact rational residual ||b - A x||^2 is compared with
     the recorded R^2; together with L this re-proves ||x - x*|| <= ||r|| / lambda_lo <= R / lambda_lo <= eps.
Usage: python independent10a.py <out_json> <run_root>..."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, random, time
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'PHASE5_DEV' / 'arith')); sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import independent5 as I5                                            # noqa: E402  (unchanged Phase 5 module)
import runner6 as R6                                                 # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

NEW = ('RPS-EFT', 'RPS-OPT', 'RPS-OPT-S09', 'M2A-EFT', 'M2A-EFT-S05')


def load_all(roots):
    recs = []
    for root in roots:
        for f in Path(root).rglob('*.jsonl'):
            if '_warmup' in f.parts:
                continue
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); r['_dir'] = str(f.parent); recs.append(r)
    return recs


def main():
    out = Path(sys.argv[1]); roots = sys.argv[2:]
    t0 = time.time(); recs = [r for r in load_all(roots) if r['status'] == 'CERTIFIED']
    res = {'lambda_proofs': [], 'exact_residuals': []}
    # L: matrices to prove
    picks = [('E3-TS', 'S1', 0, 0), ('E3-TS', 'S1', 0, 10), ('E3-TS', 'S1', 0, 20), ('D3-TS', 'S1', 0, 0),
             ('D3-TS', 'S1', 0, 10), ('D3-TS', 'S1', 0, 20), ('E3-TS', 'S2', 1, 10), ('D3-TS', 'S2', 1, 10)]
    for fid, size, seed, t in picks:
        cl = [float.fromhex(r['lambda_lo']) for r in recs if r['family'] == fid and r['size'] == size and r['seed'] == seed and r['t'] == t]
        if not cl:
            continue
        lam = max(cl)
        A, _, _ = Family3D(fid, size, seed=seed, split='dev').get(t); A = R6.ensure_canonical(A)
        s = time.time(); pr = I5.arb_congruence_pd(A, lam)
        res['lambda_proofs'].append(dict(family=fid, size=size, seed=seed, t=t, claims=len(cl), max_lambda_lo=lam,
                                         proved=pr['proved'], lapack_lambda1=pr['lapack_lambda1'],
                                         ratio_max_claim_to_lambda1=lam / pr['lapack_lambda1'], seconds=time.time() - s))
        print('L', res['lambda_proofs'][-1], flush=True)
    # E: exact residuals, sample per new arm and size
    rng = random.Random(10)
    fams = {}
    for meth in NEW:
        for size in ('S1', 'S2', 'S3'):
            pool = [r for r in recs if r['method'] == meth and r['size'] == size]
            for r in rng.sample(pool, min(len(pool), 4 if size != 'S3' else 2)):
                key = (r['family'], r['size'], r['seed'])
                if key not in fams:
                    fams[key] = Family3D(*key[:2], seed=key[2], split='dev')
                A, b, _ = fams[key].get(int(r['t'])); A = R6.ensure_canonical(A)
                x = np.load(Path(r['_dir']) / r['solution_file'])
                s = time.time(); ex = I5.exact_residual_sq(A, b, x); R = float.fromhex(r['R_hi'])
                ok = Fraction(R) ** 2 >= ex
                exr = float(ex) ** 0.5
                res['exact_residuals'].append(dict(method=meth, family=r['family'], size=size, seed=r['seed'], t=r['t'],
                                                   residual_kind=r.get('residual_kind'), R=R, exact=exr,
                                                   R_over_exact=(R / exr if exr > 0 else None), ok=bool(ok),
                                                   seconds=time.time() - s))
                print('E', res['exact_residuals'][-1], flush=True)
    res['summary'] = dict(lambda_proved=sum(p['proved'] for p in res['lambda_proofs']), lambda_total=len(res['lambda_proofs']),
                          residual_ok=sum(e['ok'] for e in res['exact_residuals']), residual_total=len(res['exact_residuals']),
                          seconds=time.time() - t0)
    out.write_text(json.dumps(res, indent=1))
    print('SUMMARY', res['summary'])


if __name__ == '__main__':
    main()
