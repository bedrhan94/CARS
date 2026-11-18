"""Phase 10B independent arithmetic on the held-out records (no M0, EFT or Cholesky code; unchanged Phase 5 module).
  E  exact residuals: pre-registered random sample (random.Random(10_2026)) of CERTIFIED records: per arm (all seven),
     per size (S2, S3): 4 records from ST cells and 2 from MT cells. The exact rational ||b - A x||^2 (fmpq) is compared
     with the recorded R^2; with a valid lambda_lo this re-proves ||x - x*|| <= ||r|| / lambda_lo <= R / lambda_lo <= eps.
  L  lambda: Arb ball congruence (independent5.arb_congruence_pd) proving lambda_min(A) > the LARGEST lambda_lo
     claimed by any arm in any mode for that matrix. Mandatory picks (S2): per family (seed 201, t 0), (seed 203, t 20).
     Optional picks (S3): per family (seed 202, t 10), each in its own process with a 30-min cap (UNAVAILABLE if capped).
Usage: python independent10b.py E <out_json> <run_root>...
       python independent10b.py L <out_json> <family> <size> <seed> <t> <run_root>..."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, random, time
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(K.CARS / 'PHASE5_DEV' / 'arith')); sys.path.insert(0, str(K.CARS / 'PHASE6_DEV' / 'lean6'))
import independent5 as I5                                            # noqa: E402  unchanged Phase 5 module
import runner6 as R6                                                 # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

MANDATORY_L = [(f, 'S2', 201, 0) for f in K.FAMILIES] + [(f, 'S2', 203, 20) for f in K.FAMILIES]
SAMPLE_SIZES = tuple(os.environ.get('CARS10B_SAMPLE_SIZES', 'S2,S3').split(','))   # override only for the dev harness test
OPTIONAL_L = [(f, 'S3', 202, 10) for f in K.FAMILIES]


def load_all(roots):
    recs = []
    for root in roots:
        for f in sorted(Path(root).rglob('*.jsonl')):
            if '_warmup' in f.parts or f.parent.name.endswith('_retry'):
                continue
            for l in f.read_text().splitlines():
                if l.strip():
                    r = json.loads(l); r['_dir'] = str(f.parent); r['_mode'] = 'MT' if '_MT' in str(f) else 'ST'
                    recs.append(r)
    return recs


def fam_of(key):
    split = K.seed_guard(key[2], 'heldout' if key[2] in K.HELDOUT_SEEDS else 'dev')   # dev only for the harness test
    if split == 'heldout':
        K.check_freeze()
    return Family3D(key[0], key[1], seed=key[2], split=split)


def exact_sample(out, roots):
    t0 = time.time(); recs = [r for r in load_all(roots) if r['status'] == 'CERTIFIED']
    rng = random.Random(10_2026); fams = {}; res = []
    for label, cfg in K.ARMS:
        meth = cfg.split('|')[1] if '|' in cfg else cfg
        for size in SAMPLE_SIZES:
            for mode, k in (('ST', 4), ('MT', 2)):
                pool = sorted((r for r in recs if r['method'] == meth and r['size'] == size and r['_mode'] == mode),
                              key=lambda r: (r['family'], r['seed'], r['repeat'], r['t']))
                for r in rng.sample(pool, min(k, len(pool))):
                    key = (r['family'], r['size'], r['seed'])
                    if key not in fams:
                        fams[key] = fam_of(key)
                    A, b, _ = fams[key].get(int(r['t'])); A = R6.ensure_canonical(A)
                    x = np.load(Path(r['_dir']) / r['solution_file'], allow_pickle=False)
                    s = time.time(); ex = I5.exact_residual_sq(A, b, x); R = float.fromhex(r['R_hi'])
                    ok = Fraction(R) ** 2 >= ex; exr = float(ex) ** 0.5
                    res.append(dict(arm=label, method=meth, mode=mode, family=r['family'], size=size, seed=r['seed'],
                                    repeat=r['repeat'], t=r['t'], residual_kind=r.get('residual_kind'), R=R, exact=exr,
                                    R_over_exact=(R / exr if exr > 0 else None), ok=bool(ok), seconds=time.time() - s))
                    print('E', res[-1], flush=True)
    summ = dict(residual_ok=sum(e['ok'] for e in res), residual_total=len(res), seconds=time.time() - t0)
    Path(out).write_text(json.dumps(dict(exact_residuals=res, summary=summ), indent=1))
    print('SUMMARY', summ)


def lambda_proof(out, fid, size, seed, t, roots):
    cl = [(float.fromhex(r['lambda_lo']), r['method'], r['_mode']) for r in load_all(roots)
          if r['status'] == 'CERTIFIED' and r['family'] == fid and r['size'] == size and r['seed'] == seed and r['t'] == t]
    lam, meth, mode = max(cl)
    A, _, _ = fam_of((fid, size, seed)).get(t); A = R6.ensure_canonical(A)
    s = time.time(); pr = I5.arb_congruence_pd(A, lam)
    res = dict(family=fid, size=size, seed=seed, t=t, n=int(A.shape[0]), claims=len(cl), max_lambda_lo=lam,
               max_claim_method=meth, max_claim_mode=mode, proved=bool(pr['proved']), lapack_lambda1=pr['lapack_lambda1'],
               ratio_max_claim_to_lambda1=lam / pr['lapack_lambda1'], seconds=time.time() - s)
    Path(out).write_text(json.dumps(res, indent=1))
    print('L', res, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'E':
        exact_sample(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == 'L':
        lambda_proof(sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6]), sys.argv[7:])
    else:
        raise SystemExit('E or L')
