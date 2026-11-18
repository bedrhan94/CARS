"""Revision 2, post hoc (referee comment: only 10 of the 84 pre-specified exact-residual records used the accurate bound).

An additional exact-residual sample drawn ONLY from certified records whose residual bound is the accurate (EFT) bound.
Pool: all confirmatory cells (ST and MT, S2 and S3), records with residual_kind == 'eft'. Strata: family x size x mode x
arm; up to 12 records per stratum, drawn with random.Random(20261116) from the sorted pool. For each record the stored
solution vector is re-read (the SHA-256 of its .npy file is checked against the record), the exact rational ||b - A x||_2^2 is computed
with the unchanged Phase 5 module (independent5.exact_residual_sq, Python Fractions) and compared with R_hi^2.
Writes exact_eft_sample.json."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import hashlib, json, random, time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(CARS / 'PHASE5_DEV' / 'arith')); sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import independent5 as I5                                            # noqa: E402
import runner6 as R6                                                 # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402

RUNS = CARS / 'FINAL_DEV' / 'phase10b' / 'runs'
PER_STRATUM = 12


def main():
    t0 = time.time()
    K.check_freeze()
    pool = defaultdict(list)
    for cfg in ('ST_S2', 'ST_S3', 'MT_S2', 'MT_S3'):
        mode = cfg.split('_')[0]
        for f in sorted((RUNS / cfg).glob('*/*/*.jsonl')):
            if '_warmup' in f.parts:
                continue
            arm = f.parent.name
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r['status'] == 'CERTIFIED' and r.get('residual_kind') == 'eft':
                    r['_dir'] = str(f.parent); r['_arm'] = arm; r['_mode'] = mode
                    pool[(r['family'], r['size'], mode, arm)].append(r)
    rng = random.Random(20261116); fams = {}; res = []
    for key in sorted(pool):
        recs = sorted(pool[key], key=lambda r: (r['seed'], r['repeat'], r['t']))
        for r in rng.sample(recs, min(PER_STRATUM, len(recs))):
            fk = (r['family'], r['size'], r['seed'])
            if fk not in fams:
                fams[fk] = Family3D(r['family'], r['size'], seed=r['seed'], split='heldout')
            A, b, _ = fams[fk].get(int(r['t'])); A = R6.ensure_canonical(A)
            xp = Path(r['_dir']) / r['solution_file']
            x = np.load(xp, allow_pickle=False)
            sha_ok = hashlib.sha256(xp.read_bytes()).hexdigest() == r['solution_sha256']      # records hash the .npy file
            s = time.time(); ex = I5.exact_residual_sq(A, b, x); R = float.fromhex(r['R_hi'])
            ok = Fraction(R) ** 2 >= ex; exr = float(ex) ** 0.5
            res.append(dict(family=r['family'], size=r['size'], mode=key[2], arm=key[3], seed=r['seed'], repeat=r['repeat'],
                            t=r['t'], R=R, exact=exr, R_over_exact=(R / exr if exr > 0 else None), ok=bool(ok),
                            solution_sha_ok=bool(sha_ok), seconds=time.time() - s))
            print(res[-1], flush=True)
    rat = [e['R_over_exact'] for e in res if e['R_over_exact']]
    summ = dict(pool_strata={'|'.join(k): len(v) for k, v in sorted(pool.items())}, pool_total=sum(len(v) for v in pool.values()),
                sampled=len(res), enclosed=sum(e['ok'] for e in res), solution_sha_ok=sum(e['solution_sha_ok'] for e in res),
                ratio_min=float(min(rat)), ratio_median=float(np.median(rat)), ratio_max=float(max(rat)), seconds=time.time() - t0)
    (HERE / 'exact_eft_sample.json').write_text(json.dumps(dict(records=res, summary=summ), indent=1))
    print('SUMMARY', json.dumps(summ))


if __name__ == '__main__':
    main()
