"""Phase 10B replay: EVERY record of EVERY arm in a cell directory is verified with the frozen Phase 10A verifier
(verify10a.verify_record10a: frozen verify5 for arms A/C, the EFT-aware rules for the new arms; CHOLMOD cross-engine
re-proof of every distinct Rump anchor; bitwise EFT recomputation). Identical logic to PHASE10A_DEV/methods/replay10a.py;
the only differences are (i) the family is rebuilt with the split recorded in the records (held-out seeds 200-204
require the pre-registration freeze), and (ii) every regenerated input is also checked against the frozen manifest.
Usage: python replay10b.py <cell_dir>   -> <cell_dir>/replay10b.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(K.CARS / 'PHASE10A_DEV' / 'methods')); sys.path.insert(0, str(K.CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402  frozen Phase 10A verifier
from replay6 import MemoFamily                                       # noqa: E402  frozen Phase 6 helper
from cars_m2.families import Family3D                                # noqa: E402


def main():
    root = Path(sys.argv[1])
    t0 = time.time()
    fams = {}; res = {'arms': {}, 'errors': [], 'totals': defaultdict(int), 'manifest_mismatches': []}
    man = None
    views = defaultdict(dict)
    for f in sorted(root.rglob('*.jsonl')):
        if '_warmup' in f.parts:
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        r0 = recs[0]
        split = K.seed_guard(int(r0['seed']), r0['split'])
        if split == 'heldout' and man is None:
            K.check_freeze(); man = K.check_inputs_frozen()['manifest']
        key = (r0['family'], r0['size'], r0['seed'])
        if key not in fams:
            fams[key] = MemoFamily(Family3D(r0['family'], r0['size'], seed=r0['seed'], split=split))
        arm = f.parent.name
        a = res['arms'].setdefault(arm, defaultdict(int))
        cv = []
        for r in recs:
            if man is not None:
                row = man[f"{r['family']}|{r['size']}|{r['seed']}"]['rows'][int(r['t'])]
                if r['matrix_sha256'] != row['matrix_sha256'] or r['rhs_sha256'] != row['rhs_sha256'] or \
                        (r.get('coeff_sha256') is not None and r['coeff_sha256'] != row['coeff_sha256']):
                    res['manifest_mismatches'].append(dict(file=f.name, t=r['t']))
            st, errs = VV.verify_record10a(r, fams[key], f.parent, exact_n=0)
            a[st] += 1; res['totals'][st] += 1
            if r['status'] == 'CERTIFIED':
                a['certified'] += 1; a['target_met'] += int(bool(r['target_met']))
            if st != 'VERIFIED':
                res['errors'].append(dict(file=f.name, t=r['t'], status=st, errors=errs))
            cv.append(VV.V5.cert_view(r))
        views[(arm,) + key][int(r0['repeat'])] = json.dumps(cv, sort_keys=True)
    res['certificate_views'] = {'|'.join(map(str, k)): {str(rep): __import__('hashlib').sha256(s.encode()).hexdigest()
                                                         for rep, s in reps.items()} for k, reps in views.items()}
    res['rump_reproofs'] = dict(VV.V5.STATS)
    res['seconds'] = time.time() - t0
    res['arms'] = {k: dict(v) for k, v in res['arms'].items()}; res['totals'] = dict(res['totals'])
    (root / 'replay10b.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('arms', 'totals', 'rump_reproofs', 'seconds')}))
    print('ERRORS', len(res['errors']), res['errors'][:5], 'MANIFEST_MISMATCHES', len(res['manifest_mismatches']))


if __name__ == '__main__':
    main()
