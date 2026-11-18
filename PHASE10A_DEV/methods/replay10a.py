"""Phase 10A replay: EVERY record of EVERY arm and repeat in a run directory is verified (verify10a: frozen verify5 for
arms A/C, the EFT-aware rules for the new arms; CHOLMOD cross-engine re-proof of every distinct Rump anchor). Also
reports whether certificate content is identical across repeats per arm (determinism).
Usage: python replay10a.py <run_dir> [exact_n] [exact_stride]   -> <run_dir>/replay10a.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402
from replay6 import MemoFamily                                       # noqa: E402  (unchanged Phase 6 helper)
from cars_m2.families import Family3D                                # noqa: E402


def main():
    root = Path(sys.argv[1]); exact_n = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1          # exact checks only for t % stride == 0
    t0 = time.time()
    fams = {}; res = {'arms': {}, 'errors': [], 'totals': defaultdict(int)}
    views = defaultdict(dict)
    for f in sorted(root.rglob('*.jsonl')):
        if '_warmup' in f.parts:
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        r0 = recs[0]
        if r0['seed'] not in (0, 1, 2) or r0['split'] != 'dev':
            raise SystemExit('dev seeds only')
        key = (r0['family'], r0['size'], r0['seed'])
        if key not in fams:
            fams[key] = MemoFamily(Family3D(r0['family'], r0['size'], seed=r0['seed'], split='dev'))
        arm = f.parent.name
        a = res['arms'].setdefault(arm, defaultdict(int))
        cv = []
        for r in recs:
            st, errs = VV.verify_record10a(r, fams[key], f.parent, exact_n=(exact_n if int(r['t']) % stride == 0 else 0))
            if exact_n and int(r['t']) % stride == 0 and r['status'] == 'CERTIFIED' and r['n'] <= exact_n:
                a['exact_checked'] += 1
            a[st] += 1; res['totals'][st] += 1
            if r['status'] == 'CERTIFIED':
                a['certified'] += 1; a['target_met'] += int(bool(r['target_met']))
            if st != 'VERIFIED':
                res['errors'].append(dict(file=f.name, t=r['t'], status=st, errors=errs))
            cv.append(VV.V5.cert_view(r))
        views[(arm,) + key][int(r0['repeat'])] = json.dumps(cv, sort_keys=True)
    ident = {}
    for k, reps in views.items():
        vals = list(reps.values())
        ident['|'.join(map(str, k))] = dict(repeats=len(vals), identical=all(x == vals[0] for x in vals))
    res['determinism'] = ident
    res['rump_reproofs'] = dict(VV.V5.STATS)
    res['seconds'] = time.time() - t0
    res['arms'] = {k: dict(v) for k, v in res['arms'].items()}; res['totals'] = dict(res['totals'])
    (root / 'replay10a.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('arms', 'totals', 'rump_reproofs', 'seconds')}, indent=1))
    print('ERRORS', len(res['errors']), res['errors'][:5])
    print('DETERMINISM', sum(1 for x in ident.values() if x['identical']), '/', len(ident))


if __name__ == '__main__':
    main()
