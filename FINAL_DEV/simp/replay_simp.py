"""SIMP3D replay: every record of every arm in a cell directory through the frozen Phase 10A verifier (verify10a), with
the family rebuilt from the frozen SIMP window (identical logic to FINAL_DEV/phase10b/replay10b.py).
Usage: python replay_simp.py <frozen_dir> <size> <cell_dir>  -> <cell_dir>/replay_simp.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import simp_family as SF                                             # noqa: E402
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'PHASE10A_DEV' / 'methods'))
import verify10a as VV                                               # noqa: E402
from replay6 import MemoFamily                                       # noqa: E402


def main():
    frozen, size, root = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    t0 = time.time(); res = {'arms': {}, 'errors': [], 'totals': defaultdict(int)}; fams = {}
    for f in sorted(root.rglob('*.jsonl')):
        if '_warmup' in f.parts:
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        r0 = recs[0]; window = r0['split'].split('-', 1)[1]
        inst = json.loads((root / next(root.glob('cell_*.json')).name).read_text())['env']['inst']
        key = (inst, window)
        if key not in fams:
            fams[key] = MemoFamily(SF.SimpFamily(frozen, size, inst, window))
        a = res['arms'].setdefault(f.parent.name, defaultdict(int))
        for r in recs:
            st, errs = VV.verify_record10a(r, fams[key], f.parent, exact_n=0)
            a[st] += 1; res['totals'][st] += 1
            if r['status'] == 'CERTIFIED':
                a['certified'] += 1; a['target_met'] += int(bool(r['target_met']))
            if st != 'VERIFIED':
                res['errors'].append(dict(file=f.name, t=r['t'], status=st, errors=errs))
    res['rump_reproofs'] = dict(VV.V5.STATS); res['seconds'] = time.time() - t0
    res['arms'] = {k: dict(v) for k, v in res['arms'].items()}; res['totals'] = dict(res['totals'])
    (root / 'replay_simp.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ('totals', 'rump_reproofs', 'seconds')}), 'ERRORS', len(res['errors']), res['errors'][:3])


if __name__ == '__main__':
    main()
