"""Replay every record in a cell directory with the frozen verifier against the FROZEN held-out inputs (FrozenFamily),
on any platform. Used for cross-platform checks: records produced on one platform are verified on another.
Usage: python replay_frozen.py <cell_dir> [<result_name>]"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, platform, time
from collections import defaultdict
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b')); sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(CARS / 'PHASE10A_DEV' / 'methods')); sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402
from replay6 import MemoFamily                                       # noqa: E402
from frozen_family import FrozenFamily                               # noqa: E402


def main():
    root = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else 'replay_frozen.json'
    t0 = time.time(); K.check_freeze()
    res = dict(platform=platform.platform(), totals=defaultdict(int), errors=[], arms=defaultdict(lambda: defaultdict(int)))
    fams = {}
    for f in sorted(root.rglob('*.jsonl')):
        if '_warmup' in f.parts:
            continue
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        key = (recs[0]['family'], recs[0]['size'], recs[0]['seed'])
        if key not in fams:
            fams[key] = MemoFamily(FrozenFamily(CARS / 'FINAL_DEV/phase10b/inputs/pass_a', *key))
        for r in recs:
            st, errs = VV.verify_record10a(r, fams[key], f.parent, exact_n=0)
            res['totals'][st] += 1; res['arms'][f.parent.name][st] += 1
            if st != 'VERIFIED':
                res['errors'].append(dict(file=f.name, t=r['t'], status=st, errors=errs))
    res['input_mismatches'] = {str(k): v._f.mismatches for k, v in fams.items() if v._f.mismatches}
    res['rump_reproofs'] = dict(VV.V5.STATS); res['seconds'] = time.time() - t0
    out = dict(res, totals=dict(res['totals']), arms={k: dict(v) for k, v in res['arms'].items()})
    (root / name).write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(platform=out['platform'], totals=out['totals'], errors=len(out['errors']),
                          input_mismatches=len(out['input_mismatches']), seconds=round(out['seconds'], 1))))


if __name__ == '__main__':
    main()
