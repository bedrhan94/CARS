"""Replay, from a clean copy of the artifact, every record whose solution vector is shipped (the primary arms of the
ST S3 repeat-0 cells), with the frozen Phase 10A verifier and the frozen inputs. Independent of the original machine's
paths; on a platform whose float assembly differs, the input check reports the mismatch instead of verifying.
Usage: python FINAL_DEV/release_tools/replay_subset.py   -> FINAL_DEV/release_tools/replay_subset_result.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
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
    t0 = time.time(); K.check_freeze()
    runs = CARS / 'FINAL_DEV' / 'phase10b' / 'runs' / 'ST_S3'
    res = dict(totals=defaultdict(int), errors=[], input_mismatches=[], arms=defaultdict(lambda: defaultdict(int)))
    fams = {}
    for f in sorted(runs.rglob('*.jsonl')):
        if '_warmup' in f.parts or not any((f.parent / 'artifacts').glob('*.npy')):
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
    for key, fam in fams.items():
        if fam._f.mismatches:
            res['input_mismatches'].append(dict(key=key, mismatches=fam._f.mismatches[:5]))
    res['seconds'] = time.time() - t0
    out = dict(res, totals=dict(res['totals']), arms={k: dict(v) for k, v in res['arms'].items()})
    (HERE / 'replay_subset_result.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(dict(totals=out['totals'], arms=out['arms'], errors=len(out['errors']),
                          input_mismatches=len(out['input_mismatches']), seconds=round(out['seconds'], 1))))


if __name__ == '__main__':
    main()
