"""Phase 6 replay of ONE cell directory: EVERY record of EVERY repeat is verified with the Phase 5 verifier
(PHASE5_DEV/lean/verify5.py, imported unchanged; frozen Phase 3B verifier for M0 records; cross-engine CHOLMOD re-proof
of every distinct Rump anchor). Only addition: the problem generator's get(t) is memoised per process and returns
COPIES (inputs are regenerated deterministically and checked against the record's SHA-256 by the verifier anyway).
Also reports P2 identity (certificate content identical across repeats, per configuration).
Usage: python replay6.py <run_plan_dir> <cell_name> [exact_n] [--cpu k]   -> <cell>/replay6.json"""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[k] = '1'
sys.dont_write_bytecode = True
import json, time
from collections import defaultdict
from pathlib import Path
import psutil
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'PHASE5_DEV' / 'lean'))
import verify5 as V                                                    # noqa: E402  Phase 5 verifier, unchanged
from cars_m2.families import Family3D                                  # noqa: E402  frozen generator


def load(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


class MemoFamily:
    def __init__(self, fam):
        self._f = fam; self._c = {}

    def get(self, t):
        if t not in self._c:
            self._c[t] = self._f.get(t)
        A, b, meta = self._c[t]
        return A.copy(), b.copy(), {k: (x.copy() if hasattr(x, 'copy') else x) for k, x in meta.items()}

    def __getattr__(self, name):
        return getattr(self._f, name)


def main():
    root = Path(sys.argv[1]); cell = root / sys.argv[2]
    exact_n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith('--') else 0
    if '--cpu' in sys.argv:
        psutil.Process().cpu_affinity([int(sys.argv[sys.argv.index('--cpu') + 1])])
    fam_id, size, s, mode = cell.name.split('_'); seed = int(s[1:])
    if seed not in (0, 1, 2):
        raise SystemExit('dev seeds only')
    t0 = time.time()
    fam = MemoFamily(Family3D(fam_id, size, seed=seed, split='dev'))
    runs = defaultdict(dict)
    for eng_dir in sorted(d for d in cell.iterdir() if d.is_dir()):
        for f in eng_dir.glob('*.jsonl'):
            recs = load(f); r0 = recs[0]
            cfg = '|'.join((r0['engine'], r0['method'], r0['track'], r0['binding']))
            runs[cfg][int(r0['repeat'])] = (recs, eng_dir)
    res = {'cell': cell.name, 'mode': mode, 'configs': {}, 'totals': defaultdict(int), 'errors': []}
    for cfg, reps in sorted(runs.items()):
        cnt = defaultdict(int); per_rep = {}
        for rep, (recs, base) in sorted(reps.items()):
            c2 = defaultdict(int)
            for r in recs:
                o, e = V.verify_record5(r, fam, base, exact_n=exact_n)
                cnt[o] += 1; c2[o] += 1; res['totals'][o] += 1
                if r['status'] == 'CERTIFIED':
                    res['totals']['certified_records'] += 1
                if o != 'VERIFIED':
                    res['errors'].append([cfg, rep, r['t'], o, e[:3]])
            per_rep[rep] = dict(c2, records=len(recs), certified=sum(r['status'] == 'CERTIFIED' for r in recs),
                                target_met=sum(bool(r.get('target_met')) for r in recs))
        r0 = reps[min(reps)][0]
        p2 = all([V.cert_view(a) for a in reps[k][0]] == [V.cert_view(a) for a in r0] for k in reps)
        res['configs'][cfg] = dict(dict(cnt), repeats=sorted(reps), per_repeat=per_rep, p2_identical=p2)
        print(cfg, json.dumps(res['configs'][cfg]), flush=True)
    res['totals'] = dict(res['totals']); res['stats'] = dict(V.STATS); res['wall_s'] = time.time() - t0
    res['exact_n'] = exact_n
    (cell / 'replay6.json').write_text(json.dumps(res, indent=1))
    print('TOTALS', res['totals'], 'stats', res['stats'], 'wall', round(res['wall_s'], 1), flush=True)
    sys.exit(2 if res['totals'].get('REJECTED') else 0)


if __name__ == '__main__':
    main()
