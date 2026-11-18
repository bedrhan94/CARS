"""Revision 2: pre-freeze checks of the fixed estimate fallback (fallback_runner.py). Untimed; no SIMP evaluation data.

1. Diff: the only difference between the frozen run_sequence_10a and run_sequence_fb is the estimate block (unified
   diff stored).
2. Phase 10B identity: on every ST S3 repeat-0 sequence (both families, seeds 200-204, already consumed), the carried-
   shift arms rebuilt with the fix (configs RPS-OPT and RPS-OPT-S09) must reproduce the certificate view of every stored
   B / B9 record exactly (the carried shift never failed there, so the fix must be inactive).
3. Forced-failure harness test (development seed 0, E3-TS S1): the carried shift is made to fail genuinely at steps
   5, 6 and 12 (the shift is multiplied by 1e8, so PARDISO reports a non-positive pivot and resets its session).
   Frozen code: records 'eigsh' (cold estimate). Fixed code: must record 'shift-invert', count the extra
   factorization and certify every step; its records are then replayed with the frozen Phase 10B verifier.
Writes check_fallback.json; exit 1 on failure."""
import os, sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import difflib, json, shutil, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402
import run_cell10b as C10                                            # noqa: E402
N = C10.N; R6 = C10.R6; v = C10.v
sys.path.insert(0, str(CARS / 'PHASE6_DEV' / 'lean6'))
import verify10a as VV                                               # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import fallback_runner as FB                                         # noqa: E402

RUN_FB, src_old, src_new = FB.build(N)
out = dict(diff=''.join(difflib.unified_diff(src_old.splitlines(True), src_new.splitlines(True), 'run_sequence_10a (frozen)',
                                             'run_sequence_fb')), identity={}, forced={}, failures=[])
work = HERE / 'check_runs'
if work.exists():
    shutil.rmtree(work)
ch = C10.code_hash()

# 2. Phase 10B identity
K.check_freeze()
RUNS = CARS / 'FINAL_DEV' / 'phase10b' / 'runs' / 'ST_S3'
for fam_id in K.FAMILIES:
    for seed in K.HELDOUT_SEEDS:
        fam = Family3D(fam_id, 'S3', seed=seed, split='heldout')
        for arm, cfg in (('B', 'RPS-OPT'), ('B9', 'RPS-OPT-S09')):
            stored = [json.loads(l) for l in next((RUNS / f'{fam_id}_S3_s{seed}_r0_ST' / arm).glob('*.jsonl')).read_text().splitlines()]
            _, rows = RUN_FB(fam, cfg, work / 'identity' / f'{fam_id}_{seed}' / arm, target=K.TARGET, repeat=0, run_order=0,
                             code_hash=ch, write_artifacts=False)
            same = [json.dumps(VV.V5.cert_view(a), sort_keys=True) == json.dumps(VV.V5.cert_view(b), sort_keys=True)
                    for a, b in zip(stored, rows)]
            key = f'{fam_id}|S3|{seed}|{arm}'
            out['identity'][key] = dict(records=len(rows), identical=sum(same), stored=len(stored),
                                        fallbacks=sum(bool(r['details'].get('estimate_fallback')) for r in rows))
            if not (all(same) and len(rows) == len(stored)):
                out['failures'].append(('identity', key))
            print(key, out['identity'][key], flush=True)


# 3. forced-failure harness test (development data)
class Forced:
    def __init__(self, fam, steps):
        self._f = fam; self._steps = set(steps); self.flag = False

    def __getattr__(self, k):
        return getattr(self._f, k)

    def get(self, t):
        self.flag = t in self._steps
        return self._f.get(t)


frozen_anchor = R6.rump_anchor_once
for label, runner in (('frozen', N.run_sequence_10a), ('fixed', RUN_FB)):
    for arm, cfg in (('B', 'RPS-OPT'), ('B9', 'RPS-OPT-S09')):
        fam = Forced(Family3D('E3-TS', 'S1', seed=0, split='dev'), (5, 6, 12))

        def forced_anchor(A, sigma, engine, tm, _fam=fam):
            if _fam.flag:                                             # the first Rump attempt of a forced step = the carry
                _fam.flag = False
                return frozen_anchor(A, sigma * 1e8, engine, tm)
            return frozen_anchor(A, sigma, engine, tm)
        R6.rump_anchor_once = forced_anchor
        try:
            d = work / 'forced' / label / arm
            _, rows = runner(fam, cfg, d, target=K.TARGET, repeat=0, run_order=0, code_hash=ch, write_artifacts=True)
        finally:
            R6.rump_anchor_once = frozen_anchor
        res = dict(certified=sum(r['status'] == 'CERTIFIED' for r in rows), records=len(rows),
                   carry_failures=[r['t'] for r in rows if r['details'].get('carry_failure')],
                   fallbacks={str(r['t']): r['details'].get('estimate_fallback') for r in rows if r['details'].get('estimate_fallback')},
                   factor_count={str(r['t']): r['factor_count'] for r in rows if r['t'] in (4, 5, 6, 12)},
                   failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}))
        out['forced'][f'{label}|{arm}'] = res
        print('FORCED', label, arm, res, flush=True)
        if label == 'fixed' and not (res['certified'] == 21 and set(res['fallbacks'].values()) == {'shift-invert'}
                                     and sorted(res['carry_failures']) == [5, 6, 12]):
            out['failures'].append(('forced', arm))

# replay of the forced fixed-arm records with the frozen Phase 10B verifier (development seed allowed)
rp = subprocess.run([sys.executable, '-B', str(CARS / 'FINAL_DEV' / 'phase10b' / 'replay10b.py'), str(work / 'forced' / 'fixed')],
                    capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
rj = json.loads((work / 'forced' / 'fixed' / 'replay10b.json').read_text())
out['forced_replay'] = dict(totals=rj['totals'], errors=len(rj['errors']), rump_reproofs=rj['rump_reproofs'])
if set(rj['totals']) != {'VERIFIED'} or rj['errors']:
    out['failures'].append(('forced_replay', rj['totals']))
out['ALL_OK'] = not out['failures']
(HERE / 'check_fallback.json').write_text(json.dumps(out, indent=1))
print('ALL_OK', out['ALL_OK'], out['failures'], out['forced_replay'])
sys.exit(0 if out['ALL_OK'] else 1)
