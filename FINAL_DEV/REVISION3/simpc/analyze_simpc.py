"""Revision 3, exploratory: analysis of the SIMP3D evaluation with compiled operations, as fixed in SIMPC_PROTOCOL.md.
Per window: unit = instance (median over the valid repeats); GM over the five instances with a 95 % t-interval (df = 4).
Validity (parity): all 21 steps certified and target-met AND all records VERIFIED by the frozen SIMP replay; a ratio
uses only cells in which both arms are valid. Pairs: B9f/D and Bf/D2 (frozen operations), B9fc/Dc and Bfc/D2c (compiled
operations), the same four pairs with the first-step eigenvalue estimate (a cold computation common to all arms)
subtracted from both costs, and the speed-ups of the compiled arms. Identity: frozen and compiled versions of the same
arm must store identical R_hi and lambda_lo sequences. Descriptive only.
Usage: python analyze_simpc.py -> analysis_simpc.json"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PAIRS = [('B9f', 'D'), ('Bf', 'D2'), ('B9fc', 'Dc'), ('Bfc', 'D2c'), ('B9f', 'B9fc'), ('Bf', 'Bfc'), ('D', 'Dc'), ('D2', 'D2c')]
XPAIRS = [('B9f', 'D'), ('Bf', 'D2'), ('B9fc', 'Dc'), ('Bfc', 'D2c')]


def gm_ci(v):
    l = np.log(np.asarray(v, float)); m = l.mean(); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None)
    se = l.std(ddof=1) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)])


def category(s):
    g, (lo, hi) = s['gm'], s['ci95']
    if g >= 1.2 and lo > 1: return 'reuse faster, beyond the 1.2 threshold'
    if lo > 1: return 'reuse faster (directional)'
    if g <= 1 / 1.2 and hi < 1: return 'per-system faster, beyond the 1/1.2 threshold'
    if hi < 1: return 'per-system faster (directional)'
    return 'no directional difference (interval contains 1)'


def ratio(C, A, B, key):
    per = defaultdict(list)
    for k, arms in C.items():
        if arms[A]['valid'] and arms[B]['valid']:
            per[k[1]].append(key(arms[A]) / key(arms[B]))
    units = {i: float(np.median(v)) for i, v in sorted(per.items())}
    if not units:
        return None
    st = gm_ci(list(units.values()))
    if st['ci95']:
        st['category'] = category(st)
    return dict(per_instance=units, runs=sum(len(v) for v in per.values()), **st)


def main():
    cells = {}
    for cj in sorted((HERE / 'runs').glob('*/cell_SIMP3D_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay_simp.json').read_text())
        arms = {}
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            arms[r['arm']] = dict(r, valid=(r['certified_target'] == 21 and ver == 21), verified=ver)
        r0 = d['results'][0]
        cells[(r0['window'], r0['inst'], r0['repeat'])] = arms
    out = dict(cells=len(cells), windows={}, identity=dict(pairs_checked=0, identical=0, mismatches=[]))
    I = out['identity']
    for key, arms in cells.items():
        for a, b in (('Bf', 'Bfc'), ('B9f', 'B9fc'), ('D', 'Dc'), ('D2', 'D2c')):
            I['pairs_checked'] += 1
            same = arms[a]['R_hi'] == arms[b]['R_hi'] and arms[a]['lambda_lo'] == arms[b]['lambda_lo']
            I['identical'] += same
            if not same:
                I['mismatches'].append(dict(cell=list(key), pair=f'{a}/{b}'))
    for w in ('early', 'late'):
        C = {k: v for k, v in cells.items() if k[0] == w}
        res = dict(pairs={}, pairs_excl_first_estimate={}, arms={}, runs=len(C), invalid=[])
        for k, arms in C.items():
            for a, x in arms.items():
                if not x['valid']:
                    res['invalid'].append(dict(cell=list(k), arm=a, certified_target=x['certified_target'], verified=x['verified']))
        for A, B in PAIRS:
            r = ratio(C, A, B, lambda x: x['measured_s'])
            if r:
                res['pairs'][f'{A}/{B}'] = r
        for A, B in XPAIRS:
            r = ratio(C, A, B, lambda x: x['measured_s'] - x['first_estimate_s'])
            if r:
                res['pairs_excl_first_estimate'][f'{A}/{B}'] = r
        for a in sorted({a for v in C.values() for a in v}):
            xs = [arms[a] for arms in C.values()]
            per_inst = defaultdict(list)
            for k, arms in C.items():
                if arms[a]['valid']:
                    per_inst[k[1]].append(arms[a]['measured_s'])
            res['arms'][a] = dict(
                level=xs[0]['level'], runs=len(xs), valid_runs=sum(x['valid'] for x in xs),
                certified_target=sum(x['certified_target'] for x in xs), records=sum(x['records'] for x in xs),
                cost_mean=float(np.mean([np.median(v) for v in per_inst.values()])) if per_inst else None,
                first_estimate_mean=float(np.mean([x['first_estimate_s'] for x in xs])),
                components_mean={c: float(np.mean([x['components'][c] for x in xs])) for c in xs[0]['components']},
                carry_failures=sum(x['carry_failures'] for x in xs),
                shift_invert=sum(f == 'shift-invert' for x in xs for f in x['estimate_fallbacks']),
                refreshes_mean=float(np.mean([x['refreshes'] for x in xs])),
                pcg_iters_mean=float(np.mean([x['pcg_iters'] for x in xs])))
        out['windows'][w] = res
    (HERE / 'analysis_simpc.json').write_text(json.dumps(out, indent=1))
    print('identity', {k: v for k, v in I.items() if k != 'mismatches'})
    for w, r in out['windows'].items():
        print(w, 'runs', r['runs'], 'invalid', len(r['invalid']))
        for pk, v in list(r['pairs'].items()) + [('x ' + k, v) for k, v in r['pairs_excl_first_estimate'].items()]:
            print('  ', pk, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, v.get('category'))
        for a, v in r['arms'].items():
            cm = v['components_mean']
            print('  ', a, v['level'], round(v['cost_mean'] or 0, 3), 'binding', round(cm.get('binding', 0), 3),
                  'first_est', round(v['first_estimate_mean'], 2), 'carry', v['carry_failures'], 'valid', v['valid_runs'])


if __name__ == '__main__':
    main()
