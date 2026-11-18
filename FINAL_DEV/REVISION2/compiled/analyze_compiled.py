"""Revision 2, exploratory: analysis of the compiled-kernel experiment, exactly as fixed in COMPILED_PROTOCOL.md.
Validity: an arm in a cell is valid only if all 21 steps are certified and meet the target AND all its records are
VERIFIED by the frozen replay. Ratios within a cell (paired); unit = seed (median over valid repeats); geometric mean
over seeds with two-sided 95 % and 90 % t-intervals (df = seeds - 1). Descriptive category of every contrast with the
thresholds of the registered rule (applied post hoc, not confirmatory), and the TOST-style margin check.
Identity: for every frozen/compiled pair of arms in every cell, the certificate views computed by the frozen replay must
be identical (same decisions, same bounds), and so must the stored R_hi sequences.
Usage: python analyze_compiled.py -> analysis_compiled.json"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132}
PAIRS = {'E3-TS': [('B9c', 'Dc'), ('B9', 'D'), ('B1c', 'Dc'), ('Bc', 'Dc'), ('B9c', 'D'), ('B9', 'Dc'), ('A', 'Cc'), ('A', 'C')],
         'D3-TS': [('Bc', 'D2c'), ('B', 'D2'), ('B1c', 'D2c'), ('B9c', 'D2c'), ('Bc', 'D2'), ('B', 'D2c'), ('A', 'Cc'), ('A', 'C')]}
PRIMARY = {'E3-TS': ('B9c', 'Dc'), 'D3-TS': ('Bc', 'D2c')}
SPEEDUPS = [('B1', 'B1c'), ('B', 'Bc'), ('B9', 'B9c'), ('C', 'Cc'), ('D', 'Dc'), ('D2', 'D2c')]


def gm_ci(v):
    l = np.log(np.asarray(v, float)); m = l.mean(); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None, ci90=None)
    se = l.std(ddof=1) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)],
                ci90=[math.exp(m - T95[n - 1] * se), math.exp(m + T95[n - 1] * se)])


def category(s):
    g, (lo, hi) = s['gm'], s['ci95']
    if g >= 1.2 and lo > 1: return 'reuse faster, beyond the 1.2 threshold'
    if lo > 1: return 'reuse faster (directional)'
    if g <= 1 / 1.2 and hi < 1: return 'per-system faster, beyond the 1/1.2 threshold'
    if hi < 1: return 'per-system faster (directional)'
    return 'no directional difference (interval contains 1)'


def main():
    cells = {}
    for cj in sorted((HERE / 'runs').glob('*/cell_compiled_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay10b.json').read_text())
        arms = {}
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            arms[r['arm']] = dict(r, valid=(r['certified_target'] == 21 and ver == 21), verified=ver)
        r0 = d['results'][0]
        views = {k.split('|')[0]: h for k, h in rp['certificate_views'].items()}
        cells[(r0['family'], r0['size'], r0['seed'], r0['repeat'])] = dict(arms=arms, env=d['env'], views=views)
    out = dict(configs={}, cells=len(cells), identity=dict(pairs_checked=0, view_identical=0, R_identical=0, mismatches=[]))
    I = out['identity']
    for key, c in cells.items():
        for P, Q in SPEEDUPS:
            if P in c['arms'] and Q in c['arms']:
                I['pairs_checked'] += 1
                vi = c['views'].get(P) is not None and c['views'].get(P) == c['views'].get(Q)
                ri = c['arms'][P]['R_hi'] == c['arms'][Q]['R_hi']
                I['view_identical'] += vi; I['R_identical'] += ri
                if not (vi and ri):
                    I['mismatches'].append(dict(cell=list(key), pair=f'{P}/{Q}', view=vi, R=ri))
    for fam, size in sorted({(k[0], k[1]) for k in cells}):
        C = {k: v for k, v in cells.items() if k[:2] == (fam, size)}
        res = dict(pairs={}, speedups={}, arms={}, invalid=[], repeats=sorted({k[3] for k in C}), seeds=sorted({k[2] for k in C}))
        for k, c in C.items():
            for a, x in c['arms'].items():
                if not x['valid']:
                    res['invalid'].append(dict(cell=list(k), arm=a, certified_target=x['certified_target'], verified=x['verified']))

        def ratio(P, Q):
            per = defaultdict(list)
            for k, c in C.items():
                if P in c['arms'] and Q in c['arms'] and c['arms'][P]['valid'] and c['arms'][Q]['valid']:
                    per[k[2]].append(c['arms'][P]['measured_s'] / c['arms'][Q]['measured_s'])
            units = {str(s): float(np.median(v)) for s, v in sorted(per.items())}
            if not units:
                return None
            st = gm_ci(list(units.values()))
            if st['ci95']:
                st['category'] = category(st)
                st['within_margin_90'] = bool(1 / 1.2 < st['ci90'][0] and st['ci90'][1] < 1.2)
            return dict(per_seed=units, runs=sum(len(v) for v in per.values()), **st)
        for P, Q in PAIRS[fam]:
            r = ratio(P, Q)
            if r:
                res['pairs'][f'{P}/{Q}'] = r
        for P, Q in SPEEDUPS:
            r = ratio(P, Q)
            if r:
                res['speedups'][f'{P}/{Q}'] = r
        for a in sorted({a for c in C.values() for a in c['arms']}):
            xs = [c['arms'][a] for c in C.values() if a in c['arms']]
            per_seed = defaultdict(list)
            for k, c in C.items():
                if a in c['arms'] and c['arms'][a]['valid']:
                    per_seed[k[2]].append(c['arms'][a]['measured_s'])
            meds = [float(np.median(v)) for v in per_seed.values()]
            cv = [float(np.std(v, ddof=1) / np.mean(v)) for v in per_seed.values() if len(v) > 1]
            res['arms'][a] = dict(
                impl=xs[0]['impl'], config=xs[0]['config'], runs=len(xs), valid_runs=sum(x['valid'] for x in xs),
                cost_mean=float(np.mean(meds)) if meds else None, cost_gm=float(np.exp(np.mean(np.log(meds)))) if meds else None,
                components_mean={c: float(np.mean([x['components'][c] for x in xs])) for c in xs[0]['components']},
                pcg_iters_mean=float(np.mean([x['pcg_iters'] for x in xs])),
                eft_calls_mean=float(np.mean([x['eft_calls'] for x in xs])),
                refreshes_mean=float(np.mean([x['refreshes'] for x in xs])),
                factorizations_mean=float(np.mean([x['factorizations'] for x in xs])),
                bindings=sorted({b for x in xs for b in x['bindings']}),
                residual_kinds=sorted({b for x in xs for b in x['residual_kinds']}),
                repeat_cv_median=float(np.median(cv)) if cv else None,
                peak_wset_GiB_max=float(max(x['peak_wset_GiB'] for x in xs)))
        res['primary_exploratory'] = '/'.join(PRIMARY[fam])
        out['configs'][f'{fam}|{size}'] = res
    fp = [c['env'][k]['dense_mm_s'] for c in cells.values() for k in ('fingerprint_before', 'fingerprint_after')]
    out['fingerprint_dense_mm_s'] = [min(fp), max(fp)] if fp else None
    out['compiled_hash'] = sorted({c['env']['compiled_hash'] for c in cells.values()})
    (HERE / 'analysis_compiled.json').write_text(json.dumps(out, indent=1))
    print('identity', {k: v for k, v in I.items() if k != 'mismatches'}, 'mismatches', len(I['mismatches']))
    for cfg, r in out['configs'].items():
        print(cfg, 'invalid', len(r['invalid']), r['invalid'][:3])
        for pk, v in list(r['pairs'].items()) + list(r['speedups'].items()):
            print('  ', pk, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, v.get('category'),
                  'TOST' if v.get('within_margin_90') else '')
        for a, v in r['arms'].items():
            cm = v['components_mean']
            print('  ', a, v['impl'], round(v['cost_mean'] or 0, 3), 'binding', round(cm.get('binding', 0), 3),
                  'residual', round(cm.get('residual', 0), 3), 'valid', v['valid_runs'], '/', v['runs'])


if __name__ == '__main__':
    main()
