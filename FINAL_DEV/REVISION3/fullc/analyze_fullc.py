"""Revision 3, exploratory: analysis of the full-compilation experiment, as fixed in FULLC_PROTOCOL.md.
Validity: an arm in a cell is valid only if all 21 steps are certified and meet the target AND all its records are
VERIFIED by the frozen replay. Ratios within a cell; unit = seed (median over valid repeats); GM over seeds with
two-sided 95 % and 90 % t-intervals (df = seeds - 1); descriptive categories with the thresholds of the pre-specified
rule, and the margin check. Identity: arms that differ only in compiled operations ('partial' vs 'full') must have
identical replay certificate views and identical stored R_hi sequences. Also PARDISO analysis counts and components.
Usage: python analyze_fullc.py -> analysis_fullc.json"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132}
ROLE = {'E3-TS': dict(P='B9', Q='D'), 'D3-TS': dict(P='B', Q='D2')}
PRIMARY_A = {'E3-TS': ('B9f', 'Df'), 'D3-TS': ('Bf', 'D2f')}
PRIMARY_C = {'E3-TS': ('B9fs', 'Dfs'), 'D3-TS': ('Bfs', 'D2fs')}


def pairs(fam):
    P, Q = ROLE[fam]['P'], ROLE[fam]['Q']
    return [(P + 'f', Q + 'f'), (P + 'fs', Q + 'fs'), (P + 'c', Q + 'c'), (P + 'fs', Q + 'f'),
            (P + 'c', P + 'f'), (Q + 'c', Q + 'f'), (P + 'f', P + 'fs'), (Q + 'f', Q + 'fs')]


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
    for cj in sorted((HERE / 'runs').glob('*/cell_fullc_*.json')):
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
        P, Q = ROLE[key[0]]['P'], ROLE[key[0]]['Q']
        for a, b in ((P + 'c', P + 'f'), (Q + 'c', Q + 'f')):
            I['pairs_checked'] += 1
            vi = c['views'].get(a) is not None and c['views'].get(a) == c['views'].get(b)
            ri = c['arms'][a]['R_hi'] == c['arms'][b]['R_hi']
            I['view_identical'] += vi; I['R_identical'] += ri
            if not (vi and ri):
                I['mismatches'].append(dict(cell=list(key), pair=f'{a}/{b}', view=vi, R=ri))
    for fam, size in sorted({(k[0], k[1]) for k in cells}):
        C = {k: v for k, v in cells.items() if k[:2] == (fam, size)}
        res = dict(pairs={}, arms={}, invalid=[], repeats=sorted({k[3] for k in C}), seeds=sorted({k[2] for k in C}))
        for k, c in C.items():
            for a, x in c['arms'].items():
                if not x['valid']:
                    res['invalid'].append(dict(cell=list(k), arm=a, certified_target=x['certified_target'], verified=x['verified']))
        for A, B in pairs(fam):
            per = defaultdict(list)
            for k, c in C.items():
                if c['arms'][A]['valid'] and c['arms'][B]['valid']:
                    per[k[2]].append(c['arms'][A]['measured_s'] / c['arms'][B]['measured_s'])
            units = {str(s): float(np.median(v)) for s, v in sorted(per.items())}
            if units:
                st = gm_ci(list(units.values()))
                if st['ci95']:
                    st['category'] = category(st)
                    st['within_margin_90'] = bool(1 / 1.2 < st['ci90'][0] and st['ci90'][1] < 1.2)
                res['pairs'][f'{A}/{B}'] = dict(per_seed=units, runs=sum(len(v) for v in per.values()), **st)
        for a in sorted({a for c in C.values() for a in c['arms']}):
            xs = [c['arms'][a] for c in C.values()]
            per_seed = defaultdict(list)
            for k, c in C.items():
                if c['arms'][a]['valid']:
                    per_seed[k[2]].append(c['arms'][a]['measured_s'])
            meds = [float(np.median(v)) for v in per_seed.values()]
            pz = [x['pardiso'] for x in xs]
            res['arms'][a] = dict(
                level=xs[0]['level'], session=xs[0]['session'], config=xs[0]['config'], runs=len(xs),
                valid_runs=sum(x['valid'] for x in xs), cost_mean=float(np.mean(meds)) if meds else None,
                components_mean={cc: float(np.mean([x['components'][cc] for x in xs])) for cc in xs[0]['components']},
                pcg_iters_mean=float(np.mean([x['pcg_iters'] for x in xs])),
                eft_calls_mean=float(np.mean([x['eft_calls'] for x in xs])),
                factorizations_mean=float(np.mean([x['factorizations'] for x in xs])),
                analyses_mean=float(np.mean([p['analyses'] for p in pz])),
                t_analyse_mean=float(np.mean([p['t_analyse'] for p in pz])),
                t_numeric_mean=float(np.mean([p['t_numeric'] for p in pz])),
                t_envelope_mean=float(np.mean([p['t_envelope'] for p in pz])),
                peak_wset_GiB_max=float(max(x['peak_wset_GiB'] for x in xs)))
        res['primary_A'] = '/'.join(PRIMARY_A[fam]); res['primary_C'] = '/'.join(PRIMARY_C[fam])
        out['configs'][f'{fam}|{size}'] = res
    fp = [c['env'][k]['dense_mm_s'] for c in cells.values() for k in ('fingerprint_before', 'fingerprint_after')]
    out['fingerprint_dense_mm_s'] = [min(fp), max(fp)] if fp else None
    (HERE / 'analysis_fullc.json').write_text(json.dumps(out, indent=1))
    print('identity', {k: v for k, v in I.items() if k != 'mismatches'}, 'mismatches', len(I['mismatches']))
    for cfg, r in out['configs'].items():
        print(cfg, 'invalid', len(r['invalid']), r['invalid'][:3])
        for pk, v in r['pairs'].items():
            print('  ', pk, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, v.get('category'),
                  'margin' if v.get('within_margin_90') else '')
        for a, v in r['arms'].items():
            cm = v['components_mean']
            print('  ', a, v['level'], v['session'], round(v['cost_mean'] or 0, 3), 'shift', round(cm.get('shift', 0), 3),
                  'binding', round(cm.get('binding', 0), 3), 'resid', round(cm.get('residual', 0), 3), 'ana', round(v['analyses_mean'], 1),
                  'valid', v['valid_runs'], '/', v['runs'])


if __name__ == '__main__':
    main()
