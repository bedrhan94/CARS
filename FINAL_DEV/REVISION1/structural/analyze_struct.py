"""Revision 1, exploratory: analysis of the structural-envelope experiment, exactly as fixed in STRUCTURAL_PROTOCOL.md.
Parity: an arm in a cell is valid only if all 21 steps are certified and meet the target AND all its records are
VERIFIED by the frozen replay. Ratios are formed within a cell (paired), unit = seed (median over the valid repeats),
geometric mean over seeds with a two-sided 95 % t-interval (df = seeds - 1). Descriptive categories (exploratory, same
thresholds as the registered rule): see CATEGORY(). Mechanism: PARDISO analyses and phase times, envelope cost, factor
fill, components, iterations, residual-bound calls, memory and repeat CV.
Usage: python analyze_struct.py -> analysis_struct.json"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PAIRS = {'E3-TS': [('B9', 'D'), ('B9s', 'Ds'), ('B9s', 'D'), ('B9', 'B9s'), ('D', 'Ds')],
         'D3-TS': [('B', 'D2'), ('Bs', 'D2s'), ('Bs', 'D2'), ('B', 'Bs'), ('D2', 'D2s')]}
PRIMARY = {'E3-TS': ('B9s', 'Ds'), 'D3-TS': ('Bs', 'D2s')}


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


def main():
    cells = {}
    for cj in sorted((HERE / 'runs').glob('*/cell_struct_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay10b.json').read_text())
        arms = {}
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            arms[r['arm']] = dict(r, valid=(r['certified_target'] == 21 and ver == 21), verified=ver)
        r0 = d['results'][0]
        cells[(r0['family'], r0['size'], r0['seed'], r0['repeat'])] = dict(arms=arms, env=d['env'])
    out = dict(configs={}, cells=len(cells))
    for fam, size in sorted({(k[0], k[1]) for k in cells}):
        C = {k: v for k, v in cells.items() if k[:2] == (fam, size)}
        res = dict(pairs={}, arms={}, invalid=[], repeats=sorted({k[3] for k in C}), seeds=sorted({k[2] for k in C}))
        for k, c in C.items():
            for a, x in c['arms'].items():
                if not x['valid']:
                    res['invalid'].append(dict(cell=list(k), arm=a, certified_target=x['certified_target'], verified=x['verified']))
        for P, Q in PAIRS[fam]:
            per = defaultdict(list)
            for k, c in C.items():
                if P in c['arms'] and Q in c['arms'] and c['arms'][P]['valid'] and c['arms'][Q]['valid']:
                    per[k[2]].append(c['arms'][P]['measured_s'] / c['arms'][Q]['measured_s'])
            units = {str(s): float(np.median(v)) for s, v in sorted(per.items())}
            if units:
                st = gm_ci(list(units.values()))
                st['category'] = category(st) if st['ci95'] else None
                res['pairs'][f'{P}/{Q}'] = dict(per_seed=units, runs=sum(len(v) for v in per.values()), **st)
        for a in sorted({a for c in C.values() for a in c['arms']}):
            xs = [c['arms'][a] for c in C.values() if a in c['arms']]
            per_seed = defaultdict(list)
            for k, c in C.items():
                if a in c['arms']:
                    per_seed[k[2]].append(c['arms'][a]['measured_s'])
            cv = [float(np.std(v, ddof=1) / np.mean(v)) for v in per_seed.values() if len(v) > 1]
            pz = [x['pardiso'] for x in xs]
            res['arms'][a] = dict(
                session=xs[0]['session'], runs=len(xs), valid_runs=sum(x['valid'] for x in xs),
                cost_mean=float(np.mean([np.median(v) for v in per_seed.values()])),
                analyses_mean=float(np.mean([p['analyses'] for p in pz])),
                analyses_range=[int(min(p['analyses'] for p in pz)), int(max(p['analyses'] for p in pz))],
                t_analyse_mean=float(np.mean([p['t_analyse'] for p in pz])),
                t_numeric_mean=float(np.mean([p['t_numeric'] for p in pz])),
                numeric_mean=float(np.mean([p['numeric'] for p in pz])),
                t_per_analysis_ms=1000 * float(np.sum([p['t_analyse'] for p in pz]) / max(1, np.sum([p['analyses'] for p in pz]))),
                t_per_numeric_ms=1000 * float(np.sum([p['t_numeric'] for p in pz]) / max(1, np.sum([p['numeric'] for p in pz]))),
                t_envelope_mean=float(np.mean([p['t_envelope'] for p in pz])),
                envelope_nnz=int(max(p['envelope_nnz'] for p in pz)),
                nnz_factor=sorted({n for x in xs for n in x['nnz_factor']}),
                pcg_iters_mean=float(np.mean([x['pcg_iters'] for x in xs])),
                eft_calls_mean=float(np.mean([x['eft_calls'] for x in xs])),
                refreshes_mean=float(np.mean([x['refreshes'] for x in xs])),
                components_mean={c: float(np.mean([x['components'][c] for x in xs])) for c in xs[0]['components']},
                repeat_cv_median=float(np.median(cv)) if cv else None, repeat_cv_max=float(max(cv)) if cv else None,
                peak_wset_GiB_max=float(max(x['peak_wset_GiB'] for x in xs)))
        res['primary_exploratory'] = '/'.join(PRIMARY[fam])
        out['configs'][f'{fam}|{size}'] = res
    fp = [c['env'][k]['dense_mm_s'] for c in cells.values() for k in ('fingerprint_before', 'fingerprint_after')]
    out['fingerprint_dense_mm_s'] = [min(fp), max(fp)] if fp else None
    (HERE / 'analysis_struct.json').write_text(json.dumps(out, indent=1))
    for cfg, r in out['configs'].items():
        print(cfg, 'invalid', len(r['invalid']))
        for pk, v in r['pairs'].items():
            print('  ', pk, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, v.get('category'))
        for a, v in r['arms'].items():
            print('  ', a, round(v['cost_mean'], 3), 'analyses', round(v['analyses_mean'], 2), v['analyses_range'],
                  'ana_ms', round(v['t_per_analysis_ms'], 1), 'num_ms', round(v['t_per_numeric_ms'], 1),
                  'env_s', round(v['t_envelope_mean'], 3), 'valid', v['valid_runs'], '/', v['runs'])


if __name__ == '__main__':
    main()
