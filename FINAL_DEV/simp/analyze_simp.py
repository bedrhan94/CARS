"""Descriptive analysis of the SIMP3D supplementary evaluation (SIMP_PROTOCOL.md section 4): per window, the ratios of
the Phase 10B primary pairs (B9/D, B/D2) and all per-system/reuse pairs; unit = instance (median over repeats); GM with a
95 % t-interval (df = 4); all arms' costs, parity, factorizations, refreshes, PCG iterations, component means.
No decision category is applied. Usage: python analyze_simp.py -> analysis_simp.json"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def gm_ci(v):
    l = np.log(v); m = l.mean(); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None)
    se = l.std(ddof=1) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)])


def main():
    cells = defaultdict(dict)
    for cj in sorted((HERE / 'eval' / 'runs').glob('*/cell_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay_simp.json').read_text())
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            cells[(r['window'], r['inst'], r['repeat'])][r['arm']] = dict(r, parity=(r['certified_target'] == 21 and ver == 21), verified=ver)
    out = {'windows': {}}
    for w in ('early', 'late'):
        runs = {k: v for k, v in cells.items() if k[0] == w}
        insts = sorted({k[1] for k in runs})
        arms = sorted({a for v in runs.values() for a in v})
        res = dict(arms={}, pairs={}, instances=insts, runs=len(runs))
        for a in arms:
            per = defaultdict(list)
            for (ww, i, rep), c in runs.items():
                if a in c:
                    per[i].append(c[a])
            res['arms'][a] = dict(cost_mean=float(np.mean([np.median([x['measured_s'] for x in v]) for v in per.values()])),
                                  parity=f"{sum(x['parity'] for v in per.values() for x in v)}/{sum(len(v) for v in per.values())}",
                                  certified_target=sum(x['certified_target'] for v in per.values() for x in v),
                                  certified=sum(x['certified'] for v in per.values() for x in v),
                                  records=21 * sum(len(v) for v in per.values()),
                                  verified=sum(x['verified'] for v in per.values() for x in v),
                                  factorizations_mean=float(np.mean([x['factorizations'] for v in per.values() for x in v])),
                                  refreshes_mean=float(np.mean([x['refreshes'] for v in per.values() for x in v])),
                                  pcg_iters_mean=float(np.mean([x['pcg_iters'] for v in per.values() for x in v])),
                                  eft_calls_mean=float(np.mean([x['eft_calls'] for v in per.values() for x in v])),
                                  components_mean={k: float(np.mean([x['components'][k] for v in per.values() for x in v]))
                                                   for k in next(iter(per.values()))[0]['components']},
                                  failure_reasons=sorted({fr for v in per.values() for x in v for fr in x['failure_reasons']}))
        for P in ('A', 'B1', 'B', 'B9'):
            for Q in ('C', 'D', 'D2'):
                per = defaultdict(list)
                for (ww, i, rep), c in runs.items():
                    if P in c and Q in c:
                        per[i].append(c[P]['measured_s'] / c[Q]['measured_s'])
                units = {i: float(np.median(v)) for i, v in per.items()}
                if units:
                    res['pairs'][f'{P}/{Q}'] = dict(per_instance=units, **gm_ci(np.array(list(units.values()))))
        out['windows'][w] = res
    (HERE / 'analysis_simp.json').write_text(json.dumps(out, indent=1))
    for w, r in out['windows'].items():
        print(w, {p: round(v['gm'], 3) for p, v in r['pairs'].items() if p in ('B9/D', 'B/D2', 'A/C')},
              {a: (round(v['cost_mean'], 3), v['parity']) for a, v in r['arms'].items()})


if __name__ == '__main__':
    main()
