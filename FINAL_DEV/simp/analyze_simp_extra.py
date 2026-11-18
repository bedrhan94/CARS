"""EXPLORATORY additions to the SIMP analysis, written after the registered descriptive analysis (analyze_simp.py) had
been run. Not part of SIMP_PROTOCOL.md. Reads the same records; writes analysis_simp_extra.json.

1. Per instance and window, for the carried-shift arms B and B9: number of uncertified steps, carry failures, failure
   reasons, steps that fell back to a cold eigenvalue estimate, and the time spent in the estimate.
2. Per window, pair ratios computed on the measured cost MINUS the eigenvalue-estimate time of step t = 0 (the cold-start
   estimate that every arm performs on the same first matrix). Unit = instance, median over repeats, GM, 95 % t-interval.
3. Repeats available per (instance, window).
Usage: python analyze_simp_extra.py"""
import json, glob, math
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}


def gm_ci(v):
    l = np.log(np.asarray(v)); m = l.mean(); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None)
    se = l.std(ddof=1) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)])


def main():
    rows = defaultdict(dict)      # (window, inst, repeat) -> arm -> info
    for cj in sorted((HERE / 'eval' / 'runs').glob('*/cell_*.json')):
        cell = json.loads(cj.read_text())
        for r in cell['results']:
            recs = [json.loads(l) for l in open(cj.parent / r['file'])]
            est0 = recs[0]['timing_s'].get('estimate', 0.0)
            rows[(r['window'], r['inst'], r['repeat'])][r['arm']] = dict(
                measured=r['measured_s'], est0=est0, target_met_all=(r['certified_target'] == 21),
                uncertified=sum(1 for x in recs if x['status'] != 'CERTIFIED'),
                carry_failures=sum(1 for x in recs if 'carry_failure' in x.get('details', {})),
                cold_fallbacks=sum(1 for x in recs if x.get('details', {}).get('estimate_fallback') == 'eigsh'),
                estimate_s=sum(x['timing_s'].get('estimate', 0.0) for x in recs),
                reasons=sorted({str(x.get('reason')) for x in recs if x['status'] != 'CERTIFIED'}))
    out = dict(repeats={}, carried_shift={}, pairs_excluding_cold_start={}, parity_by_instance={})
    for (w, i, rep) in rows:
        out['repeats'].setdefault(f'{w}|{i}', []).append(rep)
    for (w, i, rep), arms in rows.items():      # all runs of the arm on the unit met the target at all 21 steps
        for a, x in arms.items():
            k = f'{w}|{i}|{a}'
            out['parity_by_instance'][k] = out['parity_by_instance'].get(k, True) and x['target_met_all']
    for (w, i, rep), arms in sorted(rows.items()):
        for a in ('B', 'B9'):
            if a in arms:
                out['carried_shift'][f'{w}|{i}|r{rep}|{a}'] = {k: v for k, v in arms[a].items() if k != 'est0'}
    for w in ('early', 'late'):
        res = {}
        for P in ('A', 'B1', 'B', 'B9'):
            for Q in ('C', 'D', 'D2'):
                per = defaultdict(list)
                for (ww, i, rep), c in rows.items():
                    if ww == w and P in c and Q in c:
                        per[i].append((c[P]['measured'] - c[P]['est0']) / (c[Q]['measured'] - c[Q]['est0']))
                units = {i: float(np.median(v)) for i, v in per.items()}
                res[f'{P}/{Q}'] = dict(per_instance=units, **gm_ci(list(units.values())))
        res['cold_start_estimate_s'] = {i: float(np.mean([c[a]['est0'] for (ww, ii, rep), c in rows.items()
                                                          if ww == w and ii == i for a in c]))
                                        for i in sorted({k[1] for k in rows if k[0] == w})}
        res['cold_start_fraction'] = {i: float(np.mean([c[a]['est0'] / c[a]['measured'] for (ww, ii, rep), c in rows.items()
                                                        if ww == w and ii == i for a in c]))
                                      for i in sorted({k[1] for k in rows if k[0] == w})}
        out['pairs_excluding_cold_start'][w] = res
    (HERE / 'analysis_simp_extra.json').write_text(json.dumps(out, indent=1))
    for k, v in out['carried_shift'].items():
        print(k, v)
    for w, r in out['pairs_excluding_cold_start'].items():
        print(w, {p: (round(x['gm'], 3), [round(c, 3) for c in x['ci95']]) for p, x in r.items()
                  if p in ('A/D', 'B9/D', 'B/D2', 'A/D2')}, {i: round(s, 2) for i, s in r['cold_start_estimate_s'].items()})
    print(out['repeats'])


if __name__ == '__main__':
    main()
