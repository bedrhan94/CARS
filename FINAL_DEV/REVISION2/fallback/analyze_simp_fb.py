"""Revision 2, exploratory: analysis of the fixed-fallback SIMP3D evaluation, as fixed in SIMP_FB_PROTOCOL.md.
Per window (early, late): unit = instance (median over the valid repeats); GM over the 5 instances with a two-sided
95 % t-interval (df = 4). Validity (parity) of an arm in a cell: all 21 steps certified and target-met AND all its
records VERIFIED by the frozen SIMP replay; a ratio uses only cells in which both arms are valid (invalid arms are
counted and reported, never treated as slow observations). Pairs: every per-system arm (A, B1, Bf, B9f) against every
reuse arm (C, D, D2); the analogues of the Phase 10B primary pairs are B9f/D and Bf/D2. Descriptive only: the
categories use the thresholds of the registered rule but have no decision role. Fallback events: carried-shift
failures, estimate fallbacks by kind, and their cost (estimate component).
Usage: python analyze_simp_fb.py -> analysis_simp_fb.json"""
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
PER = ('A', 'B1', 'Bf', 'B9f'); REUSE = ('C', 'D', 'D2')


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
    for cj in sorted((HERE / 'runs').glob('*/cell_SIMP3D_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay_simp.json').read_text())
        arms = {}
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            arms[r['arm']] = dict(r, valid=(r['certified_target'] == 21 and ver == 21), verified=ver)
        r0 = d['results'][0]
        cells[(r0['window'], r0['inst'], r0['repeat'])] = arms
    out = dict(cells=len(cells), windows={})
    for w in ('early', 'late'):
        C = {k: v for k, v in cells.items() if k[0] == w}
        res = dict(arms={}, pairs={}, instances=sorted({k[1] for k in C}), runs=len(C), invalid=[])
        for k, arms in C.items():
            for a, x in arms.items():
                if not x['valid']:
                    res['invalid'].append(dict(cell=list(k), arm=a, certified_target=x['certified_target'], verified=x['verified']))
        for a in PER + REUSE:
            xs = [arms[a] for arms in C.values() if a in arms]
            if not xs:
                continue
            per_inst = defaultdict(list)
            for k, arms in C.items():
                if a in arms and arms[a]['valid']:
                    per_inst[k[1]].append(arms[a]['measured_s'])
            res['arms'][a] = dict(
                runs=len(xs), valid_runs=sum(x['valid'] for x in xs),
                certified=sum(x['certified'] for x in xs), certified_target=sum(x['certified_target'] for x in xs),
                records=sum(x['records'] for x in xs), verified=sum(x['verified'] for x in xs),
                cost_mean=float(np.mean([np.median(v) for v in per_inst.values()])) if per_inst else None,
                factorizations_mean=float(np.mean([x['factorizations'] for x in xs])),
                refreshes_mean=float(np.mean([x['refreshes'] for x in xs])),
                pcg_iters_mean=float(np.mean([x['pcg_iters'] for x in xs])),
                components_mean={c: float(np.mean([x['components'][c] for x in xs])) for c in xs[0]['components']},
                carry_failures=sum(x.get('carry_failures', 0) for x in xs),
                estimate_fallbacks=dict(Counter(f for x in xs for f in x.get('estimate_fallbacks', []))),
                failure_reasons=sorted({fr for x in xs for fr in x['failure_reasons']}),
                max_measured_s=float(max(x['measured_s'] for x in xs)))
        for P in PER:
            for Q in REUSE:
                per = defaultdict(list); excluded = 0
                for k, arms in C.items():
                    if P in arms and Q in arms:
                        if arms[P]['valid'] and arms[Q]['valid']:
                            per[k[1]].append(arms[P]['measured_s'] / arms[Q]['measured_s'])
                        else:
                            excluded += 1
                units = {i: float(np.median(v)) for i, v in sorted(per.items())}
                if units:
                    st = gm_ci(list(units.values()))
                    if st['ci95']:
                        st['category'] = category(st)
                    res['pairs'][f'{P}/{Q}'] = dict(per_instance=units, runs=sum(len(v) for v in per.values()),
                                                    excluded_runs=excluded, **st)
        out['windows'][w] = res
    (HERE / 'analysis_simp_fb.json').write_text(json.dumps(out, indent=1))
    for w, r in out['windows'].items():
        print(w, 'runs', r['runs'], 'invalid', len(r['invalid']), r['invalid'][:4])
        for p in ('B9f/D', 'Bf/D2', 'B1/D', 'B1/D2', 'A/C'):
            if p in r['pairs']:
                v = r['pairs'][p]
                print('  ', p, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, v.get('category'),
                      'runs', v['runs'], 'excluded', v['excluded_runs'])
        for a, v in r['arms'].items():
            print('  ', a, round(v['cost_mean'] or 0, 3), 'valid', v['valid_runs'], '/', v['runs'], 'carry_fail', v['carry_failures'],
                  v['estimate_fallbacks'], 'est_s', round(v['components_mean'].get('estimate', 0), 3), v['failure_reasons'])


if __name__ == '__main__':
    main()
