"""Revision 3, exploratory: analysis of the large-size experiment, as fixed in LARGE_PROTOCOL.md (descriptive only).
Validity: all 21 steps certified and target-met AND all records VERIFIED by the frozen replay. Ratios per seed (one
repeat); GM over the three development seeds with a two-sided 95 % t-interval (df = 2), reported with the per-seed values;
no category is applied. Pairs: B9/D (frozen operations), B9f/Df (compiled), B9fs/Dfs (compiled + envelope), B9fs/Df,
and the speed-ups. Per-step component costs (factorization, binding, residual, PCG, shift, estimate), PARDISO
analyses and factor size, compared with the same arms at sizes S2 and S3 from the full-compilation experiment
(analysis_fullc.json), to show how the components scale with n. Identity: B9/B9f and D/Df store identical R_hi.
Usage: python analyze_large.py -> analysis_large.json"""
import json, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
T975 = {1: 12.706, 2: 4.303}
PAIRS = [('B9', 'D'), ('B9f', 'Df'), ('B9fs', 'Dfs'), ('B9fs', 'Df'), ('B9', 'B9f'), ('D', 'Df'), ('B9f', 'B9fs'), ('Df', 'Dfs')]


def gm_ci(v):
    l = np.log(np.asarray(v, float)); m = l.mean(); n = len(l)
    if n < 2:
        return dict(gm=math.exp(m), n=n, ci95=None)
    se = l.std(ddof=1) / math.sqrt(n)
    return dict(gm=math.exp(m), n=n, ci95=[math.exp(m - T975[n - 1] * se), math.exp(m + T975[n - 1] * se)])


def per_step(x):
    c = x['components']; f = max(1, x['factorizations'])
    return dict(factorization_per_factorization_s=c.get('anchor_F', 0) / f, binding_per_step_s=c.get('binding', 0) / 21,
                residual_per_step_s=c.get('residual', 0) / 21, pcg_per_step_s=c.get('solve', 0) / 21,
                shift_per_factorization_s=c.get('shift', 0) / f, estimate_s=c.get('estimate', 0),
                pcg_iters_per_step=x['pcg_iters'] / 21)


def main():
    cells = {}
    for cj in sorted((HERE / 'runs').glob('*/cell_large_*.json')):
        d = json.loads(cj.read_text()); rp = json.loads((cj.parent / 'replay10b.json').read_text())
        arms = {}
        for r in d['results']:
            ver = rp['arms'].get(r['arm'], {}).get('VERIFIED', 0)
            arms[r['arm']] = dict(r, valid=(r['certified_target'] == 21 and ver == 21), verified=ver)
        cells[d['results'][0]['seed']] = dict(arms=arms, env=d['env'], replay_totals=rp['totals'])
    out = dict(cells=len(cells), seeds=sorted(cells), pairs={}, arms={}, invalid=[], identity={})
    for s, c in cells.items():
        for a, x in c['arms'].items():
            if not x['valid']:
                out['invalid'].append(dict(seed=s, arm=a, certified_target=x['certified_target'], verified=x['verified']))
        out['identity'][str(s)] = dict(B9_B9f=c['arms']['B9']['R_hi'] == c['arms']['B9f']['R_hi'],
                                       D_Df=c['arms']['D']['R_hi'] == c['arms']['Df']['R_hi'])
    for A, B in PAIRS:
        per = {str(s): c['arms'][A]['measured_s'] / c['arms'][B]['measured_s'] for s, c in cells.items()
               if c['arms'][A]['valid'] and c['arms'][B]['valid']}
        if per:
            out['pairs'][f'{A}/{B}'] = dict(per_seed=per, **gm_ci(list(per.values())))
    for a in ('B9', 'D', 'B9f', 'Df', 'B9fs', 'Dfs'):
        xs = [c['arms'][a] for c in cells.values() if a in c['arms']]
        if not xs:
            continue
        ps = [per_step(x) for x in xs]
        out['arms'][a] = dict(level=xs[0]['level'], session=xs[0]['session'], runs=len(xs), valid_runs=sum(x['valid'] for x in xs),
                              cost_mean=float(np.mean([x['measured_s'] for x in xs])),
                              components_mean={k: float(np.mean([x['components'][k] for x in xs])) for k in xs[0]['components']},
                              per_step={k: float(np.mean([p[k] for p in ps])) for k in ps[0]},
                              factorizations_mean=float(np.mean([x['factorizations'] for x in xs])),
                              refreshes_mean=float(np.mean([x['refreshes'] for x in xs])),
                              eft_calls_mean=float(np.mean([x['eft_calls'] for x in xs])),
                              analyses_mean=float(np.mean([x['pardiso']['analyses'] for x in xs])),
                              nnz_factor=sorted({n for x in xs for n in x['nnz_factor']}),
                              peak_wset_GiB_max=float(max(x['peak_wset_GiB'] for x in xs)))
    fc = HERE.parent / 'fullc' / 'analysis_fullc.json'
    if fc.exists():
        f = json.loads(fc.read_text()); sc = {}
        for size in ('S2', 'S3'):
            r = f['configs'].get(f'E3-TS|{size}')
            if r:
                sc[size] = {a: dict(cost_mean=r['arms'][a]['cost_mean'],
                                    per_step=dict(factorization_per_factorization_s=r['arms'][a]['components_mean'].get('anchor_F', 0) / max(1, r['arms'][a]['factorizations_mean']),
                                                  binding_per_step_s=r['arms'][a]['components_mean'].get('binding', 0) / 21,
                                                  residual_per_step_s=r['arms'][a]['components_mean'].get('residual', 0) / 21,
                                                  pcg_per_step_s=r['arms'][a]['components_mean'].get('solve', 0) / 21))
                            for a in ('B9f', 'Df', 'B9fs', 'Dfs')}
        out['scaling_reference'] = sc
    (HERE / 'analysis_large.json').write_text(json.dumps(out, indent=1))
    print('cells', out['cells'], 'invalid', out['invalid'], 'identity', out['identity'])
    for pk, v in out['pairs'].items():
        print('  ', pk, round(v['gm'], 3), [round(x, 3) for x in v['ci95']] if v['ci95'] else None, {k: round(x, 3) for k, x in v['per_seed'].items()})
    for a, v in out['arms'].items():
        print('  ', a, round(v['cost_mean'], 2), {k: round(x, 4) for k, x in v['per_step'].items()}, 'ana', v['analyses_mean'], 'fac', v['factorizations_mean'])


if __name__ == '__main__':
    main()
