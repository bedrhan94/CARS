"""Phase 10A analysis of the paired development runs (descriptive; no decision rule is applied in Phase 10A).
Statistical unit: (family, dev seed). Per unit, each arm's per-sequence certified cost is the MEDIAN over its timing
repeats; ratios are per-system / M2-A (historical convention: > 1 means M2-A is faster) computed per (seed, repeat)
inside one process and summarised by the median over repeats; across seeds: geometric mean with a two-sided 95 %
t-interval on log ratios (df = seeds - 1) and [min, max]. Measured cost A = all components except artifact I/O;
counterfactual B = A - anchor_V - (estimate for per-system methods), as frozen.
Usage: python analyze10a.py -> analysis10a.json"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P10 = HERE.parent
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776}
ARM_NAME = {'A': 'RPS (frozen, reference)', 'B1': 'RPS-EFT (ablation)', 'B': 'RPS-OPT', 'B9': 'RPS-OPT-S09', 'C': 'M2-A (frozen, reference)',
            'D': 'M2A-EFT', 'D2': 'M2A-EFT-S05'}


def cells(stage):
    out = []
    for f in sorted((P10 / 'runs' / stage).rglob('cell_*.json')):
        if '_warmup' in f.parts:
            continue
        out.append(json.loads(f.read_text()))
    return out


def gm_ci(vals):
    l = np.log(np.asarray(vals, float)); m = float(l.mean()); k = len(l)
    if k < 2:
        return math.exp(m), None, None
    s = float(l.std(ddof=1)); h = T975.get(k - 1, 1.96) * s / math.sqrt(k)
    return math.exp(m), math.exp(m - h), math.exp(m + h)


def summarise(stage):
    rows = [r for c in cells(stage) for r in c['results']]
    by = defaultdict(list)
    for r in rows:
        by[(r['family'] + '@' + r['size'], r['seed'], r['arm'])].append(r)
    fams = sorted({k[0] for k in by}); seeds = sorted({k[1] for k in by}); arms = sorted({k[2] for k in by})
    out = {'per_unit': {}, 'arms': {}, 'ratios': {}, 'parity': {}}
    for (fam, seed, arm), rs in sorted(by.items()):
        A = [r['measured_s'] for r in rs]; B = [r['counterfactual_s'] for r in rs]
        out['per_unit'][f'{fam}|s{seed}|{arm}'] = dict(
            repeats=len(rs), measured_median=float(np.median(A)), measured_cv=(float(np.std(A, ddof=1) / np.mean(A)) if len(A) > 1 else None),
            counterfactual_median=float(np.median(B)), certified=rs[0]['certified'], target_met=rs[0]['certified_target'],
            records=rs[0]['records'], factorizations=rs[0]['factorizations'], pcg_iters=rs[0]['pcg_iters'],
            estimates=rs[0].get('estimates'), restorations=rs[0].get('restorations'),
            components={k: float(np.median([r['components'][k] for r in rs])) for k in rs[0]['components']},
            all_repeats_certified=[r['certified'] for r in rs], all_repeats_target=[r['certified_target'] for r in rs],
            peak_wset_GiB=max(r.get('peak_wset_GiB', 0) for r in rs))
    for fam in fams:
        for arm in arms:
            units = [out['per_unit'].get(f'{fam}|s{s}|{arm}') for s in seeds]
            units = [u for u in units if u]
            if not units:
                continue
            out['arms'][f'{fam}|{arm}'] = dict(name=ARM_NAME.get(arm, arm), measured_mean_over_seeds=float(np.mean([u['measured_median'] for u in units])),
                                               counterfactual_mean=float(np.mean([u['counterfactual_median'] for u in units])),
                                               certified=sum(u['certified'] for u in units), target_met=sum(u['target_met'] for u in units),
                                               records=sum(u['records'] for u in units), factorizations=sum(u['factorizations'] for u in units),
                                               pcg_iters=sum(u['pcg_iters'] for u in units),
                                               estimates=sum((u['estimates'] or 0) for u in units), restorations=sum((u['restorations'] or 0) for u in units),
                                               components_mean={k: float(np.mean([u['components'][k] for u in units])) for k in units[0]['components']},
                                               measured_cv_median=(float(np.median([u['measured_cv'] for u in units if u['measured_cv'] is not None]))
                                                                   if any(u['measured_cv'] is not None for u in units) else None))
        # paired ratios (per seed: median over repeats of within-process ratios)
        pairs = [('A', 'C'), ('B', 'D'), ('B', 'D2'), ('B9', 'D'), ('B9', 'D2'), ('B1', 'D'), ('B1', 'D2'), ('A', 'D'), ('A', 'D2'),
                 ('A', 'B'), ('A', 'B9'), ('A', 'B1'), ('B1', 'B'), ('B9', 'B'), ('C', 'D'), ('D', 'D2'), ('B', 'C')]
        for x, y in pairs:
            for kind in ('measured_s', 'counterfactual_s'):
                per_seed = []
                for s in seeds:
                    rx = {r['repeat']: r for r in by.get((fam, s, x), [])}; ry = {r['repeat']: r for r in by.get((fam, s, y), [])}
                    reps = sorted(set(rx) & set(ry))
                    if reps:
                        per_seed.append(float(np.median([rx[k][kind] / ry[k][kind] for k in reps])))
                if per_seed:
                    g, lo, hi = gm_ci(per_seed)
                    out['ratios'][f'{fam}|{x}/{y}|{kind}'] = dict(gm=g, ci95=[lo, hi], per_seed=per_seed, min=min(per_seed), max=max(per_seed))
    return out


def manifest_check(stage):
    man = json.loads((P10 / 'inputs_manifest10a.json').read_text())['manifest']
    bad = 0; tot = 0
    for f in (P10 / 'runs' / stage).rglob('*.jsonl'):
        if '_warmup' in f.parts:
            continue
        for l in f.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l); key = f"{r['family']}|{r['size']}|{r['seed']}"
            m = man.get(key)
            tot += 1
            if m is None or m[int(r['t'])]['matrix_sha256'] != r['matrix_sha256'] or m[int(r['t'])]['rhs_sha256'] != r['rhs_sha256']:
                bad += 1
    return dict(records=tot, mismatches=bad)


def main():
    res = {}
    for stage in ('V1', 'V1E', 'V1B9', 'P2', 'D3', 'P2_6arm', 'D3_6arm'):
        if (P10 / 'runs' / stage).exists():
            res[stage] = summarise(stage); res[stage]['input_manifest_check'] = manifest_check(stage)
            reps = [json.loads(p.read_text()) for p in (P10 / 'runs' / stage).glob('*/replay10a.json')]
            if reps:
                tot = defaultdict(int); det = [0, 0]; rump = defaultdict(int)
                for r in reps:
                    for k, x in r['totals'].items():
                        tot[k] += x
                    det[0] += sum(1 for v in r['determinism'].values() if v['identical']); det[1] += len(r['determinism'])
                    for k, x in r['rump_reproofs'].items():
                        rump[k] += x
                res[stage]['replay'] = dict(cells=len(reps), totals=dict(tot), rump=dict(rump), determinism_identical=det[0],
                                            determinism_total=det[1], errors=sum(len(r['errors']) for r in reps))
    (HERE / 'analysis10a.json').write_text(json.dumps(res, indent=1))
    for stage, s in res.items():
        print('==', stage, 'manifest', s['input_manifest_check'], 'replay', s.get('replay', {}).get('totals'))
        for k, a in s['arms'].items():
            print('  ', k, 'A=%.3f' % a['measured_mean_over_seeds'], 'B=%.3f' % a['counterfactual_mean'], 'cert', a['certified'], 'tgt', a['target_met'], '/', a['records'],
                  'fac', a['factorizations'], 'it', a['pcg_iters'], 'est', a['estimates'], 'restore', a['restorations'], 'cv', a['measured_cv_median'])
        for k, r in s['ratios'].items():
            if k.endswith('measured_s'):
                print('   ratio', k, 'GM %.3f' % r['gm'], 'CI', [None if x is None else round(x, 3) for x in r['ci95']], 'seeds', [round(x, 3) for x in r['per_seed']])


if __name__ == '__main__':
    main()
