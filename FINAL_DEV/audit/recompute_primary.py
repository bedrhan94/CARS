"""Stage 7 audit: independent recomputation of the Phase 10B primary statistic from the raw cell files, written
separately from analyze10b.py (different code path: raw per-record JSONL sums instead of cell summaries, scipy.stats
t quantiles instead of a table, parity from the per-record replay files). Compares with analysis10b.json.
Usage: python recompute_primary.py"""
import json, math, statistics
from pathlib import Path
from scipy import stats

CARS = Path(__file__).resolve().parents[2]
P10B = CARS / 'FINAL_DEV' / 'phase10b'
PRIMARY = {'E3-TS': ('B9', 'D'), 'D3-TS': ('B', 'D2')}
SEEDS = (200, 201, 202, 203, 204)


def seq_cost(cell_dir, arm):
    f = next((cell_dir / arm).glob('*.jsonl'))
    recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    cost = sum(sum(v for k, v in r['timing_s'].items() if k not in ('io', 'measured', 'counterfactual')) for r in recs)
    ok = len(recs) == 21 and all(r['status'] == 'CERTIFIED' and r['target_met'] for r in recs)
    return cost, ok, sum(r['timing_s']['measured'] for r in recs)


def main():
    an = json.loads((P10B / 'analysis10b.json').read_text())
    out = {}
    for mode, size, key in (('ST', 'S3', 'primary'), ('ST', 'S2', 'secondary_st_s2'), ('MT', 'S3', 'secondary_mt_s3'), ('MT', 'S2', 'secondary_mt_s2')):
        for fam, (P, Q) in PRIMARY.items():
            units = {}; parity = True; maxdiff = 0.0
            for s in SEEDS:
                rs = []
                for rep in (0, 1, 2):
                    cd = P10B / 'runs' / f'{mode}_{size}' / f'{fam}_{size}_s{s}_r{rep}_{mode}'
                    if (cd.parent / (cd.name + '_retry')).exists():
                        cd = cd.parent / (cd.name + '_retry')
                    rp = json.loads((cd / 'replay10b.json').read_text())
                    cp, okp, mp = seq_cost(cd, P); cq, okq, mq = seq_cost(cd, Q)
                    maxdiff = max(maxdiff, abs(cp - mp) / mp, abs(cq - mq) / mq)
                    parity &= okp and okq and rp['arms'][P].get('VERIFIED') == 21 and rp['arms'][Q].get('VERIFIED') == 21
                    rs.append(cp / cq)
                units[s] = statistics.median(rs)
            logs = [math.log(v) for v in units.values()]
            m = statistics.fmean(logs); se = statistics.stdev(logs) / math.sqrt(len(logs))
            q975, q95 = stats.t.ppf(0.975, len(logs) - 1), stats.t.ppf(0.95, len(logs) - 1)
            gm = math.exp(m); ci95 = [math.exp(m - q975 * se), math.exp(m + q975 * se)]; ci90 = [math.exp(m - q95 * se), math.exp(m + q95 * se)]
            sec = an[key][0] if isinstance(an[key], list) else an[key]
            ref = sec[fam]['stats']
            out[f'{mode}|{size}|{fam}'] = dict(gm=gm, ci95=ci95, ci90=ci90, parity=parity, components_vs_measured_max_rel=maxdiff,
                                               ref_gm=ref['gm'], ref_ci95=ref['ci95'], ref_ci90=ref['ci90'],
                                               max_abs_diff=max(abs(gm - ref['gm']), *(abs(a - b) for a, b in zip(ci95, ref['ci95'])),
                                                                *(abs(a - b) for a, b in zip(ci90, ref['ci90']))),
                                               ref_category=sec[fam]['category'])
            print(mode, size, fam, f'gm {gm:.6f} vs {ref["gm"]:.6f}', 'CI95', [round(x, 6) for x in ci95], 'parity', parity,
                  'maxdiff', f"{out[f'{mode}|{size}|{fam}']['max_abs_diff']:.2e}")
    (Path(__file__).parent / 'recompute_primary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
