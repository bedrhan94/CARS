"""Cross-platform software-stack check (Linux/WSL2 on the SAME hardware; descriptive, not a replication).
For every Linux ST S3 repeat-0 cell (frozen held-out inputs, seeds 200-204, both families, all seven arms):
  - replay status on Linux (replay_linux.json) and on Windows (replay_windows.json);
  - parity (all 21 certificates target-met) per arm;
  - bitwise comparison of each certificate (lambda_lo, R_hi, eps, status, matrix hash) with the Windows record of the
    same arm, seed and step from the registered campaign (runs/ST_S3, repeat 0);
  - primary-pair cost ratios on Linux (one repeat) next to the Windows repeat-0 ratios of the same cells.
Usage: python analyze_xplat.py -> analysis_xplat.json"""
import json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
WIN = CARS / 'FINAL_DEV' / 'phase10b' / 'runs' / 'ST_S3'
PRIMARY = {'E3-TS': ('B9', 'D'), 'D3-TS': ('B', 'D2')}
KEYS = ('status', 'matrix_sha256', 'lambda_lo', 'R_hi', 'eps', 'target_met')


def recs(cell, arm):
    f = next((cell / arm).glob('*.jsonl'))
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def main():
    out = dict(cells={}, totals=dict(records=0, linux_verified=0, windows_verified=0, identical_certificates=0,
                                     same_matrix=0, parity_valid_arms=0, arms=0))
    ratios = {f: dict(linux=[], windows=[]) for f in PRIMARY}
    for cell in sorted((HERE / 'linux' / 'cells').glob('*_ST')):
        if not cell.is_dir():
            continue
        fam, size, s, r, mode = cell.name.split('_'); seed = int(s[1:])
        cj = json.loads(next(cell.glob('cell_*.json')).read_text())
        lin = json.loads((cell / 'replay_linux.json').read_text()) if (cell / 'replay_linux.json').exists() else {}
        win_rp = json.loads((cell / 'replay_windows.json').read_text()) if (cell / 'replay_windows.json').exists() else {}
        wcell = WIN / f'{fam}_{size}_s{seed}_r0_ST'
        wres = {x['arm']: x for x in json.loads(next(wcell.glob('cell_*.json')).read_text())['results']}
        c = dict(linux_replay=lin.get('totals'), windows_replay=win_rp.get('totals'), arms={})
        for x in cj['results']:
            a = x['arm']; L = recs(cell, a); W = recs(wcell, a)
            ident = sum(1 for p, q in zip(L, W) if all(p.get(k) == q.get(k) for k in KEYS))
            same_m = sum(1 for p, q in zip(L, W) if p['matrix_sha256'] == q['matrix_sha256'])
            par = x['certified_target'] == 21
            c['arms'][a] = dict(parity=par, certified_target=x['certified_target'], identical_certificates=ident,
                                same_matrix=same_m, linux_measured_s=x['measured_s'], windows_measured_s=wres[a]['measured_s'])
            T = out['totals']; T['records'] += len(L); T['identical_certificates'] += ident; T['same_matrix'] += same_m
            T['arms'] += 1; T['parity_valid_arms'] += par
        T = out['totals']
        T['linux_verified'] += (lin.get('totals') or {}).get('VERIFIED', 0)
        T['windows_verified'] += (win_rp.get('totals') or {}).get('VERIFIED', 0)
        P, Q = PRIMARY[fam]
        ratios[fam]['linux'].append(c['arms'][P]['linux_measured_s'] / c['arms'][Q]['linux_measured_s'])
        ratios[fam]['windows'].append(c['arms'][P]['windows_measured_s'] / c['arms'][Q]['windows_measured_s'])
        out['cells'][cell.name] = c
    # EXPLORATORY: mean cost components of the primary arms on both platforms, and the memory-stream fingerprint
    comps = {}
    for fam, (P, Q) in PRIMARY.items():
        for a in (P, Q):
            for plat in ('linux', 'windows'):
                acc = {}
                for cell in sorted((HERE / 'linux' / 'cells').glob(f'{fam}_*_ST')):
                    if not cell.is_dir():
                        continue
                    src = cell if plat == 'linux' else WIN / cell.name
                    x = {r['arm']: r for r in json.loads(next(src.glob('cell_*.json')).read_text())['results']}[a]
                    for k, v in dict(x.get('components', {}), measured=x['measured_s']).items():
                        acc.setdefault(k, []).append(v)
                comps[f'{fam}|{a}|{plat}'] = {k: sum(v) / len(v) for k, v in acc.items()}
    out['components_mean_exploratory'] = comps
    st = {'linux': [], 'windows': []}
    for cell in sorted((HERE / 'linux' / 'cells').glob('*_ST')):
        if not cell.is_dir():
            continue
        for plat, src in (('linux', cell), ('windows', WIN / cell.name)):
            env = json.loads(next(src.glob('cell_*.json')).read_text())['env']
            for k in ('fingerprint_before', 'fingerprint_after', 'fp_before', 'fp_after'):
                if k in env and 'stream_s' in env[k]:
                    st[plat].append(env[k]['stream_s'])
    out['stream_fingerprint_s'] = {p: [min(v), max(v)] if v else None for p, v in st.items()}
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v)) if v else None
    out['primary_ratio_repeat0'] = {f: dict(pair='/'.join(PRIMARY[f]), linux_per_seed=v['linux'], windows_per_seed=v['windows'],
                                            linux_gm=gm(v['linux']), windows_gm=gm(v['windows'])) for f, v in ratios.items()}
    env = (HERE / 'linux' / 'environment.txt').read_text(errors='replace') if (HERE / 'linux' / 'environment.txt').exists() else ''
    out['linux_environment_excerpt'] = [l for l in env.splitlines() if l.startswith(('PRETTY', 'Python', 'Model name', 'Linux'))][:6]
    (HERE / 'analysis_xplat.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out['totals']), json.dumps({f: (round(v['linux_gm'], 3), round(v['windows_gm'], 3)) for f, v in out['primary_ratio_repeat0'].items()}))


if __name__ == '__main__':
    main()
