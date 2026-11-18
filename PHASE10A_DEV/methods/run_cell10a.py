"""Phase 10A worker: ONE paired cell (family, size, dev seed, repeat) in a fresh process, single-threaded, pinned to
logical CPU 2 (a P-core; identical to Phases 5/6). Six arms run in this process in a rotated order:
    A   RPS|P1F      frozen Phase 6 runner (reference per-system)
    B1  RPS-EFT      per-system + EFT strategy (ablation)
    B   RPS-OPT      per-system + carried shift + sigma 0.5 + EFT strategy
    B9  RPS-OPT-S09  per-system + carried shift + sigma 0.9 + EFT strategy
    C   RPS|M2A      frozen Phase 6 runner (reference M2-A)
    D   M2A-EFT      M2-A + EFT strategy (frozen sigma policy)
    D2  M2A-EFT-S05  M2-A + EFT strategy + sigma 0.5 (symmetric sigma policy)
Guards: development seeds 0-2, split 'dev', families E3-TS / D3-TS (and SAFE-S / SAFE-P for negative controls),
sizes S0/S1/S2, and S3 only with --diag (single repeat). Held-out seeds are refused."""
import os
import sys
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import argparse, hashlib, json, time                                 # noqa: E402
from pathlib import Path                                             # noqa: E402
import psutil                                                        # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runner10a as N                                                # noqa: E402
R6 = N.R6
from cars_m2.families import Family3D                                # noqa: E402

ARMS = [('A', 'RPS|P1F'), ('B1', 'RPS-EFT'), ('B', 'RPS-OPT'), ('B9', 'RPS-OPT-S09'), ('C', 'RPS|M2A'), ('D', 'M2A-EFT'),
        ('D2', 'M2A-EFT-S05')]
TARGET = 1e-8                       # identical to the Phase 5/6 records (runner default); see the Phase 10A reports


def code_hash():
    h = hashlib.sha256()
    files = [HERE / 'runner10a.py', HERE / 'run_cell10a.py', HERE.parent / 'eft' / 'eft_residual.py',
             R6.CARS / 'PHASE6_DEV' / 'lean6' / 'runner6.py', R6.CARS / 'PHASE5_DEV' / 'lean' / 'fastbind.py']
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def run_arm(label, cfg, fam, out, repeat, order, ch):
    if cfg.startswith('RPS|'):
        meth = cfg.split('|')[1]
        return R6.run_sequence_5(fam, meth, 'RPS', out / label, track='T1', binding='lean', shared=None,
                                 repeat=repeat, run_order=order, code_hash=ch, write_artifacts=True, target=TARGET)
    return N.run_sequence_10a(fam, cfg, out / label, target=TARGET, repeat=repeat, run_order=order, code_hash=ch,
                              write_artifacts=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family'); ap.add_argument('--size'); ap.add_argument('--seed', type=int)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--out'); ap.add_argument('--diag', action='store_true')
    ap.add_argument('--arms', default='A,B1,B,B9,C,D,D2')
    a = ap.parse_args()
    if a.seed not in (0, 1, 2):
        raise SystemExit('DEV GUARD: development seeds 0, 1, 2 only')
    if a.family not in ('E3-TS', 'D3-TS', 'SAFE-S', 'SAFE-P'):
        raise SystemExit('family guard')
    if a.size not in ('S0', 'S1', 'S2') and not (a.size == 'S3' and a.diag):
        raise SystemExit('SIZE GUARD: S0/S1/S2, or S3 only as a --diag diagnostic')
    p = psutil.Process()
    try:
        p.cpu_affinity([2]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print('ENV_WARNING', repr(e), flush=True)
    R6._MKL.get()
    ch = code_hash()
    sel = [x for x in ARMS if x[0] in a.arms.split(',')]
    k = (a.repeat + a.seed) % len(sel); sel = sel[k:] + sel[:k]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # symmetric untimed warm-up: every arm once on E3-TS S0 dev seed 0 (library initialisation, first-call effects);
    # results are written to a scratch subdirectory and never analysed
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg in sel:
        run_arm(label, cfg, wfam, out / '_warmup', 0, 0, ch)
    fam = Family3D(a.family, a.size, seed=a.seed, split='dev')
    results = []
    for order, (label, cfg) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows = run_arm(label, cfg, fam, out, a.repeat, order, ch)
        wall = time.perf_counter() - t0
        mi = p.memory_info()
        summ = dict(arm=label, config=cfg, family=a.family, size=a.size, seed=a.seed, repeat=a.repeat, order=order,
                    n=fam.n, file=str(path), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    unverified=sum(r['status'] == 'UNVERIFIED' for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    counterfactual_s=sum(r['timing_s']['counterfactual'] for r in rows),
                    io_s=sum(r['timing_s']['io'] for r in rows),
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    refreshes=sum(bool(r['details'].get('refresh')) for r in rows),
                    estimates=sum(int(r['details'].get('estimates', 0)) for r in rows),
                    restorations=sum(int(r['details'].get('restorations', 0)) for r in rows),
                    reasons=sorted({str(r['details'].get('decision_reason')) for r in rows}),
                    failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}),
                    residual_kinds=sorted({str(r.get('residual_kind')) for r in rows if r['status'] == 'CERTIFIED'}),
                    eps_max=max((float.fromhex(r['eps']) for r in rows if r.get('eps')), default=None),
                    rss_GiB=mi.rss / 2 ** 30, peak_wset_GiB=getattr(mi, 'peak_wset', 0) / 2 ** 30, wall_s=wall)
        results.append(summ)
        print('SUMMARY ' + json.dumps(summ), flush=True)
    env = dict(affinity=p.cpu_affinity(), code_hash=ch, target=TARGET.hex(), peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30)
    (out / f'cell_{a.family}_{a.size}_s{a.seed}_r{a.repeat}.json').write_text(json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
