"""Revision 2, exploratory (post hoc) worker: ONE SIMP3D cell (instance, window, repeat) with the carried-shift arms
using the fixed estimate fallback (fallback_runner.py). Everything else is the frozen SIMP cell
(FINAL_DEV/simp/run_cell_simp.py): frozen SIMP inputs hash-checked before the warm-up, frozen settings and per-window
target, single thread pinned to logical CPU 2, HIGH priority, the same symmetric untimed warm-up, rotated order
k = (repeat + instance) mod 7. Arms:
    A  RPS|P1F   B1 RPS-EFT   Bf RPS-OPT (fixed fallback)   B9f RPS-OPT-S09 (fixed fallback)   (per-system)
    C  RPS|M2A   D  M2A-EFT   D2 M2A-EFT-S05                                                     (reuse)
The frozen B and B9 arms are not re-run (their SIMP failures are documented from the original 12 runs).
Usage: python run_cell_simp_fb.py --frozen DIR --size S3 --inst KEY --window early|late --repeat r --out DIR"""
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
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
SIMP = CARS / 'FINAL_DEV' / 'simp'
sys.path.insert(0, str(SIMP)); sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import simp_family as SF                                             # noqa: E402  frozen SIMP family wrapper
import run_cell10b as C10                                            # noqa: E402  frozen helpers (code_hash, fingerprint, run_arm)
import common10b as K                                                # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import fallback_runner as FB                                         # noqa: E402

RUN_FB, _, _ = FB.build(C10.N)
ARMS = [('A', 'RPS|P1F'), ('B1', 'RPS-EFT'), ('Bf', 'RPS-OPT'), ('B9f', 'RPS-OPT-S09'), ('C', 'RPS|M2A'),
        ('D', 'M2A-EFT'), ('D2', 'M2A-EFT-S05')]
FIXED = ('Bf', 'B9f')


def fb_hash():
    h = hashlib.sha256()
    for f in ('fallback_runner.py', 'run_cell_simp_fb.py'):
        h.update(f.encode()); h.update((HERE / f).read_bytes())
    return h.hexdigest()


def run_arm(label, cfg, fam, out, repeat, order, ch):
    if label in FIXED:
        return RUN_FB(fam, cfg, out / label, target=K.TARGET, repeat=repeat, run_order=order, code_hash=ch,
                      write_artifacts=True)
    return C10.run_arm(label, cfg, fam, out, repeat, order, ch)


def main():
    ap = argparse.ArgumentParser()
    for a_ in ('--frozen', '--size', '--inst', '--window', '--out'):
        ap.add_argument(a_)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--target', type=float, default=None)
    a = ap.parse_args()
    rec = (SIMP / 'simp_settings.sha256').read_text().split()[0]
    if hashlib.sha256((SIMP / 'simp_settings.json').read_bytes()).hexdigest() != rec:
        raise SystemExit('SIMP settings changed after freezing')
    settings = json.loads((SIMP / 'simp_settings.json').read_text())
    target = a.target if a.target is not None else float.fromhex(settings['targets'][f'{a.inst}|{a.window}'])
    p = psutil.Process()
    try:
        p.cpu_affinity([2]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print('ENV_WARNING', repr(e), flush=True)
    C10.R6._MKL.get()
    K.TARGET = target                                                # run_arm reads K.TARGET
    ch = C10.code_hash()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = SF.SimpFamily(a.frozen, a.size, a.inst, a.window)
    rows_h = json.loads((Path(a.frozen) / 'simp_hashes.json').read_text())[f'{a.inst}|{a.window}']
    bad = []
    for row in rows_h:
        A, b, meta = fam.get(row['t']); A = SF.R6.ensure_canonical(A)
        if SF.v.matrix_sha256(A) != row['matrix_sha256'] or SF.v.vec_sha256(b) != row['rhs_sha256'] or \
                meta['coeff_sha256'] != row['coeff_sha256']:
            bad.append(row['t'])
    if bad:
        print('INPUT_MISMATCH', bad, flush=True); sys.exit(3)
    env = dict(mode='ST', affinity=p.cpu_affinity(), code_hash=ch, fb_hash=fb_hash(), target=float(target).hex(),
               inst=a.inst, window=a.window, fingerprint_before=C10.fingerprint(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(ARMS); k = (a.repeat + fam.seed) % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg in sel:
        run_arm(label, cfg, wfam, out / '_warmup', 0, 0, ch)
    results = []
    for order, (label, cfg) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows = run_arm(label, cfg, fam, out, a.repeat, order, ch)
        wall = time.perf_counter() - t0
        summ = dict(arm=label, config=cfg, fixed_fallback=label in FIXED, inst=a.inst, window=a.window, size=a.size,
                    repeat=a.repeat, order=order, n=fam.n, file=str(Path(path).relative_to(out)), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    io_s=sum(r['timing_s']['io'] for r in rows),
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in C10.R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    eft_calls=sum(int(r['solver'].get('eft_calls', 0)) for r in rows),
                    refreshes=sum(bool(r['details'].get('refresh')) for r in rows),
                    carry_failures=sum(r['details'].get('carry_failure') is not None for r in rows),
                    estimate_fallbacks=sorted(str(r['details'].get('estimate_fallback')) for r in rows
                                              if r['details'].get('estimate_fallback')),
                    reasons=sorted({str(r['details'].get('decision_reason')) for r in rows}),
                    failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}),
                    eps_max=max((float.fromhex(r['eps']) for r in rows if r.get('eps')), default=None),
                    lambda_lo_min=min((float.fromhex(r['lambda_lo']) for r in rows if r.get('lambda_lo')), default=None),
                    peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30, wall_s=wall)
        results.append(summ)
        print('SUMMARY ' + json.dumps({k2: summ[k2] for k2 in ('arm', 'measured_s', 'certified_target', 'carry_failures',
                                                                'estimate_fallbacks')}), flush=True)
    env.update(fingerprint_after=C10.fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / f'cell_SIMP3D_{a.inst}_{a.window}_r{a.repeat}.json').write_text(json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
