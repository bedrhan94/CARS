"""Revision 2, exploratory (post hoc) worker: ONE paired cell of the compiled-kernel experiment in a fresh process.

Arms (13): the seven frozen Phase 10A arms, and the six arms that use the binding and/or the accurate residual bound
with the compiled kernels of compiled_kernels.py (same method, same decisions, bit-identical bounds and binding results):
    A   RPS|P1F      (frozen only: uses neither the binding nor the accurate residual bound)
    B1  / B1c  RPS-EFT        B  / Bc  RPS-OPT        B9 / B9c  RPS-OPT-S09        (per-system; residual bound compiled)
    C   / Cc   RPS|M2A        (reuse, frozen Phase 6 runner; binding compiled)
    D   / Dc   M2A-EFT        D2 / D2c M2A-EFT-S05                              (reuse; binding and residual compiled)
Order: interleaved list A, B1, B1c, B, Bc, B9, B9c, C, Cc, D, Dc, D2, D2c rotated by k = (repeat + seed) mod 13.
Single thread pinned to logical CPU 2 (performance core), HIGH priority, threads fixed before any numerical import;
symmetric untimed warm-up (every arm once on E3-TS S0, development seed 0; this also loads/compiles the kernels).
Inputs: frozen Phase 10B held-out sequences (seeds 200-204, ALREADY CONSUMED), verified against the frozen manifest
before the warm-up (exit 3 on mismatch). Development seeds 0-2 only at S0/S1 (harness smoke). Sizes S2/S3 only.
Usage: python run_cell_compiled.py --family F --size S --seed k --repeat r --out DIR"""
import os
import sys

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = '1'
sys.dont_write_bytecode = True
import argparse, hashlib, json, platform, time                      # noqa: E402
from pathlib import Path                                             # noqa: E402
import numpy as np                                                   # noqa: E402
import psutil                                                        # noqa: E402
HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b'))
import common10b as K                                                # noqa: E402  frozen guards and constants
import run_cell10b as RC                                             # noqa: E402  frozen helpers (code_hash, fingerprint, verify_inputs, run_arm)
N = RC.N; R6 = RC.R6; v = RC.v
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import compiled_kernels as CK                                        # noqa: E402

CK.install(N, R6, N.EFT, v)
ARMS = [('A', 'RPS|P1F', 'frozen'), ('B1', 'RPS-EFT', 'frozen'), ('B1c', 'RPS-EFT', 'compiled'),
        ('B', 'RPS-OPT', 'frozen'), ('Bc', 'RPS-OPT', 'compiled'), ('B9', 'RPS-OPT-S09', 'frozen'),
        ('B9c', 'RPS-OPT-S09', 'compiled'), ('C', 'RPS|M2A', 'frozen'), ('Cc', 'RPS|M2A', 'compiled'),
        ('D', 'M2A-EFT', 'frozen'), ('Dc', 'M2A-EFT', 'compiled'), ('D2', 'M2A-EFT-S05', 'frozen'),
        ('D2c', 'M2A-EFT-S05', 'compiled')]


def compiled_hash():
    h = hashlib.sha256()
    for f in ('compiled_kernels.py', 'run_cell_compiled.py'):
        h.update(f.encode()); h.update((HERE / f).read_bytes())
    return h.hexdigest()


def run_arm(label, cfg, kind, fam, out, repeat, order, ch):
    CK.use(kind, N, R6)
    try:
        return RC.run_arm(label, cfg, fam, out, repeat, order, ch)
    finally:
        CK.use('frozen', N, R6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family'); ap.add_argument('--size'); ap.add_argument('--seed', type=int)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--out')
    a = ap.parse_args()
    if a.family not in K.FAMILIES:
        raise SystemExit('family guard')
    split = 'heldout' if a.seed in K.HELDOUT_SEEDS else 'dev'
    K.seed_guard(a.seed, split)
    man_rows = None; agg = None
    if split == 'heldout':
        if a.size not in ('S2', 'S3'):
            raise SystemExit('SIZE GUARD: S2 or S3 only (no S4)')
        agg = K.check_freeze()
        man_rows = K.check_inputs_frozen()['manifest'][f'{a.family}|{a.size}|{a.seed}']['rows']
    elif a.size not in ('S0', 'S1'):
        raise SystemExit('SIZE GUARD: development cells are harness smoke tests at S0/S1 only')
    p = psutil.Process()
    try:
        p.cpu_affinity([2]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print('ENV_WARNING', repr(e), flush=True)
    ch = RC.code_hash()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = Family3D(a.family, a.size, seed=a.seed, split=split)
    t_in = time.perf_counter()
    bad = RC.verify_inputs(fam, man_rows) if man_rows is not None else []
    input_check = dict(checked=(len(man_rows) if man_rows else 0), mismatches=bad, seconds=time.perf_counter() - t_in)
    if bad:
        (out / 'INPUT_MISMATCH.json').write_text(json.dumps(input_check, indent=1)); sys.exit(3)
    env = dict(mode='ST', threads=1, affinity=p.cpu_affinity(), priority=int(p.nice()), python=platform.python_version(),
               numpy=np.__version__, numba=CK.nb.__version__, code_hash=ch, code_aggregate_10b=agg,
               compiled_hash=compiled_hash(), target=float(K.TARGET).hex(), split=split, input_check=input_check,
               fingerprint_before=RC.fingerprint(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(ARMS)
    k = (a.repeat + a.seed) % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    t_w = time.perf_counter()
    for label, cfg, kind in sel:                   # symmetric untimed warm-up (never analysed)
        run_arm(label, cfg, kind, wfam, out / '_warmup', 0, 0, ch)
    env['warmup_s'] = time.perf_counter() - t_w
    results = []
    for order, (label, cfg, kind) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows = run_arm(label, cfg, kind, fam, out, a.repeat, order, ch)
        wall = time.perf_counter() - t0
        mi = p.memory_info()
        summ = dict(arm=label, config=cfg, impl=kind, family=a.family, size=a.size, seed=a.seed, split=split,
                    repeat=a.repeat, order=order, n=fam.n, file=str(Path(path).relative_to(out)), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    unverified=sum(r['status'] == 'UNVERIFIED' for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    io_s=sum(r['timing_s']['io'] for r in rows),
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    eft_calls=sum(int(r['solver'].get('eft_calls', 0)) for r in rows),
                    refreshes=sum(bool(r['details'].get('refresh')) for r in rows),
                    estimates=sum(int(r['details'].get('estimates', 0)) for r in rows),
                    bindings=sorted({str(r['details'].get('binding')) for r in rows}),
                    reasons=sorted({str(r['details'].get('decision_reason')) for r in rows}),
                    failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}),
                    residual_kinds=sorted({str(r.get('residual_kind')) for r in rows if r['status'] == 'CERTIFIED'}),
                    R_hi=[r.get('R_hi') for r in rows], eps_max=max((float.fromhex(r['eps']) for r in rows if r.get('eps')), default=None),
                    rss_GiB=mi.rss / 2 ** 30, peak_wset_GiB=getattr(mi, 'peak_wset', 0) / 2 ** 30, wall_s=wall)
        results.append(summ)
        print('SUMMARY ' + json.dumps({k2: summ[k2] for k2 in ('arm', 'measured_s', 'certified_target')}), flush=True)
    env.update(fingerprint_after=RC.fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
               peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30)
    (out / f'cell_compiled_{a.family}_{a.size}_s{a.seed}_r{a.repeat}_ST.json').write_text(
        json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
