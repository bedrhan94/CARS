"""Revision 1, exploratory (post hoc) worker: ONE paired cell of the structural-envelope experiment in a fresh process.

Arms (frozen Phase 10A methods; only the PARDISO session class differs, see struct_session.py):
    E3-TS: B9 (RPS-OPT-S09, frozen session)  D (M2A-EFT, frozen session)  B9s, Ds (same methods, structural envelope)
    D3-TS: B  (RPS-OPT, frozen session)      D2 (M2A-EFT-S05, frozen)     Bs, D2s (same methods, structural envelope)
The frozen sessions are instrumented (phase timing, counts) without any change in behaviour.
Single thread, pinned to logical CPU 2 (performance core), HIGH priority, threads fixed before any numerical import;
symmetric untimed warm-up (every arm once on E3-TS S0 development seed 0); rotated order k = (repeat + seed) mod 4.
Inputs: the frozen Phase 10B held-out sequences (seeds 200-204, ALREADY CONSUMED by the confirmatory analysis); every
input is verified against the frozen manifest before the warm-up (exit 3 on mismatch). Development seeds 0-2 only at
S0/S1 (harness smoke). Sizes S2/S3 only.
Usage: python run_cell_struct.py --family F --size S --seed k --repeat r --out DIR"""
import os
import sys

for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
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
import run_cell10b as RC                                             # noqa: E402  frozen helpers (code_hash, fingerprint, verify_inputs)
N = RC.N; R6 = RC.R6
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(HERE))
import struct_session as SS                                          # noqa: E402

SS.install(R6)
ARMS = {'E3-TS': [('B9', 'RPS-OPT-S09', 'orig'), ('D', 'M2A-EFT', 'orig'),
                  ('B9s', 'RPS-OPT-S09', 'struct'), ('Ds', 'M2A-EFT', 'struct')],
        'D3-TS': [('B', 'RPS-OPT', 'orig'), ('D2', 'M2A-EFT-S05', 'orig'),
                  ('Bs', 'RPS-OPT', 'struct'), ('D2s', 'M2A-EFT-S05', 'struct')]}


def struct_hash():
    h = hashlib.sha256()
    for f in ('struct_session.py', 'run_cell_struct.py'):
        h.update(f.encode()); h.update((HERE / f).read_bytes())
    return h.hexdigest()


def run_arm(label, cfg, kind, fam, out, repeat, order, ch):
    SS.use(kind, fam.mesh)
    path, rows = N.run_sequence_10a(fam, cfg, out / label, target=K.TARGET, repeat=repeat, run_order=order,
                                    code_hash=ch, write_artifacts=True)
    return path, rows, SS.take_stats()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family'); ap.add_argument('--size'); ap.add_argument('--seed', type=int)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--out')
    a = ap.parse_args()
    if a.family not in ARMS:
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
               numpy=np.__version__, code_hash=ch, code_aggregate_10b=agg, struct_hash=struct_hash(),
               target=float(K.TARGET).hex(), split=split, input_check=input_check, fingerprint_before=RC.fingerprint(),
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(ARMS[a.family])
    k = (a.repeat + a.seed) % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg, kind in sel:                   # symmetric untimed warm-up (never analysed)
        run_arm(label, cfg, kind, wfam, out / '_warmup', 0, 0, ch)
    SS.take_stats()
    results = []
    for order, (label, cfg, kind) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows, st = run_arm(label, cfg, kind, fam, out, a.repeat, order, ch)
        wall = time.perf_counter() - t0
        mi = p.memory_info()
        summ = dict(arm=label, config=cfg, session=kind, family=a.family, size=a.size, seed=a.seed, split=split,
                    repeat=a.repeat, order=order, n=fam.n, file=str(Path(path).relative_to(out)), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    io_s=sum(r['timing_s']['io'] for r in rows),
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    eft_calls=sum(int(r['solver'].get('eft_calls', 0)) for r in rows),
                    refreshes=sum(bool(r['details'].get('refresh')) for r in rows),
                    failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}),
                    nnz_factor=sorted({int(l.get('nnz_factor') or 0) for r in rows for l in (r.get('lambda_chain') or [])
                                       if l.get('kind') == 'rump'}),
                    pardiso=st, rss_GiB=mi.rss / 2 ** 30, peak_wset_GiB=getattr(mi, 'peak_wset', 0) / 2 ** 30, wall_s=wall)
        results.append(summ)
        print('SUMMARY ' + json.dumps({k2: summ[k2] for k2 in ('arm', 'measured_s', 'certified_target', 'pardiso')}), flush=True)
    env.update(fingerprint_after=RC.fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
               peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30)
    (out / f'cell_struct_{a.family}_{a.size}_s{a.seed}_r{a.repeat}_ST.json').write_text(
        json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
