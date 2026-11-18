"""Paired cell on FROZEN held-out inputs (exact-hex coefficients and rhs via FrozenFamily) for other platforms: the same
seven frozen arms, rotation, warm-up and records as FINAL_DEV/phase10b/run_cell10b.py, but the coefficients are read
instead of recomputed, so the cell does not depend on the platform's exp/sin. Every regenerated matrix is compared
with the frozen SHA-256; mismatches are recorded (a platform difference of the float assembly), not hidden.
Usage: python run_cell_frozen.py --family F --size S --seed k --repeat r --mode ST|MT --out DIR"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path                                             # noqa: E402
HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'phase10b')); sys.path.insert(0, str(HERE))
import run_cell10b as C10                                            # noqa: E402  (sets threads from --mode before numpy)
import argparse, json, os, platform, time                            # noqa: E402
import psutil                                                        # noqa: E402
from frozen_family import FrozenFamily                               # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family'); ap.add_argument('--size'); ap.add_argument('--seed', type=int)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--out'); ap.add_argument('--mode', default='ST')
    a = ap.parse_args()
    agg = C10.K.check_freeze()
    p = psutil.Process()
    try:
        p.cpu_affinity(C10.CPUS)
    except Exception as e:
        print('ENV_WARNING affinity', repr(e), flush=True)
    try:
        p.nice(psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -5)
    except Exception as e:
        print('ENV_WARNING priority', repr(e), flush=True)
    lib = C10.R6._MKL.get()
    ch = C10.code_hash(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = FrozenFamily(CARS / 'FINAL_DEV/phase10b/inputs/pass_a', a.family, a.size, a.seed)
    for t in range(fam.T + 1):
        fam.get(t)
    pre = list(fam.mismatches)
    env = dict(mode=C10.MODE, threads=C10.THREADS, platform=platform.platform(), python=platform.python_version(),
               mkl_max_threads=int(lib.MKL_Get_Max_Threads()) if hasattr(lib, 'MKL_Get_Max_Threads') else None,
               affinity=p.cpu_affinity() if hasattr(p, 'cpu_affinity') else None, code_hash=ch, code_aggregate_10b=agg,
               input_mismatches=pre, fingerprint_before=C10.fingerprint(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(C10.K.ARMS); k = (a.repeat + a.seed) % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg in sel:
        C10.run_arm(label, cfg, wfam, out / '_warmup', 0, 0, ch)
    results = []
    for order, (label, cfg) in enumerate(sel):
        t0 = time.perf_counter(); path, rows = C10.run_arm(label, cfg, fam, out, a.repeat, order, ch)
        results.append(dict(arm=label, config=cfg, family=a.family, size=a.size, seed=a.seed, repeat=a.repeat, mode=C10.MODE,
                            records=len(rows), certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                            certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                            measured_s=sum(r['timing_s']['measured'] for r in rows),
                            components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in C10.R6.COMPONENTS},
                            pcg_iters=sum(r['solver'].get('iters', 0) for r in rows), factorizations=sum(r['factor_count'] for r in rows),
                            wall_s=time.perf_counter() - t0))
        print('SUMMARY ' + json.dumps(results[-1]), flush=True)
    env.update(fingerprint_after=C10.fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    (out / f'cell_frozen_{a.family}_{a.size}_s{a.seed}_r{a.repeat}_{C10.MODE}.json').write_text(json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE input_mismatches', len(pre), flush=True)


if __name__ == '__main__':
    main()
