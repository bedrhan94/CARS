"""Revision 3, exploratory (post hoc) worker: ONE cell of the large-size experiment (LARGE_PROTOCOL.md) in a fresh
process. E3-TS geometry with N = 32 (n = 104 544), development split, one development seed; inputs verified against
the frozen manifest (inputs_manifest_large.json) before the warm-up (exit 3 on mismatch).
Arms (frozen methods; implementation level and PARDISO session differ):
    B9, D       RPS-OPT-S09 / M2A-EFT, frozen operations, frozen (instrumented) session
    B9f, Df     all compiled operations (binding, accurate residual enclosure, shift construction), frozen session
    B9fs, Dfs   all compiled operations, structural-envelope session
Order [B9, D, B9f, Df, B9fs, Dfs] rotated by k = seed mod 6. Single thread pinned to logical CPU 2, HIGH priority;
symmetric untimed warm-up on E3-TS S0, development seed 0.
Harness smoke test only: --size S1 (development seed 0, no manifest).
Usage: python run_cell_large.py --seed s --out DIR [--size S5|S1]"""
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
import common10b as K                                                # noqa: E402
import run_cell10b as RC                                             # noqa: E402
N = RC.N; R6 = RC.R6; v = RC.v
from cars_m2.families import Family3D                                # noqa: E402
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'REVISION1' / 'structural'))
import struct_session as SS                                          # noqa: E402
sys.path.insert(0, str(CARS / 'FINAL_DEV' / 'REVISION3' / 'fullc'))
import compiled_full as CF                                           # noqa: E402

SS.install(R6)
CF.install(N, R6, N.EFT, v)
ARMS = [('B9', 'RPS-OPT-S09', 'frozen', 'orig'), ('D', 'M2A-EFT', 'frozen', 'orig'),
        ('B9f', 'RPS-OPT-S09', 'full', 'orig'), ('Df', 'M2A-EFT', 'full', 'orig'),
        ('B9fs', 'RPS-OPT-S09', 'full', 'struct'), ('Dfs', 'M2A-EFT', 'full', 'struct')]


def large_hash():
    h = hashlib.sha256()
    for f in (HERE / 'run_cell_large.py', CARS / 'FINAL_DEV/REVISION3/fullc/compiled_full.py',
              CARS / 'FINAL_DEV/REVISION2/compiled/compiled_kernels.py', CARS / 'FINAL_DEV/REVISION1/structural/struct_session.py'):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def run_arm(label, cfg, level, session, fam, out, order, ch):
    CF.use(level, N, R6); SS.use(session, fam.mesh)
    try:
        return RC.run_arm(label, cfg, fam, out, 0, order, ch)
    finally:
        CF.use('frozen', N, R6); SS.use('orig', fam.mesh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int); ap.add_argument('--out'); ap.add_argument('--size', default='S5')
    a = ap.parse_args()
    if a.size == 'S5':
        if a.seed not in (0, 1, 2):
            raise SystemExit('SEED GUARD: development seeds 0-2 only')
        man = json.loads((HERE / 'inputs_manifest_large.json').read_text())['manifest'][f'E3-TS|S5|{a.seed}']['rows']
    elif a.size == 'S1' and a.seed == 0:
        man = None
    else:
        raise SystemExit('SIZE GUARD: S5 (experiment) or S1 seed 0 (harness smoke) only')
    p = psutil.Process()
    try:
        p.cpu_affinity([2]); p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print('ENV_WARNING', repr(e), flush=True)
    ch = RC.code_hash()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = Family3D('E3-TS', a.size, seed=a.seed, split='dev')
    t_in = time.perf_counter(); bad = []
    if man is not None:
        for row in man:
            A, b, meta = fam.get(row['t']); A = R6.ensure_canonical(A)
            if (v.matrix_sha256(A), v.vec_sha256(b), meta['coeff_sha256']) != (row['matrix_sha256'], row['rhs_sha256'], row['coeff_sha256']):
                bad.append(row['t'])
    input_check = dict(checked=len(man or []), mismatches=bad, seconds=time.perf_counter() - t_in)
    if bad:
        (out / 'INPUT_MISMATCH.json').write_text(json.dumps(input_check, indent=1)); sys.exit(3)
    env = dict(mode='ST', threads=1, affinity=p.cpu_affinity(), priority=int(p.nice()), python=platform.python_version(),
               numpy=np.__version__, numba=CF.nb.__version__, code_hash=ch, large_hash=large_hash(), target=float(K.TARGET).hex(),
               size=a.size, n=fam.n, input_check=input_check, fingerprint_before=RC.fingerprint(),
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(ARMS); k = a.seed % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg, level, session in sel:
        run_arm(label, cfg, level, session, wfam, out / '_warmup', 0, ch)
    SS.take_stats()
    results = []
    for order, (label, cfg, level, session) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows = run_arm(label, cfg, level, session, fam, out, order, ch)
        wall = time.perf_counter() - t0
        st = SS.take_stats(); mi = p.memory_info()
        summ = dict(arm=label, config=cfg, level=level, session=session, size=a.size, seed=a.seed, order=order, n=fam.n,
                    file=str(Path(path).relative_to(out)), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    first_estimate_s=rows[0]['timing_s'].get('estimate', 0.0) if rows else None,
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    eft_calls=sum(int(r['solver'].get('eft_calls', 0)) for r in rows),
                    refreshes=sum(bool(r['details'].get('refresh')) for r in rows),
                    failure_reasons=sorted({str(r.get('reason')) for r in rows if r['status'] != 'CERTIFIED'}),
                    R_hi=[r.get('R_hi') for r in rows], lambda_lo=[r.get('lambda_lo') for r in rows],
                    nnz_factor=sorted({int(l.get('nnz_factor') or 0) for r in rows for l in (r.get('lambda_chain') or [])
                                       if l.get('kind') == 'rump'}),
                    pardiso=st, rss_GiB=mi.rss / 2 ** 30, peak_wset_GiB=getattr(mi, 'peak_wset', 0) / 2 ** 30, wall_s=wall)
        results.append(summ)
        print('SUMMARY ' + json.dumps({k2: summ[k2] for k2 in ('arm', 'measured_s', 'certified_target')}), flush=True)
    env.update(fingerprint_after=RC.fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
               peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30)
    (out / f'cell_large_E3-TS_{a.size}_s{a.seed}.json').write_text(json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
