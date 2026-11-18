"""Phase 10B worker: ONE paired cell (family, size, seed, repeat, thread mode) in a fresh process.

All seven frozen Phase 10A arms run in this process in a rotated order (k = (repeat + seed) mod 7), after a symmetric
untimed warm-up (every arm once on E3-TS S0 development seed 0):
    A  RPS|P1F (frozen Phase 6 runner)   B1 RPS-EFT   B RPS-OPT   B9 RPS-OPT-S09   (per-system)
    C  RPS|M2A (frozen Phase 6 runner)   D  M2A-EFT   D2 M2A-EFT-S05               (certificate reuse)
(The docstring of the frozen PHASE10A_DEV/methods/run_cell10a.py says "Six arms"; its ARMS list, which is what runs,
has these seven; see PHASE10B_PREDATA_AMENDMENTS.md.)
Thread modes (set before any numerical library is imported):
    ST = 1 thread (MKL/PARDISO, OpenMP, OpenBLAS), pinned to logical CPU 2 (a P-core), HIGH priority  (primary)
    MT = 8 threads, pinned to logical CPUs 0,2,...,14 (one per P-core), HIGH priority                 (secondary)
Held-out cells (seeds 200-204, split 'heldout') require the pre-registration freeze record, unchanged frozen code and
the frozen input manifest; every input of the cell is verified against the manifest BEFORE the warm-up (hard stop on
mismatch, exit code 3). Development seeds 0-2 are allowed only at S0/S1 (harness smoke).
Usage: python run_cell10b.py --family F --size S --seed k --repeat r --mode ST|MT --out DIR"""
import os
import sys


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


MODE = _arg('--mode', 'ST')
THREADS, CPUS = {'ST': (1, [2]), 'MT': (8, [0, 2, 4, 6, 8, 10, 12, 14])}[MODE]
for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[k] = str(THREADS)
os.environ['MKL_DYNAMIC'] = 'FALSE'; os.environ['CARS_P6_THREADS'] = str(THREADS)
sys.dont_write_bytecode = True
import argparse, hashlib, json, platform, time                      # noqa: E402
from pathlib import Path                                             # noqa: E402
import numpy as np                                                   # noqa: E402
import psutil                                                        # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402
sys.path.insert(0, str(K.CARS / 'PHASE10A_DEV' / 'methods'))
import runner10a as N                                                # noqa: E402  frozen Phase 10A runner
R6 = N.R6
from cars import _reference_m0 as v                                  # noqa: E402
from cars_m2.families import Family3D                                # noqa: E402


def code_hash():
    """Identical to PHASE10A_DEV/methods/run_cell10a.code_hash (records stay comparable with Phase 10A)."""
    h = hashlib.sha256()
    M = K.CARS / 'PHASE10A_DEV' / 'methods'
    files = [M / 'runner10a.py', M / 'run_cell10a.py', K.CARS / 'PHASE10A_DEV' / 'eft' / 'eft_residual.py',
             K.CARS / 'PHASE6_DEV' / 'lean6' / 'runner6.py', K.CARS / 'PHASE5_DEV' / 'lean' / 'fastbind.py']
    for f in files:
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()


def fingerprint():
    """Same micro-benchmark as Phase 6 (dense matrix products, streaming sqrt) plus CPU frequency and load."""
    rng = np.random.default_rng(1); M = rng.random((300, 300)); s = time.perf_counter()
    for _ in range(40):
        M = M @ M; M /= np.abs(M).max()
    t_mm = time.perf_counter() - s
    x = rng.random(2_000_000); s = time.perf_counter()
    for _ in range(20):
        x = np.sqrt(x + 1.0)
    f = psutil.cpu_freq()
    return {'dense_mm_s': t_mm, 'stream_s': time.perf_counter() - s, 'cpu_freq_mhz': (f.current if f else None),
            'cpu_percent_system': psutil.cpu_percent(interval=0.2), 'available_GiB': psutil.virtual_memory().available / 2 ** 30}


def run_arm(label, cfg, fam, out, repeat, order, ch):
    if cfg.startswith('RPS|'):
        return R6.run_sequence_5(fam, cfg.split('|')[1], 'RPS', out / label, track='T1', binding='lean', shared=None,
                                 repeat=repeat, run_order=order, code_hash=ch, write_artifacts=True, target=K.TARGET)
    return N.run_sequence_10a(fam, cfg, out / label, target=K.TARGET, repeat=repeat, run_order=order, code_hash=ch,
                              write_artifacts=True)


def verify_inputs(fam, man_rows):
    """Untimed: every matrix, rhs and coefficient vector of the cell must equal the frozen manifest bit for bit."""
    bad = []
    for row in man_rows:
        A, b, meta = fam.get(row['t']); A = R6.ensure_canonical(A)
        got = dict(matrix_sha256=v.matrix_sha256(A), rhs_sha256=v.vec_sha256(b), coeff_sha256=meta.get('coeff_sha256'))
        for k2, val in got.items():
            if val != row[k2]:
                bad.append(dict(t=row['t'], field=k2))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family'); ap.add_argument('--size'); ap.add_argument('--seed', type=int)
    ap.add_argument('--repeat', type=int, default=0); ap.add_argument('--out'); ap.add_argument('--mode', default='ST')
    a = ap.parse_args()
    if a.family not in K.FAMILIES:
        raise SystemExit('family guard')
    split = 'heldout' if a.seed in K.HELDOUT_SEEDS else 'dev'
    K.seed_guard(a.seed, split)
    agg = None; man_rows = None
    if split == 'heldout':
        if a.size not in K.SIZES:
            raise SystemExit('SIZE GUARD: held-out cells are S2 or S3 only')
        agg = K.check_freeze()
        man_rows = K.check_inputs_frozen()['manifest'][f'{a.family}|{a.size}|{a.seed}']['rows']
    elif a.size not in ('S0', 'S1'):
        raise SystemExit('SIZE GUARD: development cells are harness smoke tests at S0/S1 only')
    p = psutil.Process()
    try:
        p.cpu_affinity(CPUS); p.nice(psutil.HIGH_PRIORITY_CLASS)
    except Exception as e:
        print('ENV_WARNING', repr(e), flush=True)
    lib = R6._MKL.get()
    try:
        mkl_threads = int(lib.MKL_Get_Max_Threads())
    except AttributeError:
        mkl_threads = None
    ch = code_hash()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    fam = Family3D(a.family, a.size, seed=a.seed, split=split)
    t_in = time.perf_counter()
    bad = verify_inputs(fam, man_rows) if man_rows is not None else []
    input_check = dict(checked=(len(man_rows) if man_rows else 0), mismatches=bad, seconds=time.perf_counter() - t_in)
    if bad:
        (out / 'INPUT_MISMATCH.json').write_text(json.dumps(input_check, indent=1))
        print('INPUT_MISMATCH', bad[:5], flush=True)
        sys.exit(3)
    env = dict(mode=MODE, threads=THREADS, mkl_max_threads=mkl_threads, affinity=p.cpu_affinity(), priority=int(p.nice()),
               threads_env={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                           'MKL_DYNAMIC', 'CARS_P6_THREADS')},
               python=platform.python_version(), numpy=np.__version__, code_hash=ch, code_aggregate_10b=agg,
               target=float(K.TARGET).hex(), split=split, input_check=input_check, fingerprint_before=fingerprint(),
               started=time.strftime('%Y-%m-%d %H:%M:%S'))
    sel = list(K.ARMS)
    k = (a.repeat + a.seed) % len(sel); sel = sel[k:] + sel[:k]
    wfam = Family3D('E3-TS', 'S0', seed=0, split='dev')
    for label, cfg in sel:                       # symmetric untimed warm-up (never analysed)
        run_arm(label, cfg, wfam, out / '_warmup', 0, 0, ch)
    results = []
    for order, (label, cfg) in enumerate(sel):
        t0 = time.perf_counter()
        path, rows = run_arm(label, cfg, fam, out, a.repeat, order, ch)
        wall = time.perf_counter() - t0
        mi = p.memory_info()
        summ = dict(arm=label, config=cfg, family=a.family, size=a.size, seed=a.seed, split=split, repeat=a.repeat,
                    mode=MODE, order=order, n=fam.n, file=str(Path(path).relative_to(out)), records=len(rows),
                    certified=sum(r['status'] == 'CERTIFIED' for r in rows),
                    certified_target=sum(r['status'] == 'CERTIFIED' and bool(r['target_met']) for r in rows),
                    unverified=sum(r['status'] == 'UNVERIFIED' for r in rows),
                    factorizations=sum(r['factor_count'] for r in rows),
                    measured_s=sum(r['timing_s']['measured'] for r in rows),
                    counterfactual_s=sum(r['timing_s']['counterfactual'] for r in rows),
                    io_s=sum(r['timing_s']['io'] for r in rows),
                    components={c: sum(r['timing_s'].get(c, 0.0) for r in rows) for c in R6.COMPONENTS},
                    pcg_iters=sum(r['solver'].get('iters', 0) for r in rows),
                    eft_calls=sum(int(r['solver'].get('eft_calls', 0)) for r in rows),
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
    env.update(fingerprint_after=fingerprint(), finished=time.strftime('%Y-%m-%d %H:%M:%S'),
               peak_wset_GiB=getattr(p.memory_info(), 'peak_wset', 0) / 2 ** 30)
    (out / f'cell_{a.family}_{a.size}_s{a.seed}_r{a.repeat}_{MODE}.json').write_text(
        json.dumps({'env': env, 'results': results}, indent=1))
    print('CELL_DONE', flush=True)


if __name__ == '__main__':
    main()
