"""Revision 2, exploratory: execution plan of the compiled-kernel experiment (COMPILED_PROTOCOL.md).
Cells (fresh process each, run_cell_compiled.py); the primary size S3 first, ordered by repeat so that a budget stop
leaves balanced data, then the secondary size S2:
    for size in (S3, S2): for repeat in 0,1,2: for family in (E3-TS, D3-TS): for seed in 200..204.
Guards: 90 min of cumulative cell wall time (benchmark budget), 30 min per process, 16 GiB peak memory per process;
a process that hits a guard is killed and the plan stops. After the cells, every completed cell is replayed with the
frozen Phase 10B replay (replay10b.py; untimed, outside the benchmark budget); any REJECTED/UNAVAILABLE record or input
mismatch is reported as a hard stop. Ledger: budget_compiled.json; log: plan_compiled.log.
Usage: python run_plan_compiled.py"""
import json, subprocess, sys, time
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
PY = sys.executable
RUNS = HERE / 'runs'
BUDGET_S, PROC_S, MEM_GIB = 90 * 60, 30 * 60, 16.0
LEDGER = HERE / 'budget_compiled.json'
SIZES = ('S3', 'S2')
FAMILIES = ('E3-TS', 'D3-TS')
SEEDS = (200, 201, 202, 203, 204)


def log(msg):
    with open(HERE / 'plan_compiled.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def run_proc(args, logfile, charge):
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(seconds=0.0, entries=[])
    t0 = time.time(); why = None
    with open(logfile, 'w') as f:
        p = subprocess.Popen([PY, '-B'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE),
                             env=dict(__import__('os').environ, PYTHONDONTWRITEBYTECODE='1'))
        pp = psutil.Process(p.pid); peak = 0.0
        while p.poll() is None:
            time.sleep(0.5)
            try:
                mi = pp.memory_info(); peak = max(peak, getattr(mi, 'peak_wset', mi.rss) / 2 ** 30)
            except psutil.Error:
                pass
            el = time.time() - t0
            if peak > MEM_GIB: why = 'MEMORY_CEILING'; p.kill(); break
            if el > PROC_S: why = 'PROCESS_TIMEOUT'; p.kill(); break
            if charge and led['seconds'] + el > BUDGET_S: why = 'BUDGET'; p.kill(); break
        p.wait()
    el = time.time() - t0
    if charge:
        led['seconds'] += el; led['entries'].append([time.strftime('%H:%M:%S'), Path(logfile).stem, round(el, 1)])
        LEDGER.write_text(json.dumps(led))
    return p.returncode, why, el


def main():
    RUNS.mkdir(exist_ok=True)
    others = [q.info for q in psutil.process_iter(['pid', 'name', 'cmdline'])
              if q.info['name'] and 'python' in q.info['name'].lower() and q.pid != psutil.Process().pid
              and q.pid != psutil.Process().ppid()]
    log(f'START plan; other python processes: {len(others)}')
    done = []; stop = None
    for size in SIZES:
        for rep in (0, 1, 2):
            for fam in FAMILIES:
                for seed in SEEDS:
                    name = f'{fam}_{size}_s{seed}_r{rep}'
                    rc, why, el = run_proc(['run_cell_compiled.py', '--family', fam, '--size', size, '--seed', str(seed),
                                            '--repeat', str(rep), '--out', str(RUNS / name)], RUNS / f'{name}.log', True)
                    log(f'{name} rc={rc} why={why} {el:.1f}s')
                    if rc != 0 or why:
                        stop = f'{why or "EXIT"} at {name}'; break
                    done.append(name)
                if stop: break
            if stop: break
        if stop: break
    log(f'CELLS DONE {len(done)}; stop={stop}')
    rep_tot = {}
    for name in done:
        rc, why, el = run_proc([str(CARS / 'FINAL_DEV' / 'phase10b' / 'replay10b.py'), str(RUNS / name)],
                               RUNS / f'{name}_replay.log', False)
        r = json.loads((RUNS / name / 'replay10b.json').read_text()) if (RUNS / name / 'replay10b.json').exists() else {}
        tot = r.get('totals', {}); rep_tot[name] = tot
        bad = {k: v for k, v in tot.items() if k != 'VERIFIED'}
        log(f'replay {name} rc={rc} {tot} errors={len(r.get("errors", []))} mismatches={len(r.get("manifest_mismatches", []))}')
        if rc != 0 or bad or r.get('errors') or r.get('manifest_mismatches'):
            log(f'HARD STOP: replay problem in {name}'); break
    (HERE / 'plan_compiled.json').write_text(json.dumps(dict(cells=done, stop=stop, replay=rep_tot,
                                                           budget_s=json.loads(LEDGER.read_text())['seconds']), indent=1))
    log('PLAN DONE')


if __name__ == '__main__':
    main()
