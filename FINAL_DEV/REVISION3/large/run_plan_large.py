"""Revision 3, exploratory: execution plan of the large-size experiment (LARGE_PROTOCOL.md).
Cells (fresh process each, run_cell_large.py): development seeds 0, 1, 2 in this order, one repeat. Guards: 90 min of
cumulative cell wall time, 45 min per process, 16 GiB peak memory; a process that hits a guard is killed and the plan
stops. After the cells, every completed cell is replayed with the frozen Phase 10B replay (untimed); any REJECTED or
UNAVAILABLE record is a hard stop. Ledger: budget_large.json; log: plan_large.log. Usage: python run_plan_large.py"""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
PY = sys.executable
RUNS = HERE / 'runs'
BUDGET_S, PROC_S, MEM_GIB = 90 * 60, 45 * 60, 16.0
LEDGER = HERE / 'budget_large.json'


def log(msg):
    with open(HERE / 'plan_large.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def run_proc(args, logfile, charge):
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(seconds=0.0, entries=[])
    t0 = time.time(); why = None; peak = 0.0
    with open(logfile, 'w') as f:
        p = subprocess.Popen([PY, '-B'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE),
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        pp = psutil.Process(p.pid)
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
        led['seconds'] += el; led['entries'].append([time.strftime('%H:%M:%S'), Path(logfile).stem, round(el, 1), round(peak, 2)])
        LEDGER.write_text(json.dumps(led))
    return p.returncode, why, el


def main():
    RUNS.mkdir(exist_ok=True)
    others = [q.info for q in psutil.process_iter(['pid', 'name']) if q.info['name'] and 'python' in q.info['name'].lower()
              and q.pid not in (psutil.Process().pid, psutil.Process().ppid())]
    log(f'START plan; other python processes: {len(others)}')
    done = []; stop = None
    for seed in (0, 1, 2):
        name = f'E3-TS_S5_s{seed}'
        rc, why, el = run_proc(['run_cell_large.py', '--seed', str(seed), '--out', str(RUNS / name)], RUNS / f'{name}.log', True)
        log(f'{name} rc={rc} why={why} {el:.1f}s')
        if rc != 0 or why:
            stop = f'{why or "EXIT"} at {name}'; break
        done.append(name)
    log(f'CELLS DONE {len(done)}; stop={stop}')
    reps = {}
    for name in done:
        rc, why, el = run_proc([str(CARS / 'FINAL_DEV' / 'phase10b' / 'replay10b.py'), str(RUNS / name)], RUNS / f'{name}_replay.log', False)
        r = json.loads((RUNS / name / 'replay10b.json').read_text()) if (RUNS / name / 'replay10b.json').exists() else {}
        reps[name] = r.get('totals')
        log(f'replay {name} rc={rc} {r.get("totals")} errors={len(r.get("errors", []))} {el:.1f}s')
        if rc != 0 or set(r.get('totals', {'x': 1})) - {'VERIFIED'} or r.get('errors'):
            log(f'HARD STOP: replay problem in {name}'); break
    (HERE / 'plan_large.json').write_text(json.dumps(dict(cells=done, stop=stop, replay=reps,
                                                          budget_s=json.loads(LEDGER.read_text())['seconds']), indent=1))
    log('PLAN DONE')


if __name__ == '__main__':
    main()
