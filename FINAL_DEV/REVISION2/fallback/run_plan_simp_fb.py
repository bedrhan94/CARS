"""Revision 2, exploratory: execution plan of the fixed-fallback SIMP3D evaluation (SIMP_FB_PROTOCOL.md).
30 cells = 5 instances x 2 windows x 3 repeats, one ST process per cell (run_cell_simp_fb.py), ordered by repeat so
that a budget stop leaves balanced data; every cell is replayed immediately with the frozen SIMP replay
(FINAL_DEV/simp/replay_simp.py; untimed, outside the budget). Guards: 90 min cumulative cell wall time, 30 min per
process, 16 GiB peak memory, frozen SIMP settings unchanged; hard stop on any non-VERIFIED record.
Ledger: budget_simp_fb.json; log: plan_simp_fb.log. Usage: python run_plan_simp_fb.py"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
CARS = next(p for p in HERE.parents if (p / 'FINAL_DEV').is_dir() and (p / 'reports').is_dir())
SIMP = CARS / 'FINAL_DEV' / 'simp'
PY = sys.executable
LEDGER = HERE / 'budget_simp_fb.json'
BUDGET_S, PROC_S, MEM_GIB = 90 * 60, 30 * 60, 16.0
RUNS = HERE / 'runs'


def log(msg):
    with open(HERE / 'plan_simp_fb.log', 'a') as f:
        f.write(time.strftime('%H:%M:%S ') + msg + '\n')
    print(msg, flush=True)


def run_proc(args, logfile, charge):
    led = json.loads(LEDGER.read_text()) if LEDGER.exists() else dict(seconds=0.0, entries=[])
    t0 = time.time(); why = None
    with open(logfile, 'w') as f:
        p = subprocess.Popen([PY, '-B'] + args, stdout=f, stderr=subprocess.STDOUT, cwd=str(HERE),
                             env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
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
    rec = (SIMP / 'simp_settings.sha256').read_text().split()[0]
    if hashlib.sha256((SIMP / 'simp_settings.json').read_bytes()).hexdigest() != rec:
        raise SystemExit('SIMP settings changed after freezing')
    frozen = SIMP / 'eval' / 'pass_a'
    man = json.loads((frozen / 'simp_manifest.json').read_text())
    RUNS.mkdir(exist_ok=True)
    others = [q.info for q in psutil.process_iter(['pid', 'name']) if q.info['name'] and 'python' in q.info['name'].lower()
              and q.pid not in (psutil.Process().pid, psutil.Process().ppid())]
    log(f'START plan; other python processes: {len(others)}')
    done = []; stop = None; replays = {}
    for rep in (0, 1, 2):
        for key in sorted(man['instances']):
            for w in ('early', 'late'):
                name = f'{key}_{w}_r{rep}'; out = RUNS / name
                rc, why, el = run_proc(['run_cell_simp_fb.py', '--frozen', str(frozen), '--size', 'S3', '--inst', key,
                                        '--window', w, '--repeat', str(rep), '--out', str(out)], RUNS / f'{name}.log', True)
                log(f'{name} rc={rc} why={why} {el:.1f}s')
                if rc != 0 or why:
                    stop = f'{why or "EXIT"} at {name}'; break
                done.append(name)
                rc2, why2, el2 = run_proc([str(SIMP / 'replay_simp.py'), str(frozen), 'S3', str(out)], RUNS / f'{name}_replay.log', False)
                rp = json.loads((out / 'replay_simp.json').read_text()) if (out / 'replay_simp.json').exists() else None
                replays[name] = (rp or {}).get('totals')
                log(f'replay {name} rc={rc2} {replays[name]} errors={len((rp or {}).get("errors", [1]))} {el2:.1f}s')
                if rc2 != 0 or rp is None or rp['errors'] or set(rp['totals']) - {'VERIFIED'}:
                    stop = f'HARD_STOP replay at {name}'; break
            if stop: break
        if stop: break
    (HERE / 'plan_simp_fb.json').write_text(json.dumps(dict(cells=done, stop=stop, replay=replays,
                                                            budget_s=json.loads(LEDGER.read_text())['seconds']), indent=1))
    log(f'PLAN DONE cells={len(done)} stop={stop}')


if __name__ == '__main__':
    main()
