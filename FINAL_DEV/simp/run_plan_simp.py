"""SIMP3D evaluation plan: 5 instances x 2 windows x 3 repeats, one ST process per cell, each replayed immediately.
Guards: 1.5 h budget (ledger budget_simp.json), 30 min per process, 16 GiB peak memory, settings hash unchanged.
Usage: python run_plan_simp.py"""
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import psutil

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
PY = str(CARS / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
LEDGER = HERE / 'budget_simp.json'
BUDGET_S, PROC_S, MEM_GIB = 1.5 * 3600, 30 * 60, 16.0


def used():
    return json.loads(LEDGER.read_text())['seconds'] if LEDGER.exists() else 0.0


def charge(sec):
    d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'seconds': 0.0, 'entries': []}
    d['seconds'] += sec; d['entries'].append([time.strftime('%H:%M:%S'), round(sec, 1)]); LEDGER.write_text(json.dumps(d))


def run_proc(cmd, log):
    t0 = time.time(); why = None; peak = 0.0
    with open(log, 'w') as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), cwd=str(HERE))
        ps = psutil.Process(p.pid)
        try:
            while p.poll() is None:
                try:
                    mi = ps.memory_info(); peak = max(peak, getattr(mi, 'peak_wset', mi.rss) / 2 ** 30)
                except psutil.Error:
                    pass
                if peak > MEM_GIB: why = 'MEMORY_CEILING'; p.kill(); break
                if time.time() - t0 > PROC_S: why = 'PROCESS_TIMEOUT'; p.kill(); break
                if used() + time.time() - t0 > BUDGET_S: why = 'BUDGET'; p.kill(); break
                time.sleep(0.5)
            p.wait()
        finally:
            charge(time.time() - t0)
    return p.returncode, why


def main():
    rec = (HERE / 'simp_settings.sha256').read_text().split()[0]
    if hashlib.sha256((HERE / 'simp_settings.json').read_bytes()).hexdigest() != rec:
        raise SystemExit('settings changed after freezing')
    frozen = HERE / 'eval' / 'pass_a'
    man = json.loads((frozen / 'simp_manifest.json').read_text())
    log = {'cells': []}; status = 'COMPLETE'
    for rep in (0, 1, 2):
        for key in sorted(man['instances']):
            for w in ('early', 'late'):
                out = HERE / 'eval' / 'runs' / f'{key}_{w}_r{rep}'
                out.parent.mkdir(parents=True, exist_ok=True)
                code, why = run_proc([PY, '-B', 'run_cell_simp.py', '--frozen', str(frozen), '--size', 'S3', '--inst', key,
                                      '--window', w, '--repeat', str(rep), '--out', str(out)], out.parent / f'{out.name}.log')
                if why or code != 0:
                    status = f'STOP {why or code} at {out.name}'; break
                code, why = run_proc([PY, '-B', 'replay_simp.py', str(frozen), 'S3', str(out)], out.parent / f'{out.name}_replay.log')
                rp = json.loads((out / 'replay_simp.json').read_text()) if (out / 'replay_simp.json').exists() else None
                log['cells'].append(dict(cell=out.name, replay=(rp or {}).get('totals'), errors=len((rp or {}).get('errors', [1]))))
                print(out.name, (rp or {}).get('totals'), flush=True)
                if why or code != 0 or rp is None or rp['errors'] or set(rp['totals']) - {'VERIFIED'}:
                    status = f'HARD_STOP replay at {out.name}'; break
            if status != 'COMPLETE':
                break
        if status != 'COMPLETE':
            break
    log.update(status=status, budget_used_s=used())
    (HERE / 'eval' / 'plan_simp.json').write_text(json.dumps(log, indent=1))
    print('SIMP PLAN', status, round(used() / 60, 1), 'min')


if __name__ == '__main__':
    main()
