"""Run the registered Phase 10B independent arithmetic checks (amendment A8) with the frozen independent10b.py:
  E  the exact-residual sample (one process),
  L  Arb lambda proofs: mandatory S2 picks (4) and optional S3 picks (2, capped at 30 min each).
These checks are untimed, so they run in parallel processes; each process's wall time is charged to the campaign ledger
(budget10b.json), conservatively as the sum over processes. A capped optional pick is recorded as UNAVAILABLE.
Usage: python run_independent10b.py"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
P10B = CARS / 'FINAL_DEV' / 'phase10b'
PY = str(CARS / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
OUT = P10B / 'independent'
ROOTS = [str(P10B / 'runs' / s) for s in ('ST_S3', 'ST_S2', 'MT_S3', 'MT_S2')]
MANDATORY = [(f, 'S2', 201, 0) for f in ('E3-TS', 'D3-TS')] + [(f, 'S2', 203, 20) for f in ('E3-TS', 'D3-TS')]
OPTIONAL = [(f, 'S3', 202, 10) for f in ('E3-TS', 'D3-TS')]
CAP = 30 * 60


def charge(sec, what):
    led = P10B / 'budget10b.json'
    d = json.loads(led.read_text())
    d['seconds'] += sec; d['entries'].append([time.strftime('%Y-%m-%d %H:%M:%S'), round(sec, 2), what])
    led.write_text(json.dumps(d, indent=0))


def main():
    OUT.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    jobs = [('E', OUT / 'exact10b.json', [PY, '-B', str(P10B / 'independent10b.py'), 'E', str(OUT / 'exact10b.json'), *ROOTS], False)]
    for f, s, sd, t in MANDATORY + OPTIONAL:
        o = OUT / f'lambda_{f}_{s}_s{sd}_t{t}.json'
        jobs.append((f'L {f} {s} {sd} {t}', o, [PY, '-B', str(P10B / 'independent10b.py'), 'L', str(o), f, s, str(sd), str(t), *ROOTS],
                     (f, s, sd, t) in OPTIONAL))
    procs = []
    for name, o, cmd, optional in jobs:
        log = open(OUT / (o.stem + '.log'), 'w')
        procs.append(dict(name=name, out=o, optional=optional, t0=time.time(), log=log,
                          p=subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(P10B))))
    status = {}
    while any(x['p'].poll() is None for x in procs):
        for x in procs:
            if x['p'].poll() is None and time.time() - x['t0'] > CAP:
                x['p'].kill(); x['killed'] = True
        time.sleep(5)
    for x in procs:
        x['p'].wait(); x['log'].close(); sec = time.time() - x['t0']; charge(sec, 'independent ' + x['name'])
        st = 'UNAVAILABLE (30-min cap)' if x.get('killed') else ('OK' if x['p'].returncode == 0 and x['out'].exists() else f"FAILED exit {x['p'].returncode}")
        status[x['name']] = dict(status=st, seconds=round(sec, 1), optional=x['optional'])
        print(x['name'], st, round(sec, 1), flush=True)
    (OUT / 'independent_status.json').write_text(json.dumps(status, indent=1))


if __name__ == '__main__':
    main()
