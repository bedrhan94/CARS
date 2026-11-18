"""Phase 10A orchestrator (development data only). Every cell runs in a fresh process (run_cell10a.py), every stage is
followed by a full replay (replay10a.py). Budget guards: 30 min per process, 4 h total; any non-zero exit or any
REJECTED replay record stops the plan (hard stop).
  stage V1  validation: S0 and S1, E3-TS / D3-TS, seeds 0-2, one repeat; S0 negative-control families SAFE-S, SAFE-P;
            replay with exact checks (exact_n = 200, i.e. all S0 records)
  stage P2  paired benchmark: S2, E3-TS / D3-TS, seeds 0-2, repeats 0-2 (rotated arm order); replay
  stage D3  S3 diagnostic (target attainment): E3-TS seeds 0-2 and D3-TS seed 0, one repeat; replay
  stage V1B9 validation of the added arm B9 (as V1 + V1E, arm B9 only)
  stage V1E validation of the EFT path: as V1 for seed 0 with CARS10A_FORCE_EFT=1 (new arms only)
Usage: python run_plan10a.py V1|V1E|V1B9|P2|D3"""
import json, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
PY = str(HERE.parent / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
BUDGET = HERE / 'budget10a.json'
LIMIT_TOTAL = 4 * 3600; LIMIT_PROC = 1800


def spent():
    return json.loads(BUDGET.read_text())['seconds'] if BUDGET.exists() else 0.0


def charge(s):
    BUDGET.write_text(json.dumps({'seconds': spent() + s}))


def run(cmd, log):
    if spent() > LIMIT_TOTAL:
        raise SystemExit('BUDGET_STOP')
    t0 = time.time()
    try:
        with open(log, 'a', encoding='utf-8') as f:
            f.write('\n$ ' + ' '.join(cmd) + '\n'); f.flush()
            p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=LIMIT_PROC,
                               env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    finally:
        charge(time.time() - t0)
    if p.returncode != 0:
        raise SystemExit(f'HARD_STOP nonzero exit {p.returncode}: {cmd}')


def replay(root, exact_n, log):
    """Replay every cell directory separately (one fresh process per cell); exact checks only for S0 cells."""
    tot = {}
    for d in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        en = exact_n if '_S0_' in d.name else 0
        stride = 20 if d.name.startswith('D3-TS_S0') else 1          # n = 128: rational elimination ~20 s/record
        run([PY, str(HERE / 'methods' / 'replay10a.py'), str(d), str(en), str(stride)], log)
        r = json.loads((d / 'replay10a.json').read_text())
        if r['totals'].get('REJECTED', 0) or r['totals'].get('UNAVAILABLE', 0):
            raise SystemExit(f'HARD_STOP replay {d}: {r["totals"]}')
        for k, x in r['totals'].items():
            tot[k] = tot.get(k, 0) + x
    return tot


def main():
    stage = sys.argv[1]; log = HERE / 'logs' / f'plan_{stage}.log'
    cell = str(HERE / 'methods' / 'run_cell10a.py')
    if stage == 'V1':
        root = HERE / 'runs' / 'V1'
        for size in ('S0', 'S1'):
            for fam in ('E3-TS', 'D3-TS'):
                for seed in (0, 1, 2):
                    run([PY, cell, '--family', fam, '--size', size, '--seed', str(seed), '--repeat', '0',
                         '--out', str(root / f'{fam}_{size}_s{seed}')], log)
        for fam in ('SAFE-S', 'SAFE-P'):
            run([PY, cell, '--family', fam, '--size', 'S0', '--seed', '0', '--repeat', '0', '--out', str(root / f'{fam}_S0_s0')], log)
        print('V1 replay', replay(root, 200, log), flush=True)
    elif stage == 'V1E':
        root = HERE / 'runs' / 'V1E'
        import os as _os
        _os.environ['CARS10A_FORCE_EFT'] = '1'
        for size in ('S0', 'S1'):
            for fam in ('E3-TS', 'D3-TS'):
                run([PY, cell, '--family', fam, '--size', size, '--seed', '0', '--repeat', '0', '--arms', 'B1,B,D,D2',
                     '--out', str(root / f'{fam}_{size}_s0')], log)
        _os.environ.pop('CARS10A_FORCE_EFT')
        print('V1E replay', replay(root, 200, log), flush=True)
    elif stage == 'V1B9':
        root = HERE / 'runs' / 'V1B9'
        for size in ('S0', 'S1'):
            for fam in ('E3-TS', 'D3-TS'):
                for seed in (0, 1, 2):
                    run([PY, cell, '--family', fam, '--size', size, '--seed', str(seed), '--repeat', '0', '--arms', 'B9',
                         '--out', str(root / f'{fam}_{size}_s{seed}')], log)
        for fam in ('SAFE-S', 'SAFE-P'):
            run([PY, cell, '--family', fam, '--size', 'S0', '--seed', '0', '--repeat', '0', '--arms', 'B9', '--out', str(root / f'{fam}_S0_s0')], log)
        import os as _os
        _os.environ['CARS10A_FORCE_EFT'] = '1'
        for size in ('S0', 'S1'):
            for fam in ('E3-TS', 'D3-TS'):
                run([PY, cell, '--family', fam, '--size', size, '--seed', '0', '--repeat', '0', '--arms', 'B9',
                     '--out', str(root / f'{fam}_{size}_s0_forcedEFT')], log)
        _os.environ.pop('CARS10A_FORCE_EFT')
        print('V1B9 replay', replay(root, 200, log), flush=True)
    elif stage == 'P2':
        root = HERE / 'runs' / 'P2'
        for rep in (0, 1, 2):
            for fam in ('E3-TS', 'D3-TS'):
                for seed in (0, 1, 2):
                    run([PY, cell, '--family', fam, '--size', 'S2', '--seed', str(seed), '--repeat', str(rep),
                         '--out', str(root / f'{fam}_S2_s{seed}')], log)
        print('P2 replay', replay(root, 0, log), flush=True)
    elif stage == 'D3':
        root = HERE / 'runs' / 'D3'
        for fam, seed in (('E3-TS', 0), ('E3-TS', 1), ('E3-TS', 2), ('D3-TS', 0)):
            run([PY, cell, '--family', fam, '--size', 'S3', '--seed', str(seed), '--repeat', '0', '--diag',
                 '--out', str(root / f'{fam}_S3_s{seed}')], log)
        print('D3 replay', replay(root, 0, log), flush=True)
    print('STAGE_DONE', stage, 'budget_s', round(spent(), 1))


if __name__ == '__main__':
    main()
