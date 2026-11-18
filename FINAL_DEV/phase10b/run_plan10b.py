"""Phase 10B plan runner: executes the registered cells in a fixed order, one fresh process per cell, each followed by
the full replay of that cell. Resource guards (pre-registered): campaign budget 6 h (ledger budget10b.json, charged in a
finally block), 30 min per process, peak memory ceiling 16 GiB per process (polled watchdog). Hard stops: input
mismatch (exit 3), replay REJECTED/UNAVAILABLE, manifest mismatch in replay, a second failure of a crashed cell.
A crashed cell (non-zero exit other than 3) is repeated once into <cell>_retry and the repetition is reported.
Stages:  SMOKE (development seed 0, S0/S1, ST and MT; harness test before the freeze)
         ST_S3, ST_S2, MT_S3, MT_S2 (held-out seeds 200-204, 3 repeats; require the freeze)
         R2_ST_S3 (same-machine second session after the campaign: ST S3, one repeat with index 3; not decisive)
Usage: python run_plan10b.py STAGE"""
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common10b as K                                                # noqa: E402

PY = str(K.CARS / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
RUNS = HERE / 'runs'
LEDGER = HERE / ('budget10b_smoke.json' if sys.argv[1:2] == ['SMOKE'] else 'budget10b.json')   # smoke is not campaign
BUDGET_S = 6 * 3600; PROC_S = 30 * 60; MEM_GIB = 16.0


def charge(sec):
    d = json.loads(LEDGER.read_text()) if LEDGER.exists() else {'seconds': 0.0, 'entries': []}
    d['seconds'] += sec; d['entries'].append([time.strftime('%Y-%m-%d %H:%M:%S'), round(sec, 2)])
    LEDGER.write_text(json.dumps(d, indent=0))
    return d['seconds']


def used():
    return json.loads(LEDGER.read_text())['seconds'] if LEDGER.exists() else 0.0


def run_proc(cmd, log):
    """Run one process with the per-process timeout and the memory watchdog; returns (exit, seconds, peak_GiB, why)."""
    t0 = time.time(); peak = 0.0; why = None
    with open(log, 'w', encoding='utf-8') as lf:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        pr = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(HERE))
        ps = psutil.Process(pr.pid)
        try:
            while pr.poll() is None:
                try:
                    mi = ps.memory_info(); peak = max(peak, getattr(mi, 'peak_wset', mi.rss) / 2 ** 30)
                except psutil.Error:
                    pass
                if peak > MEM_GIB:
                    why = 'MEMORY_CEILING'; pr.kill(); break
                if time.time() - t0 > PROC_S:
                    why = 'PROCESS_TIMEOUT'; pr.kill(); break
                if used() + (time.time() - t0) > BUDGET_S:
                    why = 'CAMPAIGN_BUDGET'; pr.kill(); break
                time.sleep(0.5)
            pr.wait()
        finally:
            sec = time.time() - t0; charge(sec)
    return pr.returncode, sec, peak, why


def cells(stage):
    if stage == 'SMOKE':
        return [(f, s, 0, 0, m) for m in ('ST', 'MT') for s in ('S0', 'S1') for f in K.FAMILIES]
    if stage == 'R2_ST_S3':                     # same-machine second session (amendment A10): one repeat, index 3
        return [(f, 'S3', seed, 3, 'ST') for seed in K.HELDOUT_SEEDS for f in K.FAMILIES]
    mode, size = stage.split('_')
    if mode not in K.MODES or size not in K.SIZES:
        raise SystemExit('unknown stage')
    return [(f, size, seed, rep, mode) for rep in (0, 1, 2) for seed in K.HELDOUT_SEEDS for f in K.FAMILIES]


def main():
    stage = sys.argv[1]
    if stage != 'SMOKE':
        K.check_freeze(); K.check_inputs_frozen()
    root = RUNS / stage; root.mkdir(parents=True, exist_ok=True)
    log = {'stage': stage, 'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'cells': []}
    status = 'COMPLETE'
    for fam, size, seed, rep, mode in cells(stage):
        name = f'{fam}_{size}_s{seed}_r{rep}_{mode}'
        entry = dict(cell=name)
        for attempt in (0, 1):
            cdir = root / (name if attempt == 0 else name + '_retry')
            cmd = [PY, str(HERE / 'run_cell10b.py'), '--family', fam, '--size', size, '--seed', str(seed),
                   '--repeat', str(rep), '--mode', mode, '--out', str(cdir)]
            cdir.mkdir(parents=True, exist_ok=True)
            code, sec, peak, why = run_proc(cmd, cdir / 'cell.log')
            entry.update({f'attempt{attempt}': dict(exit=code, seconds=round(sec, 1), peak_GiB=round(peak, 3), stop=why)})
            print(name, 'attempt', attempt, 'exit', code, round(sec, 1), 's', why or '', flush=True)
            if why in ('CAMPAIGN_BUDGET', 'MEMORY_CEILING', 'PROCESS_TIMEOUT'):
                status = 'RESOURCE_STOP:' + why; break
            if code == 3:
                status = 'HARD_STOP:INPUT_MISMATCH'; break
            if code == 0:
                break
        else:
            status = 'HARD_STOP:CELL_FAILED_TWICE'
        if status != 'COMPLETE':
            log['cells'].append(entry); break
        code, sec, peak, why = run_proc([PY, str(HERE / 'replay10b.py'), str(cdir)], cdir / 'replay.log')
        rp = json.loads((cdir / 'replay10b.json').read_text()) if (cdir / 'replay10b.json').exists() else None
        entry['replay'] = dict(exit=code, seconds=round(sec, 1), stop=why,
                               totals=(rp or {}).get('totals'), errors=len((rp or {}).get('errors', [1])),
                               manifest_mismatches=len((rp or {}).get('manifest_mismatches', [1])))
        log['cells'].append(entry)
        print(name, 'replay', entry['replay'], flush=True)
        if why:
            status = 'RESOURCE_STOP:' + why; break
        if code != 0 or rp is None or rp['errors'] or rp['manifest_mismatches'] or \
                set(rp['totals']) - {'VERIFIED'}:
            status = 'HARD_STOP:REPLAY'; break
    log.update(status=status, finished=time.strftime('%Y-%m-%d %H:%M:%S'), budget_used_s=round(used(), 1))
    (root / f'plan_{stage}.json').write_text(json.dumps(log, indent=1))
    print('STAGE', stage, status, 'budget_used_h', round(used() / 3600, 3), flush=True)
    if status != 'COMPLETE':
        sys.exit(2)


if __name__ == '__main__':
    main()
