"""SIMP3D development pilot and settings (SIMP_PROTOCOL.md section 2), applied mechanically:
  1. generate the pilot instance (v_f = 0.5, N = 12) for E_min in {1e-9, 1e-3}, both windows;
  2. run one paired cell per (E_min, window) with tau = 1e-8 and replay it;
  3. E_min = 1e-9 if arm A certifies all 21 steps of both windows at 1e-9, else 1e-3;
     target = absolute 1e-8 if arm A is target-met at every pilot step (at the chosen E_min), else relative
     tau = 1e-8 * ||u_window_start||_2 per (instance, window);
  4. write simp_settings.json (emin, target rule) and pilot_report.json.
Usage: python pilot_simp.py"""
import json, os, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARS = HERE.parents[1]
PY = str(CARS / 'PHASE5_DEV' / '.venv' / 'Scripts' / 'python.exe')
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')


def run(cmd, log):
    with open(log, 'w') as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=ENV, cwd=str(HERE), timeout=1800)
    return r.returncode


def main():
    pil = HERE / 'pilot'; pil.mkdir(exist_ok=True)
    assert run([PY, '-B', 'gen_simp.py', str(pil / 'inputs'), 'pilot'], pil / 'gen.log') == 0
    man = json.loads((pil / 'inputs' / 'simp_manifest.json').read_text())
    rep = {'cells': {}}
    for key in man['instances']:
        for w in ('early', 'late'):
            out = pil / 'runs' / f'{key}_{w}'
            code = run([PY, '-B', 'run_cell_simp.py', '--frozen', str(pil / 'inputs'), '--size', 'S2', '--inst', key,
                        '--window', w, '--repeat', '0', '--out', str(out), '--target', '1e-8'], pil / f'cell_{key}_{w}.log')
            rcode = run([PY, '-B', 'replay_simp.py', str(pil / 'inputs'), 'S2', str(out)], pil / f'replay_{key}_{w}.log')
            cell = json.loads(next(out.glob('cell_*.json')).read_text()) if code == 0 else None
            rp = json.loads((out / 'replay_simp.json').read_text()) if rcode == 0 else None
            rep['cells'][f'{key}|{w}'] = dict(exit=code, replay_exit=rcode, replay_totals=(rp or {}).get('totals'),
                                              arms={r['arm']: dict(certified=r['certified'], target=r['certified_target'],
                                                                   measured_s=r['measured_s'], refreshes=r['refreshes'],
                                                                   failure_reasons=r['failure_reasons'], eps_max=r['eps_max'],
                                                                   lambda_lo_min=r['lambda_lo_min'])
                                                    for r in (cell or {}).get('results', [])})
            print(key, w, code, rcode, {a: (v['certified'], v['target']) for a, v in rep['cells'][f'{key}|{w}']['arms'].items()}, flush=True)
    A = lambda key, w, f: rep['cells'][f'{key}|{w}']['arms'].get('A', {}).get(f, 0)
    k9 = [k for k in man['instances'] if 'emin1e-09' in k][0]; k3 = [k for k in man['instances'] if 'emin1e-03' in k][0]
    emin_ok = all(A(k9, w, 'certified') == 21 for w in ('early', 'late'))
    emin = 1e-9 if emin_ok else 1e-3
    kc = k9 if emin_ok else k3
    abs_ok = all(A(kc, w, 'target') == 21 for w in ('early', 'late'))
    settings = dict(emin=emin, target_rule=('absolute 1e-8' if abs_ok else 'relative 1e-8 * ||u_window_start||_2'),
                    decided_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    evidence=dict(emin_1e9_A_certified={w: A(k9, w, 'certified') for w in ('early', 'late')},
                                  chosen_A_target_met={w: A(kc, w, 'target') for w in ('early', 'late')}), targets={})
    (HERE / 'simp_settings.json').write_text(json.dumps(settings, indent=1))
    (pil / 'pilot_report.json').write_text(json.dumps(rep, indent=1))
    print('SETTINGS', json.dumps(settings))


if __name__ == '__main__':
    main()
